"""Real-space equivariant mixing kernels."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from tpen.data.real import (
    Feature,
    Interaction,
    common_real_batch_size,
    common_real_dtype,
    common_real_particle_count,
    zero_block,
)
from tpen.dependencies import require_torch, require_torch_nn
from tpen.equivariance import EquivariantMap
from tpen.data.paths import PathMetadata, VirtualPath, load_default_path_metadata
from tpen.nn.mixing_kernel import (
    Aggregation,
    MixingImplementation,
    execute_binary,
    normalize_aggregation,
    normalize_implementation,
)

torch = require_torch(feature="TPEN equivariant mixing")
nn = require_torch_nn(feature="TPEN equivariant mixing")


class EquivariantMixing(EquivariantMap):
    """Bilinear virtual-support real-space mixing module.

    Mathematical reference: ``main.typ`` section "Equivariant Mixing" and the
    "Model Workflow" TPEN layer block. The implemented contraction is

    ``h^c_{I,p} = sum_{J_[s]\\im(tau)} W_p^{c<-c1 c2}
    x^{c1}_{J o tau1} x^{c2}_{J o tau2}``.

    In code, one :class:`VirtualPath` is the path index
    ``p = (s, m, m1, m2, tau, tau1, tau2)``. A ``virtual_tuple`` is ``J``;
    ``select_tuple(virtual_tuple, tau)`` gives the output tuple ``I``; and
    ``tau1``/``tau2`` select the two input tuples from the same virtual
    support. The path axis is deliberately preserved in :class:`Interaction`
    so the later path-aggregation stage can choose how to combine mechanisms.

    The slow implementation is a literal correctness reference that loops over
    paths and ordered distinct virtual tuples exactly as written in the PR
    plan. The vectorized implementation batches virtual tuples path-by-path and
    is tested against the slow oracle.

    Parameters
    ----------
    max_order : int
        Maximum input/output body order.
    max_virtual_order : int or None, optional
        Maximum virtual support order. Defaults to `max_order`.
    paths : PathMetadata, tuple of VirtualPath, or None, optional
        Precomputed paths. If ``None``, canonical paths are generated.
    output_embedding : {"canonical", "full"}, optional
        Path family used when generating paths.
    aggregation : {"sum", "completion_mean"}, optional
        Whether to sum over completions or average over compatible completions
        for each output tuple and path.
    channels : int or mapping
        Input feature channels per body order. This is architecture metadata
        and is independent of particle count.
    left_channels, right_channels : int, mapping, or None, optional
        Input channels for asymmetric two-input mixing. If omitted, `channels`
        is used for both sides.
    out_channels : int, mapping, or None, optional
        Output channels per target order. ``None`` preserves `channels`.
    initial_weight : float, optional
        Initial value for each path weight.
    implementation : {"slow", "vectorized"}, optional
        Mixing kernel implementation. ``"slow"`` keeps the literal loop oracle;
        ``"vectorized"`` batches virtual tuples path-by-path and should match
        the slow reference exactly.
    activation : torch.nn.Module, callable, or None, optional
        Owned shape-preserving activation ``Gamma`` applied to every
        positive-order output block after the bilinear contraction (TPEN layer
        contract, MIG-TPEN-000 section 2.2). ``None`` keeps the identity and
        preserves the pre-TPEN behavior exactly. Pointwise choices transform
        each entry independently; the opt-in
        :class:`ChannelPreservingMLPActivation` may mix channels at each fixed
        path and tuple position while preserving the channel width. The
        activation is applied to the full block, including non-distinct tuple
        entries that mixing never writes, so a ``Gamma(0) != 0`` choice writes
        an invariant constant onto those entries. The activation never mixes
        paths or particle indices.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        max_order: int,
        *,
        max_virtual_order: int | None = None,
        paths: PathMetadata | tuple[VirtualPath, ...] | None = None,
        output_embedding: Literal["canonical", "full"] = "canonical",
        aggregation: str | Aggregation = Aggregation.SUM,
        channels: int | Mapping[int, int],
        left_channels: int | Mapping[int, int] | None = None,
        right_channels: int | Mapping[int, int] | None = None,
        out_channels: int | Mapping[int, int] | None = None,
        initial_weight: float = 1.0,
        implementation: str | MixingImplementation = MixingImplementation.SLOW,
        activation: "nn.Module | Callable[[torch.Tensor], torch.Tensor] | None" = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.activation = activation
        self.max_order = int(max_order)
        self.max_virtual_order = self.max_order if max_virtual_order is None else int(max_virtual_order)
        if self.max_order <= 0:
            raise ValueError(f"max_order must be positive, got {self.max_order}")
        if self.max_virtual_order <= 0:
            raise ValueError(f"max_virtual_order must be positive, got {self.max_virtual_order}")
        self.aggregation = normalize_aggregation(aggregation)
        self.implementation = normalize_implementation(implementation)
        self.output_embedding = output_embedding
        self.initial_weight = float(initial_weight)
        self.left_channels = _normalize_channels(
            channels if left_channels is None else left_channels,
            max_order=self.max_order,
            name="left_channels",
        )
        self.right_channels = _normalize_channels(
            channels if right_channels is None else right_channels,
            max_order=self.max_order,
            name="right_channels",
        )
        self.out_channels = _normalize_channels(
            channels if out_channels is None else out_channels,
            max_order=self.max_order,
            name="out_channels",
        )
        if isinstance(paths, PathMetadata):
            self.paths = self._paths_from_metadata(paths)
        elif paths is None:
            metadata = load_default_path_metadata(output_embedding)
            if metadata.max_order < self.max_order or metadata.max_virtual_order < self.max_virtual_order:
                raise ValueError(
                    "Saved path metadata only covers "
                    f"max_order={metadata.max_order}, max_virtual_order={metadata.max_virtual_order}; "
                    "pass explicit PathMetadata for larger generated path families"
                )
            self.paths = self._paths_from_metadata(metadata)
        else:
            self.paths = tuple(paths)
        self.weights = nn.ParameterDict()
        self._initialize_weights()

    def forward_impl(self, x1: Feature, x2: Feature | None = None) -> Interaction:
        """Mix one or two real feature states into path-resolved interactions."""

        return self._forward_impl(x1, x2, apply_activation=True)

    def forward_pre_activation(self, x1: Feature, x2: Feature | None = None) -> Interaction:
        """Return raw paths for :class:`CompositeMixing` before common Gamma."""

        return self._forward_impl(x1, x2, apply_activation=False)

    def _forward_impl(
        self, x1: Feature, x2: Feature | None, *, apply_activation: bool
    ) -> Interaction:
        """Run the TP contraction with an explicitly selected activation boundary."""

        x2 = x1 if x2 is None else x2
        x1.validate()
        x2.validate()
        n_particles = common_real_particle_count(x1, x2)
        batch_size = common_real_batch_size(x1, x2)
        dtype = common_real_dtype(x1, x2)
        device = x1.blocks[0].device if x1.blocks else None
        output_blocks: list[torch.Tensor] = [
            zero_block(batch_size=batch_size, paths=0, device=device, dtype=dtype)
        ]
        for order in range(1, self.max_order + 1):
            active_paths = self._paths_for_order(order, x1=x1, x2=x2)
            out_channels = self._out_channels(order)
            # ``order`` is the output body order m in main.typ. The block shape
            # is [batch, c_out, p, I_1, ..., I_m], i.e. h^c_{I,p}.
            block = torch.zeros(
                (batch_size, out_channels, len(active_paths), *((n_particles,) * order)),
                device=device,
                dtype=dtype,
            )
            # ``completion_mean`` divides by the number of virtual completions
            # J that collapse to a fixed output tuple I for a path p. This is a
            # normalized variant of the same sum in main.typ, useful when
            # different I have different completion counts near small n.
            weights = tuple(
                self._weight_for(path, x1=x1, x2=x2, out_channels=out_channels)
                for path in active_paths
            )
            block = execute_binary(
                tuple(active_paths),
                weights,
                tuple(x1.blocks[path.m1] for path in active_paths),
                tuple(x2.blocks[path.m2] for path in active_paths),
                n_particles=n_particles,
                output_order=order,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                output_channels=out_channels,
                aggregation=self.aggregation,
                implementation=self.implementation,
            )
            # Owned shape-preserving Gamma on the full block (TPEN contract).
            # Applied after completion averaging so Gamma sees the final mixed
            # values; a channel-preserving MLP may mix only the channel axis.
            if apply_activation and self.activation is not None:
                block = self.activation(block)
            output_blocks.append(block)
        return Interaction(output_blocks)

    def _paths_for_order(self, order: int, *, x1: Feature, x2: Feature) -> list[VirtualPath]:
        return [
            path
            for path in self.paths
            if path.m == order and path.m1 < len(x1.blocks) and path.m2 < len(x2.blocks)
        ]

    def _paths_from_metadata(self, metadata: PathMetadata) -> tuple[VirtualPath, ...]:
        return tuple(
            path
            for path in metadata.all_paths()
            if path.s <= self.max_virtual_order
            and path.m <= self.max_order
            and path.m1 <= self.max_order
            and path.m2 <= self.max_order
        )

    def _out_channels(self, order: int) -> int:
        return self.out_channels[order]

    def _initialize_weights(self) -> None:
        for path in self.paths:
            key = f"g{path.global_id}"
            if key in self.weights:
                continue
            shape = (
                self.out_channels[path.m],
                self.left_channels[path.m1],
                self.right_channels[path.m2],
            )
            self.weights[key] = nn.Parameter(torch.full(shape, self.initial_weight))

    def _weight_for(
        self,
        path: VirtualPath,
        *,
        x1: Feature,
        x2: Feature,
        out_channels: int,
    ) -> torch.Tensor:
        left_channels = self.left_channels[path.m1]
        right_channels = self.right_channels[path.m2]
        _validate_feature_channels(x1, path.m1, left_channels, name="left input")
        _validate_feature_channels(x2, path.m2, right_channels, name="right input")
        shape = (out_channels, left_channels, right_channels)
        key = f"g{path.global_id}"
        if key not in self.weights:
            raise RuntimeError(f"Missing eager EquivariantMixing weight for path {path.global_id}")
        weight = self.weights[key]
        if tuple(weight.shape) != shape:
            raise ValueError(f"Path {path.global_id} weight shape {tuple(weight.shape)} does not match {shape}")
        return weight


def _normalize_channels(value: int | Mapping[int, int], *, max_order: int, name: str) -> dict[int, int]:
    if isinstance(value, Mapping):
        channels = {int(order): int(count) for order, count in value.items()}
        missing = [order for order in range(1, max_order + 1) if order not in channels]
        if missing:
            raise ValueError(f"{name} is missing orders {missing}")
    else:
        count = int(value)
        channels = {order: count for order in range(1, max_order + 1)}
    for order, count in channels.items():
        if order < 1 or order > max_order:
            raise ValueError(f"{name} contains order {order} outside [1, {max_order}]")
        if count <= 0:
            raise ValueError(f"{name}[{order}] must be positive, got {count}")
    return dict(sorted(channels.items()))


def _validate_feature_channels(feature: Feature, order: int, expected: int, *, name: str) -> None:
    if order >= len(feature.blocks):
        raise ValueError(f"{name} has no order-{order} block")
    actual = int(feature.blocks[order].shape[1])
    if actual != expected:
        raise ValueError(f"{name} order-{order} channels {actual} do not match configured {expected}")


__all__ = ["EquivariantMixing"]
