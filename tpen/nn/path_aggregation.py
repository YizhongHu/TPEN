"""Learned real-space aggregation over path-resolved interactions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from operator import index

from tpen.data.real import Interaction, Update, zero_block
from tpen.dependencies import require_torch, require_torch_nn
from tpen.equivariance import EquivariantMap
from tpen.nn.initialization import TorchInitializer
from tpen.data.paths import PathLayout, PathMetadata, VirtualPath, load_default_path_metadata

torch = require_torch(feature="TPEN path aggregation")
nn = require_torch_nn(feature="TPEN path aggregation")


class PathAggregation(EquivariantMap):
    """Aggregate path-resolved real interactions into a feature update.

    Mathematical reference: TPEN design document "Aggregation" and
    MIG-TPEN-000 section 2.2. The implemented map is

    ``u^c_I = Gamma_c( sum_p U^{(m)}[c, p] h^c_{I,p} )``

    per output order ``m``: the path axis is contracted per input channel
    with a learned weight, then the owned activation ``Gamma_c`` is applied.
    Per decision D3 the first activation form is elementwise with
    ``C_out = C_in``. The opt-in channel-preserving MLP activation also keeps
    ``C_out = C_in`` while mixing channels at each fixed tuple position; this
    changes only ``Gamma_c`` behind the same activation signature.

    The input contract is :class:`Interaction` with blocks of shape
    ``[batch, channels, paths, indices...]``. The output contract is
    :class:`Update` with blocks of shape ``[batch, channels, indices...]``.
    The weights are shared over batch and tuple positions ``I`` and never mix
    channels or particle indices, which preserves permutation equivariance.

    Parameters
    ----------
    max_order : int
        Maximum tuple order to aggregate.
    channels : int or mapping
        Input (= output) channels per tuple order.
    layout : PathLayout or None, optional
        Static union layout shared with CompositeMixing. When supplied, its
        path counts are used directly and incoming Interaction fingerprints
        must match.
    max_virtual_order : int or None, optional
        Maximum virtual support order used when deriving path counts from
        metadata. Defaults to `max_order`.
    paths : PathMetadata, tuple of VirtualPath, or None, optional
        Path metadata used to derive path counts. If ``None``, checked-in path
        metadata for `output_embedding` is loaded.
    output_embedding : str, optional
        Path family used when loading default metadata.
    path_counts_by_order : mapping of int to int or None, optional
        Explicit path counts, mainly for tests or custom path families.
    activation : torch.nn.Module, callable, or None, optional
        Owned shape-preserving activation ``Gamma_c`` applied after the path
        contraction. ``None`` keeps the identity. Pointwise choices transform
        entries independently; the opt-in channel-preserving MLP may mix
        channels at each fixed tuple position. Neither form mixes particles or
        the contracted path axis.
    initializer : TorchInitializer or None, optional
        Explicit side-effect-free initializer for learned path weights. If
        ``None``, weights use the legacy PyTorch global-RNG Xavier initializer.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        max_order: int,
        channels: int | Mapping[int, int],
        layout: PathLayout | None = None,
        max_virtual_order: int | None = None,
        paths: PathMetadata | tuple[VirtualPath, ...] | None = None,
        output_embedding: str = "canonical",
        path_counts_by_order: Mapping[int, int] | None = None,
        activation: "nn.Module | Callable[[torch.Tensor], torch.Tensor] | None" = None,
        initializer: TorchInitializer | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_order = int(max_order)
        self.max_virtual_order = self.max_order if max_virtual_order is None else int(max_virtual_order)
        if self.max_order <= 0:
            raise ValueError(f"max_order must be positive, got {self.max_order}")
        if self.max_virtual_order <= 0:
            raise ValueError(f"max_virtual_order must be positive, got {self.max_virtual_order}")
        self.channels_by_order = _normalize_positive_channels(channels, max_order=self.max_order, name="channels")
        self.layout = layout
        if layout is not None:
            if layout.output_orders.values != tuple(range(1, self.max_order + 1)):
                raise ValueError("PathAggregation layout must cover contiguous output orders")
            for order in layout.output_orders.values:
                if self.channels_by_order[order] != layout.output_channels.for_order(order):
                    raise ValueError(f"channels disagree with layout at order {order}")
        if layout is not None:
            if path_counts_by_order is not None:
                raise ValueError("pass either layout or path_counts_by_order, not both")
            self.path_counts_by_order = {
                order: layout.count_for_order(order) for order in range(1, self.max_order + 1)
            }
        elif path_counts_by_order is None:
            self.path_counts_by_order = _path_counts_by_order(
                paths=paths,
                output_embedding=output_embedding,
                max_order=self.max_order,
                max_virtual_order=self.max_virtual_order,
            )
        else:
            self.path_counts_by_order = _normalize_nonnegative_channels(
                path_counts_by_order,
                max_order=self.max_order,
                name="path_counts_by_order",
            )
        self.activation = activation
        self.initializer = initializer
        self.weights = nn.ParameterDict()
        self._initialize_weights()

    def forward_impl(self, x: Interaction) -> Update:
        """Return the path-aggregated real-space feature update."""

        x.validate()
        if self.layout is not None and x.fingerprint != self.layout.fingerprint:
            raise ValueError("PathAggregation received an Interaction with the wrong layout fingerprint")
        if not x.blocks:
            # A valid empty interaction aggregates to a valid empty update.
            return Update([])
        batch_size = x.batch_size
        device = x.blocks[0].device
        dtype = x.blocks[0].dtype
        output_blocks: list[torch.Tensor] = [
            zero_block(batch_size=batch_size, device=device, dtype=dtype)
        ]
        for order in range(1, len(x.blocks)):
            output_blocks.append(self.aggregate_block(order, x.blocks[order]))
        return Update(output_blocks)

    def aggregate_block(self, order: int, tensor: torch.Tensor) -> torch.Tensor:
        """Aggregate one order block with the learned path weights.

        Parameters
        ----------
        order : int
            Output body order of the block.
        tensor : torch.Tensor
            Path-resolved block with shape
            ``[batch, channels, paths, indices...]``.

        Returns
        -------
        torch.Tensor
            Aggregated block with shape ``[batch, channels, indices...]``.
        """

        weight = self._weight_for(order, tensor)
        # u^c_I = sum_p U[c, p] h^c_{I,p}: the path axis is contracted per
        # channel; batch and tuple indices pass through untouched, so the map
        # is shared over particles and cannot break equivariance.
        aggregated = torch.einsum("bcp...,cp->bc...", tensor, weight)
        if self.activation is not None:
            aggregated = self.activation(aggregated)
        return aggregated

    def key(self, order: int) -> str:
        """Return the stable parameter key for one output order."""

        return f"o{int(order)}"

    def _weight_for(self, order: int, tensor: torch.Tensor) -> torch.Tensor:
        if order < 1 or order > self.max_order:
            raise ValueError(f"PathAggregation received order {order} outside [1, {self.max_order}]")
        in_channels = int(tensor.shape[1])
        path_count = int(tensor.shape[2])
        expected_channels = self.channels_by_order[order]
        expected_paths = self.path_counts_by_order[order]
        if in_channels != expected_channels:
            raise ValueError(
                f"PathAggregation order-{order} channels are {in_channels}, expected {expected_channels}"
            )
        if path_count != expected_paths:
            raise ValueError(
                f"PathAggregation order-{order} path count is {path_count}, expected {expected_paths}"
            )
        key = self.key(order)
        if key not in self.weights:
            raise RuntimeError(f"Missing eager PathAggregation weight for order {order}")
        weight = self.weights[key]
        shape = (expected_channels, expected_paths)
        if tuple(weight.shape) != shape:
            raise ValueError(
                f"PathAggregation order-{order} weight has shape {tuple(weight.shape)}, expected {shape}"
            )
        return weight

    def _initialize_weights(self) -> None:
        for order in range(1, self.max_order + 1):
            key = self.key(order)
            shape = (
                self.channels_by_order[order],
                self.path_counts_by_order[order],
            )
            weight = torch.empty(shape)
            if weight.numel() > 0:
                if self.initializer is None:
                    nn.init.xavier_uniform_(weight)
                else:
                    self.initializer.spawn(f"order_{order}").xavier_uniform_(weight)
            self.weights[key] = nn.Parameter(weight)


def _normalize_positive_channels(
    value: int | Mapping[int, int],
    *,
    max_order: int,
    name: str,
) -> dict[int, int]:
    channels = _normalize_nonnegative_channels(value, max_order=max_order, name=name)
    for order, count in channels.items():
        if count <= 0:
            raise ValueError(f"{name}[{order}] must be positive, got {count}")
    return channels


def _normalize_nonnegative_channels(
    value: int | Mapping[int, int],
    *,
    max_order: int,
    name: str,
) -> dict[int, int]:
    if isinstance(value, Mapping):
        normalized = {}
        for raw_order, raw_channels in value.items():
            order = _nonnegative_int(raw_order, f"{name} key")
            if order < 1 or order > max_order:
                raise ValueError(f"{name} contains order {order} outside [1, {max_order}]")
            channels = _nonnegative_int(raw_channels, f"{name}[{order}]")
            normalized[order] = channels
        missing = [order for order in range(1, max_order + 1) if order not in normalized]
        if missing:
            raise ValueError(f"{name} is missing orders {missing}")
        return dict(sorted(normalized.items()))
    channels = _nonnegative_int(value, name)
    return {order: channels for order in range(1, max_order + 1)}


def _path_counts_by_order(
    *,
    paths: PathMetadata | tuple[VirtualPath, ...] | None,
    output_embedding: str,
    max_order: int,
    max_virtual_order: int,
) -> dict[int, int]:
    if isinstance(paths, PathMetadata):
        all_paths = paths.all_paths()
    elif paths is None:
        metadata = load_default_path_metadata(output_embedding)
        if metadata.max_order < max_order or metadata.max_virtual_order < max_virtual_order:
            raise ValueError(
                "Saved path metadata only covers "
                f"max_order={metadata.max_order}, max_virtual_order={metadata.max_virtual_order}; "
                "pass explicit PathMetadata for larger generated path families"
            )
        all_paths = metadata.all_paths()
    else:
        all_paths = list(paths)
    return {
        order: sum(
            1
            for path in all_paths
            if path.m == order
            and path.s <= max_virtual_order
            and path.m <= max_order
            and path.m1 <= max_order
            and path.m2 <= max_order
        )
        for order in range(1, max_order + 1)
    }


def _nonnegative_int(value: object, name: str) -> int:
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be nonnegative, got {result}")
    return result


__all__ = ["PathAggregation"]
