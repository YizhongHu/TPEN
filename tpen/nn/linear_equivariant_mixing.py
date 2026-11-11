"""Unary linear support-path mixing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from tpen.data.paths import (
    LinearPathMetadata,
    LinearPathPolicy,
    NormalizedChannels,
    SupportPath,
)
from tpen.data.real import Feature, Interaction, common_real_batch_size, common_real_dtype, common_real_particle_count, zero_block
from tpen.dependencies import require_torch, require_torch_nn
from tpen.equivariance import EquivariantMap
from tpen.nn.mixing_kernel import (
    Aggregation,
    MixingImplementation,
    build_unary_index_plan,
    execute_unary,
    normalize_implementation,
)

torch = require_torch(feature="TPEN linear equivariant mixing")
nn = require_torch_nn(feature="TPEN linear equivariant mixing")


class LinearEquivariantMixing(EquivariantMap):
    """Unary linear permutation-equivariant support-path producer.

    The path metadata and all ``W_q`` parameters are constructed eagerly.
    Forward only creates device-local index tensors through the shared slow
    unary kernel; it never discovers paths, writes metadata, or registers
    parameters.

    Parameters
    ----------
    max_order : int
        Maximum output and default input body order.
    channels : int or mapping
        Input channels by body order.
    input_orders : tuple of int, optional
        Constructor-owned input orders for ``orbit_complete`` metadata.
    output_orders : tuple of int, optional
        Output orders to materialize; defaults to ``1..max_order``.
    out_channels : int or mapping, optional
        Output channels by output order. Defaults to ``channels``.
    policy : {"coordinate_neighbor", "orbit_complete", "explicit"}, optional
        Static path-family policy. The shipped default is
        ``coordinate_neighbor``.
    metadata : LinearPathMetadata, optional
        Precomputed immutable metadata. It is used verbatim.
    explicit : tuple of SupportPath, optional
        Explicit ordered paths when ``policy="explicit"``.
    aggregation : {"sum", "completion_mean"}, optional
        Reduction over support completions.
    implementation : {"slow", "vectorized"}, optional
        Unary execution strategy. The vectorized strategy gathers source
        tuples, contracts pathwise, and scatters into the output block.
    initial_weight : float, optional
        Initial value for every eagerly allocated ``W_q``.
    """

    def __init__(
        self,
        max_order: int,
        *,
        channels: int | Mapping[int, int],
        input_orders: tuple[int, ...] | None = None,
        output_orders: tuple[int, ...] | None = None,
        out_channels: int | Mapping[int, int] | None = None,
        policy: LinearPathPolicy | str = LinearPathPolicy.COORDINATE_NEIGHBOR,
        metadata: LinearPathMetadata | None = None,
        explicit: tuple[SupportPath, ...] | None = None,
        aggregation: Aggregation | str = Aggregation.COMPLETION_MEAN,
        implementation: MixingImplementation | str = MixingImplementation.SLOW,
        initial_weight: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.max_order = int(max_order)
        if self.max_order <= 0:
            raise ValueError(f"max_order must be positive, got {self.max_order}")
        self.aggregation = Aggregation(aggregation)
        self.implementation = normalize_implementation(implementation)
        self.metadata = metadata or LinearPathMetadata.generate(
            max_order=self.max_order,
            policy=policy,
            input_orders=input_orders,
            output_orders=output_orders,
            explicit=explicit,
        )
        self.paths = self.metadata.all_paths()
        self.input_channels = _normalize_channels(channels, self.metadata.input_orders.values, "channels")
        self.output_channels = _normalize_channels(
            channels if out_channels is None else out_channels,
            self.metadata.output_orders.values,
            "out_channels",
        )
        self.initial_weight = float(initial_weight)
        self.weights = nn.ParameterList()
        for path in self.paths:
            self.weights.append(
                nn.Parameter(torch.full((self.output_channels.for_order(path.output_order), self.input_channels.for_order(path.input_order)), self.initial_weight))
            )

    def forward_impl(self, x: Feature) -> Interaction:
        """Apply every static unary path to a feature state."""

        x.validate()
        n_particles = common_real_particle_count(x)
        batch_size = common_real_batch_size(x)
        dtype = common_real_dtype(x)
        device = x.blocks[0].device
        for order in self.metadata.input_orders.values:
            _validate_feature_channels(x, order, self.input_channels.for_order(order))

        output_blocks = [zero_block(batch_size=batch_size, paths=0, device=device, dtype=dtype)]
        path_offset = 0
        for output_order in range(1, self.max_order + 1):
            output_channel_count = self.output_channels.for_order(output_order)
            paths = self.metadata.paths_for_output_order(output_order) if output_order in self.metadata.output_orders.values else ()
            if not paths:
                output_blocks.append(torch.zeros((batch_size, output_channel_count, 0, *((n_particles,) * output_order)), device=device, dtype=dtype))
                continue
            # Index tensors are execution caches: rebuilding them per forward
            # permits runtime N/device changes without changing static paths or
            # registered weights. The kernel still executes one path at a time.
            path_blocks = []
            group_start = 0
            while group_start < len(paths):
                input_order = paths[group_start].input_order
                group_end = group_start + 1
                while group_end < len(paths) and paths[group_end].input_order == input_order:
                    group_end += 1
                group_paths = paths[group_start:group_end]
                plans = tuple(
                    build_unary_index_plan(path, n_particles, device=device) for path in group_paths
                )
                weights = tuple(
                    self.weights[path_offset + index]
                    for index in range(group_start, group_end)
                )
                path_blocks.append(
                    execute_unary(
                        group_paths,
                        weights,
                        x.blocks[input_order],
                        n_particles=n_particles,
                        output_order=output_order,
                        batch_size=batch_size,
                        output_channels=output_channel_count,
                        aggregation=self.aggregation,
                        implementation=self.implementation,
                        index_plans=plans,
                    )
                )
                group_start = group_end
            output_blocks.append(torch.cat(path_blocks, dim=2))
            path_offset += len(paths)
        return Interaction(output_blocks)

    def forward_pre_activation(self, x: Feature) -> Interaction:
        """Return raw paths through the common producer interface."""

        return self.forward_impl(x)


def _normalize_channels(
    value: int | Mapping[int, int], orders: tuple[int, ...], name: str
) -> NormalizedChannels:
    """Normalize channel configuration into an immutable typed record."""

    if isinstance(value, Mapping):
        pairs = tuple((int(order), int(value[order])) for order in orders if order in value)
        missing = [order for order in orders if order not in value]
        if missing:
            raise ValueError(f"{name} is missing orders {missing}")
    else:
        count = int(value)
        pairs = tuple((order, count) for order in orders)
    if any(count <= 0 for _, count in pairs):
        raise ValueError(f"{name} channel counts must be positive")
    return NormalizedChannels(pairs)


def _validate_feature_channels(feature: Feature, order: int, expected: int) -> None:
    """Validate one configured feature block without reflective lookup."""

    if order >= len(feature.blocks):
        raise ValueError(f"feature has no order-{order} block")
    actual = int(feature.blocks[order].shape[1])
    if actual != expected:
        raise ValueError(f"feature order-{order} channels {actual} do not match configured {expected}")


__all__ = ["LinearEquivariantMixing"]
