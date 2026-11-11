"""Parameter-free typed execution kernels for support-path mixing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tpen.data.indices import (
    flatten_tuple_indices,
    ordered_tuple_tensor,
    select_tuple_tensor,
)
from tpen.data.paths import SupportPath, VirtualPath
from tpen.dependencies import require_torch

torch = require_torch(feature="TPEN mixing kernels")


class Aggregation(str, Enum):
    """Closed set of support-completion reductions."""

    SUM = "sum"
    COMPLETION_MEAN = "completion_mean"


class MixingImplementation(str, Enum):
    """Closed set of support-kernel implementations."""

    SLOW = "slow"
    VECTORIZED = "vectorized"


@dataclass(frozen=True)
class UnaryIndexPlan:
    """Typed per-forward indices for one unary support path."""

    supports: torch.Tensor
    output_indices: torch.Tensor
    input_indices: torch.Tensor
    flat_output_indices: torch.Tensor


@dataclass(frozen=True)
class BinaryIndexPlan:
    """Typed per-forward indices for one binary virtual-support path."""

    supports: torch.Tensor
    output_indices: torch.Tensor
    left_indices: torch.Tensor
    right_indices: torch.Tensor
    flat_output_indices: torch.Tensor


def normalize_aggregation(value: str | Aggregation) -> Aggregation:
    """Normalize a configuration boundary value to :class:`Aggregation`."""

    if isinstance(value, Aggregation):
        return value
    try:
        return Aggregation(value)
    except ValueError as error:
        raise ValueError(f"Unsupported aggregation {value!r}") from error


def normalize_implementation(value: str | MixingImplementation) -> MixingImplementation:
    """Normalize a configuration boundary value to an implementation choice."""

    if isinstance(value, MixingImplementation):
        return value
    try:
        return MixingImplementation(value)
    except ValueError as error:
        raise ValueError(f"Unsupported mixing implementation {value!r}") from error


def enumerate_supports(
    n_particles: int,
    support_order: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Enumerate ordered distinct supports for one forward pass.

    The returned tensor is an execution cache only. It contains no parameters
    and does not affect path membership or module state.
    """

    return ordered_tuple_tensor(n_particles, support_order, distinct=True, device=device)


def build_unary_index_plan(
    path: SupportPath,
    n_particles: int,
    *,
    device: torch.device | str | None = None,
) -> UnaryIndexPlan:
    """Build typed output/input gather indices for one unary path."""

    supports = enumerate_supports(n_particles, path.support_order, device=device)
    output_indices = select_tuple_tensor(supports, path.tau_out)
    input_indices = select_tuple_tensor(supports, path.tau_in)
    return UnaryIndexPlan(
        supports=supports,
        output_indices=output_indices,
        input_indices=input_indices,
        flat_output_indices=flatten_tuple_indices(output_indices, n_particles),
    )


def build_binary_index_plan(
    path: VirtualPath,
    n_particles: int,
    *,
    device: torch.device | str | None = None,
) -> BinaryIndexPlan:
    """Build typed output/left/right gather indices for one TP path."""

    supports = enumerate_supports(n_particles, path.s, device=device)
    output_indices = select_tuple_tensor(supports, path.tau)
    left_indices = select_tuple_tensor(supports, path.tau1)
    right_indices = select_tuple_tensor(supports, path.tau2)
    return BinaryIndexPlan(
        supports=supports,
        output_indices=output_indices,
        left_indices=left_indices,
        right_indices=right_indices,
        flat_output_indices=flatten_tuple_indices(output_indices, n_particles),
    )


def _gather_tuple_values(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather dense tuple blocks in the plan's support order."""

    return source[(slice(None), slice(None), *indices.unbind(dim=1))]


def _reduce_completions(
    values: torch.Tensor,
    counts: torch.Tensor | None,
    aggregation: Aggregation,
) -> torch.Tensor:
    """Apply a zero-safe completion reduction to a path-resolved block."""

    if aggregation is Aggregation.SUM:
        return values
    if counts is None:
        raise RuntimeError("completion_mean requires completion counts")
    return values / counts.clamp_min(1).unsqueeze(0).unsqueeze(0)


def execute_unary(
    paths: tuple[SupportPath, ...],
    weights: tuple[torch.Tensor, ...],
    source: torch.Tensor,
    *,
    n_particles: int,
    output_order: int,
    batch_size: int,
    output_channels: int,
    aggregation: Aggregation,
    implementation: MixingImplementation,
    index_plans: tuple[UnaryIndexPlan, ...] | None = None,
) -> torch.Tensor:
    """Execute a parameter-free unary support-path contraction.

    Parameters
    ----------
    paths, weights : tuple
        Static path records and their already-validated eager weights.
    source : torch.Tensor
        One feature block with shape ``[batch, channels, ...tuple axes]``.
    n_particles : int
        Runtime particle count used only to build the index cache.
    output_channels : int
        Number of channels in the returned path block.
    aggregation : Aggregation
        Sum or zero-safe completion mean.
    implementation : MixingImplementation
        Literal or vectorized execution strategy.
    index_plans : tuple of UnaryIndexPlan, optional
        Per-forward typed index cache. When supplied, it must contain one plan
        per path; otherwise plans are built from the static paths for this
        execution. The cache contains no parameters or mutable module state.
    """

    aggregation = normalize_aggregation(aggregation)
    implementation = normalize_implementation(implementation)
    if len(paths) != len(weights):
        raise ValueError("unary paths and weights must have matching lengths")
    block = torch.zeros(
        (batch_size, output_channels, len(paths), *((n_particles,) * output_order)),
        device=source.device,
        dtype=source.dtype,
    )
    if not paths:
        return block
    counts = (
        torch.zeros((len(paths), *((n_particles,) * output_order)), device=source.device, dtype=source.dtype)
        if aggregation is Aggregation.COMPLETION_MEAN
        else None
    )
    plans = (
        tuple(build_unary_index_plan(path, n_particles, device=source.device) for path in paths)
        if index_plans is None
        else index_plans
    )
    if len(plans) != len(paths):
        raise ValueError("unary paths and index plans must have matching lengths")
    block_flat = block.reshape(*block.shape[:3], -1)
    for path_index, (path, weight) in enumerate(zip(paths, weights)):
        plan = plans[path_index]
        if implementation is MixingImplementation.SLOW:
            for output_tuple, input_tuple in zip(
                plan.output_indices.tolist(), plan.input_indices.tolist()
            ):
                input_value = source[(slice(None), slice(None), *input_tuple)]
                contribution = torch.einsum("oc,bc->bo", weight, input_value)
                block[(slice(None), slice(None), path_index, *output_tuple)] += contribution
                if counts is not None:
                    counts[(path_index, *output_tuple)] += 1
        else:
            input_values = _gather_tuple_values(source, plan.input_indices)
            contribution = torch.einsum("oc,bcv->bov", weight, input_values)
            block_flat[:, :, path_index].scatter_add_(
                2,
                plan.flat_output_indices.reshape(1, 1, -1).expand(source.shape[0], output_channels, -1),
                contribution,
            )
            if counts is not None:
                counts[path_index].reshape(-1).scatter_add_(
                    0,
                    plan.flat_output_indices,
                    torch.ones_like(plan.flat_output_indices, dtype=counts.dtype),
                )
    return _reduce_completions(block, counts, aggregation)


def execute_binary(
    paths: tuple[VirtualPath, ...],
    weights: tuple[torch.Tensor, ...],
    left: tuple[torch.Tensor, ...],
    right: tuple[torch.Tensor, ...],
    *,
    n_particles: int,
    output_order: int,
    batch_size: int,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
    output_channels: int,
    aggregation: Aggregation,
    implementation: MixingImplementation,
) -> torch.Tensor:
    """Execute a parameter-free bilinear virtual-support contraction."""

    aggregation = normalize_aggregation(aggregation)
    implementation = normalize_implementation(implementation)
    if not (len(paths) == len(weights) == len(left) == len(right)):
        raise ValueError("binary paths, weights, and input blocks must have matching lengths")
    block = torch.zeros(
        (batch_size, output_channels, len(paths), *((n_particles,) * output_order)),
        device=device,
        dtype=dtype,
    )
    if not paths:
        return block
    counts = (
        torch.zeros((len(paths), *((n_particles,) * output_order)), device=block.device, dtype=block.dtype)
        if aggregation is Aggregation.COMPLETION_MEAN
        else None
    )
    block_flat = block.reshape(*block.shape[:3], -1)
    for path_index, (path, weight, left_block, right_block) in enumerate(zip(paths, weights, left, right)):
        plan = build_binary_index_plan(path, n_particles, device=block.device)
        if implementation is MixingImplementation.SLOW:
            for output_tuple, left_tuple, right_tuple in zip(
                plan.output_indices.tolist(), plan.left_indices.tolist(), plan.right_indices.tolist()
            ):
                left_value = left_block[(slice(None), slice(None), *left_tuple)]
                right_value = right_block[(slice(None), slice(None), *right_tuple)]
                contribution = torch.einsum("ocd,bc,bd->bo", weight, left_value, right_value)
                block[(slice(None), slice(None), path_index, *output_tuple)] += contribution
                if counts is not None:
                    counts[(path_index, *output_tuple)] += 1
        else:
            left_values = _gather_tuple_values(left_block, plan.left_indices)
            right_values = _gather_tuple_values(right_block, plan.right_indices)
            contribution = torch.einsum("ocd,bcv,bdv->bov", weight, left_values, right_values)
            block_flat[:, :, path_index].scatter_add_(
                2,
                plan.flat_output_indices.reshape(1, 1, -1).expand(block.shape[0], output_channels, -1),
                contribution,
            )
            if counts is not None:
                counts[path_index].reshape(-1).scatter_add_(
                    0,
                    plan.flat_output_indices,
                    torch.ones_like(plan.flat_output_indices, dtype=counts.dtype),
                )
    return _reduce_completions(block, counts, aggregation)


__all__ = [
    "Aggregation",
    "BinaryIndexPlan",
    "MixingImplementation",
    "UnaryIndexPlan",
    "build_binary_index_plan",
    "build_unary_index_plan",
    "enumerate_supports",
    "execute_binary",
    "execute_unary",
    "normalize_aggregation",
    "normalize_implementation",
]
