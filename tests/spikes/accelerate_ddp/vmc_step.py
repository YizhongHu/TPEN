"""Accelerate-backed VMC score-function step and reduction instrumentation.

Mirrors ``tests/spikes/native_ddp/vmc_step.py`` step for step, so a difference in
outcome between the two spikes is attributable to the runtime rather than to the
step's structure. Two places diverge, both deliberately and both recorded:
``accelerator.backward`` replaces ``Tensor.backward``, and the reduction counter
is installed on Accelerate's prepared wrapper rather than on a DDP object TPEN
constructed itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from tests.spikes.accelerate_ddp.model_access import ModelAccess
from tests.spikes.accelerate_ddp.runtime import DistributedRuntime
from tests.spikes.accelerate_ddp.statistics import (
    FiniteStatistics,
    centered_terms,
    reduce_statistics,
)


@dataclass
class GradientReductionCounter:
    """Count only gradient-reducer buckets, not ordinary diagnostic collectives.

    Registered as a DDP communication hook on the model Accelerate prepared. The
    hook performs the all-reduce and the world-size division EXPLICITLY rather
    than delegating, so the averaging convention A-G1 asserts is visible in this
    file instead of being an assumption about what the wrapper does.
    """

    world_size: int
    count: int = 0

    def communication_hook(self, _state, bucket):
        """All-reduce one bucket and perform the DDP average explicitly."""

        self.count += 1
        buffer = bucket.buffer()
        torch.distributed.all_reduce(buffer, op=torch.distributed.ReduceOp.SUM)
        buffer.div_(self.world_size)
        future = torch.futures.Future()
        future.set_result(buffer)
        return future


@dataclass(frozen=True)
class StepObservation:
    """Scalar and tensor evidence emitted by one Accelerate-backed update."""

    stats: FiniteStatistics
    global_loss: float
    local_surrogate_loss: float
    scale_factor: float
    local_gradients: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor]
    gradient_reductions: int
    coordinate_gradient: torch.Tensor
    logabs: torch.Tensor
    gradient_accumulation_steps: int


def install_gradient_counter(
    access: ModelAccess, runtime: DistributedRuntime
) -> GradientReductionCounter:
    """Install one reducer hook on the model Accelerate prepared.

    Raises rather than degrading if the prepared model exposes no
    ``register_comm_hook``. A counter that silently failed to install would read
    zero reductions under every workload and make A-G6's invariant pass
    vacuously -- the exact shape A-G6 exists to rule out.
    """

    counter = GradientReductionCounter(world_size=runtime.world_size)
    register = getattr(access.prepared_model, "register_comm_hook", None)
    if register is None:
        raise RuntimeError(
            "the prepared model exposes no register_comm_hook, so gradient "
            "reductions cannot be counted; A-G6 would pass vacuously"
        )
    register(counter, counter.communication_hook)
    return counter


def prepare_statistics(runtime: DistributedRuntime, energy: torch.Tensor) -> FiniteStatistics:
    """Run the rank-statistics collective before any backward call."""

    return reduce_statistics(runtime, energy)


def make_local_surrogate(
    logabs: torch.Tensor,
    energy: torch.Tensor,
    stats: FiniteStatistics,
    *,
    world_size: int,
) -> tuple[torch.Tensor, float]:
    """Build the W-scaled local surrogate and return its scale.

    The ``W`` factor exists because the reducer AVERAGES gradients across ranks.
    Each rank backpropagates ``2W/M * sum_i(m_i (E_i - mu) logabs_i)`` so that the
    post-average gradient equals the global ``2/M * sum`` over the concatenation.
    ``mu`` and ``M`` are GLOBAL, from the collective that ran before this call --
    rank-local centering is the documented wrong answer and is what A-G1b shows
    failing.
    """

    if stats.finite_count == 0:
        # The empty shard remains parameter-connected, but global M == 0 is
        # refused by the worker before this function is used for backward.
        return (logabs * 0.0).sum(), 0.0
    scale = 2.0 * world_size / stats.finite_count
    return scale * centered_terms(logabs, energy, stats).sum(), scale


def run_score_function_step(
    access: ModelAccess,
    runtime: DistributedRuntime,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    energy: torch.Tensor,
    stats: FiniteStatistics,
    counter: GradientReductionCounter,
    *,
    before_backward: Callable[[], None] | None = None,
) -> StepObservation:
    """Run exactly one Accelerate-backed score update after coordinate work."""

    # Coordinate autograd on the RAW module: a prepared forward here would enlist
    # kinetic work in gradient synchronization, which A-G6 forbids.
    _, coordinate_gradient = access.coordinate_forward(features)

    # Diagnostic-only local gradients, computed off the raw module through
    # torch.autograd.grad so they never touch .grad and never trigger a reduction.
    raw_logabs = access.raw_model(features)
    diagnostic_surrogate, scale = make_local_surrogate(
        raw_logabs, energy, stats, world_size=runtime.world_size
    )
    local_gradients = {
        name: gradient.detach().clone()
        for (name, _parameter), gradient in zip(
            access.raw_model.named_parameters(),
            torch.autograd.grad(
                diagnostic_surrogate,
                tuple(access.raw_model.parameters()),
                allow_unused=True,
            ),
            strict=True,
        )
        if gradient is not None
    }

    optimizer.zero_grad(set_to_none=True)
    logabs = access.score_forward(features)
    local_surrogate, scale = make_local_surrogate(
        logabs, energy, stats, world_size=runtime.world_size
    )
    if before_backward is not None:
        before_backward()

    # THE ACCELERATE-SPECIFIC HAZARD A-G1 MUST CLOSE. `Accelerator.backward`
    # divides the loss by `gradient_accumulation_steps` before calling backward.
    # At the default of 1 that is a no-op, but an unexamined non-default would
    # apply a hidden 1/n on top of the reducer's own averaging and corrupt every
    # gradient comparison while every arm still passed. The value is asserted
    # here AND carried into the observation, so the receipt records what was in
    # force rather than what was assumed.
    accumulation_steps = int(getattr(runtime.accelerator, "gradient_accumulation_steps", 1))
    if accumulation_steps != 1:
        raise RuntimeError(
            "gradient_accumulation_steps must be 1 for the score-function step: "
            f"got {accumulation_steps}, which would scale the loss by 1/{accumulation_steps} "
            "on top of the reducer average"
        )
    runtime.accelerator.backward(local_surrogate)

    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in access.raw_model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()

    local_term = float(centered_terms(logabs.detach(), energy, stats).sum().item())
    global_term = sum(float(value) for value in runtime.all_gather_objects(local_term))
    # The PUBLISHED loss is the global scientific loss, never the world-size
    # compensated surrogate that was actually backpropagated.
    global_loss = 2.0 * global_term / stats.finite_count if stats.finite_count else 0.0
    return StepObservation(
        stats=stats,
        global_loss=global_loss,
        local_surrogate_loss=float(local_surrogate.detach().item()),
        scale_factor=scale,
        local_gradients=local_gradients,
        gradients=gradients,
        gradient_reductions=counter.count,
        coordinate_gradient=coordinate_gradient.detach(),
        logabs=logabs.detach(),
        gradient_accumulation_steps=accumulation_steps,
    )


def run_closure_step(
    access: ModelAccess,
    runtime: DistributedRuntime,
    optimizer: torch.optim.LBFGS,
    features: torch.Tensor,
    energy: torch.Tensor,
    stats: FiniteStatistics,
    counter: GradientReductionCounter,
) -> tuple[StepObservation, int, int]:
    """Run a globally synchronized closure through ``accelerator.backward``.

    A-E3's question, and the reason it must be ESTABLISHED rather than inherited
    from the sibling lane: what does Accelerate's abstraction do with a
    closure-driven optimizer -- does it HIDE, EXPOSE, or FORBID the re-entrant
    backward?

    What is measured here: whether ``accelerator.backward`` can be called
    repeatedly inside one ``optimizer.step(closure)``, and how many gradient
    reductions that costs per closure call.

    THE POLICY IS DELIBERATE, not incidental. LBFGS mutates parameters and its
    history after EVERY closure return, so a rank whose gradient was not globally
    reduced would diverge from its peers immediately, and a later synchronized
    backward cannot repair optimizer state that has already advanced. Suppressing
    synchronization -- via ``no_sync`` or otherwise -- is therefore unsafe on this
    path, and early stopping is safe only because there is no specially designated
    unsynchronized final call to miss. Every state-mutating iterate pays one
    reduction. That cost is a DG0 input, not something to hide.
    """

    closure_calls = 0
    synchronized_calls = 0

    def closure() -> torch.Tensor:
        nonlocal closure_calls, synchronized_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        logabs = access.score_forward(features)
        local_surrogate, _ = make_local_surrogate(
            logabs, energy, stats, world_size=runtime.world_size
        )
        runtime.accelerator.backward(local_surrogate)
        local_term = float(centered_terms(logabs.detach(), energy, stats).sum().item())
        global_term = sum(float(v) for v in runtime.all_gather_objects(local_term))
        synchronized_calls += 1
        # Return the GLOBAL objective, so every rank's optimizer sees the same
        # value and cannot take a different search direction from its peers.
        return torch.tensor(
            2.0 * global_term / stats.finite_count, dtype=local_surrogate.dtype
        )

    optimizer.step(closure)

    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in access.raw_model.named_parameters()
        if parameter.grad is not None
    }
    raw_logabs, coordinate_gradient = access.coordinate_forward(features)
    local_surrogate, scale = make_local_surrogate(
        raw_logabs, energy, stats, world_size=runtime.world_size
    )
    local_term = float(centered_terms(raw_logabs.detach(), energy, stats).sum().item())
    global_term = sum(float(v) for v in runtime.all_gather_objects(local_term))
    observation = StepObservation(
        stats=stats,
        global_loss=2.0 * global_term / stats.finite_count,
        local_surrogate_loss=float(local_surrogate.detach().item()),
        scale_factor=scale,
        local_gradients={},
        gradients=gradients,
        gradient_reductions=counter.count,
        coordinate_gradient=coordinate_gradient.detach(),
        logabs=raw_logabs.detach(),
        gradient_accumulation_steps=int(
            getattr(runtime.accelerator, "gradient_accumulation_steps", 1)
        ),
    )
    return observation, closure_calls, synchronized_calls


__all__ = [
    "GradientReductionCounter",
    "StepObservation",
    "install_gradient_counter",
    "make_local_surrogate",
    "prepare_statistics",
    "run_closure_step",
    "run_score_function_step",
]
