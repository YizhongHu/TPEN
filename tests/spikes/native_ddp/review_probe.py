"""Small independent probes used by the round-one reviewer tests.

This module deliberately contains no TPEN imports and no distributed launch
logic.  It supplies an execution-bound call counter for the sampler and
kinetic raw-model paths, which the worker's derived telemetry cannot provide.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tests.spikes.native_ddp.model_access import SemanticWavefunction
from tests.spikes.native_ddp.sampler import RankLocalSampler
from tests.spikes.native_ddp.seed import SeedPartition


@dataclass(frozen=True)
class RawCallCounts:
    """Observed raw-model calls for sampling and kinetic work."""

    sampling: int
    kinetic: int


def count_raw_model_calls(*, mcmc_steps: int, kinetic_forwards: int) -> RawCallCounts:
    """Count calls made by actual sampler and kinetic execution paths."""

    if mcmc_steps < 0 or kinetic_forwards < 0:
        raise ValueError("work counts must be nonnegative")
    model = SemanticWavefunction()
    calls = {"sampling": 0, "kinetic": 0}

    def count_sampling(_module, _inputs, _output) -> None:
        calls["sampling"] += 1

    model.register_forward_hook(count_sampling)
    sampler = RankLocalSampler(SeedPartition(base_seed=50_000, rank=0, world_size=2))
    sampler.advance(model, mcmc_steps)
    sampling_calls = calls["sampling"]

    for _ in range(kinetic_forwards):
        coordinates = torch.zeros((4, 2), dtype=torch.float64, requires_grad=True)
        model(coordinates)
        calls["kinetic"] += 1
    return RawCallCounts(sampling=sampling_calls, kinetic=calls["kinetic"])


__all__ = ["RawCallCounts", "count_raw_model_calls"]
