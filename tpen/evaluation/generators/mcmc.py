"""MCMC-backed evaluation generators."""

from __future__ import annotations

from typing import Any

import torch

from tpen.evaluation.bundle import GeneratedConfigurations
from tpen.evaluation.protocols import EvaluationContext
from tpen.sampling.protocols import SamplingCollector


class MCMCGenerator:
    """Generate evaluation configurations from an existing sampler."""

    name = "mcmc"

    def __init__(
        self,
        *,
        sampler: SamplingCollector,
        seed: int | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.sampler = sampler
        self.seed = None if seed is None else int(seed)
        self.max_samples = None if max_samples is None else int(max_samples)

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Collect sampler configurations and expose bookkeeping metadata."""

        walkers, sampler_stats = self.sampler.collect_samples(
            model, device=context.device, emitter=context.emitter
        )
        batch = walkers.make_batch().flatten_samples()
        if self.max_samples is not None and self.max_samples >= 0:
            batch = _slice_batch(batch, 0, min(self.max_samples, batch.batch_size))
        sample_index = torch.arange(batch.batch_size, device=batch.device)
        # The typed SamplerStats record is stored as-is; flattening to metric
        # names is owned by the record and happens at the summary edge.
        metadata: dict[str, Any] = {
            "sample_index": sample_index,
            "sampler_stats": sampler_stats,
        }
        if self.seed is not None:
            metadata["seed"] = self.seed
        if "walker_index" not in metadata:
            metadata["walker_index"] = sample_index
        return GeneratedConfigurations(batch=batch, metadata=metadata)


def _slice_batch(batch, start: int, end: int):
    from tpen.evaluation.calculators.local_energy import slice_flat_batch

    return slice_flat_batch(batch.flatten_samples(), start, end)


__all__ = ["MCMCGenerator"]
