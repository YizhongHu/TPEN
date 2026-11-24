"""Tests for MCMC-backed evaluation generators."""

from __future__ import annotations

from pathlib import Path

import torch

from tpen.data.batch import ElectronBatch, Walkers
from tpen.evaluation.generators import MCMCGenerator
from tpen.evaluation.protocols import EvaluationContext
from tpen.sampling import SamplerStats


class RecordingSampler:
    def collect_samples(self, model, *, device=None, emitter=None):
        del emitter
        positions = torch.zeros(2, 1, 3, dtype=torch.float64)
        walkers = Walkers(positions=positions)
        stats = SamplerStats(
            acceptance_rate=1.0,
            n_walkers=positions.shape[0],
            burn_in=0,
            n_steps=1,
            proposal_scale=0.1,
        )
        return walkers, stats


def test_mcmc_generator_seed_does_not_mutate_global_torch_rng() -> None:
    torch.manual_seed(999)
    before = torch.get_rng_state()
    generator = MCMCGenerator(sampler=RecordingSampler(), seed=123)
    context = EvaluationContext(
        namespace="eval",
        artifact_level="metrics_only",
        task_failure_policy="fail_fast",
        device=torch.device("cpu"),
        dtype=torch.float64,
        seed=123,
        run_dir=Path("/tmp"),
        task_output_dir=Path("/tmp"),
        metadata={},
    )

    generated = generator.generate(model=None, context=context)

    assert isinstance(generated.batch, ElectronBatch)
    assert generated.metadata["seed"] == 123
    # Bookkeeping metadata carries the typed record, not a flattened dict.
    assert isinstance(generated.metadata["sampler_stats"], SamplerStats)
    torch.testing.assert_close(torch.get_rng_state(), before)
