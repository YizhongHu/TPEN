"""Smoke test: the Hooke pair sampler yields typed, fixed-spin walkers."""

from __future__ import annotations

import torch

from tpen.sampling import SamplerStats
from tests.helpers.hooke_models import build_tiny_sampler, build_tiny_spenn


def test_sampler_produces_typed_walkers_with_fixed_spin() -> None:
    model = build_tiny_spenn()
    sampler = build_tiny_sampler()

    walkers, stats = sampler.collect_samples(model, emitter=None)
    batch = walkers.make_batch()

    n_walkers = batch.positions.shape[0]
    assert batch.positions.shape == (n_walkers, 2, 3)
    assert batch.spins is not None
    assert batch.spins.shape == (n_walkers, 2)
    # Fixed (1 up, 1 down) population, preserved across MCMC steps.
    expected = torch.tensor([1.0, -1.0], dtype=torch.float64)
    assert torch.equal(batch.spins[0], expected)
    assert torch.all(batch.spins == expected)

    assert isinstance(stats, SamplerStats)
    assert 0.0 <= stats.acceptance_rate <= 1.0
    assert stats.n_walkers == n_walkers
