"""Exact radial occupancy and wrong-target controls independent of learned forwards.

Bands are frozen from independent analytic draws BEFORE examining any sampler
output. The final retained draw contains independent walkers; serially repeated
positions are never counted as independent observations in these bands. All 64
draws and their radial counts are retained for inspection, not substituted for
the learned-target energy trajectory endpoint.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.physics.one_electron import OneElectronExponential, OneElectronGaussian
from tpen.sampling.metropolis import MetropolisSampler

PROBABILITIES = np.array([.1, .25, .5, .75, .9, .99, .999])


def _cdf(system, radius):
    if system == "hooke":
        # Maxwell radius for three independent N(0,1) coordinates.
        return math.erf(radius / math.sqrt(2)) - math.sqrt(2 / math.pi) * radius * math.exp(-radius * radius / 2)
    return 1 - math.exp(-2 * radius) * (1 + 2 * radius + 2 * radius * radius)


def _quantile(system, probability):
    low, high = 0., 40.
    for _ in range(80):
        mid = (low + high) / 2
        if _cdf(system, mid) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def _independent_radii(system, rng, shape):
    if system == "hooke":
        return np.linalg.norm(rng.normal(size=(*shape, 3)), axis=-1)
    # Gamma(3, rate2) from three uniforms, independent of the reference forward.
    return -np.log(rng.uniform(size=(*shape, 3))).sum(axis=-1) / 2


def _deviation(counts):
    expected = 4096 * PROBABILITIES
    return np.max(np.abs(counts - expected) / np.sqrt(expected * (1 - PROBABILITIES)), axis=-1)


class _AmplitudeTarget(nn.Module):
    """Equivalent acceptance-ratio mutant: 2*delta(half logabs)=delta(logabs)."""

    def __init__(self, reference):
        super().__init__()
        self.reference = reference

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        output = self.reference(batch)
        return WavefunctionOutput(logabs=output.logabs / 2, sign=output.sign)


@pytest.mark.parametrize("system", ["hooke", "hydrogen"])
def test_sampler_radial_oracle_and_wrong_target_mutant(tmp_path, system):
    rng = np.random.default_rng(61001 if system == "hooke" else 61002)
    radii = np.array([_quantile(system, p) for p in PROBABILITIES])
    # Simulate 4096 independent TWO-stream tests of all seven CDF cuts.
    # Repeated 256-row chunks bound memory while preserving a fixed seed.
    deviations = []
    for _ in range(32):
        draws = _independent_radii(system, rng, (256, 4096))
        counts = (draws[..., None] <= radii).sum(axis=1)
        deviations.extend(_deviation(counts).tolist())
    maxima = np.asarray(deviations).reshape(4096, 2).max(axis=1)
    bound = float(np.quantile(maxima, .999, method="higher"))
    calibration = {"system": system, "seed": 61001 if system == "hooke" else 61002,
                   "probabilities": PROBABILITIES.tolist(), "radii": radii.tolist(),
                   "simultaneous_bound": bound, "replicates": 4096,
                   "assessment": "two_final_draws_with_independent_walkers"}
    (tmp_path / "calibration-before-sampler.json").write_text(json.dumps(calibration, indent=2))
    reference = OneElectronGaussian() if system == "hooke" else OneElectronExponential()
    outcomes = []
    for mutant in (False, True):
        model = _AmplitudeTarget(reference) if mutant else reference
        scores = []
        for stream in (0, 1):
            sampler = MetropolisSampler(n_walkers=4096, burn_in=1000, n_steps=20,
                                        proposal_scale=.5, seed=62001 + stream,
                                        n_electrons=1, spatial_dim=3, dtype=torch.float64)
            counts_by_draw = []
            for draw in range(64):
                walkers, _ = sampler.collect_samples(model, reset=(draw == 0), device="cpu")
                radius = walkers.positions.norm(dim=-1).reshape(-1).numpy()
                counts_by_draw.append((radius[:, None] <= radii).sum(axis=0))
            score = float(_deviation(counts_by_draw[-1]))
            scores.append(score)
            np.savez(tmp_path / f"{'mutant' if mutant else 'control'}-stream{stream}.npz",
                     radial_counts=np.asarray(counts_by_draw), final_positions=walkers.positions.numpy())
        outcomes.append({"mutant": mutant, "scores": scores, "passes": max(scores) <= bound})
    (tmp_path / "sampler-verdict.json").write_text(json.dumps({**calibration, "outcomes": outcomes}, indent=2))
    assert outcomes[0]["passes"], outcomes
    assert not outcomes[1]["passes"], outcomes
