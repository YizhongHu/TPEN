"""Calibration controls proposed by the PR 622 T2 reviewer.

These tests have not been run locally. Run the unmodified head and the named
calibration mutants on Cannon before accepting this review packet.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent


@pytest.fixture
def bands_module():
    """Load an isolated study module so oracle patches cannot leak to the canary."""

    spec = importlib.util.spec_from_file_location(
        "h_v2_review_r1_moment_bands", HERE / "moment_bands.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The study loader temporarily publishes its scalar oracle in sys.modules.
    # Restore any prior entry so unrelated tests retain their original module.
    previous_oracle = sys.modules.get("h_v2_analytic_oracle")
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        if previous_oracle is None:
            sys.modules.pop("h_v2_analytic_oracle", None)
        else:
            sys.modules["h_v2_analytic_oracle"] = previous_oracle


def test_default_band_reduces_pointwise_oracle_samples_at_contract_quantiles(
    bands_module, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kill broad constants, min/max bands, wrong replicate count and ddof=1.

    A controlled oracle gives replicate j the five energies
    j + (j+1)*[-2, -1, 0, 1, 2]. Its mean is j and population variance is
    2*(j+1)**2. Thus both ordered moment columns and the two interpolated
    endpoints are known without calling the implementation's reducers.
    """

    calls = 0

    def controlled_energy(radius: float, *, alpha: float) -> float:
        nonlocal calls
        assert math.isfinite(radius) and radius > 0.0
        assert alpha == 1.2
        replicate, walker = divmod(calls, 5)
        calls += 1
        return float(replicate + (replicate + 1) * (walker - 2))

    monkeypatch.setattr(bands_module._oracle, "local_energy", controlled_energy)
    band = bands_module.calibrate_moment_band(alpha=1.2, sample_count=5, seed=120001)

    assert calls == 1024 * 5
    assert band.shape == (2, 2)
    # Default linear quantiles use zero-based positions 1.023 and 1021.977.
    expected = np.array([
        [1.023, 0.977 * 2 * 2**2 + 0.023 * 2 * 3**2],
        [1021.977, 0.023 * 2 * 1022**2 + 0.977 * 2 * 1023**2],
    ])
    np.testing.assert_allclose(band, expected, rtol=1e-12, atol=1e-12)


def test_oracle_receives_independent_gamma3_rate_2_4_radii(
    bands_module, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kill a wrong radial rate/shape or reuse of one draw in every replicate.

    The exact Gamma(3, 2.4) CDF maps the input radii to Uniform(0, 1).
    Check its decile masses independently of the log-uniform construction;
    0.02 mass slack is over eight binomial standard errors at this size.
    """

    radii = []
    scalar_energy = bands_module._oracle.local_energy

    def record_radius(radius: float, *, alpha: float) -> float:
        radii.append(radius)
        return scalar_energy(radius, alpha=alpha)

    monkeypatch.setattr(bands_module._oracle, "local_energy", record_radius)
    bands_module.calibrate_moment_band(
        alpha=1.2, sample_count=64, seed=120003, replicate_count=256,
    )

    assert len(radii) == 64 * 256
    # An exact duplicate is a strong regression signal for reused RNG state.
    assert len(set(radii)) == len(radii)
    scaled = 2.4 * np.asarray(radii)
    probabilities = 1.0 - np.exp(-scaled) * (1.0 + scaled + scaled**2 / 2.0)
    masses = np.histogram(probabilities, bins=np.linspace(0.0, 1.0, 11))[0] / len(radii)
    np.testing.assert_allclose(masses, 0.1, rtol=0.0, atol=0.02)
