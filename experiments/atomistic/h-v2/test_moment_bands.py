"""Unit controls for the calibrated imperfect-state moment bands (`moment_bands.py`).

Isolated from the exact-hydrogen canary so a defect in the calibration
itself -- rather than in the canary's evaluation route -- is diagnosable on
its own. See ``test_hydrogen_exact_canary.py`` for the canary's own use of
``calibrate_moment_band`` against its final-draw moments.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"h_v2_{name}", HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


moment_bands = _module("moment_bands")
oracle = _module("analytic_oracle")

# Must match the exact-hydrogen canary's final retained draw walker count
# (`N_WALKERS` in `test_hydrogen_exact_canary.py`): walkers are independent
# after burn-in, while successive draws of one walker are not, so only a
# walker count is a valid calibration sample size.
N_WALKERS = 1024


def test_calibrated_band_contains_the_closed_form_moments() -> None:
    band = moment_bands.calibrate_moment_band(alpha=1.2, sample_count=N_WALKERS, seed=120001)
    control = oracle.analytic_control(1.2)
    assert control["born_mean_hartree"] == pytest.approx(-0.48)
    assert control["variance_hartree2"] == pytest.approx(0.0576)
    assert band[0, 0] <= control["born_mean_hartree"] <= band[1, 0]
    assert band[0, 1] <= control["variance_hartree2"] <= band[1, 1]


def test_calibrated_band_rejects_nonpositive_sizes() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        moment_bands.calibrate_moment_band(alpha=1.2, sample_count=0, seed=1)
    with pytest.raises(ValueError, match="replicate_count"):
        moment_bands.calibrate_moment_band(alpha=1.2, sample_count=8, seed=1, replicate_count=0)


def test_calibrated_band_reproducible_for_a_fixed_seed() -> None:
    first = moment_bands.calibrate_moment_band(alpha=1.2, sample_count=256, seed=7, replicate_count=64)
    second = moment_bands.calibrate_moment_band(alpha=1.2, sample_count=256, seed=7, replicate_count=64)
    assert (first == second).all()
