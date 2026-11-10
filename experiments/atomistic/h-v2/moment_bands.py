"""Calibrated empirical moment bands for the imperfect (alpha != 1) one-electron
hydrogen state.

Standard library plus NumPy only, per the experiment boundary -- the scalar
oracle math itself is deferred to ``analytic_oracle.local_energy``, never to
a learned or core forward. Used by the exact-hydrogen canary
(``test_hydrogen_exact_canary.py``) to bound its final-draw mean/variance,
and exercised directly by this module's own calibration test
(``test_moment_bands.py``).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"h_v2_{name}", HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_oracle = _module("analytic_oracle")


def calibrate_moment_band(
    *, alpha: float, sample_count: int, seed: int, replicate_count: int = 1024,
) -> np.ndarray:
    """Return the [0.001, 0.999] replicate-quantile band of (mean, variance).

    Each of ``replicate_count`` replicates draws ``sample_count`` independent
    walkers: a radius from ``Gamma(shape=3, rate=2*alpha)`` -- the exact Born
    radial density under the imperfect one-electron hydrogen state -- paired
    with a uniformly random direction, matching the sampling convention in
    ``experiments/one_electron_variance/test_reference_controls.py::
    test_imperfect_reference_empirical_moments_against_independent_draws``.
    The oracle's pointwise local energy (``analytic_oracle.local_energy``) is
    evaluated at each walker's radius; the returned array's rows are the
    0.001 and 0.999 replicate quantiles, and its columns are the sample mean
    and the sample variance, i.e. ``band[0]`` is the lower edge
    ``(mean_low, variance_low)`` and ``band[1]`` is the upper edge.

    ``sample_count`` must equal the number of independent walkers in the
    consumer's final retained draw (callers are responsible for that
    equality; walkers are independent after burn-in while successive draws
    of one walker are not, so only a walker count is a valid calibration
    sample size here).
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if replicate_count <= 0:
        raise ValueError("replicate_count must be positive")

    rng = np.random.default_rng(seed)
    moments = np.empty((replicate_count, 2), dtype=np.float64)
    for replicate in range(replicate_count):
        # Sum of three rate-(2*alpha) exponentials == Gamma(shape=3, rate=2*alpha).
        radius = -np.log(rng.uniform(size=(sample_count, 3))).sum(axis=-1) / (2.0 * alpha)
        direction = rng.normal(size=(sample_count, 3))
        unit_direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
        positions = radius[:, None] * unit_direction
        walker_radii = np.linalg.norm(positions, axis=-1)
        energy = np.fromiter(
            (_oracle.local_energy(float(r), alpha=alpha) for r in walker_radii),
            dtype=np.float64, count=sample_count,
        )
        moments[replicate, 0] = energy.mean()
        moments[replicate, 1] = energy.var()
    return np.quantile(moments, [0.001, 0.999], axis=0)


__all__ = ["calibrate_moment_band"]
