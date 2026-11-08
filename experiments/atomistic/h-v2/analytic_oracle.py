"""Independent scalar controls for the one-electron hydrogen protocol.

The module intentionally uses only the Python standard library.  Its formulas
are not imported from :mod:`tpen`; the configured TPEN route is tested
separately and is the subject of the mutation controls.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable


def local_energy(radius: float, alpha: float = 1.0, charge: float = 1.0) -> float:
    """Return the scalar local energy for ``psi=exp(-alpha*r)`` and ``-Z/r``.

    ``radius == 0`` is undefined for the Coulomb term and is rejected rather
    than regularized.
    """

    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("hydrogen local energy requires a finite radius > 0")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    if not math.isfinite(charge) or charge <= 0.0:
        raise ValueError("charge must be finite and positive")
    return -0.5 * alpha * alpha + (alpha - charge) / radius


def gamma3_cdf(radius: float, alpha: float = 1.0) -> float:
    """Return the Born radial CDF for ``Gamma(shape=3, rate=2*alpha)``."""

    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and nonnegative")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    t = 2.0 * alpha * radius
    return 1.0 - math.exp(-t) * (1.0 + t + 0.5 * t * t)


def gamma3_quantile(probability: float, alpha: float = 1.0) -> float:
    """Invert :func:`gamma3_cdf` by a deterministic bisection."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    low, high = 0.0, 1.0 / alpha
    while gamma3_cdf(high, alpha) < probability:
        high *= 2.0
    for _ in range(120):
        middle = (low + high) / 2.0
        if gamma3_cdf(middle, alpha) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def sample_born_radii(count: int, seed: int) -> list[float]:
    """Draw independent Born radii as sums of three rate-two exponentials."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    rng = random.Random(seed)
    return [-math.log(rng.random()) / 2.0 - math.log(rng.random()) / 2.0 - math.log(rng.random()) / 2.0 for _ in range(count)]


def _absolute_deviation_survival(threshold: float, alpha: float) -> float:
    """Return P(|E_L - median(E_L)| > threshold) in closed form."""

    if threshold < 0.0 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite and nonnegative")
    if alpha == 1.0:
        return 0.0 if threshold >= 0.0 else 1.0
    median_radius = gamma3_quantile(0.5, alpha)
    coefficient = abs(alpha - 1.0)
    median_value = local_energy(median_radius, alpha)
    # E(r) is monotone in 1/r.  The two inverse-radius crossings bound the
    # central interval; either crossing can be outside the physical domain.
    lower_inverse = 1.0 / median_radius + threshold / coefficient
    upper_inverse = 1.0 / median_radius - threshold / coefficient
    lower_mass = gamma3_cdf(1.0 / lower_inverse, alpha)
    upper_mass = 0.0 if upper_inverse <= 0.0 else 1.0 - gamma3_cdf(1.0 / upper_inverse, alpha)
    # Keep the explicit median calculation live: it guards accidental changes
    # to the centering convention in this independent instrument.
    if not math.isfinite(median_value):
        raise ArithmeticError("nonfinite analytic median")
    return lower_mass + upper_mass


def closed_form_w(alpha: float = 1.2, probability: float = 0.999) -> float:
    """Return the no-interpolation ``P_probability`` absolute deviation."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be strictly between zero and one")
    if alpha == 1.0:
        return 0.0
    low, high = 0.0, 1.0
    while _absolute_deviation_survival(high, alpha) > 1.0 - probability:
        high *= 2.0
    for _ in range(120):
        middle = (low + high) / 2.0
        if _absolute_deviation_survival(middle, alpha) > 1.0 - probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def exceedance_probability(threshold: float, alpha: float = 1.2) -> float:
    """Return the exact absolute-deviation exceedance probability."""

    return _absolute_deviation_survival(threshold, alpha)


def zero_event_upper_bound(n: int, confidence: float = 0.975) -> float:
    """Return a one-sided binomial upper bound when zero events were seen."""

    if n <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("n must be positive and confidence must be in (0, 1)")
    return 1.0 - (1.0 - confidence) ** (1.0 / n)


def analytic_control(alpha: float) -> dict[str, float | int | str]:
    """Return reproducible exact-state controls for one selected ``alpha``."""

    if alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    # Under the Born density E[1/r] = alpha for Gamma(3, 2*alpha).
    mean = 0.5 * alpha * alpha - alpha
    variance = alpha * alpha * (alpha - 1.0) ** 2
    median = local_energy(gamma3_quantile(0.5, alpha), alpha)
    return {
        "alpha": alpha,
        "born_mean_hartree": mean,
        "variance_hartree2": variance,
        "median_local_energy_hartree": median,
        "w_p99_9_hartree": closed_form_w(alpha),
        "exceedance_gt_0.01": exceedance_probability(0.01, alpha),
        "exceedance_gt_1": exceedance_probability(1.0, alpha),
        "radius_zero": "rejected",
    }


__all__ = [
    "analytic_control",
    "closed_form_w",
    "exceedance_probability",
    "gamma3_cdf",
    "gamma3_quantile",
    "local_energy",
    "sample_born_radii",
    "zero_event_upper_bound",
]
