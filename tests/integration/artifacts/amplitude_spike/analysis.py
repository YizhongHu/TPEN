"""Centered-logabs amplitude-risk reduction, per note amplitude-risk-design-20261106.

Pure NumPy. No torch import, so this module is usable to reduce already-written
CSV/NumPy raw tables without a TPEN environment.

Convention (frozen by amplitude-risk-decision-20261106, section 1):
for one batch/draw of N walkers, ``ell = logabs`` including every additive
factor. ``m`` is the ordinary median of the FINITE ``ell`` values (even count:
arithmetic midpoint of the two central sorted values). For a configured
negative offset ``tau``::

    low_count(tau) = count(finite ell_i AND ell_i - m < tau) + count(ell_i == -inf)
    amplitude_risk_fraction(tau) = low_count(tau) / N

The comparison is STRICT (``<``). NaN and +inf are reported separately and
retained in N, never counted as low. N=0 or no finite logabs is an explicit
unavailable status, never a healthy zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CANDIDATE_THRESHOLDS = (-3.0, -4.0, -5.0, -7.0, -10.0)
QUANTILE_LEVELS = (0.0001, 0.001, 0.01, 0.05, 0.5)


def _finite_median(values: np.ndarray) -> float | None:
    """Ordinary median of the finite entries; midpoint convention for even count."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    sorted_finite = np.sort(finite)
    n = sorted_finite.size
    mid = n // 2
    if n % 2:
        return float(sorted_finite[mid])
    return float((sorted_finite[mid - 1] + sorted_finite[mid]) / 2.0)


@dataclass
class DrawReduction:
    """One draw's (or one group's) centered-logabs reduction."""

    total_count: int
    finite_logabs_count: int
    nan_count: int
    pos_inf_count: int
    neg_inf_count: int
    sign_nonfinite_count: int
    sign_zero_count: int
    sign_other_count: int
    median: float | None
    quantiles: dict[float, float] = field(default_factory=dict)
    min_centered: float | None = None
    low_counts: dict[float, int] = field(default_factory=dict)
    fractions: dict[float, float | None] = field(default_factory=dict)
    unavailable: bool = False


def reduce_draw(
    ell: np.ndarray,
    sign: np.ndarray,
    *,
    thresholds: tuple[float, ...] = CANDIDATE_THRESHOLDS,
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS,
) -> DrawReduction:
    """Reduce one full draw/batch of aligned ``logabs``/``sign`` values.

    Parameters
    ----------
    ell : np.ndarray
        1-D ``logabs`` values, full sampled batch (no finite-energy mask applied).
    sign : np.ndarray
        1-D sign values aligned with ``ell``.
    """

    ell = np.asarray(ell, dtype=np.float64).reshape(-1)
    sign = np.asarray(sign, dtype=np.float64).reshape(-1)
    if ell.shape != sign.shape:
        raise ValueError(f"shape mismatch: ell {ell.shape} vs sign {sign.shape}")

    n_total = int(ell.size)
    nan_mask = np.isnan(ell)
    pos_inf_mask = np.isposinf(ell)
    neg_inf_mask = np.isneginf(ell)
    finite_mask = np.isfinite(ell)

    sign_nonfinite_mask = ~np.isfinite(sign)
    sign_zero_mask = np.isfinite(sign) & (sign == 0.0)
    sign_other_mask = np.isfinite(sign) & (sign != 0.0) & (sign != 1.0) & (sign != -1.0)

    median = _finite_median(ell)

    if n_total == 0 or median is None:
        return DrawReduction(
            total_count=n_total,
            finite_logabs_count=int(finite_mask.sum()),
            nan_count=int(nan_mask.sum()),
            pos_inf_count=int(pos_inf_mask.sum()),
            neg_inf_count=int(neg_inf_mask.sum()),
            sign_nonfinite_count=int(sign_nonfinite_mask.sum()),
            sign_zero_count=int(sign_zero_mask.sum()),
            sign_other_count=int(sign_other_mask.sum()),
            median=None,
            unavailable=True,
        )

    centered_finite = ell[finite_mask] - median
    quantiles = {
        level: float(np.quantile(centered_finite, level)) for level in quantile_levels
    }
    min_centered = float(np.min(centered_finite))

    low_counts: dict[float, int] = {}
    fractions: dict[float, float | None] = {}
    for tau in thresholds:
        low = int(np.count_nonzero(finite_mask & (ell - median < tau)))
        low += int(neg_inf_mask.sum())
        low_counts[tau] = low
        fractions[tau] = low / n_total if n_total > 0 else None

    return DrawReduction(
        total_count=n_total,
        finite_logabs_count=int(finite_mask.sum()),
        nan_count=int(nan_mask.sum()),
        pos_inf_count=int(pos_inf_mask.sum()),
        neg_inf_count=int(neg_inf_mask.sum()),
        sign_nonfinite_count=int(sign_nonfinite_mask.sum()),
        sign_zero_count=int(sign_zero_mask.sum()),
        sign_other_count=int(sign_other_mask.sum()),
        median=median,
        quantiles=quantiles,
        min_centered=min_centered,
        low_counts=low_counts,
        fractions=fractions,
    )


def low_event_mask(ell: np.ndarray, median: float, tau: float) -> np.ndarray:
    """Boolean mask of rows counted as a low-amplitude event at offset ``tau``."""

    ell = np.asarray(ell, dtype=np.float64)
    finite_mask = np.isfinite(ell)
    neg_inf_mask = np.isneginf(ell)
    return (finite_mask & (ell - median < tau)) | neg_inf_mask


def aggregate_stream(
    draws: list[DrawReduction],
    walker_indices: list[np.ndarray],
    ells: list[np.ndarray],
    *,
    thresholds: tuple[float, ...] = CANDIDATE_THRESHOLDS,
) -> dict:
    """Aggregate per-draw reductions into one stream's totals and distinct-walker counts."""

    total_n = sum(d.total_count for d in draws)
    totals = {
        "total_count": total_n,
        "finite_logabs_count": sum(d.finite_logabs_count for d in draws),
        "nan_count": sum(d.nan_count for d in draws),
        "pos_inf_count": sum(d.pos_inf_count for d in draws),
        "neg_inf_count": sum(d.neg_inf_count for d in draws),
        "sign_nonfinite_count": sum(d.sign_nonfinite_count for d in draws),
        "sign_zero_count": sum(d.sign_zero_count for d in draws),
        "sign_other_count": sum(d.sign_other_count for d in draws),
        "n_draws": len(draws),
    }
    per_tau: dict[float, dict] = {}
    for tau in thresholds:
        event_count = 0
        distinct_walkers: set[int] = set()
        for draw, walkers, ell in zip(draws, walker_indices, ells):
            if draw.median is None:
                continue
            mask = low_event_mask(ell, draw.median, tau)
            event_count += int(mask.sum())
            distinct_walkers.update(int(w) for w in np.asarray(walkers)[mask])
        per_tau[tau] = {
            "event_count": event_count,
            "distinct_walker_count": len(distinct_walkers),
            "fraction": (event_count / total_n) if total_n > 0 else None,
        }
    return {"totals": totals, "by_threshold": per_tau}


def block_bootstrap_fraction(
    draws: list[DrawReduction],
    walker_indices: list[np.ndarray],
    ells: list[np.ndarray],
    *,
    thresholds: tuple[float, ...] = CANDIDATE_THRESHOLDS,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Whole-draw block bootstrap over the fraction statistic, per predeclared tau.

    Resamples whole draws (every draw keeps its own walkers and its own
    already-computed per-draw median together as one block), recomputing the
    aggregate fraction(tau) each replicate. Reports mean/std/2.5%/97.5% of the
    bootstrap distribution, or an explicit ``insufficient_resolution`` flag
    when every replicate has zero events (the statistic cannot resolve a rate
    from zero hits alone).
    """

    rng = np.random.default_rng(seed)
    n_draws = len(draws)
    results: dict[float, dict] = {}
    if n_draws == 0:
        return {tau: {"insufficient_resolution": True} for tau in thresholds}

    # Precompute per-draw, per-tau (event_count, n) so resampling is cheap.
    per_draw_tau: dict[float, list[tuple[int, int]]] = {tau: [] for tau in thresholds}
    for draw, walkers, ell in zip(draws, walker_indices, ells):
        n = draw.total_count
        for tau in thresholds:
            if draw.median is None:
                per_draw_tau[tau].append((0, n))
                continue
            mask = low_event_mask(ell, draw.median, tau)
            per_draw_tau[tau].append((int(mask.sum()), n))

    for tau in thresholds:
        pairs = per_draw_tau[tau]
        events = np.array([p[0] for p in pairs], dtype=np.float64)
        counts = np.array([p[1] for p in pairs], dtype=np.float64)
        boot_fracs = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            idx = rng.integers(0, n_draws, size=n_draws)
            e = events[idx].sum()
            c = counts[idx].sum()
            boot_fracs[b] = (e / c) if c > 0 else np.nan
            if b == 0:
                pass
        valid = boot_fracs[np.isfinite(boot_fracs)]
        total_events = int(events.sum())
        if total_events == 0:
            results[tau] = {
                "insufficient_resolution": True,
                "observed_event_count": 0,
                "mean": 0.0,
                "std": 0.0,
                "p2_5": 0.0,
                "p97_5": 0.0,
            }
            continue
        results[tau] = {
            "insufficient_resolution": False,
            "observed_event_count": total_events,
            "mean": float(np.mean(valid)) if valid.size else None,
            "std": float(np.std(valid)) if valid.size else None,
            "p2_5": float(np.quantile(valid, 0.025)) if valid.size else None,
            "p97_5": float(np.quantile(valid, 0.975)) if valid.size else None,
        }
    return results


def group_within_draw(
    ell: np.ndarray,
    sign: np.ndarray,
    walker_index: np.ndarray,
    *,
    group_size: int = 256,
    thresholds: tuple[float, ...] = CANDIDATE_THRESHOLDS,
) -> list[DrawReduction]:
    """Split one draw into fixed walker-index groups and center WITHIN each group.

    Groups are correlated reductions of the same draw (quantifying the
    training-batch median effect at 256 walkers), never reclassified as
    independent data.
    """

    ell = np.asarray(ell, dtype=np.float64)
    sign = np.asarray(sign, dtype=np.float64)
    walker_index = np.asarray(walker_index, dtype=np.int64)
    order = np.argsort(walker_index)
    ell_sorted = ell[order]
    sign_sorted = sign[order]
    n = ell_sorted.size
    groups = []
    for start in range(0, n, group_size):
        stop = min(start + group_size, n)
        groups.append(reduce_draw(ell_sorted[start:stop], sign_sorted[start:stop], thresholds=thresholds))
    return groups


__all__ = [
    "CANDIDATE_THRESHOLDS",
    "QUANTILE_LEVELS",
    "DrawReduction",
    "reduce_draw",
    "low_event_mask",
    "aggregate_stream",
    "block_bootstrap_fraction",
    "group_within_draw",
]
