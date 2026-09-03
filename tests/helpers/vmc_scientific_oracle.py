"""Process-free scientific oracle for VMC DDP correctness (DF0/DF1).

Independent float64 reference implementing the mathematics of
``ddp-scientific-oracle-2026-08-31``: one distributed VMC iteration must equal
one single-process iteration over the stable concatenation of every
rank-local sample. This module NEVER imports or calls
``tpen.training.vmc.compute_vmc_objective``, which is the SUBJECT these tests
exist to check, not a reference for it.

Global reductions never average per-shard means. Every reduction merges
mergeable count/mean/M2 (Chan/Welford) sufficient-statistic packets, so the
merge algebra itself -- not just the final number -- is what a caller tests
against.

Tolerance policy (``ddp-scientific-oracle-2026-08-31``): floating comparisons
use a documented summation-error envelope derived from machine epsilon and
``sum(abs(x))``, never a bare unexplained epsilon. See
:func:`summation_error_envelope` and :func:`loss_tolerance_envelope` for the
derivation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

Tensor = torch.Tensor


# --- mergeable sufficient statistics (Chan/Welford) -------------------------


@dataclass(frozen=True)
class SufficientStatistics:
    """Mergeable count/mean/M2 packet (parallel Chan/Welford combination).

    A zero-count packet is the identity element of :meth:`merge`.

    Parameters
    ----------
    count : int
        Number of samples this packet summarizes.
    mean : float
        Sample mean. Meaningless when ``count == 0`` (any value is discarded
        by :meth:`merge`, since a zero-count packet is the identity).
    m2 : float
        Sum of squared deviations from ``mean``, i.e. ``sum((x_i - mean)**2)``.
        ``variance = m2 / count`` (population variance).
    """

    count: int
    mean: float
    m2: float

    @property
    def variance(self) -> float:
        """Return the population variance, or ``0.0`` for an empty packet."""

        return self.m2 / self.count if self.count > 0 else 0.0

    @classmethod
    def from_values(cls, values: Tensor) -> "SufficientStatistics":
        """Build a packet directly from a 1-D float64 tensor of samples."""

        if values.dtype != torch.float64:
            raise TypeError("SufficientStatistics.from_values requires float64")
        count = int(values.numel())
        if count == 0:
            return cls.EMPTY
        mean = float(values.mean().item())
        m2 = float(((values - mean) ** 2).sum().item())
        return cls(count=count, mean=mean, m2=m2)

    def merge(self, other: "SufficientStatistics") -> "SufficientStatistics":
        """Combine two independently accumulated packets.

        Parallel Chan/Welford combination formula. A zero-count packet on
        either side returns the other packet unchanged (identity element),
        which also avoids a division by zero in the general formula below.
        """

        if self.count == 0:
            return other
        if other.count == 0:
            return self
        n = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / n
        m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / n
        return SufficientStatistics(count=n, mean=mean, m2=m2)


SufficientStatistics.EMPTY = SufficientStatistics(count=0, mean=0.0, m2=0.0)


# --- global finite-energy reduction -----------------------------------------


@dataclass(frozen=True)
class GlobalFiniteStatistics:
    """Global reduction of one Hamiltonian's per-shard local-energy tensors.

    Parameters
    ----------
    n_total : int
        Sum of raw (finite + nonfinite) sample counts across every shard.
    n_finite : int
        Sum of finite sample counts across every shard.
    mean : float
        Global mean over every finite sample (never a per-shard mean).
    variance : float
        Global population variance over every finite sample.
    """

    n_total: int
    n_finite: int
    mean: float
    variance: float


def reduce_energy_shards(shards: Sequence[Tensor]) -> GlobalFiniteStatistics:
    """Merge one :class:`SufficientStatistics` per shard's finite entries.

    Shards are folded left to right through :meth:`SufficientStatistics.merge`,
    never concatenated first -- concatenating first would test only the
    single-shard formula and never exercise the merge algebra a distributed
    reducer actually needs.
    """

    packet = SufficientStatistics.EMPTY
    n_total = 0
    for shard in shards:
        if shard.dtype != torch.float64:
            raise TypeError("reduce_energy_shards requires float64 shards")
        n_total += int(shard.numel())
        finite = shard[torch.isfinite(shard)]
        packet = packet.merge(SufficientStatistics.from_values(finite))
    return GlobalFiniteStatistics(
        n_total=n_total,
        n_finite=packet.count,
        mean=packet.mean,
        variance=packet.variance,
    )


# --- global VMC objective ---------------------------------------------------


@dataclass(frozen=True)
class OracleObjectiveResult:
    """Independent global VMC objective, mirroring ``VMCObjectiveResult``.

    Parameters
    ----------
    loss : torch.Tensor
        Scalar float64 loss. Differentiable through ``logabs_shards`` when
        those tensors require grad.
    metrics : dict
        JSON-safe scalar metrics, keyed to match
        ``tpen.training.vmc.compute_vmc_objective`` where the concepts agree.
    per_sample_gradients : tuple of torch.Tensor
        Analytic ``d(loss)/d(logabs)`` per shard, computed directly from the
        closed-form score-function derivative rather than by calling
        ``torch.autograd``. Used as an independent cross-check against
        autograd in :func:`analytic_gradient_matches_autograd`-style tests.
    """

    loss: Tensor
    metrics: dict[str, float | int]
    per_sample_gradients: tuple[Tensor, ...]


def oracle_vmc_objective(
    logabs_shards: Sequence[Tensor],
    energy_shards: Sequence[Tensor],
    *,
    scale_factor: float = 2.0,
) -> OracleObjectiveResult:
    """Compute the global VMC objective from per-shard logabs/energy tensors.

    Implements the ``ddp-scientific-oracle-2026-08-31`` formula directly:
    ``mu = sum(m_i E_i) / M``, ``L = scale_factor/M * sum(m_i (E_i - mu) logabs_i)``,
    where ``m_i`` is the finite-energy indicator and ``M`` is the global
    finite count. ``mu`` and every ``E_i`` are detached; gradient flows only
    through ``logabs_i``.

    Parameters
    ----------
    logabs_shards : sequence of torch.Tensor
        Per-shard log-amplitude tensors (float64). May require grad.
    energy_shards : sequence of torch.Tensor
        Per-shard local-energy tensors (float64), one per ``logabs`` shard,
        matching shapes pairwise.
    scale_factor : float, optional
        Matches ``tpen.training.vmc.compute_vmc_objective``'s default of ``2``.

    Returns
    -------
    OracleObjectiveResult

    Raises
    ------
    ValueError
        If shapes mismatch pairwise, if the global finite count is zero
        (oracle-note case M4), or if any finite-energy sample pairs with a
        nonfinite ``logabs`` entry (the poisoning policy the oracle note
        requires; see divergence D1 for today's production behavior on this
        input shape, which this function does NOT reproduce).
    """

    if len(logabs_shards) != len(energy_shards):
        raise ValueError("oracle_vmc_objective requires one energy shard per logabs shard")
    for logabs, energy in zip(logabs_shards, energy_shards, strict=True):
        if logabs.shape != energy.shape:
            raise ValueError(
                f"shard shape mismatch: logabs {tuple(logabs.shape)} vs energy {tuple(energy.shape)}"
            )
        if logabs.dtype != torch.float64 or energy.dtype != torch.float64:
            raise TypeError("oracle_vmc_objective requires float64 logabs and energy shards")
        finite_energy_mask = torch.isfinite(energy)
        if bool((finite_energy_mask & ~torch.isfinite(logabs)).any().item()):
            raise ValueError(
                "oracle_vmc_objective: a finite-energy sample paired with a "
                "nonfinite logabs entry poisons the objective; this must "
                "trigger a coordinated failure, not silent joint filtering"
            )

    stats = reduce_energy_shards([energy.detach() for energy in energy_shards])
    if stats.n_finite == 0:
        raise ValueError("oracle_vmc_objective: global finite-energy count is zero")

    mu = stats.mean
    scale_over_m = scale_factor / stats.n_finite

    loss = torch.zeros((), dtype=torch.float64)
    per_sample_gradients: list[Tensor] = []
    for logabs, energy in zip(logabs_shards, energy_shards, strict=True):
        mask = torch.isfinite(energy)
        centered = torch.where(mask, energy.detach() - mu, torch.zeros_like(energy))
        contribution = scale_over_m * (centered * torch.where(mask, logabs, torch.zeros_like(logabs)))
        loss = loss + contribution.sum()
        # Analytic derivative: d(loss)/d(logabs_i) = scale_factor/M * m_i * (E_i - mu).
        # mu and every E_i are detached, so mu carries no chain rule through logabs.
        per_sample_gradients.append(scale_over_m * mask.to(torch.float64) * centered)

    metrics: dict[str, float | int] = {
        "loss": float(loss.detach().item()),
        "energy": mu,
        "energy_variance": stats.variance,
        "energy_std": stats.variance**0.5,
        "energy_stderr": (stats.variance / stats.n_finite) ** 0.5,
        "local_energy_n_finite": stats.n_finite,
        "local_energy_n_total": stats.n_total,
        "local_energy_finite_fraction": (
            float(stats.n_finite / stats.n_total) if stats.n_total else 0.0
        ),
        "local_energy_nonfinite_count": stats.n_total - stats.n_finite,
    }
    return OracleObjectiveResult(
        loss=loss, metrics=metrics, per_sample_gradients=tuple(per_sample_gradients)
    )


# --- gradient clipping: global vs. naive per-shard --------------------------


def clip_by_global_norm(gradient: Tensor, clip_norm: float) -> Tensor:
    """Scale ``gradient`` down to ``clip_norm`` if its norm exceeds it.

    Mirrors the semantics of ``torch.nn.utils.clip_grad_norm_`` (a uniform
    rescale of the whole vector, applied once), reimplemented independently
    here so :func:`oracle_global_clip` composes it correctly at the GLOBAL
    (post-concatenation) level rather than per shard.
    """

    norm = float(gradient.norm().item())
    if norm <= clip_norm:
        return gradient
    return gradient * (clip_norm / norm)


def oracle_global_clip(shard_gradients: Sequence[Tensor], clip_norm: float) -> Tensor:
    """Clip the GLOBAL sum of per-shard gradients, never a per-shard clip.

    Oracle-note M9's decisive case: large, nearly-opposing per-shard
    gradients whose GLOBAL sum is small must clip to (at most) the global
    sum's own norm, not to a per-shard-clipped-then-summed value. Clipping
    happens once, after every shard's contribution is already summed.
    """

    total = torch.zeros_like(shard_gradients[0])
    for gradient in shard_gradients:
        total = total + gradient
    return clip_by_global_norm(total, clip_norm)


def naive_per_shard_clip_then_sum(shard_gradients: Sequence[Tensor], clip_norm: float) -> Tensor:
    """The WRONG comparison point: clip each shard, then sum the results.

    Exists only so a test can show it disagrees with
    :func:`oracle_global_clip` on the large-opposing-gradients fixture; this
    function is never the oracle's own behavior.
    """

    total = torch.zeros_like(shard_gradients[0])
    for gradient in shard_gradients:
        total = total + clip_by_global_norm(gradient, clip_norm)
    return total


# --- tolerance envelope ------------------------------------------------------


def summation_error_envelope(values: Tensor) -> float:
    """Return the first-order float64 summation rounding-error bound.

    ``eps * sum(abs(values))``: the standard backward-error bound on the
    rounding error of summing ``values`` in float64 (Higham-style analysis),
    to leading order in ``eps``.
    """

    eps = torch.finfo(torch.float64).eps
    return eps * float(values.abs().sum().item())


def loss_tolerance_envelope(terms: Tensor) -> float:
    """Return the derived (not asserted) tolerance for one loss/gradient sum.

    The score-function reduction chains three dependent float64 rounding
    steps per contributing term: (1) centering ``E_i - mu``, (2) the product
    ``(E_i - mu) * logabs_i``, (3) the reduction sum itself. Each step
    contributes an independent first-order relative perturbation bounded by
    machine epsilon; to leading order in ``eps`` (dropping ``O(eps**2)``
    cross terms, standard practice since float64 ``eps ~ 1e-16``), three
    independent rounding steps compose ADDITIVELY into a ``3 * eps *
    sum(abs(terms))`` bound. This is a derivation from the named operations,
    not an empirically chosen margin.
    """

    return 3.0 * summation_error_envelope(terms)


__all__ = [
    "GlobalFiniteStatistics",
    "OracleObjectiveResult",
    "SufficientStatistics",
    "clip_by_global_norm",
    "loss_tolerance_envelope",
    "naive_per_shard_clip_then_sum",
    "oracle_global_clip",
    "oracle_vmc_objective",
    "reduce_energy_shards",
    "summation_error_envelope",
]
