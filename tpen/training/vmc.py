"""Canonical VMC training objective and VMC-native training metrics.

This module is the single source of truth for the VMC score-function objective
used by `VMCTrainer`. It returns one differentiable scalar ``loss`` for
``optimizer.step()`` alongside detached, JSON-safe training metrics. Per-term
local-energy summaries are metrics (not loss components): they may be computed
from the same local-energy batch, but they never form a second public objective
surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tpen.dependencies import require_torch

torch = require_torch(feature="VMC objective")


@dataclass(frozen=True)
class VMCObjectiveResult:
    """Differentiable VMC objective plus detached JSON-safe training metrics.

    Parameters
    ----------
    loss : torch.Tensor
        Differentiable scalar surrogate objective for ``optimizer.step()``. The
        only object in this result that carries an autograd graph.
    metrics : dict
        Detached, JSON-safe training metrics (Python scalars only).
    """

    loss: torch.Tensor
    metrics: dict[str, float | int]


#: The two admissible ways to treat a non-finite local-energy row.
#:
#: ``"fail"`` refuses the step. ``"mask"`` excludes those rows and reports the
#: count, which is the historical behaviour.
#:
#: WHY THIS IS A CHOICE AND NOT A DEFAULT. Masking is NOT a random subsample.
#: Non-finite local energies do not occur uniformly -- they occur where the
#: local energy is pathological: near nodes, at coalescence, in the tail,
#: wherever the wavefunction misbehaves. Those are precisely the regions
#: carrying the physics being measured. Dropping them is therefore a
#: SYSTEMATICALLY SELECTED subsample, and the resulting energy estimator is
#: biased in a direction nobody has characterised.
#:
#: ``local_energy_nonfinite_count`` tells you HOW MANY rows were dropped. It
#: does not tell you WHAT BIAS dropping them introduced, and there is no way to
#: recover that from the count. So "not silently accepted, it is counted" is
#: true and insufficient: **a biased estimator with a diagnostic attached is
#: still a biased estimator**, and it produces a plausible number rather than a
#: crash, which is the expensive kind of error.
#:
#: A scientist choosing ``"mask"`` should be choosing a known-biased estimator
#: on purpose. That is why it is reachable only by explicit declaration, and
#: why the helium-importance closed schema requires the declaration to be
#: present rather than inherited.
#:
#: THIS POLICY DOES NOT COVER EVERY ROW-DROPPING PATH. It governs rows whose
#: LOCAL ENERGY is non-finite. `tpen.training.sr.StochasticReconfigurationUpdate`
#: has a SECOND, unconditional drop -- rows whose parameter SCORES are
#: non-finite -- in the same expression that consults this policy. That drop
#: carries the same selection-bias character and is NOT governed here.
#:
#: It is not simply an oversight, which is why it is an open decision rather
#: than a gap: a non-finite score row would poison the ENTIRE QGT through the
#: outer product, so the cost of NOT dropping it is categorically different
#: from the local-energy case -- one bad local energy biases a mean, one bad
#: score can destroy the solve. "Refuse the step" versus "protect the solve" is
#: therefore a real question with a different answer available.
#:
#: Tracked as item ``02859027-7dbf-492d-8538-2ef7e28d1cee``. Named here because
#: a reader of this policy would otherwise reasonably conclude it covers
#: "non-finite rows" in general, and it does not.
NONFINITE_LOCAL_ENERGY_POLICIES = ("fail", "mask")

#: Historical default, kept so existing non-HI callers are unchanged by the
#: introduction of the policy. The helium-importance schema does not rely on
#: this default: it REQUIRES the policy to be declared, so an HI configuration
#: cannot reach masking by omission.
DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY = "mask"


def resolve_nonfinite_local_energy_policy(policy: object) -> str:
    """Validate and normalize a non-finite local-energy policy name.

    Parameters
    ----------
    policy : object
        Candidate policy name.

    Returns
    -------
    str
        One of `NONFINITE_LOCAL_ENERGY_POLICIES`.

    Raises
    ------
    ValueError
        If `policy` is not an admissible name. Refused rather than defaulted,
        because a typo silently falling back to ``"mask"`` would reintroduce
        exactly the unchosen-estimator failure the policy exists to prevent.
    """

    if not isinstance(policy, str) or policy not in NONFINITE_LOCAL_ENERGY_POLICIES:
        raise ValueError(
            f"nonfinite local-energy policy must be one of "
            f"{list(NONFINITE_LOCAL_ENERGY_POLICIES)}, got {policy!r}"
        )
    return policy


def compute_vmc_objective(
    logabs: torch.Tensor,
    local_energy: torch.Tensor,
    *,
    scale_factor: float = 2.0,
    nonfinite_policy: str = DEFAULT_NONFINITE_LOCAL_ENERGY_POLICY,
) -> VMCObjectiveResult:
    """Compute the VMC score-function objective and training metrics.

    The returned loss is differentiable with respect to ``logabs``. Local-energy
    values are detached before forming the score-function objective, so the
    gradient flows only through ``logabs``.

    Non-finite local-energy samples are excluded from the objective and from the
    energy summary metrics. The function raises if no finite samples remain.

    Parameters
    ----------
    logabs : torch.Tensor
        Log absolute wavefunction values with shape ``[batch]``. Carries the
        autograd graph used for backpropagation.
    local_energy : torch.Tensor
        Per-sample total local energy with shape ``[batch]``.
    scale_factor : float, optional
        Multiplicative factor on the score-function objective. The default ``2``
        corresponds to gradients of an expectation under ``|psi|^2``.

    Returns
    -------
    VMCObjectiveResult
        Differentiable ``loss`` and detached, JSON-safe ``metrics``.

    Raises
    ------
    ValueError
        If ``logabs`` and ``local_energy`` shapes differ, or if no finite
        local-energy sample remains.

    Notes
    -----
    ``energy_stderr`` is an **IID-only** standard error: ``sigma / sqrt(N)``
    over finite samples. The batch is a set of correlated MCMC walkers, so this
    understates the true uncertainty by roughly ``sqrt(tau_int)``. It is a
    training progress signal, not a reportable error bar, and is retained
    unchanged because it has existing consumers.

    The correlation-aware quantity is the MCSE from
    :func:`tpen.statistics.produce_trajectory_statistics`, which requires a
    ``[draw, walker]`` trajectory that does not exist at this call site. Do not
    reinterpret this metric as an MCSE.
    """

    if logabs.shape != local_energy.shape:
        raise ValueError(
            "logabs and local_energy must have the same shape, "
            f"got {tuple(logabs.shape)} and {tuple(local_energy.shape)}"
        )

    resolved_policy = resolve_nonfinite_local_energy_policy(nonfinite_policy)

    finite_mask = torch.isfinite(local_energy)
    n_total = int(local_energy.numel())
    n_finite = int(finite_mask.sum().item())

    if resolved_policy == "fail" and n_finite != n_total:
        raise ValueError(
            "cannot compute VMC objective: "
            f"{n_total - n_finite} of {n_total} local-energy samples are non-finite and "
            "the active policy is 'fail'. Masking them would drop a systematically "
            "selected subsample -- non-finite rows occur where the local energy is "
            "pathological -- so the resulting estimator would be biased by an "
            "uncharacterised amount. Declare 'mask' explicitly to accept that "
            "known-biased estimator"
        )

    # Retained under EVERY policy: with no finite row there is no estimator at
    # all, biased or otherwise, so this is not a policy question.
    if n_finite == 0:
        raise ValueError("cannot compute VMC objective: no finite local-energy samples")

    finite_logabs = logabs[finite_mask]
    finite_energy = local_energy[finite_mask].detach()

    energy = finite_energy.mean()
    centered_energy = finite_energy - energy

    loss = scale_factor * torch.mean(centered_energy * finite_logabs)

    if n_finite > 1:
        energy_variance = finite_energy.var(unbiased=False)
    else:
        energy_variance = torch.zeros((), device=finite_energy.device, dtype=finite_energy.dtype)

    energy_std = torch.sqrt(energy_variance)
    energy_stderr = energy_std / float(n_finite) ** 0.5

    metrics: dict[str, float | int] = {
        "loss": float(loss.detach().item()),
        "energy": float(energy.detach().item()),
        "energy_variance": float(energy_variance.detach().item()),
        "energy_std": float(energy_std.detach().item()),
        "energy_stderr": float(energy_stderr.detach().item()),
        "local_energy_n_finite": n_finite,
        "local_energy_n_total": n_total,
        "local_energy_finite_fraction": float(n_finite / n_total) if n_total else 0.0,
        "local_energy_nonfinite_count": n_total - n_finite,
    }

    return VMCObjectiveResult(loss=loss, metrics=metrics)


def hamiltonian_term_metric_prefix(name: str) -> str:
    """Return the metric-key prefix for a named Hamiltonian term.

    The prefix is derived from the resolved term name (the ``dict`` key, or the
    snake-case class name for a sequence; see
    `tpen.physics.hamiltonian.normalize_hamiltonian_terms`). Names are unique,
    so prefixes are deterministic and collision-free. Training per-term metrics
    use this prefix directly for the finite mean and append suffixes such as
    ``_variance``, ``_std``, ``_stderr``, ``_n_finite``, ``_n_total``,
    ``_finite_fraction``, and ``_nonfinite_count`` for companion statistics.
    """

    return f"energy_term_{name}"


def summarize_local_energy_terms(
    terms: Mapping[str, torch.Tensor],
) -> dict[str, float | int]:
    """Summarize per-Hamiltonian-term local-energy tensors as training metrics.

    Term metric keys are derived from the resolved term names (see
    `hamiltonian_term_metric_prefix`). For a resolved name ``kinetic``, the
    finite mean is logged as ``energy_term_kinetic`` and companion statistics
    use suffixes like ``energy_term_kinetic_variance``. These are metrics only
    -- they never form part of the optimizer objective.

    Parameters
    ----------
    terms : Mapping of str to torch.Tensor
        Per-term local-energy tensors keyed by resolved term name, as produced
        by ``local_energy(..., return_terms=True).terms``.

    Returns
    -------
    dict
        Detached, JSON-safe per-term metrics (Python scalars only).

    Raises
    ------
    ValueError
        If any term has no finite samples.

    Notes
    -----
    Each ``{prefix}_stderr`` is an **IID-only** standard error, exactly as in
    `compute_vmc_objective`: it ignores serial correlation between MCMC walkers and so
    understates the true uncertainty. Per-term error bars are diagnostic only.
    Do not reinterpret them as MCSE; there is no per-term trajectory producer.
    """

    metrics: dict[str, float | int] = {}

    for name, values in terms.items():
        prefix = hamiltonian_term_metric_prefix(name)

        finite_mask = torch.isfinite(values)
        n_total = int(values.numel())
        n_finite = int(finite_mask.sum().item())

        if n_finite == 0:
            raise ValueError(f"cannot summarize local-energy term {prefix}: no finite samples")

        finite_values = values[finite_mask].detach()
        energy = finite_values.mean()

        if n_finite > 1:
            variance = finite_values.var(unbiased=False)
        else:
            variance = torch.zeros((), device=finite_values.device, dtype=finite_values.dtype)

        std = torch.sqrt(variance)
        stderr = std / float(n_finite) ** 0.5

        metrics[prefix] = float(energy.detach().item())
        metrics[f"{prefix}_variance"] = float(variance.detach().item())
        metrics[f"{prefix}_std"] = float(std.detach().item())
        metrics[f"{prefix}_stderr"] = float(stderr.detach().item())
        metrics[f"{prefix}_n_finite"] = n_finite
        metrics[f"{prefix}_n_total"] = n_total
        metrics[f"{prefix}_finite_fraction"] = float(n_finite / n_total) if n_total else 0.0
        metrics[f"{prefix}_nonfinite_count"] = n_total - n_finite

    return metrics


def summarize_logabs(logabs: torch.Tensor) -> dict[str, float]:
    """Summarize log-amplitude values into finite-aware scalar metrics.

    Parameters
    ----------
    logabs : torch.Tensor
        Log absolute wavefunction values with shape ``[batch]``.

    Returns
    -------
    dict
        Scalar metrics ``logabs_mean``, ``logabs_min``, ``logabs_max``, and
        ``nonfinite_logabs_fraction``. Statistics are computed over finite
        entries and are ``nan`` when no entry is finite.
    """

    n = int(logabs.numel())
    finite_mask = torch.isfinite(logabs)
    n_finite = int(finite_mask.sum().item())
    if n_finite > 0:
        finite = logabs[finite_mask]
        mean = float(finite.mean().item())
        minimum = float(finite.min().item())
        maximum = float(finite.max().item())
    else:
        mean = float("nan")
        minimum = float("nan")
        maximum = float("nan")
    return {
        "logabs_mean": mean,
        "logabs_min": minimum,
        "logabs_max": maximum,
        "nonfinite_logabs_fraction": float((n - n_finite) / n) if n > 0 else float("nan"),
    }


__all__ = [
    "VMCObjectiveResult",
    "compute_vmc_objective",
    "hamiltonian_term_metric_prefix",
    "summarize_local_energy_terms",
    "summarize_logabs",
]
