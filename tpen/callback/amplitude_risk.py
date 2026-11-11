"""Descriptive low-amplitude occupancy fraction, gauge-invariant by design."""

from __future__ import annotations

from typing import Any, ClassVar

from tpen.artifacts import RunContext
from tpen.dependencies import require_torch
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence


class AmplitudeRiskFraction(StatefulCallback[TrainerState]):
    """Log the fraction of walkers with centered log-amplitude below each tau.

    For one training batch, with ``ell`` the just-completed iteration's
    ``logabs`` and ``m`` the median of its FINITE entries (arithmetic midpoint
    of the two central sorted values for an even count), this logs, for every
    configured threshold ``tau`` (a negative offset), the fraction of the full
    batch for which ``ell - m < tau`` strictly, OR ``ell == -inf``. It also
    logs the finite/nonfinite logabs breakdown and the sign-category counts.

    Parameters
    ----------
    thresholds : sequence of float
        Nonempty, finite, distinct negative log-amplitude offsets. Required;
        there is no default, so a configuration that omits it is refused at
        construction rather than silently measuring nothing.
    every_n_steps, start_step, max_calls, probability, seed
        Durable-step cadence, consumed by `pop_step_cadence`.

    Notes
    -----
    **This is descriptive, not a classifier.** Adding any constant to the
    wavefunction (hence to every entry of ``ell``, and to ``m``) leaves every
    ``ell - m`` and therefore every reported fraction unchanged: the statistic
    is a gauge-invariant property of the batch's *relative* amplitude spread,
    not of its absolute scale. It is not an energy, normalizability, or
    sampler-correctness certificate, and a zero fraction is not a health
    verdict -- see ``amplitude-risk-call-20261106`` for the frozen study
    thresholds and their descriptive labels. No fraction computed here gates
    anything; `tpen.callback.health.DataIntegrity` owns run-halting policy.

    This callback reads `tpen.training.state.TrainerState.wavefunction_output`
    from the JUST-COMPLETED iteration -- the pre-update batch that produced it,
    per `tpen.training.trainer.VMCTrainer` -- and never triggers a fresh
    forward pass. It fires on every completed iteration admitted by its
    cadence, including one whose optimizer update was skipped, because the
    batch and its wavefunction output are still the ones this iteration
    actually evaluated.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, *, thresholds: Any, **kwargs: Any) -> None:
        resolved = _validate_thresholds(thresholds)
        cadence = pop_step_cadence(kwargs)
        super().__init__(
            typed_groups=(
                SubscriptionGroup(selectors=(Subscription.of(TrainingIterationCompleted),)),
            ),
            **kwargs,
        )
        self.thresholds = resolved
        self._steps = StepCadenceGate(cadence)

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Compute and log the centered low-amplitude occupancy fraction."""

        event = occurrence.event
        if not isinstance(event, TrainingIterationCompleted):
            return
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return

        output = state.wavefunction_output
        if output is None:
            return
        metrics = amplitude_risk_metrics(output.logabs, output.sign, self.thresholds)
        context.log(metrics, step=step, namespace="train/amplitude_risk")


def _validate_thresholds(thresholds: Any) -> tuple[float, ...]:
    """Validate and freeze the configured threshold list.

    A nonempty, finite, distinct list of negative log-amplitude offsets. There
    is no core default: a configuration that omits ``thresholds`` is refused
    here rather than silently measuring an arbitrary built-in scale.
    """

    if thresholds is None:
        raise ValueError("AmplitudeRiskFraction requires 'thresholds'; there is no default")
    resolved = tuple(float(value) for value in thresholds)
    if not resolved:
        raise ValueError("AmplitudeRiskFraction 'thresholds' must be nonempty")
    for value in resolved:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"AmplitudeRiskFraction thresholds must be finite, got {value!r}")
        if value >= 0.0:
            raise ValueError(
                f"AmplitudeRiskFraction thresholds must be negative offsets, got {value!r}"
            )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"AmplitudeRiskFraction thresholds must be distinct, got {resolved!r}")
    return resolved


def amplitude_risk_metrics(
    logabs: Any, sign: Any, thresholds: tuple[float, ...]
) -> dict[str, Any]:
    """Return the gauge-invariant centered low-amplitude occupancy metrics.

    Parameters
    ----------
    logabs : Tensor
        One-dimensional log-amplitude values for the full sampled batch.
    sign : Tensor
        One-dimensional sign values aligned one-to-one with ``logabs``.
    thresholds : tuple of float
        Frozen, validated negative offsets (see `_validate_thresholds`).

    Returns
    -------
    dict
        ``total_count``; ``finite_logabs_count``; ``nan_logabs_count``;
        ``positive_infinite_logabs_count``; ``negative_infinite_logabs_count``;
        ``sign_nonfinite_count``; ``sign_zero_count``;
        ``sign_other_invalid_count``; ``available`` (bool); and, per
        threshold, ``low_amplitude_count.<tau>`` and
        ``low_amplitude_fraction.<tau>`` -- ``None`` for both when
        ``available`` is ``False`` (``total_count == 0`` or no finite logabs),
        which is distinct from a healthy, measured zero.

    Notes
    -----
    ``total_count`` is the FULL sampled batch size, independent of any
    local-energy finite mask: `N` in the denominator never shrinks because
    some other quantity was invalid. The comparison against each ``tau`` is
    strict (``<``), and a recorded ``-inf`` logabs counts as low-amplitude
    exactly once per threshold, never twice over.
    """

    torch = require_torch(feature="AmplitudeRiskFraction")
    logabs = logabs.detach()
    sign = sign.detach()
    if logabs.ndim != 1 or sign.ndim != 1:
        raise ValueError(
            f"logabs and sign must be one-dimensional, got ndim {logabs.ndim} vs "
            f"{sign.ndim}"
        )
    if logabs.shape != sign.shape:
        raise ValueError(
            f"logabs and sign must share a shape, got {tuple(logabs.shape)} vs "
            f"{tuple(sign.shape)}"
        )

    total_count = int(logabs.numel())
    nan_mask = torch.isnan(logabs)
    pos_inf_mask = logabs == float("inf")
    neg_inf_mask = logabs == float("-inf")
    finite_mask = torch.isfinite(logabs)
    finite_count = int(finite_mask.sum().item())

    sign_nan_or_inf = ~torch.isfinite(sign)
    sign_zero_mask = (sign == 0) & torch.isfinite(sign)
    sign_other_invalid_mask = (
        torch.isfinite(sign) & (sign != -1) & (sign != 0) & (sign != 1)
    )

    metrics: dict[str, Any] = {
        "total_count": total_count,
        "finite_logabs_count": finite_count,
        "nan_logabs_count": int(nan_mask.sum().item()),
        "positive_infinite_logabs_count": int(pos_inf_mask.sum().item()),
        "negative_infinite_logabs_count": int(neg_inf_mask.sum().item()),
        "sign_nonfinite_count": int(sign_nan_or_inf.sum().item()),
        "sign_zero_count": int(sign_zero_mask.sum().item()),
        "sign_other_invalid_count": int(sign_other_invalid_mask.sum().item()),
    }

    available = total_count > 0 and finite_count > 0
    metrics["available"] = available
    if not available:
        for tau in thresholds:
            key = _tau_key(tau)
            metrics[f"low_amplitude_count.{key}"] = None
            metrics[f"low_amplitude_fraction.{key}"] = None
        return metrics

    finite_values = logabs[finite_mask]
    median = _finite_median(finite_values, torch)
    centered = finite_values - median
    neg_inf_count = int(neg_inf_mask.sum().item())
    for tau in thresholds:
        key = _tau_key(tau)
        low_finite = int((centered < tau).sum().item())
        low_count = low_finite + neg_inf_count
        metrics[f"low_amplitude_count.{key}"] = low_count
        metrics[f"low_amplitude_fraction.{key}"] = low_count / total_count
    return metrics


def _finite_median(values: Any, torch: Any) -> Any:
    """Return the ordinary median, using the midpoint convention when even.

    ``torch.median`` returns the LOWER of the two central values for an even
    count, which is not the arithmetic midpoint this callback's contract
    requires; this always sorts and computes the midpoint explicitly so an
    even-count batch is centered the same way an odd-count one is.
    """

    sorted_values, _ = torch.sort(values)
    n = sorted_values.numel()
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    lower = sorted_values[mid - 1]
    upper = sorted_values[mid]
    return lower + (upper - lower) / 2


def _tau_key(tau: float) -> str:
    """Return a metric-name-safe spelling of one threshold value."""

    return repr(tau)


__all__ = ["AmplitudeRiskFraction", "amplitude_risk_metrics"]
