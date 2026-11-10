"""Tests for the gauge-invariant AmplitudeRiskFraction callback."""

from __future__ import annotations

import pytest
import torch

from tpen.callback import AmplitudeRiskFraction
from tpen.callback.amplitude_risk import amplitude_risk_metrics
from tpen.data.batch import WavefunctionOutput
from tests.unit.callback.support import RecordingContext, deliver_completed_iteration, training_state

STEP = 3
THRESHOLDS = (-3.0, -4.0, -5.0, -7.0, -10.0)


def _state(logabs: list[float], sign: list[float] | None = None) -> object:
    n = len(logabs)
    return training_state(
        wavefunction_output=WavefunctionOutput(
            logabs=torch.tensor(logabs, dtype=torch.float64),
            sign=torch.ones(n, dtype=torch.float64) if sign is None else torch.tensor(sign, dtype=torch.float64),
        ),
    )


def _handle(callback: AmplitudeRiskFraction, state: object) -> RecordingContext:
    context = RecordingContext()
    deliver_completed_iteration(callback, context, state, step=STEP)
    return context


def test_rejects_missing_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        AmplitudeRiskFraction(thresholds=None)


def test_rejects_empty_thresholds() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        AmplitudeRiskFraction(thresholds=())


def test_rejects_nonfinite_threshold() -> None:
    with pytest.raises(ValueError, match="finite"):
        AmplitudeRiskFraction(thresholds=(float("nan"),))


def test_rejects_positive_threshold() -> None:
    with pytest.raises(ValueError, match="negative"):
        AmplitudeRiskFraction(thresholds=(3.0,))


def test_rejects_duplicate_thresholds() -> None:
    with pytest.raises(ValueError, match="distinct"):
        AmplitudeRiskFraction(thresholds=(-3.0, -3.0))


def test_odd_count_median_is_middle_value() -> None:
    # Sorted: -5,-4,-3,-2,-1 -> median -3. Centered: -2,-1,0,1,2.
    logabs = [-1.0, -3.0, -5.0, -2.0, -4.0]
    metrics = amplitude_risk_metrics(
        torch.tensor(logabs, dtype=torch.float64),
        torch.ones(5, dtype=torch.float64),
        (-2.0,),
    )
    # centered < -2 strictly: only -5-(-3)=-2 is not < -2 (equal). None below -2.
    assert metrics["low_amplitude_count.-2.0"] == 0
    assert metrics["low_amplitude_fraction.-2.0"] == 0.0


def test_even_count_median_uses_midpoint_convention() -> None:
    # Sorted: -4,-2,0,2 -> midpoint of -2 and 0 is -1.
    logabs = [0.0, -4.0, -2.0, 2.0]
    metrics = amplitude_risk_metrics(
        torch.tensor(logabs, dtype=torch.float64),
        torch.ones(4, dtype=torch.float64),
        (-3.0,),
    )
    # centered = logabs - (-1) = 1, -3, -1, 3. Strictly below -3: none (-3 == -3, not <).
    assert metrics["low_amplitude_count.-3.0"] == 0
    # Nudge one value so exactly one centered entry clears the threshold.
    logabs2 = [0.0, -4.01, -2.0, 2.0]
    metrics2 = amplitude_risk_metrics(
        torch.tensor(logabs2, dtype=torch.float64),
        torch.ones(4, dtype=torch.float64),
        (-3.0,),
    )
    assert metrics2["low_amplitude_count.-3.0"] == 1


def test_tie_at_threshold_boundary_is_strict() -> None:
    # median 0; one value centered exactly at tau=-3 must NOT count (strict <).
    logabs = [0.0, 0.0, -3.0]
    metrics = amplitude_risk_metrics(
        torch.tensor(logabs, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
        (-3.0,),
    )
    assert metrics["low_amplitude_count.-3.0"] == 0
    logabs_below = [0.0, 0.0, -3.0001]
    metrics_below = amplitude_risk_metrics(
        torch.tensor(logabs_below, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
        (-3.0,),
    )
    assert metrics_below["low_amplitude_count.-3.0"] == 1


def test_gauge_shift_leaves_every_fraction_unchanged() -> None:
    torch.manual_seed(0)
    logabs = torch.randn(64, dtype=torch.float64) * 2.0
    sign = torch.ones(64, dtype=torch.float64)
    base = amplitude_risk_metrics(logabs, sign, THRESHOLDS)
    shifted = amplitude_risk_metrics(logabs + 137.5, sign, THRESHOLDS)
    for tau in THRESHOLDS:
        key = f"low_amplitude_fraction.{tau!r}"
        assert base[key] == shifted[key]
        assert base[f"low_amplitude_count.{tau!r}"] == shifted[f"low_amplitude_count.{tau!r}"]


def test_global_sign_flip_leaves_sign_categories_unchanged() -> None:
    logabs = torch.tensor([-1.0, -2.0, -3.0, 0.0], dtype=torch.float64)
    sign = torch.tensor([1.0, -1.0, 0.0, 1.0], dtype=torch.float64)
    base = amplitude_risk_metrics(logabs, sign, THRESHOLDS)
    flipped = amplitude_risk_metrics(logabs, -sign, THRESHOLDS)
    assert base["sign_zero_count"] == flipped["sign_zero_count"]
    assert base["sign_nonfinite_count"] == flipped["sign_nonfinite_count"]
    assert base["sign_other_invalid_count"] == flipped["sign_other_invalid_count"]
    for tau in THRESHOLDS:
        key = f"low_amplitude_fraction.{tau!r}"
        assert base[key] == flipped[key]


def test_negative_infinity_counts_as_low_amplitude_once_per_threshold() -> None:
    logabs = torch.tensor([0.0, 0.0, float("-inf")], dtype=torch.float64)
    sign = torch.ones(3, dtype=torch.float64)
    metrics = amplitude_risk_metrics(logabs, sign, (-3.0, -100.0))
    # median of finite values {0,0} is 0; -inf is low at every threshold, once.
    assert metrics["negative_infinite_logabs_count"] == 1
    assert metrics["low_amplitude_count.-3.0"] == 1
    assert metrics["low_amplitude_count.-100.0"] == 1
    assert metrics["total_count"] == 3


def test_positive_infinity_and_nan_are_not_low_amplitude_but_are_retained() -> None:
    logabs = torch.tensor([0.0, 0.0, float("inf"), float("nan")], dtype=torch.float64)
    sign = torch.ones(4, dtype=torch.float64)
    metrics = amplitude_risk_metrics(logabs, sign, (-1.0,))
    assert metrics["total_count"] == 4
    assert metrics["finite_logabs_count"] == 2
    assert metrics["positive_infinite_logabs_count"] == 1
    assert metrics["nan_logabs_count"] == 1
    assert metrics["low_amplitude_count.-1.0"] == 0


def test_exact_sign_zero_is_counted_and_not_invalid() -> None:
    logabs = torch.tensor([0.0, 0.0], dtype=torch.float64)
    sign = torch.tensor([0.0, 1.0], dtype=torch.float64)
    metrics = amplitude_risk_metrics(logabs, sign, (-1.0,))
    assert metrics["sign_zero_count"] == 1
    assert metrics["sign_other_invalid_count"] == 0
    assert metrics["sign_nonfinite_count"] == 0


def test_sign_nonfinite_and_other_invalid_values_are_disambiguated() -> None:
    logabs = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    sign = torch.tensor([float("nan"), 0.5, -1.0], dtype=torch.float64)
    metrics = amplitude_risk_metrics(logabs, sign, (-1.0,))
    assert metrics["sign_nonfinite_count"] == 1
    assert metrics["sign_other_invalid_count"] == 1


def test_one_sample_batch_is_available_with_zero_centered_value() -> None:
    logabs = torch.tensor([-7.0], dtype=torch.float64)
    sign = torch.ones(1, dtype=torch.float64)
    metrics = amplitude_risk_metrics(logabs, sign, (-3.0,))
    assert metrics["available"] is True
    assert metrics["total_count"] == 1
    assert metrics["finite_logabs_count"] == 1
    # median equals the single value, so centered value is 0, never below tau.
    assert metrics["low_amplitude_count.-3.0"] == 0
    assert metrics["low_amplitude_fraction.-3.0"] == 0.0


def test_empty_input_is_unavailable_not_a_healthy_zero() -> None:
    logabs = torch.empty(0, dtype=torch.float64)
    sign = torch.empty(0, dtype=torch.float64)
    metrics = amplitude_risk_metrics(logabs, sign, (-3.0, -5.0))
    assert metrics["available"] is False
    assert metrics["total_count"] == 0
    assert metrics["low_amplitude_count.-3.0"] is None
    assert metrics["low_amplitude_fraction.-3.0"] is None


def test_all_nonfinite_input_is_unavailable_not_a_healthy_zero() -> None:
    logabs = torch.full((5,), float("nan"), dtype=torch.float64)
    sign = torch.ones(5, dtype=torch.float64)
    metrics = amplitude_risk_metrics(logabs, sign, (-3.0,))
    assert metrics["available"] is False
    assert metrics["total_count"] == 5
    assert metrics["finite_logabs_count"] == 0
    assert metrics["low_amplitude_count.-3.0"] is None


def test_shape_mismatch_between_logabs_and_sign_is_refused() -> None:
    logabs = torch.zeros(3, dtype=torch.float64)
    sign = torch.ones(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="shape"):
        amplitude_risk_metrics(logabs, sign, (-3.0,))


def test_independent_of_local_energy_mask() -> None:
    """The denominator is the full sampled batch, never a local-energy mask."""

    state = _state([0.0, 0.0, -20.0, 0.0])
    state.local_energy = torch.tensor([1.0, float("nan"), 3.0, float("inf")], dtype=torch.float64)

    callback = AmplitudeRiskFraction(thresholds=(-3.0,))
    metrics = _handle(callback, state).latest("train/amplitude_risk")

    assert metrics["total_count"] == 4
    assert metrics["low_amplitude_count.-3.0"] == 1


def test_cadence_gates_on_durable_step_not_run_local_count() -> None:
    callback = AmplitudeRiskFraction(thresholds=(-3.0,), every_n_steps=2)
    state = _state([0.0, 0.0, 0.0])

    context = RecordingContext()
    deliver_completed_iteration(callback, context, state, step=1, count=1)
    assert context.by_namespace("train/amplitude_risk") == []
    deliver_completed_iteration(callback, context, state, step=2, count=1)
    assert len(context.by_namespace("train/amplitude_risk")) == 1


def test_handles_no_wavefunction_output_without_error() -> None:
    callback = AmplitudeRiskFraction(thresholds=(-3.0,))
    state = training_state()

    context = _handle(callback, state)

    assert context.records == []
