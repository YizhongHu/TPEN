"""PR630 T2 probes; proposed for Cannon execution, not run by the reviewer.

The rank and large-finite-offset probes expose defects at bbbbe00a. The other
probes are expected passing controls with their targeted mutants named below.
All expectations are independent literals or public event/logger behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.callback import AmplitudeRiskFraction
from tpen.callback import amplitude_risk
from tpen.callback.amplitude_risk import amplitude_risk_metrics
from tpen.data.batch import WavefunctionOutput
from tpen.training.events import TrainingIteration, TrainingIterationCompleted
from tests.helpers.run_context import RecordingLogger, make_run_context
from tests.unit.callback.support import (
    RecordingContext,
    deliver_completed_iteration,
    training_state,
)


@pytest.mark.parametrize("values", [0.0, [[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]]])
def test_review_refuses_non_vector_inputs_instead_of_pooling_draws(values) -> None:
    """Catch accepting scalar inputs or pooling draw-by-walker matrices.

    Each row of the matrix has zero low-amplitude occupancy. Flattening it
    silently produces a pooled median of 50 and a spurious fraction of 1/2.
    """

    logabs = torch.tensor(values, dtype=torch.float64)
    with pytest.raises(ValueError):
        amplitude_risk_metrics(logabs, torch.ones_like(logabs), (-3.0,))


@pytest.mark.parametrize(
    ("dtype", "offset"), [(torch.float32, 2.0**127), (torch.float64, 2.0**1023)]
)
def test_review_constant_finite_gauge_shift_cannot_overflow_median(dtype, offset) -> None:
    """Catch an overflowing even midpoint, without threshold-roundoff ambiguity."""

    zeros = torch.zeros(2, dtype=dtype)
    shifted = zeros + offset
    assert torch.isfinite(shifted).all()
    base = amplitude_risk_metrics(zeros, torch.ones_like(zeros), (-3.0,))
    result = amplitude_risk_metrics(shifted, torch.ones_like(shifted), (-3.0,))
    # All values are identical, so every centered value must be exactly zero.
    assert result["finite_logabs_count"] == 2
    assert result["available"] is True
    assert result["low_amplitude_count.-3.0"] == 0
    assert result["low_amplitude_fraction.-3.0"] == 0.0
    assert result == base


def test_review_mixed_invalid_values_match_hand_counted_full_population() -> None:
    """Catch mean centering, denominator shrinkage, coercion and sign-inf omission."""

    logabs = torch.tensor(
        [-10.0, -3.0, 0.0, 0.0, 100.0, float("-inf"), float("inf"), float("nan")],
        dtype=torch.float64,
    )
    sign = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, 0.5, -1.0, 1.0, 1.0],
        dtype=torch.float64,
    )
    # Finite-only median = 0. Exactly -10 and -inf count; -3 is an equality.
    result = amplitude_risk_metrics(logabs, sign, (-3.0, -5.0))
    assert result == {
        "total_count": 8,
        "finite_logabs_count": 5,
        "nan_logabs_count": 1,
        "positive_infinite_logabs_count": 1,
        "negative_infinite_logabs_count": 1,
        "sign_nonfinite_count": 3,
        "sign_zero_count": 1,
        "sign_other_invalid_count": 1,
        "available": True,
        "low_amplitude_count.-3.0": 2,
        "low_amplitude_fraction.-3.0": 0.25,
        "low_amplitude_count.-5.0": 2,
        "low_amplitude_fraction.-5.0": 0.25,
    }


def test_review_minus_infinity_without_a_finite_center_is_unavailable() -> None:
    """Catch declaring -inf-only populations available or assigning fraction one."""

    logabs = torch.tensor([float("-inf"), float("-inf")], dtype=torch.float64)
    result = amplitude_risk_metrics(logabs, torch.zeros_like(logabs), (-3.0, -5.0))
    assert result["total_count"] == 2
    assert result["negative_infinite_logabs_count"] == 2
    assert result["sign_zero_count"] == 2
    assert result["available"] is False
    for tau in (-3.0, -5.0):
        assert result[f"low_amplitude_count.{tau!r}"] is None
        assert result[f"low_amplitude_fraction.{tau!r}"] is None


def test_review_hydra_refuses_an_omitted_threshold_field() -> None:
    """Catch a constructor default that silently enables an unconfigured scale."""

    config = OmegaConf.create({"_target_": "tpen.callback.AmplitudeRiskFraction"})
    with pytest.raises(InstantiationException, match="thresholds"):
        instantiate(config)


def test_review_dispatch_uses_each_supplied_batch_including_skipped_updates(
    tmp_path: Path,
) -> None:
    """Catch wrong subscription, post-update forward, stale output or pooled batches.

    This exercises the real dispatcher. Trainer-side assignment timing is a
    separate static audit; the supplied state represents its pre-update output.
    """

    def forbidden_forward(*args, **kwargs):
        raise AssertionError("amplitude callback must not call the updated model")

    callback = AmplitudeRiskFraction(thresholds=(-3.0,))
    logger = RecordingLogger()
    context = make_run_context(tmp_path, callbacks=[callback], loggers=[logger])
    state = training_state(model=forbidden_forward)
    for step, values, applied in (
        (8, [0.0, 0.0, -10.0], False),
        (9, [100.0, 100.0, 100.0], True),
    ):
        state.wavefunction_output = WavefunctionOutput(
            logabs=torch.tensor(values, dtype=torch.float64),
            sign=torch.ones(3, dtype=torch.float64),
        )
        state.optimizer_step = applied
        context.emit(
            TrainingIterationCompleted(iteration=TrainingIteration(step=step)), state=state
        )

    records = logger.by_namespace("train/amplitude_risk")
    assert [record.step for record in records] == [8, 9]
    assert [record.metrics["total_count"] for record in records] == [3, 3]
    assert [record.metrics["low_amplitude_count.-3.0"] for record in records] == [1, 0]
    assert [record.metrics["low_amplitude_fraction.-3.0"] for record in records] == [1 / 3, 0.0]


def test_review_ineligible_cadence_performs_no_reduction(monkeypatch) -> None:
    """Catch moving expensive reduction ahead of the cadence filter."""

    def forbidden_reduction(*args, **kwargs):
        raise AssertionError("off-cadence iteration must not reduce amplitudes")

    monkeypatch.setattr(amplitude_risk, "amplitude_risk_metrics", forbidden_reduction)
    state = training_state(
        wavefunction_output=WavefunctionOutput(
            logabs=torch.zeros(3, dtype=torch.float64), sign=torch.ones(3, dtype=torch.float64)
        )
    )
    context = RecordingContext()
    deliver_completed_iteration(
        AmplitudeRiskFraction(thresholds=(-3.0,), every_n_steps=2), context, state, step=1
    )
    assert context.records == []


def test_review_reconstructed_callback_preserves_durable_cadence_records() -> None:
    """Catch callback-local offsets or retained batch history changing resumed output."""

    def collect(steps):
        callback = AmplitudeRiskFraction(thresholds=(-3.0,), every_n_steps=2, start_step=1)
        context = RecordingContext()
        for count, step in enumerate(steps, start=1):
            state = training_state(
                wavefunction_output=WavefunctionOutput(
                    logabs=torch.tensor([float(step), 0.0, -10.0], dtype=torch.float64),
                    sign=torch.ones(3, dtype=torch.float64),
                )
            )
            deliver_completed_iteration(callback, context, state, step=step, count=count)
        return context.records

    continuous = collect(range(8))
    reconstructed = collect(range(4)) + collect(range(4, 8))
    assert [record["step"] for record in continuous] == [1, 3, 5, 7]
    assert reconstructed == continuous
