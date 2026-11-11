"""Tests for the DataIntegrity runtime-check callback."""

from __future__ import annotations

import pytest
import torch

from tpen.callback import DataIntegrity
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.training.state import TrainerState
from tests.unit.callback.support import (
    RecordingContext,
    deliver_completed_iteration,
    training_state,
)

STEP = 1


def _finite_state() -> TrainerState:
    return training_state(
        batch=ElectronBatch(
            positions=torch.zeros(4, 2, 3, dtype=torch.float64),
            spins=torch.tensor([[1.0, -1.0]] * 4, dtype=torch.float64),
        ),
        local_energy=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        loss=torch.tensor(1.5, dtype=torch.float64),
        wavefunction_output=WavefunctionOutput(
            logabs=torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float64),
            sign=torch.ones(4, dtype=torch.float64),
        ),
    )


def _handle(callback: DataIntegrity, state: TrainerState) -> RecordingContext:
    context = RecordingContext()
    deliver_completed_iteration(callback, context, state, step=STEP)
    return context


def test_passes_on_finite_state() -> None:
    context = _handle(DataIntegrity(fail_fast=True), _finite_state())

    metrics = context.latest("checks/data_integrity")
    assert metrics["passed"] is True
    assert metrics["local_energy_nonfinite_fraction"] == 0.0
    assert metrics["local_energy_finite_count"] == 4
    assert metrics["local_energy_total_count"] == 4
    assert metrics["logabs_nonfinite_fraction"] == 0.0
    assert metrics["logabs_finite_count"] == 4
    assert metrics["logabs_total_count"] == 4
    assert metrics["loss_is_finite"] is True
    # Typed schema validation is delegated to WavefunctionOutput.validate(...).
    assert metrics["output_validated"] is True


def test_typed_output_validation_failure_is_reported() -> None:
    state = _finite_state()
    # Output has 4 samples but the batch has 3 configurations: the typed
    # validator should reject the sample-shape/batch-size mismatch.
    state.batch = ElectronBatch(
        positions=torch.zeros(3, 2, 3, dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0]] * 3, dtype=torch.float64),
    )

    context = _handle(DataIntegrity(fail_fast=False), state)

    metrics = context.latest("checks/data_integrity")
    assert metrics["output_validated"] is False
    assert metrics["passed"] is False


def test_counts_disambiguate_empty_from_all_nonfinite() -> None:
    all_nan = _finite_state()
    all_nan.local_energy = torch.full((4,), float("nan"), dtype=torch.float64)
    empty = _finite_state()
    empty.local_energy = torch.empty(0, dtype=torch.float64)

    nan_metrics = _handle(DataIntegrity(fail_fast=False), all_nan).latest("checks/data_integrity")
    empty_metrics = _handle(DataIntegrity(fail_fast=False), empty).latest("checks/data_integrity")

    # Both share fraction 1.0 but the counts tell them apart.
    assert nan_metrics["local_energy_nonfinite_fraction"] == 1.0
    assert empty_metrics["local_energy_nonfinite_fraction"] == 1.0
    assert (nan_metrics["local_energy_finite_count"], nan_metrics["local_energy_total_count"]) == (0, 4)
    assert (empty_metrics["local_energy_finite_count"], empty_metrics["local_energy_total_count"]) == (0, 0)


def test_valid_batch_logs_typed_validity_metrics() -> None:
    from tpen.data.batch import ElectronBatch

    state = _finite_state()
    state.batch = ElectronBatch(
        positions=torch.zeros(4, 2, 3, dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0]] * 4, dtype=torch.float64),
    )

    metrics = _handle(DataIntegrity(fail_fast=False), state).latest("checks/data_integrity")

    assert metrics["batch_validated"] is True
    assert metrics["batch_positions_nonfinite_fraction"] == 0.0
    assert metrics["batch_n_electrons"] == 2


def test_fails_on_nan_local_energy() -> None:
    state = _finite_state()
    state.local_energy = torch.tensor([1.0, float("nan"), 3.0, 4.0], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="local_energy_nonfinite_fraction"):
        _handle(DataIntegrity(fail_fast=True), state)


def test_fails_on_inf_logabs() -> None:
    state = _finite_state()
    state.wavefunction_output = WavefunctionOutput(
        logabs=torch.tensor([0.0, float("inf"), 0.0, 0.0], dtype=torch.float64),
        sign=torch.ones(4, dtype=torch.float64),
    )

    with pytest.raises(RuntimeError, match="logabs_nonfinite_fraction"):
        _handle(DataIntegrity(fail_fast=True), state)


def test_fails_on_nonfinite_loss() -> None:
    state = _finite_state()
    state.loss = torch.tensor(float("nan"), dtype=torch.float64)

    with pytest.raises(RuntimeError, match="loss is not finite"):
        _handle(DataIntegrity(fail_fast=True), state)


def test_strict_sign_values_catches_invalid_sign() -> None:
    state = _finite_state()
    state.wavefunction_output = WavefunctionOutput(
        logabs=torch.zeros(4, dtype=torch.float64),
        sign=torch.tensor([1.0, 0.5, -1.0, 1.0], dtype=torch.float64),
    )

    context = _handle(DataIntegrity(fail_fast=False, strict_sign_values=True), state)

    metrics = context.latest("checks/data_integrity")
    assert metrics["sign_invalid_fraction"] == pytest.approx(0.25)
    assert metrics["passed"] is False


def test_logs_nonfinite_fractions_without_raising_when_not_fail_fast() -> None:
    state = _finite_state()
    state.local_energy = torch.tensor([1.0, float("inf"), float("nan"), 4.0], dtype=torch.float64)

    context = _handle(DataIntegrity(fail_fast=False), state)

    metrics = context.latest("checks/data_integrity")
    assert metrics["local_energy_nonfinite_fraction"] == pytest.approx(0.5)
    assert metrics["passed"] is False


def test_batch_without_validate_method_fails() -> None:
    state = _finite_state()
    # A plain dict is not a typed batch: it exposes no validate() contract.
    state.batch = {"positions": torch.tensor([[float("nan")]], dtype=torch.float64)}

    context = _handle(DataIntegrity(fail_fast=False), state)

    metrics = context.latest("checks/data_integrity")
    assert metrics["batch_validated"] is False
    assert metrics["passed"] is False


def test_logged_step_comes_from_the_event_not_the_state() -> None:
    # See the matching note in test_gradient_stats.py: `TrainerState` value
    # fields are stale at any boundary above their end-of-body assignment, so
    # the coordinate has to ride the typed event.
    state = _finite_state()
    state.step = -1
    context = RecordingContext()

    deliver_completed_iteration(DataIntegrity(), context, state, step=6)

    assert [r["step"] for r in context.by_namespace("checks/data_integrity")] == [6]


def test_cadence_gates_on_the_durable_step_not_the_occurrence_count() -> None:
    # A run-local `Occurrence.count` restarts after a checkpoint restore; the
    # durable step does not. Holding the count fixed proves which one gates.
    callback = DataIntegrity(every_n_steps=5)
    context = RecordingContext()
    state = _finite_state()

    for step in range(12):
        deliver_completed_iteration(callback, context, state, step=step, count=1)

    assert [r["step"] for r in context.by_namespace("checks/data_integrity")] == [0, 5, 10]


def test_every_n_steps_only_abort_decision_is_stable_across_restart() -> None:
    """A durable-only cadence sees the same bad step before and after restore."""

    bad_state = _finite_state()
    bad_state.local_energy = torch.tensor([1.0, float("nan"), 3.0, 4.0], dtype=torch.float64)

    uninterrupted = DataIntegrity(every_n_steps=1, fail_fast=True)
    _handle(uninterrupted, _finite_state())
    with pytest.raises(RuntimeError, match="local_energy_nonfinite_fraction"):
        _handle(uninterrupted, bad_state)

    resumed = DataIntegrity(every_n_steps=1, fail_fast=True)
    with pytest.raises(RuntimeError, match="local_energy_nonfinite_fraction"):
        _handle(resumed, bad_state)


def test_unrestored_max_calls_would_change_the_abort_decision() -> None:
    """The refused cadence demonstrates why the admission rule is required."""

    bad_state = _finite_state()
    bad_state.local_energy = torch.tensor([1.0, float("nan"), 3.0, 4.0], dtype=torch.float64)

    uninterrupted = DataIntegrity(every_n_steps=1, max_calls=1, fail_fast=True)
    _handle(uninterrupted, _finite_state())
    _handle(uninterrupted, bad_state)

    resumed = DataIntegrity(every_n_steps=1, max_calls=1, fail_fast=True)
    with pytest.raises(RuntimeError, match="local_energy_nonfinite_fraction"):
        _handle(resumed, bad_state)
