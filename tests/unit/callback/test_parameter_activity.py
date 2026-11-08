"""Unit tests for the `ParameterActivity` callback.

These tests drive `ParameterActivity.handle_occurrence` directly with
hand-built typed occurrences and models, the same style
`tests/unit/callback/test_initial_checkpoint.py` uses. Gradient norms are
computed independently of the callback under test (a manual sum-of-squares
square root), and models used to probe the foreign-optimizer-carrier case are
constructed independently from the optimizer under test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tpen.callback.parameter_activity import ParameterActivity
from tpen.training.events import TrainingIteration, TrainingIterationCompleted, UpdateCompleted, UpdateSkipped
from tpen.training.state import TrainerState
from tpen.events import Occurrence


class _Trainer:
    """Trainer double exposing only the `completed_updates` counter this callback reads."""

    def __init__(self, completed_updates: int) -> None:
        self.completed_updates = completed_updates


def _independent_l2_norm(tensor: torch.Tensor) -> float:
    """Compute an L2 norm without going through the callback's own code path."""

    return math.sqrt(float((tensor.detach() ** 2).sum().item()))


def _update_completed(*, count: int = 1) -> Occurrence[UpdateCompleted]:
    return Occurrence(event=UpdateCompleted(iteration=TrainingIteration(step=0)), count=count)


def _update_skipped(*, count: int = 1) -> Occurrence[UpdateSkipped]:
    return Occurrence(event=UpdateSkipped(iteration=TrainingIteration(step=0)), count=count)


def _iteration_completed(*, count: int = 1) -> Occurrence[TrainingIterationCompleted]:
    return Occurrence(event=TrainingIterationCompleted(iteration=TrainingIteration(step=0)), count=count)


class _ThreeParamModel(torch.nn.Module):
    """One parameter off the computation path, one forced to an exact zero
    gradient, and one ordinary parameter with a nonzero gradient."""

    def __init__(self) -> None:
        super().__init__()
        self.used = torch.nn.Parameter(torch.tensor([2.0, 3.0], dtype=torch.float64))
        self.zeroed = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.tensor([5.0], dtype=torch.float64))

    def loss(self) -> torch.Tensor:
        # `zeroed` enters the graph multiplied by a constant zero, so autograd
        # populates an exactly-zero gradient tensor for it rather than leaving
        # `.grad` unset. `unused` never enters the graph at all.
        return (self.used ** 2).sum() + (self.zeroed * 0.0).sum()


def _rows_by_name(payload: dict) -> dict[str, dict]:
    return {row["name"]: row for row in payload["parameters"]}


def test_classifies_absent_zero_and_nonzero_gradients(tmp_path: Path) -> None:
    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    used_grad_before = model.used.grad.detach().clone()

    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    callback.handle_occurrence(_update_completed(), object(), state)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = _rows_by_name(payload)
    assert rows["unused"]["gradient"] == "absent"
    assert rows["unused"]["gradient_norm"] is None
    assert rows["zeroed"]["gradient"] == "zero"
    assert rows["zeroed"]["gradient_norm"] == pytest.approx(0.0)
    assert rows["used"]["gradient"] == "nonzero"
    assert rows["used"]["gradient_norm"] == pytest.approx(
        _independent_l2_norm(used_grad_before)
    )
    assert payload["completed_updates"] == 1
    # Gradients must be bitwise unchanged by the callback.
    assert torch.equal(model.used.grad, used_grad_before)


def test_optimizer_without_live_model_parameters_reports_every_parameter_absent(
    tmp_path: Path,
) -> None:
    """A foreign-carrier optimizer must not be trusted even with real gradients present."""

    model = _ThreeParamModel()
    # An optimizer built over an independently constructed model's parameters,
    # never the live model's -- the exact foreign-carrier-params shape.
    foreign_model = _ThreeParamModel()
    optimizer = torch.optim.SGD(foreign_model.parameters(), lr=0.1)
    model.loss().backward()
    assert model.used.grad is not None and bool(model.used.grad.count_nonzero())

    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    callback.handle_occurrence(_update_completed(), object(), state)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = _rows_by_name(payload)
    for row in rows.values():
        assert row["gradient"] == "absent"
        assert row["gradient_norm"] is None


def test_no_write_on_update_skipped_or_unrelated_event(tmp_path: Path) -> None:
    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=0))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    callback.handle_occurrence(_update_skipped(), object(), state)
    callback.handle_occurrence(_iteration_completed(), object(), state)

    assert not output_path.exists()


def test_writes_exactly_once_on_first_applied_update(tmp_path: Path) -> None:
    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    callback.handle_occurrence(_update_completed(count=1), object(), state)
    mtime_before = output_path.stat().st_mtime_ns

    # A second applied update must not attempt a second write at all -- the
    # `_written` guard short-circuits before the file is ever reopened, so
    # even an attempted (and would-be-refused) second write cannot happen.
    state.trainer.completed_updates = 2
    callback.handle_occurrence(_update_completed(count=2), object(), state)

    assert output_path.stat().st_mtime_ns == mtime_before


def test_refuses_to_silently_overwrite_an_existing_inventory(tmp_path: Path) -> None:
    """Resumed-run behavior: refuse rather than silently overwrite."""

    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    output_path.write_text("stale-from-a-prior-attempt", encoding="utf-8")
    callback = ParameterActivity(output_path)

    with pytest.raises(FileExistsError):
        callback.handle_occurrence(_update_completed(), object(), state)

    assert output_path.read_text(encoding="utf-8") == "stale-from-a-prior-attempt"
