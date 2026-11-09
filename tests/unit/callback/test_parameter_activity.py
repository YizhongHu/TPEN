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
from datetime import UTC
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.callback.parameter_activity import ParameterActivity
from tpen.training.events import TrainingIteration, TrainingIterationCompleted, UpdateCompleted, UpdateSkipped
from tpen.training.state import TrainerState
from tpen.events import Occurrence


def _make_context(tmp_path: Path, *, run_id: str = "parameter-activity-unit") -> RunContext:
    """Build a minimal real `RunContext`: `handle_occurrence_impl` is type-checked
    against the concrete `RunContext` class, so a duck-typed stand-in fails that
    check before any assertion runs. This callback never reads from `context`."""

    artifact_manager = ArtifactManager(
        tmp_path / "run-artifacts" / run_id,
        experiment="parameter-activity",
        sector="unit",
        run_id=run_id,
        layout="flat",
    )
    artifact_manager.make_dirs()
    metadata = RunMetadata(
        run_id=run_id,
        run_name=run_id,
        timestamp="2026-11-08T12:00:00+00:00",
        timezone="UTC",
        git_commit="test-sha",
        git_branch="test-branch",
        dirty_worktree=False,
        command="pytest",
        config_path="test.yaml",
        resolved_config_path=str(artifact_manager.path("resolved_config.yaml")),
        run_dir=str(artifact_manager.run_dir),
        device="cpu",
        dtype="float64",
    )
    return RunContext(
        cfg=OmegaConf.create({}),
        source_cfg=OmegaConf.create({}),
        artifact_manager=artifact_manager,
        metadata=metadata,
        clock=RunClock(timezone="UTC", tzinfo=UTC),
    )


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

    callback.handle_occurrence(_update_completed(), _make_context(tmp_path), state)

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

    callback.handle_occurrence(_update_completed(), _make_context(tmp_path), state)

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

    context = _make_context(tmp_path)
    callback.handle_occurrence(_update_skipped(), context, state)
    callback.handle_occurrence(_iteration_completed(), context, state)

    assert not output_path.exists()


def test_writes_exactly_once_on_first_applied_update(tmp_path: Path) -> None:
    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    context = _make_context(tmp_path)
    callback.handle_occurrence(_update_completed(count=1), context, state)
    mtime_before = output_path.stat().st_mtime_ns

    # A second applied update must not attempt a second write at all -- the
    # `_written` guard short-circuits before the file is ever reopened, so
    # even an attempted (and would-be-refused) second write cannot happen.
    state.trainer.completed_updates = 2
    callback.handle_occurrence(_update_completed(count=2), context, state)

    assert output_path.stat().st_mtime_ns == mtime_before


class _NonfiniteGradModel(torch.nn.Module):
    """Parameters carrying genuine NaN/Inf/signed-zero/absent gradients.

    No gradient here comes from a hand-constructed tensor stand-in: every
    populated ``.grad`` is autograd's own output for a loss built with real
    nonfinite intermediates, the same way `LegacyAutogradUpdate` would hand
    an optimizer a nonfinite gradient it still applies.
    """

    def __init__(self) -> None:
        super().__init__()
        self.nan_param = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
        self.inf_param = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
        self.signed_zero_param = torch.nn.Parameter(
            torch.tensor([0.0, -5.0], dtype=torch.float64)
        )
        self.unused_param = torch.nn.Parameter(torch.tensor([7.0], dtype=torch.float64))

    def loss(self) -> torch.Tensor:
        # Multiplying by a scalar NaN/Inf makes autograd's own backward pass
        # produce a nonfinite gradient (the constant multiplier itself),
        # rather than a hand-constructed nonfinite tensor assigned to `.grad`.
        nan_term = (self.nan_param * float("nan")).sum()
        inf_term = (self.inf_param * float("inf")).sum()
        # `signed_zero_param` multiplied by an exact negative zero keeps its
        # gradient exactly zero (signed) rather than NaN, isolating the
        # signed-zero preservation question from the nonfinite one.
        zero_term = (self.signed_zero_param * torch.tensor([-0.0, 0.0], dtype=torch.float64)).sum()
        return nan_term + inf_term + zero_term


class _FiniteTwoGroupModel(torch.nn.Module):
    """Two parameters split across two distinct optimizer param groups."""

    def __init__(self) -> None:
        super().__init__()
        self.group_a = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        self.group_b = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))

    def loss(self) -> torch.Tensor:
        return (self.group_a ** 2).sum() + (self.group_b ** 2).sum()


def test_nonfinite_gradient_does_not_leave_an_unparseable_inventory(tmp_path: Path) -> None:
    """Regression: a NaN/Inf gradient must not strand a malformed artifact.

    `json.dumps(..., allow_nan=False)` raises on the offending float. This
    probe does not assert that the callback catches or re-raises any
    particular way -- only that whatever is left behind at `output_path`,
    if anything, is either absent or a complete, parseable JSON document.
    """

    model = _NonfiniteGradModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    assert model.nan_param.grad is not None and bool(torch.isnan(model.nan_param.grad).any())
    assert model.inf_param.grad is not None and bool(torch.isinf(model.inf_param.grad).any())

    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    with pytest.raises(ValueError):
        callback.handle_occurrence(_update_completed(), _make_context(tmp_path), state)

    if output_path.exists():
        # Must be a complete, parseable document -- never a truncated prefix.
        json.loads(output_path.read_text(encoding="utf-8"))


def test_retry_after_nonfinite_failure_does_not_wrongly_refuse_a_fresh_attempt(
    tmp_path: Path,
) -> None:
    """A resumed/retried run must be able to attempt the write again.

    `FileExistsError` is documented as reserved for a genuine prior
    inventory at the same path. A failed first attempt that nonetheless
    claimed the path with a partial artifact would make this retry raise
    `FileExistsError` instead of re-attempting (and again failing on) the
    nonfinite encode -- conflating "a real inventory is already here" with
    "a doomed write touched this path once".
    """

    model = _NonfiniteGradModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()

    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    with pytest.raises(ValueError):
        callback.handle_occurrence(_update_completed(), _make_context(tmp_path), state)

    # Retry via a fresh context identity, the same way `_written` would be
    # cleared across a resumed run's new `RunContext`.
    second_exc: Exception | None = None
    try:
        callback.handle_occurrence(_update_completed(), _make_context(tmp_path), state)
    except ValueError:
        pass
    except FileExistsError as exc:
        second_exc = exc

    assert second_exc is None, (
        "retry after a failed nonfinite write must not be refused as though a "
        "legitimate prior inventory already existed at this path"
    )


def test_two_optimizer_param_groups_both_track_their_parameters(tmp_path: Path) -> None:
    model = _FiniteTwoGroupModel()
    optimizer = torch.optim.SGD(
        [
            {"params": [model.group_a], "lr": 0.1},
            {"params": [model.group_b], "lr": 0.2},
        ]
    )
    model.loss().backward()
    grad_a_before = model.group_a.grad.detach().clone()
    grad_b_before = model.group_b.grad.detach().clone()

    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)
    callback.handle_occurrence(_update_completed(), _make_context(tmp_path), state)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    rows = _rows_by_name(payload)
    assert rows["group_a"]["gradient"] == "nonzero"
    assert rows["group_b"]["gradient"] == "nonzero"
    # Bitwise-unchanged: the two-group membership check never mutates a
    # gradient from either group.
    assert torch.equal(model.group_a.grad, grad_a_before)
    assert torch.equal(model.group_b.grad, grad_b_before)


def test_absent_and_signed_zero_gradients_are_preserved_bitwise(tmp_path: Path) -> None:
    model = _NonfiniteGradModel()
    optimizer = torch.optim.SGD(
        [model.nan_param, model.inf_param, model.signed_zero_param], lr=0.1
    )
    model.loss().backward()
    assert model.unused_param.grad is None
    zero_grad_before = model.signed_zero_param.grad.detach().clone()
    assert torch.equal(
        torch.signbit(zero_grad_before), torch.tensor([True, False])
    ), "fixture must produce a genuinely signed-zero gradient to be meaningful"

    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    with pytest.raises(ValueError):
        callback.handle_occurrence(_update_completed(), _make_context(tmp_path), state)

    # The nonfinite encode must not have mutated any gradient along the way,
    # including the ones that were never implicated in the failure.
    assert model.unused_param.grad is None
    assert torch.equal(model.signed_zero_param.grad, zero_grad_before)
    assert torch.equal(
        torch.signbit(model.signed_zero_param.grad), torch.tensor([True, False])
    )


def test_reused_callback_refuses_a_second_context_once_a_write_already_landed(
    tmp_path: Path,
) -> None:
    """A callback reused across two `RunContext` identities resets `_written`.

    The second context's attempt reopens the same `output_path` exclusive-
    create, which must raise `FileExistsError` against the first context's
    completed (finite, parseable) inventory -- the documented resumed-run
    refusal, exercised here against a *successful* first write rather than
    a crashed one.
    """

    model = _FiniteTwoGroupModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    first_context = _make_context(tmp_path, run_id="first-context")
    callback.handle_occurrence(_update_completed(), first_context, state)
    assert output_path.exists()

    second_context = _make_context(tmp_path, run_id="second-context")
    with pytest.raises(FileExistsError):
        callback.handle_occurrence(_update_completed(), second_context, state)


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
        callback.handle_occurrence(_update_completed(), _make_context(tmp_path), state)

    assert output_path.read_text(encoding="utf-8") == "stale-from-a-prior-attempt"
