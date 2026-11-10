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
from tpen.checkpoint import CheckpointRestored
from tpen.checkpoint.restore import RestoreReport
from tpen.training.events import TrainingIteration, TrainingIterationCompleted, TrainingStarted, UpdateCompleted, UpdateSkipped
from tpen.training.state import TrainerState
from tpen.events import Occurrence, Started


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


def _update_completed(*, count: int = 1, step: int = 0) -> Occurrence[UpdateCompleted]:
    return Occurrence(event=UpdateCompleted(iteration=TrainingIteration(step=step)), count=count)


def _checkpoint_restored(
    *, next_iteration: int = 0, completed_updates: int = 0
) -> Occurrence[CheckpointRestored]:
    report = RestoreReport(
        mode="train_resume",
        checkpoint_dir="/tmp/source",
        next_iteration=next_iteration,
        completed_updates=completed_updates,
        loaded_model=True,
        loaded_optimizer=True,
        loaded_trainer=True,
        loaded_sampler=True,
        loaded_rng=True,
    )
    return Occurrence(event=CheckpointRestored(report=report), count=1)


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


def test_verified_no_op_on_resumed_lineage_with_prior_inventory(tmp_path: Path) -> None:
    """A resumed lineage (nonzero step) with a verified prior inventory writes nothing.

    Never records a later update's gradients as though they were the
    lineage's first -- the resume-safety contract this callback now shares
    with `InitialCheckpoint`.
    """

    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    output_path = tmp_path / "parameter_activity.json"

    source_state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    source_callback = ParameterActivity(output_path)
    source_callback.handle_occurrence(_update_completed(step=0), _make_context(tmp_path), source_state)
    before = output_path.read_text(encoding="utf-8")
    mtime_before = output_path.stat().st_mtime_ns

    # A fresh callback instance, as a resumed run would construct, observing
    # the resumed lineage's own first applied update at a nonzero step. The
    # restore report is delivered first, the same order the runner emits it.
    resumed_model = _ThreeParamModel()
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    resumed_model.loss().backward()
    resumed_state = TrainerState(
        model=resumed_model, optimizer=resumed_optimizer, trainer=_Trainer(completed_updates=4)
    )
    resumed_callback = ParameterActivity(output_path)
    resumed_context = _make_context(tmp_path, run_id="resumed")
    resumed_callback.handle_occurrence(
        _checkpoint_restored(next_iteration=3, completed_updates=3), resumed_context, None
    )
    resumed_callback.handle_occurrence(
        _update_completed(step=3), resumed_context, resumed_state
    )

    assert resumed_callback._written
    assert output_path.read_text(encoding="utf-8") == before
    assert output_path.stat().st_mtime_ns == mtime_before

    # A redelivery (same context identity) is still a no-op, guarded by
    # `_written`.
    resumed_callback.handle_occurrence(
        _update_completed(step=3, count=2), resumed_context, resumed_state
    )
    assert output_path.stat().st_mtime_ns == mtime_before


def test_refuses_resumed_lineage_with_no_prior_inventory(tmp_path: Path) -> None:
    """A resumed lineage with nothing at `output_path` is refused, never written.

    Pins both halves of the contract in one probe: a missing path is
    refused, and so is a distinct per-attempt path configured for the
    resumed arm -- recording at the next update would mislabel update N+1's
    gradients as the lineage's first, which is worse than a refusal.
    """

    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=4))

    for output_path in (tmp_path / "parameter_activity.json", tmp_path / "parameter_activity_fresh_attempt.json"):
        callback = ParameterActivity(output_path)
        context = _make_context(tmp_path)
        callback.handle_occurrence(_checkpoint_restored(next_iteration=3, completed_updates=3), context, None)
        with pytest.raises(ValueError, match="fresh first-applied-update lineage"):
            callback.handle_occurrence(_update_completed(step=3), context, state)
        assert not output_path.exists()


def test_refuses_resumed_lineage_with_a_malformed_prior_inventory(tmp_path: Path) -> None:
    """A resumed lineage whose prior inventory does not verify is refused.

    Pins that the no-op path actually verifies structure (expected keys,
    interpretation string, row shape), rather than treating any existing
    file as proof.
    """

    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=4))
    output_path = tmp_path / "parameter_activity.json"
    output_path.write_text("not even json", encoding="utf-8")
    callback = ParameterActivity(output_path)
    context = _make_context(tmp_path)
    callback.handle_occurrence(_checkpoint_restored(next_iteration=3, completed_updates=3), context, None)

    with pytest.raises(ValueError, match="fresh first-applied-update lineage"):
        callback.handle_occurrence(_update_completed(step=3), context, state)

    assert output_path.read_text(encoding="utf-8") == "not even json"


def test_refuses_a_restored_lineage_even_with_a_zero_step_first_update(tmp_path: Path) -> None:
    """A `CheckpointRestored` report, not just the step, must gate the write.

    Mirrors `InitialCheckpoint`'s equivalent probe: a lineage resumed from a
    checkpoint written at the zero boundary can have its first applied
    update still read step zero, indistinguishable from a fresh lineage by
    step alone.
    """

    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    state = TrainerState(model=model, optimizer=optimizer, trainer=_Trainer(completed_updates=1))
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)
    # Same context identity for both deliveries: a new `RunContext` instance
    # resets `_restored` (see `_reset_typed_state`), which would otherwise
    # silently defeat this exact probe.
    context = _make_context(tmp_path)

    callback.handle_occurrence(_checkpoint_restored(), context, None)

    with pytest.raises(ValueError, match="fresh first-applied-update lineage"):
        callback.handle_occurrence(_update_completed(step=0), context, state)

    assert not output_path.exists()


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


def test_review_fresh_lineage_records_first_update_after_a_skipped_iteration(tmp_path: Path) -> None:
    """R1-1: iteration zero is not necessarily the first applied update.

    Rejects the mutant that infers resume solely from UpdateCompleted.step.
    The typed event sequence follows the runner/trainer order, in one context.
    """

    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _Trainer(completed_updates=0)
    trainer.next_iteration = 0
    state = TrainerState(model=model, optimizer=optimizer, trainer=trainer)
    context = _make_context(tmp_path)
    output_path = tmp_path / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    callback.handle_occurrence(Occurrence(event=TrainingStarted(), count=1), context, None)
    callback.handle_occurrence(
        Occurrence(event=Started(TrainingIteration(step=0)), count=1), context, state
    )
    callback.handle_occurrence(_update_skipped(), context, state)
    trainer.next_iteration = 1
    callback.handle_occurrence(_iteration_completed(), context, state)
    assert not output_path.exists()

    callback.handle_occurrence(
        Occurrence(event=Started(TrainingIteration(step=1)), count=2), context, state
    )
    model.loss().backward()
    optimizer.step()
    trainer.completed_updates = 1
    callback.handle_occurrence(_update_completed(step=1), context, state)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["completed_updates"] == 1
    assert _rows_by_name(payload)["used"]["gradient_norm"] == pytest.approx(math.sqrt(52))
    before = output_path.read_bytes()
    trainer.completed_updates = 2
    callback.handle_occurrence(_update_completed(step=2, count=2), context, state)
    assert output_path.read_bytes() == before


def test_review_resume_without_prior_updates_never_writes_a_fresh_inventory(tmp_path: Path) -> None:
    """R1-1 green control: C > 0, U = 0 is still a restored lineage.

    Rejects a repair/mutant that uses completed_updates == 1 to declare the
    first resumed applied update fresh. Refusal may happen at any boundary.
    """

    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = _Trainer(completed_updates=0)
    trainer.next_iteration = 2
    state = TrainerState(model=model, optimizer=optimizer, trainer=trainer)
    context = _make_context(tmp_path)
    output_path = tmp_path / "new-attempt" / "parameter_activity.json"
    callback = ParameterActivity(output_path)

    with pytest.raises(ValueError, match="ParameterActivity"):
        callback.handle_occurrence(
            _checkpoint_restored(next_iteration=2, completed_updates=0), context, None
        )
        callback.handle_occurrence(Occurrence(event=TrainingStarted(), count=1), context, None)
        callback.handle_occurrence(
            Occurrence(event=Started(TrainingIteration(step=2)), count=1), context, state
        )
        model.loss().backward()
        optimizer.step()
        trainer.completed_updates = 1
        callback.handle_occurrence(_update_completed(step=2), context, state)

    assert not output_path.exists()


def test_review_resume_preserves_inventory_across_multiple_later_updates(tmp_path: Path) -> None:
    """R1 green control: C > U resume never reopens the original inventory.

    Catches a resume overwrite, writing on the second resumed update, and a
    fresh/resume decision lost when TrainingStarted follows CheckpointRestored.
    """

    model = _ThreeParamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.loss().backward()
    trainer = _Trainer(completed_updates=1)
    state = TrainerState(model=model, optimizer=optimizer, trainer=trainer)
    output_path = tmp_path / "parameter_activity.json"
    ParameterActivity(output_path).handle_occurrence(_update_completed(), _make_context(tmp_path), state)
    before = output_path.read_bytes()
    mtime_before = output_path.stat().st_mtime_ns

    # Distinct live gradients make an attempted rewrite observably different.
    optimizer.zero_grad(set_to_none=True)
    (model.loss() * 7).backward()
    context = _make_context(tmp_path, run_id="review-resumed")
    callback = ParameterActivity(output_path)
    trainer.next_iteration = 3
    callback.handle_occurrence(
        _checkpoint_restored(next_iteration=3, completed_updates=1), context, None
    )
    callback.handle_occurrence(Occurrence(event=TrainingStarted(), count=1), context, None)
    for count, step in enumerate((3, 4), start=1):
        callback.handle_occurrence(
            Occurrence(event=Started(TrainingIteration(step=step)), count=count), context, state
        )
        trainer.completed_updates += 1
        callback.handle_occurrence(_update_completed(step=step, count=count), context, state)
        trainer.next_iteration = step + 1
        callback.handle_occurrence(
            Occurrence(event=TrainingIterationCompleted(TrainingIteration(step=step)), count=count),
            context,
            state,
        )
        assert output_path.read_bytes() == before
        assert output_path.stat().st_mtime_ns == mtime_before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("completed_updates", "corrupt"),
        ("shape", "corrupt"),
        ("gradient", "unknown-status"),
        ("gradient_norm", "corrupt"),
        ("gradient_norm", float("nan")),
        ("interpretation", "wrong-interpretation"),
    ],
    ids=["counter-type", "shape-type", "gradient-enum", "norm-type", "nonfinite-norm", "interpretation-control"],
)
def test_review_resume_refuses_invalid_inventory_values(
    tmp_path: Path, field: str, value: object
) -> None:
    """R1-2: preserving every key does not preserve artifact integrity.

    Rejects the key-presence-only validator. The interpretation case is a
    green control that catches a parse-only verification mutant; no live
    gradient recomputation or comparison is needed.
    """

    row = {
        "name": "weight", "shape": [2], "requires_grad": True,
        "gradient": "nonzero", "gradient_norm": 1.0,
    }
    payload = {
        "completed_updates": 1,
        "interpretation": "observed_on_first_applied_update_not_structural_proof",
        "parameters": [row],
    }
    if field in ("completed_updates", "interpretation"):
        payload[field] = value
    else:
        row[field] = value
    output_path = tmp_path / "parameter_activity.json"
    # The NaN case deliberately writes the non-standard token json.loads
    # accepts by default but the fresh writer's allow_nan=False forbids.
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    before = output_path.read_bytes()
    context = _make_context(tmp_path)
    callback = ParameterActivity(output_path)
    state = TrainerState(trainer=_Trainer(completed_updates=2))

    with pytest.raises(ValueError, match="ParameterActivity"):
        callback.handle_occurrence(
            _checkpoint_restored(next_iteration=3, completed_updates=1), context, None
        )
        callback.handle_occurrence(Occurrence(event=TrainingStarted(), count=1), context, None)
        callback.handle_occurrence(_update_completed(step=3), context, state)

    assert output_path.read_bytes() == before
