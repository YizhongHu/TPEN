"""Tests for the RuntimeEquivariance callback (multi-checker, artifact writing)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen.artifacts import ArtifactRoutingError
from tpen.callback import RuntimeEquivariance
from tpen.equivariance.checks import EquivarianceCheckResult
from tests.unit.callback.support import (
    RecordingContext,
    deliver_completed_iteration,
    training_state,
)


class FakePassingChecker:
    def __init__(self, n_comparisons: int = 3) -> None:
        self.calls = 0
        self.n_comparisons = n_comparisons

    def run(self, state) -> EquivarianceCheckResult:
        self.calls += 1
        return EquivarianceCheckResult(
            passed=True, metrics={"max_abs_error": 0.0}, n_comparisons=self.n_comparisons
        )


class FakeFailingChecker:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, state) -> EquivarianceCheckResult:
        self.calls += 1
        return EquivarianceCheckResult(
            passed=False,
            metrics={"max_abs_error": 1.0},
            n_comparisons=2,
            failures=["boom"],
            artifact={"checker_class": "FakeFailingChecker", "failures": ["boom"]},
        )


class FakeMisreportingChecker:
    """Returns a comparison count in ``metrics`` that contradicts the field.

    The published record must follow the typed field, not this. A checker that
    could overwrite the key from its own free-form metrics would be able to
    claim it compared something while the contract said it compared nothing --
    which is the failure this key exists to expose.
    """

    def run(self, state) -> EquivarianceCheckResult:
        return EquivarianceCheckResult(
            passed=True,
            metrics={"max_abs_error": 0.0, "n_comparisons": 99},
            n_comparisons=0,
        )


def _handle(callback: RuntimeEquivariance, context: RecordingContext, step: int = 1) -> None:
    # ``state.step`` is deliberately left at the constructor default: the step
    # this callback encodes into an artifact PATH must come from the typed
    # event, so a state carrying a different value would expose a wrong read.
    deliver_completed_iteration(callback, context, training_state(), step=step)


def test_runs_all_checkers_and_logs_class_derived_namespaces() -> None:
    passing, failing = FakePassingChecker(), FakeFailingChecker()
    context = RecordingContext()

    _handle(RuntimeEquivariance(checkers=[passing, failing], fail_fast=False), context)

    assert passing.calls == 1 and failing.calls == 1
    namespaces = {record["namespace"] for record in context.records}
    assert namespaces == {"checks/equivariance/fake_passing", "checks/equivariance/fake_failing"}
    passing_metrics = context.latest("checks/equivariance/fake_passing")
    assert passing_metrics["passed"] is True
    assert passing_metrics["checker_class"] == "FakePassingChecker"


def test_duplicate_names_warn_and_disambiguate_without_failing() -> None:
    context = RecordingContext()
    with pytest.warns(UserWarning, match="duplicate checker name 'fake_passing'"):
        callback = RuntimeEquivariance(
            checkers=[FakePassingChecker(), FakePassingChecker()], fail_fast=False
        )

    _handle(callback, context)

    namespaces = {record["namespace"] for record in context.records}
    assert namespaces == {"checks/equivariance/fake_passing", "checks/equivariance/fake_passing_1"}


def test_writes_artifact_and_adds_artifact_path_metric(tmp_path: Path) -> None:
    context = RecordingContext()
    callback = RuntimeEquivariance(
        checkers=[FakeFailingChecker()], fail_fast=False, artifact_dir=tmp_path
    )

    _handle(callback, context, step=12)

    metrics = context.latest("checks/equivariance/fake_failing")
    artifact_path = Path(metrics["artifact_path"])
    assert artifact_path == tmp_path / "fake_failing" / "step_000012" / "failure.json"
    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text())["checker_class"] == "FakeFailingChecker"


class FakeFailingCheckerWithName(FakeFailingChecker):
    """A failing checker with an explicit, caller-controlled ``name``.

    ``_checker_base_name`` reads this verbatim, so it stands in for a
    checker whose registered owner cannot validate what the checker chose
    to call itself (review-r3 R3.2)."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


def test_r3_2_an_absolute_checker_name_is_refused_and_predecessor_evidence_is_untouched(
    tmp_path: Path,
) -> None:
    """(R3.2) regression: an absolute checker ``name`` would otherwise
    discard ``artifact_dir`` entirely under plain ``Path`` joining
    (``root / "/absolute/predecessor/checks"`` resolves to the absolute
    path alone), letting a registered callback overwrite a predecessor file
    outside its claimed destination. The derived path is checked to resolve
    beneath ``artifact_dir`` before any write, refusing this instead."""

    root = tmp_path / "artifact_root"
    root.mkdir()
    predecessor_dir = tmp_path / "predecessor"
    predecessor_file = predecessor_dir / "checks" / "step_000012" / "failure.json"
    predecessor_file.parent.mkdir(parents=True)
    predecessor_file.write_text('{"predecessor": true}')

    checker = FakeFailingCheckerWithName(name=str(predecessor_dir / "checks"))
    callback = RuntimeEquivariance(checkers=[checker], fail_fast=False, artifact_dir=root)

    with pytest.raises(ArtifactRoutingError):
        _handle(callback, RecordingContext(), step=12)

    assert predecessor_file.read_text() == '{"predecessor": true}'
    assert list(root.rglob("*")) == []


def test_r3_2_a_traversal_checker_name_is_refused(tmp_path: Path) -> None:
    """(R3.2) regression: a ``..``-bearing checker name also escapes
    ``artifact_dir`` through plain ``Path`` joining; refused the same way."""

    root = tmp_path / "artifact_root"
    root.mkdir()
    checker = FakeFailingCheckerWithName(name="../escape")
    callback = RuntimeEquivariance(checkers=[checker], fail_fast=False, artifact_dir=root)

    with pytest.raises(ArtifactRoutingError):
        _handle(callback, RecordingContext(), step=12)

    assert list(root.rglob("*")) == []


def test_r3_2_a_normal_checker_name_still_publishes(tmp_path: Path) -> None:
    """(R3.2) negative: an ordinary checker name is unaffected -- the
    containment check is a floor, not a new restriction on legitimate
    names."""

    root = tmp_path / "artifact_root"
    checker = FakeFailingCheckerWithName(name="custom_checker")
    callback = RuntimeEquivariance(checkers=[checker], fail_fast=False, artifact_dir=root)

    _handle(callback, RecordingContext(), step=12)

    artifact_path = root / "custom_checker" / "step_000012" / "failure.json"
    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text())["checker_class"] == "FakeFailingChecker"


def test_no_artifact_written_without_artifact_dir() -> None:
    context = RecordingContext()
    callback = RuntimeEquivariance(checkers=[FakeFailingChecker()], fail_fast=False)

    _handle(callback, context)

    assert "artifact_path" not in context.latest("checks/equivariance/fake_failing")


def test_fail_fast_raises_on_failure() -> None:
    callback = RuntimeEquivariance(checkers=[FakeFailingChecker()], fail_fast=True)

    with pytest.raises(RuntimeError, match="fake_failing"):
        _handle(callback, RecordingContext())


def test_no_raise_when_not_fail_fast() -> None:
    context = RecordingContext()

    _handle(RuntimeEquivariance(checkers=[FakeFailingChecker()], fail_fast=False), context)

    assert context.latest("checks/equivariance/fake_failing")["passed"] is False


def test_checkers_run_only_on_scheduled_steps() -> None:
    checker = FakePassingChecker()
    callback = RuntimeEquivariance(checkers=[checker], every_n_steps=2, fail_fast=False)

    for step in range(0, 5):
        deliver_completed_iteration(
            callback, RecordingContext(), training_state(), step=step
        )

    assert checker.calls == 3  # steps 0, 2, 4


def test_artifact_step_comes_from_the_event_not_the_state(tmp_path: Path) -> None:
    # This is the only migrated callback whose step reaches a durable PATH.
    # `f"{-1:06d}"` renders "-00001", so a callback reading a stale
    # ``state.step`` would write ``step_-00001/`` on the first iteration and
    # collide iteration k with k-1 thereafter -- silently. The state here holds
    # exactly that stale default while the event says 12.
    context = RecordingContext()
    callback = RuntimeEquivariance(
        checkers=[FakeFailingChecker()], fail_fast=False, artifact_dir=tmp_path
    )
    stale = training_state(step=-1)

    deliver_completed_iteration(callback, context, stale, step=12)

    metrics = context.latest("checks/equivariance/fake_failing")
    assert Path(metrics["artifact_path"]) == tmp_path / "fake_failing" / "step_000012" / "failure.json"
    assert not (tmp_path / "fake_failing" / "step_-00001").exists()


def test_cadence_gates_on_the_durable_step_not_the_occurrence_count() -> None:
    # A run-local `Occurrence.count` restarts after a checkpoint restore; the
    # durable step does not. This is the most expensive check in the stack, so
    # a schedule that shifted phase on a resumed run would be visible.
    checker = FakePassingChecker()
    callback = RuntimeEquivariance(checkers=[checker], every_n_steps=5, fail_fast=False)
    context = RecordingContext()

    for step in range(12):
        deliver_completed_iteration(callback, context, training_state(), step=step, count=1)

    assert [r["step"] for r in context.by_namespace("checks/equivariance/fake_passing")] == [0, 5, 10]
    assert checker.calls == 3


# --- an empty checker list must be loud, not silently absent ---


@pytest.mark.parametrize("fail_fast", [True, False])
def test_empty_checker_list_is_rejected_at_construction(fail_fast: bool) -> None:
    # Every record this callback publishes comes from inside the per-checker
    # loop, so an empty list makes the whole `checks/equivariance/*` namespace
    # vanish rather than fail -- a vacuous ABSENCE, which is harder to notice
    # than a vacuous pass because it leaves no record to inspect. Parametrised
    # over `fail_fast` because a log-time failing record would have been silent
    # in the `False` mode, which is exactly the case being closed.
    with pytest.raises(ValueError, match="requires at least one checker"):
        RuntimeEquivariance(checkers=[], fail_fast=fail_fast)


def test_empty_checker_rejection_survives_other_constructor_arguments() -> None:
    # `every_n_steps` is popped from kwargs before the base class runs, so the
    # rejection has to happen ahead of that bookkeeping rather than after it.
    with pytest.raises(ValueError, match="requires at least one checker"):
        RuntimeEquivariance(checkers=(), every_n_steps=5, fail_fast=False)


def test_a_configured_checker_always_produces_a_record() -> None:
    # The positive half of the same contract: construction succeeded, so the
    # namespace must appear. Guards the loop against a future gate that could
    # skip the `context.log` call and take the namespace with it.
    context = RecordingContext()

    _handle(RuntimeEquivariance(checkers=[FakePassingChecker()], fail_fast=False), context)

    assert context.by_namespace("checks/equivariance/fake_passing")


# --- n_comparisons: a zero-comparison pass must be structurally visible ---


def test_comparison_count_is_published_on_every_record() -> None:
    context = RecordingContext()
    callback = RuntimeEquivariance(
        checkers=[FakePassingChecker(n_comparisons=7), FakeFailingChecker()], fail_fast=False
    )

    _handle(callback, context)

    assert context.latest("checks/equivariance/fake_passing")["n_comparisons"] == 7
    assert context.latest("checks/equivariance/fake_failing")["n_comparisons"] == 2


def test_zero_comparison_pass_is_visible_in_the_record() -> None:
    # `passed` is well defined over an empty comparison set -- no permutation
    # failed because none was tested -- so without this key the record would
    # read as a healthy check that measured nothing.
    context = RecordingContext()
    callback = RuntimeEquivariance(
        checkers=[FakePassingChecker(n_comparisons=0)], fail_fast=True
    )

    _handle(callback, context)

    metrics = context.latest("checks/equivariance/fake_passing")
    assert metrics["passed"] is True
    assert metrics["n_comparisons"] == 0


def test_comparison_count_comes_from_the_field_not_checker_metrics() -> None:
    # Mirrors how `passed` is published: the typed field is authoritative, so a
    # checker cannot dress an empty check up as a busy one via free-form metrics.
    context = RecordingContext()

    _handle(RuntimeEquivariance(checkers=[FakeMisreportingChecker()], fail_fast=False), context)

    assert context.latest("checks/equivariance/fake_misreporting")["n_comparisons"] == 0
