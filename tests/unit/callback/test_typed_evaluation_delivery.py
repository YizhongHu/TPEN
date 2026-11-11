"""Delivery tests for the evaluation callbacks migrated onto the typed path.

These exist because of a ruling recorded on item ``62593af4``. The dispatcher in
`tpen.artifacts.RunContext._dispatch_occurrence` SKIPS a
`tpen.callback.StatefulCallback` whose domain does not match the state it was
handed, and that same branch fires when the emitter passes no state at all.
Skipping is correct and specified, but it means an emitter that forgets
``state=`` produces a callback that observes nothing, silently, with no error
anywhere -- the identical failure shape as defect ``933b5f78``, where
`GradientStats` reported ``passed: true`` while observing zero gradients.

So every migrated stateful callback gets a test asserting it really is
delivered its state at the boundary it subscribes to, and a paired test pinning
that a forgotten ``state=`` produces total silence. The tests drive the REAL
dispatcher and the REAL `tpen.evaluation.Evaluator`; a `RunContext` stand-in
would override the very method under test.

`tpen.callback.timing.DiagnosticTiming` is the sharpest case: its published
``diagnostics/<task>/failed`` flag is readable ONLY from the state, so ADR-E008's
mechanism is load-bearing for a durable metric key rather than a convenience.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from torch import nn

from tpen.callback import (
    ArtifactIndex,
    Callback,
    DiagnosticTiming,
    EvaluationComponentTiming,
    EvaluationTiming,
    FailureLog,
    StatefulCallback,
    SubscriptionGroup,
)
from tpen.evaluation.events import ComponentFailed, EvaluationTaskRun
from tpen.evaluation.results import TaskResult
from tpen.evaluation.state import EvaluationRunState
from tpen.events import DomainState, Occurrence, Subscription, ended
from tpen.run_events import RunCompleted, RunFailed
from tests.helpers.evaluation_components import (
    FailingCalculator,
    FailingGenerator,
    MissingFieldSummary,
    multi_task_evaluator,
    single_task_evaluator,
)
from tests.helpers.run_context import RecordingLogger, make_run_context


class FakeClock:
    """Deterministic clock; exhaustion is an error, so no call goes unnoticed."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        if not self.values:
            raise AssertionError("fake clock exhausted")
        return self.values.pop(0)


def _model() -> nn.Module:
    return nn.Linear(1, 1)


# --------------------------------------------------------------------------
# The two stateful callbacks: state really arrives
# --------------------------------------------------------------------------


def test_diagnostic_timing_receives_state_at_the_task_boundary(tmp_path: Path) -> None:
    """A real evaluator run delivers the task result, so the record is written."""

    logger = RecordingLogger()
    clock = FakeClock([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.5])
    callback = DiagnosticTiming()
    context = make_run_context(
        tmp_path, callbacks=[callback], loggers=[logger], monotonic_clock=clock
    )

    single_task_evaluator(tmp_path).evaluate(model=_model(), context=context)

    assert "diagnostics/energy" in logger.namespaces(), "DiagnosticTiming observed nothing"
    assert logger.latest("diagnostics/energy") == {"time_sec": 2.5}
    assert logger.steps("diagnostics/energy") == [0]


def test_artifact_index_receives_state_at_the_task_boundary(tmp_path: Path) -> None:
    """A real evaluator run delivers the task result, so the index is written."""

    context = make_run_context(tmp_path, callbacks=[ArtifactIndex()])

    single_task_evaluator(tmp_path).evaluate(model=_model(), context=context)

    written = json.loads((Path(context.run_dir) / "diagnostics" / "index.json").read_text())
    assert [task["namespace"] for task in written["tasks"]] == ["eval/energy"]
    assert written["tasks"][0] == {
        "name": "energy",
        "namespace": "eval/energy",
        "output_dir": str(tmp_path / "energy"),
        "status": "success",
        "artifacts": [],
    }


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("DiagnosticTiming", lambda: DiagnosticTiming(clock=FakeClock([0.0, 1.0]))),
        ("ArtifactIndex", ArtifactIndex),
    ],
)
def test_a_task_scope_opened_without_state_delivers_nothing(
    tmp_path: Path, name: str, build: Any
) -> None:
    """Absent state is skipped, not raised -- which is why the tests above exist.

    This pins the hazard rather than a fix: a forgotten ``state=`` at the
    emitting site really does produce total silence, so the positive tests are
    the only thing standing between that mistake and an empty metric namespace
    nobody notices.
    """

    logger = RecordingLogger()
    context = make_run_context(tmp_path, callbacks=[build()], loggers=[logger])

    with context.scope(
        EvaluationTaskRun(name="energy", namespace="eval/energy", output_dir=tmp_path)
    ):
        pass

    assert logger.records == [], f"{name} logged despite receiving no state"
    assert not (Path(context.run_dir) / "diagnostics" / "index.json").exists()


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("DiagnosticTiming", lambda: DiagnosticTiming(clock=FakeClock([0.0]))),
        ("ArtifactIndex", ArtifactIndex),
    ],
)
def test_a_task_body_that_raises_publishes_nothing(tmp_path: Path, name: str, build: Any) -> None:
    """``Ended`` fires from a ``finally``, so it is reached with an empty state.

    The legacy path emitted neither ``task_end`` nor ``task_failed`` when the
    evaluator raised out of a task body, so neither the ``diagnostics/`` record
    nor the index entry existed. Both callbacks must keep that silence rather
    than publish a duration or an entry with a guessed status.
    """

    logger = RecordingLogger()
    callback = build()
    context = make_run_context(tmp_path, callbacks=[callback], loggers=[logger])
    state = EvaluationRunState()

    with pytest.raises(RuntimeError, match="evaluator blew up"):
        with context.scope(
            EvaluationTaskRun(name="energy", namespace="eval/energy", output_dir=tmp_path),
            state=state,
        ):
            raise RuntimeError("evaluator blew up")

    assert logger.records == [], f"{name} logged from an empty state"
    assert not (Path(context.run_dir) / "diagnostics" / "index.json").exists()


def test_a_foreign_domain_state_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    """One run may carry several domains' states; a mismatch must not kill it."""

    class _ForeignState(DomainState):
        pass

    logger = RecordingLogger()
    context = make_run_context(
        tmp_path, callbacks=[DiagnosticTiming(clock=FakeClock([]))], loggers=[logger]
    )

    with context.scope(
        EvaluationTaskRun(name="energy", namespace="eval/energy", output_dir=tmp_path),
        state=_ForeignState(),
    ):
        pass

    assert logger.records == []


# --------------------------------------------------------------------------
# The three state-free callbacks: delivered without state, and correct
# --------------------------------------------------------------------------


def test_the_state_free_callbacks_observe_the_same_run(tmp_path: Path) -> None:
    """A plain `Callback` sees every boundary regardless of the state beside it.

    Run alongside the stateful pair, this is the cheap version of the mixed-fleet
    check: the same occurrences that a wrong-domain `StatefulCallback` would be
    skipped for are delivered in full to a state-free one. A genuinely mixed
    single run does not exist -- `Train` and `Evaluate` are separate runners --
    so this is the limit of what can be proven here, not full coverage.
    """

    logger = RecordingLogger()
    callbacks = [
        EvaluationComponentTiming(),
        FailureLog(),
        DiagnosticTiming(),
        ArtifactIndex(),
    ]
    context = make_run_context(
        tmp_path,
        callbacks=callbacks,
        loggers=[logger],
        monotonic_clock=FakeClock([0.0, 0.0, 1.0, 2.0, 2.5, 4.0, 4.25, 9.0]),
    )

    single_task_evaluator(tmp_path).evaluate(model=_model(), context=context)

    assert logger.latest("eval/perf/energy") == {
        "generator_time_sec": 1.0,
        "calculator/identity_time_sec": 0.5,
        "summary/metric_time_sec": 0.25,
    }
    assert logger.latest("diagnostics/energy") == {"time_sec": 9.0}
    assert (Path(context.run_dir) / "diagnostics" / "index.json").exists()


def test_evaluation_timing_is_delivered_its_suite_boundaries(tmp_path: Path) -> None:
    """`EvaluationTiming` is fully data-free: no state, no payload, no operation."""

    from tpen.evaluation.events import EvaluationCompleted, EvaluationStarted

    logger = RecordingLogger()
    clock = FakeClock([2.0, 5.5])
    context = make_run_context(
        tmp_path,
        callbacks=[EvaluationTiming()],
        loggers=[logger],
        monotonic_clock=clock,
    )

    context.emit(EvaluationStarted())
    context.emit(EvaluationCompleted())

    assert logger.latest("eval/perf") == {"wall_time_sec": 3.5}
    assert logger.steps("eval/perf") == [0]


# --------------------------------------------------------------------------
# What the state holds AT each boundary, observed rather than reasoned about
# --------------------------------------------------------------------------


class _StateProbe(StatefulCallback[EvaluationRunState]):
    """Record ``state.task_result`` at whichever boundaries it is given."""

    state_type: ClassVar[type[DomainState]] = EvaluationRunState

    def __init__(self, *groups: SubscriptionGroup) -> None:
        super().__init__(typed_groups=groups)
        self.observed: list[TaskResult | None] = []

    def handle_occurrence_impl(
        self, occurrence: Occurrence[Any], context: Any, state: EvaluationRunState
    ) -> None:
        self.observed.append(state.task_result)


def test_the_state_is_still_empty_when_a_component_failure_is_reported(tmp_path: Path) -> None:
    """`ComponentFailed` fires before the task result is written, and is not
    where a subscriber should look for it.

    The evaluator assigns ``state.task_result`` as the LAST statement of the task
    scope's body, so at every earlier boundary the field still holds ``None``.
    `FailureLog` is unaffected because it reads the failure off the event; a
    callback that reached for the state here would silently see nothing. This is
    the same stale-cursor hazard the training side measured, and it is asserted
    from an observed value rather than argued from the source.
    """

    probe = _StateProbe(SubscriptionGroup(selectors=(Subscription.of(ComponentFailed),)))
    context = make_run_context(tmp_path, callbacks=[probe])

    result = single_task_evaluator(
        tmp_path, generator=FailingGenerator()
    ).evaluate(model=_model(), context=context)

    assert result.status == "failed"
    assert probe.observed == [None]


def test_the_task_ended_boundary_carries_the_failed_result(tmp_path: Path) -> None:
    """Evaluation failure is a VALUE, so `Ended` observes it on the failure path.

    This is what lets `DiagnosticTiming` publish ``failed`` without a typed
    task-failure event: the same boundary carries the outcome either way.
    """

    probe = _StateProbe(SubscriptionGroup(selectors=(ended(EvaluationTaskRun),)))
    context = make_run_context(tmp_path, callbacks=[probe])

    result = single_task_evaluator(
        tmp_path, calculators=[FailingCalculator()]
    ).evaluate(model=_model(), context=context)

    assert result.status == "failed"
    assert len(probe.observed) == 1
    assert probe.observed[0] is result.task_results[0]
    assert probe.observed[0].failed is True


# --------------------------------------------------------------------------
# The durable outputs, end to end through a real evaluator
# --------------------------------------------------------------------------


def test_diagnostic_timing_publishes_failed_for_a_failed_task(tmp_path: Path) -> None:
    logger = RecordingLogger()
    context = make_run_context(
        tmp_path,
        callbacks=[DiagnosticTiming()],
        loggers=[logger],
        monotonic_clock=FakeClock([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]),
    )

    single_task_evaluator(tmp_path, calculators=[FailingCalculator()]).evaluate(
        model=_model(), context=context
    )

    assert logger.latest("diagnostics/energy") == {"time_sec": 3.0, "failed": True}


def test_diagnostic_timing_publishes_failed_for_a_partially_failed_task(tmp_path: Path) -> None:
    """A partial failure is a failure for this flag, matching the legacy split.

    The string path selected ``task_failed`` for ``partial_failed`` too, so the
    typed path must agree; `tpen.evaluation.results.TaskResult.failed` is the one
    place that set is now spelled.
    """

    logger = RecordingLogger()
    context = make_run_context(
        tmp_path,
        callbacks=[DiagnosticTiming()],
        loggers=[logger],
        monotonic_clock=FakeClock([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),
    )

    result = single_task_evaluator(tmp_path, summaries=[MissingFieldSummary()]).evaluate(
        model=_model(), context=context
    )

    assert result.task_results[0].status == "partial_failed"
    assert logger.latest("diagnostics/energy") == {"time_sec": 1.5, "failed": True}


def test_failure_log_writes_one_line_per_component_failure(tmp_path: Path) -> None:
    context = make_run_context(tmp_path, callbacks=[FailureLog()])

    single_task_evaluator(tmp_path, calculators=[FailingCalculator()]).evaluate(
        model=_model(), context=context
    )

    lines = [
        json.loads(line)
        for line in (Path(context.run_dir) / "diagnostics" / "failures.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(lines) == 1
    assert lines[0]["component"] == "broken"
    assert lines[0]["component_type"] == "calculator"
    assert lines[0]["error_type"] == "RuntimeError"


def test_the_artifact_index_covers_every_task_in_a_multi_task_suite(tmp_path: Path) -> None:
    context = make_run_context(tmp_path, callbacks=[ArtifactIndex()])

    multi_task_evaluator(tmp_path, "first", "second").evaluate(model=_model(), context=context)

    written = json.loads((Path(context.run_dir) / "diagnostics" / "index.json").read_text())
    assert [task["name"] for task in written["tasks"]] == ["first", "second"]


def test_an_empty_suite_still_writes_an_index(tmp_path: Path) -> None:
    """The one thing `ArtifactIndex`'s run-level subscription still buys.

    Every other run writes the index from the task boundary, so dropping the
    subscription would be byte-identical -- except here, where there is no task
    boundary at all and the empty index would silently stop existing.

    Driven through the REAL dispatcher and through `RunCompleted` ALONE, with no
    legacy string emitted anywhere: that is the whole assertion. The retired
    ``run_end`` trigger reached this callback through `_CallbackCore.handle`,
    which a typed occurrence never enters, so a migration that declared the
    group but never received a delivery would leave this file the only thing
    between it and an index that stopped being written.
    """

    index = ArtifactIndex()
    context = make_run_context(tmp_path, callbacks=[index])

    context.emit(RunCompleted())

    written = json.loads((Path(context.run_dir) / "diagnostics" / "index.json").read_text())
    assert written == {"tasks": []}


def test_an_empty_suite_writes_no_index_when_the_run_failed(tmp_path: Path) -> None:
    """`RunCompleted` is success-only, and so was the string it replaces.

    ``run_end`` was the last statement of ``Train.run`` and ``Evaluate.run``
    before their ``return``, never a ``finally``, so a crashed run never wrote
    an index. Selecting `RunFailed` here as well would invent one.

    NOT DISCRIMINATING on its own: it also passes against the pre-migration
    tree, where `ArtifactIndex` subscribed no run-level typed event at all and
    so wrote nothing for a different reason. It is here to pin the selector
    against a later widening, not to prove this migration.
    """

    index = ArtifactIndex()
    context = make_run_context(tmp_path, callbacks=[index])

    context.emit(RunFailed(exception_type="ValueError", exception_message="boom"))

    assert not (Path(context.run_dir) / "diagnostics" / "index.json").exists()


# --------------------------------------------------------------------------
# A stale ``triggers:`` key cannot resurrect the deleted string path
# --------------------------------------------------------------------------

_MIGRATED: list[tuple[str, Any, tuple[str, ...], tuple[str, ...]]] = [
    (
        "ArtifactIndex",
        ArtifactIndex,
        (),
        ("on_task_end", "on_task_failed", "on_run_end"),
    ),
    (
        "FailureLog",
        FailureLog,
        (),
        ("on_generator_failed", "on_calculator_failed", "on_summary_failed", "on_artifact_failed"),
    ),
    (
        "EvaluationTiming",
        lambda: EvaluationTiming(clock=FakeClock([])),
        (),
        ("on_evaluate_start", "on_evaluate_end", "on_exception"),
    ),
    (
        "EvaluationComponentTiming",
        lambda: EvaluationComponentTiming(clock=FakeClock([])),
        (),
        (
            "on_generator_start",
            "on_generator_end",
            "on_calculator_start",
            "on_calculator_end",
            "on_summary_start",
            "on_summary_end",
            "on_task_end",
            "on_task_failed",
        ),
    ),
    (
        "DiagnosticTiming",
        lambda: DiagnosticTiming(clock=FakeClock([])),
        (),
        (
            "on_diagnostic_start",
            "on_diagnostic_end",
            "on_diagnostic_failed",
            "on_task_start",
            "on_task_end",
            "on_task_failed",
        ),
    ),
]


@pytest.mark.parametrize(
    ("name", "build", "triggers", "dead"),
    _MIGRATED,
    ids=[entry[0] for entry in _MIGRATED],
)
def test_no_migrated_callback_still_answers_a_deleted_trigger(
    name: str, build: Any, triggers: tuple[str, ...], dead: tuple[str, ...]
) -> None:
    """Double firing is structurally impossible, not merely unconfigured.

    Each class dropped its ``on_<name>`` methods, so the legacy dispatch in
    `tpen.callback.base._CallbackCore.handle` finds nothing to call even if a
    config still carries the key. The residual triggers are asserted exactly, so
    they cannot be widened back by accident.

    `EvaluationTiming` lost its last one to item ``39eacd99``: ``exception`` is
    now `tpen.run_events.RunFailed`. `ArtifactIndex` lost its last one,
    ``run_end``, once a subscription group could declare that it needs no domain
    state -- so every entry in this table is now empty, and any non-empty one
    would be a regression.
    """

    del triggers
    callback = build()
    assert not hasattr(callback, "triggers")
    for method in dead:
        assert not hasattr(callback, method), f"{name}.{method} survived the migration"


def test_a_component_selector_may_not_be_split_across_cadence_groups() -> None:
    """ADR-E002: overlapping selectors are rejected loudly, and that is fatal.

    `EvaluationComponentTiming` puts every `ComponentRun` selector in ONE group
    for this reason. The rule is pinned here as well as on the vocabulary,
    because it is a mistake a future subscriber to a single component kind would
    make while editing a callback, not while editing the events.
    """

    from tpen.evaluation.events import CalculatorRun, ComponentRun
    from tpen.events import started

    class _Split(Callback):
        def __init__(self) -> None:
            super().__init__(
                typed_groups=(
                    SubscriptionGroup(selectors=(started(ComponentRun),)),
                    SubscriptionGroup(selectors=(started(CalculatorRun),)),
                )
            )

    with pytest.raises(ValueError, match="overlapping deliveries"):
        _Split()


def test_the_failure_log_emits_ascii_only_bytes(tmp_path: Path) -> None:
    """``FailureLog`` calls ``json.dumps`` itself, so it needs its own pin.

    One of two routed writers the primitive's own ASCII test could not reach,
    because that test file is deliberately torch-free and this module imports
    torch transitively. MEASURED before this existed: ``ensure_ascii=False`` in
    ``callback/evaluation.py`` left the full suite green.

    A failure payload is exactly where non-ASCII is likeliest -- exception text
    carries whatever the underlying library emitted, including non-ASCII paths
    and messages.
    """

    path = tmp_path / "failures.jsonl"
    # A real RunContext, not None: this module is typeguard-checked at runtime,
    # so the annotation is enforced even though ``_path`` ignores the context
    # when an explicit path is set.
    context = make_run_context(tmp_path)

    FailureLog(path=path)._append(context, {"message": "caf\u00e9 \u2014 \u4e2d"})

    raw = path.read_bytes()
    assert raw.isascii(), f"FailureLog emitted non-ASCII bytes: {raw!r}"
