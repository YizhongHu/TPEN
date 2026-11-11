"""Delivery and emit-site tests for the typed run lifecycle (item ``39eacd99``).

Two distinct hazards are covered here, and they need different kinds of test.

DELIVERY. Required by the ruling on item ``62593af4``: a migrated callback gets a
test proving it is really reached at the boundary it subscribes to, driven through
the REAL `tpen.artifacts.RunContext` dispatcher, because a `RunContext` stand-in
would override the very routing under test.

THE EMIT SITES. `tpen.run_events` has exactly one emitter, `tpen.run`, and every
delivery test below emits its own events, so all of them would keep passing if
`run_from_config` dropped an emit. A forgotten emit is silent -- no ``config.yaml``,
no ``metadata.json``, no ``runtime/wall_time_sec``, and no error anywhere. The
end-to-end tests drive the real `run_from_config` for that reason, and they are
the only thing standing between that mistake and a run that quietly stops
recording itself.

Also pinned here: `tpen.callback.Status` and `tpen.callback.ArtifactIndex` now
have NO legacy triggers left. The MEASURED reason they once had to, when this
slice landed, was that both are `StatefulCallback`s, the run lifecycle carries no
domain state, and the dispatcher's ``isinstance(state, callback.state_type)``
branch therefore skipped them at every run-level boundary -- silently.

Item ``24f91145`` removed that constraint by moving the gate from per-callback to
per-GROUP, so ``test_a_state_free_run_event_now_reaches_a_stateful_callback``
records the same scenario with the opposite outcome, and the two callbacks
consumed the mechanism directly afterwards. Their own delivery tests live in
``test_status.py`` and ``test_typed_evaluation_delivery.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from omegaconf import OmegaConf

from tpen.artifacts import RunContext, RunResult
from tpen.callback import (
    ArtifactIndex,
    Callback,
    ConfigSnapshot,
    EvaluationTiming,
    Metadata,
    ResolvedConfigSnapshot,
    ResourceUsage,
    RunTiming,
    StatefulCallback,
    Status,
    SubscriptionGroup,
)
from tpen.evaluation.events import EvaluationStarted
from tpen.events import DomainState, Occurrence, Subscription
from tpen.run import run_from_config
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tpen.runner import Runner
from tests.helpers.run_context import RecordingLogger, make_run_context


class FakeClock:
    """Deterministic clock; exhaustion is an error, so no call goes unnoticed."""

    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        if not self.values:
            raise AssertionError("clock exhausted")
        return self.values.pop(0)


def _failure() -> RunFailed:
    return RunFailed(exception_type="RuntimeError", exception_message="boom")


# --------------------------------------------------------------------------
# Per-callback delivery, through the real dispatcher
# --------------------------------------------------------------------------


def test_metadata_records_a_status_at_each_of_the_three_boundaries(tmp_path: Path) -> None:
    """One data-free callback, three moments, three durable statuses."""

    output = tmp_path / "metadata.json"
    context = make_run_context(tmp_path, callbacks=[Metadata(output_path=output)])

    context.emit(RunStarted())
    assert json.loads(output.read_text())["status"] == "running"

    context.emit(RunCompleted())
    assert json.loads(output.read_text())["status"] == "completed"

    context.emit(_failure())
    written = json.loads(output.read_text())
    assert written["status"] == "failed"
    assert written["exception_type"] == "RuntimeError"
    assert written["exception_message"] == "boom"


def test_metadata_reads_the_failure_off_the_typed_event(tmp_path: Path) -> None:
    """The two fields come from the event, not from an untyped payload probe.

    The legacy path pulled a live exception out of ``event.payload`` and derived
    both strings here. `tpen.run` already had them, so they now ride
    `RunFailed`; this asserts the values survive the move rather than being
    silently replaced by placeholders.
    """

    output = tmp_path / "metadata.json"
    context = make_run_context(tmp_path, callbacks=[Metadata(output_path=output)])

    context.emit(RunFailed(exception_type="FileNotFoundError", exception_message="no /x"))

    written = json.loads(output.read_text())
    assert (written["exception_type"], written["exception_message"]) == (
        "FileNotFoundError",
        "no /x",
    )


def test_both_snapshots_write_once_at_the_start_boundary(tmp_path: Path) -> None:
    """``config.yaml`` keeps its interpolations; ``resolved_config.yaml`` loses them."""

    context = make_run_context(tmp_path)
    context.cfg = OmegaConf.create({"a": 1, "b": "${a}"})
    context.source_cfg = OmegaConf.create({"a": 1, "b": "${a}"})
    source = tmp_path / "snap" / "config.yaml"
    resolved = tmp_path / "snap" / "resolved_config.yaml"
    context.callbacks = [
        ConfigSnapshot(output_path=source),
        ResolvedConfigSnapshot(output_path=resolved),
    ]

    context.emit(RunStarted())

    assert "${a}" in source.read_text()
    assert "${a}" not in resolved.read_text()

    # Neither terminal boundary rewrites a snapshot.
    source.unlink()
    resolved.unlink()
    context.emit(RunCompleted())
    context.emit(_failure())
    assert not source.exists() and not resolved.exists()


def test_run_timing_measures_the_successful_run(tmp_path: Path) -> None:
    logger = RecordingLogger()
    clock = FakeClock([10.0, 12.5])
    callback = RunTiming(wall_clock=FakeClock([100.0, 103.0]))
    context = make_run_context(
        tmp_path, callbacks=[callback], loggers=[logger], monotonic_clock=clock
    )

    context.emit(RunStarted())
    context.emit(RunCompleted())

    assert [dict(record.metrics) for record in logger.by_namespace("runtime")] == [
        {"start_time_unix": 100.0},
        {"end_time_unix": 103.0, "wall_time_sec": 2.5},
    ]
    assert logger.steps("runtime") == [0, 0]


def test_run_timing_reports_failed_exactly_once_per_failed_run(tmp_path: Path) -> None:
    """The collapse of ``run_failed`` and ``exception`` into one event, pinned.

    `tpen.run` emitted both strings back to back with the same payload, and this
    callback's shipped default ``triggers`` answered both, so a failed run logged
    ``runtime`` TWICE -- with a slightly later ``end_time_unix`` the second time,
    so not even identically. The clock is exhausted after two reads, which is why
    a regression to two firings raises here rather than merely comparing unequal.
    """

    logger = RecordingLogger()
    clock = FakeClock([1.0, 4.0])
    callback = RunTiming(wall_clock=FakeClock([10.0, 13.0]))
    context = make_run_context(
        tmp_path, callbacks=[callback], loggers=[logger], monotonic_clock=clock
    )

    context.emit(RunStarted())
    context.emit(_failure())

    records = logger.by_namespace("runtime")
    assert len(records) == 2
    assert dict(records[-1].metrics) == {
        "end_time_unix": 13.0,
        "wall_time_sec": 3.0,
        "failed": True,
    }


def test_resource_usage_resets_at_the_start_and_reports_once_at_the_first_end(tmp_path: Path) -> None:
    logger = RecordingLogger()
    callback = ResourceUsage(peak_rss_mb_reader=lambda: 512.0)
    context = make_run_context(tmp_path, callbacks=[callback], loggers=[logger])

    context.emit(RunStarted())
    assert logger.records == []

    context.emit(RunCompleted())
    assert logger.latest("runtime")["peak_memory_mb"] == 512.0

    context.emit(_failure())
    # One report per logical run: a second terminal boundary on the same
    # context (contract #9) does not produce a second record.
    assert len(logger.by_namespace("runtime")) == 1


def test_evaluation_timing_reports_failed_off_the_typed_run_event(tmp_path: Path) -> None:
    """The last legacy run-level trigger in the codebase, now typed.

    ``eval/perf {failed: True}`` has exactly one writer, because the evaluation
    domain has no suite-level failure moment of its own. It used to arrive on the
    ``exception`` string; it now arrives on `RunFailed`, and the metric is
    unchanged (ADR-E006).
    """

    logger = RecordingLogger()
    clock = FakeClock([2.0, 6.0])
    callback = EvaluationTiming()
    context = make_run_context(
        tmp_path, callbacks=[callback], loggers=[logger], monotonic_clock=clock
    )

    context.emit(EvaluationStarted())
    context.emit(_failure())

    assert dict(logger.latest("eval/perf")) == {"wall_time_sec": 4.0, "failed": True}
    assert logger.steps("eval/perf") == [0]


def test_evaluation_timing_reports_nothing_when_the_suite_never_started(tmp_path: Path) -> None:
    """A run that failed before evaluating has no evaluation duration to report."""

    logger = RecordingLogger()
    context = make_run_context(
        tmp_path, callbacks=[EvaluationTiming(clock=FakeClock([]))], loggers=[logger]
    )

    context.emit(_failure())

    assert logger.records == []


# --------------------------------------------------------------------------
# The emit sites in `tpen.run`, which no delivery test above can cover
# --------------------------------------------------------------------------


class _NoopRunner(Runner):
    """Runner that succeeds without touching torch, a model, or a sampler."""

    def run(self, context: RunContext) -> RunResult:
        del context
        return RunResult(status="completed")


class _CompletionCallbackThatFails(Callback):
    """Subscriber that turns a clean `RunCompleted` into a harness failure.

    Writer-authored for contract #9 (R1-repair item 003dada1): models the
    one logical run that reaches `RunCompleted` and is then reported as
    `RunFailed` because a later subscriber blew up while handling it.
    """

    def __init__(self) -> None:
        super().__init__(
            typed_groups=(
                SubscriptionGroup(selectors=(Subscription.of(RunStarted),)),
                SubscriptionGroup(selectors=(Subscription.of(RunCompleted),)),
            )
        )

    def handle_occurrence_impl(
        self, occurrence: Occurrence[Any], context: RunContext
    ) -> None:
        del context
        if isinstance(occurrence.event, RunCompleted):
            raise RuntimeError("completion callback exploded")


class _RaisingRunner(Runner):
    """Runner that fails inside ``run``, after the start boundary has fired."""

    def run(self, context: RunContext) -> RunResult:
        del context
        raise RuntimeError("runner exploded")


def _cfg(tmp_path: Path, runner_target: str) -> Any:
    return OmegaConf.create(
        {
            "experiment": {"name": "lifecycle", "sector": "unit", "run_name": "lifecycle"},
            "run": {"root": str(tmp_path), "run_id": None, "dir": None},
            "runtime": {"seed": 0},
            "runner": {"_target_": runner_target},
            "callbacks": [
                {
                    "_target_": "tpen.callback.ConfigSnapshot",
                    "output_path": "${run.dir}/config.yaml",
                },
                {
                    "_target_": "tpen.callback.ResolvedConfigSnapshot",
                    "output_path": "${run.dir}/resolved_config.yaml",
                },
                {
                    "_target_": "tpen.callback.Metadata",
                    "output_path": "${run.dir}/metadata.json",
                },
                {"_target_": "tpen.callback.RunTiming"},
            ],
            "loggers": [
                {"_target_": "tpen.logging.JSONL", "path": "${run.dir}/metrics.jsonl"}
            ],
        }
    )


def _run_dir(tmp_path: Path) -> Path:
    dirs = list(tmp_path.glob("lifecycle/unit/*"))
    assert len(dirs) == 1, dirs
    return dirs[0]


def _runtime_metrics(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return [record for record in records if record.get("namespace") == "runtime"]


def _occurrences(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "occurrences.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_the_harness_emits_the_whole_success_lifecycle(tmp_path: Path) -> None:
    """End-to-end: every run-level artifact is written by a typed subscriber.

    Each of these four files or series used to depend on a legacy string
    trigger, and each would silently stop existing if `run_from_config` forgot
    an emit.
    """

    target = f"{__name__}._NoopRunner"
    assert run_from_config(_cfg(tmp_path, target), config_path="x", command="pytest") == 0

    run_dir = _run_dir(tmp_path)
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "resolved_config.yaml").is_file()
    assert json.loads((run_dir / "metadata.json").read_text())["status"] == "completed"

    runtime = _runtime_metrics(run_dir)
    assert "start_time_unix" in runtime[0]["metrics"]
    assert "wall_time_sec" in runtime[-1]["metrics"]
    assert "failed" not in runtime[-1]["metrics"]

    names = [record["event"] for record in _occurrences(run_dir)]
    assert names[0] == "tpen.run_events.RunStarted"
    assert names[-1] == "tpen.run_events.RunCompleted"


def test_the_harness_emits_run_started_exactly_once(tmp_path: Path) -> None:
    """One typed emitter, so no de-duplication flag is needed on this path.

    The legacy ``run_start`` string is emitted by BOTH `tpen.run` and each
    runner, which is why `RunContext` and `tpen.runner.Runner` each carry a
    ``_run_start_emitted`` guard. The typed lifecycle has a single emitter
    instead. A second `RunStarted` would re-render the status boxes, rewrite the
    snapshots, and restart the run clock, so the count matters.
    """

    target = f"{__name__}._NoopRunner"
    assert run_from_config(_cfg(tmp_path, target), config_path="x", command="pytest") == 0

    names = [record["event"] for record in _occurrences(_run_dir(tmp_path))]
    assert names.count("tpen.run_events.RunStarted") == 1
    assert names.count("tpen.run_events.RunCompleted") == 1


def test_resource_usage_reports_once_per_run_when_a_later_callback_fails_after_completion(
    tmp_path: Path,
) -> None:
    """Contract #9 (R1-repair item 003dada1): one logical run, one report.

    `RunCompleted` fires and `ResourceUsage` answers it; a later subscriber
    then blows up handling that same event, so the harness reports the run as
    `RunFailed` on top of the completion that already happened. Scoped to
    what R1 can assert directly: `metrics.jsonl`, and to `ResourceUsage`
    alone -- `cfg.callbacks` carries only `ResourceUsage` and the failing
    subscriber, not the default `RunTiming`, which has its own unfixed,
    unreviewed double-report on this same `RunCompleted`/`RunFailed` pair.
    Mixing it in would make this test depend on a fix this item does not
    own. The per-rank `profiles/.../resources.jsonl` artifact is R3's
    launcher/rank-writer surface, which does not exist at this layer
    either -- that assertion belongs to R3, not here.
    """

    cfg = _cfg(tmp_path, f"{__name__}._NoopRunner")
    cfg.callbacks = [
        {"_target_": "tpen.callback.ResourceUsage"},
        {"_target_": f"{__name__}._CompletionCallbackThatFails"},
    ]

    assert run_from_config(cfg, config_path="x", command="pytest") == 1

    run_dir = _run_dir(tmp_path)
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len([record for record in metrics if record.get("namespace") == "runtime"]) == 1
    assert len([record for record in metrics if record.get("namespace") == "process"]) == 1


def test_resource_usage_writes_exactly_one_device_and_process_profile_line_once_when_completion_callback_fails(
    tmp_path: Path,
) -> None:
    """Writer-authored for contract #9 (ruling `ruling-contract-9-test-disposed`).

    The reviewer's own version of this cross-layer check (22b47d28) asserted
    `profiles/rank-00000/resources.jsonl` through `_cfg()`'s DEFAULT callback
    list, which also carries `RunTiming`. `RunTiming` logs to the same
    `runtime` namespace as `ResourceUsage` on its own (see
    `test_the_harness_emits_run_failed_once_and_records_the_failure` above),
    so the reviewer's `runtime`-count-of-1 assertion is unsatisfiable under
    that callback set regardless of any fix -- disposed rather than adopted.
    This test isolates the callback list to `[ResourceUsage,
    _CompletionCallbackThatFails]`, exactly as R1's sibling test above does,
    and asserts the property the reviewer was reaching for at the layer
    where the rank-local artifact actually exists: one logical run that
    reaches `RunCompleted` and is then reported `RunFailed` by a later
    subscriber still writes exactly one device profile record and one
    process profile record, not two of each.
    """

    cfg = _cfg(tmp_path, f"{__name__}._NoopRunner")
    cfg.callbacks = [
        {"_target_": "tpen.callback.ResourceUsage"},
        {"_target_": f"{__name__}._CompletionCallbackThatFails"},
    ]

    assert run_from_config(cfg, config_path="x", command="pytest") == 1

    run_dir = _run_dir(tmp_path)
    profile_lines = [
        json.loads(line)
        for line in (run_dir / "profiles" / "rank-00000" / "resources.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [line["scope"] for line in profile_lines] == ["device", "process"]


def test_the_harness_emits_run_failed_once_and_records_the_failure(tmp_path: Path) -> None:
    """The failure boundary: one typed event where two strings used to fire.

    ``runtime`` carries exactly ONE failed record, which is the observable
    difference the collapse makes, and ``metadata.json`` carries the failure
    identity that used to travel in an untyped payload.
    """

    target = f"{__name__}._RaisingRunner"
    assert run_from_config(_cfg(tmp_path, target), config_path="x", command="pytest") == 1

    run_dir = _run_dir(tmp_path)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["exception_type"] == "RuntimeError"
    assert metadata["exception_message"] == "runner exploded"

    failed = [record for record in _runtime_metrics(run_dir) if record["metrics"].get("failed")]
    assert len(failed) == 1

    occurrences = _occurrences(run_dir)
    assert [record["event"] for record in occurrences].count("tpen.run_events.RunFailed") == 1
    (record,) = [
        entry for entry in occurrences if entry["event"] == "tpen.run_events.RunFailed"
    ]
    assert record["fields"] == {
        "exception_type": "RuntimeError",
        "exception_message": "runner exploded",
    }


def test_a_run_that_never_reached_its_runner_still_reports_both_boundaries(
    tmp_path: Path,
) -> None:
    """`RunStarted` precedes runner construction, so a bad runner still reports.

    This is the shape of the real ``invalid load path`` failure: the run dies
    before any runner exists. `RunStarted` has already fired, so the failure
    boundary finds a `RunTiming` with a live start timestamp and can report a
    duration rather than nothing.
    """

    cfg = _cfg(tmp_path, "builtins.object")
    assert run_from_config(cfg, config_path="x", command="pytest") == 1

    run_dir = _run_dir(tmp_path)
    assert (run_dir / "config.yaml").is_file()
    assert json.loads((run_dir / "metadata.json").read_text())["status"] == "failed"
    failed = [record for record in _runtime_metrics(run_dir) if record["metrics"].get("failed")]
    assert len(failed) == 1
    assert "wall_time_sec" in failed[0]["metrics"]


def test_a_raising_subscriber_on_the_failure_path_does_not_mask_the_failure(
    tmp_path: Path,
) -> None:
    """The typed emit is guarded exactly as the legacy string emits were.

    This path runs after the run has already failed, from a context that may be
    half-constructed. An emit that propagated would replace the exception the
    user needs with a callback error, which is the worst possible outcome here.
    """

    cfg = _cfg(tmp_path, f"{__name__}._RaisingRunner")
    cfg.callbacks.append({"_target_": f"{__name__}._ExplodingOnFailure"})

    assert run_from_config(cfg, config_path="x", command="pytest") == 1

    # The original failure still reached the durable record.
    error = json.loads((_run_dir(tmp_path) / "error.json").read_text())
    assert error["exception_type"] == "RuntimeError"
    assert error["exception_message"] == "runner exploded"


class _ExplodingOnFailure(Metadata):
    """Subscriber that raises at the failure boundary and nowhere else."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(output_path="/dev/null", **kwargs)

    def handle_occurrence_impl(
        self, occurrence: Occurrence[Any], context: RunContext
    ) -> None:
        if isinstance(occurrence.event, RunFailed):
            raise RuntimeError("subscriber exploded while reporting a failure")


# --------------------------------------------------------------------------
# The blocker this slice measured, and its removal in ``24f91145``
# --------------------------------------------------------------------------


class _ForeignState(DomainState):
    """Stand-in for some domain's state, so the skip has something to mismatch."""


class _RunLevelStateful(StatefulCallback[_ForeignState]):
    """A stateful callback subscribing the run lifecycle, as it once could not.

    Its group declares ``stateless``, which is exactly the capability
    ``24f91145`` added and this slice measured to be missing. Before it, a
    `StatefulCallback` subscribing a run-level event observed nothing at all,
    because `RunContext._dispatch_occurrence` decided delivery once per callback
    on ``isinstance(state, callback.state_type)`` and the run lifecycle carries
    no domain state.
    """

    state_type: ClassVar[type[DomainState]] = _ForeignState

    def __init__(self) -> None:
        super().__init__(
            typed_groups=(
                SubscriptionGroup(selectors=(Subscription.of(EvaluationStarted),)),
                SubscriptionGroup(
                    selectors=(Subscription.of(RunStarted),), stateless=True
                ),
            ),
        )
        self.seen: list[Any] = []
        self.domain_seen: list[Any] = []

    def handle_occurrence_impl(
        self, occurrence: Occurrence[Any], context: RunContext, state: _ForeignState
    ) -> None:
        del context, state
        self.domain_seen.append(occurrence.event)

    def handle_stateless_occurrence_impl(
        self, occurrence: Occurrence[Any], context: RunContext
    ) -> None:
        del context
        self.seen.append(occurrence.event)


def test_a_state_free_run_event_now_reaches_a_stateful_callback(tmp_path: Path) -> None:
    """The constraint this slice recorded, and its removal.

    This test asserted the opposite when #182 landed: a `StatefulCallback`
    subscribing a run-level event observed NOTHING, silently, because a callback
    declares one ``state_type`` and "observes a domain AND the run lifecycle" was
    inexpressible. That was the whole remaining blocker for ``85870732``.

    ``24f91145`` moved the state gate from per-callback to per-GROUP, so the same
    class can now declare one group that wants its domain's state and another
    that wants a state-free boundary. Kept here rather than deleted so the record
    shows the constraint and its removal in one place; the mechanism's own tests
    are in ``test_stateless_subscription_group.py``.
    """

    callback = _RunLevelStateful()
    context = make_run_context(tmp_path, callbacks=[callback])

    context.emit(RunStarted())
    assert callback.seen == [RunStarted()]

    # And the domain group on the SAME callback still receives its state, which
    # is what makes this a capability rather than a relaxed filter.
    state = _ForeignState()
    context.emit(EvaluationStarted(), state=state)
    assert callback.domain_seen == [EvaluationStarted()]


def test_status_is_reached_at_every_run_boundary_by_the_real_dispatcher(
    tmp_path: Path,
) -> None:
    """`Status` through `RunContext`, which is what its whole artifact depends on.

    Required by the ruling on item ``62593af4``: the delivery itself is the
    thing that can silently stop happening, and only the real dispatcher can
    show it happening. `Status` is a `StatefulCallback`, so this is the path
    where `_dispatch_occurrence` hands it ``state=None`` and its group's
    ``stateless`` declaration is what keeps the occurrence from being dropped.

    Three separate runs rather than one sequence, because a run reaches at most
    one terminal boundary and a single ``status.json`` would only show the last.
    """

    statuses = []
    for name, event in (
        ("started", RunStarted()),
        ("completed", RunCompleted()),
        ("failed", RunFailed(exception_type="ValueError", exception_message="boom")),
    ):
        output_path = tmp_path / f"{name}.json"
        status = Status(output_path=output_path, terminal=False)
        context = make_run_context(tmp_path / name, callbacks=[status])
        context.emit(RunStarted())
        if not isinstance(event, RunStarted):
            context.emit(event)
        statuses.append(json.loads(output_path.read_text()))

    assert [entry["status"] for entry in statuses] == ["running", "completed", "failed"]
    assert statuses[2]["exception_type"] == "ValueError"
    assert statuses[2]["exception_message"] == "boom"


def test_no_callback_anywhere_still_answers_a_run_level_string() -> None:
    """The precondition item ``85870732`` deletes the string path under.

    `Status` and `ArtifactIndex` were the LAST two consumers of a run-level
    string anywhere in `tpen`; every other callback reached ``triggers=()`` in
    an earlier slice. With them migrated, ``run_start``, ``run_end``,
    ``run_failed`` and ``exception`` are emitted to nobody at all.

    Asserted over every callback class rather than over those two, because the
    claim that makes the deletion safe is the universal one, and a NEW callback
    reaching for a run-level string is exactly what would quietly falsify it.
    """

    from tpen import callback as callback_package

    run_level = {"run_start", "run_end", "run_failed", "exception"}
    offenders = {}
    for name in callback_package.__all__:
        attribute = getattr(callback_package, name)
        if not isinstance(attribute, type) or not issubclass(
            attribute, (Callback, StatefulCallback)
        ):
            continue
        answered = run_level & {
            method[len("on_") :] for method in dir(attribute) if method.startswith("on_")
        }
        if answered:
            offenders[name] = sorted(answered)

    assert offenders == {}


# --------------------------------------------------------------------------
# A stale `triggers:` key cannot resurrect the legacy path for these six
# --------------------------------------------------------------------------


_MIGRATED: list[tuple[str, Any]] = [
    ("Metadata", lambda **kw: Metadata(output_path="metadata.json", **kw)),
    ("ConfigSnapshot", lambda **kw: ConfigSnapshot(output_path="config.yaml", **kw)),
    (
        "ResolvedConfigSnapshot",
        lambda **kw: ResolvedConfigSnapshot(output_path="resolved.yaml", **kw),
    ),
    ("RunTiming", RunTiming),
    ("ResourceUsage", ResourceUsage),
    ("EvaluationTiming", lambda **kw: EvaluationTiming(clock=FakeClock([]), **kw)),
    # The last two to arrive, and the only `StatefulCallback`s in the table.
    # They belong in it on exactly the same terms as the six above: a stateless
    # subscription group let them answer the run lifecycle without a string, so
    # there is no longer any sense in which they are a special case.
    ("Status", Status),
    ("ArtifactIndex", ArtifactIndex),
]


@pytest.mark.parametrize(("name", "build"), _MIGRATED, ids=[entry[0] for entry in _MIGRATED])
def test_a_migrated_callback_is_trigger_free_and_answers_no_run_level_string(
    name: str, build: Any
) -> None:
    """Double firing is structurally impossible, not merely unconfigured."""

    callback = build()
    assert not hasattr(callback, "triggers")
    for method in ("on_run_start", "on_run_end", "on_exception", "on_run_failed"):
        assert not hasattr(callback, method), f"{name}.{method} survived the migration"


def test_the_shared_snapshot_base_is_abstract() -> None:
    """The two snapshots share a subscription plan, not a usable callback.

    Rejected at construction rather than at save time, the same reason
    `StatefulCallback` and `tpen.training.events.TrainingPhase` reject their
    undeclared bases there.
    """

    from tpen.callback.snapshot import _RunStartSnapshot

    with pytest.raises(TypeError, match="abstract"):
        _RunStartSnapshot(output_path="config.yaml")


@pytest.mark.parametrize(("name", "build"), _MIGRATED, ids=[entry[0] for entry in _MIGRATED])
def test_a_stale_triggers_key_is_rejected_loudly(name: str, build: Any) -> None:
    """ADR-E002: a config may not name events, and silence here would be worse.

    ``triggers`` was a REQUIRED positional argument on three of these and a
    defaulted one on two more, so several shipped configs passed it. Left in
    ``**kwargs`` it would reach `_CallbackCore` and re-arm the legacy path
    alongside the typed one, firing every handler twice. It reaches a
    duplicate-argument ``TypeError`` instead.
    """

    del name
    with pytest.raises(TypeError, match="triggers"):
        build(triggers=["run_start"])
