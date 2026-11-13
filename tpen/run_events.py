"""Typed events for one configured run's own lifecycle.

Every other typed vocabulary sits beside the code that emits it:
`tpen.training.events`, `tpen.evaluation.events`, `tpen.checkpoint.events`. The
run lifecycle is the one part of the vocabulary that belongs to no domain, and
its emitter is `tpen.run.run_from_config`. It cannot live there: `tpen.run`
imports Hydra, `tpen.runner`, and `tpen.callback` itself, so a callback
importing its events back would be both circular and heavy, and it would drag
torch into `tpen.callback.timing`, which
``test_callback_timing_import_stays_torch_free`` forbids. This module therefore
imports nothing but `tpen.events`.

THE RUN HARNESS IS THE SOLE EMITTER of all three. That is a decision, not an
accident of where the code sits. `tpen.runner.Train` and `tpen.runner.Evaluate`
also emit the legacy ``run_start`` and ``run_end`` strings, and the ``run_start``
duplication is why `tpen.artifacts.RunContext` carries a ``_run_start_emitted``
flag and `tpen.runner.Runner.emit` carries a matching one: two emitters for one
moment, suppressed after the fact by name. A single emitter needs no such flag.
It also means `RunCompleted` fires for ANY configured runner rather than only the
two that remember to emit, and it puts `RunFailed` -- which only the harness can
observe, because the runners deliberately own no exception handling -- with the
other two boundaries of the same lifecycle instead of splitting one lifecycle
across two modules.

ALL THREE ARE POINT EVENTS, NEVER AN `Operation`. A run scope would have to
bracket ``runner.run``, and `tpen.artifacts.RunContext.scope` emits its ``Ended``
record from a ``finally``. Every one of the callbacks below would then run its
completion path on a CRASHED run: `tpen.callback.Metadata` would write
``status: completed`` over a failure, `tpen.callback.timing.RunTiming` would log
``runtime`` without ``failed: True`` and consume its own start timestamp so the
later failure boundary reported nothing, and
`tpen.callback.timing.EvaluationTiming` would stop writing ``eval/perf
{failed: True}`` altogether. This is the fourth time the ``finally`` has been
caught in this program (crashed training iterations, ``eval/perf/failed``, a
terminal checkpoint on an abort, and `TrainingCompleted` in PR #181); the answer
each time was a point event.
"""

from __future__ import annotations

from dataclasses import dataclass

from tpen.events import Event


@dataclass(frozen=True)
class RunStarted(Event):
    """The run context exists and the harness is about to build its runner.

    Emitted from `tpen.run.run_from_config` after `tpen.artifacts.RunContext` is
    constructed and the RNGs are seeded, at the same point as the legacy
    ``run_start`` string. Everything a subscriber needs at this moment --
    metadata, the source config, the resolved config -- is already on the
    context, so the event carries no fields (ADR-E003: no field without a
    present consumer).
    """


@dataclass(frozen=True)
class RunCompleted(Event):
    """The configured runner returned without raising.

    RAISE-FREE PATH ONLY, which is the whole reason this is not the ``Ended``
    boundary of a scope. See the module docstring.

    "Completed" here names the MOMENT, not the verdict: an evaluation suite
    whose tasks all failed still returns a `tpen.artifacts.RunResult` and still
    reaches this moment, exactly as it reached the legacy ``run_end`` string.
    That distinction is a status field on the result, and manufacturing a
    SECOND EVENT to carry it is what ADR-E007 forbids -- so the field rides
    THIS event rather than becoming a lifecycle moment of its own.

    Attributes
    ----------
    status : str
        `tpen.artifacts.RunResult.status` verbatim as the runner returned it:
        ``"completed"``, or ``"failed"`` for a suite that aggregated to failed.

        The two durable writers, `tpen.callback.Status` and
        `tpen.callback.Metadata`, record this instead of hardcoding success.
        Both used to write ``completed`` at this boundary unconditionally, so a
        failed evaluation left ``status.json`` and ``metadata.json`` claiming a
        success that `tpen.artifacts.RunResult` had already contradicted --
        and ``experiments/toolkit/task_state.py`` banks a row as done on
        exactly that ``completed``.

        Defaults to ``"completed"``: a runner that returns anything other than
        a `RunResult` reports no status to contradict the moment, and every
        emitter of this event outside `tpen.run.run_from_config` is a test
        driving the boundary itself.
    """

    status: str = "completed"
    result: object | None = None


@dataclass(frozen=True)
class RunFailed(Event):
    """The run raised out of the harness and is being reported as failed.

    ONE EVENT REPLACES TWO STRINGS. `tpen.run.run_from_config` emitted
    ``run_failed`` and then ``exception`` on consecutive lines with the SAME
    payload, and the consumer census found nothing that distinguished them:
    `tpen.callback.timing.RunTiming.on_run_failed` and
    `tpen.callback.ResourceUsage.on_run_failed` were documented as aliases and
    called the identical private method as their ``on_exception`` twin, while
    `tpen.callback.Metadata`, `tpen.callback.Status`, and
    `tpen.callback.timing.EvaluationTiming` answered only ``exception``. Two
    names for one moment, and for any callback subscribed to both -- which was
    the shipped default for `RunTiming` and `ResourceUsage`, and the configured
    setting for `ResourceUsage` in ``pair_stability.yaml`` -- the same work ran
    twice, writing ``runtime`` twice per failure. Collapsing them makes that
    duplication structurally impossible.

    The two fields are what the two artifact writers actually read, and no more.
    `tpen.run` already computed both into the legacy payload, so this moves
    values rather than inventing them; ``phase`` and ``traceback`` travel in the
    same payload but have no event consumer -- they reach ``error.json`` through
    `tpen.artifacts.write_error_artifact` directly -- so they are not fields
    here (ADR-E003).

    Carrying the failure's identity is not the payload-dict pattern ADR-E007
    rejects. That rule bars an event minted, or a field added, to ferry AMBIENT
    DOMAIN STATE to a subscriber; ADR-E007's own test is whether the event would
    be worth emitting if nobody needed its data, and a run failing is a moment
    worth recording regardless. What failed is constitutive of that moment, the
    same way `tpen.evaluation.events.ComponentFailed` carries its
    `tpen.evaluation.results.EvaluationFailure` and
    `tpen.checkpoint.events.CheckpointRestored` carries its report.

    Strings rather than the live exception object, for two reasons. They are
    exactly what both consumers use (``type(exc).__name__`` and ``str(exc)``),
    and a frozen event field holding a live traceback-bearing exception would
    keep every frame's locals alive for the rest of the process. They also
    serialize: `tpen.artifacts.write_occurrence_artifact` records typed fields
    into ``occurrences.jsonl``, where a bare ``BaseException`` would collapse to
    a type marker and the message would be silently dropped.

    Parameters
    ----------
    exception_type : str
        Class name of the raised exception.
    exception_message : str
        ``str()`` of the raised exception.
    """

    exception_type: str
    exception_message: str


__all__ = ["RunCompleted", "RunFailed", "RunStarted"]
