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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tpen.events import Event


RUN_OUTCOME_SCHEMA = "tpen.run-outcome/v1"
RUN_OUTCOME_STATUSES = frozenset({"completed", "planned_yield", "failed", "unknown", "refused"})
_CHECKPOINT_REF_KEYS = frozenset(
    {
        "schema",
        "checkpoint_dir",
        "schema_version",
        "kind",
        "next_iteration",
        "completed_updates",
        "manifest_sha256",
        "model_sha256",
        "provenance",
        "content_id",
    }
)


class RunOutcomeDecodeError(ValueError):
    """Raised when a durable S4 run outcome is not a closed valid record."""


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Decoded, typed contents of the S4 ``run-outcome.json`` artifact.

    Parameters
    ----------
    status : str
        Typed result status returned by the core runner.
    run_dir : str or None
        Producer-reported run directory, when available.
    error : str or None
        Producer-reported error text, when available.
    checkpoint_ref : CheckpointRef or None
        Full checkpoint identity named by the typed result.
    next_iteration, completed_updates, target_iterations : int or None
        Independent producer counters.
    exception_type, publication_stage : str or None
        Optional failure and publication diagnostics.
    schema : str or None, optional
        Explicit ``tpen.run-outcome/v1`` marker, or ``None`` for the legacy
        pre-marker record shape.
    """

    status: str
    run_dir: str | None
    error: str | None
    checkpoint_ref: Any | None
    next_iteration: int | None
    completed_updates: int | None
    target_iterations: int | None
    exception_type: str | None
    publication_stage: str | None
    schema: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunOutcome":
        """Decode one closed S4 outcome mapping and its checkpoint identity."""

        if not isinstance(data, Mapping):
            raise RunOutcomeDecodeError("run outcome must be a mapping")
        expected = {
            "schema",
            "status",
            "run_dir",
            "error",
            "checkpoint_ref",
            "next_iteration",
            "completed_updates",
            "target_iterations",
            "exception_type",
            "publication_stage",
        }
        legacy_expected = expected - {"schema"}
        data_keys = set(data)
        if (data_keys != expected and data_keys != legacy_expected) or (
            "schema" in data and data.get("schema") != RUN_OUTCOME_SCHEMA
        ):
            raise RunOutcomeDecodeError("run outcome has an invalid schema or key set")
        status = data["status"]
        if type(status) is not str or status not in RUN_OUTCOME_STATUSES:
            raise RunOutcomeDecodeError(f"unsupported run outcome status {status!r}")
        for key in ("run_dir", "error", "exception_type", "publication_stage"):
            value = data[key]
            if value is not None and type(value) is not str:
                raise RunOutcomeDecodeError(f"run outcome {key} must be a string or null")
        counters: dict[str, int | None] = {}
        for key in ("next_iteration", "completed_updates", "target_iterations"):
            value = data[key]
            if value is not None and (type(value) is not int or value < 0):
                raise RunOutcomeDecodeError(f"run outcome {key} must be a non-negative integer or null")
            counters[key] = value
        ref_data = data["checkpoint_ref"]
        ref = None
        if ref_data is not None:
            if not isinstance(ref_data, Mapping):
                raise RunOutcomeDecodeError("run outcome checkpoint_ref must be a mapping or null")
            if set(ref_data) != _CHECKPOINT_REF_KEYS:
                raise RunOutcomeDecodeError("run outcome checkpoint_ref has an invalid key set")
            try:
                from tpen.checkpoint import CheckpointRef

                ref = CheckpointRef.from_mapping(ref_data)
            except (KeyError, TypeError, ValueError) as exc:
                raise RunOutcomeDecodeError(f"invalid run outcome checkpoint_ref: {exc}") from exc
        return cls(
            status=status,
            run_dir=data["run_dir"],
            error=data["error"],
            checkpoint_ref=ref,
            next_iteration=counters["next_iteration"],
            completed_updates=counters["completed_updates"],
            target_iterations=counters["target_iterations"],
            exception_type=data["exception_type"],
            publication_stage=data["publication_stage"],
            schema=data.get("schema"),
        )


def read_run_outcome(path: str | Path) -> RunOutcome:
    """Read and decode one S4 ``run-outcome.json`` artifact.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the producer-written outcome artifact.

    Returns
    -------
    RunOutcome
        Strictly decoded typed result.

    Raises
    ------
    RunOutcomeDecodeError
        If the file cannot be read, parsed, or decoded as the closed schema.
    """

    outcome_path = Path(path)
    try:
        with outcome_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RunOutcomeDecodeError(f"cannot read {outcome_path}: {exc}") from exc
    return RunOutcome.from_mapping(data)


def has_explicit_run_outcome_schema(path: str | Path) -> bool:
    """Return whether a JSON outcome carries the current explicit schema marker.

    Invalid or unmarked files return ``False`` so inspection can preserve the
    historical v1/v2 path for pre-CX-01 artifacts.
    """

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, ValueError):
        return False
    return isinstance(data, Mapping) and data.get("schema") == RUN_OUTCOME_SCHEMA


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


__all__ = [
    "RUN_OUTCOME_SCHEMA",
    "RUN_OUTCOME_STATUSES",
    "RunCompleted",
    "RunFailed",
    "RunOutcome",
    "RunOutcomeDecodeError",
    "RunStarted",
    "has_explicit_run_outcome_schema",
    "read_run_outcome",
]
