"""Callbacks for evaluation task artifacts and failures.

Importing this module resolves `tpen.evaluation`, and therefore torch, because
`ArtifactIndex` declares its ``state_type`` as a class fact. `tpen.callback`
loads it lazily for that reason; the timing callbacks stay torch-free by
resolving their event types inside ``__init__`` instead, which this module
cannot do for a ClassVar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from tpen.artifacts import RunContext
from tpen.durable_append import append_record
from tpen.evaluation.events import ComponentFailed, EvaluationTaskRun
from tpen.evaluation.state import EvaluationRunState
from tpen.events import DomainState, Ended
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription, ended
from tpen.run_events import RunCompleted

from .base import Callback, StatefulCallback
from .cadence import SubscriptionGroup


class ArtifactIndex(StatefulCallback[EvaluationRunState]):
    """Maintain a compact index of evaluation task artifacts.

    Observes the end of every `tpen.evaluation.events.EvaluationTaskRun` scope
    and reads the finished task straight off the evaluation domain's state
    object (ADR-E008). What it used to do instead is the shape ADR-E007 rejects:
    `tpen.evaluation.results.TaskResult` was flattened into a mapping by
    ``to_payload()`` and re-parsed here with five ``.get()`` probes and a
    ``str(payload.get("namespace", payload.get("name", "")))`` fallback chain --
    defending against a shape the frozen result structurally cannot produce.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Index destination. Defaults to ``<run_dir>/diagnostics/index.json``.
    """

    state_type: ClassVar[type[DomainState]] = EvaluationRunState

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            typed_groups=(
                SubscriptionGroup(selectors=(ended(EvaluationTaskRun),)),
                # The run boundary is load-bearing exactly once: `_write`
                # already runs after every task, so on any run with at least one
                # task this group is byte-identical to not having it, but on an
                # EMPTY suite it is the only thing that writes the
                # ``{"tasks": []}`` index at all. Deleting a durable artifact is
                # not a refactor (ADR-E006).
                #
                # `RunCompleted` is the faithful replacement for the ``run_end``
                # string this used to answer, MEASURED rather than assumed. That
                # string was emitted by the runners as the last statement of
                # ``Train.run`` and ``Evaluate.run`` before their ``return``,
                # never from a ``finally``, so it fired on the success path only;
                # `RunCompleted` is emitted by the harness immediately after
                # ``runner.run`` returns, with nothing between the two points. A
                # group selecting `RunFailed` too would therefore write an index
                # on crashed runs where none was written before, which is why
                # this selects one event and `tpen.callback.Status` -- which DID
                # answer the failure boundary, under a different string -- selects
                # three.
                SubscriptionGroup(
                    selectors=(Subscription.of(RunCompleted),), stateless=True
                ),
            ),
            **kwargs,
        )
        self.path = None if path is None else Path(path)
        self._task_run_type = EvaluationTaskRun
        self._tasks: dict[str, dict[str, Any]] = {}

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: EvaluationRunState,
    ) -> None:
        """Record the task that just finished and rewrite the index."""

        event = occurrence.event
        if not isinstance(event, Ended) or not isinstance(
            event.operation, self._task_run_type
        ):
            return
        result = state.task_result
        if result is None:
            # `scope` fires ``Ended`` from a ``finally``, so this boundary is
            # also reached when the evaluator raises out of the task body -- and
            # then nothing was ever written to the state. The legacy path emitted
            # neither ``task_end`` nor ``task_failed`` there, so no index entry
            # was recorded either. Preserve that rather than inventing one.
            return
        # Named, typed field access on a frozen result: no probing, no fallback.
        self._tasks[result.namespace] = {
            "name": result.name,
            "namespace": result.namespace,
            "output_dir": str(result.output_dir),
            "status": result.status,
            "artifacts": [artifact.to_dict() for artifact in result.artifacts],
        }
        self._write(context)

    def handle_stateless_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Flush the artifact index at the end of a run that completed."""

        if not isinstance(occurrence.event, RunCompleted):
            return
        self._write(context)

    def _write(self, context: RunContext) -> None:
        path = self._path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tasks": list(self._tasks.values()),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _path(self, context: RunContext) -> Path:
        if self.path is not None:
            return self.path
        return Path(context.run_dir) / "diagnostics" / "index.json"


class FailureLog(Callback):
    """Append structured evaluation failures to ``diagnostics/failures.jsonl``.

    Reads `tpen.evaluation.events.ComponentFailed`, whose ``failure`` field is
    the `tpen.evaluation.results.EvaluationFailure` itself. The legacy path
    flattened that same object with ``to_dict()`` into an event payload and this
    callback re-parsed it back out behind an ``isinstance(dict)`` guard; the
    written line is unchanged, because it was always ``to_dict()`` either way.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Log destination. Defaults to ``<run_dir>/diagnostics/failures.jsonl``.
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        # One event covers a raising generator, calculator, or summary and a
        # summary whose bundle dependencies are absent, so the four separate
        # ``<kind>_failed`` triggers collapse into a single selector. There is
        # deliberately no task-level subscription: the task's own result repeats
        # these same failures, and observing both would write each one twice.
        super().__init__(
            typed_groups=(
                SubscriptionGroup(selectors=(Subscription.of(ComponentFailed),)),
            ),
            **kwargs,
        )
        self.path = None if path is None else Path(path)
        self._component_failed_type = ComponentFailed

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Append one component failure to the log."""

        event = occurrence.event
        if not isinstance(event, self._component_failed_type):
            return
        self._append(context, event.failure.to_dict())

    def _append(self, context: RunContext, failure: dict[str, Any]) -> None:
        append_record(self._path(context), json.dumps(failure, sort_keys=True))

    def _path(self, context: RunContext) -> Path:
        if self.path is not None:
            return self.path
        return Path(context.run_dir) / "diagnostics" / "failures.jsonl"


__all__ = ["ArtifactIndex", "FailureLog"]
