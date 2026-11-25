"""Timing callback for typed sampler burn-in and production scopes."""

from __future__ import annotations

import time
from typing import Any, Callable

from tpen.artifacts import RunContext
from tpen.events import Ended, Event as TypedEvent, Occurrence, Started, Subscription, ended, started
from tpen.sampling.events import SamplerBurnIn, SamplerProduction

from ..cadence import SubscriptionGroup
from .base import Callback, _occurrence_time, _sync_device


class SamplingScopeTiming(Callback):
    """Aggregate successful sampler scopes under typed train/eval parents."""

    def __init__(self, *, accelerator_synchronize: bool = False,
                 clock: Callable[[], float] | None = None, **kwargs: Any) -> None:
        from tpen.evaluation.events import EvaluationTaskRun
        from tpen.training.events import TrainingIteration, TrainingIterationCompleted

        super().__init__(typed_groups=(SubscriptionGroup(selectors=(
            started(TrainingIteration), ended(TrainingIteration),
            started(EvaluationTaskRun), ended(EvaluationTaskRun),
            started(SamplerBurnIn), ended(SamplerBurnIn),
            started(SamplerProduction), ended(SamplerProduction),
            Subscription.of(TrainingIterationCompleted),
        )),), **kwargs)
        self.accelerator_synchronize = bool(accelerator_synchronize)
        self.clock = time.perf_counter if clock is None else clock
        self._training_type = TrainingIteration
        self._training_completed_type = TrainingIterationCompleted
        self._evaluation_type = EvaluationTaskRun
        self._sampler_types = (SamplerBurnIn, SamplerProduction)
        # Parent scopes can nest (for example, an evaluation task inside a
        # training iteration).  Keep the full stack so ending the inner scope
        # restores the enclosing sampler attribution target.
        self._parents: list[tuple[str, int | str]] = []
        self._starts: dict[tuple[type[object], int], tuple[str, int | str, float]] = {}
        self._durations: dict[tuple[str, int | str], dict[str, float]] = {}

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        event = occurrence.event
        if isinstance(event, (Started, Ended)):
            operation = event.operation
            if isinstance(operation, self._training_type):
                if isinstance(event, Started):
                    self._parents.append(("train", operation.step))
                if isinstance(event, Ended):
                    self._clear_parent("train", operation.step)
                    self._pop_parent("train", operation.step)
                return
            if isinstance(operation, self._evaluation_type):
                if isinstance(event, Started):
                    self._parents.append(("eval", operation.name))
                else:
                    self._flush("eval", operation.name, context)
                    self._pop_parent("eval", operation.name)
                return
            if isinstance(operation, self._sampler_types):
                key = (type(operation), occurrence.count)
                if isinstance(event, Started):
                    parent_kind: str
                    parent: int | str
                    if not self._parents:
                        raise RuntimeError("sampler scope requires an active training or evaluation parent")
                    parent_kind, parent = self._parents[-1]
                    _sync_device(self.accelerator_synchronize)
                    self._starts[key] = (parent_kind, parent, _occurrence_time(occurrence, self.clock))
                else:
                    record = self._starts.pop(key, None)
                    if record is not None and event.succeeded:
                        kind, parent, start = record
                        _sync_device(self.accelerator_synchronize)
                        metric = operation.metric_fragment + "_time_sec"
                        values = self._durations.setdefault((kind, parent), {})
                        values[metric] = values.get(metric, 0.0) + _occurrence_time(occurrence, self.clock) - start
                return
        if isinstance(event, self._training_completed_type):
            self._flush("train", event.iteration.step, context)

    def _flush(self, kind: str, parent: int | str, context: RunContext) -> None:
        metrics = self._durations.pop((kind, parent), None)
        if metrics:
            namespace = "train/perf" if kind == "train" else f"eval/perf/{parent}"
            context.log(metrics, step=0 if kind == "eval" else int(parent), namespace=namespace)
        self._starts = {key: value for key, value in self._starts.items() if value[:2] != (kind, parent)}

    def _clear_parent(self, kind: str, parent: int | str) -> None:
        self._durations.pop((kind, parent), None)
        self._starts = {key: value for key, value in self._starts.items() if value[:2] != (kind, parent)}

    def _pop_parent(self, kind: str, parent: int | str) -> None:
        target = (kind, parent)
        for index in range(len(self._parents) - 1, -1, -1):
            if self._parents[index] == target:
                del self._parents[index]
                return

    def _reset_typed_state(self) -> None:
        self._parents.clear()
        self._starts.clear()
        self._durations.clear()


__all__ = ["SamplingScopeTiming"]
