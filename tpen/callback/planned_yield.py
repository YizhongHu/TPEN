"""Runtime callback for a published-boundary planned yield."""

from __future__ import annotations

from typing import Any, ClassVar

from tpen.artifacts import RunContext, RunResult
from tpen.checkpoint.reference import CheckpointRef
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import SubscriptionGroup


class PlannedYieldSignal(RuntimeError):
    """Internal control signal carrying an already classified run outcome."""

    def __init__(self, result: RunResult) -> None:
        super().__init__(result.status)
        self.result = result


class PlannedYield(StatefulCallback[TrainerState]):
    """Stop at one published, full-state checkpoint boundary below H."""

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, target_iteration: int, target_iterations: int, *, checkpoint: Any = None) -> None:
        if type(target_iteration) is not int or target_iteration < 0:
            raise ValueError("target_iteration must be a non-negative integer")
        if type(target_iterations) is not int or target_iterations < 0:
            raise ValueError("target_iterations must be a non-negative integer")
        self.target_iteration = target_iteration
        self.target_iterations = target_iterations
        self.checkpoint = checkpoint
        self.observed = False
        self._published_context: object = None
        self._published_ref: CheckpointRef | None = None
        self._published_profile: str | None = None
        super().__init__(
            typed_groups=(
                SubscriptionGroup(selectors=(Subscription.of(TrainingIterationCompleted),)),
            )
        )

    def handle_occurrence_impl(self, occurrence: Occurrence[TypedEvent], context: RunContext, state: TrainerState) -> None:
        """Classify only the requested durable training boundary."""

        next_iteration = int(getattr(state.trainer, "next_iteration"))
        if next_iteration != self.target_iteration:
            return
        self.observed = True
        callback = self.checkpoint or next(
            (candidate for candidate in context.callbacks if hasattr(candidate, "published_ref")),
            None,
        )
        ref = getattr(callback, "published_ref", None)
        profile = getattr(callback, "published_payload_profile", None)
        completed_updates = int(getattr(state.trainer, "completed_updates"))
        if self._published_context is not context:
            self._published_context = context
            self._published_ref = None
            self._published_profile = None
        if isinstance(ref, CheckpointRef) and profile == "train_resume":
            self._published_ref = ref
            self._published_profile = profile
        ref = getattr(self, "_published_ref", None)
        profile = getattr(self, "_published_profile", None)
        if (
            not isinstance(ref, CheckpointRef)
            or profile != "train_resume"
            or ref.next_iteration != next_iteration
            or ref.completed_updates != completed_updates
        ):
            raise PlannedYieldSignal(RunResult(
                status="unknown", error="planned yield boundary lacks a published full-state checkpoint",
                next_iteration=next_iteration, completed_updates=completed_updates,
                target_iterations=self.target_iterations,
            ))
        if next_iteration >= self.target_iterations:
            raise PlannedYieldSignal(RunResult(
                status="refused", checkpoint_ref=ref, error="planned yield must be below the fixed horizon H",
                next_iteration=next_iteration, completed_updates=completed_updates,
                target_iterations=self.target_iterations,
            ))
        raise PlannedYieldSignal(RunResult(
            status="planned_yield", checkpoint_ref=ref, next_iteration=next_iteration,
            completed_updates=completed_updates, target_iterations=self.target_iterations,
        ))

    def validate_target(self, runner: object) -> None:
        """Refuse a target that is not below the live trainer horizon."""
        live_h = getattr(getattr(runner, "trainer", None), "max_steps", None)
        binding = getattr(runner, "restore_binding", None)
        bound_h = getattr(binding, "target_iterations", None)
        if live_h != self.target_iterations or (bound_h is not None and bound_h != self.target_iterations):
            raise PlannedYieldSignal(RunResult(
                status="refused",
                error=("planned yield target does not match the live trainer "
                       "max_steps or restore binding"),
                target_iterations=self.target_iterations,
            ))
        if self.target_iteration >= live_h:
            raise PlannedYieldSignal(RunResult(
                status="refused",
                error="planned yield must be below the fixed horizon H",
                target_iterations=self.target_iterations,
            ))

    def _reset_typed_state(self) -> None:
        self._published_context = None
        self._published_ref = None
        self._published_profile = None
        self.observed = False


__all__ = ["PlannedYield", "PlannedYieldSignal"]
