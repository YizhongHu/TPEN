"""An opt-in checkpoint of the actual initial training state."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from tpen.artifacts import RunContext
from tpen.checkpoint import CheckpointRestored, TrainResume, save_checkpoint
from tpen.checkpoint.artifact import checkpoint_step_dir_name, is_complete_checkpoint_dir
from tpen.events import DomainState, Occurrence, Started, Subscription, started
from tpen.events import Event as TypedEvent
from tpen.training.events import TrainingIteration
from tpen.training.state import TrainerState

from .base import CallbackContext, StatefulCallback
from .cadence import SubscriptionGroup


class InitialCheckpoint(StatefulCallback[TrainerState]):
    """Save boundary zero before sampling or updating the initial model.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Keep-all checkpoint directory shared with the periodic writer.

    Notes
    -----
    A restored lineage whose first iteration still reads step zero is
    refused: this callback's own zero-update checkpoint is written by a
    fresh lineage only, never one that itself started from a restore (see
    ``CheckpointRestored`` handling below). A resumed lineage whose first
    iteration step is greater than zero instead becomes a VERIFIED no-op: it
    checks that ``step_000000`` already exists and is complete (the
    ``COMPLETE`` marker and manifest) in this same keep-all ``output_dir``,
    then writes nothing. A resumed lineage with no such checkpoint present is
    refused, never written late.

    On a fresh lineage it writes exactly once, through the checkpoint
    package's saver, with the full ``TrainResume`` payload (model, optimizer,
    trainer, sampler, and random state). It neither reconstructs the model nor
    draws any samples.

    It subscribes to ``started(TrainingIteration)``, which the trainer emits
    on entry to the per-iteration scope, strictly BEFORE the nested
    ``CollectSamples`` scope that draws the first batch of walkers (see
    ``tpen.training.trainer.Trainer.fit``). So the write this callback performs
    always precedes the first sample of a fresh run.

    The durable counters alone do not distinguish a fresh lineage from one
    restored from a checkpoint whose own counters happen to be zero (the
    exact lineage this callback itself produces, resumed into a new output
    directory): both read ``next_iteration == completed_updates == 0``. So
    this callback ALSO subscribes, through a second, stateless, class-owned
    group, to ``tpen.checkpoint.CheckpointRestored`` -- the restore report
    `tpen.runner.train.Train.run` emits strictly before it calls
    ``trainer.fit`` (and so strictly before any ``TrainingIteration`` starts).
    Observing one of those reports this run is a restored lineage, not a
    fresh one, regardless of what its restored counters say.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, output_dir: str | Path) -> None:
        super().__init__(typed_groups=(
            SubscriptionGroup(selectors=(started(TrainingIteration),)),
            SubscriptionGroup(selectors=(Subscription.of(CheckpointRestored),), stateless=True),
        ))
        self.output_dir = Path(output_dir)
        self._written = False
        self._restored = False

    def handle_stateless_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: CallbackContext
    ) -> None:
        """Record that this run's lineage was restored, not fresh."""
        del context
        if isinstance(occurrence.event, CheckpointRestored):
            self._restored = True

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext, state: TrainerState
    ) -> None:
        """Publish the pre-sampling zero boundary, or verify it on resume."""
        if self._written:
            return
        event = occurrence.event
        if not isinstance(event, Started) or not isinstance(event.operation, TrainingIteration):
            return
        trainer = state.trainer
        if trainer is None:
            raise ValueError("InitialCheckpoint requires state.trainer")
        progress = trainer.state_dict()
        fresh_counters = event.operation.step == 0 and all(
            progress[key] == 0 for key in ("next_iteration", "completed_updates")
        )
        if fresh_counters and not self._restored:
            # save_checkpoint refuses a non-equivalent existing publication,
            # which is exactly the collision policy this callback wants: an
            # unrelated or differently-shaped existing checkpoint at this path
            # must fail loudly, never be silently overwritten or silently
            # treated as already done.
            save_checkpoint(
                output_dir=self.output_dir, next_iteration=0, completed_updates=0,
                model=state.model, optimizer=state.optimizer, trainer=state.trainer,
                sampler=state.sampler, context=context, payload=TrainResume(),
            )
            self._written = True
            return
        if fresh_counters:
            # `self._restored`: zero counters, but a `CheckpointRestored`
            # report proves this lineage is not actually fresh.
            raise ValueError("InitialCheckpoint requires a fresh zero-update lineage")
        # Resumed lineage (nonzero step, or the non-standard nonzero-counter
        # state at step zero pinned by the unit tests below): never write an
        # update-zero checkpoint late. Only a VERIFIED prior publication in
        # this same output_dir makes this a no-op.
        zero_dir = self.output_dir / checkpoint_step_dir_name(0)
        if not is_complete_checkpoint_dir(zero_dir):
            raise ValueError(
                "InitialCheckpoint requires a fresh zero-update lineage or a "
                f"verified prior update-zero checkpoint at {zero_dir}; found neither"
            )
        self._written = True

    def _reset_typed_state(self) -> None:
        self._written = False
        self._restored = False


__all__ = [
    "InitialCheckpoint",
]
