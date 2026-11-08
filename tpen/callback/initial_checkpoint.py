"""An opt-in checkpoint of the actual initial training state."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from tpen.artifacts import RunContext
from tpen.checkpoint import TrainResume, save_checkpoint
from tpen.events import DomainState, Occurrence, Started, started
from tpen.events import Event as TypedEvent
from tpen.training.events import TrainingIteration
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import SubscriptionGroup


class InitialCheckpoint(StatefulCallback[TrainerState]):
    """Save boundary zero before sampling or updating the initial model.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Keep-all checkpoint directory shared with the periodic writer.

    Notes
    -----
    A resumed or non-zero lineage is refused: this callback is for fresh
    independent lineages only. It writes exactly once, through the checkpoint
    package's saver, with the full ``TrainResume`` payload (model, optimizer,
    trainer, sampler, and random state). It neither reconstructs the model nor
    draws any samples.

    It subscribes to ``started(TrainingIteration)``, which the trainer emits
    on entry to the per-iteration scope, strictly BEFORE the nested
    ``CollectSamples`` scope that draws the first batch of walkers (see
    ``tpen.training.trainer.Trainer.fit``). So the write this callback performs
    always precedes the first sample of a fresh run.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, output_dir: str | Path) -> None:
        super().__init__(typed_groups=(SubscriptionGroup(
            selectors=(started(TrainingIteration),),
        ),))
        self.output_dir = Path(output_dir)
        self._written = False

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext, state: TrainerState
    ) -> None:
        """Publish exactly the pre-sampling zero boundary."""
        if self._written:
            return
        event = occurrence.event
        if not isinstance(event, Started) or not isinstance(event.operation, TrainingIteration):
            return
        trainer = state.trainer
        if trainer is None:
            raise ValueError("InitialCheckpoint requires state.trainer")
        progress = trainer.state_dict()
        if event.operation.step != 0 or any(
            progress[key] != 0 for key in ("next_iteration", "completed_updates")
        ):
            raise ValueError("InitialCheckpoint requires a fresh zero-update lineage")
        # save_checkpoint refuses a non-equivalent existing publication, which
        # is exactly the collision policy this callback wants: an unrelated or
        # differently-shaped existing checkpoint at this path must fail loudly,
        # never be silently overwritten or silently treated as already done.
        save_checkpoint(
            output_dir=self.output_dir, next_iteration=0, completed_updates=0,
            model=state.model, optimizer=state.optimizer, trainer=state.trainer,
            sampler=state.sampler, context=context, payload=TrainResume(),
        )
        self._written = True

    def _reset_typed_state(self) -> None:
        self._written = False


__all__ = [
    "InitialCheckpoint",
]
