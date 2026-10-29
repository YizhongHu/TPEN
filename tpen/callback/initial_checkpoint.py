"""An opt-in checkpoint of the actual initial training state."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from tpen.artifacts import RunContext
from tpen.callback.base import StatefulCallback
from tpen.callback.cadence import SubscriptionGroup
from tpen.checkpoint import TrainResume, save_checkpoint
from tpen.events import DomainState, Event, Occurrence, Started, started
from tpen.training.events import TrainingIteration
from tpen.training.state import TrainerState


class InitialCheckpoint(StatefulCallback[TrainerState]):
    """Save boundary zero before sampling or updating the initial model.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Keep-all checkpoint directory shared with the periodic writer.

    Notes
    -----
    A resumed run is refused: this callback is for fresh independent lineages.
    It uses the checkpoint owner's saver, including optimizer, sampler and
    random state. It neither reconstructs the model nor draws any samples.
    Zero-iteration runs have no training-iteration boundary and are outside
    this callback's contract.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, output_dir: str | Path) -> None:
        super().__init__(typed_groups=(SubscriptionGroup(
            selectors=(started(TrainingIteration),),
        ),))
        self.output_dir = Path(output_dir)
        self._written = False

    def handle_occurrence_impl(
        self, occurrence: Occurrence[Event], context: RunContext, state: TrainerState
    ) -> None:
        """Publish exactly the pre-sampling zero boundary."""
        if self._written:
            return
        event = occurrence.event
        if not isinstance(event, Started) or not isinstance(event.operation, TrainingIteration):
            return
        progress = state.trainer.state_dict()
        if event.operation.step != 0 or any(
            progress[key] != 0 for key in ("next_iteration", "completed_updates")
        ):
            raise ValueError("InitialCheckpoint requires a fresh zero-update lineage")
        # The saver fails on collisions; an old artifact is not an initial-state receipt.
        save_checkpoint(
            output_dir=self.output_dir, next_iteration=0, completed_updates=0,
            model=state.model, optimizer=state.optimizer, trainer=state.trainer,
            sampler=state.sampler, context=context, payload=TrainResume(),
        )
        self._written = True

    def _reset_typed_state(self) -> None:
        self._written = False
