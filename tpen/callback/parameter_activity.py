"""Record which parameters received gradients on the first applied update."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from tpen.artifacts import RunContext
from tpen.callback.base import StatefulCallback
from tpen.callback.cadence import SubscriptionGroup
from tpen.events import DomainState, Event, Occurrence, Subscription
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState


class ParameterActivity(StatefulCallback[TrainerState]):
    """Write a first-update gradient inventory without changing any gradients.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Exclusive-create destination for the parameter inventory.

    Notes
    -----
    A zero gradient on this batch is not proof of structural inactivity. The
    receipt distinguishes absent, zero, and nonzero gradients explicitly.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, output_path: str | Path) -> None:
        super().__init__(typed_groups=(SubscriptionGroup(
            selectors=(Subscription.of(TrainingIterationCompleted),),
        ),))
        self.output_path = Path(output_path)
        self._written = False

    def handle_occurrence_impl(
        self, occurrence: Occurrence[Event], context: RunContext, state: TrainerState
    ) -> None:
        """Observe the still-live gradients after the first completed update."""
        if self._written or not state.optimizer_step:
            return
        rows = []
        for name, parameter in state.model.named_parameters():
            gradient = parameter.grad
            rows.append({"name": name, "shape": list(parameter.shape),
                         "requires_grad": parameter.requires_grad,
                         "gradient": "absent" if gradient is None else "nonzero" if bool(gradient.count_nonzero()) else "zero",
                         "gradient_norm": None if gradient is None else float(gradient.detach().norm().item())})
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("x") as handle:
            json.dump({"completed_updates": state.trainer.completed_updates,
                       "interpretation": "observed_on_first_applied_update_not_structural_proof",
                       "parameters": rows}, handle, indent=2, allow_nan=False)
            handle.write("\n")
        self._written = True

    def _reset_typed_state(self) -> None:
        self._written = False
