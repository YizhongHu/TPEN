"""Record which parameters actually received gradients on the first update."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

from tpen.artifacts import RunContext
from tpen.events import DomainState, Occurrence, Subscription
from tpen.events import Event as TypedEvent
from tpen.training.events import UpdateCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import SubscriptionGroup


class ParameterActivity(StatefulCallback[TrainerState]):
    """Write a one-time, first-applied-update gradient inventory.

    Parameters
    ----------
    output_path : str or pathlib.Path
        Exclusive-create destination for the parameter inventory, following
        the run's artifact-manager conventions (for example
        ``${run.dir}/parameter_activity.json`` in a run config) rather than an
        arbitrary absolute path.

    Notes
    -----
    This observes exactly one batch: a zero or absent gradient here is not
    proof that a parameter is structurally disconnected from the objective,
    only that this particular batch did not move it. The inventory records,
    per named model parameter, its shape, ``requires_grad``, and a gradient
    state of ``"absent"``, ``"zero"``, or ``"nonzero"`` together with the
    gradient's L2 norm (``None`` when absent).

    A parameter is reported ``"absent"`` whenever its gradient tensor is
    ``None`` OR it is not one of the tensors the live optimizer's
    ``param_groups`` actually track (compared by object identity). The
    second condition exists independently of the gradient check itself: an
    optimizer holding a disjoint set of parameter tensors -- the exact
    foreign-carrier failure class fixed elsewhere for the training path
    itself (see pull request 537) -- would otherwise still show "real"
    gradients from autograd on the live model while the optimizer trains
    none of it. This callback never reads or mutates the optimizer's state
    beyond that membership check, and never modifies any gradient.

    It subscribes to `tpen.training.events.UpdateCompleted`, which the
    trainer emits only after an optimizer update has actually applied and
    returned (never for an iteration whose update method declined to step).
    So nothing is written for an iteration with no applied update, and the
    inventory is written exactly once, on the first `UpdateCompleted` this
    callback observes.

    Because ``output_path`` is opened with exclusive create, a resumed run
    whose output directory already carries a prior inventory at the same
    path raises ``FileExistsError`` rather than silently overwriting it; a
    resumed run that wants its own inventory must be configured with a
    distinct per-attempt path.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, output_path: str | Path) -> None:
        super().__init__(typed_groups=(
            SubscriptionGroup(selectors=(Subscription.of(UpdateCompleted),)),
        ))
        self.output_path = Path(output_path)
        self._written = False

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext, state: TrainerState
    ) -> None:
        """Write the first-applied-update gradient inventory, once."""
        del context
        if self._written:
            return
        if not isinstance(occurrence.event, UpdateCompleted):
            return
        model = state.model
        if model is None:
            raise ValueError("ParameterActivity requires state.model")
        tracked_ids = _tracked_parameter_ids(state.optimizer)
        rows = [
            _parameter_row(name, parameter, tracked=id(parameter) in tracked_ids)
            for name, parameter in model.named_parameters()
        ]
        trainer = state.trainer
        completed_updates = None if trainer is None else trainer.completed_updates
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("x", encoding="utf-8") as handle:
            json.dump(
                {
                    "completed_updates": completed_updates,
                    "interpretation": "observed_on_first_applied_update_not_structural_proof",
                    "parameters": rows,
                },
                handle, indent=2, allow_nan=False,
            )
            handle.write("\n")
        self._written = True

    def _reset_typed_state(self) -> None:
        self._written = False


def _tracked_parameter_ids(optimizer: Any) -> set[int]:
    """Return identities of every tensor the optimizer's param groups track."""
    ids: set[int] = set()
    if optimizer is None:
        return ids
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            ids.add(id(parameter))
    return ids


def _parameter_row(name: str, parameter: Any, *, tracked: bool) -> dict[str, Any]:
    """Classify one named parameter's gradient without mutating it."""
    gradient = parameter.grad
    if gradient is None or not tracked:
        status = "absent"
        norm = None
    else:
        gradient = gradient.detach()
        status = "nonzero" if bool(gradient.count_nonzero()) else "zero"
        norm = float(gradient.norm().item())
    return {
        "name": name,
        "shape": list(parameter.shape),
        "requires_grad": bool(parameter.requires_grad),
        "gradient": status,
        "gradient_norm": norm,
    }


__all__ = [
    "ParameterActivity",
]
