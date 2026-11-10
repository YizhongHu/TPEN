"""Record which parameters actually received gradients on the first update."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, ClassVar

from tpen.artifacts import RunContext
from tpen.checkpoint import CheckpointRestored
from tpen.events import DomainState, Occurrence, Subscription
from tpen.events import Event as TypedEvent
from tpen.training.events import UpdateCompleted
from tpen.training.state import TrainerState

from .base import CallbackContext, StatefulCallback
from .cadence import SubscriptionGroup

_INVENTORY_INTERPRETATION = "observed_on_first_applied_update_not_structural_proof"
_INVENTORY_KEYS = {"completed_updates", "interpretation", "parameters"}
_PARAMETER_ROW_KEYS = {"name", "shape", "requires_grad", "gradient", "gradient_norm"}
_GRADIENT_STATUSES = {"absent", "zero", "nonzero"}


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

    A restored lineage whose first applied update still reads step zero is
    refused, exactly as ``InitialCheckpoint`` refuses the same case: this
    callback's own first-update write happens on a fresh lineage only. A
    resumed lineage whose first applied update reads a nonzero step instead
    becomes a VERIFIED no-op: it checks that ``output_path`` already holds a
    complete, well-formed inventory (the expected top-level keys, the
    expected interpretation string, and a nonempty list of per-parameter rows
    each carrying the expected row keys), and records nothing -- never a
    later update's gradients mislabeled as the first one. A resumed lineage
    with no such inventory present, at this path or a distinct per-attempt
    one, is refused, never written late. 'Verified' means artifact
    integrity, the same standard ``InitialCheckpoint`` holds with its
    ``COMPLETE`` marker and manifest, not recomputed gradient values: this
    callback never recomputes or compares against the live model's current
    gradients, which legitimately differ update to update.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(self, output_path: str | Path) -> None:
        super().__init__(typed_groups=(
            SubscriptionGroup(selectors=(Subscription.of(UpdateCompleted),)),
            SubscriptionGroup(selectors=(Subscription.of(CheckpointRestored),), stateless=True),
        ))
        self.output_path = Path(output_path)
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
        """Write the first-applied-update gradient inventory, or verify it on resume."""
        del context
        if self._written:
            return
        event = occurrence.event
        if not isinstance(event, UpdateCompleted):
            return
        if not self._restored:
            # Origin comes from the `CheckpointRestored` report alone, never
            # from the step of this first applied update: a fresh lineage can
            # skip iteration 0 entirely (no update applies there) and first
            # apply an update at a nonzero step, which must still record as
            # the lineage's first inventory.
            self._write_fresh_inventory(state)
            self._written = True
            return
        if event.iteration.step == 0:
            # Restored lineage whose first applied update still reads step
            # zero: indistinguishable from fresh by step alone, so the
            # `CheckpointRestored` report is what refuses it.
            raise ValueError("ParameterActivity requires a fresh first-applied-update lineage")
        # Resumed lineage (nonzero step): never record a later update's
        # gradients as though they were the first. Only a VERIFIED prior
        # inventory at `output_path` makes this a no-op.
        if not _is_verified_inventory(self.output_path):
            raise ValueError(
                "ParameterActivity requires a fresh first-applied-update lineage or a "
                f"verified prior inventory at {self.output_path}; found neither"
            )
        self._written = True

    def _write_fresh_inventory(self, state: TrainerState) -> None:
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
        # Serialize the whole payload before touching the destination: a
        # non-finite gradient norm makes `allow_nan=False` raise, and doing
        # this before the exclusive-create open means a rejected payload
        # never leaves a truncated, unparseable file claiming the path.
        payload = json.dumps(
            {
                "completed_updates": completed_updates,
                "interpretation": _INVENTORY_INTERPRETATION,
                "parameters": rows,
            },
            indent=2, allow_nan=False,
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")

    def _reset_typed_state(self) -> None:
        self._written = False
        self._restored = False


def _is_verified_inventory(path: Path) -> bool:
    """Return whether `path` holds a complete, well-formed parameter inventory.

    Structural and value verification: the expected top-level keys, the
    expected interpretation string, a nonempty list of per-parameter rows
    each carrying the expected row keys with well-typed values, a finite or
    null gradient norm, a recognized gradient status, and a valid counter.
    This never recomputes or compares gradient VALUES -- they legitimately
    differ update to update, so a resumed lineage's "verified" artifact is
    one that parses and has the right shape and types, not one whose
    numbers match anything current.
    """

    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or set(payload) != _INVENTORY_KEYS:
        return False
    if payload["interpretation"] != _INVENTORY_INTERPRETATION:
        return False
    if not _is_valid_counter(payload["completed_updates"]):
        return False
    rows = payload["parameters"]
    if not isinstance(rows, list) or not rows:
        return False
    return all(_is_verified_row(row) for row in rows)


def _is_valid_counter(value: Any) -> bool:
    """Return whether `value` is a valid counter: null, or a non-negative int."""

    if value is None:
        return True
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_verified_row(row: Any) -> bool:
    """Return whether one parameter row has the expected keys and well-typed values."""

    if not isinstance(row, dict) or set(row) != _PARAMETER_ROW_KEYS:
        return False
    shape = row["shape"]
    if not isinstance(shape, list) or not all(
        isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape
    ):
        return False
    if not isinstance(row["requires_grad"], bool):
        return False
    if row["gradient"] not in _GRADIENT_STATUSES:
        return False
    return _is_finite_or_null(row["gradient_norm"])


def _is_finite_or_null(value: Any) -> bool:
    """Return whether `value` is `None`, or a finite (non-bool) real number."""

    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


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
