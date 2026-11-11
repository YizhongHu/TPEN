"""Training checkpoint callback.

Checkpoint artifact format, hashing, and restore behavior are owned by
``tpen.checkpoint``. This callback is only the lifecycle adapter that receives
runner state through typed events and asks the package-owned saver to write a
directory checkpoint.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from tpen.artifacts import RunContext
from tpen.checkpoint import (
    CheckpointPayload,
    CheckpointSchedule,
    ModelOnly,
    TrainResume,
    checkpoint_step_dir_name,
    reconcile_publication,
    save_checkpoint,
)
from tpen.checkpoint.artifact import is_complete_checkpoint_dir
from tpen.checkpoint.schema import read_manifest
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.training.events import (
    TrainingCompleted,
    TrainingIterationCompleted,
    UpdateCompleted,
)
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence

_LOGGER = logging.getLogger("tpen")


class Checkpoint(StatefulCallback[TrainerState]):
    """Write directory checkpoints from typed training state.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory into which checkpoints are written.
    periodic : bool, optional
        Write a checkpoint for each cadence-eligible completed iteration.
    terminal : bool, optional
        Write one checkpoint when the training loop finishes.
    schedule : CheckpointSchedule or None, optional
        Semantic periodic schedule keyed on durable ``completed_updates``.
        Every configured schedule also admits the terminal boundary; terminal
        publication is never cadence-gated.
    payload : CheckpointPayload or None, optional
        Immutable contract for the files and restore intents this stream
        publishes.  When omitted, the historical all-components payload is
        preserved unless the legacy save flags select another component set.
    keep_last : int or None, optional
        Compatibility input accepted and ignored. TPEN retains all committed
        checkpoints.
    save_optimizer, save_trainer, save_sampler, save_rng : bool or None, optional
        Whether to include train-resume state components.  ``None`` lets an
        explicit payload choose; with no payload, the historical all-component
        default is preserved.
    **kwargs
        Forwarded to `StatefulCallback` (e.g. legacy ``every_n_steps``).

    Notes
    -----
    **Which typed events the two writes land on, and why it matters.**

    The periodic write fires at `tpen.training.events.TrainingIterationCompleted`
    -- the SAME boundary the four ``fail_fast`` health checks fire on -- and not
    at `tpen.training.events.UpdateCompleted`, which reads more naturally
    ("an update completed, persist it") and is therefore the mistake this note
    exists to prevent. ``UpdateCompleted`` is emitted immediately after
    ``optimizer.step()`` returns, i.e. BEFORE any health check can run, so a
    ``fail_fast`` abort would persist the very model version that failed its own
    check and advance ``latest.json`` to it. Sharing the health checks' boundary
    restores the property that a failed check leaves no checkpoint for its step,
    because dispatch within one occurrence is the callback-list loop and an
    exception propagates out of it before later callbacks run.

    That property is pinned by a behavioural test rather than by this comment or
    by callback order (see ``tests/unit/training/test_health_checkpoint_order.py``),
    because ordering alone is an undeclared contract that a config edit would
    break silently.

    ``UpdateCompleted`` is still subscribed, but only to ARM the periodic write:
    it records which iteration applied an optimizer update, and the write at
    ``TrainingIterationCompleted`` is skipped for any iteration that did not.
    A vacuum iteration opens no optimizer scope and emits ``UpdateSkipped``,
    which this callback does not subscribe to, so it arms nothing. Selection is
    unconditional: with no ``every_n_steps`` configured, the periodic path
    writes on every completed update, not on every attempted iteration.

    The terminal write fires at `tpen.training.events.TrainingCompleted`, which
    the runner emits after the loop returns. Update selection deliberately does
    not apply to it: a run whose final iteration skipped its update, and a run
    whose loop body never executed at all, must both still write a terminal
    checkpoint.

    **The two progress counters, and which job each has.**

    Both are required keys of ``trainer.state_dict()`` and are read together on
    every path, so a trainer that cannot report its resume cursor fails loudly
    rather than mislabelling a directory. Neither is ever ``TrainerState.step``,
    whose value is ``-1`` before the first iteration and would render a
    directory named ``step_-00001``.

    ``next_iteration`` is *identity*: it names the checkpoint directory, so a
    restored run continues from exactly the iteration the name encodes. A run
    with ``max_steps=500`` writes its terminal checkpoint as ``step_000500``,
    while training metrics and other step callbacks keep their existing 0-based
    loop step indices.

    ``completed_updates`` is the periodic path's *cadence* coordinate:
    ``every_n_steps`` counts applied optimizer updates, so periodic checkpoints
    are spaced by real training progress rather than by attempted iterations.

    The split is deliberate and both halves are durable, so a resumed run
    checkpoints at the same points an uninterrupted one does. Neither is the
    run-local `tpen.events.Occurrence.count`, which restarts at 1 after a
    restore and would silently shift a resumed run's checkpoint phase.

    **Keep-all compatibility warning.** A non-``None`` ``keep_last`` emits one
    warning when this callback is constructed. Multiple writes through that
    callback emit no further warning; reconstructing the callback mid-run emits
    another warning. This is a construction-scoped diagnostic, not a
    process-global once-guard. The current launcher constructs one callback per
    ``python -u run.py`` process, so it currently amounts to once per run and
    once per scan cell; no current execution path reconstructs it mid-run.
    """

    # ClassVar: the runtime authority for typed state delivery.
    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(
        self,
        output_dir: str | Path,
        *,
        schedule: CheckpointSchedule | None = None,
        payload: CheckpointPayload | None = None,
        periodic: bool = True,
        terminal: bool = True,
        keep_last: int | None = None,
        save_optimizer: bool | None = None,
        save_trainer: bool | None = None,
        save_sampler: bool | None = None,
        save_rng: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if not periodic and not terminal:
            raise ValueError(
                "Checkpoint writes nothing with periodic=False and terminal=False"
            )
        cadence = pop_step_cadence(kwargs)
        # Subscriptions are class-owned under ADR-E002 -- no config names an
        # event -- but WHICH of this class's two writes an instance performs is
        # a semantic option, which is the alternative that ADR names for a
        # callback with genuinely different policy modes. It replaces the
        # former string selection of step completion / training completion that
        # configs used to spell, one flag per former trigger.
        selectors: tuple[Subscription, ...] = ()
        if periodic:
            selectors += (
                Subscription.of(UpdateCompleted),
                Subscription.of(TrainingIterationCompleted),
            )
        if terminal:
            selectors += (Subscription.of(TrainingCompleted),)
        super().__init__(
            # One group holding every selector, so `validate_subscription_groups`
            # has nothing to overlap. `cadence=None` is deliberate: a group
            # `Cadence` gates on the run-local `Occurrence.count`, which restarts
            # after a restore, so the window is applied to a durable counter
            # below instead.
            typed_groups=(SubscriptionGroup(selectors=selectors),),
            **kwargs,
        )
        self.output_dir = Path(output_dir)
        # A supplied semantic schedule owns periodic boundary selection.  The
        # legacy scalar gate remains for existing configs until the composed
        # callback migration, but both paths use durable completed_updates.
        self.schedule = schedule
        self.payload = payload
        self.periodic = bool(periodic)
        self.terminal = bool(terminal)
        self.keep_last = keep_last
        if keep_last is not None:
            _LOGGER.warning(
                "keep_last=%r is ignored; TPEN retains all checkpoints; "
                "checkpoint storage grows without bound; see the keep-all "
                "checkpoint policy in tpen/metrics_naming.md.",
                keep_last,
            )
        self.save_optimizer = save_optimizer
        self.save_trainer = save_trainer
        self.save_sampler = save_sampler
        self.save_rng = save_rng
        # The legacy gate applies only to periodic writes. Terminal publication
        # is an explicit semantic boundary and never consults this gate.
        self._steps = StepCadenceGate(cadence)
        # Trainer step of the most recent iteration that applied an optimizer
        # update. Matched against the step the completed-iteration event carries,
        # so it is self-clearing: a later iteration that skipped its update
        # simply never matches, and no explicit per-iteration reset is required.
        self._updated_iteration_step: int | None = None

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Arm, write periodically, or write terminally, per typed boundary."""

        event = occurrence.event
        if isinstance(event, UpdateCompleted):
            # `occurrence.count` is deliberately unused: it is run-local and
            # restarts after a restore. The durable coordinates are read from
            # the trainer when a write is actually considered.
            self._updated_iteration_step = int(event.iteration.step)
            return
        if isinstance(event, TrainingIterationCompleted):
            self._write_periodic(int(event.iteration.step), context, state)
            return
        if isinstance(event, TrainingCompleted):
            self._write_terminal(context, state)

    def _write_periodic(self, step: int, context: RunContext, state: TrainerState) -> None:
        """Write this iteration's checkpoint if it is eligible and in cadence.

        The cadence window is consulted BEFORE update selection, reproducing the
        legacy path's order, where the window lived in ``should_run`` and
        selection in the handler it gated. That order is what ``max_calls``
        counts, so swapping the two would change when a bounded callback stops.

        Parameters
        ----------
        step : int
            Durable zero-based trainer step, read from the typed event. Never
            ``state.step``, whose value fields are stale above their assignment.
        """

        next_iteration, completed_updates = _trainer_progress(state)
        if self.schedule is not None:
            should_run = self.schedule.should_run(completed_updates)
        else:
            should_run = self._steps.should_run(completed_updates)
        if not should_run:
            return
        if self._updated_iteration_step != step:
            return
        self._save(context, state, next_iteration, completed_updates)

    def _write_terminal(self, context: RunContext, state: TrainerState) -> None:
        """Write the final checkpoint for the run's resume cursor."""

        next_iteration, completed_updates = _trainer_progress(state)
        if self.schedule is not None and not self.schedule.should_run(
            completed_updates, terminal=True
        ):
            schedule_type = type(self.schedule).__name__
            raise RuntimeError(
                f"{schedule_type} violated CheckpointSchedule: terminal publication "
                "must not be suppressed by cadence (should_run(..., terminal=True) "
                "returned False)"
            )
        # Terminal publication is its own semantic boundary. It must remain
        # observable when periodic cadence misses. The durable completed-update
        # count is passed as metadata, while the terminal decision is explicit.
        self._save(context, state, next_iteration, completed_updates)

    def _save(
        self,
        context: RunContext,
        state: TrainerState,
        next_iteration: int,
        completed_updates: int,
    ) -> None:
        final_dir = self.output_dir / checkpoint_step_dir_name(next_iteration)
        if is_complete_checkpoint_dir(final_dir):
            self._validate_existing_publication(final_dir)
            reconcile_publication(self.output_dir, final_dir)
            return
        model = state.model
        if model is None:
            raise ValueError("Checkpoint callback requires state.model")
        save_checkpoint(
            output_dir=self.output_dir,
            next_iteration=next_iteration,
            completed_updates=completed_updates,
            model=model,
            # The composed payload owns which components are written. Passing
            # the available state through lets an explicit TrainResume payload
            # use its defaults (the legacy flags are still honoured by the
            # saver when they are explicitly set to False).
            optimizer=state.optimizer,
            trainer=state.trainer,
            sampler=state.sampler,
            context=context,
            payload=self.payload,
            save_optimizer=self.save_optimizer,
            save_trainer=self.save_trainer,
            save_sampler=self.save_sampler,
            save_rng=self.save_rng,
        )

    def _validate_existing_publication(self, final_dir: Path) -> None:
        """Allow only an equivalent re-publication at this stream root.

        A complete step directory is the collision boundary for one configured
        stream (its canonical ``output_dir``).  Replaying the same payload is
        idempotent, while a different payload must fail before ``save_checkpoint``
        can write or publish anything for the second stream.
        """

        existing = _existing_components(final_dir)
        configured = _configured_components(
            self.payload,
            save_optimizer=self.save_optimizer,
            save_trainer=self.save_trainer,
            save_sampler=self.save_sampler,
            save_rng=self.save_rng,
        )
        if existing is not None and existing == configured:
            return
        raise ValueError(
            f"checkpoint stream collision at {final_dir}: existing payload="
            f"{_payload_label(existing)!r}, configured payload="
            f"{_payload_label(configured)!r}"
        )

    def _reset_typed_state(self) -> None:
        """Drop the armed update when the owning RunContext identity changes."""

        self._updated_iteration_step = None


def _trainer_progress(state: TrainerState) -> tuple[int, int]:
    """Read the trainer's two required progress counters from typed state.

    Parameters
    ----------
    state : TrainerState
        The loop's training state. Its ``trainer`` field is read by name; no
        payload keys and no arbitrary attributes are probed.

    Returns
    -------
    tuple of int
        ``(next_iteration, completed_updates)``.

    Raises
    ------
    ValueError
        If the state carries no trainer.
    TypeError
        If the trainer does not expose a ``state_dict()`` returning a mapping.
    KeyError
        If either counter is missing. Both are required keys with no default:
        a trainer that cannot report its resume cursor must not be allowed to
        mislabel a checkpoint directory.
    """

    trainer = state.trainer
    if trainer is None:
        raise ValueError("Checkpoint callback requires state.trainer")
    state_dict = getattr(trainer, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("trainer must expose state_dict() for checkpoint progress")
    progress = state_dict()
    if not isinstance(progress, Mapping):
        raise TypeError("trainer.state_dict() must return a mapping")
    return int(progress["next_iteration"]), int(progress["completed_updates"])


_PAYLOAD_COMPONENTS = frozenset(("model", "optimizer", "trainer", "sampler", "rng"))


def _configured_components(
    payload: CheckpointPayload | None,
    *,
    save_optimizer: bool | None,
    save_trainer: bool | None,
    save_sampler: bool | None,
    save_rng: bool | None,
) -> frozenset[str]:
    """Return the effective component set this callback would publish."""

    if payload is not None:
        resolved = {
            component: (component in payload.required_files if flag is None else flag)
            for component, flag in (
                ("optimizer", save_optimizer),
                ("trainer", save_trainer),
                ("sampler", save_sampler),
                ("rng", save_rng),
            )
        }
        payload.validate_save_flags({"model": True, **resolved})
        return frozenset(payload.required_files)
    return frozenset(
        component
        for component, flag in (
            ("model", True),
            ("optimizer", save_optimizer),
            ("trainer", save_trainer),
            ("sampler", save_sampler),
            ("rng", save_rng),
        )
        if flag is not False
    )


def _existing_components(final_dir: Path) -> frozenset[str] | None:
    """Read the effective component set of a complete existing checkpoint."""

    manifest = read_manifest(final_dir / "manifest.json", mode="model_only")
    if manifest.payload is not None:
        # Validate payload metadata when present, but use the files map as the
        # collision key so metadata cannot hide an unexpected extra component.
        CheckpointPayload.from_manifest(manifest.payload)
    return frozenset(name for name in manifest.files if name in _PAYLOAD_COMPONENTS)


def _payload_label(components: frozenset[str] | None) -> str | None:
    """Render a component set for a collision diagnostic."""

    if components is None:
        return None
    if components == frozenset(("model",)):
        return ModelOnly().name
    if components == _PAYLOAD_COMPONENTS:
        return TrainResume().name
    return f"components={sorted(components)!r}"


__all__ = [
    "Checkpoint",
]
