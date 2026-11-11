"""Delivery tests for `Checkpoint` and `Status`, the last two state readers.

Required by the ruling recorded on item ``62593af4``. The dispatcher in
`tpen.artifacts.RunContext._dispatch_occurrence` SKIPS a `StatefulCallback`
whose declared domain does not match the state it was handed, and the same
branch fires when the emitter passes no state at all. Skipping is correct and
specified, but it means an emitter that forgets ``state=`` produces a callback
that observes nothing, silently, with no error anywhere -- the same shape as
defect ``933b5f78``, where `GradientStats` reported ``passed: true`` while
observing zero gradients.

So every migrated callback gets a test asserting it really is delivered its
state at the boundary it subscribes to. These use the REAL dispatcher, and the
end-to-end cases use the REAL runner: a `RunContext` stand-in would override the
very method under test, and only the runner can prove that the emitting SITE
passes state.

`TrainingCompleted` is a brand new event with exactly one emit site, so the
runner-level test below is the one that would catch a forgotten ``state=``
there. Without it the terminal checkpoint would silently stop being written.
"""

from __future__ import annotations

import logging

import pytest
import torch

from tpen.callback import Checkpoint, Status
from tpen.checkpoint import ModelOnly
from tpen.runner import Train
from tpen.training.events import (
    TrainingCompleted,
    TrainingIteration,
    TrainingIterationCompleted,
    UpdateCompleted,
)
from tpen.training.state import TrainerState
from tpen.training.trainer import VMCTrainer
from tests.helpers.hooke_models import (
    build_tiny_hamiltonian_terms,
    build_tiny_sampler,
    build_tiny_spenn,
)
from tests.helpers.run_context import make_run_context
from tests.unit.callback.support import make_sampler_stats, training_state


class _Trainer:
    """Trainer stand-in reporting only the two required progress counters."""

    def __init__(self, *, next_iteration: int, completed_updates: int) -> None:
        self.next_iteration = next_iteration
        self.completed_updates = completed_updates

    def state_dict(self) -> dict[str, int]:
        return {
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
        }


class _Sampler:
    def mcmc_state_dict(self) -> dict[str, bool]:
        return {"has_burned_in": True}


def _savable_state(*, step: int, next_iteration: int, completed_updates: int) -> TrainerState:
    model = torch.nn.Linear(2, 1)
    return training_state(
        step=step,
        metrics={"loss": 0.5},
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        trainer=_Trainer(
            next_iteration=next_iteration, completed_updates=completed_updates
        ),
        sampler=_Sampler(),
    )


def test_checkpoint_receives_state_at_the_completed_iteration_boundary(tmp_path) -> None:
    """The dispatcher hands `Checkpoint` its domain state, and it writes."""

    checkpoint_dir = tmp_path / "checkpoints"
    callback = Checkpoint(output_dir=checkpoint_dir, terminal=False, every_n_steps=1)
    context = make_run_context(tmp_path, callbacks=[callback])
    state = _savable_state(step=0, next_iteration=1, completed_updates=1)
    iteration = TrainingIteration(step=0)

    context.emit(UpdateCompleted(iteration=iteration), state=state)
    context.emit(TrainingIterationCompleted(iteration=iteration), state=state)

    assert (checkpoint_dir / "step_000001" / "COMPLETE").exists()


def test_checkpoint_receives_state_at_the_training_completed_boundary(tmp_path) -> None:
    """The same, for the newly minted terminal event."""

    checkpoint_dir = tmp_path / "checkpoints"
    callback = Checkpoint(output_dir=checkpoint_dir, periodic=False)
    context = make_run_context(tmp_path, callbacks=[callback])

    context.emit(
        TrainingCompleted(),
        state=_savable_state(step=-1, next_iteration=7, completed_updates=7),
    )

    assert (checkpoint_dir / "step_000007" / "COMPLETE").exists()


def test_independent_root_checkpoint_streams_publish_at_one_boundary(tmp_path) -> None:
    """Root-owned streams write independently when their roots are distinct."""

    first_dir = tmp_path / "model-checkpoints"
    second_dir = tmp_path / "resume-checkpoints"
    callbacks = [
        Checkpoint(output_dir=first_dir, payload=ModelOnly(), periodic=False),
        Checkpoint(output_dir=second_dir, payload=ModelOnly(), periodic=False),
    ]
    context = make_run_context(tmp_path, callbacks=callbacks)

    context.emit(
        TrainingCompleted(),
        state=_savable_state(step=-1, next_iteration=7, completed_updates=7),
    )

    assert (first_dir / "step_000007" / "COMPLETE").exists()
    assert (second_dir / "step_000007" / "COMPLETE").exists()
    assert len((first_dir / "publications.jsonl").read_text().splitlines()) == 1
    assert len((second_dir / "publications.jsonl").read_text().splitlines()) == 1


def test_status_receives_state_at_the_completed_iteration_boundary(
    tmp_path, caplog: pytest.LogCaptureFixture
) -> None:
    """The dispatcher hands `Status` its domain state, and it renders a line."""

    callback = Status(color="never", train_lines=True, include=["train/loss"])
    context = make_run_context(tmp_path, callbacks=[callback])
    state = training_state(
        step=3, metrics={"loss": 0.25}, sampler_stats=make_sampler_stats()
    )

    with caplog.at_level(logging.INFO, logger="tpen.status"):
        context.emit(
            TrainingIterationCompleted(iteration=TrainingIteration(step=3)), state=state
        )

    assert caplog.records[-1].getMessage() == "[train] step=3 loss=0.25"


@pytest.mark.parametrize("event", ["iteration", "training"], ids=["iteration", "training"])
def test_a_boundary_emitted_without_state_delivers_nothing(tmp_path, event: str) -> None:
    """Absent state is skipped, not raised -- which is why the tests above exist.

    This pins the hazard rather than the fix: a forgotten ``state=`` really does
    produce total silence, so the positive tests are the only thing standing
    between that mistake and a run that silently stops checkpointing.
    """

    checkpoint_dir = tmp_path / "checkpoints"
    callback = Checkpoint(output_dir=checkpoint_dir, every_n_steps=1)
    context = make_run_context(tmp_path, callbacks=[callback])

    if event == "iteration":
        context.emit(TrainingIterationCompleted(iteration=TrainingIteration(step=0)))
    else:
        context.emit(TrainingCompleted())

    assert not checkpoint_dir.exists()


def test_the_train_runner_passes_state_at_the_terminal_boundary(tmp_path) -> None:
    """End-to-end: `Train` really carries state on the new terminal event.

    The unit tests above emit `TrainingCompleted` themselves, so they would keep
    passing if `tpen.runner.Train` dropped its ``state=``. This drives the real
    runner, which is the single site where that mistake would be made.
    """

    checkpoint_dir = tmp_path / "checkpoints"
    callback = Checkpoint(output_dir=checkpoint_dir, periodic=False)
    context = make_run_context(tmp_path, callbacks=[callback])
    model = build_tiny_spenn()

    Train(
        model=model,
        sampler=build_tiny_sampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=lambda params: torch.optim.Adam(params, lr=0.01),
        trainer=VMCTrainer(max_steps=2, log_every_n_steps=1),
    ).run(context)

    assert (checkpoint_dir / "step_000002" / "COMPLETE").exists()


def test_the_terminal_boundary_fires_when_the_loop_body_never_ran(tmp_path) -> None:
    """``max_steps=0`` still owes -- and writes -- a terminal checkpoint.

    The whole reason the terminal moment needed its own event: no iteration
    runs, so no `TrainingIterationCompleted` is ever emitted, and
    `TrainerState.step` is still ``-1``. The directory is named from the
    trainer's resume cursor, so it is ``step_000000`` rather than the
    ``step_-00001`` that ``state.step`` would have produced.
    """

    checkpoint_dir = tmp_path / "checkpoints"
    callback = Checkpoint(output_dir=checkpoint_dir, periodic=False)
    context = make_run_context(tmp_path, callbacks=[callback])
    model = build_tiny_spenn()

    Train(
        model=model,
        sampler=build_tiny_sampler(),
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=lambda params: torch.optim.Adam(params, lr=0.01),
        trainer=VMCTrainer(max_steps=0, log_every_n_steps=1),
    ).run(context)

    assert (checkpoint_dir / "step_000000" / "COMPLETE").exists()
    assert sorted(path.name for path in checkpoint_dir.glob("step_*")) == ["step_000000"]
