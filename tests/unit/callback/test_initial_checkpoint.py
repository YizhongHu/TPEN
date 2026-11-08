"""Unit tests for the `InitialCheckpoint` callback.

These tests drive `InitialCheckpoint.handle_occurrence` directly with
hand-built typed occurrences, the same style `tests/unit/callback/test_checkpoint.py`
uses for the periodic `Checkpoint` callback. The bitwise-resume and
ordering-relative-to-sampling claims need a real trainer loop and live in
`tests/unit/training/test_initial_checkpoint_integration.py` instead.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.callback import Checkpoint
from tpen.callback.initial_checkpoint import InitialCheckpoint
from tpen.checkpoint import checkpoint_step_dir_name, save_checkpoint
from tpen.events import Occurrence, Started
from tpen.training.events import TrainingIteration, UpdateCompleted, TrainingIterationCompleted
from tpen.training.state import TrainerState


def _cfg():
    return OmegaConf.create(
        {
            "model": {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 2},
            "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01},
            "trainer": {"_target_": "tests.Trainer", "max_steps": 2},
            "sampler": {"_target_": "tests.Sampler", "n_steps": 5},
            "hamiltonian_terms": {"constant": {"_target_": "tests.ConstantHamiltonian"}},
            "run": {"run_id": "run", "dir": "/tmp/run"},
            "study": {"name": "unit", "config_id": "cfg"},
        }
    )


class _InitialCheckpointContext(RunContext):
    """Minimal real `RunContext` satisfying `handle_occurrence`'s typeguard."""

    def __init__(self, tmp_path: Path) -> None:
        artifact_manager = ArtifactManager(
            tmp_path / "run-artifacts",
            experiment="initial-checkpoint",
            sector="unit",
            run_id="initial-checkpoint-unit",
            layout="flat",
        )
        artifact_manager.make_dirs()
        self.cfg = _cfg()
        self.source_cfg = OmegaConf.create({})
        self.artifact_manager = artifact_manager
        self.metadata = RunMetadata(
            run_id="initial-checkpoint-unit",
            run_name="initial-checkpoint-unit",
            timestamp="2026-11-08T12:00:00+00:00",
            timezone="UTC",
            git_commit="test-sha",
            git_branch="test-branch",
            dirty_worktree=False,
            command="pytest",
            config_path="test.yaml",
            resolved_config_path=str(artifact_manager.path("resolved_config.yaml")),
            run_dir=str(artifact_manager.run_dir),
            device="cpu",
            dtype="float64",
        )
        self.clock = RunClock(timezone="UTC", tzinfo=UTC)
        self.callbacks = []
        self.loggers = []
        self.topology = None
        self.profile_writer = None
        self._occurrence_counts = {}


class _Trainer:
    """Trainer double exposing exactly the required progress state_dict."""

    def __init__(self, *, next_iteration: int, completed_updates: int) -> None:
        self.next_iteration = int(next_iteration)
        self.completed_updates = int(completed_updates)
        self.loaded = None

    def state_dict(self) -> dict[str, int]:
        return {
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, state) -> None:
        self.loaded = dict(state)


class _Sampler:
    def __init__(self) -> None:
        self.loaded = None

    def mcmc_state_dict(self) -> dict[str, object]:
        return {"has_burned_in": False, "position": torch.ones(1)}

    def load_mcmc_state_dict(self, state) -> None:
        self.loaded = dict(state)


def _fresh_state(*, step: int = -1) -> TrainerState:
    """Build a fresh-lineage pre-sampling `TrainerState` (next_iteration=0)."""

    model = torch.nn.Linear(3, 2).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return TrainerState(
        step=step,
        model=model,
        optimizer=optimizer,
        trainer=_Trainer(next_iteration=0, completed_updates=0),
        sampler=_Sampler(),
    )


def _started_iteration(step: int, *, count: int = 1) -> Occurrence[Started[TrainingIteration]]:
    return Occurrence(event=Started(TrainingIteration(step=step)), count=count)


def _manifest(checkpoint_dir: Path) -> dict:
    return json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))


def test_writes_full_train_resume_payload_once(tmp_path: Path) -> None:
    """Exactly one complete TrainResume checkpoint is written at boundary zero."""

    context = _InitialCheckpointContext(tmp_path)
    output_dir = tmp_path / "checkpoints"
    callback = InitialCheckpoint(output_dir)
    context.callbacks = [callback]
    state = _fresh_state()

    callback.handle_occurrence(_started_iteration(0, count=1), context, state)

    final_dir = output_dir / checkpoint_step_dir_name(0)
    assert (final_dir / "COMPLETE").is_file()
    manifest = _manifest(final_dir)
    assert {"model", "optimizer", "trainer", "sampler", "rng"} <= set(manifest["files"])
    assert manifest["next_iteration"] == 0
    assert manifest["completed_updates"] == 0

    # A second Started(TrainingIteration) delivery -- e.g. a stray redelivery --
    # must not attempt a second write. The `_written` guard short-circuits
    # before `save_checkpoint` is ever called again, so even an attempted
    # (and would-be-refused) second write cannot happen.
    mtime_before = (final_dir / "COMPLETE").stat().st_mtime_ns
    callback.handle_occurrence(_started_iteration(0, count=2), context, state)
    assert (final_dir / "COMPLETE").stat().st_mtime_ns == mtime_before


def test_refuses_resumed_lineage_nonzero_step(tmp_path: Path) -> None:
    """A resumed run's first iteration (nonzero step) is refused, not written."""

    context = _InitialCheckpointContext(tmp_path)
    output_dir = tmp_path / "checkpoints"
    callback = InitialCheckpoint(output_dir)
    context.callbacks = [callback]
    state = TrainerState(
        step=2,
        model=torch.nn.Linear(3, 2).double(),
        optimizer=torch.optim.Adam(torch.nn.Linear(3, 2).double().parameters(), lr=0.01),
        trainer=_Trainer(next_iteration=3, completed_updates=3),
        sampler=_Sampler(),
    )

    with pytest.raises(ValueError, match="fresh zero-update lineage"):
        callback.handle_occurrence(_started_iteration(3, count=1), context, state)

    assert not output_dir.exists()


def test_refuses_nonzero_completed_updates_even_at_step_zero(tmp_path: Path) -> None:
    """A trainer reporting nonzero completed_updates at step 0 is still refused.

    This is not a realistic trainer state (next_iteration tracks step), but it
    pins that the refusal checks BOTH counters independently, not just the
    iteration step.
    """

    context = _InitialCheckpointContext(tmp_path)
    output_dir = tmp_path / "checkpoints"
    callback = InitialCheckpoint(output_dir)
    context.callbacks = [callback]
    state = TrainerState(
        step=-1,
        model=torch.nn.Linear(3, 2).double(),
        optimizer=torch.optim.Adam(torch.nn.Linear(3, 2).double().parameters(), lr=0.01),
        trainer=_Trainer(next_iteration=0, completed_updates=1),
        sampler=_Sampler(),
    )

    with pytest.raises(ValueError, match="fresh zero-update lineage"):
        callback.handle_occurrence(_started_iteration(0, count=1), context, state)

    assert not output_dir.exists()


def test_refuses_an_existing_checkpoint_rather_than_overwriting(tmp_path: Path) -> None:
    """An existing checkpoint at step_000000 is refused, never overwritten."""

    output_dir = tmp_path / "checkpoints"
    # Pre-populate step_000000 with a DIFFERENT model's weights, standing in
    # for "something already wrote this path" (e.g. a prior attempt, or an
    # unrelated run pointed at the same output_dir by mistake).
    decoy_model = torch.nn.Linear(3, 2).double()
    with torch.no_grad():
        decoy_model.weight.fill_(7.0)
    save_checkpoint(
        output_dir=output_dir,
        next_iteration=0,
        completed_updates=0,
        model=decoy_model,
        optimizer=torch.optim.Adam(decoy_model.parameters(), lr=0.01),
        trainer=_Trainer(next_iteration=0, completed_updates=0),
        sampler=_Sampler(),
        context=_InitialCheckpointContext(tmp_path),
    )
    final_dir = output_dir / checkpoint_step_dir_name(0)
    before = (final_dir / "model.pt").read_bytes()

    context = _InitialCheckpointContext(tmp_path)
    callback = InitialCheckpoint(output_dir)
    context.callbacks = [callback]
    state = _fresh_state()

    with pytest.raises(FileExistsError):
        callback.handle_occurrence(_started_iteration(0, count=1), context, state)

    # Byte-for-byte unchanged: no silent overwrite, no partial write residue.
    assert (final_dir / "model.pt").read_bytes() == before


def test_ignores_non_training_iteration_started_events(tmp_path: Path) -> None:
    """A Started boundary for a different operation type is simply skipped."""

    from tpen.training.events import CollectSamples

    context = _InitialCheckpointContext(tmp_path)
    output_dir = tmp_path / "checkpoints"
    callback = InitialCheckpoint(output_dir)
    context.callbacks = [callback]
    state = _fresh_state()

    occurrence = Occurrence(event=Started(CollectSamples(step=0)), count=1)
    callback.handle_occurrence(occurrence, context, state)

    assert not output_dir.exists()
    assert not callback._written


def test_coexists_idempotently_with_a_periodic_checkpoint_at_update_zero(tmp_path: Path) -> None:
    """The default periodic Checkpoint callback's step-0 write is a no-op republication.

    `Checkpoint._save` checks `is_complete_checkpoint_dir` before writing, and
    with matching default (all-component) payloads an already-complete
    directory is reconciled rather than re-written. `InitialCheckpoint` always
    runs first (it fires on the TrainingIteration Started boundary, strictly
    before the TrainingIterationCompleted boundary the periodic callback's
    write is gated on), so this pins "no double write, no collision" for the
    common default-configuration case.
    """

    output_dir = tmp_path / "checkpoints"
    context = _InitialCheckpointContext(tmp_path)
    initial = InitialCheckpoint(output_dir)
    periodic = Checkpoint(output_dir)
    context.callbacks = [initial, periodic]
    state = _fresh_state()

    # InitialCheckpoint fires on Started(TrainingIteration) -- before sampling.
    initial.handle_occurrence(_started_iteration(0, count=1), context, state)
    final_dir = output_dir / checkpoint_step_dir_name(0)
    written_bytes = (final_dir / "model.pt").read_bytes()

    # The periodic callback is armed by UpdateCompleted and writes at
    # TrainingIterationCompleted for the same step, as it would in a real run
    # whose first iteration actually applied an optimizer update.
    periodic.handle_occurrence(
        Occurrence(event=UpdateCompleted(iteration=TrainingIteration(step=0)), count=1),
        context,
        state,
    )
    periodic.handle_occurrence(
        Occurrence(
            event=TrainingIterationCompleted(iteration=TrainingIteration(step=0)), count=1
        ),
        context,
        state,
    )

    # Still exactly one checkpoint directory, content unchanged (reconciled,
    # not rewritten).
    assert list(output_dir.glob("step_*")) == [final_dir]
    assert (final_dir / "model.pt").read_bytes() == written_bytes


def test_collides_with_a_differently_configured_periodic_checkpoint_at_update_zero(
    tmp_path: Path,
) -> None:
    """A periodic Checkpoint configured for a DIFFERENT payload collides loudly.

    `Checkpoint._validate_existing_publication` compares the existing
    component set against the configured one and raises on mismatch -- this is
    the defined collision behaviour, inherited unchanged from the periodic
    callback, not new coupling introduced by `InitialCheckpoint`.
    """

    output_dir = tmp_path / "checkpoints"
    context = _InitialCheckpointContext(tmp_path)
    initial = InitialCheckpoint(output_dir)
    # Configured to skip the sampler component -- a different payload shape
    # than InitialCheckpoint's full TrainResume write.
    periodic = Checkpoint(output_dir, save_sampler=False)
    context.callbacks = [initial, periodic]
    state = _fresh_state()

    initial.handle_occurrence(_started_iteration(0, count=1), context, state)

    with pytest.raises(ValueError, match="checkpoint stream collision"):
        periodic.handle_occurrence(
            Occurrence(event=UpdateCompleted(iteration=TrainingIteration(step=0)), count=1),
            context,
            state,
        )
        periodic.handle_occurrence(
            Occurrence(
                event=TrainingIterationCompleted(iteration=TrainingIteration(step=0)), count=1
            ),
            context,
            state,
        )
