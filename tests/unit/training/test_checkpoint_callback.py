"""Unit tests for the Checkpoint callback."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import tpen
from tpen.artifacts import RunContext
from tpen.callback import Checkpoint
from tpen.callback.terminal_logging import configure_terminal_logging
from tpen.checkpoint import (
    CheckpointCatalog,
    CheckpointRef,
    EveryNUpdates,
    ExplicitUpdates,
    ModelOnly,
    TrainResume,
    checkpoint_hashes,
    read_publications,
    save_checkpoint,
)
from tpen.checkpoint.hashing import file_sha256
from tpen.events import Occurrence
from tpen.training.events import (
    TrainingCompleted,
    TrainingIteration,
    TrainingIterationCompleted,
    UpdateCompleted,
)
from tpen.training.state import TrainerState


class _CheckpointContext(RunContext):
    """Minimal `RunContext` carrying only what `save_checkpoint` reads.

    This subclasses `RunContext` rather than duck-typing a `SimpleNamespace`
    because `tpen.callback.StatefulCallback.handle_occurrence` annotates its
    context parameter and the suite runs typeguard over the ``tpen`` package, so
    the typed dispatch path rejects a stand-in that is not really a
    `RunContext`.

    The dataclass ``__init__`` is bypassed on purpose -- an artifact manager,
    clock, and logger list are irrelevant to checkpoint writing -- and
    ``run_dir`` is overridden because the base class resolves it through the
    artifact manager this stub never builds. Mirrors the established
    ``tests/unit/callback/support.py::RecordingContext`` pattern.
    """

    def __init__(self, cfg, metadata, run_dir: str) -> None:
        self.cfg = cfg
        self.metadata = metadata
        self._run_dir = Path(run_dir)

    @property
    def run_dir(self) -> Path:
        """Return the stubbed run directory."""

        return self._run_dir

    def __repr__(self) -> str:
        # The inherited dataclass repr reads fields this stub never sets, so it
        # would raise while pytest renders a failure. Keep failures readable.
        return f"_CheckpointContext(run_dir={self._run_dir!r})"


def _state(
    step: int,
    *,
    next_iteration: int | None = None,
    completed_updates: int | None = None,
) -> TrainerState:
    """Build loop state for one completed iteration.

    Both counters default to ``step + 1``, the ordinary case where every
    attempted iteration applied its optimizer update. Pass them separately to
    model a vacuum iteration, where the cursor advances but the update count
    does not.
    """

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = _Trainer(
        next_iteration=step + 1 if next_iteration is None else next_iteration,
        completed_updates=step + 1 if completed_updates is None else completed_updates,
    )
    return TrainerState(
        step=step,
        metrics={"loss": 0.5, "energy": 1.25},
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=_SamplerWithMCMCState(),
    )


class _Trainer:
    def __init__(self, *, next_iteration: int, completed_updates: int) -> None:
        self.next_iteration = int(next_iteration)
        self.completed_updates = int(completed_updates)

    def state_dict(self) -> dict[str, int]:
        # Directory numbering follows the resume cursor; cadence follows the
        # applied-update count. Both are required manifest fields.
        return {
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
        }


class _SamplerWithMCMCState:
    def mcmc_state_dict(self) -> dict:
        return {"has_burned_in": True}


class _RecordingSchedule:
    """Capture callback decisions to pin the durable terminal coordinate."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def should_run(self, completed_updates: int, terminal: bool = False) -> bool:
        self.calls.append((completed_updates, terminal))
        return True


class _RejectingTerminalSchedule:
    """Deliberately violate the terminal-admission schedule invariant."""

    def should_run(self, completed_updates: int, terminal: bool = False) -> bool:
        return not terminal


@dataclass
class _FailOncePublish:
    """Wrap one directly imported catalog operation with one deterministic failure."""

    delegate: Callable[[CheckpointCatalog, CheckpointRef], CheckpointRef]
    failures_remaining: int = 1

    def __call__(self, catalog: CheckpointCatalog, ref: CheckpointRef) -> CheckpointRef:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise OSError("catalog publication failed once")
        return self.delegate(catalog, ref)


class _FailOnceCatalog(CheckpointCatalog):
    """Catalog that fails before its first append, without dynamic patching."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self._publish_once = _FailOncePublish(CheckpointCatalog.publish)

    def publish(self, ref: CheckpointRef) -> CheckpointRef:
        return self._publish_once(self, ref)


@dataclass(frozen=True)
class _CheckpointSnapshot:
    """Byte snapshot used to prove a repair never rewrites its payload."""

    files: tuple[tuple[str, bytes], ...]


def _checkpoint_snapshot(checkpoint_dir: Path) -> _CheckpointSnapshot:
    return _CheckpointSnapshot(
        tuple(
            (str(path.relative_to(checkpoint_dir)), path.read_bytes())
            for path in sorted(checkpoint_dir.rglob("*"))
            if path.is_file()
        )
    )


def _iteration(
    callback: Checkpoint,
    state: TrainerState,
    context,
    *,
    applied_update: bool = True,
    occurrence_count: int = 1,
) -> None:
    """Replay one trainer iteration's typed deliveries in loop order.

    Mirrors `tpen.training.VMCTrainer.fit`: an iteration that applied its
    optimizer update emits `UpdateCompleted` immediately after
    ``optimizer.step()`` returns, then `TrainingIterationCompleted` at the end
    of the body. A vacuum iteration emits `UpdateSkipped` instead, which
    `Checkpoint` does not subscribe to, so ``applied_update=False`` arms
    nothing.

    The same `context` object must reach every delivery, exactly as at runtime:
    `tpen.callback.StatefulCallback` resets typed state when the context
    identity changes.

    Parameters
    ----------
    occurrence_count : int, optional
        Run-local occurrence coordinate carried by the typed event. Varied
        independently of ``completed_updates`` so tests can pin that cadence
        follows the durable counter rather than this one.
    """

    iteration = TrainingIteration(step=state.step)
    if applied_update:
        callback.handle_occurrence(
            Occurrence(event=UpdateCompleted(iteration=iteration), count=occurrence_count),
            context,
            state,
        )
    callback.handle_occurrence(
        Occurrence(
            event=TrainingIterationCompleted(iteration=iteration), count=occurrence_count
        ),
        context,
        state,
    )


def _finish(
    callback: Checkpoint, state: TrainerState, context, *, occurrence_count: int = 1
) -> None:
    """Deliver the terminal boundary the runner emits once `fit` returns."""

    callback.handle_occurrence(
        Occurrence(event=TrainingCompleted(), count=occurrence_count), context, state
    )


def test_checkpoint_writes_step_directory_and_latest_pointer(tmp_path) -> None:
    callback = Checkpoint(
        output_dir=tmp_path / "checkpoints", terminal=False, every_n_steps=1
    )

    _iteration(callback, _state(2), _context())

    ckpt_dir = tmp_path / "checkpoints"
    step_dir = ckpt_dir / "step_000003"
    assert step_dir.is_dir()
    assert (step_dir / "manifest.json").exists()
    assert (step_dir / "resolved_config.yaml").exists()
    assert (step_dir / "model.pt").exists()
    assert (step_dir / "optimizer.pt").exists()
    assert (step_dir / "trainer.json").exists()
    assert (step_dir / "sampler.pt").exists()
    assert (step_dir / "rng.pt").exists()
    assert (step_dir / "COMPLETE").exists()
    assert (ckpt_dir / "latest.json").exists()
    assert not (ckpt_dir / "step_000003.tmp").exists()


def test_non_null_keep_last_warns_once_per_callback(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="tpen"):
        callback = Checkpoint(output_dir=tmp_path, keep_last=1, terminal=False)
        _iteration(callback, _state(0), _context())
        _iteration(callback, _state(1), _context())
        assert sorted(path.name for path in tmp_path.glob("step_*")) == [
            "step_000001",
            "step_000002",
        ]

        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "keep_last=1 is ignored" in message
        assert "checkpoint storage grows without bound" in message
        assert "tpen/metrics_naming.md" in message

        Checkpoint(output_dir=tmp_path, keep_last=1, terminal=False)
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 2
    assert callback.keep_last == 1


def test_none_keep_last_does_not_warn(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="tpen"):
        Checkpoint(output_dir=tmp_path, keep_last=None, terminal=False)

    assert [record for record in caplog.records if record.levelno == logging.WARNING] == []


def test_keep_last_warning_reaches_configured_terminal_stderr(tmp_path, capsys) -> None:
    configure_terminal_logging(enabled=True, level="warning", color="never")
    Checkpoint(output_dir=tmp_path, keep_last=3, terminal=False)

    assert "keep_last=3 is ignored" in capsys.readouterr().err


def test_checkpoint_composes_schedule_and_payload(tmp_path) -> None:
    callback = Checkpoint(
        output_dir=tmp_path,
        schedule=EveryNUpdates(2),
        payload=ModelOnly(),
        terminal=False,
    )

    _iteration(callback, _state(1), _context())

    step_dir = tmp_path / "step_000002"
    manifest = json.loads((step_dir / "manifest.json").read_text())
    assert manifest["payload"] == ModelOnly().to_manifest()
    assert (step_dir / "model.pt").exists()
    assert not (step_dir / "optimizer.pt").exists()
    assert not (step_dir / "trainer.json").exists()
    assert not (step_dir / "sampler.pt").exists()
    assert not (step_dir / "rng.pt").exists()


def test_no_checkpoint_is_written_at_the_update_boundary(tmp_path) -> None:
    """The periodic write lands on the completed ITERATION, not on the update.

    `UpdateCompleted` fires immediately after ``optimizer.step()`` returns,
    which is before any health check can run. Writing there would persist a
    model version no check has yet accepted, and would make defect ``195c3ff3``
    strictly worse than the state PR #175 left it in. Its only job here is to
    arm the write that the later boundary performs.
    """

    callback = Checkpoint(output_dir=tmp_path, terminal=False, every_n_steps=1)
    context = _context()
    state = _state(0)
    iteration = TrainingIteration(step=0)

    callback.handle_occurrence(
        Occurrence(event=UpdateCompleted(iteration=iteration), count=1), context, state
    )
    assert sorted(path.name for path in tmp_path.glob("step_*")) == []
    assert not (tmp_path / "latest.json").exists()

    callback.handle_occurrence(
        Occurrence(event=TrainingIterationCompleted(iteration=iteration), count=1),
        context,
        state,
    )
    assert (tmp_path / "step_000001" / "COMPLETE").exists()


def test_checkpoint_payload_contains_expected_keys(tmp_path) -> None:
    callback = Checkpoint(output_dir=tmp_path, terminal=False, every_n_steps=1)
    state = _state(3)

    _iteration(callback, state, _context())

    manifest = torch.load(tmp_path / "step_000004" / "model.pt", weights_only=False)
    assert set(manifest) == set(state.model.state_dict())
    sampler_state = torch.load(tmp_path / "step_000004" / "sampler.pt", weights_only=False)
    assert sampler_state == {"has_burned_in": True}


def test_checkpoint_respects_every_n_steps_filter(tmp_path) -> None:
    callback = Checkpoint(output_dir=tmp_path, terminal=False, every_n_steps=2)
    context = _context()

    _iteration(callback, _state(0), context)
    assert not (tmp_path / "step_000001").exists()

    _iteration(callback, _state(1), context)
    assert (tmp_path / "step_000002").exists()


def test_checkpoint_cadence_counts_completed_updates(tmp_path) -> None:
    """Cadence is spaced by applied updates; identity stays the resume cursor."""

    callback = Checkpoint(output_dir=tmp_path, terminal=False, every_n_steps=5)

    # Five completed updates hits the cadence, but the run attempted six
    # iterations, so the directory is named for the cursor, not the count.
    _iteration(callback, _state(5, next_iteration=6, completed_updates=5), _context())

    assert (tmp_path / "step_000006").exists()
    assert not (tmp_path / "step_000005").exists()


def test_checkpoint_cadence_fires_once_per_update_completed(tmp_path) -> None:
    """One firing per `UpdateCompleted`; a skipped update fires nothing."""

    callback = Checkpoint(output_dir=tmp_path, terminal=False, every_n_steps=1)
    context = _context()

    # Iteration 0 applies its update: cursor 1, one completed update.
    _iteration(callback, _state(0, next_iteration=1, completed_updates=1), context)
    # Iteration 1 is a zero-electron vacuum. It emits `UpdateSkipped`, which
    # this callback does not subscribe to, so nothing is armed and the write is
    # skipped -- even though the cursor advanced and every_n_steps=1 would
    # otherwise admit the unchanged count.
    _iteration(
        callback,
        _state(1, next_iteration=2, completed_updates=1),
        context,
        applied_update=False,
    )
    # Iteration 2 applies an update again, so the write fires exactly once more.
    _iteration(callback, _state(2, next_iteration=3, completed_updates=2), context)

    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_000001",
        "step_000003",
    ]


def test_checkpoint_cadence_phase_survives_resume(tmp_path) -> None:
    """Cadence follows the durable counter, not the run-local occurrence count.

    A resumed run restarts `Occurrence.count` at 1. If that were the cadence
    coordinate, the resumed run would checkpoint at different points than an
    uninterrupted run that reached the same ``completed_updates``. Gating on
    the trainer's durable counter keeps the two aligned.
    """

    uninterrupted_dir = tmp_path / "uninterrupted"
    uninterrupted = Checkpoint(
        output_dir=uninterrupted_dir, terminal=False, every_n_steps=3
    )
    context = _context()
    # Nine iterations, each applying its update: completed_updates 1..9.
    for step in range(9):
        _iteration(uninterrupted, _state(step), context, occurrence_count=step + 1)

    resumed_dir = tmp_path / "resumed"
    resumed = Checkpoint(output_dir=resumed_dir, terminal=False, every_n_steps=3)
    resumed_context = _context()
    # Restored mid-phase at completed_updates=4, so the next admitted count is
    # 6, not 4 + 3. Occurrence counts restart at 1 for the new context.
    for offset, step in enumerate(range(4, 9)):
        _iteration(resumed, _state(step), resumed_context, occurrence_count=offset + 1)

    uninterrupted_names = sorted(path.name for path in uninterrupted_dir.glob("step_*"))
    resumed_names = sorted(path.name for path in resumed_dir.glob("step_*"))
    assert uninterrupted_names == ["step_000003", "step_000006", "step_000009"]
    # Same checkpoint points, minus the ones already written before the restore.
    assert resumed_names == ["step_000006", "step_000009"]
    # Occurrence-count cadence would have fired on the resumed run's third
    # delivery, at completed_updates=7. It must not.
    assert not (resumed_dir / "step_000007").exists()


def test_checkpoint_without_cadence_still_requires_a_completed_update(tmp_path) -> None:
    """No `every_n_steps` means every completed update, not every iteration."""

    callback = Checkpoint(output_dir=tmp_path, terminal=False)
    context = _context()

    _iteration(callback, _state(0, next_iteration=1, completed_updates=1), context)
    assert (tmp_path / "step_000001").exists()

    # Update selection is unconditional rather than part of the cadence window,
    # so a vacuum iteration writes nothing even with no window configured.
    _iteration(
        callback,
        _state(1, next_iteration=2, completed_updates=1),
        context,
        applied_update=False,
    )
    assert not (tmp_path / "step_000002").exists()


def test_terminal_checkpoint_survives_a_final_vacuum_iteration(tmp_path) -> None:
    """Update selection must not reach the terminal write."""

    callback = Checkpoint(output_dir=tmp_path, every_n_steps=1)
    context = _context()

    _iteration(callback, _state(0, next_iteration=1, completed_updates=1), context)
    # The run's last iteration is a zero-electron vacuum, so no periodic
    # checkpoint fires and the armed step stays behind the iteration's own.
    vacuum = _state(1, next_iteration=2, completed_updates=1)
    _iteration(callback, vacuum, context, applied_update=False)
    # The terminal boundary carries that same mutated state. It must still write.
    _finish(callback, vacuum, context)

    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_000001",
        "step_000002",
    ]


def test_terminal_checkpoint_written_when_the_loop_body_never_ran(tmp_path) -> None:
    """`max_steps=0` or a fully-resumed run still writes its terminal checkpoint.

    This is the case that forced a typed terminal event to exist at all. No
    iteration ran, so no `TrainingIterationCompleted` was ever emitted and there
    is no "last iteration" to hang the terminal write on; `TrainingCompleted`
    fires from the runner once `fit` returns, whether or not the body executed.
    """

    callback = Checkpoint(output_dir=tmp_path, periodic=False, every_n_steps=1)
    # No iteration executed: `TrainerState.step` is still -1 and nothing was
    # ever armed, but the resume cursor is 5.
    state = _state(-1, next_iteration=5, completed_updates=5)

    _finish(callback, state, _context())

    assert (tmp_path / "step_000005" / "COMPLETE").exists()
    # `state.step` is -1 here, and `f"step_{-1:06d}"` renders `step_-00001`.
    # The coordinate comes from the trainer's resume cursor, never from state.
    assert sorted(path.name for path in tmp_path.glob("step_*")) == ["step_000005"]


@pytest.mark.parametrize("schedule", [EveryNUpdates(2), ExplicitUpdates([2])])
def test_terminal_checkpoint_ignores_periodic_schedule(
    tmp_path, schedule
) -> None:
    """A terminal boundary is published even when its update count misses."""

    callback = Checkpoint(output_dir=tmp_path, schedule=schedule)
    state = _state(2, next_iteration=3, completed_updates=1)

    _finish(callback, state, _context())

    assert (tmp_path / "step_000003" / "COMPLETE").exists()


def test_terminal_schedule_receives_durable_completed_updates(tmp_path) -> None:
    schedule = _RecordingSchedule()
    callback = Checkpoint(output_dir=tmp_path, schedule=schedule)
    state = _state(2, next_iteration=3, completed_updates=1)

    _finish(callback, state, _context())

    assert schedule.calls == [(1, True)]


def test_terminal_schedule_violation_fails_loudly(tmp_path) -> None:
    callback = Checkpoint(
        output_dir=tmp_path,
        schedule=_RejectingTerminalSchedule(),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"_RejectingTerminalSchedule violated CheckpointSchedule: terminal "
            r"publication must not be suppressed"
        ),
    ):
        _finish(callback, _state(2, next_iteration=3, completed_updates=1), _context())

    assert not (tmp_path / "step_000003").exists()


def test_checkpoint_writes_terminal_checkpoint_without_step_cadence(tmp_path) -> None:
    callback = Checkpoint(output_dir=tmp_path, periodic=False)
    state = _state(3, next_iteration=4)

    _finish(callback, state, _context())

    assert (tmp_path / "step_000004" / "COMPLETE").exists()
    assert (tmp_path / "latest.json").exists()


def test_checkpoint_terminal_skips_existing_complete_checkpoint(tmp_path) -> None:
    callback = Checkpoint(output_dir=tmp_path, every_n_steps=1)
    context = _context()
    state = _state(1, next_iteration=2)

    _iteration(callback, state, context)
    _finish(callback, state, context)

    assert sorted(path.name for path in tmp_path.glob("step_*")) == ["step_000002"]


def test_checkpoint_identical_republication_is_idempotent(tmp_path) -> None:
    state = _state(1, next_iteration=2, completed_updates=2)
    first = Checkpoint(output_dir=tmp_path, payload=ModelOnly(), periodic=False)
    second = Checkpoint(output_dir=tmp_path, payload=ModelOnly(), periodic=False)

    _finish(first, state, _context())
    manifest_before = (tmp_path / "step_000002" / "manifest.json").read_bytes()
    _finish(second, state, _context())

    assert (tmp_path / "step_000002" / "manifest.json").read_bytes() == manifest_before
    assert len((tmp_path / "publications.jsonl").read_text().splitlines()) == 1


def test_checkpoint_retry_repairs_catalog_after_rename_failure(tmp_path) -> None:
    """A retry publishes a committed directory when its first append failed."""

    root = tmp_path / "checkpoints"
    state = _state(1, next_iteration=2, completed_updates=2)
    failing_catalog = _FailOnceCatalog(root / "publications.jsonl")

    with pytest.raises(OSError, match="catalog publication failed once"):
        save_checkpoint(
            output_dir=root,
            next_iteration=2,
            completed_updates=2,
            model=state.model,
            optimizer=state.optimizer,
            trainer=state.trainer,
            sampler=state.sampler,
            context=_context(),
            publication_catalog=failing_catalog,
        )

    final_dir = root / "step_000002"
    assert final_dir.is_dir()
    assert not (root / "publications.jsonl").exists()
    assert not (root / "latest.json").exists()
    payload_before = _checkpoint_snapshot(final_dir)

    _finish(Checkpoint(output_dir=root, periodic=False), state, _context())

    assert len(read_publications(root / "publications.jsonl")) == 1
    latest = json.loads((root / "latest.json").read_text())
    assert latest["checkpoint_dir"] == final_dir.name
    assert _checkpoint_snapshot(final_dir) == payload_before


def test_checkpoint_retry_repairs_latest_without_duplicate_catalog_row(tmp_path) -> None:
    """A retry repairs a stale pointer after the row was already appended."""

    root = tmp_path / "checkpoints"
    state = _state(1, next_iteration=2, completed_updates=2)
    _finish(Checkpoint(output_dir=root, periodic=False), state, _context())

    final_dir = root / "step_000002"
    payload_before = _checkpoint_snapshot(final_dir)
    assert len(read_publications(root / "publications.jsonl")) == 1
    # This is the durable state after catalog append but before latest replace.
    (root / "latest.json").unlink()

    _finish(Checkpoint(output_dir=root, periodic=False), state, _context())

    assert len(read_publications(root / "publications.jsonl")) == 1
    latest = json.loads((root / "latest.json").read_text())
    assert latest["checkpoint_dir"] == final_dir.name
    assert _checkpoint_snapshot(final_dir) == payload_before


def test_checkpoint_legacy_manifest_uses_component_set_for_republication(tmp_path) -> None:
    """A pre-payload manifest remains idempotent when its files are equivalent."""

    state = _state(1, next_iteration=2, completed_updates=2)
    first = Checkpoint(output_dir=tmp_path, periodic=False)

    _finish(first, state, _context())
    manifest_path = tmp_path / "step_000002" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest.pop("payload") == TrainResume().to_manifest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    manifest_before = manifest_path.read_bytes()

    # A real pre-payload checkpoint had these bytes when its catalog row was
    # created. Re-seed the append-only row from that legacy artifact rather
    # than treating a post-publication manifest rewrite as a valid retry.
    (tmp_path / "publications.jsonl").unlink()
    (tmp_path / "latest.json").unlink()
    CheckpointCatalog(tmp_path / "publications.jsonl").publish(
        CheckpointRef.from_directory(tmp_path / "step_000002")
    )

    # Explicit legacy flags are the ordinary pre-composition configuration.
    second = Checkpoint(
        output_dir=tmp_path,
        periodic=False,
        save_optimizer=True,
        save_trainer=True,
        save_sampler=True,
        save_rng=True,
    )
    _finish(second, state, _context())

    assert manifest_path.read_bytes() == manifest_before
    assert len((tmp_path / "publications.jsonl").read_text().splitlines()) == 1


def test_checkpoint_mixed_flags_compare_the_effective_component_set(tmp_path) -> None:
    """Independent legacy flags remain valid stream identities."""

    state = _state(1, next_iteration=2, completed_updates=2)
    flags = {
        "save_optimizer": False,
        "save_trainer": True,
        "save_sampler": False,
        "save_rng": True,
    }
    first = Checkpoint(output_dir=tmp_path, periodic=False, **flags)
    second = Checkpoint(output_dir=tmp_path, periodic=False, **flags)

    _finish(first, state, _context())
    _finish(second, state, _context())

    step_dir = tmp_path / "step_000002"
    assert (step_dir / "model.pt").exists()
    assert (step_dir / "trainer.json").exists()
    assert (step_dir / "rng.pt").exists()
    assert not (step_dir / "optimizer.pt").exists()
    assert not (step_dir / "sampler.pt").exists()
    assert len((tmp_path / "publications.jsonl").read_text().splitlines()) == 1


def test_checkpoint_non_equivalent_payload_collision_fails_before_writing(tmp_path) -> None:
    state = _state(1, next_iteration=2, completed_updates=2)
    first = Checkpoint(output_dir=tmp_path, payload=ModelOnly(), periodic=False)
    second = Checkpoint(output_dir=tmp_path, payload=TrainResume(), periodic=False)

    _finish(first, state, _context())
    manifest_before = (tmp_path / "step_000002" / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="checkpoint stream collision"):
        _finish(second, state, _context())

    assert (tmp_path / "step_000002" / "manifest.json").read_bytes() == manifest_before
    assert not (tmp_path / "step_000002" / "optimizer.pt").exists()
    assert len((tmp_path / "publications.jsonl").read_text().splitlines()) == 1


def test_terminal_updates_latest_when_cadence_misses_terminal_step(tmp_path) -> None:
    periodic = Checkpoint(output_dir=tmp_path, terminal=False, every_n_steps=2)
    terminal = Checkpoint(output_dir=tmp_path, periodic=False)
    context = _context()

    _iteration(periodic, _state(1, next_iteration=2), context)
    final_state = _state(2, next_iteration=3)
    _iteration(periodic, final_state, context)
    _finish(terminal, final_state, context)

    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_000002",
        "step_000003",
    ]
    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["checkpoint_dir"] == "step_000003"
    assert latest["step"] == 3


def test_periodic_only_checkpoint_ignores_the_terminal_boundary(tmp_path) -> None:
    """``terminal: false`` is the config's old ``triggers: [step_end]``."""

    callback = Checkpoint(output_dir=tmp_path, terminal=False)
    context = _context()

    _finish(callback, _state(4, next_iteration=5), context)

    assert sorted(path.name for path in tmp_path.glob("step_*")) == []


def test_terminal_only_checkpoint_ignores_completed_iterations(tmp_path) -> None:
    """``periodic: false`` is the config's old ``triggers: [train_end]``."""

    callback = Checkpoint(output_dir=tmp_path, periodic=False)
    context = _context()

    _iteration(callback, _state(0), context)

    assert sorted(path.name for path in tmp_path.glob("step_*")) == []


def test_checkpoint_that_would_write_nothing_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="periodic"):
        Checkpoint(output_dir=tmp_path, periodic=False, terminal=False)


def _context() -> _CheckpointContext:
    """Build a real `RunContext` carrying resolved config and metadata."""

    cfg = OmegaConf.create(
        {
            "study": {"name": "test_study", "config_id": "lr=0.001_channels=4"},
            "model": {"_target_": "torch.nn.Linear", "in_features": 2, "out_features": 1},
            "runtime": {"device": "cpu", "dtype": "float64"},
        }
    )
    metadata = SimpleNamespace(
        run_id="run",
        device="cpu",
        dtype="float64",
        git_commit="deadbeef",
        git_branch="main",
        dirty_worktree=False,
        command="pytest",
        extra={"python_version": "3.12.0", "torch_version": torch.__version__},
    )
    # `metadata` stays a namespace: it is only ever read attribute-wise by
    # `save_checkpoint`, which annotates its context as `Any`. Only the context
    # itself has to be a real `RunContext`, because it crosses the annotated
    # typed-occurrence dispatch boundary.
    return _CheckpointContext(cfg=cfg, metadata=metadata, run_dir="/tmp/run")


def test_checkpoint_payload_uses_structured_schema(tmp_path) -> None:
    callback = Checkpoint(output_dir=tmp_path, terminal=False, every_n_steps=1)
    context = _context()

    _iteration(callback, _state(1), context)

    checkpoint_dir = tmp_path / "step_000002"
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["kind"] == "tpen.checkpoint"
    # v2 names both counters instead of one ambiguous `step`.
    assert "step" not in manifest
    assert manifest["next_iteration"] == 2
    assert manifest["completed_updates"] == 2
    assert manifest["files"]["model"] == "model.pt"
    config_hashes = checkpoint_hashes(context.cfg)
    assert {key: manifest["hashes"][key] for key in config_hashes} == config_hashes
    for component, relative in manifest["files"].items():
        assert manifest["hashes"][f"{component}_sha256"] == file_sha256(
            checkpoint_dir / relative
        )
    assert manifest["runtime"]["device"] == "cpu"
    assert manifest["runtime"]["dtype"] == "float64"
    assert manifest["runtime"]["torch_version"] == torch.__version__
    assert manifest["provenance"]["config_id"] == "lr=0.001_channels=4"
    assert manifest["provenance"]["study_name"] == "test_study"
    assert manifest["provenance"]["git_sha"] == "deadbeef"
    assert manifest["provenance"]["tpen_version"] == tpen.__version__


def test_checkpoint_fails_loudly_when_required_state_is_missing(tmp_path) -> None:
    callback = Checkpoint(output_dir=tmp_path, terminal=False, every_n_steps=1)
    state = _state(1)
    state.trainer = None

    with pytest.raises(ValueError, match="trainer"):
        _iteration(callback, state, _context())


def test_checkpoint_owns_no_legacy_scheduling_hooks(tmp_path) -> None:
    """Nothing routes this callback by an event NAME any more.

    A stale ``triggers:`` key left in a config cannot make it fire, because the
    legacy dispatch in `_CallbackCore.handle` finds no ``on_<name>`` method to
    call. Double firing is structurally impossible rather than merely
    unconfigured.
    """

    assert "should_run" not in Checkpoint.__dict__
    assert "_legacy_cadence_step" not in Checkpoint.__dict__

    callback = Checkpoint(output_dir=tmp_path)

    assert not hasattr(callback, "triggers")
    assert not hasattr(callback, "on_step_end")
    assert not hasattr(callback, "on_train_end")


def test_checkpoint_cadence_reads_completed_updates_and_nothing_else(tmp_path) -> None:
    """Only `completed_updates` gates the periodic window -- no step fallbacks."""

    callback = Checkpoint(output_dir=tmp_path, terminal=False, every_n_steps=2)

    # The counters are deliberately diverged, so this pins which one the window
    # reads: `completed_updates=3` is odd and closes the gate, while the resume
    # cursor `next_iteration=4` would have opened it. `state.step` and the
    # event's own step would also have passed a step-based gate.
    _iteration(callback, _state(2, next_iteration=4, completed_updates=3), _context())

    assert sorted(path.name for path in tmp_path.glob("step_*")) == []


def test_checkpoint_cadence_requires_a_trainer_on_state(tmp_path) -> None:
    """No `global_step`/`state.step` probing survives: the trainer is required."""

    callback = Checkpoint(output_dir=tmp_path, periodic=False, every_n_steps=2)
    state = _state(1)
    state.trainer = None

    with pytest.raises(ValueError, match="trainer"):
        _finish(callback, state, _context())
