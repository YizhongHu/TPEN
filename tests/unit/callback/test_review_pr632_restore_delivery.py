"""Review probes for PR632 restored-lineage delivery and context reset."""

from __future__ import annotations

import json
import re
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.callback.raw_training_batch_recorder import RawTrainingBatchRecorder, TrainingBatchReader
from tpen.checkpoint import CheckpointRestored
from tpen.checkpoint.restore import RestoreReport
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.events import Occurrence
from tpen.run import run_from_config
from tpen.training.events import TrainingCompleted, TrainingIteration, TrainingIterationCompleted
from tpen.training.state import TrainerState


FIXTURE = Path(__file__).resolve().parents[2] / "integration" / "artifacts" / "training" / "vmc_smoke.yaml"
REFUSAL = "RawTrainingBatchRecorder refuses restored lineage before capture or publication [unsupported-callback]"


def _cfg(
    *, max_steps: int, root: Path, load: Path | None = None, with_initial_checkpoint: bool = False
) -> DictConfig:
    cfg = OmegaConf.load(FIXTURE)
    cfg.trainer.max_steps = max_steps
    cfg.run.root = str(root)
    if with_initial_checkpoint:
        cfg.callbacks.append(
            OmegaConf.create(
                {
                    "_target_": "tpen.callback.InitialCheckpoint",
                    "output_dir": "${run.dir}/checkpoints",
                }
            )
        )
    cfg.callbacks.append(
        OmegaConf.create(
            {
                "_target_": "tpen.callback.RawTrainingBatchRecorder",
                "path": str(root / "training-history"),
                "partition_size_steps": 1,
                "every_n_steps": 1,
            }
        )
    )
    if load is not None:
        cfg.runner.load = {"mode": "train_resume", "path": str(load)}
    return cfg


def _run(cfg: DictConfig, *, raise_exceptions: bool = False) -> int:
    return run_from_config(
        cfg,
        config_path=str(FIXTURE),
        command="pytest",
        raise_exceptions=raise_exceptions,
    )


def _single_run(root: Path) -> Path:
    runs = list(root.glob("vmc_smoke/*/*"))
    assert len(runs) == 1, f"expected one run directory, found {runs}"
    return runs[0]


def _checkpoint(run_dir: Path, step: int) -> Path:
    checkpoint = run_dir / "checkpoints" / f"step_{step:06d}"
    assert (checkpoint / "COMPLETE").is_file()
    return checkpoint


def _occurrences(run_dir: Path, name: str) -> list[dict]:
    return [
        record
        for record in (json.loads(line) for line in (run_dir / "occurrences.jsonl").read_text().splitlines())
        if record["event"].endswith(name)
    ]


@pytest.mark.parametrize(
    ("max_steps", "source_step"),
    [
        pytest.param(1, 0, id="restored-zero-counters"),
        pytest.param(2, 1, id="restored-nonzero-intermediate-cursor"),
        pytest.param(1, 1, id="restored-terminal-no-loop"),
    ],
)
def test_real_runner_refuses_restored_recorder_before_history_write(
    tmp_path: Path, max_steps: int, source_step: int
) -> None:
    """A real restore reaches the recorder refusal before target history creation."""

    source_root = tmp_path / "source"
    source_cfg = _cfg(max_steps=max_steps, root=source_root, with_initial_checkpoint=True)
    if source_step == 1:
        for callback in source_cfg.callbacks:
            if callback.get("_target_") == "tpen.callback.Checkpoint":
                callback.schedule.every_n = 1
    assert _run(source_cfg) == 0
    source_dir = _single_run(source_root)
    source_checkpoint = _checkpoint(source_dir, source_step)

    target_root = tmp_path / "target"
    target_cfg = _cfg(max_steps=max_steps, root=target_root, load=source_checkpoint)
    with pytest.raises(ValueError, match=re.escape(REFUSAL)):
        _run(target_cfg, raise_exceptions=True)

    target_dir = _single_run(target_root)
    restored = _occurrences(target_dir, "CheckpointRestored")
    assert len(restored) == 1
    report = restored[0]["fields"]["report"]
    assert report["mode"] == "train_resume"
    assert report["next_iteration"] == source_step
    assert report["completed_updates"] == source_step
    assert report["loaded_model"] is True
    assert report["loaded_optimizer"] is True
    assert report["loaded_trainer"] is True
    assert report["loaded_sampler"] is True
    assert report["loaded_rng"] is True
    assert not (target_dir / "training-history").exists()


def _context(tmp_path: Path, name: str, callback: RawTrainingBatchRecorder) -> RunContext:
    manager = ArtifactManager(tmp_path / name, "callback", "review", name, layout="flat")
    manager.make_dirs()
    return RunContext(
        cfg=OmegaConf.create({}),
        source_cfg=OmegaConf.create({}),
        artifact_manager=manager,
        metadata=RunMetadata(
            run_id=name,
            run_name=name,
            timestamp="2026-01-01T00:00:00+00:00",
            timezone="UTC",
            git_commit="review",
            git_branch="review",
            dirty_worktree=False,
            command="pytest",
            config_path="review.yaml",
            resolved_config_path=str(manager.path("resolved_config.yaml")),
            run_dir=str(manager.run_dir),
            device="cpu",
            dtype="float64",
        ),
        clock=RunClock(timezone="UTC", tzinfo=UTC),
        callbacks=[callback],
    )


def _restored_occurrence() -> CheckpointRestored:
    return CheckpointRestored(
        RestoreReport(
            mode="train_resume",
            checkpoint_dir="/source/checkpoint",
            next_iteration=3,
            completed_updates=2,
            loaded_model=True,
            loaded_optimizer=True,
            loaded_trainer=True,
            loaded_sampler=True,
            loaded_rng=True,
        )
    )


def _state() -> TrainerState:
    return TrainerState(
        trainer=SimpleNamespace(
            next_iteration=0,
            completed_updates=0,
            state_dict=lambda: {"next_iteration": 0, "completed_updates": 0},
        ),
        batch=ElectronBatch(torch.tensor([[[0.0, 1.0, 2.0]]])),
        local_energy=torch.tensor([1.0]),
        local_energy_terms={"kinetic": torch.tensor([1.0])},
        wavefunction_output=WavefunctionOutput(torch.tensor([0.0]), torch.ones(1)),
        optimizer_step=False,
    )


def test_recorder_reset_after_real_context_emit_restores_fresh_delivery(tmp_path: Path) -> None:
    """A reused recorder refuses in context A, then writes in fresh context B."""

    history_root = tmp_path / "training-history"
    callback = RawTrainingBatchRecorder(history_root, partition_size_steps=1, every_n_steps=1)
    context_a = _context(tmp_path, "context-a", callback)
    context_a.emit(_restored_occurrence())
    with pytest.raises(ValueError, match=re.escape(REFUSAL)):
        with context_a.scope(TrainingIteration(0), state=_state()):
            pass

    context_b = _context(tmp_path, "context-b", callback)
    state = _state()
    with context_b.scope(TrainingIteration(0), state=state):
        pass
    context_b.emit(TrainingIterationCompleted(TrainingIteration(0)), state=state)
    context_b.emit(TrainingCompleted(), state=state)

    rows = TrainingBatchReader(history_root).read()
    assert rows["iteration_step"].tolist() == [0]
