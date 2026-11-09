"""Focused T1 probes for the raw training-history recorder."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from tpen.artifacts import ArtifactManager, RunClock, RunContext, RunMetadata
from tpen.callback.raw_training_batch_recorder import (
    RawTrainingBatchRecorder,
    TrainingBatchPartitionWriter,
    TrainingBatchReader,
    TrainingBatchRecord,
)
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.events import Occurrence
from tpen.training.events import TrainingCompleted, TrainingIteration, TrainingIterationCompleted
from tpen.training.state import TrainerState
from tpen.training.trainer import VMCTrainer
from tests.helpers.hooke_models import build_tiny_hamiltonian_terms, build_tiny_sampler, build_tiny_spenn


def _context(tmp_path: Path, callbacks: list[object] | None = None) -> RunContext:
    manager = ArtifactManager(tmp_path / "artifacts", "review", "unit", "run", layout="flat")
    manager.make_dirs()
    return RunContext(
        cfg=OmegaConf.create({"trainer": {"max_steps": 2}}),
        source_cfg=OmegaConf.create({}),
        artifact_manager=manager,
        metadata=RunMetadata(
            run_id="run",
            run_name="run",
            timestamp="2026-01-01T00:00:00+00:00",
            timezone="UTC",
            git_commit="review-sha",
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
        callbacks=callbacks or [],
    )


def _record(step: int, *, applied: bool = True) -> TrainingBatchRecord:
    return TrainingBatchRecord(
        iteration_step=step,
        completed_updates=step + int(applied),
        update_applied=applied,
        walker_index=np.arange(2),
        positions=np.asarray([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]]),
        local_energy=np.asarray([step + 0.25, step + 0.5]),
        term_energies={"kinetic": np.asarray([1.0, 2.0]), "potential": np.asarray([-0.5, -0.25])},
        logabs=np.asarray([10.0 + step, 11.0 + step]),
        sign=np.ones(2),
        finite=np.ones(2, dtype=bool),
    )


def _state(step: int) -> TrainerState:
    return TrainerState(
        trainer=SimpleNamespace(completed_updates=int(step > 0)),
        batch=ElectronBatch(torch.tensor([[[0.0, 1.0, 2.0]], [[3.0, 4.0, 5.0]]])),
        local_energy=torch.tensor([step + 0.25, step + 0.5]),
        local_energy_terms={"kinetic": torch.tensor([1.0, 2.0]), "potential": torch.tensor([-0.5, -0.25])},
        wavefunction_output=WavefunctionOutput(torch.tensor([10.0 + step, 11.0 + step]), torch.ones(2)),
        optimizer_step=step > 0,
    )


def _emit_steps(callback: RawTrainingBatchRecorder, context: RunContext, steps: list[int]) -> None:
    for step in steps:
        callback.handle_occurrence(
            Occurrence(TrainingIterationCompleted(TrainingIteration(step)), 1),
            context,
            _state(step),
        )
    callback.handle_occurrence(Occurrence(TrainingCompleted(), 1), context, _state(steps[-1]))


@pytest.mark.parametrize(
    ("start_step", "every_n_steps", "partition_size_steps", "steps"),
    [
        (0, 3, 5, [0, 3, 6, 9]),
        (1, 2, 4, [1, 3, 5, 7]),
        (0, 2, 4, [0, 2, 4, 6]),
    ],
)
def test_cadence_crossing_partitions_emits_scheduled_steps(
    tmp_path: Path,
    start_step: int,
    every_n_steps: int,
    partition_size_steps: int,
    steps: list[int],
) -> None:
    """A cadence must survive raw partition boundaries, including an offset."""

    callback = RawTrainingBatchRecorder(
        tmp_path,
        partition_size_steps=partition_size_steps,
        every_n_steps=every_n_steps,
        start_step=start_step,
    )
    _emit_steps(callback, _context(tmp_path), steps)
    rows = TrainingBatchReader(tmp_path).read()
    np.testing.assert_array_equal(np.unique(rows["iteration_step"]), steps)


def _publish_three(root: Path) -> None:
    for index in range(3):
        writer = TrainingBatchPartitionWriter(root, partition_index=index, planned_steps=[index])
        writer.append(_record(index))
        writer.publish()


def test_reader_roundtrip_refuses_broken_manifest_parent_hash(tmp_path: Path) -> None:
    _publish_three(tmp_path)
    TrainingBatchReader(tmp_path).validate()
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["partitions"][1]["parent_manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="lineage"):
        TrainingBatchReader(tmp_path).validate()


def test_reader_refuses_removed_middle_manifest_entry(tmp_path: Path) -> None:
    _publish_three(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["partitions"].pop(1)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="lineage"):
        TrainingBatchReader(tmp_path).validate()


def test_reader_refuses_duplicate_iteration_walker_grid_across_partitions(tmp_path: Path) -> None:
    for index in range(2):
        writer = TrainingBatchPartitionWriter(tmp_path, partition_index=index, planned_steps=[0])
        writer.append(_record(0))
        writer.publish()
    with pytest.raises(ValueError, match="duplicate"):
        TrainingBatchReader(tmp_path).validate()


def test_real_trainer_capture_matches_pre_update_model_and_differs_after_update(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    callback = RawTrainingBatchRecorder(history_root, partition_size_steps=10, every_n_steps=1)
    context = _context(tmp_path, [callback])
    torch.manual_seed(20261109)
    model = build_tiny_spenn()
    pre_model = build_tiny_spenn()
    pre_model.load_state_dict(model.state_dict())
    sampler = build_tiny_sampler()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Keep the oracle on the first pre-update batch; later steps legitimately
    # use parameters changed by the optimizer and need a separate pre-state.
    trainer = VMCTrainer(max_steps=1, log_every_n_steps=1, return_terms=True)
    trainer.fit(
        model=model,
        sampler=sampler,
        hamiltonian_terms=build_tiny_hamiltonian_terms(),
        optimizer=optimizer,
        context=context,
        emit=lambda *args, **kwargs: None,
    )
    callback.close(context)

    rows = TrainingBatchReader(history_root).read()
    assert np.any(rows["update_applied"])
    assert int(rows["completed_updates"].max()) >= 1
    positions = torch.as_tensor(rows["positions"], dtype=torch.float64)
    spins = torch.tensor([[1.0, -1.0]] * len(rows["positions"]), dtype=torch.float64)
    batch = ElectronBatch(positions=positions, spins=spins)
    pre_model.eval()
    model.eval()
    with torch.no_grad():
        pre_logabs = pre_model(batch).logabs.cpu().numpy()
        post_logabs = model(batch).logabs.cpu().numpy()
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(rows["logabs"], pre_logabs + 1.0, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(rows["logabs"], pre_logabs, rtol=1e-10, atol=1e-10)
    assert not np.allclose(pre_logabs, post_logabs, rtol=1e-10, atol=1e-10)
