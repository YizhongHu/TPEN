"""Fail-closed recorder binding and durable segment boundary contracts."""

from pathlib import Path

import pytest

from tpen.artifacts import RunResult
from tpen.callback.raw_training_batch_recorder import RawTrainingBatchRecorder, TrainingHistoryReader
from tpen.checkpoint import CheckpointRestored
from tpen.checkpoint.restore import RestoreReport
from tpen.events import Occurrence
from tpen.run_events import RunCompleted
from tpen.training.state import TrainerState
from tests.unit.callback.test_raw_training_batch_recorder import _record, _run_context, _trainer
from tests.unit.callback.test_raw_training_batch_recorder_v3 import _write_real_checkpoint


def _source(tmp_path: Path):
    context = _run_context(tmp_path / "source")
    root = Path(context.metadata.run_dir) / "training_history"
    recorder = RawTrainingBatchRecorder(root, format_version=3, partition_size_steps=4)
    for step in range(3):
        recorder._capture_v3(_record(step), context, TrainerState(trainer=_trainer(step + 1)), scheduled=True)
    recorder.handle_stateless_occurrence_impl(
        Occurrence(RunCompleted(status="planned_yield", result=RunResult(
            status="planned_yield", next_iteration=3, completed_updates=3, target_iterations=6,
        )), 1), context,
    )
    checkpoint = _write_real_checkpoint(
        Path(context.metadata.run_dir) / "checkpoints", step=3,
        run_id=context.metadata.run_id, run_dir=Path(context.metadata.run_dir),
    )
    return context, root, checkpoint


def _successor(tmp_path: Path, checkpoint: Path, *, every: int = 1):
    context = _run_context(tmp_path / "successor")
    root = Path(context.metadata.run_dir) / "training_history"
    recorder = RawTrainingBatchRecorder(
        root, format_version=3, partition_size_steps=4, every_n_steps=every,
    )
    recorder.handle_stateless_occurrence_impl(
        Occurrence(CheckpointRestored(RestoreReport(
            mode="train_resume", checkpoint_dir=str(checkpoint), next_iteration=3, completed_updates=3,
        )), 1), context,
    )
    return context, root, recorder


def test_partial_window_continues_without_duplicates(tmp_path: Path) -> None:
    _, source, checkpoint = _source(tmp_path)
    context, root, recorder = _successor(tmp_path, checkpoint)
    recorder._resume_v3(context, {"next_iteration": 3, "completed_updates": 3})
    recorder._capture_v3(_record(3), context, TrainerState(trainer=_trainer(4)), scheduled=True)
    recorder._capture_v3(_record(4), context, TrainerState(trainer=_trainer(5)), scheduled=True)
    recorder._finish_v3()
    reader = TrainingHistoryReader(root)
    reader.validate(require_complete=True)
    assert reader.read()["iteration_step"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert TrainingHistoryReader(source).completion == {"next_iteration": 3, "completed_updates": 3}


def test_counter_mismatch_refused_before_publication(tmp_path: Path) -> None:
    _, _, checkpoint = _source(tmp_path)
    context, root, recorder = _successor(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="counters disagree"):
        recorder._resume_v3(context, {"next_iteration": 4, "completed_updates": 3})
    assert not root.exists()


def test_cadence_mismatch_refused_before_publication(tmp_path: Path) -> None:
    _, _, checkpoint = _source(tmp_path)
    context, root, recorder = _successor(tmp_path, checkpoint, every=2)
    with pytest.raises(ValueError, match="cadence"):
        recorder._resume_v3(context, {"next_iteration": 3, "completed_updates": 3})
    assert not root.exists()


def test_corrupted_parent_refused_before_publication(tmp_path: Path) -> None:
    _, source, checkpoint = _source(tmp_path)
    bundle = next((source / "fragments").glob("*.npz"))
    bundle.write_bytes(b"corrupt")
    context, root, recorder = _successor(tmp_path, checkpoint)
    with pytest.raises(ValueError):
        recorder._resume_v3(context, {"next_iteration": 3, "completed_updates": 3})
    assert not root.exists()


def test_yield_requires_observed_boundary(tmp_path: Path) -> None:
    context = _run_context(tmp_path)
    recorder = RawTrainingBatchRecorder(tmp_path / "history", format_version=3)
    with pytest.raises(ValueError, match="did not observe"):
        recorder.handle_stateless_occurrence_impl(
            Occurrence(RunCompleted(status="planned_yield", result=RunResult(
                status="planned_yield", next_iteration=1, completed_updates=1, target_iterations=3,
            )), 1), context,
        )


def test_yield_seals_durable_partial_window(tmp_path: Path) -> None:
    _, root, _ = _source(tmp_path)
    reader = TrainingHistoryReader(root)
    reader.validate(require_complete=True)
    assert reader.completion == {"next_iteration": 3, "completed_updates": 3}
