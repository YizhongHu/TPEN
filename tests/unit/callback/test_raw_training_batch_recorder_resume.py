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


def test_v3_unscheduled_step_skips_host_transfer(tmp_path: Path) -> None:
    from types import SimpleNamespace
    import numpy as np
    from tpen.training.events import TrainingIteration, TrainingIterationCompleted

    class _NoCpuTensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

        def reshape(self, *shape: int):
            if shape == (-1,):
                total = 1
                for dim in self.shape:
                    total *= dim
                return _NoCpuTensor((total,))
            return _NoCpuTensor(shape)

        def detach(self):
            return self

        def cpu(self):
            raise AssertionError("unscheduled v3 step must not copy tensor to host")

    context = _run_context(tmp_path)
    root = Path(context.metadata.run_dir) / "training_history"
    recorder = RawTrainingBatchRecorder(root, format_version=3, every_n_steps=2, partition_size_steps=4)
    recorder._capture_v3(_record(0), context, TrainerState(trainer=_trainer(1)), scheduled=True)

    unscheduled_state = TrainerState(
        trainer=_trainer(2),
        batch=SimpleNamespace(positions=_NoCpuTensor((2, 2, 3)), n_electrons=2, spatial_dim=3),
        local_energy=_NoCpuTensor((2,)),
        local_energy_terms={"kinetic": _NoCpuTensor((2,))},
        wavefunction_output=SimpleNamespace(logabs=_NoCpuTensor((2,)), sign=_NoCpuTensor((2,))),
    )
    recorder.handle_occurrence_impl(
        Occurrence(TrainingIterationCompleted(TrainingIteration(step=1)), 1),
        context,
        unscheduled_state,
    )
    assert recorder._v3_counters == (2, 2)
    reader = TrainingHistoryReader(root)
    assert set(reader.read()["iteration_step"].tolist()) == {0}


def test_invalid_segments_path_refused(tmp_path: Path) -> None:
    """Symlinked or non-directory segments paths must fail closed, never drop successor rows."""
    import shutil

    _, source, checkpoint = _source(tmp_path)
    recorder = RawTrainingBatchRecorder(source, format_version=3, partition_size_steps=4)
    context = _run_context(tmp_path / "source")
    recorder._restore_report = RestoreReport(mode="train_resume", checkpoint_dir=checkpoint)
    recorder._resume_v3(context, {"next_iteration": 3, "completed_updates": 3})
    recorder._capture_v3(_record(3), context, TrainerState(trainer=_trainer(4)), scheduled=True)
    recorder.handle_stateless_occurrence_impl(
        Occurrence(RunCompleted(status="completed", result=RunResult(
            status="completed", next_iteration=4, completed_updates=4, target_iterations=4,
        )), 1), context,
    )
    seg_dir = source / "segments"
    preserved = tmp_path / "preserved_segments"
    shutil.move(str(seg_dir), str(preserved))

    # Symlinked segments directory (valid and dangling) must be refused.
    seg_dir.symlink_to(preserved, target_is_directory=True)
    with pytest.raises(ValueError, match="segments directory"):
        TrainingHistoryReader(source).read()
    with pytest.raises(ValueError, match="segments directory"):
        TrainingHistoryReader(source).validate(require_complete=True)
    seg_dir.unlink()

    seg_dir.symlink_to(tmp_path / "missing_target", target_is_directory=True)
    with pytest.raises(ValueError, match="segments directory"):
        TrainingHistoryReader(source).read()
    seg_dir.unlink()

    # Non-directory segments path must be refused.
    seg_dir.write_text("not a dir", encoding="utf-8")
    with pytest.raises(ValueError, match="segments directory"):
        TrainingHistoryReader(source).read()
    seg_dir.unlink()

    # Symlinked segment child must be refused rather than falling back to parent.
    seg_dir.mkdir()
    child_target = next(preserved.iterdir())
    (seg_dir / child_target.name).symlink_to(child_target, target_is_directory=True)
    with pytest.raises(ValueError, match="segment entry"):
        TrainingHistoryReader(source).read()


def test_v2_preserves_term_insertion_order_while_v3_sorts(tmp_path: Path) -> None:
    """Default v2 keeps insertion order for term names; opt-in v3 canonicalizes sorted order."""
    import json
    from types import SimpleNamespace
    import numpy as np
    from tpen.training.events import TrainingIteration, TrainingIterationCompleted

    class _FakeTensor:
        def __init__(self, array: np.ndarray) -> None:
            self._array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return self._array

    state = TrainerState(
        trainer=_trainer(1),
        batch=SimpleNamespace(positions=_FakeTensor(np.zeros((2, 2, 3), dtype=np.float64)), n_electrons=2, spatial_dim=3),
        local_energy=_FakeTensor(np.zeros((2,), dtype=np.float64)),
        local_energy_terms={
            "potential": _FakeTensor(np.ones((2,), dtype=np.float64)),
            "kinetic": _FakeTensor(np.full((2,), 2.0, dtype=np.float64)),
        },
        wavefunction_output=SimpleNamespace(
            logabs=_FakeTensor(np.zeros((2,), dtype=np.float64)),
            sign=_FakeTensor(np.ones((2,), dtype=np.float64)),
        ),
    )

    v2_ctx = _run_context(tmp_path / "v2")
    v2_root = Path(v2_ctx.metadata.run_dir) / "training_history"
    v2_rec = RawTrainingBatchRecorder(v2_root)
    v2_rec.handle_occurrence_impl(Occurrence(TrainingIterationCompleted(TrainingIteration(step=0)), 1), v2_ctx, state)
    v2_rec.close(v2_ctx)
    v2_manifest = json.loads((v2_root / "manifest.json").read_text(encoding="utf-8"))
    v2_sidecar = json.loads((v2_root / v2_manifest["partitions"][0]["sidecar"]).read_text(encoding="utf-8"))
    assert v2_sidecar["term_names"] == ["potential", "kinetic"]

    v3_ctx = _run_context(tmp_path / "v3")
    v3_root = Path(v3_ctx.metadata.run_dir) / "training_history"
    v3_rec = RawTrainingBatchRecorder(v3_root, format_version=3)
    v3_rec.handle_occurrence_impl(Occurrence(TrainingIterationCompleted(TrainingIteration(step=0)), 1), v3_ctx, state)
    v3_rec.handle_stateless_occurrence_impl(
        Occurrence(RunCompleted(status="completed", result=RunResult(
            status="completed", next_iteration=1, completed_updates=1, target_iterations=1,
        )), 1), v3_ctx,
    )
    v3_header = json.loads((v3_root / "history.json").read_text(encoding="utf-8"))
    assert v3_header["contract"]["term_names"] == ["kinetic", "potential"]
    assert set(TrainingHistoryReader(v3_root).read()["iteration_step"].tolist()) == {0}
