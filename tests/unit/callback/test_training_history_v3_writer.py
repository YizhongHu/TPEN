"""Unit contracts for the library-only v3 durable writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tpen.callback.raw_training_batch_recorder import TrainingBatchReader, TrainingBatchRecord
from tpen.callback.training_history_v3_writer import TrainingHistoryV3Writer


def _contract() -> dict[str, object]:
    return {
        "coordinate": "iteration_step", "start_step": 0, "every_n_steps": 1,
        "partition_size_steps": 4, "walker_count": 2, "n_electrons": 1,
        "spatial_dim": 3, "term_names": ["kinetic"],
        "capture_semantics": "pre_update_model_forward",
    }


def _record(step: int) -> TrainingBatchRecord:
    return TrainingBatchRecord(
        iteration_step=step, completed_updates=step, update_applied=True,
        walker_index=np.arange(2, dtype="<i8"),
        positions=np.asarray([[[step, 0.0, 1.0]], [[step, 1.0, 2.0]]], dtype="<f8"),
        local_energy=np.asarray([step + .25, step + .5], dtype="<f8"),
        term_energies={"kinetic": np.asarray([1., 2.], dtype="<f8")},
        logabs=np.asarray([step + 1., step + 2.], dtype="<f8"),
        sign=np.ones(2, dtype="<f8"), finite=np.ones(2, dtype=np.bool_),
    )


def _writer(root: Path) -> TrainingHistoryV3Writer:
    return TrainingHistoryV3Writer(
        root, stream_id="11111111-1111-4111-8111-111111111111",
        run_id="writer-test", contract=_contract(),
        provenance={"config_hash": "a" * 64},
    )


def test_writer_round_trip_and_exact_write_set(tmp_path: Path) -> None:
    root = tmp_path / "history"
    writer = _writer(root)
    writer.append_capture(_record(0))
    writer.append_capture(_record(1))
    writer.complete(next_iteration=2, completed_updates=2)
    TrainingBatchReader(root).validate(require_complete=True)
    rows = TrainingBatchReader(root).read()
    np.testing.assert_array_equal(rows["iteration_step"], [0, 0, 1, 1])
    assert {path.relative_to(root).as_posix() for path in root.rglob("*")} == {
        "history.json", "events.jsonl", "fragments", "fragments/step-000000000000.npz",
        "fragments/step-000000000000.json", "fragments/step-000000000001.npz",
        "fragments/step-000000000001.json",
    }


def test_writer_refuses_missing_capture_and_poisoned_writer(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "history")
    with pytest.raises(ValueError, match="missing"):
        writer.complete(next_iteration=2, completed_updates=2)
    assert writer.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        writer.append_capture(_record(0))


def test_writer_refuses_nonmonotone_capture(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "history")
    writer.append_capture(_record(0))
    with pytest.raises(ValueError, match="strictly increasing"):
        writer.append_capture(_record(0))


def test_writer_fsync_failure_poisoning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    writer = _writer(tmp_path / "history")
    import tpen.callback.training_history_v3_writer as module

    original = module._fsync
    calls = 0

    def fail_once(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls > 2:
            raise OSError("injected fsync failure")
        original(fd)

    monkeypatch.setattr(module, "_fsync", fail_once)
    with pytest.raises(OSError, match="fsync"):
        writer.append_capture(_record(0))
    assert writer.poisoned
    assert writer.event_count == 1

