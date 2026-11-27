"""Unit contracts for the library-only v3 durable writer."""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest

from tpen.callback.raw_training_batch_recorder import TrainingBatchReader, TrainingBatchRecord
from tpen.callback.training_history_v3_writer import EVENTS_FILENAME, TrainingHistoryV3Writer
from tests.unit.callback.test_raw_training_batch_recorder_v3 import _pin, _write_real_checkpoint, _write_v3_fixture


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


def test_writer_accepts_symlinked_caller_root_and_uses_canonical_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real" / "history"
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_root.parent, target_is_directory=True)

    writer = _writer(alias_parent / "history")

    assert writer.root == real_root
    writer.append_capture(_record(0))
    writer.complete(next_iteration=1, completed_updates=1)
    TrainingBatchReader(alias_parent / "history").validate(require_complete=True)


def test_writer_refuses_symlink_below_canonical_root(tmp_path: Path) -> None:
    root = tmp_path / "history"
    root.mkdir()
    target = tmp_path / "outside-events.jsonl"
    target.write_bytes(b"")
    (root / EVENTS_FILENAME).symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        _writer(root)


def test_writer_matches_r1_fixture_bytes_for_shared_scenario(tmp_path: Path) -> None:
    stream_id = "32323232-3232-4232-8232-323232323232"
    fixture = tmp_path / "fixture"
    _write_v3_fixture(fixture, stream_id=stream_id, steps=[0, 1], complete=2)
    writer = TrainingHistoryV3Writer(
        tmp_path / "writer", stream_id=stream_id, run_id=f"run-{stream_id[:8]}",
        contract=_contract(), provenance={"config_hash": "3" * 64},
    )
    writer.append_capture(_record(0))
    writer.append_capture(_record(1))
    writer.complete(next_iteration=2, completed_updates=2)
    assert (fixture / "history.json").read_bytes() == (writer.root / "history.json").read_bytes()
    assert (fixture / "events.jsonl").read_bytes() == (writer.root / "events.jsonl").read_bytes()
    for name in ("step-000000000000.npz", "step-000000000000.json", "step-000000000001.npz", "step-000000000001.json"):
        assert (fixture / "fragments" / name).read_bytes() == (writer.root / "fragments" / name).read_bytes()


def test_writer_refuses_missing_capture_and_poisoned_writer(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "history")
    with pytest.raises(ValueError, match="missing"):
        writer.complete(next_iteration=2, completed_updates=2)
    assert writer.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        writer.append_capture(_record(0))


def _resume_stub(root: Path, *, contract: dict[str, object], cut: int, frontier: int = 0) -> TrainingHistoryV3Writer:
    return TrainingHistoryV3Writer(
        root, stream_id="34343434-3434-4434-8434-343434343434", run_id="resume-test",
        contract=contract, origin="train_resume", supersede_from_step=cut,
        inherited_seal_frontier=frontier,
        checkpoint={"path": str((root.parent / "checkpoint").resolve()), "manifest_sha256": "a" * 64,
                    "model_sha256": "b" * 64, "next_iteration": cut, "completed_updates": cut},
    )


def test_resume_cadence_phase_and_inherited_full_seal_frontier(tmp_path: Path) -> None:
    contract = _contract() | {"every_n_steps": 50, "partition_size_steps": 1000}
    parent = tmp_path / "cadence-parent"
    parent_writer = TrainingHistoryV3Writer(parent, stream_id="34343434-3434-4434-8434-343434343434", run_id="resume-parent", contract=contract, provenance={"config_hash": "a" * 64})
    for step in range(0, 473, 50):
        parent_writer.append_capture(_record(step))
    parent_writer.complete(next_iteration=473, completed_updates=473)
    parent_root, parent_count, parent_prefix = _pin(parent)
    checkpoint = _write_real_checkpoint(tmp_path / "checkpoint", step=473, run_id="resume-parent", run_dir=parent.parent)
    checkpoint_data = {"path": str(checkpoint.resolve()), "manifest_sha256": hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest(), "model_sha256": hashlib.sha256((checkpoint / "model.pt").read_bytes()).hexdigest(), "next_iteration": 473, "completed_updates": 473}
    writer = TrainingHistoryV3Writer(tmp_path / "cadence", stream_id="35353535-3535-4535-8535-353535353535", run_id="cadence", contract=contract, origin="train_resume", parent={"root": str(parent_root.resolve()), "event_count": parent_count, "prefix_sha256": parent_prefix}, checkpoint=checkpoint_data, supersede_from_step=473, provenance={"config_hash": "b" * 64})
    for step in range(500, 1000, 50):
        writer.append_capture(_record(step))
    writer.seal_window(0)
    writer.complete(next_iteration=1000, completed_updates=10)
    TrainingBatchReader(writer.root).validate(require_complete=True)
    np.testing.assert_array_equal(TrainingBatchReader(writer.root).read()["iteration_step"], np.repeat(np.arange(0, 1000, 50), 2))

    parent = tmp_path / "cadence-short-parent"
    parent_writer = TrainingHistoryV3Writer(parent, stream_id="36363636-3636-4636-8636-363636363636", run_id="resume-parent-short", contract=contract, provenance={"config_hash": "c" * 64})
    for step in range(0, 473, 50):
        parent_writer.append_capture(_record(step))
    parent_writer.complete(next_iteration=473, completed_updates=473)
    parent_root, parent_count, parent_prefix = _pin(parent)
    checkpoint = _write_real_checkpoint(tmp_path / "checkpoint-short", step=473, run_id="resume-parent-short", run_dir=parent.parent)
    checkpoint_data = {"path": str(checkpoint.resolve()), "manifest_sha256": hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest(), "model_sha256": hashlib.sha256((checkpoint / "model.pt").read_bytes()).hexdigest(), "next_iteration": 473, "completed_updates": 473}
    writer = TrainingHistoryV3Writer(tmp_path / "cadence-short", stream_id="37373737-3737-4737-8737-373737373737", run_id="cadence-short", contract=contract, origin="train_resume", parent={"root": str(parent_root.resolve()), "event_count": parent_count, "prefix_sha256": parent_prefix}, checkpoint=checkpoint_data, supersede_from_step=473, provenance={"config_hash": "d" * 64})
    for step in range(500, 725, 50):
        writer.append_capture(_record(step))
    writer.complete(next_iteration=725, completed_updates=5)
    TrainingBatchReader(writer.root).validate(require_complete=True)
    np.testing.assert_array_equal(TrainingBatchReader(writer.root).read()["iteration_step"], np.repeat(np.arange(0, 725, 50), 2))

    parent = tmp_path / "inherited-parent"
    parent_writer = TrainingHistoryV3Writer(parent, stream_id="38383838-3838-4838-8838-383838383838", run_id="resume-parent-inherited", contract=_contract(), provenance={"config_hash": "e" * 64})
    for step in range(4):
        parent_writer.append_capture(_record(step))
    parent_writer.seal_window(0)
    parent_writer.complete(next_iteration=4, completed_updates=4)
    parent_root, parent_count, parent_prefix = _pin(parent)
    checkpoint = _write_real_checkpoint(tmp_path / "checkpoint-inherited", step=4, run_id="resume-parent-inherited", run_dir=parent.parent)
    checkpoint_data = {"path": str(checkpoint.resolve()), "manifest_sha256": hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest(), "model_sha256": hashlib.sha256((checkpoint / "model.pt").read_bytes()).hexdigest(), "next_iteration": 4, "completed_updates": 4}
    small = TrainingHistoryV3Writer(tmp_path / "inherited", stream_id="39393939-3939-4939-8939-393939393939", run_id="inherited", contract=_contract(), origin="train_resume", parent={"root": str(parent_root.resolve()), "event_count": parent_count, "prefix_sha256": parent_prefix}, checkpoint=checkpoint_data, supersede_from_step=4, inherited_seal_frontier=1, provenance={"config_hash": "f" * 64})
    small.complete(next_iteration=4, completed_updates=4)
    TrainingBatchReader(small.root).validate(require_complete=True)
    np.testing.assert_array_equal(TrainingBatchReader(small.root).read()["iteration_step"], np.repeat(np.arange(0, 4), 2))
    continuing = TrainingHistoryV3Writer(tmp_path / "continuing", stream_id="40404040-4040-4040-8040-404040404040", run_id="continuing", contract=_contract(), origin="train_resume", parent={"root": str(parent_root.resolve()), "event_count": parent_count, "prefix_sha256": parent_prefix}, checkpoint=checkpoint_data, supersede_from_step=4, inherited_seal_frontier=1, provenance={"config_hash": "1" * 64})
    continuing.append_capture(_record(4))
    continuing.complete(next_iteration=5, completed_updates=5)
    TrainingBatchReader(continuing.root).validate(require_complete=True)
    np.testing.assert_array_equal(TrainingBatchReader(continuing.root).read()["iteration_step"], np.repeat(np.arange(0, 5), 2))

    phased = _resume_stub(tmp_path / "phase", contract=_contract() | {"start_step": 1, "every_n_steps": 2}, cut=4)
    assert phased._expected_capture_count(0, 10) == 3


def test_writer_refuses_nonmonotone_capture(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "history")
    writer.append_capture(_record(2))
    with pytest.raises(ValueError, match="strictly increasing"):
        writer.append_capture(_record(1))


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
    with pytest.raises(RuntimeError, match="poisoned"):
        writer.append_capture(_record(0))


@pytest.mark.parametrize("variant", ["complete", "torn", "orphan"])
def test_nonempty_root_refused_in_all_states(tmp_path: Path, variant: str) -> None:
    root = tmp_path / variant
    if variant == "complete":
        _write_v3_fixture(root, stream_id="22222222-2222-4222-8222-222222222222", complete=0)
    elif variant == "torn":
        _write_v3_fixture(root, stream_id="23232323-2323-4232-8232-232323232323")
        (root / "events.jsonl").write_bytes((root / "events.jsonl").read_bytes() + b'{"type":"complete"}')
    else:
        root.mkdir(parents=True)
        (root / "fragments").mkdir()
        (root / "fragments" / "orphan.npz").write_bytes(b"orphan")
    with pytest.raises(FileExistsError, match="fresh, empty, non-symlink"):
        _writer(root)


def test_nonempty_root_without_fragments_is_refused_by_fresh_root_guard(tmp_path: Path) -> None:
    # A stray file and no fragments/ directory: only the fresh-root guard can
    # refuse this root (fragments.mkdir would otherwise succeed).
    root = tmp_path / "stray"
    root.mkdir()
    (root / "stray.txt").write_bytes(b"stray")
    with pytest.raises(FileExistsError, match="fresh, empty, non-symlink"):
        _writer(root)
    assert sorted(path.name for path in root.iterdir()) == ["stray.txt"]

def test_empty_root_and_symlinked_caller_root_are_accepted(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    _writer(empty)
    target = tmp_path / "target"
    target.mkdir()
    component = tmp_path / "component"
    component.symlink_to(target, target_is_directory=True)
    writer = _writer(component / "history")
    assert writer.root == target / "history"

    (writer.root / EVENTS_FILENAME).unlink()
    outside = tmp_path / "outside-events.jsonl"
    outside.write_bytes(b"")
    (writer.root / EVENTS_FILENAME).symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        _writer(writer.root)


def test_root_creation_fsyncs_new_directory_entries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tpen.callback.training_history_v3_writer as module
    trace: list[str] = []
    original = module._fsync_directory
    monkeypatch.setattr(module, "_fsync_directory", lambda path: (trace.append(path.name), original(path))[1])
    _writer(tmp_path / "missing" / "history")
    assert trace[:3] == [tmp_path.name, "missing", "history"]


def test_root_creation_directory_fsync_failure_propagates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tpen.callback.training_history_v3_writer as module
    original = module._fsync_directory

    def fail_parent(path: Path) -> None:
        if path == tmp_path / "missing":
            raise OSError("injected parent directory fsync failure")
        original(path)

    monkeypatch.setattr(module, "_fsync_directory", fail_parent)
    with pytest.raises(OSError, match="parent directory fsync"):
        _writer(tmp_path / "missing" / "history")


@pytest.mark.parametrize("mutation", ["finite", "completed_updates", "positions", "walker_index"])
def test_append_admission_rejects_mutated_record_without_log_write(tmp_path: Path, mutation: str) -> None:
    writer = _writer(tmp_path / mutation)
    record = _record(0)
    before = (writer.root / "events.jsonl").read_bytes()
    if mutation == "finite":
        record.finite[0] = False
    elif mutation == "completed_updates":
        object.__setattr__(record, "completed_updates", 2)
    elif mutation == "walker_index":
        record.walker_index[:] = 0
    else:
        record.positions[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        writer.append_capture(record)
    assert writer.poisoned
    assert (writer.root / "events.jsonl").read_bytes() == before


def test_complete_rejects_negative_and_extra_seals(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "negative")
    with pytest.raises(ValueError, match="invalid or duplicate"):
        writer.seal_window(-1)
    writer = TrainingHistoryV3Writer(
        tmp_path / "extra", stream_id="35353535-3535-4535-8535-353535353535",
        run_id="extra", contract=_contract() | {"start_step": 100}, provenance={"config_hash": "5" * 64},
    )
    writer.seal_window(0)
    with pytest.raises(ValueError, match="outside"):
        writer.complete(next_iteration=0, completed_updates=0)


def test_resume_origin_written_by_writer_and_read_by_r1(tmp_path: Path) -> None:
    parent = tmp_path / "parent" / "core" / "training_history"
    parent.parent.mkdir(parents=True)
    parent_writer = TrainingHistoryV3Writer(
        parent, stream_id="24242424-2424-4242-8242-242424242424",
        run_id="parent-run", contract=_contract(), provenance={"config_hash": "a" * 64},
    )
    parent_writer.append_capture(_record(0))
    parent_writer.complete(next_iteration=1, completed_updates=1)
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint = _write_real_checkpoint(checkpoint_root, step=1, run_id="parent-run", run_dir=parent.parent)
    child = tmp_path / "child"
    parent_root, parent_count, parent_prefix = _pin(parent)
    manifest = checkpoint / "manifest.json"
    checkpoint_data = {
        "path": str(checkpoint.resolve()),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((checkpoint / "model.pt").read_bytes()).hexdigest(),
        "next_iteration": 1, "completed_updates": 1,
    }
    child_writer = TrainingHistoryV3Writer(
        child, stream_id="25252525-2525-4252-8252-252525252525", run_id="child-run",
        contract=_contract(), origin="train_resume",
        parent={"root": str(parent_root.resolve()), "event_count": parent_count, "prefix_sha256": parent_prefix},
        checkpoint=checkpoint_data, supersede_from_step=1, provenance={"config_hash": "b" * 64},
    )
    assert json.loads((child / "events.jsonl").read_text())["type"] == "resume"
    child_writer.complete(next_iteration=1, completed_updates=1)
    TrainingBatchReader(child).validate(require_complete=True)


def test_creation_and_capture_order_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tpen.callback.training_history_v3_writer as module

    trace: list[str] = []
    original_private, original_file_fsync, original_publish, original_dir, original_log, original_log_fsync = (
        module._write_private, module._fsync_file, module._publish_exclusive,
        module._fsync_directory, module._append_log_line, module._fsync_log
    )
    def private(path: Path, data: bytes, *, fsync: bool = True) -> None:
        # Private objects are fsynced before exclusive publication under a
        # dot-prefixed temporary name; record the eventual object name so the
        # creation ordering assertion observes the origin fsyncs.
        name = path.name
        if fsync:
            trace.append(name + " fsync")
        original_private(path, data, fsync=fsync)
    def file_fsync(path: Path) -> None:
        trace.append(path.name.removeprefix(".").removesuffix(".tmp") + " fsync")
        original_file_fsync(path)
    def publish(temp: Path, final: Path) -> None:
        trace.append("publish " + final.name)
        original_publish(temp, final)
    def directory(path: Path) -> None:
        trace.append("dir " + path.name)
        original_dir(path)
    def log(path: Path, line: bytes) -> None:
        trace.append("log append")
        original_log(path, line)
    def log_fsync(fd: int) -> None:
        trace.append("log fsync")
        original_log_fsync(fd)
    monkeypatch.setattr(module, "_write_private", private)
    monkeypatch.setattr(module, "_fsync_file", file_fsync)
    monkeypatch.setattr(module, "_publish_exclusive", publish)
    monkeypatch.setattr(module, "_fsync_directory", directory)
    monkeypatch.setattr(module, "_append_log_line", log)
    monkeypatch.setattr(module, "_fsync_log", log_fsync)
    writer = _writer(tmp_path / "history")
    creation = list(trace)
    assert creation[0].startswith("dir ")
    assert creation[1:] == ["dir history", "history.json fsync", "events.jsonl fsync", "dir history", "log append", "log fsync", "dir history"]
    trace.clear()
    writer.append_capture(_record(0))
    capture = trace
    assert capture.count("step-000000000000.npz fsync") == 1
    assert capture.count("step-000000000000.json fsync") == 1
    assert capture.count("dir fragments") == 1
    assert capture.count("log fsync") == 1
    assert capture.index("step-000000000000.npz fsync") < capture.index("step-000000000000.json fsync")
    assert capture.index("step-000000000000.json fsync") < capture.index("publish step-000000000000.npz")
    assert capture.index("publish step-000000000000.json") < capture.index("dir fragments")
    assert capture.index("dir fragments") < capture.index("log append") < capture.index("log fsync")


@pytest.mark.parametrize("point", ["npz", "sidecar", "fragments", "log"])
def test_each_fsync_point_poisoning_preserves_committed_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, point: str) -> None:
    import tpen.callback.training_history_v3_writer as module
    writer = _writer(tmp_path / point)
    original = module._fsync
    count = 0
    def fail(fd: int) -> None:
        nonlocal count
        count += 1
        if (point == "npz" and count == 1) or (point == "sidecar" and count == 2) or (point == "fragments" and count == 3) or (point == "log" and count == 4):
            raise OSError(point)
        original(fd)
    monkeypatch.setattr(module, "_fsync", fail)
    with pytest.raises(OSError, match=point):
        writer.append_capture(_record(0))
    assert writer.poisoned and writer.event_count == 1
    TrainingBatchReader(writer.root).validate()


def test_publish_collision_and_mid_npz_failure_poison(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tpen.callback.training_history_v3_writer as module
    writer = _writer(tmp_path / "collision")
    final = writer.fragments / "step-000000000000.npz"
    final.write_bytes(b"collision")
    with pytest.raises(FileExistsError, match="exclusive publication collision"):
        writer.append_capture(_record(0))
    assert writer.poisoned and writer.event_count == 1
    assert final.read_bytes() == b"collision"
    writer = _writer(tmp_path / "npz")
    original = module._npz_bytes
    monkeypatch.setattr(module, "_npz_bytes", lambda record: (_ for _ in ()).throw(RuntimeError("mid-npz")))
    with pytest.raises(RuntimeError, match="mid-npz"):
        writer.append_capture(_record(0))
    assert writer.poisoned and writer.event_count == 1
    assert (writer.root / "events.jsonl").read_bytes().count(b"\n") == 1
    monkeypatch.setattr(module, "_npz_bytes", original)


def test_short_os_write_is_completed_and_reader_validates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tpen.callback.training_history_v3_writer as module
    original = module.os.write
    shortened = False
    def short(fd: int, data: bytes) -> int:
        nonlocal shortened
        if not shortened and len(data) > 10:
            shortened = True
            return original(fd, data[:-1])
        return original(fd, data)
    monkeypatch.setattr(module.os, "write", short)
    writer = _writer(tmp_path / "short")
    writer.append_capture(_record(0))
    assert shortened
    TrainingBatchReader(writer.root).validate()


def test_writer_matches_fixture_reader_view(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    _write_v3_fixture(fixture, stream_id="28282828-2828-4282-8282-282828282828", steps=[0, 1])
    writer = _writer(tmp_path / "writer")
    writer.append_capture(_record(0))
    writer.append_capture(_record(1))
    fixture_rows = TrainingBatchReader(fixture).read()
    writer_rows = TrainingBatchReader(writer.root).read()
    assert fixture_rows.keys() == writer_rows.keys()
    for name in fixture_rows:
        np.testing.assert_array_equal(fixture_rows[name], writer_rows[name])


def test_seal_complete_and_contract_refusals(tmp_path: Path) -> None:
    incomplete = _writer(tmp_path / "incomplete")
    incomplete.append_capture(_record(0))
    with pytest.raises(ValueError, match="coverage"):
        incomplete.seal_window(0)

    writer = _writer(tmp_path / "seals")
    writer.append_capture(_record(0))
    writer.append_capture(_record(1))
    writer.append_capture(_record(2))
    writer.append_capture(_record(3))
    writer.seal_window(0)
    with pytest.raises(ValueError, match="invalid or duplicate"):
        writer.seal_window(0)

    writer = _writer(tmp_path / "complete")
    for step in range(4):
        writer.append_capture(_record(step))
    writer.seal_window(0)
    writer.complete(next_iteration=4, completed_updates=3)
    with pytest.raises(RuntimeError, match="complete"):
        writer.append_capture(_record(4))

    writer = _writer(tmp_path / "complete-again")
    writer.complete(next_iteration=0, completed_updates=0)
    with pytest.raises(RuntimeError, match="complete"):
        writer.complete(next_iteration=4, completed_updates=3)
    bad = TrainingHistoryV3Writer(tmp_path / "bad", stream_id="29292929-2929-4292-8292-292929292929", run_id="bad", contract=_contract() | {"every_n_steps": 2}, provenance={"config_hash": "e" * 64})
    with pytest.raises(ValueError, match="cadence"):
        bad.append_capture(_record(1))


def test_off_cadence_refused(tmp_path: Path) -> None:
    writer = TrainingHistoryV3Writer(tmp_path / "cadence", stream_id="30303030-3030-4030-8030-303030303030", run_id="cadence", contract=_contract() | {"every_n_steps": 2}, provenance={"config_hash": "f" * 64})
    with pytest.raises(ValueError, match="cadence"):
        writer.append_capture(_record(1))


def test_contract_shape_nonfinite_and_term_mismatch_refused(tmp_path: Path) -> None:
    shape = TrainingHistoryV3Writer(tmp_path / "shape", stream_id="31313131-3131-4131-8131-313131313131", run_id="shape", contract=_contract() | {"walker_count": 3}, provenance={"config_hash": "1" * 64})
    with pytest.raises(ValueError, match="walker"):
        shape.append_capture(_record(0))
    with pytest.raises(ValueError, match="finite"):
        TrainingBatchRecord(iteration_step=0, completed_updates=0, update_applied=True, walker_index=np.arange(2), positions=np.full((2, 1, 3), np.nan), local_energy=np.ones(2), term_energies={"kinetic": np.ones(2)}, logabs=np.ones(2), sign=np.ones(2), finite=np.ones(2, dtype=np.bool_))
    terms = _writer(tmp_path / "terms")
    with pytest.raises(ValueError, match="term"):
        terms.append_capture(TrainingBatchRecord(iteration_step=0, completed_updates=0, update_applied=True, walker_index=np.arange(2), positions=np.zeros((2, 1, 3)), local_energy=np.ones(2), term_energies={"other": np.ones(2)}, logabs=np.ones(2), sign=np.ones(2), finite=np.ones(2, dtype=np.bool_)))


def test_partial_seal_requires_completion_boundary(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "partial")
    writer.append_capture(_record(0))
    with pytest.raises(ValueError, match="partial"):
        writer.seal_window(0, end_step=1)


def test_append_after_complete_refused(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "after")
    writer.complete(next_iteration=0, completed_updates=0)
    with pytest.raises(RuntimeError, match="complete"):
        writer.append_capture(_record(0))


def test_complete_counters_refused(tmp_path: Path) -> None:
    writer = _writer(tmp_path / "counters")
    with pytest.raises(ValueError, match="counters"):
        writer.complete(next_iteration=0, completed_updates=1)


def test_crossed_window_without_scheduled_capture_is_sealable(tmp_path: Path) -> None:
    contract = _contract() | {"every_n_steps": 200, "partition_size_steps": 100}
    writer = TrainingHistoryV3Writer(tmp_path / "cross", stream_id="26262626-2626-4262-8262-262626262626", run_id="cross", contract=contract, provenance={"config_hash": "c" * 64})
    writer.append_capture(_record(0))
    writer.seal_window(0)
    writer.seal_window(1)
    writer.complete(next_iteration=200, completed_updates=1)
    TrainingBatchReader(writer.root).validate(require_complete=True)


def test_memory_bound_and_no_fragment_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _writer(tmp_path / "memory")
    writer.append_capture(_record(0))
    tracemalloc.start()
    for step in range(1, 4):
        writer.append_capture(_record(step))
    small = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    assert not hasattr(writer, "_records")
    assert small < 2_000_000
    import tpen.callback.training_history_v3_writer as module
    monkeypatch.setattr(module.np, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("writer reread fragment")))
    writer = _writer(tmp_path / "no-read")
    writer.append_capture(_record(0))


def test_memory_bound_across_many_groups(tmp_path: Path) -> None:
    contract = _contract() | {"partition_size_steps": 1}
    writer = TrainingHistoryV3Writer(
        tmp_path / "many-groups", stream_id="36363636-3636-4636-8636-363636363636",
        run_id="many-groups", contract=contract, provenance={"config_hash": "6" * 64},
    )
    tracemalloc.start()
    for step in range(60):
        writer.append_capture(_record(step))
        writer.seal_window(step)
        if step == 9:
            baseline = tracemalloc.get_traced_memory()[0]
    writer.complete(next_iteration=60, completed_updates=60)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert current - baseline < 1_000_000
    assert peak - baseline < 2_000_000
    assert not any(isinstance(value, (dict, set)) and "group" in name.lower() for name, value in vars(writer).items())


def test_bounded_cost_measurement_prints_wall_time_and_bytes(tmp_path: Path) -> None:
    contract = _contract() | {"every_n_steps": 50, "partition_size_steps": 1000, "walker_count": 64, "n_electrons": 2}
    writer = TrainingHistoryV3Writer(tmp_path / "cost", stream_id="27272727-2727-4272-8272-272727272727", run_id="cost", contract=contract, provenance={"config_hash": "d" * 64})
    started = time.perf_counter()
    for step in range(0, 1000, 50):
        record = _record(step)
        record = TrainingBatchRecord(iteration_step=step, completed_updates=step, update_applied=True, walker_index=np.arange(64, dtype="<i8"), positions=np.zeros((64, 2, 3), dtype="<f8"), local_energy=np.ones(64), term_energies={"kinetic": np.ones(64)}, logabs=np.ones(64), sign=np.ones(64), finite=np.ones(64, dtype=np.bool_))
        writer.append_capture(record)
    elapsed = time.perf_counter() - started
    total_bytes = sum(path.stat().st_size for path in writer.root.rglob('*') if path.is_file())
    print(f"writer_cost wall_time_sec={elapsed:.6f} bytes={total_bytes}")
    warnings.warn(UserWarning(f"writer_cost wall_time_sec={elapsed:.6f} bytes={total_bytes}"))

