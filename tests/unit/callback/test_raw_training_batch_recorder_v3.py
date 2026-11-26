"""Tests and a test-only byte-exact fixture writer for training_history/v3."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

import tpen.callback.raw_training_batch_recorder as recorder_module
from tpen.callback.raw_training_batch_recorder import TrainingBatchReader


def _pretty(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def _compact(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def _digest(header: bytes, line: bytes) -> str:
    return hashlib.sha256(hashlib.sha256(header).digest() + line).hexdigest()


def _contract() -> dict[str, object]:
    return {
        "coordinate": "iteration_step",
        "start_step": 0,
        "every_n_steps": 1,
        "partition_size_steps": 4,
        "walker_count": 2,
        "n_electrons": 1,
        "spatial_dim": 3,
        "term_names": ["kinetic"],
        "capture_semantics": "pre_update_model_forward",
    }


def _write_v3_fixture(
    root: Path,
    *,
    stream_id: str,
    origin: str = "fresh",
    steps: list[int] | None = None,
    complete: int | None = None,
    parent: tuple[Path, int, str] | None = None,
    cut: int = 0,
    contract: dict[str, object] | None = None,
) -> None:
    """Emit only the approved v3 bytes; this is not a production writer."""

    contract = contract or _contract()
    header = {
        "schema": "training_history/v3",
        "stream_id": stream_id,
        "run_id": f"run-{stream_id[:8]}",
        "origin": origin,
        "contract": contract,
    }
    root.mkdir(parents=True)
    (root / "fragments").mkdir()
    header_bytes = _pretty(header)
    (root / "history.json").write_bytes(header_bytes)
    events: list[dict[str, object]] = []
    if origin == "fresh":
        events.append({"type": "genesis", "previous_event_sha256": None})
    else:
        assert parent is not None
        parent_root, event_count, prefix = parent
        events.append({
            "type": "resume", "previous_event_sha256": None,
            "parent": {"root": str(parent_root.resolve()), "event_count": event_count, "prefix_sha256": prefix},
            "checkpoint": {
                "path": str((root / "checkpoint").resolve()),
                "manifest_sha256": "1" * 64,
                "model_sha256": "2" * 64,
                "next_iteration": cut,
                "completed_updates": cut,
            },
            "supersede_from_step": cut,
        })
    previous: str | None = None
    for step in steps or []:
        arrays: dict[str, np.ndarray] = {
            "iteration_step": np.full(2, step, dtype="<i8"),
            "completed_updates": np.full(2, step, dtype="<i8"),
            "update_applied": np.ones(2, dtype=np.bool_),
            "walker_index": np.arange(2, dtype="<i8"),
            "positions": np.asarray([[[step, 0.0, 1.0]], [[step, 1.0, 2.0]]], dtype="<f8"),
            "local_energy": np.asarray([step + 0.25, step + 0.5], dtype="<f8"),
            "logabs": np.asarray([step + 1.0, step + 2.0], dtype="<f8"),
            "sign": np.ones(2, dtype="<f8"),
            "finite": np.ones(2, dtype=np.bool_),
            "term/kinetic": np.asarray([1.0, 2.0], dtype="<f8"),
        }
        stem = f"step-{step:012d}"
        bundle = root / "fragments" / f"{stem}.npz"
        np.savez_compressed(bundle, **arrays)
        sidecar = {
            "schema": "training_history/v3",
            "source": "tpen.callback.RawTrainingBatchRecorder",
            "capture_semantics": "pre_update_model_forward",
            "stream_id": stream_id,
            "contract_sha256": hashlib.sha256(_pretty(contract)).hexdigest(),
            "group_index": step // int(contract["partition_size_steps"]),
            "start_step": step,
            "end_step": step + 1,
            "walker_count": 2,
            "row_count": 2,
            "term_names": contract["term_names"],
            "columns": sorted(arrays),
            "compressed_filename": f"fragments/{stem}.npz",
            "compressed_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            "logical_sha256": _logical_digest(arrays),
            "provenance": {
                "run_id": header["run_id"], "config_hash": "3" * 64,
                "source_revision": None, "sampler_settings": {}, "acceptance_proposal_stats": [],
            },
        }
        sidecar_path = root / "fragments" / f"{stem}.json"
        sidecar_path.write_bytes(_pretty(sidecar))
        events.append({
            "type": "fragment", "previous_event_sha256": None, "group_index": step // int(contract["partition_size_steps"]),
            "start_step": step, "end_step": step + 1,
            "sidecar": f"fragments/{stem}.json", "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        })
    if complete is not None:
        partition = int(contract["partition_size_steps"])
        for group in range((complete + partition - 1) // partition):
            events.append({
                "type": "window_seal", "previous_event_sha256": None, "group_index": group,
                "start_step": group * partition, "end_step": min((group + 1) * partition, complete),
            })
        events.append({"type": "complete", "previous_event_sha256": None, "next_iteration": complete, "completed_updates": complete})
    lines: list[bytes] = []
    for event in events:
        event["previous_event_sha256"] = previous
        line = _compact(event)
        lines.append(line)
        previous = _digest(header_bytes, line)
    (root / "events.jsonl").write_bytes(b"".join(lines))


def _logical_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _pin(root: Path) -> tuple[Path, int, str]:
    lines = (root / "events.jsonl").read_bytes().splitlines(keepends=True)
    header = (root / "history.json").read_bytes()
    previous = None
    for line in lines:
        previous = _digest(header, line)
    assert previous is not None
    return root, len(lines), previous


def _rewrite_events(root: Path, events: list[dict[str, object]]) -> None:
    """Rewrite only the event log, preserving the approved chain encoding."""

    header = (root / "history.json").read_bytes()
    lines: list[bytes] = []
    previous = None
    for event in events:
        event["previous_event_sha256"] = previous
        line = _compact(event)
        lines.append(line)
        previous = _digest(header, line)
    (root / "events.jsonl").write_bytes(b"".join(lines))


def test_v3_genesis_only_round_trip_and_completion(tmp_path: Path) -> None:
    root = tmp_path / "genesis"
    _write_v3_fixture(root, stream_id="11111111-1111-4111-8111-111111111111", complete=0)
    reader = TrainingBatchReader(root)
    reader.validate(require_complete=True)
    assert reader.read() == {}
    assert reader.completion == {"next_iteration": 0, "completed_updates": 0}


def test_v3_resume_uses_frozen_parent_prefix(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    _write_v3_fixture(parent, stream_id="22222222-2222-4222-8222-222222222222", steps=[0, 1, 2, 3])
    child = tmp_path / "child"
    _write_v3_fixture(child, stream_id="33333333-3333-4333-8333-333333333333", origin="train_resume", steps=[2, 3], parent=_pin(parent), cut=2, complete=4)
    rows = TrainingBatchReader(child).read()
    np.testing.assert_array_equal(rows["iteration_step"], [0, 0, 1, 1, 2, 2, 3, 3])
    TrainingBatchReader(child).validate(require_complete=True)


def test_v3_multi_generation_chain_is_sorted_and_inclusive(tmp_path: Path) -> None:
    first = tmp_path / "first"
    _write_v3_fixture(first, stream_id="44444444-4444-4444-8444-444444444444", steps=[0, 1])
    second = tmp_path / "second"
    _write_v3_fixture(second, stream_id="55555555-5555-4555-8555-555555555555", origin="train_resume", steps=[2, 3], parent=_pin(first), cut=2)
    third = tmp_path / "third"
    _write_v3_fixture(third, stream_id="66666666-6666-4666-8666-666666666666", origin="train_resume", steps=[4], parent=_pin(second), cut=4, complete=5)
    rows = TrainingBatchReader(third).read(end_step=4)
    np.testing.assert_array_equal(rows["iteration_step"], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])


def test_v3_torn_non_lf_tail_is_ignored_without_repair(tmp_path: Path) -> None:
    root = tmp_path / "open"
    _write_v3_fixture(root, stream_id="77777777-7777-4777-8777-777777777777", steps=[0])
    before = (root / "events.jsonl").read_bytes()
    (root / "events.jsonl").write_bytes(before + b'{"type":"complete"}')
    assert TrainingBatchReader(root).read()["iteration_step"].tolist() == [0, 0]
    assert (root / "events.jsonl").read_bytes() == before + b'{"type":"complete"}'


def test_v3_refuses_chain_corruption(tmp_path: Path) -> None:
    root = tmp_path / "chain"
    _write_v3_fixture(root, stream_id="88888888-8888-4888-8888-888888888888", steps=[0])
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    events[1]["previous_event_sha256"] = "0" * 64
    (root / "events.jsonl").write_bytes(b"".join(_compact(event) for event in events))
    with pytest.raises(ValueError, match="chain|predecessor|canonical"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_sidecar_hash_corruption(tmp_path: Path) -> None:
    root = tmp_path / "sidecar"
    _write_v3_fixture(root, stream_id="99999999-9999-4999-8999-999999999999", steps=[0])
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    events[1]["sidecar_sha256"] = "0" * 64
    _rewrite_events(root, events)
    with pytest.raises(ValueError, match="sidecar"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_bundle_hash_corruption(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_v3_fixture(root, stream_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", steps=[0])
    sidecar_path = root / "fragments" / "step-000000000000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["compressed_sha256"] = "0" * 64
    sidecar_path.write_bytes(_pretty(sidecar))
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    events[1]["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    _rewrite_events(root, events)
    with pytest.raises(ValueError, match="bundle|hash"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_duplicate_active_grid(tmp_path: Path) -> None:
    root = tmp_path / "duplicate"
    _write_v3_fixture(root, stream_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", steps=[0])
    original = recorder_module._validate_v3_sidecar

    def duplicate_rows(*args, **kwargs):
        arrays, sidecar = original(*args, **kwargs)
        arrays["walker_index"] = arrays["walker_index"].copy()
        arrays["walker_index"][1] = 0
        return arrays, sidecar

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(recorder_module, "_validate_v3_sidecar", duplicate_rows)
    try:
        with pytest.raises(ValueError, match="duplicate active"):
            TrainingBatchReader(root).validate()
    finally:
        monkeypatch.undo()


def test_v3_require_complete_refuses_open(tmp_path: Path) -> None:
    root = tmp_path / "open-required"
    _write_v3_fixture(root, stream_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc", steps=[0])
    with pytest.raises(ValueError, match="open|complete"):
        TrainingBatchReader(root).validate(require_complete=True)


def test_v3_refuses_cycle(tmp_path: Path) -> None:
    root = tmp_path / "cycle"
    _write_v3_fixture(root, stream_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd", steps=[0])
    with pytest.raises(ValueError, match="cycle"):
        TrainingBatchReader(root)._load_v3_node(root.resolve(), stack=(root.resolve(),), seen_streams=set(), limit=None)


def test_v3_refuses_duplicate_stream_id(tmp_path: Path) -> None:
    parent = tmp_path / "duplicate-parent"
    stream_id = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
    _write_v3_fixture(parent, stream_id=stream_id, steps=[0])
    child = tmp_path / "duplicate-child"
    _write_v3_fixture(child, stream_id=stream_id, origin="train_resume", parent=_pin(parent), cut=1)
    with pytest.raises(ValueError, match="stream_id"):
        TrainingBatchReader(child)._load_v3_node(child.resolve(), stack=(), seen_streams={stream_id}, limit=None)


def test_v3_refuses_missing_file(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    _write_v3_fixture(root, stream_id="ffffffff-ffff-4fff-8fff-ffffffffffff", steps=[0])
    sidecar = root / "fragments" / "step-000000000000.json"
    sidecar_bytes = sidecar.read_bytes()
    sidecar.unlink()
    original_read_bytes = Path.read_bytes

    def read_missing_sidecar(path: Path) -> bytes:
        if path == sidecar:
            return sidecar_bytes
        return original_read_bytes(path)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Path, "read_bytes", read_missing_sidecar)
    with pytest.raises(ValueError, match="bundle|missing"):
        try:
            TrainingBatchReader(root).validate()
        finally:
            monkeypatch.undo()


def test_v3_refuses_extra_header_key(tmp_path: Path) -> None:
    root = tmp_path / "extra-key"
    _write_v3_fixture(root, stream_id="10101010-1010-4101-8101-101010101010")
    header = json.loads((root / "history.json").read_text())
    header["unexpected"] = True
    (root / "history.json").write_bytes(_pretty(header))
    with pytest.raises(ValueError, match="keys|canonical"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_incompatible_contract(tmp_path: Path) -> None:
    parent = tmp_path / "contract-parent"
    _write_v3_fixture(parent, stream_id="12121212-1212-4121-8121-121212121212", steps=[0])
    child = tmp_path / "contract-child"
    child_contract = _contract()
    child_contract["partition_size_steps"] = 8
    _write_v3_fixture(child, stream_id="13131313-1313-4131-8131-131313131313", origin="train_resume", parent=_pin(parent), cut=1, complete=1, contract=child_contract)
    with pytest.raises(ValueError):
        TrainingBatchReader(child).validate()


def test_v3_refuses_invalid_cut(tmp_path: Path) -> None:
    parent = tmp_path / "cut-parent"
    _write_v3_fixture(parent, stream_id="14141414-1414-4141-8141-141414141414", steps=[0, 1])
    child = tmp_path / "cut-child"
    _write_v3_fixture(child, stream_id="15151515-1515-4151-8151-151515151515", origin="train_resume", steps=[3], parent=_pin(parent), cut=3)
    with pytest.raises(ValueError, match="prefix|coverage"):
        TrainingBatchReader(child).validate()


def test_v3_refuses_unpinned_malformed_lf_line(tmp_path: Path) -> None:
    root = tmp_path / "malformed-line"
    _write_v3_fixture(root, stream_id="16161616-1616-4161-8161-161616161616", steps=[0, 1])
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    lines = [_compact(events[0]), b" " + _compact(events[1])]
    events[1]["previous_event_sha256"] = _digest((root / "history.json").read_bytes(), lines[0])
    lines[1] = b" " + _compact(events[1])
    (root / "events.jsonl").write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="JSON|canonical"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_changed_bytes_within_parent_pin(tmp_path: Path) -> None:
    parent = tmp_path / "pinned-parent"
    _write_v3_fixture(parent, stream_id="17171717-1717-4171-8171-171717171717")
    child = tmp_path / "pinned-child"
    _write_v3_fixture(child, stream_id="18181818-1818-4181-8181-181818181818", origin="train_resume", parent=_pin(parent), cut=0)
    header = json.loads((parent / "history.json").read_text())
    header["run_id"] = "changed-run"
    (parent / "history.json").write_bytes(_pretty(header))
    with pytest.raises(ValueError, match="prefix|chain|pin"):
        TrainingBatchReader(child).validate()


def test_v3_refuses_seal_inheritance_violation(tmp_path: Path) -> None:
    parent = tmp_path / "sealed-parent"
    _write_v3_fixture(parent, stream_id="19191919-1919-4191-8191-191919191919", steps=[0, 1, 2, 3], complete=4)
    child = tmp_path / "sealed-child"
    _write_v3_fixture(child, stream_id="20202020-2020-4202-8202-202020202020", origin="train_resume", steps=[2, 3], parent=_pin(parent), cut=2, complete=4)
    with pytest.raises(ValueError, match="seal|cut"):
        TrainingBatchReader(child).validate()


def test_v3_legacy_history_cannot_pass_require_complete(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "training_history/v1", "partitions": []}) + "\n")
    with pytest.raises(ValueError, match="legacy|complete"):
        TrainingBatchReader(tmp_path).validate(require_complete=True)


MUTANTS = [
    {"id": "M01", "target": "tpen/callback/raw_training_batch_recorder.py", "old": 'if events and event["previous_event_sha256"] != digests[-1]:', "new": "if False:", "killer": "test_v3_refuses_chain_corruption"},
    {"id": "M02", "target": "tpen/callback/raw_training_batch_recorder.py", "old": 'if not sidecar_path.is_file() or _sha256(sidecar_path) != event["sidecar_sha256"]:', "new": "if False:", "killer": "test_v3_refuses_sidecar_hash_corruption"},
    {"id": "M03", "target": "tpen/callback/raw_training_batch_recorder.py", "old": 'if not bundle_path.is_file() or _sha256(bundle_path) != sidecar["compressed_sha256"]:', "new": "if False:", "killer": "test_v3_refuses_bundle_hash_corruption"},
    {"id": "M04", "target": "tpen/callback/raw_training_batch_recorder.py", "old": 'if len(identities) != len(result["iteration_step"]):', "new": "if False:", "killer": "test_v3_refuses_duplicate_active_grid"},
    {"id": "M05", "target": "tpen/callback/raw_training_batch_recorder.py", "old": "if require_complete and node.completion is None:", "new": "if False:", "killer": "test_v3_require_complete_refuses_open"},
    {"id": "M06", "target": "tpen/callback/raw_training_batch_recorder.py", "old": "if root in stack:", "new": "if False:", "killer": "test_v3_refuses_cycle"},
    {"id": "M07", "target": "tpen/callback/raw_training_batch_recorder.py", "old": "if stream_id in seen_streams:", "new": "if False:", "killer": "test_v3_refuses_duplicate_stream_id"},
    {"id": "M08", "target": "tpen/callback/raw_training_batch_recorder.py", "old": "if not history_path.is_file() or not events_path.is_file():", "new": "if False:", "killer": "test_v3_refuses_missing_file"},
    {"id": "M09", "target": "tpen/callback/raw_training_batch_recorder.py", "old": "if set(value) != allowed:", "new": "if False:", "killer": "test_v3_refuses_extra_header_key"},
    {"id": "M10", "target": "tpen/callback/raw_training_batch_recorder.py", "old": 'if parent_node.header["contract"] != header["contract"]:', "new": "if False:", "killer": "test_v3_refuses_incompatible_contract"},
    {"id": "M11", "target": "tpen/callback/raw_training_batch_recorder.py", "old": "if parent_node is not None and not _v3_has_scheduled_prefix(parent_node.rows, header[\"contract\"], cut):", "new": "if False:", "killer": "test_v3_refuses_invalid_cut"},
    {"id": "M12", "target": "tpen/callback/raw_training_batch_recorder.py", "old": "if not isinstance(event, Mapping) or line != _compact_json(event):", "new": "if False:", "killer": "test_v3_refuses_unpinned_malformed_lf_line"},
    {"id": "M13", "target": "tpen/callback/raw_training_batch_recorder.py", "old": "if limit is not None and digests[-1] != self._expected_pin_digest(root, limit):", "new": "if False:", "killer": "test_v3_refuses_changed_bytes_within_parent_pin"},
    {"id": "M14", "target": "tpen/callback/raw_training_batch_recorder.py", "old": 'if int(seal["end_step"]) > cut and int(seal["start_step"]) < cut:', "new": "if False:", "killer": "test_v3_refuses_seal_inheritance_violation"},
    {"id": "M15", "target": "tpen/callback/raw_training_batch_recorder.py", "old": 'if require_complete:\n            raise ValueError("require_complete is unavailable for legacy training histories")', "new": 'if False:\n            raise ValueError("require_complete is unavailable for legacy training histories")', "killer": "test_v3_legacy_history_cannot_pass_require_complete"},
]


def test_mutant_rows_are_executable() -> None:
    """The Cannon mutation harness can apply every row without guessing."""

    root = Path(__file__).resolve().parents[3]
    test_roots = (root / "tests", root / "executor" / "tests")
    for row in MUTANTS:
        target = root / row["target"]
        assert target.is_file(), row
        source = target.read_text()
        assert source.count(row["old"]) == 1, row
        compile(source.replace(row["old"], row["new"], 1), str(target), "exec")
        definitions = sum(
            len(re.findall(rf"^def {re.escape(row['killer'])}\s*\(", path.read_text(), re.MULTILINE))
            for test_root in test_roots
            if test_root.is_dir()
            for path in test_root.rglob("*.py")
        )
        assert definitions == 1, row
