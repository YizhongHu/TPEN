"""Tests and a test-only byte-exact fixture writer for training_history/v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

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
) -> None:
    """Emit only the approved v3 bytes; this is not a production writer."""

    contract = _contract()
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
            "group_index": step // 4,
            "start_step": step,
            "end_step": step + 1,
            "walker_count": 2,
            "row_count": 2,
            "term_names": ["kinetic"],
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
            "type": "fragment", "previous_event_sha256": None, "group_index": step // 4,
            "start_step": step, "end_step": step + 1,
            "sidecar": f"fragments/{stem}.json", "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        })
    if complete is not None:
        for group in range((complete + 3) // 4):
            events.append({
                "type": "window_seal", "previous_event_sha256": None, "group_index": group,
                "start_step": group * 4, "end_step": min((group + 1) * 4, complete),
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
    _write_v3_fixture(root, stream_id="88888888-8888-4888-8888-888888888888", steps=[0, 1])
    lines = (root / "events.jsonl").read_bytes().splitlines(keepends=True)
    lines[1] = lines[1].replace(b'"previous_event_sha256":"', b'"previous_event_sha256":"0', 1)
    (root / "events.jsonl").write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="chain|predecessor|canonical"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_sidecar_hash_corruption(tmp_path: Path) -> None:
    root = tmp_path / "sidecar"
    _write_v3_fixture(root, stream_id="99999999-9999-4999-8999-999999999999", steps=[0])
    path = root / "fragments" / "step-000000000000.json"
    data = bytearray(path.read_bytes())
    data[-2] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="sidecar"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_bundle_hash_corruption(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    _write_v3_fixture(root, stream_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", steps=[0])
    path = root / "fragments" / "step-000000000000.npz"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="bundle|hash"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_duplicate_active_grid(tmp_path: Path) -> None:
    root = tmp_path / "duplicate"
    _write_v3_fixture(root, stream_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb", steps=[0, 1])
    lines = (root / "events.jsonl").read_bytes().splitlines(keepends=True)
    event = json.loads(lines[2])
    event["start_step"] = 0
    event["end_step"] = 1
    lines[2] = _compact(event)
    (root / "events.jsonl").write_bytes(b"".join(lines))
    with pytest.raises(ValueError):
        TrainingBatchReader(root).validate()


def test_v3_require_complete_refuses_open(tmp_path: Path) -> None:
    root = tmp_path / "open-required"
    _write_v3_fixture(root, stream_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc", steps=[0])
    with pytest.raises(ValueError, match="open|complete"):
        TrainingBatchReader(root).validate(require_complete=True)


def test_v3_refuses_cycle(tmp_path: Path) -> None:
    root = tmp_path / "cycle"
    _write_v3_fixture(root, stream_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd", origin="train_resume", parent=(root, 1, "0" * 64), cut=0)
    with pytest.raises(ValueError, match="cycle"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_duplicate_stream_id(tmp_path: Path) -> None:
    parent = tmp_path / "duplicate-parent"
    stream_id = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
    _write_v3_fixture(parent, stream_id=stream_id, steps=[0])
    child = tmp_path / "duplicate-child"
    _write_v3_fixture(child, stream_id=stream_id, origin="train_resume", parent=_pin(parent), cut=1)
    with pytest.raises(ValueError, match="stream_id"):
        TrainingBatchReader(child).validate()


def test_v3_refuses_missing_file(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    _write_v3_fixture(root, stream_id="ffffffff-ffff-4fff-8fff-ffffffffffff", steps=[0])
    (root / "fragments" / "step-000000000000.npz").unlink()
    with pytest.raises(ValueError, match="bundle|missing"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_extra_header_key(tmp_path: Path) -> None:
    root = tmp_path / "extra-key"
    _write_v3_fixture(root, stream_id="10101010-1010-4101-8101-101010101010", steps=[0])
    header = json.loads((root / "history.json").read_text())
    header["unexpected"] = True
    (root / "history.json").write_bytes(_pretty(header))
    with pytest.raises(ValueError, match="keys|canonical"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_incompatible_contract(tmp_path: Path) -> None:
    parent = tmp_path / "contract-parent"
    _write_v3_fixture(parent, stream_id="12121212-1212-4121-8121-121212121212", steps=[0])
    child = tmp_path / "contract-child"
    _write_v3_fixture(child, stream_id="13131313-1313-4131-8131-131313131313", origin="train_resume", parent=_pin(parent), cut=1, complete=1)
    header = json.loads((child / "history.json").read_text())
    header["contract"]["partition_size_steps"] = 8
    (child / "history.json").write_bytes(_pretty(header))
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
    _write_v3_fixture(root, stream_id="16161616-1616-4161-8161-161616161616", steps=[0])
    with (root / "events.jsonl").open("ab") as handle:
        handle.write(b"{not-json}\n")
    with pytest.raises(ValueError, match="JSON|canonical"):
        TrainingBatchReader(root).validate()


def test_v3_refuses_changed_bytes_within_parent_pin(tmp_path: Path) -> None:
    parent = tmp_path / "pinned-parent"
    _write_v3_fixture(parent, stream_id="17171717-1717-4171-8171-171717171717", steps=[0, 1])
    child = tmp_path / "pinned-child"
    _write_v3_fixture(child, stream_id="18181818-1818-4181-8181-181818181818", origin="train_resume", steps=[2], parent=_pin(parent), cut=2, complete=3)
    parent_bytes = (parent / "events.jsonl").read_bytes()
    (parent / "events.jsonl").write_bytes(parent_bytes.replace(b'"start_step":1', b'"start_step":2', 1))
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
    {"id": "M01", "target": "event chain", "old": "skip previous digest", "new": "require previous digest", "killer": "test_v3_refuses_chain_corruption"},
    {"id": "M02", "target": "sidecar hash", "old": "trust sidecar", "new": "hash sidecar", "killer": "test_v3_refuses_sidecar_hash_corruption"},
    {"id": "M03", "target": "bundle hash", "old": "trust bundle", "new": "hash bundle", "killer": "test_v3_refuses_bundle_hash_corruption"},
    {"id": "M04", "target": "active grid", "old": "allow duplicate row", "new": "reject duplicate row", "killer": "test_v3_refuses_duplicate_active_grid"},
    {"id": "M05", "target": "completion", "old": "allow open", "new": "require complete", "killer": "test_v3_require_complete_refuses_open"},
    {"id": "M06", "target": "cycle", "old": "follow parent blindly", "new": "reject cycles", "killer": "test_v3_refuses_cycle"},
    {"id": "M07", "target": "stream identity", "old": "allow duplicate ids", "new": "reject duplicate ids", "killer": "test_v3_refuses_duplicate_stream_id"},
    {"id": "M08", "target": "physical existence", "old": "ignore missing bundle", "new": "reject missing bundle", "killer": "test_v3_refuses_missing_file"},
    {"id": "M09", "target": "header keys", "old": "allow extra header", "new": "reject extra header", "killer": "test_v3_refuses_extra_header_key"},
    {"id": "M10", "target": "contract", "old": "ignore contract drift", "new": "reject contract drift", "killer": "test_v3_refuses_incompatible_contract"},
    {"id": "M11", "target": "cut", "old": "skip prefix coverage", "new": "require prefix coverage", "killer": "test_v3_refuses_invalid_cut"},
    {"id": "M12", "target": "LF records", "old": "skip malformed line", "new": "reject malformed line", "killer": "test_v3_refuses_unpinned_malformed_lf_line"},
    {"id": "M13", "target": "parent pin", "old": "trust changed prefix", "new": "verify changed prefix", "killer": "test_v3_refuses_changed_bytes_within_parent_pin"},
    {"id": "M14", "target": "seal inheritance", "old": "cross cut seal", "new": "reject crossing seal", "killer": "test_v3_refuses_seal_inheritance_violation"},
    {"id": "M15", "target": "legacy completion", "old": "complete legacy", "new": "refuse legacy", "killer": "test_v3_legacy_history_cannot_pass_require_complete"},
]


def test_v3_mutants_table_is_well_formed() -> None:
    assert len({row["id"] for row in MUTANTS}) == len(MUTANTS)
    assert len({row["old"] for row in MUTANTS}) == len(MUTANTS)
    assert len({row["killer"] for row in MUTANTS}) == len(MUTANTS)
    assert all(set(row) == {"id", "target", "old", "new", "killer"} for row in MUTANTS)
