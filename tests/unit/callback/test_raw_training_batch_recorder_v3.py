"""Tests and a test-only byte-exact fixture writer for training_history/v3."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import tpen.callback.raw_training_batch_recorder as recorder_module
from tpen.callback.raw_training_batch_recorder import TrainingBatchReader
from tpen.checkpoint import save_checkpoint


def _pretty(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode()


def _compact(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def _digest(header: bytes, line: bytes) -> str:
    return hashlib.sha256(hashlib.sha256(header).digest() + line).hexdigest()


def _write_real_checkpoint(
    root: Path, *, step: int, run_id: str, run_dir: Path | None = None
) -> Path:
    """Produce the checkpoint reference through the public checkpoint writer."""

    context = SimpleNamespace(
        cfg=OmegaConf.create({"study": {"name": "v3-fixture", "config_id": "v3"}}),
        metadata=SimpleNamespace(
            run_id=run_id,
            git_commit="fixture-revision",
            git_branch="fixture",
            dirty_worktree=False,
            command="fixture",
            device="cpu",
            dtype="float64",
            extra={},
        ),
        run_dir=root if run_dir is None else run_dir,
    )
    return save_checkpoint(
        output_dir=root,
        next_iteration=step,
        completed_updates=step,
        model=torch.nn.Linear(1, 1).double(),
        context=context,
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )


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
    emit_seals: bool = True,
    checkpoint_run_dir: Path | None = None,
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
        parent_header = json.loads((parent_root / "history.json").read_text())
        checkpoint_root = root / "checkpoint"
        checkpoint_dir = _write_real_checkpoint(
            checkpoint_root,
            step=cut,
            run_id=parent_header["run_id"],
            run_dir=parent_root.parent if checkpoint_run_dir is None else checkpoint_run_dir,
        )
        events.append({
            "type": "resume", "previous_event_sha256": None,
            "parent": {"root": str(parent_root.resolve()), "event_count": event_count, "prefix_sha256": prefix},
            "checkpoint": {
                "path": str(checkpoint_dir.resolve()),
                "manifest_sha256": "0" * 64,
                "model_sha256": "0" * 64,
                "next_iteration": cut,
                "completed_updates": cut,
            },
            "supersede_from_step": cut,
        })
        checkpoint = events[-1]["checkpoint"]
        checkpoint_dir = checkpoint_root / f"step_{cut:06d}"
        manifest = checkpoint_dir / "manifest.json"
        checkpoint["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        checkpoint["model_sha256"] = hashlib.sha256((checkpoint_dir / "model.pt").read_bytes()).hexdigest()
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
        if emit_seals:
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


def _append_event(root: Path, event: dict[str, object]) -> None:
    """Append one canonical event after an already-created parent pin."""

    header = (root / "history.json").read_bytes()
    lines = (root / "events.jsonl").read_bytes().splitlines(keepends=True)
    event["previous_event_sha256"] = _digest(header, lines[-1]) if lines else None
    with (root / "events.jsonl").open("ab") as handle:
        handle.write(_compact(event))


def test_v3_genesis_only_round_trip_and_completion(tmp_path: Path) -> None:
    root = tmp_path / "genesis"
    _write_v3_fixture(root, stream_id="11111111-1111-4111-8111-111111111111", complete=0)
    reader = TrainingBatchReader(root)
    reader.validate(require_complete=True)
    assert reader.read() == {}
    assert reader.completion == {"next_iteration": 0, "completed_updates": 0}


def test_v3_optional_measurements_and_newer_canonical_uuid(tmp_path: Path) -> None:
    root = tmp_path / "optional"
    _write_v3_fixture(root, stream_id="36363636-3636-7363-8636-363636363636", steps=[0])
    sidecar_path = root / "fragments" / "step-000000000000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["measurements"] = {}
    sidecar_path.write_bytes(_pretty(sidecar))
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    events[1]["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    _rewrite_events(root, events)
    reader = TrainingBatchReader(root)
    reader.validate()
    assert reader.provenance[0]["history_root"] == str(root.resolve())
    assert reader.provenance[0]["stream_id"] == "36363636-3636-7363-8636-363636363636"


def test_v3_read_then_completion_preserves_provenance(tmp_path: Path) -> None:
    root = tmp_path / "completion-provenance"
    _write_v3_fixture(root, stream_id="3a3a3a3a-3a3a-43a3-83a3-3a3a3a3a3a3a", steps=[0, 1], complete=2)
    reader = TrainingBatchReader(root)
    reader.read(start_step=1, end_step=1)
    before = dict(reader.provenance)
    assert reader.completion == {"next_iteration": 2, "completed_updates": 2}
    assert reader.provenance == before


def test_v3_empty_read_clears_provenance_after_prior_read(tmp_path: Path) -> None:
    root = tmp_path / "empty-read-provenance"
    _write_v3_fixture(root, stream_id="3b3b3b3b-3b3b-43b3-83b3-3b3b3b3b3b3b", steps=[0, 1], complete=2)
    reader = TrainingBatchReader(root)
    reader.read()
    assert reader.provenance
    assert reader.read(start_step=2) == {}
    assert reader.provenance == {}


def test_v3_sidecar_json_integer_types_are_strict(tmp_path: Path) -> None:
    root = tmp_path / "strict-int"
    _write_v3_fixture(root, stream_id="37373737-3737-4373-8373-373737373737", steps=[0])
    sidecar_path = root / "fragments" / "step-000000000000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["row_count"] = 2.0
    sidecar_path.write_bytes(_pretty(sidecar))
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    events[1]["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    _rewrite_events(root, events)
    with pytest.raises(ValueError, match="integer"):
        TrainingBatchReader(root).validate()


def test_v3_resume_uses_frozen_parent_prefix(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    _write_v3_fixture(parent, stream_id="22222222-2222-4222-8222-222222222222", steps=[0, 1, 2, 3])
    child = tmp_path / "child"
    _write_v3_fixture(child, stream_id="33333333-3333-4333-8333-333333333333", origin="train_resume", steps=[2, 3], parent=_pin(parent), cut=2, complete=4)
    _append_event(parent, {"type": "window_seal", "group_index": 0, "start_step": 0, "end_step": 4})
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


def test_v3_range_read_excludes_committed_fragments_and_empty_selection(tmp_path: Path) -> None:
    root = tmp_path / "range-root"
    _write_v3_fixture(root, stream_id="67676767-6767-4676-8676-676767676767", steps=[0, 1])
    rows = TrainingBatchReader(root).read(end_step=0)
    np.testing.assert_array_equal(rows["iteration_step"], [0, 0])
    rows = TrainingBatchReader(root).read(start_step=1)
    np.testing.assert_array_equal(rows["iteration_step"], [1, 1])
    assert TrainingBatchReader(root).read(start_step=2) == {}

    parent = tmp_path / "range-parent"
    _write_v3_fixture(parent, stream_id="68686868-6868-4868-8868-686868686868", steps=[0, 1])
    child = tmp_path / "range-child"
    _write_v3_fixture(
        child,
        stream_id="69696969-6969-4969-8969-696969696969",
        origin="train_resume",
        parent=_pin(parent),
        cut=2,
        steps=[2],
    )
    rows = TrainingBatchReader(child).read(end_step=0)
    np.testing.assert_array_equal(rows["iteration_step"], [0, 0])
    assert TrainingBatchReader(child).read(start_step=3) == {}


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
    child = tmp_path / "cycle-child"
    _write_v3_fixture(child, stream_id="dededede-dede-4ede-8ede-dededededede", origin="train_resume", parent=_pin(root), cut=1)
    events = [json.loads(line) for line in (child / "events.jsonl").read_text().splitlines()]
    events[0]["parent"]["root"] = str(child.resolve())
    _rewrite_events(child, events)
    with pytest.raises(ValueError, match="cycle"):
        TrainingBatchReader(child).validate()


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
    events_path = root / "events.jsonl"
    events_bytes = events_path.read_bytes()
    events_path.unlink()
    original_read_bytes = Path.read_bytes

    def read_missing_events(path: Path) -> bytes:
        if path == events_path:
            return events_bytes
        return original_read_bytes(path)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Path, "read_bytes", read_missing_events)
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
    with pytest.raises(ValueError, match="contract"):
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


def test_v3_refuses_wrong_parent_pin_digest(tmp_path: Path) -> None:
    parent = tmp_path / "pinned-parent"
    _write_v3_fixture(parent, stream_id="17171717-1717-4171-8171-171717171717")
    child = tmp_path / "pinned-child"
    pin = _pin(parent)
    _write_v3_fixture(child, stream_id="18181818-1818-4181-8181-181818181818", origin="train_resume", parent=(pin[0], pin[1], "0" * 64), cut=0)
    with pytest.raises(ValueError, match="prefix"):
        TrainingBatchReader(child).validate()


def test_v3_refuses_changed_bytes_within_parent_pin(tmp_path: Path) -> None:
    parent = tmp_path / "changed-parent"
    _write_v3_fixture(parent, stream_id="28282828-2828-4282-8282-282828282828")
    child = tmp_path / "changed-child"
    _write_v3_fixture(child, stream_id="29292929-2929-4292-8292-292929292929", origin="train_resume", parent=_pin(parent), cut=0)
    header = json.loads((parent / "history.json").read_text())
    header["run_id"] = "changed-run"
    (parent / "history.json").write_bytes(_pretty(header))
    with pytest.raises(ValueError, match="prefix|provenance"):
        TrainingBatchReader(child).validate()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("manifest_sha256", "0" * 64, "hashes"),
        ("model_sha256", "0" * 64, "hashes"),
        ("next_iteration", 99, "counters"),
        ("completed_updates", 99, "counters"),
        ("path", "/does/not/exist", "checkpoint"),
    ],
)
def test_v3_checkpoint_binding_mutations_are_refused(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    parent = tmp_path / "checkpoint-parent"
    _write_v3_fixture(parent, stream_id="30303030-3030-4030-8030-303030303030", steps=[0])
    child = tmp_path / "checkpoint-child"
    _write_v3_fixture(
        child,
        stream_id="31313131-3131-4131-8131-313131313131",
        origin="train_resume",
        parent=_pin(parent),
        cut=1,
    )
    events = [json.loads(line) for line in (child / "events.jsonl").read_text().splitlines()]
    events[0]["checkpoint"][field] = value
    _rewrite_events(child, events)
    with pytest.raises(ValueError, match=message):
        TrainingBatchReader(child).validate()


@pytest.mark.parametrize("locator", ["root", "latest"])
def test_v3_checkpoint_refuses_mutable_root_and_latest_locators(tmp_path: Path, locator: str) -> None:
    parent = tmp_path / f"locator-parent-{locator}"
    _write_v3_fixture(parent, stream_id="38383838-3838-4383-8383-383838383838", steps=[0])
    child = tmp_path / f"locator-child-{locator}"
    _write_v3_fixture(
        child,
        stream_id="39393939-3939-4393-8393-393939393939",
        origin="train_resume",
        parent=_pin(parent),
        cut=1,
    )
    events = [json.loads(line) for line in (child / "events.jsonl").read_text().splitlines()]
    checkpoint_root = child / "checkpoint"
    events[0]["checkpoint"]["path"] = str(
        checkpoint_root if locator == "root" else checkpoint_root / "latest.json"
    )
    _rewrite_events(child, events)
    with pytest.raises(ValueError, match="checkpoint"):
        TrainingBatchReader(child).validate()


def test_v3_checkpoint_provenance_is_bound_to_parent_history(tmp_path: Path) -> None:
    parent = tmp_path / "attempt-a" / "core" / "training_history"
    parent.parent.mkdir(parents=True)
    _write_v3_fixture(parent, stream_id="32323232-3232-4232-8232-323232323232", steps=[0])
    child = tmp_path / "provenance-child"
    _write_v3_fixture(
        child,
        stream_id="33333333-3333-4333-8333-333333333333",
        origin="train_resume",
        parent=_pin(parent),
        cut=1,
        checkpoint_run_dir=parent.parent,
    )
    # The approved layout is <attempt>/core/training_history, with the
    # checkpoint provenance bound to the containing <attempt>/core run dir.
    TrainingBatchReader(child).validate()
    event = json.loads((child / "events.jsonl").read_text().splitlines()[0])
    checkpoint_dir = child / "checkpoint" / "step_000001"
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["run_id"] = "wrong-run"
    manifest_path.write_bytes(_pretty(manifest))
    event["checkpoint"]["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _rewrite_events(child, [event])
    with pytest.raises(ValueError, match="provenance"):
        TrainingBatchReader(child).validate()


def test_v3_checkpoint_run_dir_from_foreign_attempt_is_refused(tmp_path: Path) -> None:
    parent = tmp_path / "attempt-a" / "core" / "training_history"
    parent.parent.mkdir(parents=True)
    _write_v3_fixture(parent, stream_id="40404040-4040-4404-8404-404040404040", steps=[0])
    child = tmp_path / "run-dir-child"
    _write_v3_fixture(
        child,
        stream_id="41414141-4141-4414-8414-414141414141",
        origin="train_resume",
        parent=_pin(parent),
        cut=1,
        checkpoint_run_dir=parent.parent,
    )
    manifest_path = child / "checkpoint" / "step_000001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["run_dir"] = str(tmp_path / "attempt-b" / "core")
    manifest_path.write_bytes(_pretty(manifest))
    events = [json.loads(line) for line in (child / "events.jsonl").read_text().splitlines()]
    events[0]["checkpoint"]["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _rewrite_events(child, events)
    with pytest.raises(ValueError, match="provenance"):
        TrainingBatchReader(child).validate()


def test_v3_checkpoint_history_escape_is_refused(tmp_path: Path) -> None:
    parent = tmp_path / "attempt-a" / "core" / "training_history"
    parent.parent.mkdir(parents=True)
    _write_v3_fixture(parent, stream_id="4b4b4b4b-4b4b-44b4-84b4-4b4b4b4b4b4b", steps=[0])
    child = tmp_path / "escape-child"
    _write_v3_fixture(
        child,
        stream_id="4c4c4c4c-4c4c-44c4-84c4-4c4c4c4c4c4c",
        origin="train_resume",
        parent=_pin(parent),
        cut=1,
        checkpoint_run_dir=parent.parent,
    )
    manifest_path = child / "checkpoint" / "step_000001" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"]["run_dir"] = str(parent / "nested" / "run")
    manifest_path.write_bytes(_pretty(manifest))
    events = [json.loads(line) for line in (child / "events.jsonl").read_text().splitlines()]
    events[0]["checkpoint"]["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _rewrite_events(child, events)
    with pytest.raises(ValueError, match="provenance"):
        TrainingBatchReader(child).validate()


def test_v3_checkpoint_does_not_compare_inherited_source_revision(tmp_path: Path) -> None:
    parent = tmp_path / "revision-parent"
    _write_v3_fixture(parent, stream_id="45454545-4545-4454-8454-454545454545", steps=[0])
    sidecar_path = parent / "fragments" / "step-000000000000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["provenance"]["source_revision"] = "revision-A"
    sidecar_path.write_bytes(_pretty(sidecar))
    parent_events = [json.loads(line) for line in (parent / "events.jsonl").read_text().splitlines()]
    parent_events[1]["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    _rewrite_events(parent, parent_events)

    child = tmp_path / "revision-child"
    _write_v3_fixture(
        child,
        stream_id="46464646-4646-4464-8464-464646464646",
        origin="train_resume",
        parent=_pin(parent),
        cut=1,
    )
    TrainingBatchReader(child).validate()


def test_v3_measurements_allow_hash_and_require_positive_values(tmp_path: Path) -> None:
    root = tmp_path / "measurements-valid"
    _write_v3_fixture(root, stream_id="42424242-4242-4424-8424-424242424242", steps=[0])
    sidecar_path = root / "fragments" / "step-000000000000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["measurements"] = {
        "compressed_byte_count": 1,
        "uncompressed_byte_count": 1,
        "uncompressed_sha256": "a" * 64,
        "compression_ratio": 0.5,
        "write_time_sec": 0,
    }
    sidecar_path.write_bytes(_pretty(sidecar))
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    events[1]["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    _rewrite_events(root, events)
    TrainingBatchReader(root).validate()


@pytest.mark.parametrize(
    ("measurements", "message"),
    [
        (None, "measurements"),
        ({"compressed_byte_count": 0}, "measurement compressed_byte_count"),
        ({"uncompressed_byte_count": 0}, "measurement uncompressed_byte_count"),
        ({"compression_ratio": 0}, "compression_ratio"),
    ],
)
def test_v3_measurements_refuse_null_zero_and_nonpositive_values(
    tmp_path: Path, measurements: object, message: str
) -> None:
    root = tmp_path / f"measurements-invalid-{message.replace(' ', '-') }"
    _write_v3_fixture(root, stream_id="43434343-4343-4434-8434-434343434343", steps=[0])
    sidecar_path = root / "fragments" / "step-000000000000.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["measurements"] = measurements
    sidecar_path.write_bytes(_pretty(sidecar))
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    events[1]["sidecar_sha256"] = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    _rewrite_events(root, events)
    with pytest.raises(ValueError, match=message):
        TrainingBatchReader(root).validate()


def test_v3_term_names_refuse_empty_string(tmp_path: Path) -> None:
    root = tmp_path / "empty-term"
    contract = _contract()
    contract["term_names"] = [""]
    _write_v3_fixture(root, stream_id="44444444-4444-4444-8444-444444444444", contract=contract)
    with pytest.raises(ValueError, match="term_names"):
        TrainingBatchReader(root).validate()


def test_v3_fragment_directory_symlink_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "symlink-dir"
    _write_v3_fixture(root, stream_id="34343434-3434-4434-8434-343434343434", steps=[0])
    outside = tmp_path / "outside"
    outside.mkdir()
    for path in (root / "fragments").iterdir():
        shutil.move(str(path), outside / path.name)
    (root / "fragments").rmdir()
    os.symlink(outside, root / "fragments")
    with pytest.raises(ValueError, match="symlink"):
        TrainingBatchReader(root).validate()


@pytest.mark.parametrize("component", ["sidecar", "bundle"])
def test_v3_fragment_file_symlink_is_refused(tmp_path: Path, component: str) -> None:
    root = tmp_path / f"symlink-{component}"
    _write_v3_fixture(root, stream_id="35353535-3535-4535-8535-353535353535", steps=[0])
    fragment = root / "fragments" / "step-000000000000"
    path = fragment.with_suffix(".json" if component == "sidecar" else ".npz")
    outside = tmp_path / f"outside-{component}"
    shutil.move(str(path), outside)
    os.symlink(outside, path)
    with pytest.raises(ValueError, match="symlink"):
        TrainingBatchReader(root).validate()


def test_v3_accepts_crossing_full_seal_and_discards_parent_seal(tmp_path: Path) -> None:
    parent = tmp_path / "sealed-parent"
    _write_v3_fixture(parent, stream_id="19191919-1919-4191-8191-191919191919", steps=[0, 1, 2, 3], complete=4)
    child = tmp_path / "sealed-child"
    _write_v3_fixture(child, stream_id="20202020-2020-4202-8202-202020202020", origin="train_resume", steps=[2, 3], parent=_pin(parent), cut=2, complete=4)
    reader = TrainingBatchReader(child)
    reader.validate(require_complete=True)
    np.testing.assert_array_equal(reader.read()["iteration_step"], [0, 0, 1, 1, 2, 2, 3, 3])


def test_v3_partial_parent_seal_is_not_inherited(tmp_path: Path) -> None:
    parent = tmp_path / "partial-parent"
    _write_v3_fixture(parent, stream_id="21212121-2121-4121-8121-212121212121", steps=[0, 1], complete=2)
    child = tmp_path / "partial-child"
    _write_v3_fixture(
        child,
        stream_id="22222222-2222-4222-8222-222222222222",
        origin="train_resume",
        parent=_pin(parent),
        cut=2,
        complete=2,
        emit_seals=False,
    )
    with pytest.raises(ValueError, match="window seals"):
        TrainingBatchReader(child).validate(require_complete=True)
    valid = tmp_path / "partial-child-valid"
    _write_v3_fixture(
        valid,
        stream_id="23232323-2323-4232-8232-232323232323",
        origin="train_resume",
        parent=_pin(parent),
        cut=2,
        complete=2,
    )
    TrainingBatchReader(valid).validate(require_complete=True)


def test_v3_effective_seals_propagate_transitively(tmp_path: Path) -> None:
    first = tmp_path / "seal-first"
    _write_v3_fixture(first, stream_id="24242424-2424-4242-8242-242424242424", steps=[0, 1, 2, 3], complete=4)
    second = tmp_path / "seal-second"
    _write_v3_fixture(
        second,
        stream_id="25252525-2525-4252-8252-252525252525",
        origin="train_resume",
        parent=_pin(first),
        cut=4,
        complete=4,
        emit_seals=False,
    )
    third = tmp_path / "seal-third"
    _write_v3_fixture(
        third,
        stream_id="26262626-2626-4262-8262-262626262626",
        origin="train_resume",
        parent=_pin(second),
        cut=4,
        complete=4,
        emit_seals=False,
    )
    TrainingBatchReader(third).validate(require_complete=True)


def test_v3_duplicate_active_seals_are_refused(tmp_path: Path) -> None:
    root = tmp_path / "duplicate-seal"
    _write_v3_fixture(root, stream_id="27272727-2727-4272-8272-272727272727", steps=[0, 1, 2, 3], complete=4)
    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    events.insert(-1, {"type": "window_seal", "group_index": 0, "start_step": 0, "end_step": 4})
    _rewrite_events(root, events)
    with pytest.raises(ValueError, match="duplicate active"):
        TrainingBatchReader(root).validate()


def test_v3_legacy_history_cannot_pass_require_complete(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "training_history/v1", "partitions": []}) + "\n")
    with pytest.raises(ValueError, match="legacy|complete"):
        TrainingBatchReader(tmp_path).validate(require_complete=True)
