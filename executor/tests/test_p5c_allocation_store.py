"""P5c allocation, fence, and durable receipt acceptance tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.results import OUTCOME_STATUSES, OutcomeReceipt
from tpen_executor.storage import DirectoryStore, FenceRejected
from tpen_executor.tasks import ConfiguredRun


def _group(*task_ids: str) -> ReadyGroup:
    program = ProgramSpecRef({"entrypoint": "tpen.run"}, digest({"entrypoint": "tpen.run"}))
    members = tuple(TaskSpec(task_id, f"lineage-{task_id}", program, digest({"task": task_id}), 1, "steps") for task_id in task_ids)
    return ReadyGroup("group", members, digest({"plan": 1}))


def _allocation(name: str = "allocation") -> ConfiguredAllocation:
    return ConfiguredAllocation(AllocationId(name), BackendJobId(f"job-{name}"), {"node": "n", "nested": {"slot": 2}})


def test_p5c_configured_records_freeze_round_trip_and_reject_scalars(tmp_path: Path) -> None:
    """Configured records preserve values, reject bad paths, and expose a command."""
    allocation = _allocation()
    assert ConfiguredAllocation.decode(allocation.encode()) == allocation
    with pytest.raises((AttributeError, TypeError)):
        allocation.facts["node"] = "other"
    run = ConfiguredRun("task", str(tmp_path / "config.yaml"), ["x=1"])
    assert run.command().args[-1] == "x=1"
    with pytest.raises(ValueError):
        ConfiguredRun("task", "relative.yaml")
    with pytest.raises(TypeError):
        ConfiguredRun("task", "/config.yaml", "x=1")


def test_p5c_fresh_fences_are_unique_and_directories_are_not_reused(tmp_path: Path) -> None:
    """Two starts create distinct unpredictable attempt directories."""
    store = DirectoryStore(tmp_path)
    group = _group("a")
    first = store.begin_attempt(group, "a", _allocation(), "t1")
    second = store.begin_attempt(group, "a", _allocation(), "t2")
    assert first.attempt_id != second.attempt_id
    assert first.output_dir != second.output_dir
    assert Path(first.output_dir).is_dir() and Path(second.output_dir).is_dir()
    with pytest.raises(FenceRejected):
        os.mkdir(first.output_dir)


def test_p5c_stale_and_double_finish_are_rejected_without_overwrite(tmp_path: Path) -> None:
    """The current pointer fences completion and receipts are exact-once."""
    store = DirectoryStore(tmp_path)
    group = _group("a")
    first = store.begin_attempt(group, "a", _allocation(), "t1")
    second = store.begin_attempt(group, "a", _allocation(), "t2")
    with pytest.raises(FenceRejected):
        store.finish_attempt(first, "completed", "done")
    assert not Path(first.output_dir, "outcome.json").exists()
    store.finish_attempt(second, "failed", "done")
    original = Path(second.output_dir, "outcome.json").read_bytes()
    with pytest.raises(FenceRejected):
        store.finish_attempt(second, "completed", "again")
    assert Path(second.output_dir, "outcome.json").read_bytes() == original


def test_p5c_crossed_fences_and_statuses_are_rejected(tmp_path: Path) -> None:
    """Root, group, and allocation crossings plus unknown statuses fail closed."""
    group = _group("a")
    store = DirectoryStore(tmp_path / "one")
    fence = store.begin_attempt(group, "a", _allocation(), "t1")
    other = DirectoryStore(tmp_path / "two")
    with pytest.raises(FenceRejected):
        other.finish_attempt(fence, "completed", "done")
    with pytest.raises(ValueError):
        OutcomeReceipt("a", fence.attempt_id, group.identity, "done", "unknown", fence.output_dir)
    assert OUTCOME_STATUSES == {"completed", "failed", "timed_out", "launch_failed"}


def test_p5c_manifest_rebuilds_from_receipts_for_three_members(tmp_path: Path) -> None:
    """A fresh store reconstructs all independent member attempts identically."""
    group = _group("a", "b", "c")
    allocation = _allocation()
    store = DirectoryStore(tmp_path)
    for task_id, status in zip(("a", "b", "c"), ("completed", "failed", "timed_out")):
        fence = store.begin_attempt(group, task_id, allocation, "start")
        store.finish_attempt(fence, status, "finish")
    manifest = store.manifest(group)
    assert len(manifest.attempts) == 3
    assert {item.start_receipt.allocation_id for item in manifest.attempts} == {allocation.allocation_id}
    assert DirectoryStore(tmp_path).manifest(group) == manifest


def test_p5c_tampered_receipts_are_not_skipped(tmp_path: Path) -> None:
    """Truncation and foreign receipt content are manifest errors."""
    group = _group("a")
    store = DirectoryStore(tmp_path)
    fence = store.begin_attempt(group, "a", _allocation(), "start")
    Path(fence.output_dir, "start.json").write_bytes(b"{")
    with pytest.raises((ValueError, TypeError, OSError)):
        store.manifest(group)


def test_p5c_atomic_write_failure_leaves_no_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure between durable temp write and replacement is cleaned up."""
    store = DirectoryStore(tmp_path)
    group = _group("a")
    real_replace = os.replace

    def fail_replace(source: str, destination: str) -> None:
        if destination.endswith("start.json"):
            raise OSError("injected replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.begin_attempt(group, "a", _allocation(), "start")
    assert not list(tmp_path.rglob(".tmp-*"))
