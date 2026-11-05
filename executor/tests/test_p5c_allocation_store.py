"""P5c allocation, fence, and durable receipt acceptance tests."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
import tempfile

import pytest

from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.results import OUTCOME_STATUSES, OutcomeReceipt, StartReceipt
from tpen_executor.storage import AttemptFence, DirectoryStore, FenceRejected
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


@pytest.mark.parametrize("record_name", ["configured-allocation", "configured-run", "attempt-fence"], ids=["allocation", "run", "fence"])
def test_p5c_each_public_record_is_frozen_and_hashes_consistently(record_name: str, tmp_path: Path) -> None:
    """Each public value record rejects assignment, new fields, and nested mutation."""
    allocation = _allocation()
    records = {
        "configured-allocation": allocation,
        "configured-run": ConfiguredRun("task", str(tmp_path / "config.yaml"), ("x=1",)),
        "attempt-fence": AttemptFence(str(tmp_path), _group("a").identity, "a", allocation.allocation_id.value, AttemptId("attempt"), str(tmp_path / "out")),
    }
    record = records[record_name]
    assert record == type(record)(**{field: getattr(record, field) for field in record.__dataclass_fields__ if field != "schema"})
    clone = type(record)(**{field: getattr(record, field) for field in record.__dataclass_fields__ if field != "schema"})
    assert hash(record) == hash(clone)
    with pytest.raises((AttributeError, TypeError)):
        record.new_field = "rejected"  # type: ignore[attr-defined]
    with pytest.raises((AttributeError, TypeError)):
        record.id = "changed"  # type: ignore[attr-defined]
    if record_name == "configured-allocation":
        with pytest.raises((AttributeError, TypeError)):
            record.facts["nested"]["slot"] = 99


@pytest.mark.parametrize("mutation", ["unknown", "missing", "wrong-type", "facts-type"], ids=["unknown-key", "missing-key", "wrong-type", "facts-type"])
def test_p5c_configured_allocation_decoder_rejects_each_mapping_mutation(mutation: str) -> None:
    """The allocation wire boundary rejects unknown, missing, and wrong fields."""
    mapping = _allocation().to_dict()
    if mutation == "unknown":
        mapping["unexpected"] = True
    elif mutation == "missing":
        del mapping["backend_job_id"]
    elif mutation == "wrong-type":
        mapping["allocation_id"] = 7
    else:
        mapping["facts"] = "not-a-mapping"
    with pytest.raises((ValueError, TypeError)):
        ConfiguredAllocation.from_mapping(mapping)


@pytest.mark.parametrize("bad_field", ["id", "config", "overrides"], ids=["id-type", "config-type", "overrides-type"])
def test_p5c_configured_run_rejects_each_wrong_constructor_type(tmp_path: Path, bad_field: str) -> None:
    """ConfiguredRun rejects malformed identifiers, paths, and override collections."""
    values = {"id": "task", "config": str(tmp_path / "config.yaml"), "overrides": ()}
    values[bad_field] = 7
    with pytest.raises((ValueError, TypeError)):
        ConfiguredRun(**values)  # type: ignore[arg-type]


def test_p5c_fresh_fences_are_unique_and_directories_are_not_reused(tmp_path: Path) -> None:
    """Two starts create distinct unpredictable attempt directories."""
    store = DirectoryStore(tmp_path)
    group = _group("a")
    first = store.begin_attempt(group, "a", _allocation(), "t1")
    second = store.begin_attempt(group, "a", _allocation(), "t2")
    first_start = Path(first.output_dir, "start.json").read_bytes()
    second_start = Path(second.output_dir, "start.json").read_bytes()
    third = store.begin_attempt(group, "a", _allocation(), "t3")
    assert first.attempt_id != second.attempt_id
    assert first.output_dir != second.output_dir
    assert third.output_dir not in {first.output_dir, second.output_dir}
    assert Path(first.output_dir).is_dir() and Path(second.output_dir).is_dir()
    assert Path(first.output_dir, "start.json").read_bytes() == first_start
    assert Path(second.output_dir, "start.json").read_bytes() == second_start


def test_p5c_preexisting_attempt_directory_is_refused_and_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fixed-id collision refuses without reusing or changing the directory."""
    import tpen_executor.storage as storage_module

    group = _group("a")
    task_dir = tmp_path / "groups" / group.identity / "a"
    existing = task_dir / "fixed-attempt"
    existing.mkdir(parents=True)
    marker = existing / "user-data"
    marker.write_bytes(b"preserve")
    monkeypatch.setattr(storage_module, "uuid4", lambda: type("FixedUUID", (), {"hex": "fixed-attempt"})())
    with pytest.raises(FenceRejected):
        DirectoryStore(tmp_path).begin_attempt(group, "a", _allocation(), "start")
    assert marker.read_bytes() == b"preserve"
    assert list(existing.iterdir()) == [marker]


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


def _all_bytes(root: Path) -> dict[str, bytes]:
    """Snapshot all regular files under a receipt root."""
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("crossing", ["root", "group", "task", "allocation"], ids=["cross-root", "cross-group", "cross-task", "cross-allocation"])
def test_p5c_each_crossed_fence_rejection_preserves_bytes(tmp_path: Path, crossing: str) -> None:
    """Root, group, and allocation crossings plus unknown statuses fail closed."""
    group = _group("a")
    store = DirectoryStore(tmp_path / "one")
    allocation = _allocation()
    fence = store.begin_attempt(group, "a", allocation, "t1")
    other = DirectoryStore(tmp_path / "two")
    if crossing == "root":
        crossed = fence
        target = other
    elif crossing == "group":
        crossed = replace(fence, group_digest=_group("b").identity)
        target = store
    elif crossing == "task":
        crossed = replace(fence, task_id="foreign-task")
        target = store
    else:
        crossed = replace(fence, allocation_id="foreign-allocation")
        target = store
    before = _all_bytes(Path(target.root))
    with pytest.raises(FenceRejected):
        target.finish_attempt(crossed, "completed", "done")
    assert _all_bytes(Path(target.root)) == before
    with pytest.raises(ValueError):
        OutcomeReceipt("a", fence.attempt_id, group.identity, "done", "unknown", fence.output_dir)
    assert OUTCOME_STATUSES == {"completed", "failed", "timed_out", "launch_failed"}


@pytest.mark.parametrize(
    ("bad_status", "expected_exception"),
    [("unknown", ValueError), ("", TypeError), ("completed ", ValueError), ("cancelled", ValueError)],
    ids=["unknown", "empty", "whitespace", "unvocabulary"],
)
def test_p5c_status_decode_rejects_each_unknown_value(
    tmp_path: Path, bad_status: str, expected_exception: type[Exception]
) -> None:
    """Construction and canonical decode enforce the closed status set."""
    group = _group("a")
    fence = DirectoryStore(tmp_path).begin_attempt(group, "a", _allocation(), "start")
    with pytest.raises(expected_exception):
        OutcomeReceipt("a", fence.attempt_id, group.identity, "done", bad_status, fence.output_dir)
    mapping = OutcomeReceipt("a", fence.attempt_id, group.identity, "done", "completed", fence.output_dir).to_dict()
    mapping["status"] = bad_status
    # ``text`` accepts surrounding whitespace, so this reaches vocabulary validation.
    with pytest.raises(expected_exception):
        OutcomeReceipt.decode(mapping)


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


@pytest.mark.parametrize("mutation", ["truncated", "tampered-field", "foreign-group", "unknown-schema"], ids=["truncated", "tampered-field", "foreign-group", "unknown-schema"])
def test_p5c_manifest_rejects_each_receipt_corruption(tmp_path: Path, mutation: str) -> None:
    """Manifest reconstruction never skips a malformed or foreign receipt."""
    group = _group("a")
    store = DirectoryStore(tmp_path)
    fence = store.begin_attempt(group, "a", _allocation(), "start")
    start_path = Path(fence.output_dir, "start.json")
    mapping = StartReceipt.decode(start_path.read_bytes()).to_dict()
    if mutation == "truncated":
        start_path.write_bytes(b"{")
    else:
        if mutation == "tampered-field":
            mapping["task_id"] = "foreign-task"
        elif mutation == "foreign-group":
            mapping["group_digest"] = _group("foreign").identity
        else:
            mapping["schema"] = "foreign-schema"
        start_path.write_bytes(canonical_bytes(mapping))
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
    assert not list(tmp_path.rglob("start.json"))


def test_p5c_runtime_write_trap_is_lexical_and_reentrant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Store write calls stay lexically below root while the trap avoids recursion."""
    root = tmp_path / "root"
    writes: list[str] = []
    active = False

    def guarded(path: object) -> None:
        nonlocal active
        if active:
            return
        active = True
        try:
            writes.append(os.path.abspath(os.path.normpath(os.fspath(path))))
        finally:
            active = False

    real_mkdir, real_makedirs, real_open, real_replace, real_unlink, real_mkstemp = os.mkdir, os.makedirs, os.open, os.replace, os.unlink, tempfile.mkstemp
    monkeypatch.setattr(os, "mkdir", lambda path, *args, **kwargs: (guarded(path), real_mkdir(path, *args, **kwargs))[1])
    monkeypatch.setattr(os, "makedirs", lambda path, *args, **kwargs: (guarded(path), real_makedirs(path, *args, **kwargs))[1])
    monkeypatch.setattr(os, "open", lambda path, *args, **kwargs: (guarded(path), real_open(path, *args, **kwargs))[1])
    monkeypatch.setattr(os, "replace", lambda source, destination: (guarded(source), guarded(destination), real_replace(source, destination))[2])
    monkeypatch.setattr(os, "unlink", lambda path, *args, **kwargs: (guarded(path), real_unlink(path, *args, **kwargs))[1])
    monkeypatch.setattr(tempfile, "mkstemp", lambda *args, **kwargs: (guarded(kwargs.get("dir", args[1] if len(args) > 1 else ".")), real_mkstemp(*args, **kwargs))[1])
    store = DirectoryStore(root)
    group = _group("a")
    fence = store.begin_attempt(group, "a", _allocation(), "start")
    store.finish_attempt(fence, "completed", "finish")
    root_name = os.path.abspath(os.path.normpath(os.fspath(root)))
    assert writes and all(path == root_name or path.startswith(root_name + os.sep) for path in writes)


def test_p5c_attached_allocation_delegates_with_bound_allocation() -> None:
    """Attached lifecycle delegates begin/finish and observes its configured allocation."""
    class FakeStore:
        def begin_attempt(self, group, task_id, allocation, started_at_utc):
            self.begin_args = (group, task_id, allocation, started_at_utc)
            return "fence"

        def finish_attempt(self, fence, status, finished_at_utc, result_digest=None):
            self.finish_args = (fence, status, finished_at_utc, result_digest)
            return "outcome"

    allocation = _allocation()
    fake = FakeStore()
    attached = AttachedAllocation(allocation, fake)  # type: ignore[arg-type]
    group = _group("a")
    assert attached.begin(group, "a", "start") == "fence"
    assert fake.begin_args == (group, "a", allocation, "start")
    assert attached.finish("fence", "completed", "finish", "digest") == "outcome"
    assert fake.finish_args == ("fence", "completed", "finish", "digest")
    assert attached.observe("running", "observe").allocation_id == allocation.allocation_id


def test_p5c_packaged_config_materializes_allocation_store_task_and_lifecycle(tmp_path: Path) -> None:
    """Packaged Hydra nodes compose and instantiate without scheduler behavior."""
    from hydra.utils import instantiate
    from tpen_executor.configuration import StagedHydra

    staged = StagedHydra().compose_config(overrides=[f"store.root={tmp_path}"])
    allocation = instantiate(staged.config.allocation)
    store = instantiate(staged.config.store)
    task = instantiate(staged.config.task)
    lifecycle = instantiate(staged.config.lifecycle)
    assert isinstance(allocation, ConfiguredAllocation)
    assert isinstance(store, DirectoryStore)
    assert isinstance(task, ConfiguredRun)
    assert isinstance(lifecycle, AttachedAllocation)
