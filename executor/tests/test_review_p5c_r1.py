"""Independent P5c review probes; RED asserts proposed corrected behavior."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import pytest

from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.results import OutcomeReceipt, StartReceipt
from tpen_executor.storage import DirectoryStore, FenceRejected
import tpen_executor.storage as storage


def _group(*ids):
    program = ProgramSpecRef({}, digest({}))
    return ReadyGroup("review", tuple(TaskSpec(i, i, program, digest(i), 1, "steps") for i in ids), digest({}))


def _allocation(name="one", facts=None):
    return ConfiguredAllocation(AllocationId(name), BackendJobId(name), {} if facts is None else facts)


def _join(process):
    process.join(20)
    assert not process.is_alive(), "bounded child did not finish"
    assert process.exitcode == 0, f"child failed: {process.exitcode}"


def test_r1_t1_begin_cannot_make_an_inflight_finish_publish_stale(tmp_path):
    """Serialize begin/finish or reject finish when begin wins the commit race."""
    ctx = mp.get_context("fork")
    reached, release, begun = ctx.Event(), ctx.Event(), ctx.Event()
    results = ctx.Queue()
    group, allocation = _group("a"), _allocation()
    store = DirectoryStore(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "start")

    def finisher():
        real = storage._write_new

        def paused(path, payload):
            if path.endswith("outcome.json"):
                reached.set()
                assert release.wait(10), "parent did not release finish"
            return real(path, payload)

        storage._write_new = paused
        try:
            store.finish_attempt(fence, "completed", "finish")
            results.put(("finish", "accepted"))
        except FenceRejected:
            results.put(("finish", "rejected"))

    def beginner():
        second = DirectoryStore(tmp_path).begin_attempt(group, "a", allocation, "second")
        results.put(("begin", second.attempt_id.value))
        begun.set()

    first = ctx.Process(target=finisher)
    first.start()
    assert reached.wait(10), "finish did not reach publication"
    second = ctx.Process(target=beginner)
    second.start()
    try:
        begin_won = begun.wait(2)
        if begin_won:
            # Observable after begin returns: A1 has still not completed.
            previous = next(a for a in store.manifest(group).attempts if a.attempt_id == fence.attempt_id)
            assert previous.outcome_receipt is None
    finally:
        release.set()
        _join(first)
        _join(second)
    observed = dict(results.get(timeout=5) for _ in range(2))
    if begin_won:
        assert observed["finish"] == "rejected", "finish published after a newer begin returned and manifest observed A1 unfinished"
        assert not Path(fence.output_dir, "outcome.json").exists()
    else:
        assert observed["finish"] == "accepted"


@pytest.mark.parametrize("receipt", ["start.json", "outcome.json"], ids=["start", "outcome"])
def test_r1_t2_abrupt_exit_cannot_publish_an_empty_receipt(tmp_path, receipt):
    """Kill the writer at publication; a final receipt must be absent or valid."""
    ctx = mp.get_context("fork")
    store, group, allocation = DirectoryStore(tmp_path), _group("a"), _allocation()
    fence = store.begin_attempt(group, "a", allocation, "start") if receipt == "outcome.json" else None

    def writer():
        real = storage.os.link

        def abrupt(source, destination, *args, **kwargs):
            if str(destination).endswith(receipt):
                os._exit(73)
            return real(source, destination, *args, **kwargs)

        storage.os.link = abrupt
        if fence is None:
            store.begin_attempt(group, "a", allocation, "start")
        else:
            store.finish_attempt(fence, "completed", "finish")

    child = ctx.Process(target=writer)
    child.start()
    child.join(20)
    assert not child.is_alive()
    assert child.exitcode == 73, "crash hook was not reached"
    final_files = list(tmp_path.rglob(receipt))
    assert not final_files or all(p.stat().st_size > 0 for p in final_files), "O_EXCL claim exposed a 0-byte final receipt after abrupt exit"
    for path in final_files:
        (StartReceipt if receipt == "start.json" else OutcomeReceipt).decode(path.read_bytes())


def _trace_io(monkeypatch):
    events = []
    real_mkdir, real_replace, real_fsync = os.mkdir, os.replace, os.fsync

    def mkdir(path, *args, **kwargs):
        result = real_mkdir(path, *args, **kwargs)
        events.append(("mkdir", os.path.abspath(os.fspath(path))))
        return result

    def replace_file(source, destination):
        result = real_replace(source, destination)
        events.append(("replace", os.path.abspath(os.fspath(destination))))
        return result

    def fsync(fd):
        result = real_fsync(fd)
        events.append(("fsync", os.readlink(f"/proc/self/fd/{fd}")))
        return result

    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(os, "replace", replace_file)
    monkeypatch.setattr(os, "fsync", fsync)
    return events


def test_r1_t3a_pointer_replace_is_followed_by_parent_fsync(tmp_path, monkeypatch):
    """Observe the required durability syscall; this is not a power-loss test."""
    store = DirectoryStore(tmp_path)
    events = _trace_io(monkeypatch)
    fence = store.begin_attempt(_group("a"), "a", _allocation(), "start")
    parent = str(Path(fence.output_dir).parent)
    index = events.index(("replace", str(Path(parent, "current-attempt.json"))))
    assert ("fsync", parent) in events[index + 1:], f"pointer parent was not fsynced after replacement: {events!r}"


def test_r1_t3b_new_directory_links_are_durable_before_begin_returns(tmp_path, monkeypatch):
    """Every newly created store directory needs its parent link persisted."""
    store = DirectoryStore(tmp_path)
    events = _trace_io(monkeypatch)
    store.begin_attempt(_group("a"), "a", _allocation(), "start")
    missing = [path for index, (kind, path) in enumerate(events) if kind == "mkdir" and ("fsync", str(Path(path).parent)) not in events[index + 1:]]
    assert not missing, f"new directories lack parent fsync: {missing!r}; trace={events!r}"


def test_r1_t4_manifest_rejects_a_receipt_output_path_that_disagrees_with_disk(tmp_path):
    """A redundant receipt path must match its actual fenced directory."""
    store, group = DirectoryStore(tmp_path / "store"), _group("a")
    fence = store.begin_attempt(group, "a", _allocation(), "start")
    path = Path(fence.output_dir, "start.json")
    mapping = StartReceipt.decode(path.read_bytes()).to_dict()
    mapping["output_dir"] = str(tmp_path / "foreign")
    path.write_bytes(canonical_bytes(mapping))
    with pytest.raises((ValueError, FenceRejected)):
        store.manifest(group)


@pytest.mark.parametrize("field", ["task_id", "attempt_id", "group_digest", "output_dir"], ids=["task", "attempt", "group", "path"])
def test_r1_t5_finish_rejects_every_crossed_start_binding(tmp_path, field):
    """All start bindings, not just allocation_id, must agree with the fence."""
    store, group = DirectoryStore(tmp_path), _group("a")
    fence = store.begin_attempt(group, "a", _allocation(), "start")
    path = Path(fence.output_dir, "start.json")
    mapping = StartReceipt.decode(path.read_bytes()).to_dict()
    mapping[field] = digest("foreign") if field == "group_digest" else (str(tmp_path / "foreign") if field == "output_dir" else "foreign")
    path.write_bytes(canonical_bytes(mapping))
    original = path.read_bytes()
    with pytest.raises(FenceRejected):
        store.finish_attempt(fence, "completed", "finish")
    assert path.read_bytes() == original
    assert not Path(fence.output_dir, "outcome.json").exists()


def test_r1_t6_existing_symlink_cannot_redirect_store_writes(tmp_path):
    """An in-root task symlink must not write into a sibling directory."""
    root, outside = tmp_path / "store", tmp_path / "outside"
    outside.mkdir()
    store, group = DirectoryStore(root), _group("a")
    group_dir = root / "groups" / group.identity
    group_dir.mkdir(parents=True)
    (group_dir / "a").symlink_to(outside, target_is_directory=True)
    try:
        store.begin_attempt(group, "a", _allocation(), "start")
    except (FenceRejected, OSError, ValueError):
        pass
    assert list(outside.iterdir()) == [], "store followed a task symlink and wrote outside its root"


@pytest.mark.parametrize("task_id", ["train/a", "./a"], ids=["slash", "dot-alias"])
def test_r1_t7_task_namespace_is_rejected_early_or_round_trips(tmp_path, task_id):
    """Text task IDs must either map reversibly or be refused before writes."""
    store, group = DirectoryStore(tmp_path), _group(task_id)
    try:
        fence = store.begin_attempt(group, task_id, _allocation(), "start")
    except (ValueError, FenceRejected):
        assert list(tmp_path.iterdir()) == [], "invalid task id left store artifacts"
        return
    store.finish_attempt(fence, "completed", "finish")
    manifest = DirectoryStore(tmp_path).manifest(group)
    assert len(manifest.attempts) == 1
    assert manifest.attempts[0].task_id == task_id


@pytest.mark.parametrize(("left", "right"), [(1, True), (1, 1.0), (0.0, -0.0)], ids=["int-bool", "int-float", "signed-zero"])
def test_r1_t8_equal_allocation_facts_have_equal_hashes(left, right):
    """Python mapping equality determines dataclass equality, including numbers."""
    first, second = _allocation(facts={"nested": [left]}), _allocation(facts={"nested": [right]})
    assert first == second
    assert hash(first) == hash(second), "equal ConfiguredAllocation records have unequal hashes"
    assert second in {first}


@pytest.mark.parametrize("stamp", [None, False, 0, []], ids=["none", "false", "zero", "list"])
def test_r1_t9_falsy_wrong_timestamp_types_are_rejected_before_writes(tmp_path, stamp):
    """Only the documented empty-string default may use its fallback."""
    store = DirectoryStore(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        store.begin_attempt(_group("a"), "a", _allocation(), stamp)
    assert list(tmp_path.iterdir()) == []


def test_r1_t10_attached_lifecycle_cannot_finish_another_allocation(tmp_path):
    """A lifecycle binds both begin and finish to its configured allocation."""
    store, group = DirectoryStore(tmp_path), _group("a")
    owner = AttachedAllocation(_allocation("owner"), store)
    other = AttachedAllocation(_allocation("other"), store)
    fence = owner.begin(group, "a", "start")
    with pytest.raises(FenceRejected):
        other.finish(fence, "completed", "finish")
    assert not Path(fence.output_dir, "outcome.json").exists()


def test_r1_t11a_two_processes_finish_exactly_once(tmp_path):
    """Control: concurrent completers publish one whole outcome, never overwrite."""
    ctx = mp.get_context("fork")
    ready, results = ctx.Barrier(2), ctx.Queue()
    store, group = DirectoryStore(tmp_path), _group("a")
    fence = store.begin_attempt(group, "a", _allocation(), "start")

    def writer(status):
        ready.wait(10)
        try:
            receipt = DirectoryStore(tmp_path).finish_attempt(fence, status, status)
            results.put(("accepted", receipt.encode()))
        except FenceRejected:
            results.put(("rejected", None))

    children = [ctx.Process(target=writer, args=(status,)) for status in ("completed", "failed")]
    for child in children:
        child.start()
    for child in children:
        _join(child)
    observed = [results.get(timeout=5) for _ in children]
    assert sorted(row[0] for row in observed) == ["accepted", "rejected"]
    winner = next(row[1] for row in observed if row[0] == "accepted")
    assert Path(fence.output_dir, "outcome.json").read_bytes() == winner
    assert len(store.manifest(group).attempts) == 1


def test_r1_t11b_start_is_readable_before_pointer_publication(tmp_path, monkeypatch):
    """Control: pointer never references a start that has not been published."""
    seen = []
    real = storage._write_pointer

    def checked(path, attempt_id):
        receipt = StartReceipt.decode(Path(path).parent.joinpath(attempt_id, "start.json").read_bytes())
        assert receipt.attempt_id.value == attempt_id
        seen.append(receipt)
        return real(path, attempt_id)

    monkeypatch.setattr(storage, "_write_pointer", checked)
    DirectoryStore(tmp_path).begin_attempt(_group("a"), "a", _allocation(), "start")
    assert len(seen) == 1


@pytest.mark.parametrize("damage", ["task-temp", "partial-attempt", "empty-outcome"], ids=["task-temp", "partial-dir", "empty-claim"])
def test_r1_t11c_manifest_refuses_foreign_or_partial_entries(tmp_path, damage):
    """Control: strict reconstruction surfaces invalid task entries and claims."""
    store, group = DirectoryStore(tmp_path), _group("a")
    fence = store.begin_attempt(group, "a", _allocation(), "start")
    if damage == "task-temp":
        Path(fence.output_dir).parent.joinpath(".tmp-pointer-stray").write_bytes(b"partial")
    elif damage == "partial-attempt":
        Path(fence.output_dir).parent.joinpath("partial-attempt").mkdir()
    else:
        Path(fence.output_dir, "outcome.json").touch()
    with pytest.raises((ValueError, TypeError, OSError)):
        store.manifest(group)


def test_r1_t11d_manifest_is_receipt_only_when_pointer_is_corrupt(tmp_path):
    """Control records current behavior: pointer is not an authoritative receipt."""
    store, group = DirectoryStore(tmp_path), _group("a")
    fence = store.begin_attempt(group, "a", _allocation(), "start")
    before = store.manifest(group).encode()
    Path(fence.output_dir).parent.joinpath("current-attempt.json").write_bytes(b"{")
    assert store.manifest(group).encode() == before
    with pytest.raises(FenceRejected):
        store.finish_attempt(fence, "completed", "finish")
