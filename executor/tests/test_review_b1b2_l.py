"""Fakes-only storage contract checks for B1b2-L."""

from pathlib import Path

import pytest

import tpen_executor.storage as storage_module
from tpen_executor.allocations import AllocationAttemptV2, ConfiguredAllocation, ControllerGenerationV1, LaunchIntentV1, MemberInvocationV2
from tpen_executor.contracts import AllocationAttemptId, AllocationId, AttemptId, BackendJobId, ExecutionPlanV2, GroupId, InvocationId, ProgramSpecRef, ReadyGroupV2, TaskSpec
from tpen_executor._codec import digest
from tpen_executor.storage import ControllerGenerationRejected, DirectoryStore, LaunchIntentRejected

MUTANTS = (
    ("monotonic-check-off", "executor/src/tpen_executor/storage.py", "record.generation != expected", "False", "test_generation_rejects_skipped_or_stale"),
    ("predecessor-check-off", "executor/src/tpen_executor/storage.py", "record.predecessor_record_digest != predecessor", "False", "test_generation_rejects_forked_predecessor"),
    ("nonce-check-off", "executor/src/tpen_executor/storage.py", "records[-1].claimant_nonce != nonce", "False", "test_require_current_controller_rejects_wrong_nonce"),
    ("replay-byte-compare-off", "executor/src/tpen_executor/storage.py", "record.encode() == current.encode()", "True", "test_generation_rejects_differing_replay"),
    ("intent-stale-check-off", "executor/src/tpen_executor/storage.py", "self.require_current_controller(controller_record.group_id, controller_record.generation, controller_record.claimant_nonce)", "None", "test_launch_intent_stale_controller_refuses"),
)

# No intent-exclusivity row: _write_new refuses an existing destination and create_launch_intent maps FileExistsError to the same LaunchIntentRejected.


def _group():
    config = {"program": "fake"}; program = ProgramSpecRef(config, digest(config))
    task = TaskSpec("task", "lineage", program, "a" * 64, 1, "steps")
    plan = ExecutionPlanV2(config, digest(config), ("task",), "continue", "preserve")
    return ReadyGroupV2(GroupId("group"), (task,), plan)


def _record(group, generation=1, nonce="nonce", predecessor=None):
    return ControllerGenerationV1(group.group_id, generation, AllocationId("allocation"), BackendJobId("job"), "b" * 64, nonce, predecessor)


def _started(tmp_path):
    group = _group(); store = DirectoryStore(tmp_path)
    allocation = ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})
    attempt = AllocationAttemptV2(group.group_id, AllocationAttemptId("allocation-attempt"), allocation.allocation_id, allocation.backend_job_id, "opaque", "now")
    invocation = InvocationId("invocation")
    fence = store.begin_attempt(group, "task", allocation, allocation_attempt=attempt, invocation_id=invocation)
    return group, store, fence, MemberInvocationV2(attempt, group.identity, "task", "lineage", fence.attempt_id, invocation)


def test_generation_1_acquire_and_monotonic_successor(tmp_path):
    group = _group(); store = DirectoryStore(tmp_path); first = store.acquire_controller_generation(group, _record(group))
    second = store.acquire_controller_generation(group, _record(group, 2, "next", first.record_digest))
    assert (first.generation, second.generation) == (1, 2)


def test_generation_rejects_skipped_or_stale(tmp_path):
    group = _group(); store = DirectoryStore(tmp_path); first = store.acquire_controller_generation(group, _record(group))
    with pytest.raises(ControllerGenerationRejected): store.acquire_controller_generation(group, _record(group, 3, "skip", first.record_digest))
    with pytest.raises(ControllerGenerationRejected): store.acquire_controller_generation(group, _record(group, 1, "old"))


def test_generation_rejects_forked_predecessor(tmp_path):
    group = _group(); store = DirectoryStore(tmp_path); store.acquire_controller_generation(group, _record(group))
    with pytest.raises(ControllerGenerationRejected): store.acquire_controller_generation(group, _record(group, 2, "fork", "c" * 64))


def test_generation_identical_replay_is_idempotent(tmp_path):
    group = _group(); store = DirectoryStore(tmp_path); record = _record(group)
    assert store.acquire_controller_generation(group, record) == store.acquire_controller_generation(group, record)


def test_generation_rejects_differing_replay(tmp_path):
    group = _group(); store = DirectoryStore(tmp_path); store.acquire_controller_generation(group, _record(group))
    with pytest.raises(ControllerGenerationRejected): store.acquire_controller_generation(group, _record(group, nonce="different"))


def test_require_current_controller_rejects_wrong_nonce(tmp_path):
    group = _group(); store = DirectoryStore(tmp_path); first = store.acquire_controller_generation(group, _record(group)); store.acquire_controller_generation(group, _record(group, 2, "next", first.record_digest))
    assert store.require_current_controller(group, 2, "next").generation == 2
    with pytest.raises(ControllerGenerationRejected): store.require_current_controller(group, 1, "nonce")
    with pytest.raises(ControllerGenerationRejected): store.require_current_controller(group, 2, "wrong")


def test_generation_survives_reopen(tmp_path):
    group = _group(); store = DirectoryStore(tmp_path); record = store.acquire_controller_generation(group, _record(group))
    assert DirectoryStore(tmp_path).require_current_controller(group, 1, "nonce") == record


def test_generation_crash_before_link_leaves_no_visible_record(tmp_path, monkeypatch):
    group = _group(); store = DirectoryStore(tmp_path)
    monkeypatch.setattr(storage_module.os, "link", lambda *args: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError): store.acquire_controller_generation(group, _record(group))
    assert not list(Path(store.root).rglob("generation-*.json"))


def test_launch_intent_created_once_and_read_after_reopen(tmp_path):
    group, store, fence, binding = _started(tmp_path); controller = _record(group); store.acquire_controller_generation(group, controller)
    intent = store.create_launch_intent(fence, controller, binding)
    assert store.read_launch_intent(fence) == intent
    with pytest.raises(LaunchIntentRejected): store.create_launch_intent(fence, controller, binding)
    assert DirectoryStore(tmp_path).read_launch_intent(fence) == intent


def test_launch_intent_stale_controller_refuses(tmp_path):
    group, store, fence, binding = _started(tmp_path); first = _record(group); store.acquire_controller_generation(group, first); store.acquire_controller_generation(group, _record(group, 2, "next", first.record_digest))
    with pytest.raises(ControllerGenerationRejected): store.create_launch_intent(fence, first, binding)


def test_controller_generation_wire_is_canonical_and_strict():
    record = ControllerGenerationV1(GroupId("group"), 1, AllocationId("allocation"), BackendJobId("job"), "0" * 64, "nonce", None)
    assert ControllerGenerationV1.decode(record.encode()) == record
    with pytest.raises(ValueError): ControllerGenerationV1.decode(record.to_dict() | {"unexpected": True})


def test_launch_intent_wire_is_canonical_and_strict():
    record = LaunchIntentV1(GroupId("group"), "task", AttemptId("attempt"), InvocationId("invocation"), 1, "nonce", AllocationId("allocation"), BackendJobId("job"))
    assert LaunchIntentV1.decode(record.encode()) == record
    with pytest.raises(ValueError): LaunchIntentV1.decode(record.to_dict() | {"unexpected": True})


def test_mutant_anchors_are_unique_compile_and_name_real_killers():
    source = Path(__file__).parents[1].joinpath("src/tpen_executor/storage.py").read_text()
    names = {name for name, value in globals().items() if name.startswith("test_") and callable(value)}
    anchors = [row[2] for row in MUTANTS]
    assert len(anchors) == len(set(anchors)) >= 5
    for _, filename, before, after, killer in MUTANTS:
        assert filename.endswith("storage.py") and source.count(before) == 1 and killer in names
        compile(source.replace(before, after, 1), filename, "exec")
