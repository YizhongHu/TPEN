"""Proposed T2 behavior tests; not executed by the read-only reviewer."""

from dataclasses import replace
from pathlib import Path

import pytest

import tpen_executor.storage as storage
from tpen_executor.allocations import AllocationAttemptV2, ConfiguredAllocation, MemberInvocationV2
from tpen_executor.contracts import AllocationAttemptId, AllocationId, BackendJobId, InvocationId
from tpen_executor.recovery import MemberRecoveryInputV2
from tpen_executor.results import ReconciliationPublicationV2, StartReceipt
from tpen_executor.storage import DirectoryStore

from test_b1a_b_publication import _fixture
from test_b1a_reconciliation import _core, _group


class _InterruptedStart(RuntimeError):
    """Simulate process loss after one durable operation has completed."""


@pytest.mark.parametrize("boundary", ["mkdir", "start.json", "invocation.json", "pointer"])
def test_admitted_start_recovers_after_durable_boundary(tmp_path, monkeypatch, boundary):
    store, group, _, _, bindings, successors, fences = _fixture(tmp_path)
    request, acceptance, successor = successors["successor"]
    admission = store.record_successor_admission(
        fences["successor"], request, acceptance,
        predecessor_binding=bindings["successor"], successor_binding=successor,
    )
    allocation = ConfiguredAllocation(
        successor.allocation_attempt.allocation_id,
        successor.allocation_attempt.backend_job_id, {},
    )
    output = Path(fences["successor"].output_dir).parent / successor.attempt_id.value

    def start(target):
        return target.begin_admitted_successor(
            group, "successor", allocation, "2026-11-20T00:00:00Z", "remote",
            allocation_attempt=successor.allocation_attempt, admission=admission,
        )

    original_mkdir = storage._mkdir_new
    original_write = storage._write_new
    original_pointer = storage._write_pointer

    def interrupted_mkdir(path):
        original_mkdir(path)
        if boundary == "mkdir" and Path(path) == output:
            raise _InterruptedStart(boundary)

    def interrupted_write(path, payload, **kwargs):
        original_write(path, payload, **kwargs)
        if Path(path).parent == output and Path(path).name == boundary:
            raise _InterruptedStart(boundary)

    def interrupted_pointer(path, attempt):
        original_pointer(path, attempt)
        if boundary == "pointer" and attempt == successor.attempt_id.value:
            raise _InterruptedStart(boundary)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_mkdir_new", interrupted_mkdir)
        patch.setattr(storage, "_write_new", interrupted_write)
        patch.setattr(storage, "_write_pointer", interrupted_pointer)
        with pytest.raises(_InterruptedStart):
            start(store)

    # A reopened store must complete the same admitted start, without deleting
    # partial evidence, inventing a fresh identity, or overwriting written bytes.
    written = {path: path.read_bytes() for path in output.glob("*.json")}
    reopened = DirectoryStore(store.root)
    fence = start(reopened)
    assert fence.attempt_id == successor.attempt_id
    assert Path(fence.output_dir) == output
    assert MemberInvocationV2.decode((output / "invocation.json").read_bytes()) == successor
    assert all(path.read_bytes() == content for path, content in written.items())
    assert start(reopened) == fence
    # The returned fence must actually be current, not merely look identical.
    reopened.finish_attempt(fence, "completed", "2026-11-20T00:01:00Z")


def _completed_member(store, group, task_id, lease, invocation):
    """Create genuine durable evidence, rather than editing an invocation echo."""
    allocation = ConfiguredAllocation(lease.allocation_id, lease.backend_job_id, {})
    fence = store.begin_attempt(
        group, task_id, allocation, "2026-11-20T00:00:00Z",
        allocation_attempt=lease, invocation_id=InvocationId(invocation),
    )
    outcome = store.finish_attempt(fence, "completed", "2026-11-20T00:01:00Z")
    start = StartReceipt.decode(Path(fence.output_dir, "start.json").read_bytes())
    binding = MemberInvocationV2.decode(Path(fence.output_dir, "invocation.json").read_bytes())
    return MemberRecoveryInputV2(
        task_id, binding.lineage_id, fence.attempt_id, start, outcome, _core("completed"),
    ), binding


@pytest.mark.parametrize("location", ["controller-member", "member-member", "member-successor"])
def test_conflicting_lease_facts_park_before_publication(tmp_path, location):
    if location == "controller-member":
        store, group, lease, inputs, bindings, successors, _ = _fixture(tmp_path, ("completed",))
        # The same ID must not denote a different controller generation.
        controller = replace(lease, controller_generation="conflicting-controller")
    elif location == "member-member":
        group = _group(("a", "b"))
        store = DirectoryStore(tmp_path / "store")
        lease = AllocationAttemptV2(
            group.group_id, AllocationAttemptId("shared-lease"),
            AllocationId("allocation"), BackendJobId("job"), "controller", "utc",
        )
        a, a_binding = _completed_member(store, group, "a", lease, "invocation-a")
        b, b_binding = _completed_member(
            store, group, "b", replace(lease, controller_generation="conflicting-controller"),
            "invocation-b",
        )
        inputs, bindings, successors = (a, b), {"a": a_binding, "b": b_binding}, {}
        controller = replace(lease, allocation_attempt_id=AllocationAttemptId("publication-lease"))
    else:
        store, group, controller, inputs, bindings, successors, _ = _fixture(
            tmp_path, ("successor", "completed"),
        )
        proposed = successors["successor"][2].allocation_attempt
        completed, binding = _completed_member(
            store, group, "completed", replace(proposed, controller_generation="conflicting-controller"),
            "completed-other-generation",
        )
        inputs = (inputs[0], completed)
        bindings["completed"] = binding

    snapshot = {path: path.read_bytes() for path in Path(store.root).rglob("*.json")}
    result = store.publish_reconciliation(
        group, controller, inputs, bindings=bindings, successors=successors,
    )
    assert result.aggregate == "parked"
    assert store.read_reconciliation(group, controller) is None
    assert {path: path.read_bytes() for path in Path(store.root).rglob("*.json")} == snapshot


@pytest.mark.parametrize("duplicate_invocation", [False, True])
def test_shared_prior_lease_keeps_unique_invocations(tmp_path, duplicate_invocation):
    group = _group(("a", "b"))
    store = DirectoryStore(tmp_path / "store")
    lease = AllocationAttemptV2(
        group.group_id, AllocationAttemptId("prior-lease"),
        AllocationId("allocation"), BackendJobId("job"), "controller", "utc",
    )
    a, a_binding = _completed_member(store, group, "a", lease, "invocation-a")
    b, b_binding = _completed_member(
        store, group, "b", lease, "invocation-a" if duplicate_invocation else "invocation-b",
    )
    controller = replace(lease, allocation_attempt_id=AllocationAttemptId("new-lease"))
    inputs, bindings = (a, b), {"a": a_binding, "b": b_binding}
    result = store.publish_reconciliation(group, controller, inputs, bindings=bindings)
    record = store.read_reconciliation(group, controller)
    if duplicate_invocation:
        assert result.aggregate == "parked"
        assert record is None
    else:
        assert result.aggregate == "settled"
        assert record is not None
        encoded = record.encode()
        assert ReconciliationPublicationV2.decode(encoded).encode() == encoded
        assert store.publish_reconciliation(group, controller, inputs, bindings=bindings) == result
        assert store.read_reconciliation(group, controller).encode() == encoded
