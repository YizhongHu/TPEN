"""B1a-C fake-only successor consumption and multi-generation publication."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import AllocationAttemptV2, ConfiguredAllocation, MemberInvocationV2
from tpen_executor.contracts import AllocationAttemptId, AllocationId, AttemptId, BackendJobId, GroupId, InvocationId, ReadyGroup
from tpen_executor.recovery import MemberRecoveryInputV2
from tpen_executor.results import StartReceipt, SuccessorAdmissionRecord
from tpen_executor.storage import AdmissionFork, DirectoryStore, FenceRejected

from test_b1a_b_publication import _fixture
from test_b1a_reconciliation import _core, _group


def test_admitted_successor_consumes_exact_identity_and_replays(tmp_path):
    store, group, lease, inputs, bindings, successors, fences = _fixture(tmp_path)
    request, acceptance, successor = successors["successor"]
    admission = store.record_successor_admission(
        fences["successor"], request, acceptance,
        predecessor_binding=bindings["successor"], successor_binding=successor,
    )
    allocation = ConfiguredAllocation(successor.allocation_attempt.allocation_id,
                                      successor.allocation_attempt.backend_job_id, {})
    started = store.begin_admitted_successor(
        group, "successor", allocation, "successor-start", "remote",
        allocation_attempt=successor.allocation_attempt, admission=admission,
    )
    assert started.attempt_id == successor.attempt_id
    assert started.output_dir.endswith(successor.attempt_id.value)
    assert MemberInvocationV2.decode(Path(started.output_dir, "invocation.json").read_bytes()) == successor
    replay = store.begin_admitted_successor(
        group, "successor", allocation, "successor-start", "remote",
        allocation_attempt=successor.allocation_attempt, admission=admission,
    )
    assert replay == started
    with pytest.raises((AdmissionFork, FenceRejected)):
        store.begin_admitted_successor(
            group, "successor", allocation, "different-start", "remote",
            allocation_attempt=successor.allocation_attempt, admission=admission,
        )


def test_admitted_successor_requires_durable_admission(tmp_path):
    store, group, _lease, _inputs, bindings, successors, fences = _fixture(tmp_path)
    request, acceptance, successor = successors["successor"]
    admission = SuccessorAdmissionRecord(
        "successor", fences["successor"].attempt_id, request["core_evidence"]["checkpoint_ref"]["content_id"],
        "successor", successor.attempt_id, digest(request), acceptance, successor,
    )
    allocation = ConfiguredAllocation(successor.allocation_attempt.allocation_id,
                                      successor.allocation_attempt.backend_job_id, {})
    with pytest.raises((AdmissionFork, FenceRejected)):
        store.begin_admitted_successor(
            group, "successor", allocation, "start", "remote",
            allocation_attempt=successor.allocation_attempt, admission=admission,
        )


@pytest.mark.parametrize("field", ["group_digest", "task_id", "lineage_id", "allocation_attempt", "allocation"])
def test_admitted_successor_crossed_binding_fields_refuse(tmp_path, field):
    store, group, _lease, _inputs, bindings, successors, fences = _fixture(tmp_path)
    request, acceptance, successor = successors["successor"]
    admission = store.record_successor_admission(
        fences["successor"], request, acceptance,
        predecessor_binding=bindings["successor"], successor_binding=successor,
    )
    if field == "group_digest":
        crossed = replace(successor, group_digest="0" * 64)
    elif field == "task_id":
        crossed = replace(successor, task_id="foreign-task")
    elif field == "lineage_id":
        crossed = replace(successor, lineage_id="foreign-lineage")
    elif field == "allocation_attempt":
        crossed = replace(successor, allocation_attempt=replace(successor.allocation_attempt,
                                                               allocation_attempt_id=AllocationAttemptId("foreign-lease")))
    else:
        crossed = replace(successor, allocation_attempt=replace(successor.allocation_attempt,
                                                               allocation_id=AllocationId("foreign-allocation")))
    crossed_admission = replace(admission, successor_binding=crossed)
    allocation = ConfiguredAllocation(successor.allocation_attempt.allocation_id,
                                      successor.allocation_attempt.backend_job_id, {})
    with pytest.raises((AdmissionFork, FenceRejected)):
        store.begin_admitted_successor(
            group, "successor", allocation, "start", "remote",
            allocation_attempt=successor.allocation_attempt, admission=crossed_admission,
        )


def test_admitted_successor_predecessor_not_current_refuses(tmp_path):
    store, group, lease, _inputs, bindings, successors, fences = _fixture(tmp_path)
    request, acceptance, successor = successors["successor"]
    admission = store.record_successor_admission(
        fences["successor"], request, acceptance,
        predecessor_binding=bindings["successor"], successor_binding=successor,
    )
    allocation = ConfiguredAllocation(AllocationId("other-allocation"), BackendJobId("other-job"), {})
    other_lease = replace(lease, allocation_attempt_id=AllocationAttemptId("other-lease"),
                          allocation_id=allocation.allocation_id, backend_job_id=allocation.backend_job_id)
    store.begin_attempt(group, "successor", allocation, "other-start",
                        allocation_attempt=other_lease, invocation_id=InvocationId("other-invocation"))
    with pytest.raises(FenceRejected):
        store.begin_admitted_successor(
            group, "successor", ConfiguredAllocation(successor.allocation_attempt.allocation_id,
                                                      successor.allocation_attempt.backend_job_id, {}),
            "start", "remote", allocation_attempt=successor.allocation_attempt, admission=admission,
        )


def test_admitted_successor_rejects_v1_group(tmp_path):
    store, group, _lease, _inputs, _bindings, successors, fences = _fixture(tmp_path)
    request, acceptance, successor = successors["successor"]
    admission = store.record_successor_admission(
        fences["successor"], request, acceptance,
        predecessor_binding=_bindings["successor"], successor_binding=successor,
    )
    with pytest.raises(TypeError):
        store.begin_admitted_successor(
            ReadyGroup("legacy", group.members, group.identity), "successor",
            ConfiguredAllocation(successor.allocation_attempt.allocation_id,
                                  successor.allocation_attempt.backend_job_id, {}),
            "start", "remote", allocation_attempt=successor.allocation_attempt, admission=admission,
        )


def test_admitted_successor_different_start_after_completion_refuses(tmp_path):
    store, group, _lease, _inputs, bindings, successors, fences = _fixture(tmp_path)
    request, acceptance, successor = successors["successor"]
    admission = store.record_successor_admission(
        fences["successor"], request, acceptance,
        predecessor_binding=bindings["successor"], successor_binding=successor,
    )
    allocation = ConfiguredAllocation(successor.allocation_attempt.allocation_id,
                                      successor.allocation_attempt.backend_job_id, {})
    fence = store.begin_admitted_successor(
        group, "successor", allocation, "start", "remote",
        allocation_attempt=successor.allocation_attempt, admission=admission,
    )
    store.finish_attempt(fence, "completed", "done")
    with pytest.raises((AdmissionFork, FenceRejected)):
        store.begin_admitted_successor(
            group, "successor", allocation, "different-start", "remote",
            allocation_attempt=successor.allocation_attempt, admission=admission,
        )


def test_publication_accepts_prior_attempts_under_new_controller_lease(tmp_path):
    group = _group(("a", "b"))
    store = DirectoryStore(tmp_path / "store")
    inputs = []
    bindings = {}
    for index, task_id in enumerate(("a", "b"), 1):
        allocation = ConfiguredAllocation(AllocationId(f"allocation-{index}"), BackendJobId(f"job-{index}"), {})
        lease = AllocationAttemptV2(group.group_id, AllocationAttemptId(f"lease-{index}"),
                                    allocation.allocation_id, allocation.backend_job_id,
                                    f"controller-{index}", f"utc-{index}")
        fence = store.begin_attempt(group, task_id, allocation, f"start-{index}",
                                    allocation_attempt=lease, invocation_id=InvocationId(f"invocation-{index}"))
        outcome = store.finish_attempt(fence, "completed", f"finish-{index}")
        binding = MemberInvocationV2.decode(Path(fence.output_dir, "invocation.json").read_bytes())
        bindings[task_id] = binding
        inputs.append(MemberRecoveryInputV2(task_id, binding.lineage_id, fence.attempt_id,
                                            StartReceipt.decode(Path(fence.output_dir, "start.json").read_bytes()),
                                            outcome, _core("completed")))
    controller = AllocationAttemptV2(group.group_id, AllocationAttemptId("lease-3"),
                                     AllocationId("allocation-3"), BackendJobId("job-3"), "controller-3", "utc-3")
    result = store.publish_reconciliation(group, controller, tuple(inputs), bindings=bindings)
    assert result.aggregate == "settled"
    record = store.read_reconciliation(group, controller)
    assert record is not None
    encoded = record.encode()
    assert store.publish_reconciliation(group, controller, tuple(inputs), bindings=bindings).encode() == encoded
    assert [member.binding.allocation_attempt for member in record.members] == [bindings["a"].allocation_attempt, bindings["b"].allocation_attempt]


def test_other_group_binding_parks(tmp_path):
    fixture = list(_fixture(tmp_path))
    store, group, lease, inputs, bindings, successors, _fences = fixture
    binding = bindings["successor"]
    other_group = GroupId("other-group")
    bindings["successor"] = replace(binding, allocation_attempt=replace(binding.allocation_attempt, group_id=other_group))
    assert store.publish_reconciliation(group, lease, inputs, bindings=bindings, successors=successors).aggregate == "parked"


def test_tampered_invocation_parks(tmp_path):
    fixture = list(_fixture(tmp_path))
    store, group, lease, inputs, bindings, successors, fences = fixture
    path = Path(fences["successor"].output_dir, "invocation.json")
    tampered = replace(bindings["successor"], invocation_id=InvocationId("tampered"))
    path.write_bytes(tampered.encode())
    assert store.publish_reconciliation(group, lease, inputs, bindings=bindings, successors=successors).aggregate == "parked"


def test_start_binding_allocation_mismatch_parks(tmp_path):
    fixture = list(_fixture(tmp_path))
    store, group, lease, inputs, bindings, successors, _fences = fixture
    binding = bindings["successor"]
    bindings["successor"] = replace(binding, allocation_attempt=replace(binding.allocation_attempt, allocation_id=AllocationId("wrong")))
    assert store.publish_reconciliation(group, lease, inputs, bindings=bindings, successors=successors).aggregate == "parked"


def test_existing_single_attempt_publication_encoding_is_pinned(tmp_path):
    fixture = _fixture(tmp_path, ("completed", "terminal"))
    store, group, lease, inputs, bindings, successors, _fences = fixture
    result = store.publish_reconciliation(group, lease, inputs, bindings=bindings, successors=successors)
    record = store.read_reconciliation(group, lease)
    assert record is not None and record.reconciliation == result
    assert record.encode() == type(record).decode(record.encode()).encode()
    assert all(member.binding is None or member.binding.allocation_attempt == lease for member in record.members)


def test_mutant_anchor_uniqueness_b1ac():
    source = Path(__file__).parents[1] / "src" / "tpen_executor"
    names = set()
    anchors = set()
    for name, filename, before, _after, killer in MUTANTS:
        assert name not in names
        names.add(name)
        assert (filename, before) not in anchors, name
        anchors.add((filename, before))
        assert (source / filename).read_text().count(before) == 1, name
        assert any(killer in path.read_text() for path in Path(__file__).parent.glob("test_*.py")), name


MUTANTS = (
    ("uuid4-admitted-successor", "storage.py", "output_dir = _lexical(os.path.join(task_dir, successor.attempt_id.value))", "output_dir = _lexical(os.path.join(task_dir, AttemptId(uuid4().hex).value))", "test_admitted_successor_consumes_exact_identity_and_replays"),
    ("skip-admission-reread", "storage.py", "records = self._admission_records_for_task(task_dir, task_id)", "records = ()", "test_admitted_successor_requires_durable_admission"),
    ("force-current-lease", "storage.py", "binding.allocation_attempt.group_id != group.group_id", "False", "test_other_group_binding_parks"),
    ("restore-single-lease-gate", "storage.py", "if (binding.allocation_attempt.group_id != group.group_id\n                            or binding.lineage_id != task.lineage_id", "if binding.allocation_attempt != allocation_attempt:", "test_publication_accepts_prior_attempts_under_new_controller_lease"),
    ("drop-replay-byte-compare", "storage.py", "or Path(output_dir, \"invocation.json\").read_bytes() != invocation_bytes", "or False", "test_admitted_successor_consumes_exact_identity_and_replays"),
    ("crossed-successor-group", "storage.py", "or successor.group_digest != group.identity", "or False", "test_admitted_successor_crossed_binding_fields_refuse"),
    ("crossed-successor-task", "storage.py", "or successor.task_id != task_id", "or False", "test_admitted_successor_crossed_binding_fields_refuse"),
    ("crossed-successor-lineage", "storage.py", "or successor.lineage_id != next(member.lineage_id for member in group.members if member.task_id == task_id)", "or False", "test_admitted_successor_crossed_binding_fields_refuse"),
    ("crossed-successor-attempt", "storage.py", "or successor.allocation_attempt != allocation_attempt", "or False", "test_admitted_successor_crossed_binding_fields_refuse"),
    ("predecessor-current-gate", "storage.py", "if current != admission.predecessor_attempt_id.value:", "if False:", "test_admitted_successor_predecessor_not_current_refuses"),
)
