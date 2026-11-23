"""Unrun T2 proposals for PR #798; fake ports with real DirectoryStore.

Review-only local commit. Expected red/green outcomes are source predictions,
not runtime evidence. No implementation changes or cluster runs accompany it.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import AllocationAttemptV2, ConfiguredAllocation, ControllerGenerationV1
from tpen_executor.contracts import AllocationAttemptId, AllocationId, BackendJobId, InvocationId
from tpen_executor.dispatch.transport import LeafRequest
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle.balsam import (
    BalsamGroupRegistrationIntent, BalsamGroupRegistrationReceipt,
    GroupExitContract, make_group_application,
)
from tpen_executor.lifecycle.controller import (
    GroupControllerFactsV1, MemberLaunchResult, run_group_controller,
)
from tpen_executor.recovery import MemberRecoveryInputV2
from tpen_executor.results import ReconciliationPublicationV2, StartReceipt
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.lifecycle.local_executor import LeafCompletionContext

from test_b1a_b_publication import _fixture
from test_b1a_reconciliation import _group
from test_review_b1b2_controller import _real_fresh
from test_s3_transport_wire import _synthetic_result


class EchoAdmission:
    def admit(self, member):
        return member.core_verdict


class RecordingExecutor:
    capacity = 1

    def __init__(self, *, error=None, result=None):
        self.calls = []
        self.collected = []
        self.error = error
        self.result = MemberLaunchResult(True) if result is None else result

    def submit(self, fence, invocation):
        self.calls.append((fence, invocation))
        if self.error is not None:
            raise self.error
        allocation = ConfiguredAllocation(
            AllocationId(fence.allocation_id),
            invocation.allocation_attempt.backend_job_id,
            {},
        )
        configured = ConfiguredRun(invocation.task_id, "/tmp/fake.yaml", ())
        environment = PreparedPythonEnvironment(sys.executable)
        request = LeafRequest.for_attempt(fence, allocation, configured, environment)
        Path(fence.output_dir, "core").mkdir(parents=True, exist_ok=True)
        context = LeafCompletionContext(_synthetic_result(request), request, configured, environment)
        return self.result, context

    def collect(self, handle, timeout):
        assert 0 < timeout <= 6
        self.collected.append(handle)
        return handle


def _binding(group, job):
    return BalsamGroupRegistrationReceipt(
        BalsamGroupRegistrationIntent(group.group_id, group.identity, digest({"review": True})), job,
    )


def test_t2_successor_is_bound_to_current_controller_attempt(tmp_path):
    """RED: an unpersisted proposal must not pin a different allocation forever."""
    store, group, _, inputs, _, requests, _ = _fixture(tmp_path, remote=False)
    old_proposal = requests["successor"][2]
    allocation = ConfiguredAllocation(AllocationId("current-allocation"), BackendJobId("current-job"), {})
    facts = GroupControllerFactsV1(group, allocation, "a" * 64, "current-nonce", "utc", inputs, requests)
    executor = RecordingExecutor()
    outcome = run_group_controller(_binding(group, allocation.backend_job_id), allocation.backend_job_id,
                                   facts, store, EchoAdmission(), executor)
    assert len(executor.calls) == 1 and len(executor.collected) == 1
    fence, invocation = executor.calls[0]
    publications = list(Path(store.root).rglob(".reconciliation-publications/*.json"))
    assert len(publications) == 1
    publication = ReconciliationPublicationV2.decode(publications[0].read_bytes())
    assert invocation.allocation_attempt.allocation_attempt_id != publication.allocation_attempt.allocation_attempt_id
    assert invocation.allocation_attempt.allocation_id == allocation.allocation_id
    assert invocation.allocation_attempt.backend_job_id == allocation.backend_job_id
    assert invocation.invocation_id != old_proposal.invocation_id
    assert invocation.attempt_id == old_proposal.attempt_id == fence.attempt_id
    assert publication.members[0].admission.successor_binding == invocation
    assert outcome.member_outcomes == (("successor", "collected"),)


def test_t2_supersession_after_publication_preserves_newer_start(tmp_path, monkeypatch):
    """RED: refusing an intent must not follow a stale current-pointer write."""
    group, allocation, facts, binding, store = _real_fresh(tmp_path)
    publish = DirectoryStore.publish_reconciliation
    newer_fences = []

    def publish_then_supersede(self, group, attempt, inputs, *, bindings, successors):
        result = publish(self, group, attempt, inputs, bindings=bindings, successors=successors)
        current = self.require_current_controller(group, 1, facts.claimant_nonce)
        newer = ControllerGenerationV1(group.group_id, 2, allocation.allocation_id,
                                       allocation.backend_job_id, "a" * 64, "newer-nonce", current.record_digest)
        self.acquire_controller_generation(group, newer)
        lease = AllocationAttemptV2(group.group_id, AllocationAttemptId("newer-lease"),
                                    allocation.allocation_id, allocation.backend_job_id, "2", "utc")
        newer_fences.append(self.begin_attempt(group, "real-task", allocation, "utc", "local",
                                               allocation_attempt=lease, invocation_id=InvocationId("newer-invocation")))
        return result

    monkeypatch.setattr(DirectoryStore, "publish_reconciliation", publish_then_supersede)
    executor = RecordingExecutor()
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, EchoAdmission(), executor)
    assert len(newer_fences) == 1 and not executor.calls
    assert result.reconciliation.members[0].action == "park"
    pointer = Path(newer_fences[0].output_dir).parent / "current-attempt.json"
    assert json.loads(pointer.read_bytes())["attempt_id"] == newer_fences[0].attempt_id.value
    assert len(list(Path(store.root).rglob("start.json"))) == 1


@pytest.mark.parametrize("failure", ["exception", "malformed"])
def test_t2_admission_failure_is_local_to_member(tmp_path, failure):
    """RED: a failed port row must not suppress an independent fresh member."""
    group = _group(("bad", "good"))
    allocation = ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})
    store = DirectoryStore(tmp_path / "store")
    facts = GroupControllerFactsV1(group, allocation, "a" * 64, "nonce", "utc",
                                  tuple(MemberRecoveryInputV2(t.task_id, t.lineage_id) for t in group.members))
    calls = []

    class SelectiveAdmission:
        def admit(self, member):
            calls.append(member)
            if member.task_id == "bad":
                if failure == "exception":
                    raise RuntimeError("one member unavailable")
                return object()
            return None

    executor = RecordingExecutor()
    result = run_group_controller(_binding(group, allocation.backend_job_id), allocation.backend_job_id,
                                  facts, store, SelectiveAdmission(), executor)
    assert [item.task_id for item in calls[:2]] == [item.task_id for item in facts.member_inputs]
    assert [member.task_id for member in calls[2:]] == ["good"]
    assert calls[2].start_receipt is not None and calls[2].outcome_receipt is not None
    assert [fence.task_id for fence, _ in executor.calls] == ["good"]
    assert dict(result.member_outcomes) == {"bad": "parked", "good": "collected"}


def test_t2_completed_controller_result_reaches_balsam_success(tmp_path):
    """RED: feed the producer's actual output to the existing application."""
    store, group, lease, inputs, _, requests, _ = _fixture(tmp_path, ("completed",), remote=False)
    allocation = ConfiguredAllocation(lease.allocation_id, lease.backend_job_id, {})
    facts = GroupControllerFactsV1(group, allocation, "a" * 64, "nonce", "utc", inputs, requests)
    executor = RecordingExecutor()
    result = run_group_controller(_binding(group, allocation.backend_job_id), allocation.backend_job_id,
                                  facts, store, EchoAdmission(), executor)
    assert result.exit_code == 0 and not executor.calls
    failures = []
    application = make_group_application(SimpleNamespace(ApplicationDefinition=object),
                                        controller=lambda job: result, contract=GroupExitContract(()),
                                        record_failure=lambda job, error: failures.append(error))
    app = application()
    app.job = object()
    assert app.run() == 0
    assert failures == []


def test_t2_intent_read_io_uncertainty_does_not_launch(tmp_path, monkeypatch):
    """RED: an I/O exception is not evidence that no prior intent exists."""
    _, allocation, facts, binding, store = _real_fresh(tmp_path)

    def unreadable(self, fence):
        raise OSError("injected read failure")

    monkeypatch.setattr(DirectoryStore, "read_launch_intent", unreadable)
    executor = RecordingExecutor()
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, EchoAdmission(), executor)
    assert not executor.calls
    assert result.member_outcomes == (("real-task", "parked"),)


def test_t2_truthy_non_acknowledgement_parks(tmp_path):
    """RED: a truthy malformed executor result is not a typed acknowledgement."""
    _, allocation, facts, binding, store = _real_fresh(tmp_path)
    executor = RecordingExecutor(result={"acknowledged": False})
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, EchoAdmission(), executor)
    assert len(executor.calls) == 1
    assert result.member_outcomes == (("real-task", "parked"),)


def test_t2_typed_ack_launches_with_durable_intent(tmp_path):
    """GREEN: start, intent and real current authority precede external launch."""
    group, allocation, facts, binding, store = _real_fresh(tmp_path)
    calls = []

    class CheckingExecutor(RecordingExecutor):
        def submit(self, fence, invocation):
            current = store.require_current_controller(group, 1, facts.claimant_nonce)
            intent = store.read_launch_intent(fence)
            assert intent.controller_generation == current.generation
            assert intent.invocation_id == invocation.invocation_id
            assert Path(fence.output_dir, "start.json").is_file()
            result = super().submit(fence, invocation)
            calls.append(fence)
            return result

    checking = CheckingExecutor()
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, EchoAdmission(), checking)
    assert len(calls) == 1 and len(checking.collected) == 1 and result.member_outcomes == (("real-task", "collected"),)


def test_t2_new_generation_recovery_never_relaunches(tmp_path):
    """GREEN: recover uncertain member under a genuinely new controller nonce."""
    group, allocation, facts, binding, store = _real_fresh(tmp_path)
    first = RecordingExecutor(error=RuntimeError("lost ack"))
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, EchoAdmission(), first)
    assert len(first.calls) == 1 and result.member_outcomes == (("real-task", "parked"),)
    fence, _ = first.calls[0]
    assert store.read_launch_intent(fence)
    start = StartReceipt.decode(Path(fence.output_dir, "start.json").read_bytes())
    previous = store.require_current_controller(group, 1, facts.claimant_nonce)
    recovered = replace(facts, claimant_nonce="recovery-nonce", controller_generation=2,
                        predecessor_record_digest=previous.record_digest,
                        member_inputs=(MemberRecoveryInputV2("real-task", "lineage-real-task", start.attempt_id, start),))
    second = RecordingExecutor()
    result = run_group_controller(binding, allocation.backend_job_id, recovered, store, EchoAdmission(), second)
    assert result.reconciliation.members[0].action == "park" and not second.calls
    assert len(list(Path(store.root).rglob("launch-intent.json"))) == 1
