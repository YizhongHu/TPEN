"""Unrun, local-only delta-review proposals at bc3540ab (PR #798)."""
from pathlib import Path
import errno
import os

import pytest

from tpen_executor.allocations import ConfiguredAllocation, ControllerGenerationV1, ControllerOutcomeReceiptV1
from tpen_executor.contracts import AllocationId, BackendJobId
from tpen_executor.lifecycle.controller import GroupControllerFactsV1, MemberLaunchResult, run_group_controller
from tpen_executor.recovery import MemberRecoveryInputV2
from tpen_executor.storage import DirectoryStore, FenceRejected

from test_b1a_b_publication import _fixture
from test_b1a_reconciliation import _group
from test_review_b1b2_controller import _real_fresh
from test_review_b1b2_t2_r1 import EchoAdmission, RecordingExecutor, _binding


def _successor_facts(group, inputs, requests):
    proposal = requests["successor"][2]
    allocation = ConfiguredAllocation(proposal.allocation_attempt.allocation_id,
                                      proposal.allocation_attempt.backend_job_id, {})
    return GroupControllerFactsV1(group, allocation, "a" * 64, "delta-nonce", "utc", inputs, requests)


def _supersede(store, group, controller):
    newer = ControllerGenerationV1(group.group_id, controller.generation + 1,
                                   controller.allocation_id, controller.backend_job_id,
                                   controller.registration_receipt_digest, "delta-newer", controller.record_digest)
    store.acquire_controller_generation(group, newer)


def test_delta_failed_fresh_admission_still_requires_publication(tmp_path):
    """Predicted red: healthy launch must observe a durable mixed publication."""
    group = _group(("bad", "good"))
    allocation = ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})
    store = DirectoryStore(tmp_path / "store")
    facts = GroupControllerFactsV1(group, allocation, "a" * 64, "nonce", "utc",
                                  tuple(MemberRecoveryInputV2(t.task_id, t.lineage_id) for t in group.members))

    class Admission:
        def admit(self, member):
            if member.task_id == "bad":
                raise RuntimeError("one unavailable verdict")
            return None

    observed_publications = []

    class Executor(RecordingExecutor):
        def submit(self, fence, invocation):
            observed_publications.append(list(Path(store.root).rglob(".reconciliation-publications/*.json")))
            return super().submit(fence, invocation)

    executor = Executor()
    result = run_group_controller(_binding(group, allocation.backend_job_id), allocation.backend_job_id,
                                  facts, store, Admission(), executor)
    assert [f.task_id for f, _ in executor.calls] == ["good"]
    assert len(executor.collected) == 1
    assert dict(result.member_outcomes) == {"bad": "parked", "good": "collected"}
    assert len(observed_publications) == 1 and len(observed_publications[0]) == 1


def test_delta_failed_successor_admission_does_not_park_fresh_peer(tmp_path):
    """Predicted red: a parked successor must not leave a stale request entry."""
    store, group, _, inputs, _, requests, _ = _fixture(tmp_path, ("successor", "fresh"), remote=False)
    facts = _successor_facts(group, inputs, requests)
    calls = []

    class Admission:
        def admit(self, member):
            calls.append(member.task_id)
            if member.task_id == "successor":
                raise RuntimeError("successor unavailable")
            return member.core_verdict

    executor = RecordingExecutor()
    result = run_group_controller(_binding(group, facts.allocation.backend_job_id), facts.allocation.backend_job_id,
                                  facts, store, Admission(), executor)
    assert calls == ["successor", "fresh", "successor", "fresh"]
    assert [f.task_id for f, _ in executor.calls] == ["fresh"]
    assert len(executor.collected) == 1
    assert dict(result.member_outcomes) == {"successor": "parked", "fresh": "collected"}
    assert not list(Path(store.root).rglob("successor-admission.json"))


def test_delta_intent_lstat_error_is_not_absence(tmp_path, monkeypatch):
    """Predicted red: transient EIO below lexists must prevent the launch."""
    _, allocation, facts, binding, store = _real_fresh(tmp_path)
    lstat = os.lstat
    injected = []

    def flaky_lstat(path, *args, **kwargs):
        if str(path).endswith("/launch-intent.json") and not injected:
            injected.append(str(path))
            raise OSError(errno.EIO, "injected intent stat error")
        return lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", flaky_lstat)
    executor = RecordingExecutor()
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, EchoAdmission(), executor)
    assert len(injected) == 1
    assert not executor.calls
    assert result.member_outcomes == (("real-task", "parked"),)


@pytest.mark.parametrize("mode,phase", [("lost-ack", "executor-launch"), ("negative-ack", "launch-ack")])
def test_delta_unacknowledged_receipt_decode_failure_keeps_launch_phase(tmp_path, monkeypatch, mode, phase):
    """Predicted red: a post-call receipt error must not erase the launch phase."""
    _, allocation, facts, binding, store = _real_fresh(tmp_path)

    def corrupt_receipt(self, fence, controller_record, invocation, status, reason=""):
        # record_controller_outcome can raise this on malformed existing JSON.
        raise ValueError("controller outcome decode failed")

    monkeypatch.setattr(DirectoryStore, "record_controller_outcome", corrupt_receipt)
    executor = RecordingExecutor(error=RuntimeError("lost ack") if mode == "lost-ack" else None,
                                 result=MemberLaunchResult(False))
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, EchoAdmission(), executor)
    assert len(executor.calls) == 1
    reason = dict(result.member_reasons)["real-task"]
    assert reason.startswith(phase)
    assert "member-transition" not in reason


def test_delta_stale_successor_never_advances_predecessor_pointer(tmp_path, monkeypatch):
    """Predicted green; kills bypass of the successor authority wrapper."""
    store, group, _, inputs, _, requests, fences = _fixture(tmp_path, remote=False)
    facts = _successor_facts(group, inputs, requests)
    predecessor = fences["successor"]
    pointer = Path(predecessor.output_dir).parent / "current-attempt.json"
    before = pointer.read_bytes()
    publish = DirectoryStore.publish_reconciliation
    injected = []

    def publish_then_supersede(self, group, attempt, inputs, *, bindings, successors):
        result = publish(self, group, attempt, inputs, bindings=bindings, successors=successors)
        assert result.members[0].action == "admit_successor"
        current = self.require_current_controller(group, 1, facts.claimant_nonce)
        _supersede(self, group, current)
        injected.append(True)
        return result

    monkeypatch.setattr(DirectoryStore, "publish_reconciliation", publish_then_supersede)
    executor = RecordingExecutor()
    result = run_group_controller(_binding(group, facts.allocation.backend_job_id), facts.allocation.backend_job_id,
                                  facts, store, EchoAdmission(), executor)
    assert injected == [True] and not executor.calls
    assert result.reconciliation.members[0].action == "park"
    assert pointer.read_bytes() == before
    assert len(list(Path(store.root).rglob("start.json"))) == 1


def test_delta_real_supersession_after_intent_prevents_launch(tmp_path, monkeypatch):
    """Predicted green; distinguishes the final check from start/intent gates."""
    group, allocation, facts, binding, store = _real_fresh(tmp_path)
    create = DirectoryStore.create_launch_intent
    injected = []

    def intent_then_supersede(self, fence, controller_record, invocation):
        intent = create(self, fence, controller_record, invocation)
        _supersede(self, group, controller_record)
        injected.append(True)
        return intent

    monkeypatch.setattr(DirectoryStore, "create_launch_intent", intent_then_supersede)
    executor = RecordingExecutor()
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, EchoAdmission(), executor)
    assert injected == [True]
    assert len(list(Path(store.root).rglob("launch-intent.json"))) == 1
    assert not executor.calls
    assert result.member_outcomes == (("real-task", "parked"),)


def test_delta_persisted_successor_binding_is_never_rewritten(tmp_path):
    """Predicted green: a prior admission parks without adopting a fresh lease."""
    store, group, lease, inputs, bindings, requests, fences = _fixture(tmp_path, remote=False)
    result = store.publish_reconciliation(group, lease, inputs, bindings=bindings, successors=requests)
    assert result.members[0].action == "admit_successor"
    path = next(Path(store.root).rglob("successor-admission.json"))
    before = path.read_bytes()
    predecessor = fences["successor"]
    pointer = Path(predecessor.output_dir).parent / "current-attempt.json"
    pointer_before = pointer.read_bytes()
    facts = _successor_facts(group, inputs, requests)
    executor = RecordingExecutor()
    result = run_group_controller(_binding(group, facts.allocation.backend_job_id), facts.allocation.backend_job_id,
                                  facts, store, EchoAdmission(), executor)
    assert not executor.calls and result.reconciliation.members[0].action == "park"
    assert path.read_bytes() == before and pointer.read_bytes() == pointer_before


def test_delta_controller_outcome_round_trip_replay_and_conflict(tmp_path):
    """Predicted green: receipt is tied to intent and rejects a changed replay."""
    group, allocation, facts, binding, store = _real_fresh(tmp_path)
    executor = RecordingExecutor()
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, EchoAdmission(), executor)
    assert result.member_outcomes == (("real-task", "collected"),)
    assert len(executor.collected) == 1
    fence, invocation = executor.calls[0]
    reopened = DirectoryStore(store.root)
    intent = reopened.read_launch_intent(fence)
    path = Path(fence.output_dir, "controller-outcome.json")
    before = path.read_bytes()
    receipt = ControllerOutcomeReceiptV1.decode(before)
    assert receipt.encode() == before
    assert (receipt.attempt_id, receipt.invocation_id, receipt.controller_generation, receipt.claimant_nonce) == (
        intent.attempt_id, intent.invocation_id, intent.controller_generation, intent.claimant_nonce)
    controller = reopened.require_current_controller(group, 1, facts.claimant_nonce)
    assert reopened.record_controller_outcome(fence, controller, invocation, "launched", "acknowledged") == receipt
    with pytest.raises(FenceRejected):
        reopened.record_controller_outcome(fence, controller, invocation, "unacknowledged", "lost-ack")
    assert path.read_bytes() == before
