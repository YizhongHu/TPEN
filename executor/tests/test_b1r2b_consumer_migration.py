"""B1-R2B linked-result and controller consumer-boundary checks."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.allocations import AllocationAttemptV2, ConfiguredAllocation, ControllerGenerationV1, MemberInvocationV2
from tpen_executor.contracts import AllocationAttemptId, AllocationId, AttemptId, BackendJobId, InvocationId
from tpen_executor.dispatch.transport import LeafRequest
from tpen_executor.lifecycle.attached import AttachedAllocation
from tpen_executor.lifecycle.controller import GroupControllerFactsV1, run_group_controller
import tpen_executor.lifecycle.controller as controller_module
from tpen_executor.recovery import CoreAdmissionVerdictV2, RecoveryAdmissionInputV1, RecoveryGrantV1, admit_recovery
from tpen_executor.results import OutcomeReceipt, StartReceipt, SuccessorAdmissionRecord
from tpen_executor.lifecycle.local_executor import LeafSubmissionError, LocalLeafExecutor
from tpen_executor.storage import AdmissionFork, DirectoryStore, FenceRejected, read_successor_admission

from test_b1a_b_publication import _fixture
from test_b1r2a_recovery_admission import _facts
from test_review_b1b2_controller import _real_fresh
from test_review_b1b2_t2_r1 import RecordingExecutor, _binding
from test_b1e1_local_executor import _environment as _local_environment, _failure_result, _group as _local_group, _invocation as _local_invocation
from test_c22_output_binding import _allocation_with_cpu, _configured


MUTANTS = (
    {"id": "omit-r2-checkpoint-link", "target": "executor/src/tpen_executor/results.py", "old": '"checkpoint_path": self.recovery_admission.checkpoint_path', "new": '"checkpoint_path": "crossed"', "killer": "test_r2_link_round_trip_preserves_typed_lineage"},
    {"id": "accept-crossed-r2-predecessor", "target": "executor/src/tpen_executor/results.py", "old": "self.recovery_admission.predecessor_attempt_id != self.predecessor_attempt_id", "new": "False", "killer": "test_tampered_v3_predecessor_link_is_rejected"},
    {"id": "accept-crossed-r2-successor", "target": "executor/src/tpen_executor/results.py", "old": "self.successor_binding.attempt_id != self.successor_attempt_id", "new": "False", "killer": "test_crossed_r2_successor_link_is_rejected"},
    {"id": "skip-r2-controller-admission", "target": "executor/src/tpen_executor/lifecycle/controller.py", "old": "verdict = None if item.recovery_input is not None else admission_port.admit(admitted_item)", "new": "verdict = admission_port.admit(admitted_item)", "killer": "test_r2_input_uses_admit_recovery_not_legacy_selector"},
    {"id": "launch-before-r2-publication", "target": "executor/src/tpen_executor/lifecycle/controller.py", "old": "if published.encode() != checked.encode():", "new": "if False:", "killer": "test_r2_launch_requires_durable_publication"},
    {"id": "r2-park-submit-bypass", "target": "executor/src/tpen_executor/recovery.py", "old": "return _park(member, member.recovery_admission.reason)", "new": 'return MemberReconciliationV2(member.task_id, "admit_successor", "recovery-admitted", member.attempt_id)', "killer": "test_r2_park_refuses_before_submit"},
    {"id": "fabricate-interrupted-outcome", "target": "executor/src/tpen_executor/storage.py", "old": "if item.outcome_receipt is None:", "new": "if False:", "killer": "test_interrupted_r2_outcome_stays_observed_absence"},
    {"id": "cross-c2-2-load-path", "target": "executor/src/tpen_executor/lifecycle/controller.py", "old": "launched, handle = submit_admitted(\n                            fence, invocation, admission_record.checkpoint_path,\n                        )", "new": "launched, handle = submit_admitted(\n                            fence, invocation, \"crossed\",\n                        )", "killer": "test_c22_load_path_is_exact_admitted_checkpoint"},
    {"id": "r2-current-fence-required", "target": "executor/src/tpen_executor/storage.py", "old": "current_attempt_id = current_fence.attempt_id.value", "new": "current_attempt_id = admission.predecessor_attempt_id.value", "killer": "test_r2_input_uses_admit_recovery_not_legacy_selector"},
    {"id": "preserve-typed-park-reason", "target": "executor/src/tpen_executor/lifecycle/controller.py", "old": "elif member.action == \"park\":", "new": "elif False:", "killer": "test_controller_propagates_typed_park_reason"},
    {"id": "retain-r2-input-on-park", "target": "executor/src/tpen_executor/lifecycle/controller.py", "old": "return replace(item, recovery_input=None, recovery_admission=admission)", "new": "return replace(item, recovery_admission=admission)", "killer": "test_controller_propagates_typed_park_reason"},
    {"id": "skip-r2-final-admission", "target": "executor/src/tpen_executor/lifecycle/controller.py", "old": "if item.recovery_input is not None:", "new": "if False:", "killer": "test_r2_final_reconciliation_does_not_use_legacy_selector"},
    {"id": "require-r2-fence-for-legacy", "target": "executor/src/tpen_executor/lifecycle/controller.py", "old": "if admission_record.recovery_admission is not None:\n                        member_input", "new": "if True:\n                        member_input", "killer": "test_successor_uses_admitted_start_only"},
    {"id": "local-r2-ignores-admitted-load", "target": "executor/src/tpen_executor/lifecycle/local_executor.py", "old": "return self._submit(fence, invocation, admitted_checkpoint_path=checkpoint_path)", "new": "return self._submit(fence, invocation)", "killer": "test_local_r2_adapter_binds_admitted_checkpoint"},
    {"id": "controller-r2-plain-submit-fallback", "target": "executor/src/tpen_executor/lifecycle/controller.py", "old": "if admission_record is not None and admission_record.recovery_admission is not None:\n                        submit_admitted = getattr(executor, \"submit_admitted_successor\", None)", "new": "if admission_record is not None and False:\n                        submit_admitted = getattr(executor, \"submit_admitted_successor\", None)", "killer": "test_r2_adapter_without_admitted_boundary_refuses"},
    {"id": "publish-crossed-r2-request", "target": "executor/src/tpen_executor/storage.py", "old": "self._validate_recovery_request_binding(request, recovery_admission, member.task_id)", "new": "pass", "killer": "test_crossed_r2_request_is_refused_before_ledger"},
)


class RecordingAdmission:
    def __init__(self):
        self.calls = []

    def admit(self, member):
        self.calls.append(member.task_id)
        return member.core_verdict


class BoundaryExecutor(RecordingExecutor):
    def __init__(self, order):
        super().__init__()
        self.order = order
        self.checkpoint_paths = []

    def submit(self, fence, invocation):
        self.order.append("submit")
        return super().submit(fence, invocation)

    def submit_admitted_successor(self, fence, invocation, checkpoint_path):
        self.order.append("submit")
        self.checkpoint_paths.append(checkpoint_path)
        return RecordingExecutor.submit(self, fence, invocation)


def _real_evidence(store, current, predecessor, *, backend_job_id: str):
    evidence = _facts(Path(store.root), mode="operator_authorized_unplanned_end")
    from tpen_executor.recovery import RecoveryCheckpointFactsV1, RecoverySchedulerEndV1

    checkpoint = evidence.checkpoint
    bound_path = Path(predecessor.output_dir) / "core" / "checkpoints" / "step_000002"
    checkpoint = RecoveryCheckpointFactsV1(
        str(bound_path), checkpoint.content_id, checkpoint.next_iteration,
        checkpoint.completed_updates, checkpoint.program_digest,
        checkpoint.science_digest, checkpoint.h_digest,
    )
    return replace(
        evidence,
        task_id=current.task_id,
        group_digest=current.group_digest,
        attempt_id=current.attempt_id,
        predecessor_attempt_id=predecessor.attempt_id,
        allocation_id=current.allocation_id,
        backend_job_id=backend_job_id,
        grant=RecoveryGrantV1(
            evidence.grant.grant_id, evidence.grant.expires_at_utc,
            current.group_digest, current.task_id, evidence.grant.program_digest,
            evidence.grant.science_digest, evidence.grant.h_digest,
            evidence.grant.successor_limit,
        ),
        checkpoint=checkpoint,
        scheduler_end=RecoverySchedulerEndV1(
            current.task_id, current.group_digest, current.attempt_id,
            evidence.scheduler_end.scheduler, current.allocation_id,
            backend_job_id, evidence.scheduler_end.terminal_state,
            evidence.scheduler_end.terminal_at_utc, evidence.scheduler_end.writer_end_identity,
            evidence.scheduler_end.writer_end_at_utc,
        ),
    )


def _real_admission(store, fence, *, successor: AttemptId, backend_job_id: str, current_attempt: AttemptId | None = None):
    """Produce an R2 decision from facts bound to the real fixture store."""

    current = replace(fence, attempt_id=current_attempt) if current_attempt is not None else fence
    evidence = _real_evidence(store, current, fence, backend_job_id=backend_job_id)
    return admit_recovery(
        evidence, successor, now_utc="2026-11-24T00:02:00Z",
        trusted_store_root=store.root,
    )


def _r2_setup(tmp_path: Path, *, outcome: bool = True):
    store, group, lease, inputs, bindings, successors, fences = _fixture(
        tmp_path, ("successor",), remote=False,
    )
    predecessor = fences["successor"]
    request, acceptance, successor_binding = successors["successor"]
    current = AttachedAllocation(
        ConfiguredAllocation(lease.allocation_id, lease.backend_job_id, {}), store, lease,
    ).begin(group, "successor", "utc-current", "local", invocation_id=InvocationId("r2-current"))
    Path(current.output_dir, "core").mkdir(parents=True, exist_ok=True)
    store.finish_attempt(current, "failed", "utc-current-end")
    evidence = _real_evidence(
        store, current, predecessor, backend_job_id=lease.backend_job_id.value,
    )
    recovery_input = RecoveryAdmissionInputV1(
        evidence, successor_binding.attempt_id, "2026-11-24T00:02:00Z",
    )
    if not outcome:
        Path(current.output_dir, "outcome.json").unlink()
    current_start = StartReceipt.decode(Path(current.output_dir, "start.json").read_bytes())
    current_outcome = OutcomeReceipt.decode(Path(current.output_dir, "outcome.json").read_bytes()) if outcome else None
    item = replace(
        inputs[0],
        attempt_id=current.attempt_id,
        start_receipt=current_start,
        outcome_receipt=current_outcome,
        core_verdict=None,
        recovery_input=recovery_input,
        recovery_admission=None,
    )
    allocation = ConfiguredAllocation(AllocationId("current-allocation"), BackendJobId("current-job"), {})
    facts = GroupControllerFactsV1(
        group, allocation, "a" * 64, "r2-controller", "2026-11-24T00:03:00Z",
        (item,), {"successor": (request, acceptance, successor_binding)},
    )
    return store, group, allocation, facts, _binding(group, allocation.backend_job_id), evidence, successor_binding, current


def _controller_diagnostic(store, group, result) -> str:
    """Render complete controller and durable-store context for a failed assertion."""

    try:
        manifest = repr(store.manifest(group))
    except Exception as error:
        manifest = f"<manifest-error {type(error).__name__}: {error}>"
    records = {}
    record_names = {
        "current-attempt.json", "start.json", "outcome.json", "invocation.json",
        "launch-intent.json", "controller-outcome.json", "successor-admission.json",
    }
    for path in sorted(Path(store.root).rglob("*.json")):
        if path.name not in record_names and ".reconciliation-publications" not in path.parts:
            continue
        key = str(path.relative_to(store.root))
        try:
            records[key] = json.loads(path.read_text())
        except Exception as error:
            records[key] = f"<record-error {type(error).__name__}: {error}>"
    return (
        f"reconciliation={result.reconciliation!r}; member_outcomes={result.member_outcomes!r}; "
        f"member_reasons={result.member_reasons!r}; manifest={manifest}; durable_records={records!r}"
    )


def test_r2_link_round_trip_preserves_typed_lineage(tmp_path: Path) -> None:
    store, group, lease, _inputs, bindings, successors, fences = _fixture(tmp_path, ("successor",), remote=False)
    predecessor = fences["successor"]
    request, acceptance, successor_binding = successors["successor"]
    admission = _real_admission(
        store, predecessor, successor=successor_binding.attempt_id,
        backend_job_id=successor_binding.allocation_attempt.backend_job_id.value,
        current_attempt=AttemptId("current-r2"),
    )
    assert admission.status == "admitted"
    record = store.record_successor_admission(
        predecessor, request, acceptance,
        predecessor_binding=bindings["successor"], successor_binding=successor_binding,
        recovery_admission=admission,
    )
    wire = json.loads(record.encode())
    assert wire["predecessor"]["checkpoint_path"] == admission.checkpoint_path
    decoded = SuccessorAdmissionRecord.decode(record.encode())
    assert decoded.predecessor_attempt_id == admission.predecessor_attempt_id
    assert decoded.checkpoint_path == admission.checkpoint_path
    successor = store.begin_admitted_successor(
        group, "successor", ConfiguredAllocation(successor_binding.allocation_attempt.allocation_id, successor_binding.allocation_attempt.backend_job_id, {}),
        "2026-11-24T00:03:00Z", "local", allocation_attempt=successor_binding.allocation_attempt,
        admission=record,
    )
    result = read_successor_admission(
        successor.output_dir, task_id="successor", attempt_id=successor.attempt_id.value,
        group_digest=group.identity, science_input_digest="a" * 64, target_iterations=10,
    )
    assert result["record"]["predecessor"]["attempt_id"] == admission.predecessor_attempt_id.value
    assert result["record"]["predecessor"]["checkpoint_path"] == admission.checkpoint_path


def test_local_r2_adapter_binds_admitted_checkpoint(tmp_path: Path) -> None:
    group = _local_group(("member",))
    allocation = _allocation_with_cpu(min(os.sched_getaffinity(0)))
    seen: list = []

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        seen.append(request)
        return _failure_result(request).encode()

    executor = LocalLeafExecutor(group, allocation, lambda _program: _configured(tmp_path, task_id="member"), _local_environment, leaf_runner=runner)
    try:
        fence, invocation = _local_invocation(group, allocation, "member", tmp_path / "store")
        wrong = str(Path(fence.output_dir) / "core" / "checkpoint")
        with pytest.raises(LeafSubmissionError, match="own attempt leaf"):
            executor.submit_admitted_successor(fence, invocation, wrong)
        assert seen == [], f"runner_calls={seen!r}; wrong_path={wrong!r}"
        admitted_path = str(tmp_path / "predecessor" / "core" / "checkpoint")
        acknowledgement, handle = executor.submit_admitted_successor(fence, invocation, admitted_path)
        executor.collect(handle)
        assert acknowledgement.acknowledged, f"ack={acknowledgement!r}; requests={seen!r}"
        assert len(seen) == 1, f"requests={seen!r}"
        request = seen[0]
        assert request.attempt_delta is not None, f"request={request!r}"
        assert request.attempt_delta.load.mode == "train_resume", f"request={request!r}"
        assert request.attempt_delta.load.path == admitted_path, f"request={request!r}; expected={admitted_path!r}"
    finally:
        executor.close()


def test_r2_adapter_without_admitted_boundary_refuses(tmp_path: Path) -> None:
    store, group, allocation, facts, binding, _evidence, _successor, _current = _r2_setup(tmp_path)

    class LegacyOnlyExecutor(RecordingExecutor):
        pass

    executor = LegacyOnlyExecutor()
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, RecordingAdmission(), executor)
    diagnostic = _controller_diagnostic(store, group, result)
    assert executor.calls == [], diagnostic
    assert result.member_outcomes == (("successor", "parked"),), diagnostic
    assert result.member_reasons[0][1].startswith("executor-launch:LeafSubmissionError"), diagnostic


def test_crossed_r2_request_is_refused_before_ledger(tmp_path: Path) -> None:
    store, group, allocation, facts, _binding, _evidence, successor_binding, _current = _r2_setup(tmp_path)
    item = controller_module._apply_recovery_admission(facts.member_inputs[0], store)
    request, acceptance, proposed = facts.successor_requests["successor"]
    crossed = json.loads(json.dumps(request))
    crossed["predecessor"]["attempt_id"] = "crossed-predecessor"
    with pytest.raises(FenceRejected, match="crosses admitted facts") as error:
        DirectoryStore._validate_recovery_request_binding(crossed, item.recovery_admission, "successor")
    assert "crosses admitted facts" in str(error.value), f"error={error.value!r}; crossed={crossed!r}"
    current_binding = MemberInvocationV2.decode(Path(item.start_receipt.output_dir, "invocation.json").read_bytes())
    allocation_attempt = AllocationAttemptV2(
        group.group_id, AllocationAttemptId("r16-publication"), allocation.allocation_id,
        allocation.backend_job_id, "1", facts.observed_at_utc,
    )
    result = store.publish_reconciliation(
        group, allocation_attempt, (item,), bindings={"successor": current_binding},
        successors={"successor": (crossed, acceptance, proposed)},
        recovery_admissions={"successor": item.recovery_admission},
    )
    assert result.members[0].action == "park", f"result={result!r}; crossed={crossed!r}"
    assert result.members[0].reason_code == "invalid-durable-evidence", f"result={result!r}; store={store.root}"
    assert not list(Path(store.root).rglob("successor-admission.json")), f"result={result!r}; store={store.root}"
    assert store.read_reconciliation(group, allocation_attempt) is None, f"result={result!r}; store={store.root}"


def test_reader_rejects_crossed_embedded_request_with_recomputed_digest(tmp_path: Path) -> None:
    store, group, _allocation, facts, _binding, _evidence, _successor, current = _r2_setup(tmp_path)
    item = controller_module._apply_recovery_admission(facts.member_inputs[0], store)
    request, acceptance, successor_binding = facts.successor_requests["successor"]
    record = store.record_successor_admission(
        current, request, acceptance,
        successor_binding=successor_binding, recovery_admission=item.recovery_admission,
    )
    successor = store.begin_admitted_successor(
        group, "successor",
        ConfiguredAllocation(successor_binding.allocation_attempt.allocation_id, successor_binding.allocation_attempt.backend_job_id, {}),
        "2026-11-24T00:03:00Z", "local", allocation_attempt=successor_binding.allocation_attempt,
        admission=record, current_fence=current,
    )
    ledger = Path(store.root) / "groups" / group.identity / "successor" / "successor-admissions" / record.predecessor_attempt_id.value / "successor-admission.json"
    wire = json.loads(ledger.read_bytes())
    wire["successor_request"]["predecessor"]["attempt_id"] = "crossed-predecessor"
    wire["request_digest"] = digest(wire["successor_request"])
    ledger.write_bytes(canonical_bytes(wire))
    clean = read_successor_admission(
        successor.output_dir, task_id="successor", attempt_id=record.successor_attempt_id.value,
        group_digest=group.identity, science_input_digest=request["science_input_digest"],
        target_iterations=request["target_iterations"],
    )
    assert clean["record"]["successor"]["attempt_id"] == record.successor_attempt_id.value, f"clean={clean!r}"
    with pytest.raises(ValueError, match="successor_request crosses admission identity") as error:
        SuccessorAdmissionRecord.from_mapping(wire)
    assert "successor_request crosses admission identity" in str(error.value), f"error={error.value!r}; wire={wire!r}"


def test_crossed_r2_successor_link_is_rejected(tmp_path: Path) -> None:
    store, _group, _lease, _inputs, bindings, successors, fences = _fixture(tmp_path, ("successor",), remote=False)
    predecessor = fences["successor"]
    request, acceptance, successor_binding = successors["successor"]
    admission = _real_admission(
        store, predecessor, successor=successor_binding.attempt_id,
        backend_job_id=successor_binding.allocation_attempt.backend_job_id.value,
        current_attempt=AttemptId("current-r2"),
    )
    crossed = replace(admission, successor_attempt_id=AttemptId("other-successor"))
    with pytest.raises(ValueError, match="successor binding crosses admission identity"):
        SuccessorAdmissionRecord.from_recovery_admission(
            crossed, predecessor_task_id="successor", predecessor_content_id="a" * 64,
            successor_task_id="successor", request_digest=digest(request),
            core_acceptance_digest=acceptance, successor_binding=successor_binding,
            successor_request=request,
        )


def test_tampered_v3_predecessor_link_is_rejected(tmp_path: Path) -> None:
    store, _group, _lease, _inputs, bindings, successors, fences = _fixture(tmp_path, ("successor",), remote=False)
    predecessor = fences["successor"]
    request, acceptance, successor_binding = successors["successor"]
    admission = _real_admission(
        store, predecessor, successor=successor_binding.attempt_id,
        backend_job_id=successor_binding.allocation_attempt.backend_job_id.value,
        current_attempt=AttemptId("current-r2"),
    )
    record = store.record_successor_admission(
        predecessor, request, acceptance,
        predecessor_binding=bindings["successor"], successor_binding=successor_binding,
        recovery_admission=admission,
    )
    wire = json.loads(record.encode())
    wire["predecessor"]["attempt_id"] = "tampered-predecessor"
    wire["predecessor"]["checkpoint_path"] = admission.checkpoint_path
    with pytest.raises(ValueError, match="recovery admission crosses successor identity"):
        SuccessorAdmissionRecord.from_mapping(wire)


def test_r2_duplicate_replay_is_idempotent_and_fork_parks(tmp_path: Path) -> None:
    store, _group, _lease, _inputs, bindings, successors, fences = _fixture(tmp_path, ("successor",), remote=False)
    predecessor = fences["successor"]
    request, acceptance, successor_binding = successors["successor"]
    admission = _real_admission(
        store, predecessor, successor=successor_binding.attempt_id,
        backend_job_id=successor_binding.allocation_attempt.backend_job_id.value,
        current_attempt=AttemptId("current-r2"),
    )
    first = store.record_successor_admission(
        predecessor, request, acceptance, predecessor_binding=bindings["successor"],
        successor_binding=successor_binding, recovery_admission=admission,
    )
    assert store.record_successor_admission(
        predecessor, request, acceptance, predecessor_binding=bindings["successor"],
        successor_binding=successor_binding, recovery_admission=admission,
    ) == first
    other_attempt = AttemptId("other-successor")
    other_request = dict(request)
    other_request["successor"] = {"task_id": "successor", "attempt_id": other_attempt.value}
    other_binding = replace(successor_binding, attempt_id=other_attempt)
    other_admission = replace(admission, successor_attempt_id=other_attempt)
    with pytest.raises(AdmissionFork):
        store.record_successor_admission(
            predecessor, other_request, acceptance,
            predecessor_binding=bindings["successor"], successor_binding=other_binding,
            recovery_admission=other_admission,
        )


def test_r2_input_uses_admit_recovery_not_legacy_selector(tmp_path: Path, monkeypatch) -> None:
    store, group, allocation, facts, binding, _evidence, _successor, _current = _r2_setup(tmp_path)
    port = RecordingAdmission()
    executor = BoundaryExecutor([])
    admissions = []
    original_admit = controller_module.admit_recovery

    def admit(*args, **kwargs):
        result = original_admit(*args, **kwargs)
        admissions.append((result, kwargs["trusted_store_root"]))
        return result

    monkeypatch.setattr(controller_module, "admit_recovery", admit)
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, port, executor)
    diagnostic = _controller_diagnostic(store, group, result)
    assert result.member_outcomes == (("successor", "collected"),), diagnostic
    assert admissions and all(result.status == "admitted" and root == store.root for result, root in admissions), diagnostic
    assert port.calls == [], diagnostic
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    legacy_store, legacy_group, legacy_lease, legacy_inputs, _legacy_bindings, legacy_successors, _legacy_fences = _fixture(
        legacy_root, ("successor",), remote=False,
    )
    legacy_alloc = ConfiguredAllocation(legacy_lease.allocation_id, legacy_lease.backend_job_id, {})
    legacy_facts = GroupControllerFactsV1(
        legacy_group, legacy_alloc, "a" * 64, "legacy-controller", "2026-11-24T00:03:00Z",
        legacy_inputs, legacy_successors,
    )
    legacy_binding = _binding(legacy_group, legacy_lease.backend_job_id)
    legacy_port = RecordingAdmission()
    legacy_executor = BoundaryExecutor([])
    run_group_controller(legacy_binding, legacy_alloc.backend_job_id, legacy_facts, legacy_store, legacy_port, legacy_executor)
    assert legacy_port.calls and legacy_port.calls == ["successor"] * len(legacy_port.calls)


def test_r2_final_reconciliation_does_not_use_legacy_selector(tmp_path: Path, monkeypatch) -> None:
    store, group, allocation, facts, binding, _evidence, _successor, _current = _r2_setup(tmp_path)
    port = RecordingAdmission()
    executor = BoundaryExecutor([])
    reconciliations = []
    original_reconcile = controller_module.reconcile_group

    def reconcile(group, inputs, **kwargs):
        reconciliations.append(tuple((item.recovery_input, item.recovery_admission) for item in inputs))
        return original_reconcile(group, inputs, **kwargs)

    monkeypatch.setattr(controller_module, "reconcile_group", reconcile)
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, port, executor)
    diagnostic = _controller_diagnostic(store, group, result)
    assert len(reconciliations) == 2, diagnostic
    final_input = reconciliations[-1][0]
    assert final_input[0] is None and final_input[1] is None, diagnostic
    assert port.calls == [], diagnostic
    assert result.reconciliation.members[0].action == "preserve_completed", diagnostic
    assert result.exit_code == 0, diagnostic


@pytest.mark.parametrize("outcome,expected_action", [("terminal", "preserve_terminal"), ("unknown", "park")])
def test_r2_final_reconciliation_uses_successor_outcome(tmp_path: Path, outcome: str, expected_action: str) -> None:
    store, group, _allocation, facts, _binding, _evidence, _successor, _current = _r2_setup(tmp_path)
    admitted = controller_module._apply_recovery_admission(facts.member_inputs[0], store)
    successor = replace(
        admitted,
        recovery_input=None,
        recovery_admission=None,
        core_verdict=CoreAdmissionVerdictV2(outcome, "unknown", "unknown", "unknown", None, "successor-owned"),
    )
    result = controller_module.reconcile_group(group, (successor,))
    diagnostic = f"outcome={outcome!r}; result={result!r}; successor={successor!r}; store={store.root}"
    assert result.members[0].action == expected_action, diagnostic
    assert result.members[0].action != "admit_successor", diagnostic


@pytest.mark.parametrize("publication_mode", ["raise", "park"])
def test_r2_launch_requires_durable_publication(tmp_path: Path, monkeypatch, publication_mode: str) -> None:
    store, group, allocation, facts, binding, _evidence, _successor, _current = _r2_setup(tmp_path)
    order = []
    original = DirectoryStore.publish_reconciliation

    def publish(self, *args, **kwargs):
        order.append("publish")
        if publication_mode == "raise":
            raise RuntimeError("publication unavailable")
        result = original(self, *args, **kwargs)
        return replace(result, members=tuple(replace(member, reason_code="publication-parked") for member in result.members))

    monkeypatch.setattr(DirectoryStore, "publish_reconciliation", publish)
    executor = BoundaryExecutor(order)
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, RecordingAdmission(), executor)
    diagnostic = _controller_diagnostic(store, group, result)
    assert order == ["publish"], diagnostic
    assert executor.calls == [], diagnostic
    assert all(member.action == "park" for member in result.reconciliation.members), diagnostic


def test_r2_success_publishes_before_submit(tmp_path: Path, monkeypatch) -> None:
    store, group, allocation, facts, binding, _evidence, _successor, _current = _r2_setup(tmp_path)
    order = []
    original = DirectoryStore.publish_reconciliation

    def publish(self, *args, **kwargs):
        order.append("publish")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DirectoryStore, "publish_reconciliation", publish)
    executor = BoundaryExecutor(order)
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, RecordingAdmission(), executor)
    diagnostic = _controller_diagnostic(store, group, result)
    assert result.member_outcomes == (("successor", "collected"),), diagnostic
    assert order == ["publish", "submit"], diagnostic


def test_r2_park_refuses_before_submit(tmp_path: Path) -> None:
    store, group, allocation, facts, binding, evidence, _successor, _current = _r2_setup(tmp_path)
    unavailable = replace(evidence, checkpoint=replace(evidence.checkpoint, committed=False))
    item = replace(facts.member_inputs[0], recovery_input=replace(facts.member_inputs[0].recovery_input, evidence=unavailable))
    facts = replace(facts, member_inputs=(item,))
    direct = controller_module.reconcile_group(
        group,
        (replace(item, recovery_input=None, recovery_admission=admit_recovery(
            unavailable,
            item.recovery_input.successor_attempt_id,
            now_utc=item.recovery_input.now_utc,
            trusted_store_root=store.root,
        )),),
    )
    assert direct.members[0].action == "park", f"direct_reconciliation={direct!r}; store={store.root}"
    executor = BoundaryExecutor([])
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, RecordingAdmission(), executor)
    member = result.reconciliation.members[0]
    diagnostic = _controller_diagnostic(store, group, result)
    assert executor.calls == [], diagnostic
    assert member.action == "park", diagnostic
    assert member.reason_code == "checkpoint-unavailable", diagnostic


def test_controller_propagates_typed_park_reason(tmp_path: Path) -> None:
    store, group, allocation, facts, binding, evidence, _successor, _current = _r2_setup(tmp_path)
    unavailable = replace(evidence, checkpoint=replace(evidence.checkpoint, committed=False))
    item = replace(
        facts.member_inputs[0],
        recovery_input=replace(facts.member_inputs[0].recovery_input, evidence=unavailable),
    )
    result = run_group_controller(
        binding,
        allocation.backend_job_id,
        replace(facts, member_inputs=(item,)),
        store,
        RecordingAdmission(),
        BoundaryExecutor([]),
    )
    diagnostic = _controller_diagnostic(store, group, result)
    assert result.member_outcomes == (("successor", "parked"),), diagnostic
    assert result.member_reasons == (("successor", "checkpoint-unavailable"),), diagnostic


def test_interrupted_r2_outcome_stays_observed_absence(tmp_path: Path) -> None:
    store, group, allocation, facts, binding, _evidence, _successor, current = _r2_setup(tmp_path, outcome=False)
    executor = BoundaryExecutor([])
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, RecordingAdmission(), executor)
    diagnostic = _controller_diagnostic(store, group, result)
    assert result.member_outcomes == (("successor", "collected"),), diagnostic
    assert not Path(current.output_dir, "outcome.json").exists(), diagnostic
    assert not Path(current.output_dir, "core-outcome.json").exists(), diagnostic


def test_duplicate_controller_replay_is_idempotent(tmp_path: Path) -> None:
    store, group, allocation, facts, binding, _evidence, _successor, _current = _r2_setup(tmp_path)
    first_executor = BoundaryExecutor([])
    first = run_group_controller(binding, allocation.backend_job_id, facts, store, RecordingAdmission(), first_executor)
    second_executor = BoundaryExecutor([])
    second = run_group_controller(binding, allocation.backend_job_id, facts, store, RecordingAdmission(), second_executor)
    assert first.member_outcomes == (("successor", "collected"),)
    assert second_executor.calls == []
    assert len(list(Path(store.root).rglob("successor-admission.json"))) == 1
    assert second.member_outcomes == (("successor", "parked"),)


def test_crossed_controller_generation_parks_before_submit(tmp_path: Path, monkeypatch) -> None:
    store, group, allocation, facts, binding, _evidence, _successor, _current = _r2_setup(tmp_path)
    original = DirectoryStore.publish_reconciliation

    def publish_then_supersede(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        current = self.require_current_controller(group, facts.controller_generation, facts.claimant_nonce)
        newer = ControllerGenerationV1(
            group.group_id, facts.controller_generation + 1, allocation.allocation_id,
            allocation.backend_job_id, facts.registration_receipt_digest, "crossed-controller",
            current.record_digest,
        )
        self.acquire_controller_generation(group, newer)
        return result

    monkeypatch.setattr(DirectoryStore, "publish_reconciliation", publish_then_supersede)
    executor = BoundaryExecutor([])
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, RecordingAdmission(), executor)
    assert executor.calls == []
    assert result.member_outcomes == (("successor", "parked"),)
    assert result.member_reasons[0][1].startswith("member-transition:ControllerGenerationRejected")


def test_c22_load_path_is_exact_admitted_checkpoint(tmp_path: Path) -> None:
    store, group, allocation, facts, binding, evidence, _successor, _current = _r2_setup(tmp_path)
    executor = BoundaryExecutor([])
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, RecordingAdmission(), executor)
    diagnostic = _controller_diagnostic(store, group, result)
    assert result.member_outcomes == (("successor", "collected"),), diagnostic
    record_path = next(Path(store.root).rglob("successor-admission.json"))
    record = SuccessorAdmissionRecord.decode(record_path.read_bytes())
    assert executor.checkpoint_paths == [record.checkpoint_path], diagnostic


def test_mutant_sources_compile_and_killers_are_declared() -> None:
    names = set()
    for path in Path(__file__).parent.glob("test_*.py"):
        tree = ast.parse(path.read_text())
        names.update(
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    assert len({row["id"] for row in MUTANTS}) == len(MUTANTS)
    for row in MUTANTS:
        source = Path(__file__).parents[1].joinpath(row["target"].split("executor/", 1)[1]).read_text()
        assert source.count(row["old"]) == 1
        assert row["killer"] in names
        compile(source.replace(row["old"], row["new"], 1), row["target"], "exec")
