"""B1a-B fake-only durable publication and successor-fence falsifiers.

Real-core compatibility is unproven. These tests exercise no restore callback,
backend, scheduler, signal, or allocation replacement.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.allocations import AllocationAttemptV2, ConfiguredAllocation, MemberInvocationV2
from tpen_executor.contracts import AllocationAttemptId, AllocationId, AttemptId, BackendJobId, InvocationId
from tpen_executor.lifecycle.attached import AttachedAllocation
from tpen_executor.recovery import MemberRecoveryInputV2
from tpen_executor.results import ReconciliationPublicationV2, StartReceipt, SuccessorAdmissionRecord
from tpen_executor.storage import DirectoryStore

from test_b1a_reconciliation import _core, _group, _profile
from test_core_evidence_v3 import _planned_evidence
from test_review_p8_executor_r1 import setup_store


def _fixture(tmp_path, names=("successor",), remote=True):
    group = _group(names)
    allocation = ConfiguredAllocation(AllocationId("allocation-1"), BackendJobId("job-1"), {})
    lease = AllocationAttemptV2(group.group_id, AllocationAttemptId("lease-1"), allocation.allocation_id, allocation.backend_job_id, "controller-1", "utc-1")
    store = DirectoryStore(tmp_path / "store")
    attached = AttachedAllocation(allocation, store, lease)
    inputs, bindings, successors, fences = [], {}, {}, {}
    for name in names:
        if name == "fresh":
            inputs.append(MemberRecoveryInputV2(name, "lineage-" + name))
            continue
        fence = attached.begin(group, name, "utc-1", "remote" if remote else "local", invocation_id=InvocationId("invocation-" + name))
        fences[name] = fence
        binding = MemberInvocationV2.decode(Path(fence.output_dir, "invocation.json").read_bytes())
        bindings[name] = binding
        start = StartReceipt.decode(Path(fence.output_dir, "start.json").read_bytes())
        core_dir = Path(fence.output_dir, "core")
        core_dir.mkdir()
        status = "completed" if name == "completed" else "failed"
        if remote:
            outcome = store.finish_remote_attempt(fence, {
                "schema": "tpen-executor/leaf-teardown-v1", "group_digest": group.identity,
                "task_id": name, "attempt_id": fence.attempt_id.value,
                "allocation_id": allocation.allocation_id.value, "backend_job_id": allocation.backend_job_id.value,
                "child_group_reaped": True, "inspection": "complete",
            }, "c" * 64, status, "utc-2")
        else:
            outcome = store.finish_attempt(fence, status, "utc-2")
        if name == "successor":
            store.record_core_outcome(fence, _planned_evidence(core_dir))
            verdict = _core("successor", profile=_profile(), admission="admitted", checkpoint="valid", quiescence="valid")
            request = store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
            next_lease = replace(lease, allocation_attempt_id=AllocationAttemptId("lease-2"), allocation_id=AllocationId("allocation-2"), backend_job_id=BackendJobId("job-2"))
            successor = MemberInvocationV2(next_lease, group.identity, name, binding.lineage_id, AttemptId("successor-1"), InvocationId("invocation-next"))
            successors[name] = (request, "b" * 64, successor)
        else:
            verdict = _core("unknown" if name == "ambiguous" else name)
        inputs.append(MemberRecoveryInputV2(name, binding.lineage_id, fence.attempt_id, start, outcome, verdict))
    return store, group, lease, tuple(inputs), bindings, successors, fences


def _publish(fixture):
    store, group, lease, inputs, bindings, successors, _ = fixture
    return store.publish_reconciliation(group, lease, inputs, bindings=bindings, successors=successors)


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*.json")}


def test_publication_replay_and_exact_member_digests(tmp_path):
    fixture = _fixture(tmp_path, ("completed", "successor", "fresh", "terminal", "ambiguous"))
    store, group, lease, inputs, bindings, successors, fences = fixture
    completed_before = _files(Path(fences["completed"].output_dir))
    result = _publish(fixture)
    assert result.aggregate == "parked"
    record = store.read_reconciliation(group, lease)
    assert record is not None
    assert [member.outcome for member in record.members] == ["completed", "successor_admitted", "fresh_pending", "terminal", "parked"]
    assert record.to_dict()["member_digests"] == [digest(member.to_dict()) for member in record.members]
    assert ReconciliationPublicationV2.decode(record.encode()).encode() == record.encode()
    snapshot = _files(Path(store.root))
    reopened = DirectoryStore(store.root)
    assert reopened.publish_reconciliation(group, lease, inputs, bindings=bindings, successors=successors) == result
    assert reopened.read_reconciliation(group, lease).encode() == record.encode()
    assert _files(Path(store.root)) == snapshot
    assert _files(Path(fences["completed"].output_dir)) == completed_before
    admission = record.members[1].admission
    assert admission.successor_binding == successors["successor"][2]
    assert SuccessorAdmissionRecord.decode(admission.encode()) == admission


def test_publication_without_ambiguity_requires_replacement(tmp_path):
    fixture = _fixture(tmp_path, ("completed", "successor", "fresh", "terminal"))
    assert _publish(fixture).aggregate == "replacement_required"
    assert fixture[0].read_reconciliation(fixture[1], fixture[2]).reconciliation.terminal_count == 1


@pytest.mark.parametrize("field", ["allocation", "backend", "lease", "group", "lineage", "invocation", "receipt", "core", "profile"])
def test_crossed_or_missing_evidence_parks_without_writes(tmp_path, field):
    fixture = list(_fixture(tmp_path))
    store, group, lease, inputs, bindings, successors, fences = fixture
    binding = bindings["successor"]
    if field in {"allocation", "backend", "lease", "group"}:
        changes = {
            "allocation": {"allocation_id": AllocationId("other")},
            "backend": {"backend_job_id": BackendJobId("other")},
            "lease": {"allocation_attempt_id": AllocationAttemptId("other")},
            "group": {"group_id": type(group.group_id)("other")},
        }[field]
        bindings["successor"] = replace(binding, allocation_attempt=replace(lease, **changes))
    elif field == "lineage":
        fixture[3] = (replace(inputs[0], lineage_id="other"),)
    elif field == "invocation":
        bindings["successor"] = replace(binding, invocation_id=InvocationId("stale"))
    elif field == "receipt":
        fixture[3] = (replace(inputs[0], outcome_receipt=replace(inputs[0].outcome_receipt, finished_at_utc="stale")),)
    elif field == "core":
        fixture[3] = (replace(inputs[0], core_verdict=None),)
    else:
        fixture[3] = (replace(inputs[0], core_verdict=replace(inputs[0].core_verdict, profile=None)),)
    snapshot = _files(Path(store.root))
    assert _publish(fixture).aggregate == "parked"
    assert _files(Path(store.root)) == snapshot
    assert store.read_reconciliation(group, lease) is None


@pytest.mark.parametrize("damage", ["missing", "allocation", "backend", "group", "task", "attempt", "noncanonical", "unconfirmed", "not-reaped", "incomplete", "core-payload"])
def test_remote_gate_negative_matrix_creates_no_binding_or_publication(tmp_path, damage):
    fixture = _fixture(tmp_path)
    store, group, lease, inputs, bindings, successors, fences = fixture
    fence = fences["successor"]
    path = Path(fence.output_dir, "remote-teardown.json")
    data = json.loads(path.read_bytes())
    fields = {"allocation": "allocation_id", "backend": "backend_job_id", "group": "group_digest", "task": "task_id", "attempt": "attempt_id"}
    if damage == "missing":
        path.unlink()
    elif damage in fields:
        data[fields[damage]] = "a" * 64 if damage == "group" else "other"
        path.write_bytes(canonical_bytes(data))
    elif damage == "noncanonical":
        path.write_text(json.dumps(data, indent=2))
    elif damage == "unconfirmed":
        path = Path(fence.output_dir, "outcome.json")
        data = json.loads(path.read_bytes())
        data["status"] = "remote-quiescence-unconfirmed"
        path.write_bytes(canonical_bytes(data))
        # Even a caller echoing the changed operational outcome must park.
        fixture = list(fixture)
        fixture[3] = (replace(inputs[0], outcome_receipt=replace(inputs[0].outcome_receipt, status="remote-quiescence-unconfirmed")),)
    elif damage == "core-payload":
        Path(fence.output_dir, "core-outcome.json").write_bytes(b'{"corrupt":true}')
    else:
        data["child_group_reaped" if damage == "not-reaped" else "inspection"] = False if damage == "not-reaped" else "not-run"
        path.write_bytes(canonical_bytes(data))
    snapshot = _files(Path(store.root))
    result = _publish(fixture)
    assert result.aggregate == "parked"
    assert _files(Path(store.root)) == snapshot
    assert not list(Path(store.root).rglob("successor-admission.json"))
    assert store.read_reconciliation(group, lease) is None


def test_commit_time_recheck_rejects_concurrent_proof_removal(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    original = DirectoryStore.build_successor_admission_request
    path = Path(fixture[6]["successor"].output_dir, "remote-teardown.json")

    def remove_after_build(self, *args, **kwargs):
        request = original(self, *args, **kwargs)
        path.unlink()
        return request

    monkeypatch.setattr(DirectoryStore, "build_successor_admission_request", remove_after_build)
    assert _publish(fixture).aggregate == "parked"
    assert not list(Path(fixture[0].root).rglob("successor-admission.json"))
    assert fixture[0].read_reconciliation(fixture[1], fixture[2]) is None


def test_fork_or_changed_successor_binding_never_overwrites(tmp_path):
    fixture = _fixture(tmp_path)
    store, group, lease, inputs, bindings, successors, fences = fixture
    request, acceptance, successor = successors["successor"]
    store.record_successor_admission(fences["successor"], request, acceptance, predecessor_binding=bindings["successor"], successor_binding=successor)
    snapshot = _files(Path(store.root))
    successors["successor"] = (request, acceptance, replace(successor, invocation_id=InvocationId("different")))
    result = _publish(fixture)
    assert result.aggregate == "parked"
    assert result.members[0].reason_code == "successor-fork"
    assert _files(Path(store.root)) == snapshot
    assert store.read_reconciliation(group, lease) is None


def test_changed_plan_and_nonidentical_republication_park(tmp_path):
    fixture = _fixture(tmp_path, ("fresh",))
    assert _publish(fixture).aggregate == "replacement_required"
    store, group, lease, inputs, bindings, successors, _ = fixture
    snapshot = _files(Path(store.root))
    changed_plan = replace(group.execution_plan, reconciliation_policy="different-policy")
    changed_group = replace(group, execution_plan=changed_plan)
    assert store.publish_reconciliation(changed_group, lease, inputs, bindings=bindings).aggregate == "parked"
    assert store.publish_reconciliation(group, replace(lease, controller_generation="other"), inputs, bindings=bindings).aggregate == "parked"
    changed_id = type(group.group_id)("other-group")
    assert store.publish_reconciliation(replace(group, group_id=changed_id), replace(lease, group_id=changed_id), inputs, bindings=bindings).aggregate == "parked"
    assert _files(Path(store.root)) == snapshot


def test_local_bound_successor_uses_existing_local_rule(tmp_path):
    fixture = _fixture(tmp_path, remote=False)
    assert not list(Path(fixture[0].root).rglob("remote-teardown.json"))
    assert _publish(fixture).aggregate == "replacement_required"
    assert fixture[0].read_reconciliation(fixture[1], fixture[2]).members[0].outcome == "successor_admitted"


def test_v1_attached_api_and_start_bytes_unchanged(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import tpen_executor.storage as storage

    store, group, allocation = setup_store(tmp_path)
    monkeypatch.setattr(storage, "uuid4", lambda: SimpleNamespace(hex="fixed-attempt"))
    attached = AttachedAllocation(allocation, store)
    fence = attached.begin(group, "a", "utc-1")
    expected = StartReceipt("a", AttemptId("fixed-attempt"), allocation.allocation_id, allocation.backend_job_id, group.identity, "utc-1", fence.output_dir, "local")
    assert Path(fence.output_dir, "start.json").read_bytes() == expected.encode()
    assert not Path(fence.output_dir, "invocation.json").exists()
    assert attached.finish(fence, "failed", "utc-2").status == "failed"
    assert attached.observe("running", "utc-3") == allocation.observe("running", "utc-3")


@pytest.mark.parametrize("field", ["member_digests", "reconciliation", "extra", "missing"])
def test_closed_publication_rejects_corruption(tmp_path, field):
    fixture = _fixture(tmp_path, ("fresh",))
    _publish(fixture)
    record = fixture[0].read_reconciliation(fixture[1], fixture[2])
    data = record.to_dict()
    if field == "member_digests":
        data[field][0] = "0" * 64
    elif field == "reconciliation":
        data[field]["aggregate"] = "settled"
    elif field == "extra":
        data["extra"] = True
    else:
        del data["allocation_attempt"]
    with pytest.raises((TypeError, ValueError)):
        ReconciliationPublicationV2.decode(data)


def test_changed_evidence_cannot_merge_into_publication(tmp_path):
    fixture = list(_fixture(tmp_path, ("terminal",)))
    assert _publish(fixture).aggregate == "settled"
    snapshot = _files(Path(fixture[0].root))
    item = fixture[3][0]
    fixture[3] = (replace(item, core_verdict=replace(item.core_verdict, reason="different-verdict")),)
    result = _publish(fixture)
    assert result.aggregate == "parked"
    assert result.members[0].reason_code == "reconciliation-publication-conflict"
    assert _files(Path(fixture[0].root)) == snapshot


# Executable mutation specification for the independent cluster verifier.
# Each edit is exact and unique; tests below validate the table's source anchors.
# Runtime kills are intentionally UNMEASURED in this implementation receipt.
MUTANTS = (
    ("conflate-allocation-backend", "allocations.py", '            AllocationId.from_mapping(data["allocation_id"]),', '            AllocationId(data["allocation_id"]["value"]),', "test_allocation_attempt_v2_binds_all_operational_identities"),
    ("scheduler-terminal-as-core", "recovery.py", '        return _park(member, "unknown-core-result")', '        return MemberReconciliationV2(member.task_id, "preserve_terminal", "scheduler-terminal", member.attempt_id)', "test_mixed_group_emits_all_five_actions_and_park_dominates"),
    ("unsupported-native-profile", "recovery.py", '    if verdict.profile is None or verdict.profile.verdict != "supported":', '    if False:', "test_unsupported_profile_is_parked_before_restore_callback"),
    ("accept-unconfirmed-quiescence", "recovery.py", '    if member.outcome_receipt.status in NON_SETTLED_OPERATIONAL_OUTCOMES:', '    if False:', "test_remote_quiescence_unconfirmed_parks_before_restore"),
    ("ambiguity-as-fresh", "recovery.py", '        return _park(member, "unknown-core-result")', '        return MemberReconciliationV2(member.task_id, "start_fresh", "no-durable-start", None)', "test_mixed_group_emits_all_five_actions_and_park_dominates"),
    ("omit-cx06", "storage.py", "if start.transport_kind != \"local\":", "if False:", "test_remote_gate_negative_matrix_creates_no_binding_or_publication"),
    ("drop-commit-recheck", "storage.py", "            start, record = self._continuation_predecessor_locked(fence, task_dir)\n            rebuilt =", "            start = StartReceipt.decode(Path(fence.output_dir, 'start.json').read_bytes())\n            record = CoreOutcomeRecord.decode(Path(fence.output_dir, 'core-outcome.json').read_bytes())\n            rebuilt =", "test_commit_time_recheck_rejects_concurrent_proof_removal"),
    ("accept-second-successor", "storage.py", "                    raise AdmissionFork(\"fork: a different successor is already admitted for this predecessor\")", "                    return existing", "test_identical_successor_admission_replay_is_idempotent_and_fork_is_rejected"),
    ("non-idempotent-publication", "storage.py", "                    return existing.reconciliation", "                    return park('non-idempotent-replay')", "test_publication_replay_and_exact_member_digests"),
    ("merge-crossed-publication", "storage.py", "                        return park(\"reconciliation-publication-conflict\")", "                        return existing.reconciliation", "test_changed_evidence_cannot_merge_into_publication"),
    ("premature-successor-binding", "storage.py", "        request_digest = digest(request)\n        built =", "        request_digest = digest(request)\n        if successor_binding is not None:\n            _write_new(os.path.join(fence.output_dir, 'premature-successor.json'), successor_binding.encode())\n        built =", "test_remote_gate_negative_matrix_creates_no_binding_or_publication"),
    ("v1-attached-api-drift", "lifecycle/attached.py", "        begin_kwargs = {} if transport_kind == \"local\" else {\"transport_kind\": transport_kind}", "        begin_kwargs = {'transport_kind': None}", "test_v1_attached_api_and_start_bytes_unchanged"),
    ("crossed-invocation-echo", "storage.py", "        if encoded != stored.encode() or encoded != binding.encode():", "        if False:", "test_crossed_or_missing_evidence_parks_without_writes"),
    ("discard-member-digests", "results.py", "            \"member_digests\": [digest(member.to_dict()) for member in self.members],", "            \"member_digests\": [],", "test_publication_replay_and_exact_member_digests"),
)


def test_mutant_anchor_uniqueness():
    source = Path(__file__).parents[1] / "src" / "tpen_executor"
    names = set()
    for name, filename, before, after, killer in MUTANTS:
        assert name not in names
        names.add(name)
        assert (source / filename).read_text().count(before) == 1, name
        assert before != after
        assert any(killer in path.read_text() for path in Path(__file__).parent.glob("test_*.py")), name


def test_concurrent_new_attempt_invalidates_predecessor_before_commit(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    store, group, lease, _, _, _, _ = fixture
    original = DirectoryStore.build_successor_admission_request

    def replace_after_build(self, *args, **kwargs):
        request = original(self, *args, **kwargs)
        allocation = ConfiguredAllocation(lease.allocation_id, lease.backend_job_id, {})
        self.begin_attempt(group, "successor", allocation, allocation_attempt=lease, invocation_id=InvocationId("concurrent-writer"))
        return request

    monkeypatch.setattr(DirectoryStore, "build_successor_admission_request", replace_after_build)
    assert _publish(fixture).aggregate == "parked"
    assert not list(Path(store.root).rglob("successor-admission.json"))
    assert store.read_reconciliation(group, lease) is None


def test_crash_after_ledger_before_group_publication_replays(tmp_path, monkeypatch):
    import tpen_executor.storage as storage

    fixture = _fixture(tmp_path)
    original = storage._write_new

    def fail_group(path, payload, **kwargs):
        if ".reconciliation-" in str(path):
            raise OSError("injected publication failure")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(storage, "_write_new", fail_group)
    assert _publish(fixture).aggregate == "parked"
    store, group, lease, _, _, _, _ = fixture
    ledgers = list(Path(store.root).rglob("successor-admission.json"))
    assert len(ledgers) == 1
    admitted = ledgers[0].read_bytes()
    assert store.read_reconciliation(group, lease) is None
    monkeypatch.setattr(storage, "_write_new", original)
    assert _publish(fixture).aggregate == "replacement_required"
    assert ledgers[0].read_bytes() == admitted
    assert store.read_reconciliation(group, lease).members[0].outcome == "successor_admitted"


def test_different_successor_attempt_parks_after_existing_admission(tmp_path):
    fixture = _fixture(tmp_path)
    store, group, lease, inputs, bindings, successors, fences = fixture
    request, acceptance, successor = successors["successor"]
    store.record_successor_admission(fences["successor"], request, acceptance, predecessor_binding=bindings["successor"], successor_binding=successor)
    new_request = store.build_successor_admission_request(fences["successor"], AttemptId("fork"), "a" * 64, 10)
    successors["successor"] = (new_request, acceptance, replace(successor, attempt_id=AttemptId("fork")))
    snapshot = _files(Path(store.root))
    assert _publish(fixture).members[0].reason_code == "successor-fork"
    assert _files(Path(store.root)) == snapshot


@pytest.mark.parametrize("damage", ["same-lease", "same-invocation", "lineage", "group", "task", "attempt", "acceptance", "request"])
def test_invalid_successor_binding_or_acceptance_writes_nothing(tmp_path, damage):
    fixture = _fixture(tmp_path)
    store, group, lease, _, bindings, successors, _ = fixture
    request, acceptance, successor = successors["successor"]
    if damage == "same-lease":
        successor = replace(successor, allocation_attempt=lease)
    elif damage == "same-invocation":
        successor = replace(successor, invocation_id=bindings["successor"].invocation_id)
    elif damage == "lineage":
        successor = replace(successor, lineage_id="foreign")
    elif damage == "group":
        successor = replace(successor, group_digest="0" * 64)
    elif damage == "task":
        successor = replace(successor, task_id="foreign")
    elif damage == "attempt":
        successor = replace(successor, attempt_id=AttemptId("foreign"))
    elif damage == "acceptance":
        acceptance = "corrupt"
    else:
        request = dict(request, core_evidence={"corrupt": True})
    successors["successor"] = (request, acceptance, successor)
    snapshot = _files(Path(store.root))
    assert _publish(fixture).aggregate == "parked"
    assert _files(Path(store.root)) == snapshot
    assert store.read_reconciliation(group, lease) is None


def test_invocation_wire_is_closed_and_nominal(tmp_path):
    fixture = _fixture(tmp_path)
    binding = fixture[4]["successor"]
    assert MemberInvocationV2.decode(binding.encode()) == binding
    for field in binding.to_dict():
        data = binding.to_dict()
        del data[field]
        with pytest.raises((TypeError, ValueError)):
            MemberInvocationV2.decode(data)
    data = binding.to_dict()
    data["extra"] = True
    with pytest.raises((TypeError, ValueError)):
        MemberInvocationV2.decode(data)
    for field, crossed in (("invocation_id", binding.attempt_id), ("attempt_id", binding.invocation_id)):
        data = binding.to_dict()
        data[field] = crossed.to_dict()
        with pytest.raises((TypeError, ValueError)):
            MemberInvocationV2.decode(data)
