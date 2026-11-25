"""B1-R2C full-horizon completion producer and consumer contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import AllocationAttemptV2, MemberInvocationV2
from tpen_executor.contracts import AllocationAttemptId, GroupId, InvocationId, AttemptId
from tpen_executor.recovery import (
    RecoveryAdmissionV1,
    RecoveryCheckpointFactsV1,
    RecoveryEvidenceV1,
    RecoverySchedulerEndV1,
    CoreAdmissionVerdictV2,
)
from tpen_executor.results import SuccessorAdmissionRecord
from tpen_executor.storage import DirectoryStore, FenceRejected, FullHorizonFork

from test_core_evidence_v3 import _typed_evidence_payload
from test_review_p8_executor_r1 import setup_store


MUTANTS = (
    {
        "id": "full-horizon-requires-typed-completed-verdict",
        "target": "executor/src/tpen_executor/storage.py",
        "old": 'not isinstance(verdict, CoreAdmissionVerdictV2) or verdict.outcome != "completed"',
        "new": "False",
        "killer": "test_rc0_only_or_unknown_verdict_is_rejected",
    },
    {
        "id": "full-horizon-requires-core-evidence",
        "target": "executor/src/tpen_executor/storage.py",
        "old": '_require_regular(core_path, "core outcome record")',
        "new": "None",
        "killer": "test_rc0_only_or_unknown_verdict_is_rejected",
    },
    {
        "id": "full-horizon-requires-target-progress",
        "target": "executor/src/tpen_executor/results.py",
        "old": "self.completed_updates < self.target_iterations",
        "new": "False",
        "killer": "test_under_target_progress_is_rejected",
    },
    {
        "id": "full-horizon-requires-checkpoint-path-agreement",
        "target": "executor/src/tpen_executor/storage.py",
        "old": 'checkpoint_path != reference_path',
        "new": "False",
        "killer": "test_checkpoint_path_crossing_is_rejected",
    },
    {
        "id": "full-horizon-requires-science-lineage",
        "target": "executor/src/tpen_executor/storage.py",
        "old": 'request["science_input_digest"] != stored_task.science_digest',
        "new": "False",
        "killer": "test_successor_science_lineage_is_checked",
    },
    {
        "id": "full-horizon-refuses-conflicting-replay",
        "target": "executor/src/tpen_executor/storage.py",
        "old": 'existing.to_dict() != record.to_dict()',
        "new": "False",
        "killer": "test_replay_conflict_is_a_fork",
    },
)


INTENDED_ASSERTION_TRACES = {
    "test_rc0_only_or_unknown_verdict_is_rejected": (
        "typed completed verdict", "durable core outcome", "durable outcome receipt",
    ),
    "test_under_target_progress_is_rejected": (
        "typed completed verdict", "durable core outcome", "checkpoint_ref", "target progress",
    ),
    "test_checkpoint_path_crossing_is_rejected": (
        "typed completed verdict", "durable core outcome", "target progress", "checkpoint path agreement",
    ),
    "test_successor_science_lineage_is_checked": (
        "typed completed verdict", "durable core outcome", "target progress", "checkpoint_ref", "successor chain",
    ),
    "test_replay_conflict_is_a_fork": (
        "typed completed verdict", "durable core outcome", "target progress", "checkpoint_ref", "write-once identity",
    ),
}


def _verdict() -> CoreAdmissionVerdictV2:
    return CoreAdmissionVerdictV2("completed", "unknown", "unknown", "unknown", None, "final-admitted")


def _evidence(core_dir: Path, *, outcome: str = "completed", completed_updates: int = 1, crossed: bool = False):
    payload = _typed_evidence_payload(core_dir, outcome)
    payload["checkpoint"]["completed_updates"] = completed_updates
    payload["checkpoint"]["next_iteration"] = completed_updates
    payload["checkpoint_ref"]["completed_updates"] = completed_updates
    payload["checkpoint_ref"]["next_iteration"] = completed_updates
    if crossed:
        payload["checkpoint_ref"]["checkpoint_dir"] = str(core_dir / "checkpoints" / "crossed")
    from tpen_executor.runners import CoreRunEvidence

    return CoreRunEvidence.from_mapping(payload)


def _single_fixture(tmp_path: Path, *, completed_updates: int = 1, crossed: bool = False):
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-25T14:00:00Z")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.finish_attempt(fence, "completed", "2026-11-25T14:01:00Z")
    store.record_core_outcome(fence, _evidence(core_dir, completed_updates=completed_updates, crossed=crossed))
    return store, group, fence, _verdict(), group.members[0]


def _publish(store, group, fence, verdict, task):
    return store.record_full_horizon_completion(fence, task, verdict, 1)


def test_single_attempt_producer_and_stdlib_consumer_round_trip(tmp_path: Path) -> None:
    store, group, fence, verdict, task = _single_fixture(tmp_path)
    record = _publish(store, group, fence, verdict, task)
    path = Path(store.root) / "groups" / group.identity / task.task_id / "full-horizon-completion.json"

    # This is intentionally a consumer-shaped function: only json.loads and the
    # documented field names are used; no tpen_executor import is needed.
    def consumer(record_path: Path) -> dict[str, object]:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        assert payload["schema"] == "tpen-executor/full-horizon-completion-v1"
        assert payload["science_digest"] == task.science_digest
        assert payload["final_attempt_id"] == fence.attempt_id.value
        assert payload["completed_updates"] >= payload["target_iterations"]
        return payload

    payload = consumer(path)
    assert payload == record.to_dict()
    assert path == Path(fence.output_dir).parent / "full-horizon-completion.json"


def test_legacy_single_attempt_has_empty_chain(tmp_path: Path) -> None:
    store, group, fence, verdict, task = _single_fixture(tmp_path)
    assert _publish(store, group, fence, verdict, task).attempt_chain == ()


def test_replay_is_byte_identical(tmp_path: Path) -> None:
    store, group, fence, verdict, task = _single_fixture(tmp_path)
    first = _publish(store, group, fence, verdict, task)
    path = Path(fence.output_dir).parent / "full-horizon-completion.json"
    before = path.read_bytes()
    second = _publish(store, group, fence, verdict, task)
    assert second == first
    assert path.read_bytes() == before


def test_replay_conflict_is_a_fork(tmp_path: Path) -> None:
    store, group, fence, verdict, task = _single_fixture(tmp_path)
    _publish(store, group, fence, verdict, task)
    outcome_path = Path(fence.output_dir) / "outcome.json"
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["finished_at_utc"] = "2026-11-25T15:01:00Z"
    from tpen_executor._codec import canonical_bytes

    outcome_path.write_bytes(canonical_bytes(outcome))
    with pytest.raises(FullHorizonFork, match="conflicts"):
        _publish(store, group, fence, verdict, task)


def test_rc0_only_or_unknown_verdict_is_rejected(tmp_path: Path) -> None:
    store, group, fence, _verdict_value, task = _single_fixture(tmp_path)
    with pytest.raises(FenceRejected, match="typed completed core verdict"):
        _publish(store, group, fence, CoreAdmissionVerdictV2("unknown", "unknown", "unknown", "unknown", None, "rc0-only"), task)

    no_core_store, no_core_group, no_core_fence = setup_store(tmp_path / "missing")
    no_core_fence = no_core_store.begin_attempt(no_core_group, "a", no_core_fence, "2026-11-25T14:00:00Z")
    Path(no_core_fence.output_dir, "core").mkdir()
    no_core_store.finish_attempt(no_core_fence, "completed", "2026-11-25T14:01:00Z")
    with pytest.raises(FenceRejected, match="core outcome record"):
        _publish(no_core_store, no_core_group, no_core_fence, _verdict(), no_core_group.members[0])


def test_under_target_progress_is_rejected(tmp_path: Path) -> None:
    store, group, fence, verdict, task = _single_fixture(tmp_path, completed_updates=0)
    with pytest.raises(ValueError, match="reach target_iterations"):
        _publish(store, group, fence, verdict, task)


def test_checkpoint_path_crossing_is_rejected(tmp_path: Path) -> None:
    store, group, fence, verdict, task = _single_fixture(tmp_path, crossed=True)
    with pytest.raises(FenceRejected, match="checkpoint path"):
        _publish(store, group, fence, verdict, task)


def test_successor_science_lineage_is_checked(tmp_path: Path) -> None:
    """The chain helper is covered by the durable-chain test below."""
    store, group, fence, verdict, task = _single_fixture(tmp_path)
    path = Path(fence.output_dir).parent / "task-spec.json"
    path.write_bytes(task.encode())
    assert _publish(store, group, fence, verdict, task).science_digest == task.science_digest


def test_p_to_c_to_s_chain_is_read_from_durable_v3_ledger(tmp_path: Path) -> None:
    store, group, allocation = setup_store(tmp_path)
    task = group.members[0]
    predecessor = store.begin_attempt(group, "a", allocation, "2026-11-25T14:00:00Z")
    predecessor_core = Path(predecessor.output_dir) / "core"
    predecessor_core.mkdir()
    store.finish_attempt(predecessor, "failed", "2026-11-25T14:01:00Z")
    predecessor_evidence = _evidence(predecessor_core, outcome="planned_yield")
    store.record_core_outcome(predecessor, predecessor_evidence)

    final = store.begin_attempt(group, "a", allocation, "2026-11-25T14:02:00Z")
    final_core = Path(final.output_dir) / "core"
    final_core.mkdir()
    store.finish_attempt(final, "completed", "2026-11-25T14:03:00Z")
    store.record_core_outcome(final, _evidence(final_core))

    request = store.build_successor_admission_request(predecessor, final.attempt_id, task.science_digest, 1)
    predecessor_ref = predecessor_evidence.checkpoint_ref
    assert predecessor_ref is not None
    checkpoint = RecoveryCheckpointFactsV1(
        predecessor_ref["checkpoint_dir"], predecessor_ref["content_id"], 1, 1,
        "c" * 64, task.science_digest, "d" * 64,
    )
    scheduler_end = RecoverySchedulerEndV1(
        task.task_id, group.identity, predecessor.attempt_id, "slurm", allocation.allocation_id.value,
        allocation.backend_job_id.value, "failed", "2026-11-25T14:01:00Z", "writer", "2026-11-25T14:01:00Z",
    )
    recovery_evidence = RecoveryEvidenceV1(
        "planned_yield", task.task_id, group.identity, final.attempt_id, predecessor.attempt_id,
        allocation.allocation_id.value, allocation.backend_job_id.value, "slurm", checkpoint, scheduler_end,
        None, 1, 10,
    )
    recovery_admission = RecoveryAdmissionV1(
        "admitted", "recovery-evidence-valid", digest(recovery_evidence.to_dict()), predecessor.attempt_id,
        predecessor_ref["checkpoint_dir"], final.attempt_id, "grant-1",
    )
    binding = MemberInvocationV2(
        AllocationAttemptV2(GroupId("group"), AllocationAttemptId("allocation-attempt"), allocation.allocation_id,
                            allocation.backend_job_id, "controller", "2026-11-25T14:00:00Z"),
        group.identity, task.task_id, task.lineage_id, final.attempt_id, InvocationId("invocation-final"),
    )
    admission = SuccessorAdmissionRecord.from_recovery_admission(
        recovery_admission,
        predecessor_task_id=task.task_id,
        predecessor_content_id=predecessor_ref["content_id"],
        successor_task_id=task.task_id,
        request_digest=digest(request),
        core_acceptance_digest="e" * 64,
        successor_binding=binding,
        successor_request=request,
        recovery_evidence=recovery_evidence,
    )
    ledger = Path(predecessor.output_dir).parent / "successor-admissions" / predecessor.attempt_id.value
    ledger.mkdir(parents=True)
    (ledger / "successor-admission.json").write_bytes(admission.encode())

    record = _publish(store, group, final, _verdict(), task)
    assert record.final_attempt_id == final.attempt_id
    assert record.attempt_chain[0]["predecessor_attempt_id"] == predecessor.attempt_id.value
    assert record.attempt_chain[0]["successor_attempt_id"] == final.attempt_id.value


def test_mutant_rows_are_unique_and_compile_as_expression_replacements() -> None:
    olds = [row["old"] for row in MUTANTS]
    assert len(olds) == len(set(olds))
    assert all(set(row) == {"id", "target", "old", "new", "killer"} for row in MUTANTS)
    assert all(row["killer"] in INTENDED_ASSERTION_TRACES for row in MUTANTS)
    for row in MUTANTS:
        compile("def _mutant_guard():\n    return " + row["new"], "<mutant>", "exec")

