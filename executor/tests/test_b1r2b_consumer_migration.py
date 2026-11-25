"""B1-R2B linked-result and R2 consumer-boundary checks.

These tests are intentionally local/fake-only: they exercise the durable
producer and C1 reader without importing ``tpen`` or starting a scheduler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import AttemptId
from tpen_executor.recovery import RecoveryGrantV1, admit_recovery
from tpen_executor.results import SuccessorAdmissionRecord
from tpen_executor.storage import AdmissionFork, DirectoryStore, read_successor_admission

from test_b1a_b_publication import _fixture
from test_b1r2a_recovery_admission import _facts


MUTANTS = (
    {"id": "omit-r2-checkpoint-link", "target": "executor/src/tpen_executor/results.py", "old": '"checkpoint_path": self.recovery_admission.checkpoint_path', "new": '"checkpoint_path": "crossed"', "killer": "test_r2_link_round_trip_preserves_typed_lineage"},
    {"id": "accept-crossed-r2-attempt", "target": "executor/src/tpen_executor/results.py", "old": "self.recovery_admission.predecessor_attempt_id != self.predecessor_attempt_id", "new": "False", "killer": "test_crossed_r2_link_is_rejected"},
    {"id": "skip-r2-controller-admission", "target": "executor/src/tpen_executor/lifecycle/controller.py", "old": "verdict = None if admitted_item.recovery_input is not None else admission_port.admit(admitted_item)", "new": "verdict = admission_port.admit(admitted_item)", "killer": "test_r2_input_does_not_use_legacy_selector"},
    {"id": "launch-before-r2-publication", "target": "executor/src/tpen_executor/lifecycle/controller.py", "old": "published = store.publish_reconciliation(group, attempt, tuple(frozen_inputs), **publication_kwargs)", "new": "published = checked", "killer": "test_r2_launch_requires_published_admission"},
)


def _real_admission(store, fence, *, successor: AttemptId, backend_job_id: str):
    """Produce an R2 decision from facts bound to the real fixture store."""

    evidence = _facts(
        Path(store.root),
        mode="operator_authorized_unplanned_end",
    )
    checkpoint = evidence.checkpoint
    bound_path = Path(store.root) / "groups" / fence.group_digest / fence.task_id / fence.attempt_id.value / "core" / "checkpoints" / "step"
    # The test only needs the typed output; the fixture's predecessor is the
    # persisted fence, so replace the synthetic current/predecessor identities.
    from dataclasses import replace
    from tpen_executor.recovery import RecoveryCheckpointFactsV1, RecoverySchedulerEndV1

    checkpoint = RecoveryCheckpointFactsV1(
        str(bound_path), checkpoint.content_id, checkpoint.next_iteration,
        checkpoint.completed_updates, checkpoint.program_digest,
        checkpoint.science_digest, checkpoint.h_digest,
    )
    evidence = replace(
        evidence,
        task_id=fence.task_id,
        group_digest=fence.group_digest,
        attempt_id=AttemptId("current-r2"),
        predecessor_attempt_id=fence.attempt_id,
        allocation_id=fence.allocation_id,
        backend_job_id=backend_job_id,
        grant=RecoveryGrantV1(
            evidence.grant.grant_id, evidence.grant.expires_at_utc,
            fence.group_digest, fence.task_id, evidence.grant.program_digest,
            evidence.grant.science_digest, evidence.grant.h_digest,
            evidence.grant.successor_limit,
        ),
        checkpoint=checkpoint,
        scheduler_end=RecoverySchedulerEndV1(
            fence.task_id, fence.group_digest, AttemptId("current-r2"),
            evidence.scheduler_end.scheduler, fence.allocation_id,
            backend_job_id, evidence.scheduler_end.terminal_state,
            evidence.scheduler_end.terminal_at_utc, evidence.scheduler_end.writer_end_identity,
            evidence.scheduler_end.writer_end_at_utc,
        ),
    )
    return admit_recovery(
        evidence, successor, now_utc="2026-11-24T00:02:00Z",
        trusted_store_root=store.root,
    )


def test_r2_link_round_trip_preserves_typed_lineage(tmp_path: Path) -> None:
    """Real R2 output survives the supported producer and actual C1 reader."""

    store, group, _lease, _inputs, bindings, successors, fences = _fixture(
        tmp_path, ("successor",), remote=False
    )
    predecessor = fences["successor"]
    request, acceptance, successor_binding = successors["successor"]
    admission = _real_admission(
        store, predecessor, successor=successor_binding.attempt_id,
        backend_job_id=successor_binding.allocation_attempt.backend_job_id.value,
    )
    assert admission.status == "admitted"
    record = store.record_successor_admission(
        predecessor, request, acceptance,
        predecessor_binding=bindings["successor"],
        successor_binding=successor_binding,
        recovery_admission=admission,
    )
    encoded = record.encode()
    decoded = SuccessorAdmissionRecord.decode(encoded)
    assert decoded.predecessor_attempt_id == admission.predecessor_attempt_id
    assert decoded.checkpoint_path == admission.checkpoint_path
    assert decoded.predecessor["checkpoint_path"] == admission.checkpoint_path
    # C1 consumes the same durable record; it never parses checkpoint_path.
    successor = store.begin_admitted_successor(
        group, "successor", _lease_to_allocation(successor_binding),
        "2026-11-24T00:03:00Z", "local", allocation_attempt=successor_binding.allocation_attempt,
        admission=record,
    )
    result = read_successor_admission(
        successor.output_dir, task_id="successor", attempt_id=successor.attempt_id.value,
        group_digest=group.identity, science_input_digest="a" * 64, target_iterations=10,
    )
    assert result["record"]["predecessor"]["attempt_id"] == admission.predecessor_attempt_id.value
    assert result["record"]["predecessor"]["checkpoint_path"] == admission.checkpoint_path


def _lease_to_allocation(binding):
    from tpen_executor.allocations import ConfiguredAllocation

    return ConfiguredAllocation(
        binding.allocation_attempt.allocation_id,
        binding.allocation_attempt.backend_job_id,
        {},
    )


def test_crossed_r2_link_is_rejected(tmp_path: Path) -> None:
    store, _group, _lease, _inputs, bindings, successors, fences = _fixture(
        tmp_path, ("successor",), remote=False
    )
    predecessor = fences["successor"]
    request, acceptance, successor_binding = successors["successor"]
    admission = _real_admission(
        store, predecessor, successor=successor_binding.attempt_id,
        backend_job_id=successor_binding.allocation_attempt.backend_job_id.value,
    )
    crossed = type(admission)(
        admission.status, admission.reason, admission.evidence_digest,
        AttemptId("other-predecessor"), admission.checkpoint_path,
        admission.successor_attempt_id, admission.grant_id,
    )
    with pytest.raises(ValueError, match="crosses successor identity"):
        SuccessorAdmissionRecord.from_recovery_admission(
            crossed, predecessor_task_id="successor", predecessor_content_id="a" * 64,
            successor_task_id="successor", request_digest=digest(request),
            core_acceptance_digest=acceptance, successor_binding=successor_binding,
            successor_request=request,
        )


def test_r2_duplicate_replay_is_idempotent_and_fork_parks(tmp_path: Path) -> None:
    from dataclasses import replace

    store, _group, _lease, _inputs, bindings, successors, fences = _fixture(
        tmp_path, ("successor",), remote=False
    )
    predecessor = fences["successor"]
    request, acceptance, successor_binding = successors["successor"]
    admission = _real_admission(
        store, predecessor, successor=successor_binding.attempt_id,
        backend_job_id=successor_binding.allocation_attempt.backend_job_id.value,
    )
    first = store.record_successor_admission(
        predecessor, request, acceptance,
        predecessor_binding=bindings["successor"], successor_binding=successor_binding,
        recovery_admission=admission,
    )
    assert store.record_successor_admission(
        predecessor, request, acceptance,
        predecessor_binding=bindings["successor"], successor_binding=successor_binding,
        recovery_admission=admission,
    ) == first
    other_attempt = AttemptId("other-successor")
    other_request = dict(request)
    other_request["successor"] = {"task_id": "successor", "attempt_id": other_attempt.value}
    other_binding = replace(successor_binding, attempt_id=other_attempt)
    other_admission = type(admission)(
        admission.status, admission.reason, admission.evidence_digest,
        admission.predecessor_attempt_id, admission.checkpoint_path,
        other_attempt, admission.grant_id,
    )
    with pytest.raises(AdmissionFork):
        store.record_successor_admission(
            predecessor, other_request, acceptance,
            predecessor_binding=bindings["successor"], successor_binding=other_binding,
            recovery_admission=other_admission,
        )


def test_r2_input_does_not_use_legacy_selector() -> None:
    source = Path(__file__).parents[1].joinpath("src/tpen_executor/lifecycle/controller.py").read_text()
    assert "admitted_item.recovery_input is not None" in source


def test_r2_launch_requires_published_admission() -> None:
    source = Path(__file__).parents[1].joinpath("src/tpen_executor/lifecycle/controller.py").read_text()
    assert "publish_reconciliation" in source
    assert source.index("publish_reconciliation") < source.index("executor.submit")


def test_mutant_sources_compile_and_killers_are_declared() -> None:
    names = {name for name, value in globals().items() if name.startswith("test_") and callable(value)}
    assert len({row["id"] for row in MUTANTS}) == len(MUTANTS)
    for row in MUTANTS:
        source = Path(__file__).parents[1].joinpath(row["target"].split("executor/", 1)[1]).read_text()
        assert source.count(row["old"]) == 1
        assert row["killer"] in names
        compile(source.replace(row["old"], row["new"], 1), row["target"], "exec")
