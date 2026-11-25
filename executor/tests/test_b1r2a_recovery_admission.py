"""B1-R2A versioned recovery facts and fail-closed admission predicate."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import AttemptId
from tpen_executor.recovery import (
    RecoveryCheckpointFactsV1,
    RecoveryEvidenceV1,
    RecoveryGrantV1,
    RecoverySchedulerEndV1,
    admit_recovery,
)


GROUP = "a" * 64
PROGRAM = "b" * 64
SCIENCE = "manifest_sha256:" + "c" * 64 + ":row:17"
H = "d" * 64
_DEFAULT_GRANT = object()


# Each deliberate fault is scoped to the branch exercised by the corresponding
# killer.  The Cannon mutation runner consumes this tuple directly.
MUTANTS = (
    {"id": "scheduler-identity-bypass", "target": "executor/src/tpen_executor/recovery.py", "old": "evidence.scheduler_end.attempt_id != evidence.attempt_id", "new": "False", "killer": "test_scheduler_identity_mismatch_parks"},
    {"id": "terminal-state-bypass", "target": "executor/src/tpen_executor/recovery.py", "old": "evidence.scheduler_end.terminal_state not in RECOVERY_TERMINAL_STATES", "new": "False", "killer": "test_unknown_scheduler_end_parks"},
    {"id": "grant-bypass", "target": "executor/src/tpen_executor/recovery.py", "old": "return parked(\"grant-missing\")", "new": "return RecoveryAdmissionV1(\"admitted\", \"recovery-evidence-valid\", evidence_digest, evidence.predecessor_attempt_id, evidence.checkpoint_path, successor_attempt_id, \"fault-injected-grant\")", "killer": "test_missing_grant_defaults_off"},
    {"id": "expiry-bypass", "target": "executor/src/tpen_executor/recovery.py", "old": "datetime.fromisoformat(evidence.grant.expires_at_utc.replace(\"Z\", \"+00:00\")) <= datetime.fromisoformat(now_utc.replace(\"Z\", \"+00:00\"))", "new": "False", "killer": "test_expired_grant_parks"},
    {"id": "predecessor-core-bypass", "target": "executor/src/tpen_executor/recovery.py", "old": "if not _checkpoint_predecessor_core(evidence.checkpoint_path, trusted_store_root, evidence.group_digest, evidence.task_id, evidence.predecessor_attempt_id):", "new": "if False:", "killer": "test_checkpoint_path_must_be_predecessor_core"},
    {"id": "scheduler-backend-bypass", "target": "executor/src/tpen_executor/recovery.py", "old": "or evidence.scheduler_end.scheduler != evidence.expected_scheduler", "new": "or False", "killer": "test_scheduler_backend_mismatch_parks"},
    {"id": "counter-conflation", "target": "executor/src/tpen_executor/recovery.py", "old": "self.checkpoint.completed_updates > self.prior_completed_updates", "new": "self.checkpoint.next_iteration > self.prior_next_iteration", "killer": "test_next_iteration_without_completed_update_is_not_progress"},
    {"id": "repeat-progress-bypass", "target": "executor/src/tpen_executor/recovery.py", "old": "if evidence.no_progress_count >= evidence.no_progress_threshold and not evidence.real_committed_progress:", "new": "if False:", "killer": "test_repeated_identical_progress_parks"},
    {"id": "ambiguous-submit-retry", "target": "executor/src/tpen_executor/recovery.py", "old": "if evidence.external_submit_state == \"ambiguous\":", "new": "if False:", "killer": "test_ambiguous_external_submit_parks"},
)


def _facts(tmp_path: Path, *, grant: RecoveryGrantV1 | None | object = _DEFAULT_GRANT, **overrides: object) -> RecoveryEvidenceV1:
    predecessor = AttemptId("predecessor")
    current = AttemptId("current")
    checkpoint_path = tmp_path / "groups" / GROUP / "task" / predecessor.value / "core" / "checkpoints" / "step_000004"
    checkpoint = RecoveryCheckpointFactsV1(
        str(checkpoint_path),
        "e" * 64,
        5,
        4,
        PROGRAM,
        SCIENCE,
        H,
    )
    scheduler_end = RecoverySchedulerEndV1(
        "task",
        GROUP,
        current,
        "slurm",
        "allocation",
        "job",
        "failed",
        "2026-11-24T00:01:00Z",
        "writer-1",
        "2026-11-24T00:01:01Z",
    )
    base: dict[str, object] = {
        "mode": "operator_authorized_unplanned_end",
        "task_id": "task",
        "group_digest": GROUP,
        "attempt_id": current,
        "predecessor_attempt_id": predecessor,
        "allocation_id": "allocation",
        "backend_job_id": "job",
        "expected_scheduler": "slurm",
        "checkpoint": checkpoint,
        "scheduler_end": scheduler_end,
        "grant": RecoveryGrantV1("grant", "2026-11-24T01:00:00Z", GROUP, "task", PROGRAM, SCIENCE, H, 3)
        if grant is _DEFAULT_GRANT else grant,
        "budget_remaining": 1,
        "no_progress_threshold": 2,
        "no_progress_count": 0,
        "prior_content_id": "f" * 64,
        "prior_completed_updates": 3,
        "prior_next_iteration": 3,
        "external_submit_state": "not_submitted",
    }
    base.update(overrides)
    return RecoveryEvidenceV1(**base)  # type: ignore[arg-type]


def _admit(tmp_path: Path, evidence: RecoveryEvidenceV1, *, trusted_store_root: Path | None = None):
    return admit_recovery(
        evidence,
        AttemptId("successor"),
        now_utc="2026-11-24T00:02:00Z",
        trusted_store_root=str(tmp_path if trusted_store_root is None else trusted_store_root),
    )


def test_valid_unplanned_end_admits_with_typed_lineage(tmp_path: Path) -> None:
    result = _admit(tmp_path, _facts(tmp_path))
    assert result.status == "admitted"
    assert result.predecessor_attempt_id == AttemptId("predecessor")
    assert result.checkpoint_path.endswith("/predecessor/core/checkpoints/step_000004")
    assert result.grant_id == "grant"


def test_round_trip_preserves_versioned_evidence_and_admission(tmp_path: Path) -> None:
    evidence = _facts(tmp_path)
    assert RecoveryEvidenceV1.decode(evidence.encode()) == evidence
    result = _admit(tmp_path, evidence)
    assert type(result).decode(result.encode()) == result
    assert digest(evidence.to_dict()) == result.evidence_digest


def test_missing_grant_defaults_off(tmp_path: Path) -> None:
    result = _admit(tmp_path, _facts(tmp_path, grant=None))
    assert (result.status, result.reason) == ("parked", "grant-missing")


def test_expired_grant_parks(tmp_path: Path) -> None:
    grant = RecoveryGrantV1("grant", "2026-11-24T00:01:00Z", GROUP, "task", PROGRAM, SCIENCE, H, 3)
    result = _admit(tmp_path, _facts(tmp_path, grant=grant))
    assert (result.status, result.reason) == ("parked", "grant-expired")


def test_scheduler_identity_mismatch_parks(tmp_path: Path) -> None:
    scheduler_end = _facts(tmp_path).scheduler_end
    mismatched = RecoverySchedulerEndV1(
        scheduler_end.task_id, scheduler_end.group_digest, AttemptId("other"), scheduler_end.scheduler,
        scheduler_end.allocation_id, scheduler_end.backend_job_id, scheduler_end.terminal_state,
        scheduler_end.terminal_at_utc, scheduler_end.writer_end_identity, scheduler_end.writer_end_at_utc,
    )
    result = _admit(tmp_path, _facts(tmp_path, scheduler_end=mismatched))
    assert (result.status, result.reason) == ("parked", "scheduler-identity-mismatch")


def test_scheduler_backend_match_admits(tmp_path: Path) -> None:
    result = _admit(tmp_path, _facts(tmp_path, expected_scheduler="slurm"))
    assert (result.status, result.reason) == ("admitted", "recovery-evidence-valid")


def test_scheduler_backend_mismatch_parks(tmp_path: Path) -> None:
    scheduler_end = _facts(tmp_path).scheduler_end
    crossed = RecoverySchedulerEndV1(
        scheduler_end.task_id, scheduler_end.group_digest, scheduler_end.attempt_id, "pbs",
        scheduler_end.allocation_id, scheduler_end.backend_job_id, scheduler_end.terminal_state,
        scheduler_end.terminal_at_utc, scheduler_end.writer_end_identity, scheduler_end.writer_end_at_utc,
    )
    result = _admit(tmp_path, _facts(tmp_path, scheduler_end=crossed, expected_scheduler="slurm"))
    assert (result.status, result.reason) == ("parked", "scheduler-identity-mismatch")


def test_unknown_scheduler_end_parks(tmp_path: Path) -> None:
    scheduler_end = _facts(tmp_path).scheduler_end
    unknown = RecoverySchedulerEndV1(
        scheduler_end.task_id, scheduler_end.group_digest, scheduler_end.attempt_id, scheduler_end.scheduler,
        scheduler_end.allocation_id, scheduler_end.backend_job_id, "unknown", scheduler_end.terminal_at_utc,
        scheduler_end.writer_end_identity, scheduler_end.writer_end_at_utc,
    )
    result = _admit(tmp_path, _facts(tmp_path, scheduler_end=unknown))
    assert (result.status, result.reason) == ("parked", "unknown-scheduler-end")


@pytest.mark.parametrize(
    "path_kind,expected",
    [
        ("unrelated", ("parked", "checkpoint-lineage-mismatch")),
        ("current", ("parked", "checkpoint-is-current-attempt")),
        ("nested-current", ("parked", "checkpoint-is-current-attempt")),
        ("nested-foreign", ("parked", "checkpoint-lineage-mismatch")),
        ("different-store", ("parked", "checkpoint-lineage-mismatch")),
        ("bound-groups", ("admitted", "recovery-evidence-valid")),
    ],
)
def test_checkpoint_path_must_be_predecessor_core(tmp_path: Path, path_kind: str, expected: tuple[str, str]) -> None:
    trusted_root = tmp_path / "store"
    if path_kind == "unrelated":
        path = trusted_root / "groups" / GROUP / "task" / "other" / "core" / "checkpoints" / "step_000004"
    elif path_kind == "current":
        path = trusted_root / "groups" / GROUP / "task" / "current" / "core" / "checkpoints" / "step_000004"
    elif path_kind == "nested-current":
        path = trusted_root / "groups" / GROUP / "task" / "current" / "core" / "checkpoints" / "groups" / GROUP / "task" / "predecessor" / "core" / "step_000004"
    elif path_kind == "nested-foreign":
        path = trusted_root / "groups" / "foreign-group" / "foreign-task" / "foreign-attempt" / "core" / "checkpoints" / "groups" / GROUP / "task" / "predecessor" / "core" / "step_000004"
    elif path_kind == "different-store":
        path = tmp_path / "other-store" / "groups" / GROUP / "task" / "predecessor" / "core" / "checkpoints" / "step_000004"
    else:
        trusted_root = tmp_path / "groups"
        path = trusted_root / "groups" / GROUP / "task" / "predecessor" / "core" / "checkpoints" / "step_000004"
    checkpoint = RecoveryCheckpointFactsV1(str(path), "e" * 64, 5, 4, PROGRAM, SCIENCE, H)
    result = _admit(tmp_path, _facts(tmp_path, checkpoint=checkpoint), trusted_store_root=trusted_root)
    assert (result.status, result.reason) == expected


def test_next_iteration_without_completed_update_is_not_progress(tmp_path: Path) -> None:
    checkpoint = RecoveryCheckpointFactsV1(
        str(tmp_path / "groups" / GROUP / "task" / "predecessor" / "core" / "checkpoints" / "step_000004"),
        "e" * 64, 5, 3, PROGRAM, SCIENCE, H,
    )
    evidence = _facts(tmp_path, checkpoint=checkpoint, no_progress_count=2)
    result = _admit(tmp_path, evidence)
    assert (result.status, result.reason) == ("parked", "no-progress-exhausted")


def test_repeated_identical_progress_parks(tmp_path: Path) -> None:
    checkpoint = RecoveryCheckpointFactsV1(
        str(tmp_path / "groups" / GROUP / "task" / "predecessor" / "core" / "checkpoints" / "step_000004"),
        "f" * 64, 4, 3, PROGRAM, SCIENCE, H,
    )
    evidence = _facts(tmp_path, checkpoint=checkpoint, prior_content_id="f" * 64, prior_completed_updates=3, no_progress_count=2)
    result = _admit(tmp_path, evidence)
    assert (result.status, result.reason) == ("parked", "no-progress-exhausted")


def test_ambiguous_external_submit_parks(tmp_path: Path) -> None:
    result = _admit(tmp_path, _facts(tmp_path, external_submit_state="ambiguous"))
    assert (result.status, result.reason) == ("parked", "ambiguous-external-submit")


def test_mutant_table_anchors_are_unique_and_killers_exist() -> None:
    source = Path(__file__).parents[1] / "src" / "tpen_executor" / "recovery.py"
    text = source.read_text(encoding="utf-8")
    names: set[str] = set()
    anchors: set[tuple[str, str]] = set()
    for mutant in MUTANTS:
        assert mutant["id"] not in names
        names.add(mutant["id"])
        anchor = (mutant["target"], mutant["old"])
        assert anchor not in anchors
        anchors.add(anchor)
        assert mutant["old"] in text
        assert mutant["killer"] in globals()

