"""CX-04b executor request bridge and successor-admission ledger probes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen_executor.contracts import AttemptId
from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.results import (
    CoreOutcomeRecord,
    ManifestAttempt,
    RemoteTeardownReceipt,
    StartReceipt,
)
from tpen_executor.storage import AdmissionFork, DirectoryStore, FenceRejected
from tpen_executor.contracts import AllocationId, BackendJobId

from test_review_p8_executor_r1 import setup_store
from test_core_evidence_v3 import _planned_evidence, _typed_evidence


def _planned_attempt(tmp_path: Path):
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z", transport_kind="remote")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.finish_remote_attempt(
        fence,
        {
            "schema": "tpen-executor/leaf-teardown-v1",
            "group_digest": fence.group_digest,
            "task_id": fence.task_id,
            "attempt_id": fence.attempt_id.value,
            "allocation_id": fence.allocation_id,
            "backend_job_id": allocation.backend_job_id.value,
            "child_group_reaped": True,
            "inspection": "complete",
        },
        "c" * 64,
        "failed",
        "2026-11-14T00:01:00Z",
    )
    store.record_core_outcome(fence, _planned_evidence(core_dir))
    return store, group, fence


def test_local_planned_yield_is_admitted_without_remote_receipt(tmp_path: Path) -> None:
    """Local CX-04b attempts retain admission without S3.C proof."""
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.finish_attempt(fence, "failed", "2026-11-14T00:01:00Z")
    store.record_core_outcome(fence, _planned_evidence(core_dir))
    request = store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    assert request["successor"] == {"task_id": "a", "attempt_id": "successor-1"}


def test_unknown_transport_marker_fails_closed(tmp_path: Path) -> None:
    """A missing start discriminator cannot downgrade a remote attempt to local."""
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z", transport_kind="remote")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.finish_remote_attempt(
        fence,
        {
            "schema": "tpen-executor/leaf-teardown-v1",
            "group_digest": fence.group_digest,
            "task_id": fence.task_id,
            "attempt_id": fence.attempt_id.value,
            "allocation_id": fence.allocation_id,
            "backend_job_id": allocation.backend_job_id.value,
            "child_group_reaped": True,
            "inspection": "complete",
        },
        "c" * 64,
        "failed",
        "2026-11-14T00:01:00Z",
    )
    store.record_core_outcome(fence, _planned_evidence(core_dir))
    start_path = Path(fence.output_dir, "start.json")
    start = json.loads(start_path.read_text(encoding="utf-8"))
    start["schema"] = "tpen-executor/start-receipt-v1"
    start.pop("transport_kind")
    start_path.write_bytes(canonical_bytes(start))
    Path(fence.output_dir, "remote-teardown.json").unlink()
    with pytest.raises(FenceRejected, match="teardown-unconfirmed"):
        store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)


def test_forged_transport_marker_fails_closed(tmp_path: Path) -> None:
    """An invalid durable transport kind is not interpreted as local."""
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z", transport_kind="remote")
    start_path = Path(fence.output_dir, "start.json")
    start = json.loads(start_path.read_text(encoding="utf-8"))
    start["transport_kind"] = "forged"
    start_path.write_bytes(canonical_bytes(start))
    with pytest.raises(FenceRejected, match="start receipt is unavailable"):
        store.finish_attempt(fence, "failed", "2026-11-14T00:01:00Z")


def test_build_and_record_successor_admission_uses_canonical_core_record(tmp_path: Path) -> None:
    store, group, fence = _planned_attempt(tmp_path)
    receipt_path = Path(fence.output_dir, "remote-teardown.json")
    receipt = RemoteTeardownReceipt.decode(receipt_path.read_bytes())
    assert receipt_path.read_bytes() == receipt.encode()
    assert receipt.child_group_reaped is True and receipt.inspection == "complete"
    request = store.build_successor_admission_request(
        fence, AttemptId("successor-1"), "a" * 64, 10,
    )
    record = CoreOutcomeRecord.decode(Path(fence.output_dir, "core-outcome.json").read_bytes())
    assert request["predecessor"]["core_outcome_digest"] == digest(record.to_dict())
    assert request["successor"] == {"task_id": "a", "attempt_id": "successor-1"}

    admission = store.record_successor_admission(fence, request, "b" * 64)
    ledger_path = Path(fence.output_dir).parent / "successor-admissions" / fence.attempt_id.value / "successor-admission.json"
    assert json.loads(ledger_path.read_text(encoding="utf-8")) == admission.to_dict()
    manifest = store.manifest(group)
    assert manifest.segments == ()
    assert manifest.attempts[0].successor_admission == admission


def test_identical_successor_admission_replay_is_idempotent_and_fork_is_rejected(tmp_path: Path) -> None:
    store, _group, fence = _planned_attempt(tmp_path)
    request = store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    first = store.record_successor_admission(fence, request, "b" * 64)
    assert store.record_successor_admission(fence, request, "b" * 64) == first
    with pytest.raises(AdmissionFork, match="fork"):
        store.record_successor_admission(fence, request, "c" * 64)

    fork_request = store.build_successor_admission_request(fence, AttemptId("successor-2"), "a" * 64, 10)
    with pytest.raises(AdmissionFork, match="fork"):
        store.record_successor_admission(fence, fork_request, "d" * 64)


@pytest.mark.parametrize("acceptance", [None, "", "not-a-sha256-digest"])
def test_missing_or_invalid_core_acceptance_writes_no_ledger(tmp_path: Path, acceptance: object) -> None:
    store, _group, fence = _planned_attempt(tmp_path)
    request = store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    with pytest.raises((TypeError, ValueError)):
        store.record_successor_admission(fence, request, acceptance)  # type: ignore[arg-type]
    assert not (Path(fence.output_dir).parent / "successor-admissions").exists()


def test_omitted_core_acceptance_writes_no_ledger(tmp_path: Path) -> None:
    store, _group, fence = _planned_attempt(tmp_path)
    request = store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    with pytest.raises(TypeError):
        store.record_successor_admission(fence, request)  # type: ignore[call-arg]
    assert not (Path(fence.output_dir).parent / "successor-admissions").exists()


def test_manifest_ignores_leftover_admission_temp_but_rejects_unknown_extra(tmp_path: Path) -> None:
    store, group, fence = _planned_attempt(tmp_path)
    request = store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    store.record_successor_admission(fence, request, "b" * 64)
    ledger_dir = Path(fence.output_dir).parent / "successor-admissions"
    (ledger_dir / ".tmp-admission-crash-leftover").write_bytes(b"partial")
    assert store.manifest(group).attempts[0].successor_admission is not None

    predecessor_dir = ledger_dir / fence.attempt_id.value
    (predecessor_dir / "unexpected-extra").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected entries"):
        store.manifest(group)


def test_manifest_attempt_v2_round_trips_core_only_and_core_plus_admission(tmp_path: Path) -> None:
    store, group, fence = _planned_attempt(tmp_path)
    request = store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    core_only = store.manifest(group).attempts[0]
    assert core_only.successor_admission is None
    assert ManifestAttempt.decode(core_only.encode()) == core_only

    store.record_successor_admission(fence, request, "b" * 64)
    core_and_admission = store.manifest(group).attempts[0]
    assert core_and_admission.successor_admission is not None
    assert ManifestAttempt.decode(core_and_admission.encode()) == core_and_admission


def test_start_receipt_v1_byte_round_trip() -> None:
    receipt = StartReceipt(
        "task-a", AttemptId("attempt-1"), AllocationId("allocation-1"),
        BackendJobId("job-1"), "a" * 64, "2026-11-14T00:00:00Z", "/runs/a1",
    )
    assert receipt.to_dict()["schema"] == StartReceipt.schema
    assert "transport_kind" not in receipt.to_dict()
    assert StartReceipt.decode(receipt.encode()) == receipt
    assert StartReceipt.decode(canonical_bytes(receipt.to_dict())).encode() == receipt.encode()


def test_start_receipt_v2_round_trip() -> None:
    receipt = StartReceipt(
        "task-a", AttemptId("attempt-1"), AllocationId("allocation-1"),
        BackendJobId("job-1"), "a" * 64, "2026-11-14T00:00:00Z", "/runs/a1", "remote",
    )
    assert receipt.to_dict()["schema"] == StartReceipt.schema_v2
    assert receipt.to_dict()["transport_kind"] == "remote"
    assert StartReceipt.decode(receipt.encode()) == receipt


@pytest.mark.parametrize(
    "transport_kind",
    [pytest.param(None, id="null"), pytest.param([], id="list"), pytest.param({}, id="dict"), pytest.param(7, id="int"), pytest.param("missing", id="missing")],
)
def test_start_receipt_v2_rejects_invalid_transport_kind(transport_kind: object) -> None:
    receipt = StartReceipt(
        "task-a", AttemptId("attempt-1"), AllocationId("allocation-1"),
        BackendJobId("job-1"), "a" * 64, "2026-11-14T00:00:00Z", "/runs/a1",
    ).to_dict()
    receipt["schema"] = StartReceipt.schema_v2
    if transport_kind == "missing":
        pass
    else:
        receipt["transport_kind"] = transport_kind
    with pytest.raises(ValueError, match="transport_kind|keys differ"):
        StartReceipt.from_mapping(receipt)


def test_noneligible_outcome_writes_no_admission_or_segment(tmp_path: Path) -> None:
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.finish_attempt(fence, "failed", "2026-11-14T00:01:00Z")
    store.record_core_outcome(fence, _typed_evidence(core_dir, "completed"))
    with pytest.raises(FenceRejected, match="continuation-eligible"):
        store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    assert not (Path(fence.output_dir).parent / "successor-admissions").exists()
    assert not (Path(fence.output_dir) / "segment.json").exists()


def test_missing_positive_remote_receipt_refuses_without_request_or_ledger(tmp_path: Path) -> None:
    store, _group, fence = _planned_attempt(tmp_path)
    Path(fence.output_dir, "remote-teardown.json").unlink()
    with pytest.raises(FenceRejected, match="teardown-unconfirmed"):
        store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    assert not (Path(fence.output_dir).parent / "successor-admissions").exists()


def test_symlinked_remote_receipt_refuses_without_request_or_ledger(tmp_path: Path) -> None:
    store, _group, fence = _planned_attempt(tmp_path)
    receipt_path = Path(fence.output_dir, "remote-teardown.json")
    valid_copy = tmp_path / "valid-remote-teardown.json"
    receipt_path.rename(valid_copy)
    receipt_path.symlink_to(valid_copy)

    # CX6M2 drops _require_regular; read_bytes would then follow this symlink
    # and incorrectly admit the successor despite the non-regular receipt path.
    with pytest.raises(FenceRejected, match="teardown-unconfirmed"):
        store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    assert not (Path(fence.output_dir).parent / "successor-admissions").exists()


def test_unconfirmed_remote_outcome_refuses_and_writes_no_ledger(tmp_path: Path) -> None:
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.finish_attempt(fence, "remote-quiescence-unconfirmed", "2026-11-14T00:01:00Z")
    store.record_core_outcome(fence, _planned_evidence(core_dir))
    with pytest.raises(FenceRejected, match="teardown-unconfirmed"):
        store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    assert not (Path(fence.output_dir).parent / "successor-admissions").exists()


def test_unconfirmed_status_is_refused_even_with_a_positive_receipt(tmp_path: Path) -> None:
    store, _group, fence = _planned_attempt(tmp_path)
    outcome_path = Path(fence.output_dir, "outcome.json")
    payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    payload["status"] = "remote-quiescence-unconfirmed"
    outcome_path.write_bytes(canonical_bytes(payload))
    with pytest.raises(FenceRejected, match="teardown-unconfirmed"):
        store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    assert not (Path(fence.output_dir).parent / "successor-admissions").exists()


@pytest.mark.parametrize(
    "mutation",
    ["task", "attempt", "allocation", "backend", "group", "false-reaped", "not-run", "noncanonical", "malformed"],
)
def test_invalid_remote_receipt_refuses_the_same_admission_gate(tmp_path: Path, mutation: str) -> None:
    store, _group, fence = _planned_attempt(tmp_path)
    receipt_path = Path(fence.output_dir, "remote-teardown.json")
    receipt = RemoteTeardownReceipt.decode(receipt_path.read_bytes())
    payload = receipt.to_dict()
    if mutation == "task":
        payload["task_id"] = "crossed-task"
    elif mutation == "attempt":
        payload["attempt_id"] = "crossed-attempt"
    elif mutation == "allocation":
        payload["allocation_id"] = "crossed-allocation"
    elif mutation == "backend":
        payload["backend_job_id"] = "crossed-backend"
    elif mutation == "group":
        payload["group_digest"] = "d" * 64
    elif mutation == "false-reaped":
        payload["child_group_reaped"] = False
    elif mutation == "not-run":
        payload["inspection"] = "not-run"
    elif mutation == "noncanonical":
        receipt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload = None
    else:
        receipt_path.write_bytes(b"{" + b"\"schema\": \"tpen-executor/remote-teardown-v1\"}")
        payload = None
    if payload is not None:
        replacement = RemoteTeardownReceipt.from_mapping(payload)
        receipt_path.write_bytes(replacement.encode())
    with pytest.raises(FenceRejected, match="teardown-unconfirmed"):
        store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    assert not (Path(fence.output_dir).parent / "successor-admissions").exists()


def test_fresh_store_reads_positive_receipt_and_preserves_replay_fork_rules(tmp_path: Path) -> None:
    store, group, fence = _planned_attempt(tmp_path)
    request = store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    reopened = type(store)(store.root)
    first = reopened.record_successor_admission(fence, request, "b" * 64)
    assert reopened.record_successor_admission(fence, request, "b" * 64) == first
    with pytest.raises(AdmissionFork, match="fork"):
        reopened.record_successor_admission(
            fence,
            reopened.build_successor_admission_request(fence, AttemptId("successor-2"), "a" * 64, 10),
            "d" * 64,
        )
    assert reopened.manifest(group).segments == ()


def test_record_successor_admission_rechecks_teardown_under_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _group, fence = _planned_attempt(tmp_path)
    request = store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    receipt_path = Path(fence.output_dir, "remote-teardown.json")
    real_predicate = DirectoryStore._continuation_predecessor_locked
    predicate_calls = 0

    def remove_proof_after_preliminary_build(self, locked_fence, task_dir):
        nonlocal predicate_calls
        result = real_predicate(self, locked_fence, task_dir)
        predicate_calls += 1
        if predicate_calls == 1:
            receipt_path.unlink()
        return result

    monkeypatch.setattr(DirectoryStore, "_continuation_predecessor_locked", remove_proof_after_preliminary_build)
    with pytest.raises(FenceRejected, match="teardown-unconfirmed"):
        store.record_successor_admission(fence, request, "b" * 64)
    assert predicate_calls == 2
    assert not (Path(fence.output_dir).parent / "successor-admissions").exists()


def test_identical_successor_replay_refuses_after_remote_proof_removed(tmp_path: Path) -> None:
    store, group, fence = _planned_attempt(tmp_path)
    request = store.build_successor_admission_request(fence, AttemptId("successor-1"), "a" * 64, 10)
    first = store.record_successor_admission(fence, request, "b" * 64)
    receipt_path = Path(fence.output_dir, "remote-teardown.json")
    receipt_path.unlink()

    with pytest.raises(FenceRejected, match="teardown-unconfirmed"):
        store.record_successor_admission(fence, request, "b" * 64)
    ledger_path = Path(fence.output_dir).parent / "successor-admissions" / fence.attempt_id.value / "successor-admission.json"
    assert json.loads(ledger_path.read_text(encoding="utf-8")) == first.to_dict()
    assert store.manifest(group).segments == ()

