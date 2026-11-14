"""Conformance tests for immutable executor receipts and manifests."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId
from tpen_executor.results import (
    AllocationObservation,
    CoreOutcomeRecord,
    ManifestAttempt,
    OutcomeReceipt,
    RunManifest,
    StartReceipt,
)
from tpen_executor.runners import CoreRunEvidence


FIXTURES = Path(__file__).parent / "fixtures"
GROUP_DIGEST = "76a21ab64e2533a35421b67a8c16c832967fcc6cadf730c9fdfd505840389ebf"
OLD_V1_ATTEMPT_BYTES = b'{"attempt_id":"attempt-1","outcome_receipt":{"attempt_id":"attempt-1","finished_at_utc":"2026-11-04T00:01:00Z","group_digest":"76a21ab64e2533a35421b67a8c16c832967fcc6cadf730c9fdfd505840389ebf","output_dir":"/runs/group-1/task-a/attempt-1","result_digest":"18b9e78cf2fe0d58544c12e59fda8cf04b18230222d1f7fa98fd79f11f698102","schema":"tpen-executor/outcome-receipt-v1","status":"completed","task_id":"task-a"},"output_dir":"/runs/group-1/task-a/attempt-1","schema":"tpen-executor/manifest-attempt-v1","start_receipt":{"allocation_id":"allocation-1","attempt_id":"attempt-1","backend_job_id":"job-1","group_digest":"76a21ab64e2533a35421b67a8c16c832967fcc6cadf730c9fdfd505840389ebf","output_dir":"/runs/group-1/task-a/attempt-1","schema":"tpen-executor/start-receipt-v1","started_at_utc":"2026-11-04T00:00:00Z","task_id":"task-a"},"task_id":"task-a"}'


def _fixture_bytes(name: str) -> bytes:
    """Read canonical JSON text while ignoring the file's terminal newline."""

    return (FIXTURES / f"{name}.json").read_text(encoding="utf-8").rstrip("\n").encode("utf-8")


def _records() -> dict[str, object]:
    attempt = AttemptId("attempt-1")
    allocation = AllocationId("allocation-1")
    job = BackendJobId("job-1")
    start = StartReceipt("task-a", attempt, allocation, job, GROUP_DIGEST, "2026-11-04T00:00:00Z", "/runs/group-1/task-a/attempt-1")
    outcome = OutcomeReceipt("task-a", attempt, GROUP_DIGEST, "2026-11-04T00:01:00Z", "completed", start.output_dir, digest({"energy": -2.0}))
    observation = AllocationObservation(allocation, "2026-11-04T00:00:01Z", "running", {"host": "node-1", "slots": 2})
    member = ManifestAttempt("task-a", attempt, start, outcome, start.output_dir)
    manifest = RunManifest("group-1", "76a21ab64e2533a35421b67a8c16c832967fcc6cadf730c9fdfd505840389ebf", (member,))
    return {"start-receipt": start, "outcome-receipt": outcome, "allocation-observation": observation, "run-manifest": manifest}


@pytest.mark.parametrize("name,record", list(_records().items()))
def test_result_round_trip_and_golden_bytes(name: str, record: object) -> None:
    """Every result record round-trips and matches its canonical fixture."""

    decoder = type(record)
    expected = _fixture_bytes(name)
    assert decoder.from_mapping(record.to_dict()) == record
    assert record.encode() == expected
    assert decoder.decode(record.encode()) == record
    assert canonical_bytes(record.to_dict()) == expected


def test_manifest_attempt_v1_encoding_is_byte_identical() -> None:
    attempt = _records()["run-manifest"].attempts[0]  # type: ignore[union-attr]
    assert attempt.to_dict()["schema"] == ManifestAttempt.V1_SCHEMA
    assert attempt.encode() == OLD_V1_ATTEMPT_BYTES


def test_manifest_attempt_v2_requires_core_outcome_and_round_trips() -> None:
    attempt = _records()["run-manifest"].attempts[0]  # type: ignore[union-attr]
    fixture = Path(__file__).parents[2] / "tests" / "contracts" / "fixtures" / "core-run-evidence-v3-planned-yield.json"
    evidence = CoreRunEvidence.from_mapping(json.loads(fixture.read_text(encoding="utf-8")))
    core_outcome = CoreOutcomeRecord.from_evidence("task-a", attempt.attempt_id, evidence)
    versioned = ManifestAttempt(
        attempt.task_id,
        attempt.attempt_id,
        attempt.start_receipt,
        attempt.outcome_receipt,
        attempt.output_dir,
        core_outcome=core_outcome,
    )
    encoded = versioned.to_dict()
    assert encoded["schema"] == ManifestAttempt.V2_SCHEMA
    assert ManifestAttempt.from_mapping(encoded) == versioned

    v1_with_core = dict(encoded, schema=ManifestAttempt.V1_SCHEMA)
    with pytest.raises((ValueError, TypeError)):
        ManifestAttempt.from_mapping(v1_with_core)
    v2_without_core = dict(encoded)
    v2_without_core.pop("core_outcome")
    with pytest.raises((ValueError, TypeError)):
        ManifestAttempt.from_mapping(v2_without_core)


@pytest.mark.parametrize("record", list(_records().values()))
def test_result_records_are_frozen_and_sequences_are_tuples(record: object) -> None:
    """Result records reject mutation and manifests expose tuple attempts."""

    with pytest.raises((AttributeError, TypeError)):
        record.status = "failed"  # type: ignore[attr-defined]
    if isinstance(record, RunManifest):
        assert isinstance(record.attempts, tuple)
        with pytest.raises((AttributeError, TypeError)):
            record.attempts[0] = record.attempts[0]  # type: ignore[index]
    if isinstance(record, AllocationObservation):
        with pytest.raises((AttributeError, TypeError)):
            record.metadata["host"] = "other"


@pytest.mark.parametrize("record", list(_records().values()))
def test_result_decoders_reject_schema_key_and_type_mutants(record: object) -> None:
    """Result decoders reject unknown schema, missing/extra keys, and wrong types."""

    decoder = type(record)
    original = record.to_dict()
    cases = []
    unknown_schema = dict(original); unknown_schema["schema"] = "tpen-executor/unknown-v1"; cases.append(unknown_schema)
    missing = dict(original); missing.pop("schema"); cases.append(missing)
    extra = dict(original); extra["unexpected"] = True; cases.append(extra)
    wrong = dict(original); wrong[next(key for key in original if key != "schema")] = [] ; cases.append(wrong)
    for case in cases:
        with pytest.raises((ValueError, TypeError)):
            decoder.from_mapping(case)
