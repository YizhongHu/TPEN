"""Conformance tests for immutable executor receipts and manifests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId
from tpen_executor.results import (
    AllocationObservation,
    ManifestAttempt,
    OutcomeReceipt,
    RunManifest,
    StartReceipt,
)


FIXTURES = Path(__file__).parent / "fixtures"
GROUP_DIGEST = "76a21ab64e2533a35421b67a8c16c832967fcc6cadf730c9fdfd505840389ebf"


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
