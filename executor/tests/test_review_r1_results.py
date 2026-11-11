"""Independent R1 result and manifest tests for the executor."""

from __future__ import annotations

import operator

import pytest

from tpen_executor._codec import canonical_bytes, decode_json, digest
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId
from tpen_executor.results import (
    AllocationObservation,
    ManifestAttempt,
    OutcomeReceipt,
    RunManifest,
    StartReceipt,
    decode_manifest,
    encode_manifest,
)

GROUP_DIGEST = "76a21ab64e2533a35421b67a8c16c832967fcc6cadf730c9fdfd505840389ebf"


def _records() -> dict[str, object]:
    attempt = AttemptId("attempt-1")
    allocation = AllocationId("allocation-1")
    job = BackendJobId("job-1")
    start = StartReceipt("task-a", attempt, allocation, job, GROUP_DIGEST, "2026-11-04T00:00:00Z", "/runs/group-1/task-a/attempt-1")
    outcome = OutcomeReceipt("task-a", attempt, GROUP_DIGEST, "2026-11-04T00:01:00Z", "completed", start.output_dir, digest({"energy": -2.0}))
    observation = AllocationObservation(allocation, "2026-11-04T00:00:01Z", "running", {"host": "node-1", "slots": 2})
    member = ManifestAttempt("task-a", attempt, start, outcome, start.output_dir)
    manifest = RunManifest("group-1", "76a21ab64e2533a35421b67a8c16c832967fcc6cadf730c9fdfd505840389ebf", (member,))
    return {"start": start, "outcome": outcome, "observation": observation, "member": member, "manifest": manifest}


def test_manifest_attempt_requires_start_receipt_type() -> None:
    """R1-R1 (F1) — expected RED at 777700d6: ManifestAttempt requires a StartReceipt object."""

    records = _records()
    attempt = records["start"].attempt_id  # type: ignore[union-attr]
    outcome = records["outcome"]
    with pytest.raises(TypeError):
        ManifestAttempt("task-a", attempt, outcome, None, "/out")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ManifestAttempt("task-a", attempt, {"attempt_id": attempt}, None, "/out")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attempt_id": AttemptId("attempt-9")},
        {"task_id": "task-z"},
        {"outcome_receipt": OutcomeReceipt("task-a", AttemptId("attempt-9"), GROUP_DIGEST, "2026-11-04T00:01:00Z", "completed", "/runs/group-1/task-a/attempt-1", digest({"energy": -2.0}))},
        {"outcome_receipt": OutcomeReceipt("task-z", AttemptId("attempt-1"), GROUP_DIGEST, "2026-11-04T00:01:00Z", "completed", "/runs/group-1/task-a/attempt-1", digest({"energy": -2.0}))},
        {"output_dir": "/elsewhere"},
    ],
)
def test_manifest_attempt_rejects_inconsistent_receipts(kwargs: dict[str, object]) -> None:
    """R1-R2 (F2) — expected RED at 777700d6: ManifestAttempt cross-checks receipt identities and paths."""

    records = _records()
    start = records["start"]
    outcome = records["outcome"]
    attempt = records["start"].attempt_id  # type: ignore[union-attr]
    values = {"task_id": "task-a", "attempt_id": attempt, "start_receipt": start, "outcome_receipt": outcome, "output_dir": start.output_dir}  # type: ignore[union-attr]
    values.update(kwargs)
    with pytest.raises(ValueError):
        ManifestAttempt(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("task_id", "attempt_id", "output_dir"),
    [
        ("task-z", AttemptId("attempt-1"), "/runs/group-1/task-a/attempt-1"),
        ("task-a", AttemptId("attempt-9"), "/runs/group-1/task-a/attempt-1"),
        ("task-a", AttemptId("attempt-1"), "/elsewhere"),
    ],
)
def test_manifest_attempt_rejects_each_start_receipt_mismatch_without_outcome(
    task_id: str, attempt_id: AttemptId, output_dir: str
) -> None:
    """F2: each redundant field is checked even when no outcome can mask it."""

    start = _records()["start"]
    with pytest.raises(ValueError):
        ManifestAttempt(task_id, attempt_id, start, None, output_dir)  # type: ignore[arg-type]


def test_run_manifest_rejects_duplicate_attempt_ids() -> None:
    """R1-R3 (F2) — expected RED at 777700d6: RunManifest rejects duplicate attempt identifiers."""

    records = _records()
    member = records["member"]
    with pytest.raises(ValueError):
        RunManifest("group-1", digest({"group": 1}), (member, member))


def test_allocation_observation_metadata_rejects_inplace_union() -> None:
    """R1-R4 (F4) — expected RED at 777700d6: frozen observation metadata rejects in-place union."""

    observation = _records()["observation"]
    with pytest.raises(TypeError):
        operator.ior(observation.metadata, {"injected": True})  # type: ignore[union-attr]
    assert "injected" not in observation.metadata  # type: ignore[union-attr]


def test_outcome_receipt_without_result_digest_round_trips() -> None:
    """R1-R5 — expected GREEN: an absent result digest is represented by explicit JSON null."""

    records = _records()
    start = records["start"]
    receipt = OutcomeReceipt("task-a", start.attempt_id, GROUP_DIGEST, "2026-11-04T00:01:00Z", "completed", start.output_dir, None)  # type: ignore[union-attr]
    mapping = receipt.to_dict()
    assert mapping["result_digest"] is None
    assert b'"result_digest":null' in canonical_bytes(mapping)
    assert OutcomeReceipt.from_mapping(mapping) == receipt
    missing = dict(mapping)
    missing.pop("result_digest")
    with pytest.raises(ValueError):
        OutcomeReceipt.from_mapping(missing)


def test_manifest_expresses_in_flight_and_retried_attempts() -> None:
    """R1-R6 — expected GREEN: manifests preserve in-flight, retried, and empty attempt sets."""

    records = _records()
    start = records["start"]
    first = ManifestAttempt("task-a", AttemptId("attempt-1"), start, None, start.output_dir)
    second_start = StartReceipt("task-a", AttemptId("attempt-2"), AllocationId("allocation-2"), BackendJobId("job-2"), GROUP_DIGEST, "2026-11-04T00:02:00Z", "/runs/a2")
    second_outcome = OutcomeReceipt("task-a", AttemptId("attempt-2"), GROUP_DIGEST, "2026-11-04T00:02:30Z", "completed", "/runs/a2", digest({"energy": -2.0}))
    second = ManifestAttempt("task-a", AttemptId("attempt-2"), second_start, second_outcome, "/runs/a2")
    third_start = StartReceipt("task-b", AttemptId("attempt-3"), AllocationId("allocation-3"), BackendJobId("job-3"), GROUP_DIGEST, "2026-11-04T00:03:00Z", "/runs/b1")
    third = ManifestAttempt("task-b", AttemptId("attempt-3"), third_start, None, third_start.output_dir)
    manifest = RunManifest("group-1", GROUP_DIGEST, (first, second, third))
    assert RunManifest.decode(manifest.encode()) == manifest
    assert decode_manifest(encode_manifest(manifest)) == manifest
    assert [attempt.task_id for attempt in manifest.attempts] == ["task-a", "task-a", "task-b"]
    assert b'"outcome_receipt":null' in manifest.encode()
    empty = RunManifest("g", digest({"group": "g"}), ())
    assert RunManifest.decode(empty.encode()).attempts == ()


def test_manifest_decode_rejects_non_object_and_wrong_types() -> None:
    """R1-R7 — expected GREEN: manifest encoders and decoders reject wrong record shapes."""

    with pytest.raises(ValueError):
        decode_manifest(b"[]")
    with pytest.raises(TypeError):
        encode_manifest(object())  # type: ignore[arg-type]
    records = _records()
    manifest_data = records["manifest"].to_dict()  # type: ignore[union-attr]
    manifest_data["attempts"] = "x"
    with pytest.raises(TypeError):
        RunManifest.from_mapping(manifest_data)
    member_data = records["member"].to_dict()  # type: ignore[union-attr]
    member_data["outcome_receipt"] = 5
    with pytest.raises(TypeError):
        ManifestAttempt.from_mapping(member_data)
    member_data["outcome_receipt"] = None
    member_data["start_receipt"] = None
    with pytest.raises(TypeError):
        ManifestAttempt.from_mapping(member_data)


def test_receipt_decoders_reject_wrong_scalar_types() -> None:
    """R1-R8 — expected GREEN: receipt decoders enforce scalar identity, digest, state, and metadata types."""

    records = _records()
    start_data = records["start"].to_dict()  # type: ignore[union-attr]
    start_data["attempt_id"] = 7
    with pytest.raises(TypeError):
        StartReceipt.from_mapping(start_data)
    start_data["attempt_id"] = ""
    with pytest.raises(TypeError):
        StartReceipt.from_mapping(start_data)
    outcome_data = records["outcome"].to_dict()  # type: ignore[union-attr]
    outcome_data["result_digest"] = 7
    with pytest.raises(TypeError):
        OutcomeReceipt.from_mapping(outcome_data)
    outcome_data["result_digest"] = "zz"
    with pytest.raises(ValueError):
        OutcomeReceipt.from_mapping(outcome_data)
    observation_data = records["observation"].to_dict()  # type: ignore[union-attr]
    observation_data["metadata"] = []
    with pytest.raises(TypeError):
        AllocationObservation.from_mapping(observation_data)
    observation_data["metadata"] = {"host": "node-1"}
    observation_data["state"] = ""
    with pytest.raises(TypeError):
        AllocationObservation.from_mapping(observation_data)


def test_allocation_observation_nested_metadata_is_frozen_and_round_trips() -> None:
    """R1-R9 — expected GREEN: nested allocation metadata is deeply frozen and canonical on the wire."""

    metadata = {"gpus": [0, 1], "nested": {"a": [{"b": 1}]}}
    observation = AllocationObservation(AllocationId("allocation-1"), "2026-11-04T00:00:01Z", "running", metadata)
    assert observation.metadata["gpus"] == (0, 1)
    with pytest.raises(TypeError):
        observation.metadata["nested"]["a"][0]["b"] = 2
    assert AllocationObservation.from_mapping(observation.to_dict()) == observation
    expected = b'{"allocation_id":"allocation-1","metadata":{"gpus":[0,1],"nested":{"a":[{"b":1}]}},"observed_at_utc":"2026-11-04T00:00:01Z","schema":"tpen-executor/allocation-observation-v1","state":"running"}'
    assert canonical_bytes(observation.to_dict()) == expected


def test_identity_wire_forms_differ_between_receipt_and_standalone() -> None:
    """R1-R10 — expected GREEN: receipts use bare identity strings while standalone identities use envelopes."""

    records = _records()
    start = records["start"]
    assert start.to_dict()["attempt_id"] == "attempt-1"
    assert AttemptId("attempt-1").to_dict() == {"schema": "tpen-executor/attempt-id-v1", "value": "attempt-1"}
    mapping = start.to_dict()
    mapping["attempt_id"] = AttemptId("attempt-1").to_dict()
    with pytest.raises(TypeError):
        StartReceipt.from_mapping(mapping)


def test_manifest_rejects_receipt_group_digest_mismatch() -> None:
    """Orchestrator regression: receipt group identity must match its manifest."""

    records = _records()
    member = records["member"]
    with pytest.raises(ValueError):
        RunManifest("group-1", digest({"different": True}), (member,))


def test_decode_rejects_duplicate_json_keys() -> None:
    """Orchestrator regression: duplicate JSON members are not last-write-wins."""

    raw = b'{"schema":"tpen-executor/start-receipt-v1","schema":"duplicate"}'
    with pytest.raises(ValueError):
        StartReceipt.decode(raw)
    with pytest.raises(ValueError):
        decode_json(raw)
