"""Public C1 successor-admission reader contract and mutation table."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tpen_executor import (
    AdmissionAbsent,
    AdmissionIdentityMismatch,
    AdmissionIntentMismatch,
    AdmissionUnreadable,
    read_successor_admission,
)
from tpen_executor._codec import canonical_bytes, digest

from test_continuation_admission import _planned_attempt


MUTANTS = (
    ("a", "successor attempt filter", "record.successor_attempt_id.value == attempt_id", "True", "test_reader_filters_successor_attempt"),
    ("b", "ambiguous admission", "if len(matches) > 1:", "if len(matches) > 0:", "test_reader_rejects_ambiguous"),
    ("c", "group digest cross-check", "path_digest != group_digest", "False", "test_reader_rejects_crossed_group_digest"),
    ("d", "decode failure classification", "raise AdmissionUnreadable", "raise AdmissionAbsent", "test_reader_rejects_corrupt_ledger"),
    ("e", "request digest check", "digest(request) != record.request_digest", "False", "test_reader_rejects_wrong_intent"),
    ("f", "successor id in rebuild", "record.successor_attempt_id", "record.predecessor_attempt_id", "test_reader_rebuilds_successor_request"),
)


def _admitted(tmp_path: Path):
    store, _group, fence = _planned_attempt(tmp_path)
    request = store.build_successor_admission_request(fence, type(fence.attempt_id)("successor-1"), "a" * 64, 10)
    record = store.record_successor_admission(fence, request, "b" * 64)
    return fence, request, record


def test_reader_returns_fresh_record_and_rebuilt_request(tmp_path: Path) -> None:
    fence, request, record = _admitted(tmp_path)
    result = read_successor_admission(
        fence.output_dir,
        task_id=fence.task_id,
        attempt_id="successor-1",
        group_digest=fence.group_digest,
        science_input_digest="a" * 64,
        target_iterations=10,
    )
    assert result["record"] == record.to_dict()
    assert result["request"] == request
    assert digest(result["request"]) == result["record"]["request_digest"]
    result["record"]["successor"]["attempt_id"] = "mutated"
    assert read_successor_admission(
        fence.output_dir, task_id=fence.task_id, attempt_id="successor-1",
        group_digest=fence.group_digest, science_input_digest="a" * 64,
        target_iterations=10,
    )["record"] == record.to_dict()


def test_reader_rejects_absent_and_crossed_identity(tmp_path: Path) -> None:
    fence, _request, _record = _admitted(tmp_path)
    with pytest.raises(AdmissionAbsent):
        read_successor_admission(
            fence.output_dir, task_id=fence.task_id, attempt_id="missing",
            group_digest=fence.group_digest, science_input_digest="a" * 64,
            target_iterations=10,
        )
    with pytest.raises(AdmissionIdentityMismatch):
        read_successor_admission(
            fence.output_dir, task_id=fence.task_id, attempt_id="successor-1",
            group_digest="c" * 64, science_input_digest="a" * 64,
            target_iterations=10,
        )


@pytest.mark.parametrize("science_input_digest,target_iterations", [("a" * 64, 11), ("c" * 64, 10)])
def test_reader_rejects_wrong_intent(tmp_path: Path, science_input_digest: str, target_iterations: int) -> None:
    fence, _request, _record = _admitted(tmp_path)
    with pytest.raises(AdmissionIntentMismatch):
        read_successor_admission(
            fence.output_dir, task_id=fence.task_id, attempt_id="successor-1",
            group_digest=fence.group_digest, science_input_digest=science_input_digest,
            target_iterations=target_iterations,
        )


def test_reader_rejects_noncanonical_predecessor_core_outcome(tmp_path: Path) -> None:
    fence, _request, _record = _admitted(tmp_path)
    path = Path(fence.output_dir, "core-outcome.json")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(AdmissionUnreadable):
        read_successor_admission(
            fence.output_dir, task_id=fence.task_id, attempt_id="successor-1",
            group_digest=fence.group_digest, science_input_digest="a" * 64,
            target_iterations=10,
        )


def test_reader_rejects_corrupt_ledger(tmp_path: Path) -> None:
    fence, _request, _record = _admitted(tmp_path)
    ledger = Path(fence.output_dir).parent / "successor-admissions" / fence.attempt_id.value / "successor-admission.json"
    ledger.write_bytes(b"{")
    with pytest.raises(AdmissionUnreadable):
        read_successor_admission(
            fence.output_dir, task_id=fence.task_id, attempt_id="successor-1",
            group_digest=fence.group_digest, science_input_digest="a" * 64,
            target_iterations=10,
        )
