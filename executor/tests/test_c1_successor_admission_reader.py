"""Public C1 successor-admission reader contract and mutation table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tpen_executor.storage as storage
from tpen_executor import (
    AdmissionAbsent, AdmissionAmbiguous, AdmissionIdentityMismatch,
    AdmissionIntentMismatch, AdmissionUnreadable, read_successor_admission,
)
from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.contracts import AttemptId
from tpen_executor.results import SuccessorAdmissionRecord

from test_continuation_admission import _planned_attempt


MUTANTS = (
    {"id": "a", "target": "executor/src/tpen_executor/storage.py", "old": "matches = [record for record in records if record.successor_attempt_id.value == attempt_id]", "new": "matches = list(records)", "killer": "test_mutant_a_successor_filter"},
    {"id": "b", "target": "executor/src/tpen_executor/storage.py", "old": "if len(matches) > 1:\n                raise AdmissionAmbiguous", "new": "if len(matches) > 0:\n                raise AdmissionAmbiguous", "killer": "test_mutant_b_ambiguous_is_rejected"},
    {"id": "c", "target": "executor/src/tpen_executor/storage.py", "old": "if path_digest != group_digest or path_task != task_id or os.path.basename(output) != attempt_id:", "new": "if False:", "killer": "test_mutant_c_group_digest_is_checked"},
    {"id": "d", "target": "executor/src/tpen_executor/storage.py", "old": "raise AdmissionUnreadable(\"successor admission ledger is unreadable\") from error", "new": "raise AdmissionAbsent(\"successor admission ledger is unreadable\") from error", "killer": "test_mutant_d_corrupt_ledger_is_unreadable"},
    {"id": "e", "target": "executor/src/tpen_executor/storage.py", "old": "if digest(request) != record.request_digest:", "new": "if False:", "killer": "test_mutant_e_intent_digest_is_checked"},
    {"id": "f", "target": "executor/src/tpen_executor/storage.py", "old": "core_record,\n                record.successor_attempt_id,\n                science_input_digest", "new": "core_record,\n                record.predecessor_attempt_id,\n                science_input_digest", "killer": "test_mutant_f_rebuild_uses_successor_id"},
)


def _snapshot(root: Path) -> dict[str, str]:
    """Capture relative entries; the pre-existing lock is part of the snapshot."""
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        result[name + "/" if path.is_dir() else name] = "<directory>" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _make_admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create an admission and successor fence through public store APIs."""
    store, group, allocation = __import__("test_review_p8_executor_r1", fromlist=["setup_store"]).setup_store(tmp_path)
    predecessor = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z")
    core_dir = Path(predecessor.output_dir) / "core"
    core_dir.mkdir()
    store.finish_attempt(predecessor, "failed", "2026-11-14T00:01:00Z")
    from test_core_evidence_v3 import _planned_evidence
    store.record_core_outcome(predecessor, _planned_evidence(core_dir))
    request = store.build_successor_admission_request(predecessor, AttemptId("successor-1"), "a" * 64, 10)
    record = store.record_successor_admission(predecessor, request, "b" * 64)
    monkeypatch.setattr(storage, "uuid4", lambda: SimpleNamespace(hex="successor-1"))
    successor = store.begin_attempt(group, "a", allocation, "2026-11-14T00:02:00Z")
    return store, group, predecessor, successor, request, record


def _read(successor, **extra):
    args = dict(output_dir=successor.output_dir, task_id=successor.task_id, attempt_id=successor.attempt_id.value, group_digest=successor.group_digest, science_input_digest="a" * 64, target_iterations=10)
    args.update(extra)
    return read_successor_admission(**args)


def test_reader_returns_fresh_record_and_rebuilt_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _predecessor, successor, request, record = _make_admitted(tmp_path, monkeypatch)
    result = _read(successor)
    assert result["record"] == record.to_dict()
    assert result["request"] == request
    assert digest(result["request"]) == result["record"]["request_digest"]
    result["record"]["successor"]["attempt_id"] = "mutated"
    assert _read(successor)["record"] == record.to_dict()


def test_reader_rejects_absent_and_wrong_task_or_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    with pytest.raises(AdmissionAbsent):
        _read(successor, attempt_id="missing")
    with pytest.raises(AdmissionIdentityMismatch):
        _read(successor, task_id="wrong-task")
    with pytest.raises(AdmissionIdentityMismatch):
        _read(successor, attempt_id="wrong-attempt")


def test_reader_rejects_crossed_group_and_outside_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    with pytest.raises(AdmissionIdentityMismatch):
        _read(successor, group_digest="c" * 64)
    outside = tmp_path / "store" / "not-groups" / successor.group_digest / successor.task_id / successor.attempt_id.value
    with pytest.raises(AdmissionIdentityMismatch):
        read_successor_admission(str(outside), task_id=successor.task_id, attempt_id=successor.attempt_id.value, group_digest=successor.group_digest, science_input_digest="a" * 64, target_iterations=10)


def test_reader_rejects_foreign_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    ledger = Path(successor.output_dir).parent / "successor-admissions" / successor.attempt_id.value / "successor-admission.json"
    payload = json.loads(ledger.read_text())
    payload["predecessor"]["task_id"] = "foreign-task"
    ledger.write_bytes(canonical_bytes(payload))
    with pytest.raises(AdmissionIdentityMismatch):
        _read(successor)


@pytest.mark.parametrize("science_input_digest,target_iterations", [("a" * 64, 11), ("c" * 64, 10)])
def test_reader_rejects_wrong_intent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science_input_digest: str, target_iterations: int) -> None:
    _store, _group, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    with pytest.raises(AdmissionIntentMismatch):
        _read(successor, science_input_digest=science_input_digest, target_iterations=target_iterations)


def test_reader_rejects_noncanonical_predecessor_core_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    path = Path(predecessor.output_dir, "core-outcome.json")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(AdmissionUnreadable):
        _read(successor)


def test_reader_rejects_truncated_and_noncanonical_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    ledger = Path(successor.output_dir).parent / "successor-admissions" / successor.attempt_id.value / "successor-admission.json"
    original = ledger.read_bytes()
    for payload in (b"{", original + b" "):
        ledger.write_bytes(payload)
        with pytest.raises(AdmissionUnreadable):
            _read(successor)


def test_reader_rejects_ambiguous_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _predecessor, successor, _request, record = _make_admitted(tmp_path, monkeypatch)
    # Public APIs reject predecessor forks, so this minimal fixture covers ambiguity reachability.
    second = Path(successor.output_dir).parent / "successor-admissions" / "other-predecessor"
    second.mkdir()
    duplicate = SuccessorAdmissionRecord("a", AttemptId("other-predecessor"), record.predecessor_content_id, "a", record.successor_attempt_id, record.request_digest, record.core_acceptance_digest)
    (second / "successor-admission.json").write_bytes(duplicate.encode())
    with pytest.raises(AdmissionAmbiguous):
        _read(successor)


def test_reader_snapshot_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    task_dir = Path(successor.output_dir).parent
    before = _snapshot(task_dir)
    _read(successor)
    assert _snapshot(task_dir) == before
    error_calls = (
        {"task_id": "wrong-task"},
        {"attempt_id": "wrong-attempt"},
        {"group_digest": "c" * 64},
        {"science_input_digest": "c" * 64},
    )
    for call in error_calls:
        with pytest.raises((AdmissionIdentityMismatch, AdmissionIntentMismatch)):
            _read(successor, **call)
        assert _snapshot(task_dir) == before


def test_mutant_a_successor_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_reader_rejects_absent_and_wrong_task_or_attempt(tmp_path, monkeypatch)


def test_mutant_b_ambiguous_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_reader_rejects_ambiguous_fixture(tmp_path, monkeypatch)


def test_mutant_c_group_digest_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_reader_rejects_crossed_group_and_outside_layout(tmp_path, monkeypatch)


def test_mutant_d_corrupt_ledger_is_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_reader_rejects_truncated_and_noncanonical_ledger(tmp_path, monkeypatch)


def test_mutant_e_intent_digest_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_reader_rejects_wrong_intent(tmp_path, monkeypatch, "c" * 64, 10)


def test_mutant_f_rebuild_uses_successor_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_reader_returns_fresh_record_and_rebuilt_request(tmp_path, monkeypatch)
