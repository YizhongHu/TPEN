"""Public C1 successor-admission reader contract and mutation table."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os
import stat
import threading
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
from tpen_executor.results import RemoteTeardownReceipt, SuccessorAdmissionRecord
from tpen_executor.runners import EvidenceUnavailable

from test_continuation_admission import _planned_attempt


MUTANTS = (
    {"id": "a", "target": "executor/src/tpen_executor/storage.py", "old": "matches = [record for record in records if record.successor_attempt_id.value == attempt_id]", "new": "matches = list(records)", "killer": "test_mutant_a_successor_filter"},
    {"id": "b", "target": "executor/src/tpen_executor/storage.py", "old": "if len(matches) > 1:\n                raise AdmissionAmbiguous", "new": "if False:\n                raise AdmissionAmbiguous", "killer": "test_mutant_b_ambiguous_is_rejected"},
    {"id": "c", "target": "executor/src/tpen_executor/storage.py", "old": "if path_digest != group_digest or path_task != task_id or os.path.basename(output) != attempt_id:", "new": "if False:", "killer": "test_mutant_c_group_digest_is_checked"},
    {"id": "d", "target": "executor/src/tpen_executor/storage.py", "old": "raise AdmissionUnreadable(\"successor admission ledger is unreadable\") from error", "new": "raise AdmissionAbsent(\"successor admission ledger is unreadable\") from error", "killer": "test_mutant_d_corrupt_ledger_is_unreadable"},
    {"id": "e", "target": "executor/src/tpen_executor/storage.py", "old": "if digest(request) != record.request_digest:", "new": "if False:", "killer": "test_mutant_e_intent_digest_is_checked"},
    {"id": "f", "target": "executor/src/tpen_executor/storage.py", "old": "core_record,\n                record.successor_attempt_id,\n                science_input_digest", "new": "core_record,\n                record.predecessor_attempt_id,\n                science_input_digest", "killer": "test_mutant_f_rebuild_uses_successor_id"},
    {"id": "g", "target": "executor/src/tpen_executor/storage.py", "old": "except EvidenceUnavailable as error:\n                raise AdmissionUnreadable(\"predecessor core outcome is unreadable\") from error", "new": "except EvidenceUnavailable as error:\n                raise", "killer": "test_reader_translates_evidence_unavailable"},
    {"id": "h", "target": "executor/src/tpen_executor/storage.py", "old": "except FenceRejected as error:\n        raise AdmissionUnreadable(\"successor admission is unreadable\") from error", "new": "except FenceRejected as error:\n        raise", "killer": "test_reader_translates_bad_task_lock"},
    {"id": "i", "target": "executor/src/tpen_executor/storage.py", "old": "if start.transport_kind != \"local\":", "new": "if False:", "killer": "test_reader_rejects_remote_admission_without_teardown"},
    {"id": "j", "target": "executor/src/tpen_executor/storage.py", "old": "        with _task_lock(os.path.join(task_dir, _TASK_LOCK)):\n            try:\n                records = store._admission_records_for_task(task_dir, task_id)", "new": "        if True:\n            try:\n                records = store._admission_records_for_task(task_dir, task_id)", "killer": "test_reader_waits_for_task_writer"},
)


def _snapshot(root: Path) -> dict[str, str]:
    """Capture relative entries; the pre-existing lock is part of the snapshot."""
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        result[name + "/" if path.is_dir() else name] = "<directory>" if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _entry_inventory(root: Path) -> dict[str, tuple[int, int]]:
    """Inventory entries without opening regular files or special files."""
    return {
        str(path.relative_to(root)): (stat.S_IFMT(path.lstat().st_mode), path.lstat().st_size)
        for path in sorted(root.rglob("*"))
    }


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
    return store, group, allocation, predecessor, successor, request, record


def _read(successor, **extra):
    args = _read_arguments(successor)
    args.update(extra)
    return read_successor_admission(**args)


def _read_arguments(successor) -> dict[str, object]:
    return dict(
        output_dir=successor.output_dir,
        task_id=successor.task_id,
        attempt_id=successor.attempt_id.value,
        group_digest=successor.group_digest,
        science_input_digest="a" * 64,
        target_iterations=10,
    )


def _make_remote_admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create a remote predecessor with a real teardown proof and admission."""
    store, group, allocation = __import__("test_review_p8_executor_r1", fromlist=["setup_store"]).setup_store(tmp_path)
    predecessor = store.begin_attempt(
        group, "a", allocation, "2026-11-14T00:00:00Z", transport_kind="remote"
    )
    core_dir = Path(predecessor.output_dir) / "core"
    core_dir.mkdir()
    store.finish_remote_attempt(
        predecessor,
        {
            "schema": "tpen-executor/leaf-teardown-v1",
            "group_digest": predecessor.group_digest,
            "task_id": predecessor.task_id,
            "attempt_id": predecessor.attempt_id.value,
            "allocation_id": predecessor.allocation_id,
            "backend_job_id": allocation.backend_job_id.value,
            "child_group_reaped": True,
            "inspection": "complete",
        },
        "c" * 64,
        "failed",
        "2026-11-14T00:01:00Z",
    )
    store.record_core_outcome(predecessor, __import__("test_core_evidence_v3", fromlist=["_planned_evidence"])._planned_evidence(core_dir))
    receipt_path = Path(predecessor.output_dir) / "remote-teardown.json"
    receipt = RemoteTeardownReceipt.decode(receipt_path.read_bytes())
    assert receipt_path.read_bytes() == receipt.encode()
    assert receipt.child_group_reaped is True and receipt.inspection == "complete"
    request = store.build_successor_admission_request(
        predecessor, AttemptId("successor-1"), "a" * 64, 10
    )
    record = store.record_successor_admission(predecessor, request, "b" * 64)
    monkeypatch.setattr(storage, "uuid4", lambda: SimpleNamespace(hex="successor-1"))
    successor = store.begin_attempt(group, "a", allocation, "2026-11-14T00:02:00Z")
    return store, group, allocation, predecessor, successor, request, record


def _read_successor_in_child(arguments, ready, returned, results) -> None:
    ready.set()
    try:
        results.put(("ok", read_successor_admission(**arguments)))
    except BaseException as error:  # pragma: no cover - reported to parent
        results.put(("error", type(error).__name__, str(error)))
    finally:
        returned.set()


def _ledger_path(predecessor) -> Path:
    """Return the ledger path keyed by the predecessor attempt identity."""
    return Path(predecessor.output_dir).parent / "successor-admissions" / predecessor.attempt_id.value / "successor-admission.json"


def test_reader_returns_fresh_record_and_rebuilt_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _allocation, _predecessor, successor, request, record = _make_admitted(tmp_path, monkeypatch)
    result = _read(successor)
    assert result["record"] == record.to_dict()
    assert result["request"] == request
    assert digest(result["request"]) == result["record"]["request_digest"]
    result["record"]["successor"]["attempt_id"] = "mutated"
    assert _read(successor)["record"] == record.to_dict()


def test_reader_rejects_absent_and_wrong_task_or_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, group, allocation, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    # A second public begin_attempt has no admission record; its own identity reaches AdmissionAbsent.
    monkeypatch.setattr(storage, "uuid4", lambda: SimpleNamespace(hex="unadmitted-1"))
    absent = store.begin_attempt(group, "a", allocation, "2026-11-14T00:03:00Z")
    with pytest.raises(AdmissionAbsent):
        _read(absent)
    with pytest.raises(AdmissionIdentityMismatch):
        _read(successor, task_id="wrong-task")
    with pytest.raises(AdmissionIdentityMismatch):
        _read(successor, attempt_id="wrong-attempt")


def test_reader_rejects_crossed_group_and_outside_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _allocation, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    with pytest.raises(AdmissionIdentityMismatch):
        _read(successor, group_digest="c" * 64)
    outside = tmp_path / "store" / "not-groups" / successor.group_digest / successor.task_id / successor.attempt_id.value
    with pytest.raises(AdmissionIdentityMismatch):
        read_successor_admission(str(outside), task_id=successor.task_id, attempt_id=successor.attempt_id.value, group_digest=successor.group_digest, science_input_digest="a" * 64, target_iterations=10)


def test_reader_rejects_foreign_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _allocation, predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    ledger = _ledger_path(predecessor)
    payload = json.loads(ledger.read_text())
    payload["predecessor"]["task_id"] = "foreign-task"
    ledger.write_bytes(canonical_bytes(payload))
    with pytest.raises(AdmissionIdentityMismatch):
        _read(successor)


@pytest.mark.parametrize("science_input_digest,target_iterations", [("a" * 64, 11), ("c" * 64, 10)])
def test_reader_rejects_wrong_intent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science_input_digest: str, target_iterations: int) -> None:
    _store, _group, _allocation, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    with pytest.raises(AdmissionIntentMismatch):
        _read(successor, science_input_digest=science_input_digest, target_iterations=target_iterations)


def test_reader_rejects_noncanonical_predecessor_core_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _allocation, predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    path = Path(predecessor.output_dir, "core-outcome.json")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(AdmissionUnreadable):
        _read(successor)


def test_reader_translates_evidence_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _allocation, predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    _read(successor)
    path = Path(predecessor.output_dir, "core-outcome.json")
    payload = json.loads(path.read_text())
    payload["evidence"]["run_status"] = "completed"
    path.write_bytes(canonical_bytes(payload))
    after_corruption = _snapshot(Path(successor.output_dir).parent)
    with pytest.raises(AdmissionUnreadable) as caught:
        _read(successor)
    cause = caught.value.__cause__
    causes = []
    while cause is not None:
        causes.append(cause)
        cause = cause.__cause__
    assert any(isinstance(error, EvidenceUnavailable) for error in causes)
    assert _snapshot(Path(successor.output_dir).parent) == after_corruption


def test_reader_translates_bad_task_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _allocation, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    _read(successor)
    task_dir = Path(successor.output_dir).parent
    lock = task_dir / ".task.lock"
    lock.unlink()
    os.mkfifo(lock)
    before = _entry_inventory(task_dir)
    with pytest.raises(AdmissionUnreadable) as caught:
        _read(successor)
    assert isinstance(caught.value.__cause__, storage.FenceRejected)
    assert _entry_inventory(task_dir) == before


def test_reader_rejects_remote_admission_without_teardown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _allocation, predecessor, successor, request, record = _make_remote_admitted(tmp_path, monkeypatch)
    result = _read(successor)
    assert result["record"] == record.to_dict()
    assert result["request"] == request
    assert digest(result["request"]) == result["record"]["request_digest"]
    Path(predecessor.output_dir, "remote-teardown.json").unlink()
    with pytest.raises(AdmissionUnreadable):
        _read(successor)


def test_reader_waits_for_task_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _group, _allocation, predecessor, successor, _request, _record = _make_remote_admitted(tmp_path, monkeypatch)
    task_dir = Path(successor.output_dir).parent
    ledger = _ledger_path(predecessor)
    before = ledger.read_bytes()
    lock_acquired = threading.Event()
    release = threading.Event()
    original_lock = storage._task_lock
    lock_path = task_dir / ".task.lock"

    @contextmanager
    def hold_writer_lock(path: str):
        with original_lock(path):
            if Path(path) == lock_path:
                lock_acquired.set()
                assert release.wait(timeout=5)
            yield

    monkeypatch.setattr(storage, "_task_lock", hold_writer_lock)
    writer_error: list[BaseException] = []

    def public_writer() -> None:
        try:
            # This public path takes the task lock and then fails read-only
            # because the fresh successor has no predecessor outcome.
            store.build_successor_admission_request(
                successor, AttemptId("lock-probe"), "a" * 64, 10
            )
        except BaseException as error:  # expected after the lock is released
            writer_error.append(error)

    writer = threading.Thread(target=public_writer)
    writer.start()
    assert lock_acquired.wait(timeout=5)

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    returned = context.Event()
    results = context.Queue()
    reader = context.Process(
        target=_read_successor_in_child,
        args=(_read_arguments(successor), ready, returned, results),
    )
    reader.start()
    assert ready.wait(timeout=5)
    assert not returned.wait(timeout=0.5)

    release.set()
    writer.join(timeout=5)
    reader.join(timeout=5)
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert writer_error and isinstance(writer_error[0], storage.FenceRejected)
    result = results.get(timeout=5)
    assert result[0] == "ok"
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("case", ["relative", "trailing", "symlink", "predecessor"])
def test_reader_path_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    _store, _group, _allocation, predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    if case == "relative":
        with pytest.raises(AdmissionIdentityMismatch):
            read_successor_admission(os.path.relpath(successor.output_dir), task_id=successor.task_id, attempt_id=successor.attempt_id.value, group_digest=successor.group_digest, science_input_digest="a" * 64, target_iterations=10)
    elif case == "trailing":
        assert _read(successor, output_dir=successor.output_dir + "/")
    elif case == "predecessor":
        with pytest.raises(AdmissionIdentityMismatch):
            read_successor_admission(predecessor.output_dir, task_id=successor.task_id, attempt_id=successor.attempt_id.value, group_digest=successor.group_digest, science_input_digest="a" * 64, target_iterations=10)
    else:
        group_dir = Path(successor.output_dir).parents[1]
        moved = group_dir.with_name(group_dir.name + "-real")
        group_dir.rename(moved)
        os.symlink(moved, group_dir)
        try:
            with pytest.raises(AdmissionUnreadable):
                _read(successor)
        finally:
            group_dir.unlink()
            moved.rename(group_dir)


def test_reader_rejects_truncated_and_noncanonical_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _allocation, predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
    ledger = _ledger_path(predecessor)
    original = ledger.read_bytes()
    for payload in (b"{", original + b" "):
        ledger.write_bytes(payload)
        with pytest.raises(AdmissionUnreadable):
            _read(successor)


def test_reader_rejects_ambiguous_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _allocation, predecessor, successor, _request, record = _make_admitted(tmp_path, monkeypatch)
    # Public APIs reject predecessor forks, so this minimal fixture covers ambiguity reachability.
    second = _ledger_path(predecessor).parent.parent / "other-predecessor"
    second.mkdir()
    duplicate = SuccessorAdmissionRecord("a", AttemptId("other-predecessor"), record.predecessor_content_id, "a", record.successor_attempt_id, record.request_digest, record.core_acceptance_digest)
    (second / "successor-admission.json").write_bytes(duplicate.encode())
    with pytest.raises(AdmissionAmbiguous):
        _read(successor)


def test_reader_snapshot_is_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, _group, _allocation, _predecessor, successor, _request, _record = _make_admitted(tmp_path, monkeypatch)
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


def test_mutant_g_evidence_error_is_translated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_reader_translates_evidence_unavailable(tmp_path, monkeypatch)


def test_mutant_h_lock_error_is_translated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_reader_translates_bad_task_lock(tmp_path, monkeypatch)
