"""CX-04b executor request bridge and successor-admission ledger probes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen_executor.contracts import AttemptId
from tpen_executor._codec import digest
from tpen_executor.storage import AdmissionFork, FenceRejected

from test_review_p8_executor_r1 import setup_store
from test_core_evidence_v3 import _planned_evidence, _typed_evidence
from tpen_executor.results import CoreOutcomeRecord


def _planned_attempt(tmp_path: Path):
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.finish_attempt(fence, "failed", "2026-11-14T00:01:00Z")
    store.record_core_outcome(fence, _planned_evidence(core_dir))
    return store, group, fence


def test_build_and_record_successor_admission_uses_canonical_core_record(tmp_path: Path) -> None:
    store, group, fence = _planned_attempt(tmp_path)
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
    assert store.record_successor_admission(fence, request, "c" * 64) == first

    fork_request = store.build_successor_admission_request(fence, AttemptId("successor-2"), "a" * 64, 10)
    with pytest.raises(AdmissionFork, match="fork"):
        store.record_successor_admission(fence, fork_request, "d" * 64)


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

