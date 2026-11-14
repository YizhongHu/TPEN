"""Executor-side v3 storage and manifest boundary tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen_executor.runners import CoreRunEvidence, EvidenceUnavailable
from tpen_executor.storage import FenceRejected

from test_review_p8_executor_r1 import setup_store


REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNED_FIXTURE = REPO_ROOT / "tests" / "contracts" / "fixtures" / "core-run-evidence-v3-planned-yield.json"


def _planned_evidence(core_dir: Path) -> CoreRunEvidence:
    payload = json.loads(PLANNED_FIXTURE.read_text(encoding="utf-8"))
    checkpoint_dir = str(core_dir / "checkpoints" / "step_000002")
    payload["run_dir"] = str(core_dir)
    payload["checkpoint"]["checkpoint_dir"] = checkpoint_dir
    payload["checkpoint_ref"]["checkpoint_dir"] = checkpoint_dir
    return CoreRunEvidence.from_mapping(payload)


def test_storage_refuses_planned_yield_as_completed_segment(tmp_path: Path) -> None:
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.finish_attempt(fence, "completed", "2026-11-14T00:01:00Z")

    with pytest.raises(FenceRejected, match="non-completed evidence"):
        store.accept_segment(fence, _planned_evidence(core_dir))
    assert not (Path(fence.output_dir) / "segment.json").exists()


def test_storage_records_typed_planned_yield_without_segment(tmp_path: Path) -> None:
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.finish_attempt(fence, "failed", "2026-11-14T00:01:00Z")

    record = store.record_core_outcome(fence, _planned_evidence(core_dir))
    manifest = store.manifest(group)
    assert record.outcome == "planned_yield"
    assert record.continuation_eligible is True
    assert manifest.segments == ()
    assert manifest.attempts[0].core_outcome == record


def test_v3_decoder_refuses_status_outcome_mismatch() -> None:
    payload = json.loads(PLANNED_FIXTURE.read_text(encoding="utf-8"))
    payload["run_status"] = "completed"
    with pytest.raises(EvidenceUnavailable, match="run_status must equal outcome"):
        CoreRunEvidence.from_mapping(payload)
