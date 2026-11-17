"""Executor-side v3 storage and manifest boundary tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpen_executor.runners import CoreRunEvidence, EvidenceUnavailable
from tpen_executor.dispatch import LocalDispatcher
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.storage import FenceRejected
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.workers import ProcessOutcome

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


def _typed_evidence(core_dir: Path, outcome: str) -> CoreRunEvidence:
    core_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(PLANNED_FIXTURE.read_text(encoding="utf-8"))
    checkpoint_dir = str(core_dir / "checkpoints" / "step_000002")
    payload["run_dir"] = str(core_dir)
    payload["checkpoint"]["checkpoint_dir"] = checkpoint_dir
    payload["checkpoint_ref"]["checkpoint_dir"] = checkpoint_dir
    payload["outcome"] = outcome
    payload["run_status"] = outcome
    payload["verdict"] = "completed" if outcome == "completed" else "incomplete"
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


@pytest.mark.parametrize("outcome,returncode,expected_segment,expected_eligibility", [
    ("planned_yield", 1, False, True),
    ("completed", 0, True, False),
])
def test_local_dispatch_records_typed_outcome_before_segment_selection(
    tmp_path: Path, outcome: str, returncode: int, expected_segment: bool, expected_eligibility: bool,
) -> None:
    store, group, allocation = setup_store(tmp_path)

    class Process:
        def run(self, environment, args, **kwargs):
            return ProcessOutcome(tuple(args), returncode, kwargs["stdout_path"], kwargs["stderr_path"], False)

    class Evidence:
        def read(self, *args, **kwargs):
            return _typed_evidence(Path(args[2]), outcome)

    worker = SimpleNamespace(materialize=lambda env: {
        "environment": object(), "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    manifest = LocalDispatcher(worker).run(
        group,
        {"a": ConfiguredRun("a", str(tmp_path / "a.yaml"))},
        AttachedAllocation(allocation, store),
    )
    attempt = manifest.attempts[0]
    assert attempt.core_outcome is not None
    assert attempt.core_outcome.outcome == outcome
    assert attempt.core_outcome.continuation_eligible is expected_eligibility
    assert attempt.to_dict()["schema"] == "tpen-executor/manifest-attempt-v2"
    assert bool(manifest.segments) is expected_segment


def test_record_core_outcome_is_write_once_and_fenced(tmp_path: Path) -> None:
    store, group, allocation = setup_store(tmp_path)
    fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:00:00Z")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    store.finish_attempt(fence, "failed", "2026-11-14T00:01:00Z")
    evidence = _planned_evidence(core_dir)
    store.record_core_outcome(fence, evidence)
    with pytest.raises(FenceRejected, match="already has a core outcome"):
        store.record_core_outcome(fence, evidence)

    foreign_fence = store.begin_attempt(group, "a", allocation, "2026-11-14T00:02:00Z")
    Path(foreign_fence.output_dir, "core").mkdir()
    store.finish_attempt(foreign_fence, "failed", "2026-11-14T00:03:00Z")
    with pytest.raises(FenceRejected, match="core evidence"):
        store.record_core_outcome(foreign_fence, evidence)
