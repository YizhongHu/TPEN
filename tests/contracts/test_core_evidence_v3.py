"""Round trips and refusal rules for the core-owned run-evidence v3 wire."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen.artifacts import RunResult
from tpen.inspection import RUN_EVIDENCE_SCHEMA_V3, RunEvidence, inspect_run

from ._regen_fixtures import (
    PLANNED_YIELD_FIXTURE_PATH,
    UNKNOWN_FIXTURE_PATH,
    _make_run,
    normalize,
    regenerate_planned_yield,
    regenerate_unknown,
)


def test_unknown_fixture_regenerates_from_typed_result_producer() -> None:
    if not UNKNOWN_FIXTURE_PATH.exists():
        pytest.skip("Cannon-generated v3 fixtures are not present in this environment")
    fixture = json.loads(UNKNOWN_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert regenerate_unknown() == fixture
    assert RunEvidence.from_mapping(fixture).continuation_eligible is False


def test_planned_yield_fixture_regenerates_from_real_s4_entry() -> None:
    if not PLANNED_YIELD_FIXTURE_PATH.exists():
        pytest.skip("Cannon-generated v3 fixtures are not present in this environment")
    fixture = json.loads(PLANNED_YIELD_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert regenerate_planned_yield() == fixture
    evidence = RunEvidence.from_mapping(fixture)
    assert evidence.schema == RUN_EVIDENCE_SCHEMA_V3
    assert evidence.outcome == "planned_yield"
    assert evidence.continuation_eligible is True


def test_v3_round_trip_preserves_full_checkpoint_binding() -> None:
    # The fixture generator supplies the computed content_id; this test never
    # invents one.  It is skipped until Cannon produces the real-run fixture.
    if not PLANNED_YIELD_FIXTURE_PATH.exists():
        pytest.skip("Cannon-generated v3 fixtures are not present in this environment")
    fixture = json.loads(PLANNED_YIELD_FIXTURE_PATH.read_text(encoding="utf-8"))
    evidence = RunEvidence.from_mapping(fixture)
    assert evidence.checkpoint_ref is not None
    assert evidence.checkpoint_ref["schema"] == "tpen.checkpoint-ref/v1"
    assert evidence.checkpoint_ref["content_id"]
    assert evidence.checkpoint_ref["profile"] == "train_resume"


def test_outcome_status_mismatch_is_refused(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    (run_dir / "status.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    (run_dir / "run-outcome.json").write_text(
        json.dumps(
            {
                "schema": "tpen.run-outcome/v1",
                "status": "planned_yield",
                "run_dir": str(run_dir),
                "error": None,
                "checkpoint_ref": None,
                "next_iteration": 1,
                "completed_updates": 1,
                "target_iterations": 3,
                "exception_type": None,
                "publication_stage": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = inspect_run(run_dir)
    assert evidence.schema == RUN_EVIDENCE_SCHEMA_V3
    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "status-outcome-mismatch"


@pytest.mark.parametrize(
    ("status", "verdict"),
    [("completed", "completed"), ("failed", "failed"), ("unknown", "incomplete"), ("refused", "incomplete")],
)
def test_each_typed_outcome_maps_without_becoming_eligible(
    tmp_path: Path, status: str, verdict: str
) -> None:
    run_dir = _make_run(tmp_path / status)
    (run_dir / "status.json").write_text(json.dumps({"status": status}) + "\n", encoding="utf-8")
    (run_dir / "run-outcome.json").write_text(
        json.dumps(RunResult(status=status).to_dict()) + "\n", encoding="utf-8"
    )
    evidence = inspect_run(run_dir)
    assert evidence.schema == RUN_EVIDENCE_SCHEMA_V3
    assert evidence.outcome == status
    assert evidence.verdict == verdict
    assert evidence.continuation_eligible is False


def test_v1_v2_normalization_remains_unchanged(tmp_path: Path) -> None:
    evidence = inspect_run(_make_run(tmp_path))
    assert normalize(evidence, tmp_path / "attempt")["schema"] == "tpen.run-evidence/v2"
