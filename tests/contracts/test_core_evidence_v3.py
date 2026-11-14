"""Round trips and refusal rules for the core-owned run-evidence v3 wire."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tpen.artifacts import ArtifactDestination, RunResult
from tpen.callback.planned_yield import PlannedYield
from tpen.checkpoint import CheckpointRef, ModelOnly, TrainResume, append_publication
from tpen.checkpoint.manifest import CheckpointManifest
from tpen.inspection import RUN_EVIDENCE_SCHEMA_V3, RunEvidence, inspect_run
from tpen.run import PLANNED_YIELD_EXIT_CODE, run_from_config

from tests.integration.test_s4_planned_yield_public_entry import _cfg

from ._regen_fixtures import (
    PLANNED_YIELD_FIXTURE_PATH,
    UNKNOWN_FIXTURE_PATH,
    _make_run,
    normalize,
    regenerate_planned_yield,
    regenerate_unknown,
)


def test_unknown_fixture_regenerates_from_typed_result_producer() -> None:
    assert UNKNOWN_FIXTURE_PATH.exists(), "Cannon-generated unknown v3 fixture is missing"
    fixture = json.loads(UNKNOWN_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert regenerate_unknown() == fixture
    assert RunEvidence.from_mapping(fixture).continuation_eligible is False


def test_planned_yield_fixture_regenerates_from_real_s4_entry() -> None:
    assert PLANNED_YIELD_FIXTURE_PATH.exists(), "Cannon-generated planned-yield v3 fixture is missing"
    fixture = json.loads(PLANNED_YIELD_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert regenerate_planned_yield() == fixture
    assert fixture["schema"] == RUN_EVIDENCE_SCHEMA_V3
    assert fixture["outcome"] == "planned_yield"
    assert fixture["checkpoint_ref"]["profile"] == "train_resume"
    assert fixture["checkpoint_ref"]["content_id"] == "<sha256:content_id>"
    assert fixture["checkpoint_ref"]["manifest_sha256"] == "<sha256:manifest>"


def test_v3_round_trip_preserves_full_checkpoint_binding() -> None:
    # The canonical fixture validates the identity shape; the live test below
    # binds these values to the checkpoint bytes.
    assert PLANNED_YIELD_FIXTURE_PATH.exists(), "Cannon-generated planned-yield v3 fixture is missing"
    fixture = json.loads(PLANNED_YIELD_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["checkpoint_ref"]["schema"] == "tpen.checkpoint-ref/v1"
    assert fixture["checkpoint_ref"]["content_id"] == "<sha256:content_id>"
    assert fixture["checkpoint_ref"]["manifest_sha256"] == "<sha256:manifest>"
    assert fixture["checkpoint_ref"]["model_sha256"] == "<sha256:model>"
    assert fixture["checkpoint_ref"]["profile"] == "train_resume"


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


def test_real_completed_public_entry_produces_v3_evidence(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "logical", "completed")
    destination = ArtifactDestination(tmp_path / "physical")
    assert run_from_config(cfg, artifact_destination=destination, raise_exceptions=True) == 0

    evidence = inspect_run(destination.physical_root)
    assert evidence.schema == RUN_EVIDENCE_SCHEMA_V3
    assert evidence.outcome == "completed"
    assert evidence.verdict == "completed"
    assert evidence.continuation_eligible is False


def test_real_planned_yield_public_entry_produces_v3_evidence(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "logical", "planned-yield")
    destination = ArtifactDestination(tmp_path / "physical")
    assert (
        run_from_config(
            cfg,
            artifact_destination=destination,
            runtime_callbacks=(PlannedYield(1, 3),),
            raise_exceptions=True,
        )
        == PLANNED_YIELD_EXIT_CODE
    )

    evidence = inspect_run(destination.physical_root)
    assert evidence.schema == RUN_EVIDENCE_SCHEMA_V3
    assert evidence.outcome == "planned_yield"
    assert evidence.checkpoint_ref is not None
    assert evidence.checkpoint_ref["profile"] == "train_resume"
    assert evidence.continuation_eligible is True


def test_live_planned_yield_ref_binds_checkpoint_bytes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "logical", "planned-yield-live-ref")
    destination = ArtifactDestination(tmp_path / "physical")
    assert (
        run_from_config(
            cfg,
            artifact_destination=destination,
            runtime_callbacks=(PlannedYield(1, 3),),
            raise_exceptions=True,
        )
        == PLANNED_YIELD_EXIT_CODE
    )

    evidence = inspect_run(destination.physical_root)
    assert evidence.checkpoint_ref is not None
    live_ref = CheckpointRef.from_directory(evidence.checkpoint_ref["checkpoint_dir"])
    assert evidence.checkpoint_ref["content_id"] == live_ref.content_id
    assert evidence.checkpoint_ref["manifest_sha256"] == live_ref.manifest_sha256
    assert evidence.checkpoint_ref["model_sha256"] == live_ref.model_sha256


def test_pre_cx01_run_outcome_without_schema_marker_is_read_only(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    legacy_outcome = RunResult(status="completed").to_dict()
    legacy_outcome.pop("schema")
    (run_dir / "run-outcome.json").write_text(
        json.dumps(legacy_outcome) + "\n", encoding="utf-8"
    )

    evidence = inspect_run(run_dir)
    assert evidence.schema == RUN_EVIDENCE_SCHEMA_V3
    assert evidence.outcome == "completed"
    assert evidence.verdict == "completed"


def _planned_yield_fixture_run(
    root: Path,
    *,
    profile: str = "train_resume",
    publish: bool = True,
    next_iteration: int | None = None,
    completed_updates: int | None = None,
) -> tuple[Path, CheckpointRef]:
    """Build one producer-shaped planned-yield artifact for refusal tests."""

    run_dir = _make_run(root)
    checkpoint_dir = run_dir / "checkpoints" / "step_000007"
    manifest = CheckpointManifest.from_mapping(
        json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    )
    payload = TrainResume().to_manifest() if profile == "train_resume" else ModelOnly().to_manifest()
    replace(manifest, payload=payload).write(checkpoint_dir / "manifest.json")
    ref = CheckpointRef.from_directory(checkpoint_dir)
    if publish:
        append_publication(run_dir / "checkpoints" / "publications.jsonl", ref)
    (run_dir / "status.json").write_text('{"status":"planned_yield"}\n', encoding="utf-8")
    result = RunResult(
        status="planned_yield",
        checkpoint_ref=ref,
        next_iteration=ref.next_iteration if next_iteration is None else next_iteration,
        completed_updates=ref.completed_updates if completed_updates is None else completed_updates,
        target_iterations=10,
    )
    (run_dir / "run-outcome.json").write_text(
        json.dumps(result.to_dict()) + "\n", encoding="utf-8"
    )
    return run_dir, ref


def test_model_only_checkpoint_is_refused(tmp_path: Path) -> None:
    run_dir, _ = _planned_yield_fixture_run(tmp_path, profile="model_only")

    evidence = inspect_run(run_dir)
    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "model-only-checkpoint"
    assert evidence.continuation_eligible is False


def test_unpublished_planned_yield_ref_is_refused(tmp_path: Path) -> None:
    run_dir, _ = _planned_yield_fixture_run(tmp_path, publish=False)

    evidence = inspect_run(run_dir)
    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "unpublished-checkpoint-ref"


def test_planned_yield_checkpoint_counter_mismatch_is_refused(tmp_path: Path) -> None:
    run_dir, _ = _planned_yield_fixture_run(tmp_path, next_iteration=8)

    evidence = inspect_run(run_dir)
    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "outcome-checkpoint-mismatch"


def test_stale_planned_yield_ref_is_refused(tmp_path: Path) -> None:
    run_dir, _ = _planned_yield_fixture_run(tmp_path)
    (run_dir / "checkpoints" / "step_000007" / "model.pt").write_bytes(b"stale\n")

    evidence = inspect_run(run_dir)
    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "stale-checkpoint-ref"


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
