"""Unit contracts for the public, read-only run evidence seam."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from tpen.inspection import RUN_EVIDENCE_SCHEMA, RunEvidence, inspect_run


def _write_run(tmp_path: Path, *, status: str = "completed") -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(json.dumps({"status": status}) + "\n", encoding="utf-8")
    if status == "completed":
        checkpoint = run_dir / "checkpoints" / "step_000007"
        checkpoint.mkdir(parents=True)
        model = b"model-bytes"
        (checkpoint / "model.pt").write_bytes(model)
        model_hash = hashlib.sha256(model).hexdigest()
        (checkpoint / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "kind": "tpen.checkpoint",
                    "next_iteration": 7,
                    "completed_updates": 5,
                    "created_at_unix": 1.0,
                    "files": {"model": "model.pt"},
                    "hashes": {"model_sha256": model_hash},
                    "runtime": {},
                    "provenance": {},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return run_dir


def test_completed_evidence_preserves_manifest_counters(tmp_path: Path) -> None:
    evidence = inspect_run(_write_run(tmp_path))

    assert evidence.verdict == "completed"
    assert evidence.checkpoint == {
        "checkpoint_dir": str(tmp_path / "run" / "checkpoints" / "step_000007"),
        "schema_version": 2,
        "next_iteration": 7,
        "completed_updates": 5,
    }


def test_missing_status_is_a_total_refusal(tmp_path: Path) -> None:
    evidence = inspect_run(tmp_path / "missing")

    assert evidence.verdict == "refused"
    assert evidence.refusal == {"code": "missing-run-dir", "reason": evidence.refusal["reason"]}


def test_failed_and_running_statuses_do_not_claim_completion(tmp_path: Path) -> None:
    failed = inspect_run(_write_run(tmp_path / "failed", status="failed"))
    running = inspect_run(_write_run(tmp_path / "running", status="running"))

    assert (failed.verdict, failed.run_status) == ("failed", "failed")
    assert (running.verdict, running.run_status) == ("incomplete", "running")


def test_schema_round_trip_rejects_unknown_keys() -> None:
    evidence = RunEvidence(RUN_EVIDENCE_SCHEMA, "run", "incomplete", "running", None, None)
    assert RunEvidence.from_mapping(evidence.to_dict()).to_dict() == evidence.to_dict()

    encoded = evidence.to_dict()
    encoded["unexpected"] = True
    try:
        RunEvidence.from_mapping(encoded)
    except ValueError:
        pass
    else:  # pragma: no cover - assertion spelling keeps this test dependency-free
        raise AssertionError("unknown evidence keys must be rejected")


def test_import_control_excludes_execution_and_private_checkpoint_names() -> None:
    tree = ast.parse(Path("tpen/inspection.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "tpen.execution"
            assert all(not alias.name.startswith("_") for alias in node.names)
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("tpen.execution") for alias in node.names)
