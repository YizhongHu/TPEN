"""Unit contracts for the public, read-only run evidence seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tpen.inspection as inspection
from tpen.checkpoint import PUBLICATION_CATALOG_FILENAME
from tpen.checkpoint.manifest import CheckpointManifest
from tpen.inspection import RUN_EVIDENCE_SCHEMA, RunEvidence, inspect_run


def _write_status(run_dir: Path, value: object) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_checkpoint(run_dir: Path, *, complete: bool = True) -> Path:
    checkpoint = run_dir / "checkpoints" / "step_000007"
    checkpoint.mkdir(parents=True)
    model = b"model-bytes"
    (checkpoint / "model.pt").write_bytes(model)
    CheckpointManifest(
        schema_version=2,
        kind="tpen.checkpoint",
        next_iteration=7,
        completed_updates=5,
        created_at_unix=1.0,
        files={"model": "model.pt"},
        hashes={"model_sha256": hashlib.sha256(model).hexdigest()},
        runtime={},
        provenance={},
    ).write(checkpoint / "manifest.json")
    if complete:
        (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return checkpoint


def _write_run(tmp_path: Path, *, status: str = "completed", checkpoint: bool = True) -> Path:
    run_dir = tmp_path / "run"
    _write_status(run_dir, {"status": status})
    if checkpoint:
        _write_checkpoint(run_dir)
    return run_dir


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_completed_evidence_preserves_manifest_counters(tmp_path: Path) -> None:
    evidence = inspect_run(_write_run(tmp_path))

    assert evidence.verdict == "completed"
    assert evidence.checkpoint == {
        "checkpoint_dir": str((tmp_path / "run" / "checkpoints" / "step_000007").resolve()),
        "schema_version": 2,
        "next_iteration": 7,
        "completed_updates": 5,
    }


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("missing-run-dir", "missing-run-dir"),
        ("missing-status", "missing-status"),
        ("multiple-status", "multiple-status"),
        ("malformed-status", "malformed-status"),
        ("non-object-status", "malformed-status"),
        ("incomplete-checkpoint", "incomplete-checkpoint"),
        ("corrupt-checkpoint", "corrupt-checkpoint"),
        ("corrupt-publication-catalog", "corrupt-publication-catalog"),
        ("missing-checkpoint", "missing-checkpoint"),
    ],
)
def test_each_refusal_code_is_distinct_and_noncompleted(tmp_path: Path, case: str, expected_code: str) -> None:
    run_dir = tmp_path / case
    if case == "missing-run-dir":
        pass
    elif case == "missing-status":
        run_dir.mkdir()
    elif case == "multiple-status":
        _write_status(run_dir, {"status": "completed"})
        (run_dir / "nested").mkdir()
        (run_dir / "nested" / "status.json").write_text("{}\n", encoding="utf-8")
    elif case == "malformed-status":
        run_dir.mkdir()
        (run_dir / "status.json").write_text("{", encoding="utf-8")
    elif case == "non-object-status":
        _write_status(run_dir, ["completed"])
    elif case == "incomplete-checkpoint":
        _write_status(run_dir, {"status": "completed"})
        _write_checkpoint(run_dir, complete=False)
    elif case == "corrupt-checkpoint":
        run_dir = _write_run(tmp_path / "corrupt-run")
        checkpoint = run_dir / "checkpoints" / "step_000007"
        (checkpoint / "model.pt").write_bytes(b"changed-model")
    elif case == "corrupt-publication-catalog":
        run_dir = _write_run(tmp_path / "corrupt-publication-run")
        (run_dir / "checkpoints" / PUBLICATION_CATALOG_FILENAME).write_text("garbage\n", encoding="utf-8")
    elif case == "missing-checkpoint":
        run_dir = _write_run(run_dir, checkpoint=False)

    evidence = inspect_run(run_dir)
    assert evidence.verdict == "refused"
    assert evidence.verdict != "completed"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == expected_code


@pytest.mark.parametrize("status_value", ["{", ["completed"]])
def test_malformed_status_variants_refuse(tmp_path: Path, status_value: object) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    content = status_value if isinstance(status_value, str) else json.dumps(status_value)
    (run_dir / "status.json").write_text(content, encoding="utf-8")

    evidence = inspect_run(run_dir)
    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "malformed-status"


@pytest.mark.parametrize("with_catalog", [True, False])
def test_inspection_does_not_mutate_checkpoint_tree(tmp_path: Path, with_catalog: bool) -> None:
    run_dir = _write_run(tmp_path)
    if with_catalog:
        (run_dir / "checkpoints" / PUBLICATION_CATALOG_FILENAME).write_text("", encoding="utf-8")
    before = _tree_snapshot(run_dir)

    evidence = inspect_run(run_dir)

    assert evidence.verdict == "completed"
    assert _tree_snapshot(run_dir) == before


def test_schema_round_trip_is_byte_stable_and_strict() -> None:
    evidence = RunEvidence(RUN_EVIDENCE_SCHEMA, "run", "incomplete", "running", None, None)
    encoded = json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":"))
    decoded = json.loads(encoded)
    assert json.dumps(RunEvidence.from_mapping(decoded).to_dict(), sort_keys=True, separators=(",", ":")) == encoded

    with pytest.raises(ValueError):
        RunEvidence.from_mapping({**decoded, "schema": "tpen.run-evidence/v999"})
    with pytest.raises(ValueError):
        RunEvidence.from_mapping({key: value for key, value in decoded.items() if key != "refusal"})
    with pytest.raises(ValueError):
        RunEvidence.from_mapping({**decoded, "extra": True})


def test_cli_returns_zero_for_refusal_and_two_for_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    refusal_run = tmp_path / "run"
    refusal_run.mkdir()
    assert inspection._main([str(refusal_run)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "refused"

    with pytest.raises(SystemExit) as exc_info:
        inspection._main([])
    assert exc_info.value.code == 2
