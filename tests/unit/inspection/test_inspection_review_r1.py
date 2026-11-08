"""Round-one review contracts for the public run-evidence inspection seam."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import tpen.inspection as inspection
from tpen.checkpoint import (
    PUBLICATION_CATALOG_FILENAME,
    CheckpointManifest,
    CheckpointRef,
    ModelOnly,
    append_publication,
)
from tpen.inspection import RUN_EVIDENCE_SCHEMA, RunEvidence, inspect_run


def _write_status(run_dir: Path, status: object) -> None:
    """Write one status artifact for a fixture run."""

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(json.dumps(status) + "\n", encoding="utf-8")


def _write_checkpoint(
    run_dir: Path,
    *,
    step: int = 7,
    completed_updates: int = 5,
    payload: dict[str, object] | None = None,
    complete: bool = True,
) -> Path:
    """Write a minimal v2 checkpoint with a self-consistent model digest."""

    checkpoint = run_dir / "checkpoints" / f"step_{step:06d}"
    checkpoint.mkdir(parents=True)
    model = f"model-{step}".encode("utf-8")
    (checkpoint / "model.pt").write_bytes(model)
    manifest = CheckpointManifest(
        schema_version=2,
        kind="tpen.checkpoint",
        next_iteration=step,
        completed_updates=completed_updates,
        created_at_unix=1.0,
        files={"model": "model.pt"},
        hashes={"model_sha256": hashlib.sha256(model).hexdigest()},
        runtime={},
        provenance={"run_id": run_dir.name},
        payload=payload,
    )
    manifest.write(checkpoint / "manifest.json")
    if complete:
        (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return checkpoint


def _write_complete_run(run_dir: Path, *, step: int = 7, updates: int = 5) -> Path:
    """Write a completed run and return its checkpoint directory."""

    _write_status(run_dir, {"status": "completed"})
    return _write_checkpoint(
        run_dir,
        step=step,
        completed_updates=updates,
    )


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Capture file bytes and nanosecond mtimes beneath a fixture root."""

    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_nested_run_dir_reports_committed_checkpoint(tmp_path: Path) -> None:
    """The executor output root should resolve its nested run directory."""

    run_dir = tmp_path / "experiment" / "sector" / "run-id"
    _write_complete_run(run_dir)

    evidence = inspect_run(tmp_path)

    assert evidence.verdict == "completed"
    assert evidence.checkpoint is not None
    assert evidence.checkpoint["next_iteration"] == 7
    assert evidence.checkpoint["completed_updates"] == 5


@pytest.mark.parametrize("status", [7, True, ["completed"], {"k": 1}])
def test_non_string_status_value_round_trips_through_schema(tmp_path: Path, status: object) -> None:
    """Every emitted public record must be consumable by its own schema."""

    run_dir = tmp_path / "run"
    _write_status(run_dir, {"status": status})
    evidence = inspect_run(run_dir)

    decoded = json.loads(json.dumps(evidence.to_dict(), sort_keys=True))
    assert RunEvidence.from_mapping(decoded) == evidence


def test_manifest_overflow_is_refused_not_raised(tmp_path: Path) -> None:
    """Malformed numeric manifest values should become stable refusals."""

    run_dir = tmp_path / "run"
    _write_status(run_dir, {"status": "completed"})
    checkpoint = run_dir / "checkpoints" / "step_000007"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(b"model")
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "tpen.checkpoint",
                "next_iteration": 1e400,
                "completed_updates": 5,
                "created_at_unix": 1.0,
                "files": {"model": "model.pt"},
                "hashes": {},
                "runtime": {},
                "provenance": {},
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")

    evidence = inspect_run(run_dir)

    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "corrupt-checkpoint"


def test_import_control_is_prefix_based() -> None:
    """Forbidden imports include descendants and private checkpoint modules."""

    source = Path(inspection.__file__)
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    forbidden_prefixes = ("tpen.execution", "experiments", "tpen_executor")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not any(module == prefix or module.startswith(prefix + ".") for prefix in forbidden_prefixes)
            assert all(not alias.name.startswith("_") for alias in node.names)
            if module.startswith("tpen.checkpoint"):
                assert all(not segment.startswith("_") for segment in module.split("."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(alias.name == prefix or alias.name.startswith(prefix + ".") for prefix in forbidden_prefixes)


def test_published_catalog_row_is_read_and_tree_unchanged(tmp_path: Path) -> None:
    """Published train-resume evidence is read without mutating the run."""

    run_dir = tmp_path / "run"
    step_dir = _write_complete_run(run_dir)
    catalog = run_dir / "checkpoints" / PUBLICATION_CATALOG_FILENAME
    append_publication(catalog, CheckpointRef.from_directory(step_dir))
    before = _tree_snapshot(run_dir)

    evidence = inspect_run(run_dir)

    assert evidence.verdict == "completed"
    assert evidence.checkpoint == {
        "checkpoint_dir": str(step_dir),
        "schema_version": 2,
        "next_iteration": 7,
        "completed_updates": 5,
    }
    assert _tree_snapshot(run_dir) == before


def test_newer_unpublished_checkpoint_is_reported_without_writing(tmp_path: Path) -> None:
    """Inspection selects newer committed evidence but never reconciles it."""

    run_dir = tmp_path / "run"
    first = _write_complete_run(run_dir)
    _write_checkpoint(run_dir, step=9, completed_updates=8)
    catalog = run_dir / "checkpoints" / PUBLICATION_CATALOG_FILENAME
    append_publication(catalog, CheckpointRef.from_directory(first))
    before = _tree_snapshot(run_dir)

    evidence = inspect_run(run_dir)

    assert evidence.verdict == "completed"
    assert evidence.checkpoint is not None
    assert evidence.checkpoint["next_iteration"] == 9
    assert evidence.checkpoint["completed_updates"] == 8
    assert not (run_dir / "checkpoints" / "latest.json").exists()
    assert _tree_snapshot(run_dir) == before


def test_published_ref_outside_root_is_gated_by_train_resume(tmp_path: Path) -> None:
    """A model-only publication outside the run must not win selection."""

    run_dir = tmp_path / "run"
    local_step = _write_complete_run(run_dir)
    outside_root = tmp_path / "elsewhere"
    outside_step = _write_checkpoint(
        outside_root,
        step=9,
        completed_updates=9,
        payload=ModelOnly().to_manifest(),
    )
    catalog = run_dir / "checkpoints" / PUBLICATION_CATALOG_FILENAME
    append_publication(catalog, CheckpointRef.from_directory(outside_step))

    evidence = inspect_run(run_dir)

    assert evidence.verdict == "completed"
    assert evidence.checkpoint is not None
    assert evidence.checkpoint["next_iteration"] == 7
    assert Path(evidence.checkpoint["checkpoint_dir"]).parent == local_step.parent


def test_schema_round_trip_completed_and_refused_shapes(tmp_path: Path) -> None:
    """Completed and refused records remain byte-stable through the schema."""

    _write_complete_run(tmp_path / "completed")
    completed = inspect_run(tmp_path / "completed")
    refused_dir = tmp_path / "refused"
    refused_dir.mkdir()
    refused = inspect_run(refused_dir)
    for evidence in (completed, refused):
        encoded = json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":"))
        decoded = json.loads(encoded)
        restored = RunEvidence.from_mapping(decoded)
        assert restored == evidence
        assert json.dumps(restored.to_dict(), sort_keys=True, separators=(",", ":")) == encoded


def test_schema_rejects_inconsistent_verdict_payloads(tmp_path: Path) -> None:
    """Verdicts must agree with the checkpoint and refusal payload fields."""

    base = {
        "schema": RUN_EVIDENCE_SCHEMA,
        "run_dir": "run",
        "run_status": None,
        "checkpoint": None,
        "refusal": None,
    }
    valid_checkpoint = {
        "checkpoint_dir": str((tmp_path / "step_000007").resolve()),
        "schema_version": 2,
        "next_iteration": 7,
        "completed_updates": 5,
    }
    cases = [
        {"verdict": "completed", "checkpoint": None},
        {"verdict": "refused", "refusal": None},
        {"verdict": "incomplete", "checkpoint": {"x": 1}},
        {"verdict": "completed", "refusal": {"code": "x", "reason": "y"}},
    ]
    for update in cases:
        data = {**base, "verdict": "incomplete", **update}
        with pytest.raises(ValueError):
            RunEvidence.from_mapping(data)

    accepted = {**base, "verdict": "incomplete", "checkpoint": valid_checkpoint}
    assert RunEvidence.from_mapping(accepted).checkpoint == valid_checkpoint


def test_cli_module_entry_prints_one_json_line(tmp_path: Path) -> None:
    """The module entry point emits one parseable JSON record and stable codes."""

    run_dir = tmp_path / "run"
    _write_complete_run(run_dir)
    completed = subprocess.run(
        [sys.executable, "-m", "tpen.inspection", str(run_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert completed.stdout.count("\n") == 1
    parsed = json.loads(completed.stdout)
    assert RunEvidence.from_mapping(parsed).verdict == "completed"

    missing = subprocess.run(
        [sys.executable, "-m", "tpen.inspection", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 0
    assert json.loads(missing.stdout)["verdict"] == "refused"

    usage = subprocess.run(
        [sys.executable, "-m", "tpen.inspection"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert usage.returncode == 2


def test_tmp_only_checkpoint_root_is_incomplete_not_missing(tmp_path: Path) -> None:
    """A temporary checkpoint directory signals an incomplete publication."""

    run_dir = tmp_path / "run"
    _write_status(run_dir, {"status": "completed"})
    checkpoint = run_dir / "checkpoints" / "step_000007.tmp"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.pt").write_bytes(b"model")
    CheckpointManifest(
        schema_version=2,
        kind="tpen.checkpoint",
        next_iteration=7,
        completed_updates=5,
        created_at_unix=1.0,
        files={"model": "model.pt"},
        hashes={},
        runtime={},
        provenance={},
    ).write(checkpoint / "manifest.json")

    evidence = inspect_run(run_dir)

    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "incomplete-checkpoint"


@pytest.mark.parametrize("status", ["running", "failed"])
def test_noncompleted_run_with_committed_checkpoint_reports_counters(tmp_path: Path, status: str) -> None:
    """Committed train-resume facts remain visible for non-completed runs."""

    run_dir = tmp_path / status
    _write_status(run_dir, {"status": status})
    _write_checkpoint(run_dir, step=7, completed_updates=5)

    evidence = inspect_run(run_dir)

    assert evidence.verdict == ("failed" if status == "failed" else "incomplete")
    assert evidence.checkpoint is not None
    assert evidence.checkpoint["next_iteration"] == 7
    assert evidence.checkpoint["completed_updates"] == 5


def test_status_io_error_is_distinct(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_status(run_dir, {"status": "completed"})
    original_open = Path.open

    def fail_status_open(path: Path, *args: object, **kwargs: object):
        if path.name == "status.json":
            raise OSError("status temporarily unavailable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_status_open)
    evidence = inspect_run(run_dir)
    assert evidence.refusal == {"code": "status-io-error", "reason": "cannot read " + str((run_dir / "status.json").resolve()) + ": status temporarily unavailable"}


def test_checkpoint_io_error_is_distinct(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_complete_run(run_dir)
    def fail_manifest_read(manifest: object, *, mode: str, path: Path | None = None):
        raise OSError("checkpoint temporarily unavailable")

    monkeypatch.setattr(inspection, "validate_manifest_schema", fail_manifest_read)
    evidence = inspect_run(run_dir)
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "checkpoint-io-error"
