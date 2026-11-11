"""Round-two review contracts for checkpoint inspection edge cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tpen.checkpoint import (
    PUBLICATION_CATALOG_FILENAME,
    CheckpointManifest,
    CheckpointRef,
    ModelOnly,
    append_publication,
)
from tpen.inspection import inspect_run


def _write_status(run_dir: Path, status: str) -> None:
    """Write the status artifact for a fixture run."""

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps({"status": status}) + "\n", encoding="utf-8"
    )


def _write_checkpoint(
    run_dir: Path,
    *,
    step: int = 7,
    completed_updates: int = 5,
    payload: dict[str, object] | None = None,
    complete: bool = True,
    temporary: bool = False,
) -> Path:
    """Write a minimal v2 checkpoint fixture, optionally still in flight."""

    name = f"step_{step:06d}" + (".tmp" if temporary else "")
    checkpoint = run_dir / "checkpoints" / name
    checkpoint.mkdir(parents=True)
    model = f"model-{step}".encode("utf-8")
    (checkpoint / "model.pt").write_bytes(model)
    CheckpointManifest(
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
    ).write(checkpoint / "manifest.json")
    if complete:
        (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return checkpoint


@pytest.mark.parametrize("status", ["running", "failed"])
def test_noncompleted_run_with_only_tmp_checkpoint_is_not_refused(
    tmp_path: Path, status: str
) -> None:
    """An in-flight save is incomplete evidence, not a corrupt artifact.

    A ``.tmp`` checkpoint is an in-flight save rather than a corrupt artifact.
    A live or crashed run must not be reported as refused merely because that
    save has not reached its committed-directory state.
    """

    run_dir = tmp_path / "run"
    _write_status(run_dir, status)
    _write_checkpoint(run_dir, temporary=True, complete=False)

    evidence = inspect_run(run_dir)

    assert evidence.verdict == ("failed" if status == "failed" else "incomplete")
    assert evidence.checkpoint is None
    assert evidence.refusal is None


def test_published_ref_identity_is_validated_before_mode_gate(tmp_path: Path) -> None:
    """A stale published identity must refuse before payload-mode filtering."""

    run_dir = tmp_path / "run"
    _write_status(run_dir, "completed")
    (run_dir / "checkpoints").mkdir(parents=True)
    outside_step = _write_checkpoint(tmp_path / "elsewhere", step=9, completed_updates=9)
    catalog = run_dir / "checkpoints" / PUBLICATION_CATALOG_FILENAME
    append_publication(catalog, CheckpointRef.from_directory(outside_step))

    # Keep the model bytes and counters valid while making the catalog's
    # manifest digest stale and changing the payload profile to model-only.
    model = (outside_step / "model.pt").read_bytes()
    CheckpointManifest(
        schema_version=2,
        kind="tpen.checkpoint",
        next_iteration=9,
        completed_updates=9,
        created_at_unix=1.0,
        files={"model": "model.pt"},
        hashes={"model_sha256": hashlib.sha256(model).hexdigest()},
        runtime={},
        provenance={"run_id": "elsewhere"},
        payload=ModelOnly().to_manifest(),
    ).write(outside_step / "manifest.json")

    evidence = inspect_run(run_dir)

    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "corrupt-checkpoint"


def test_model_only_committed_checkpoint_is_not_reported_incomplete(tmp_path: Path) -> None:
    """A committed model-only artifact means no resumable checkpoint exists.

    The exact refusal code is the orchestrator's call: the artifact is
    committed, so ``incomplete-checkpoint`` is the wrong classification even
    though no committed train-resume checkpoint is available.
    """

    run_dir = tmp_path / "run"
    _write_status(run_dir, "completed")
    _write_checkpoint(run_dir, payload=ModelOnly().to_manifest())

    evidence = inspect_run(run_dir)

    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "no-train-resume-checkpoint"


def test_torn_publication_catalog_has_distinct_refusal_code(tmp_path: Path) -> None:
    """An unterminated catalog tail is repairable rather than corrupt."""

    run_dir = tmp_path / "run"
    _write_status(run_dir, "completed")
    catalog = run_dir / "checkpoints" / PUBLICATION_CATALOG_FILENAME
    catalog.parent.mkdir(parents=True)
    catalog.write_text('{"schema":', encoding="utf-8")

    evidence = inspect_run(run_dir)

    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "torn-publication-catalog"


def test_hash_corrupt_published_ref_outside_root_is_refused(tmp_path: Path) -> None:
    """A published checkpoint outside the run is revalidated by content hash."""

    run_dir = tmp_path / "run"
    _write_status(run_dir, "completed")
    _write_checkpoint(run_dir)
    outside_step = _write_checkpoint(tmp_path / "elsewhere", step=9, completed_updates=9)
    catalog = run_dir / "checkpoints" / PUBLICATION_CATALOG_FILENAME
    append_publication(catalog, CheckpointRef.from_directory(outside_step))
    (outside_step / "model.pt").write_bytes(b"changed")

    evidence = inspect_run(run_dir)

    assert evidence.verdict == "refused"
    assert evidence.refusal is not None
    assert evidence.refusal["code"] == "corrupt-checkpoint"
