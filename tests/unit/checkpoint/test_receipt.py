"""Tests for typed checkpoint publication size and duration receipts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tpen.checkpoint.artifact import COMPLETE_MARKER, read_latest
from tpen.checkpoint.catalog import publication_catalog_path, read_publications
from tpen.checkpoint.reference import CheckpointRef
from tpen.checkpoint.receipt import (
    PUBLICATION_RECEIPT_SCHEMA,
    CheckpointFileSize,
    CheckpointPublished,
    append_publication_receipt,
    build_publication_receipt,
    measure_checkpoint_files,
    publication_receipt_path,
)


def _manifest(step: int, *, files: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        "completed_updates": step - 1,
        "created_at_unix": 123.0,
        "files": files,
        "hashes": {},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "run", "git_sha": "deadbeef"},
    }


def _write_checkpoint(root: Path, step: int = 7, *, extra_files: dict[str, bytes] | None = None) -> tuple[Path, dict[str, str]]:
    """Write a hand-built checkpoint directory with known file byte lengths."""

    checkpoint_dir = root / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True)
    files = {
        "resolved_config": "resolved_config.yaml",
        "model": "model.pt",
        "optimizer": "optimizer.pt",
    }
    (checkpoint_dir / files["resolved_config"]).write_bytes(b"config: true\n")
    (checkpoint_dir / files["model"]).write_bytes(b"model-bytes-1234")
    (checkpoint_dir / files["optimizer"]).write_bytes(b"optimizer-bytes-56")
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(_manifest(step, files=files), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")
    if extra_files:
        for name, content in extra_files.items():
            (checkpoint_dir / name).write_bytes(content)
    return checkpoint_dir, files


def test_measure_checkpoint_files_reads_stat_sizes_after_close(tmp_path: Path) -> None:
    checkpoint_dir, files = _write_checkpoint(tmp_path)

    measured = measure_checkpoint_files(checkpoint_dir, files)

    by_component = {entry.component: entry for entry in measured}
    assert by_component["resolved_config"].size_bytes == len(b"config: true\n")
    assert by_component["model"].size_bytes == len(b"model-bytes-1234")
    assert by_component["optimizer"].size_bytes == len(b"optimizer-bytes-56")
    assert by_component["manifest"].relative_path == "manifest.json"
    assert by_component["manifest"].size_bytes == (checkpoint_dir / "manifest.json").stat().st_size
    assert by_component["complete"].relative_path == COMPLETE_MARKER
    assert by_component["complete"].size_bytes == len(b"complete\n")
    assert len(measured) == len(files) + 2


def test_measure_checkpoint_files_ignores_files_the_manifest_does_not_name(tmp_path: Path) -> None:
    """An untracked file dropped into the directory must not affect the receipt.

    This is the test that actually falsifies a scan-based implementation: a
    du/glob/os.walk stand-in would pick up ``stray.bin`` and inflate the
    total, while a strictly-named lookup over ``files`` cannot see it.
    """

    checkpoint_dir, files = _write_checkpoint(
        tmp_path, extra_files={"stray.bin": b"x" * 10_000}
    )

    measured = measure_checkpoint_files(checkpoint_dir, files)

    assert {entry.relative_path for entry in measured} == {
        *files.values(),
        "manifest.json",
        COMPLETE_MARKER,
    }
    assert sum(entry.size_bytes for entry in measured) < 10_000


def _ref_for(checkpoint_dir: Path) -> CheckpointRef:
    return CheckpointRef.from_directory(checkpoint_dir)


def test_build_publication_receipt_totals_equal_typed_file_sum(tmp_path: Path) -> None:
    checkpoint_dir, files = _write_checkpoint(tmp_path)
    ref = _ref_for(checkpoint_dir)

    receipt = build_publication_receipt(
        ref, checkpoint_dir, files, write_duration_sec=0.5, publish_duration_sec=0.1
    )

    assert receipt.summary.total_bytes == sum(entry.size_bytes for entry in receipt.files)
    assert receipt.summary.total_bytes == receipt.summary.payload_bytes + receipt.summary.metadata_bytes
    assert receipt.summary.file_count == len(receipt.files) == len(files) + 2
    # model and optimizer are payload; resolved_config/manifest/complete are metadata.
    payload_names = {"model", "optimizer"}
    expected_payload = sum(
        entry.size_bytes for entry in receipt.files if entry.component in payload_names
    )
    assert receipt.summary.payload_bytes == expected_payload
    assert receipt.summary.content_id == ref.content_id
    assert receipt.summary.checkpoint_dir == checkpoint_dir.name


def test_checkpoint_published_rejects_inconsistent_totals() -> None:
    with pytest.raises(ValueError):
        CheckpointPublished(
            checkpoint_dir="step_000001",
            content_id="0" * 64,
            file_count=1,
            payload_bytes=10,
            metadata_bytes=5,
            total_bytes=999,
            write_duration_sec=0.1,
            publish_duration_sec=0.1,
        )


def test_checkpoint_file_size_rejects_negative_bytes() -> None:
    with pytest.raises(ValueError):
        CheckpointFileSize(component="model", relative_path="model.pt", size_bytes=-1)


def test_manifest_and_complete_are_not_mutated_by_receipt_construction(tmp_path: Path) -> None:
    checkpoint_dir, files = _write_checkpoint(tmp_path)
    manifest_path = checkpoint_dir / "manifest.json"
    complete_path = checkpoint_dir / COMPLETE_MARKER
    manifest_before = manifest_path.read_bytes()
    complete_before = complete_path.read_bytes()

    ref = _ref_for(checkpoint_dir)
    build_publication_receipt(
        ref, checkpoint_dir, files, write_duration_sec=0.0, publish_duration_sec=0.0
    )

    assert manifest_path.read_bytes() == manifest_before
    assert complete_path.read_bytes() == complete_before
    # The manifest self-size cycle is resolved by omission: its own content
    # never carries a size/byte-count field about itself.
    manifest_data = json.loads(manifest_before)
    assert not any("size" in key or "bytes" in key for key in manifest_data)


def test_append_publication_receipt_is_append_only_jsonl(tmp_path: Path) -> None:
    checkpoint_dir, files = _write_checkpoint(tmp_path / "checkpoints", step=7)
    ref = _ref_for(checkpoint_dir)
    receipt = build_publication_receipt(
        ref, checkpoint_dir, files, write_duration_sec=1.0, publish_duration_sec=0.2
    )
    path = publication_receipt_path(tmp_path / "checkpoints")

    append_publication_receipt(path, receipt)
    first_bytes = path.read_bytes()

    second_checkpoint_dir, second_files = _write_checkpoint(tmp_path / "checkpoints", step=8)
    second_ref = _ref_for(second_checkpoint_dir)
    second_receipt = build_publication_receipt(
        second_ref, second_checkpoint_dir, second_files, write_duration_sec=1.0, publish_duration_sec=0.2
    )
    append_publication_receipt(path, second_receipt)

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert rows[0].encode() + b"\n" == first_bytes
    first_record = json.loads(rows[0])
    assert first_record["schema"] == PUBLICATION_RECEIPT_SCHEMA
    assert first_record["summary"]["content_id"] == ref.content_id
    second_record = json.loads(rows[1])
    assert second_record["summary"]["content_id"] == second_ref.content_id


def test_receipt_append_failure_does_not_fail_a_committed_save(tmp_path: Path) -> None:
    """A failed receipt append must not turn a committed save into a failed save.

    The receipt is telemetry: :mod:`tpen.checkpoint.receipt` frames it as a
    size/duration fact *about* a publication that already happened.  It is
    appended after ``tmp_dir.rename`` (the commit), after ``catalog.publish``,
    and after ``write_latest``; unlike those two it is not load-bearing for
    restore and has no ``reconcile_publication`` repair path.  An ``OSError``
    from the append -- in production, a quota/ENOSPC failure on the receipts
    log -- therefore reports a *failed* save for a checkpoint that is durably
    committed, published, and pointed at by ``latest.json``.

    The append is made to fail without patching the subject: the receipts log
    path is pre-created as a DIRECTORY, so ``append_jsonl``'s ``path.open("a")``
    raises ``IsADirectoryError`` (an ``OSError``).  The step directory, the
    publication catalog, and ``latest.json`` are separately named entries in
    the same root, so the induced failure cannot reach the checkpoint write.

    Assertions (2)-(5) deliberately execute in the red arm as well as the green
    one: in the red arm they *are* the evidence for the finding, recording that
    today's raise leaves a committed, published, latest-pointed checkpoint
    behind a save the trainer was told had failed.
    """

    # The real save path, exactly as the publication-receipt integration tests
    # in tests/unit/callback/test_checkpoint.py exercise it.  Imported rather
    # than re-implemented so this test cannot drift from the checkpoint that
    # suite builds, and imported inside the function so the rest of this module
    # keeps its existing torch-free import surface.
    from tests.unit.callback.test_checkpoint import _write_checkpoint as _save_real_checkpoint

    root = tmp_path / "checkpoints"
    root.mkdir(parents=True)
    receipt_path = publication_receipt_path(root)
    # A directory at the receipts log path: appendable-as-a-file is exactly the
    # property the receipt step needs and nothing else in the save path does.
    receipt_path.mkdir()

    returned: Path | None = None
    raised: BaseException | None = None
    try:
        returned = _save_real_checkpoint(tmp_path)
    except Exception as error:  # noqa: BLE001 - the failure mode under test
        raised = error

    step_dir = root / "step_000003"

    # (2) The checkpoint committed: the step directory exists and is COMPLETE.
    assert step_dir.is_dir()
    assert (step_dir / COMPLETE_MARKER).is_file()

    # (3) The publication catalog carries exactly the new checkpoint's row.
    published = read_publications(publication_catalog_path(root))
    assert [ref.checkpoint_dir.name for ref in published] == [step_dir.name]

    # (4) latest.json points at the new step.
    assert read_latest(root)["checkpoint_dir"] == step_dir.name

    # (5) No receipt line was appended -- the induced failure is real, not a
    # partially written record.
    assert receipt_path.is_dir()
    assert list(receipt_path.iterdir()) == []

    # (1) The save reported success.  RED at 406f461b: the OSError from the
    # telemetry append propagates out of save_checkpoint instead.
    assert raised is None, (
        "receipt append failure aborted an already-committed, published save: "
        f"{type(raised).__name__}: {raised}"
    )
    assert returned == step_dir
