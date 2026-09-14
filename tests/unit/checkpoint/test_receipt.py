"""Tests for typed checkpoint publication size and duration receipts."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest

from tpen.checkpoint.artifact import COMPLETE_MARKER, read_latest
from tpen.checkpoint.catalog import (
    publication_catalog_path,
    read_publications,
    reconcile_publication,
)
from tpen.checkpoint.reference import CheckpointRef
from tpen.checkpoint.receipt import (
    PUBLICATION_RECEIPT_SCHEMA,
    CheckpointFileSize,
    CheckpointPublished,
    append_publication_receipt,
    backfill_publication_receipt,
    build_publication_receipt,
    has_publication_receipt,
    iter_valid_publication_receipts,
    measure_checkpoint_files,
    publication_receipt_path,
    record_publication_receipt,
)


def _manifest(step: int, *, files: dict[str, str]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        "completed_updates": step - 1,
        "created_at_unix": 123.0,
        "files": files,
        "hashes": {
            "model_config": "a" * 64,
            "hamiltonian_config": "b" * 64,
        },
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


def test_checkpoint_published_accepts_explicit_absent_durations(tmp_path: Path) -> None:
    published = CheckpointPublished(
        checkpoint_dir="step_000001",
        content_id="0" * 64,
        file_count=1,
        payload_bytes=10,
        metadata_bytes=5,
        total_bytes=15,
        write_duration_sec=None,
        publish_duration_sec=None,
    )
    assert published.write_duration_sec is None
    assert published.publish_duration_sec is None
    # Explicit null in the serialized record, never a missing key and never 0.0.
    data = published.to_dict()
    assert "write_duration_sec" in data
    assert data["write_duration_sec"] is None
    assert "publish_duration_sec" in data
    assert data["publish_duration_sec"] is None
    assert json.loads(json.dumps(data))["write_duration_sec"] is None


def test_checkpoint_published_still_rejects_negative_durations_when_present() -> None:
    with pytest.raises(ValueError):
        CheckpointPublished(
            checkpoint_dir="step_000001",
            content_id="0" * 64,
            file_count=1,
            payload_bytes=10,
            metadata_bytes=5,
            total_bytes=15,
            write_duration_sec=-1.0,
            publish_duration_sec=None,
        )
    with pytest.raises(ValueError):
        CheckpointPublished(
            checkpoint_dir="step_000001",
            content_id="0" * 64,
            file_count=1,
            payload_bytes=10,
            metadata_bytes=5,
            total_bytes=15,
            write_duration_sec=None,
            publish_duration_sec=-1.0,
        )


def test_k_receipt_owned_code_uses_no_reflection_or_string_selected_access() -> None:
    """Enforce K's recorded source invariant independently of behavior.

    This is deliberately a source-contract adversary: the K ruling names
    direct serializers and no reflection, so a passing behavioral suite alone
    cannot establish it.  The test reports every direct getattr/hasattr call,
    including calls inside decorated definitions.
    """

    source_path = Path(__file__).parents[3] / "tpen" / "checkpoint" / "receipt.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    violations = [
        f"{source_path}:{call.lineno}:{call.col_offset}: {call.func.id}"
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in {"getattr", "hasattr"}
    ]
    assert violations == [], "reflection in K receipt-owned code: " + ", ".join(violations)


def test_has_publication_receipt_true_only_for_a_matching_valid_row(tmp_path: Path) -> None:
    checkpoint_dir, files = _write_checkpoint(tmp_path)
    ref = _ref_for(checkpoint_dir)
    path = tmp_path / "publication_receipts.jsonl"

    assert has_publication_receipt(path, ref.content_id) is False

    record_publication_receipt(
        ref, checkpoint_dir, files, path, write_duration_sec=1.0, publish_duration_sec=0.5
    )

    assert has_publication_receipt(path, ref.content_id) is True
    assert has_publication_receipt(path, "0" * 64) is False


def test_iter_valid_publication_receipts_skips_a_malformed_row_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A truncated/partial line -- the shape an interrupted append leaves -- is
    skipped, never raised, but is NOT silently discarded either: a WARNING
    naming the file and line number is logged. Both halves are required --
    a reader that silently drops corrupt data is the same defect as a writer
    that silently swallows a failure. Deliberately asymmetric with
    ``CheckpointCatalog.iter_publications``, which raises on a malformed row.
    """

    path = tmp_path / "publication_receipts.jsonl"
    path.write_text('{"schema": "tpen.checkpoint-publication-receipt/v1", "summary": {"content', encoding="utf-8")

    with caplog.at_level("WARNING", logger="tpen"):
        result = list(iter_valid_publication_receipts(path))

    assert result == []
    assert has_publication_receipt(path, "anything") is False
    assert any(
        record.levelname == "WARNING" and str(path) in record.getMessage() and "1" in record.getMessage()
        for record in caplog.records
    ), f"expected a WARNING naming {path} and line 1; got {[r.getMessage() for r in caplog.records]}"


def test_append_publication_receipt_does_not_join_onto_an_unterminated_last_line(
    tmp_path: Path,
) -> None:
    """A new append must land on its own line even if the prior line is unterminated.

    ``tpen.artifacts.append_jsonl`` now delegates to
    ``tpen.durable_append.append_record``, which opens an UNBUFFERED BINARY
    handle and issues one write carrying the record, its trailing newline, and
    a leading repair newline when the file's last byte is not already one. If
    the file's last byte is not a newline (the shape left by a body written but
    its trailing newline lost to a mid-write failure), a naive append would
    concatenate onto that line, corrupting the NEW row along with the old one
    -- which would defeat
    ``backfill_publication_receipt``'s entire purpose of recovering telemetry.
    """

    checkpoint_dir, files = _write_checkpoint(tmp_path)
    ref = _ref_for(checkpoint_dir)
    path = tmp_path / "publication_receipts.jsonl"
    path.write_text('{"unterminated": "row", "no_newline": true}', encoding="utf-8")
    assert not path.read_bytes().endswith(b"\n")

    receipt = build_publication_receipt(
        ref, checkpoint_dir, files, write_duration_sec=1.0, publish_duration_sec=0.5
    )
    append_publication_receipt(path, receipt)

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    # The first (originally unterminated) row is untouched and now parses on
    # its own line -- it was not corrupted by the new append joining onto it.
    assert json.loads(rows[0]) == {"unterminated": "row", "no_newline": True}
    # The new row parses independently and is the one just appended.
    appended = json.loads(rows[1])
    assert appended["summary"]["content_id"] == ref.content_id


def test_backfill_publication_receipt_treats_a_malformed_existing_row_as_absent(
    tmp_path: Path,
) -> None:
    """The presence predicate is 'no VALID row', not file existence or 'any row'.

    A checkpoint whose only receipt row is truncated must still be backfilled.
    """

    checkpoint_dir, files = _write_checkpoint(tmp_path)
    ref = _ref_for(checkpoint_dir)
    path = tmp_path / "publication_receipts.jsonl"
    # Simulate exactly the partial-write failure mode iter_valid_publication_receipts'
    # docstring describes: an OSError mid-write leaves a TRUNCATED, invalid-JSON
    # line -- not a complete, merely-unterminated one. A complete JSON body
    # missing only its trailing newline is still valid JSON and must NOT be
    # treated as malformed (json.loads does not require a trailing newline),
    # so the cut has to land inside the JSON body itself to be a genuine test
    # of the malformed-row path rather than an accidental valid row.
    full_row = json.dumps({"schema": PUBLICATION_RECEIPT_SCHEMA, "summary": {"content_id": ref.content_id}})
    path.write_text(full_row[: len(full_row) // 2], encoding="utf-8")
    assert not full_row[: len(full_row) // 2].strip().endswith("}")

    appended = backfill_publication_receipt(ref, checkpoint_dir, files, path)

    assert appended is True
    assert has_publication_receipt(path, ref.content_id) is True
    rows = path.read_text(encoding="utf-8").splitlines()
    # The malformed line is untouched (append-only); a second, valid line was added.
    assert len(rows) == 2
    backfilled = json.loads(rows[1])
    assert backfilled["summary"]["content_id"] == ref.content_id
    assert backfilled["summary"]["write_duration_sec"] is None
    assert backfilled["summary"]["publish_duration_sec"] is None


def test_reconcile_backfills_a_hollow_parsed_receipt_row(tmp_path: Path) -> None:
    """A parsed row without measured sizes does not suppress backfill.

    Normal receipt writes and prefix truncation cannot produce a closed JSON
    row of this shape. This is a narrow guard for externally introduced rows,
    not validation of arbitrary receipt content.
    """

    from tests.unit.callback.test_checkpoint import _write_checkpoint as _save_real_checkpoint
    from tpen.checkpoint.catalog import reconcile_publication

    final_dir = _save_real_checkpoint(tmp_path)
    root = tmp_path / "checkpoints"
    ref = _ref_for(final_dir)
    receipt_path = publication_receipt_path(root)
    receipt_path.write_text(
        json.dumps(
            {
                "schema": PUBLICATION_RECEIPT_SCHEMA,
                "summary": {"content_id": ref.content_id},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reconcile_publication(root, final_dir)

    rows = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2, "a hollow row must not suppress measured backfill"
    measured = rows[-1]
    assert measured["summary"]["content_id"] == ref.content_id
    assert measured["summary"]["total_bytes"] > 0
    assert measured["summary"]["write_duration_sec"] is None
    assert measured["summary"]["publish_duration_sec"] is None
    assert measured["files"]


def _complete_measured_receipt_row(content_id: str) -> dict[str, object]:
    """Return a presence-complete receipt row for reconciliation tests."""

    return {
        "schema": PUBLICATION_RECEIPT_SCHEMA,
        "summary": {
            "content_id": content_id,
            "file_count": 5,
            "payload_bytes": 10,
            "metadata_bytes": 5,
            "total_bytes": 15,
        },
        "files": [{"component": "model", "relative_path": "model.pt", "size_bytes": 10}],
    }


@pytest.mark.parametrize(
    "missing",
    ("file_count", "payload_bytes", "metadata_bytes", "total_bytes", "files"),
)
def test_hollow_receipt_missing_one_measured_field_is_invalid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, missing: str
) -> None:
    """Each required accounting field independently prevents receipt presence."""

    content_id = "c" * 64
    record = _complete_measured_receipt_row(content_id)
    if missing == "files":
        del record["files"]
    else:
        summary = record["summary"]
        assert isinstance(summary, dict)
        del summary[missing]
    path = tmp_path / "publication_receipts.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with caplog.at_level("WARNING", logger="tpen"):
        assert has_publication_receipt(path, content_id) is False
        assert list(iter_valid_publication_receipts(path)) == []

    assert any(
        str(path) in entry.getMessage() and ":1" in entry.getMessage()
        for entry in caplog.records
    )


def test_presence_only_receipt_validation_accepts_present_null_fields(tmp_path: Path) -> None:
    """The F2 boundary checks field presence without validating their values."""

    content_id = "c" * 64
    record = _complete_measured_receipt_row(content_id)
    record["files"] = None
    record["summary"] = {
        "content_id": content_id,
        "file_count": None,
        "payload_bytes": None,
        "metadata_bytes": None,
        "total_bytes": None,
    }
    path = tmp_path / "publication_receipts.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert has_publication_receipt(path, content_id) is True
    assert list(iter_valid_publication_receipts(path)) == [record]


def test_complete_matching_row_follows_hollow_and_unrelated_rows(tmp_path: Path) -> None:
    """Only a complete row for the requested content identity counts."""

    content_id = "c" * 64
    hollow = _complete_measured_receipt_row(content_id)
    hollow_summary = hollow["summary"]
    assert isinstance(hollow_summary, dict)
    del hollow_summary["total_bytes"]
    unrelated = _complete_measured_receipt_row("d" * 64)
    complete = _complete_measured_receipt_row(content_id)
    path = tmp_path / "publication_receipts.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (hollow, unrelated)) + "\n",
        encoding="utf-8",
    )

    assert has_publication_receipt(path, content_id) is False
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(complete) + "\n")

    assert list(iter_valid_publication_receipts(path)) == [unrelated, complete]
    assert has_publication_receipt(path, content_id) is True


def test_backfill_publication_receipt_is_a_noop_when_a_valid_row_already_exists(
    tmp_path: Path,
) -> None:
    checkpoint_dir, files = _write_checkpoint(tmp_path)
    ref = _ref_for(checkpoint_dir)
    path = tmp_path / "publication_receipts.jsonl"
    record_publication_receipt(
        ref, checkpoint_dir, files, path, write_duration_sec=1.0, publish_duration_sec=0.5
    )
    before = path.read_bytes()

    appended = backfill_publication_receipt(ref, checkpoint_dir, files, path)

    assert appended is False
    assert path.read_bytes() == before


def test_record_publication_receipt_does_not_catch_a_non_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catch is narrowly OSError; any other exception must still propagate."""

    import tpen.checkpoint.receipt as receipt_module

    checkpoint_dir, files = _write_checkpoint(tmp_path)
    ref = _ref_for(checkpoint_dir)
    path = tmp_path / "publication_receipts.jsonl"

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("not an OSError")

    monkeypatch.setattr(receipt_module, "append_publication_receipt", _boom)

    with pytest.raises(RuntimeError, match="not an OSError"):
        record_publication_receipt(
            ref, checkpoint_dir, files, path, write_duration_sec=1.0, publish_duration_sec=0.5
        )


def test_reconcile_publication_backfills_receipt_after_a_failed_append(tmp_path: Path) -> None:
    from tests.unit.callback.test_checkpoint import _write_checkpoint as _save_real_checkpoint
    from tpen.checkpoint.catalog import reconcile_publication

    root = tmp_path / "checkpoints"
    root.mkdir(parents=True)
    receipt_path = publication_receipt_path(root)
    receipt_path.mkdir()

    final_dir = _save_real_checkpoint(tmp_path)

    assert receipt_path.is_dir()
    assert list(receipt_path.iterdir()) == []

    receipt_path.rmdir()  # clear the induced failure so the backfill can write

    reconcile_publication(root, final_dir)

    assert receipt_path.is_file()
    rows = [json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["summary"]["write_duration_sec"] is None
    assert rows[0]["summary"]["publish_duration_sec"] is None
    assert rows[0]["summary"]["total_bytes"] > 0


def test_reconcile_publication_backfill_is_idempotent_across_repeated_calls(
    tmp_path: Path,
) -> None:
    from tests.unit.callback.test_checkpoint import _write_checkpoint as _save_real_checkpoint
    from tpen.checkpoint.catalog import reconcile_publication

    root = tmp_path / "checkpoints"
    final_dir = _save_real_checkpoint(tmp_path)
    receipt_path = publication_receipt_path(root)
    receipt_path.unlink()  # simulate a receipt that was never durably recorded

    reconcile_publication(root, final_dir)
    rows_after_first = receipt_path.read_text(encoding="utf-8").splitlines()
    assert len(rows_after_first) == 1

    reconcile_publication(root, final_dir)
    rows_after_second = receipt_path.read_text(encoding="utf-8").splitlines()
    assert rows_after_second == rows_after_first


def test_reconcile_older_target_backfills_catalog_and_receipt_without_regressing_newer_latest(
    tmp_path: Path,
) -> None:
    older_dir, older_files = _write_checkpoint(tmp_path, 7)
    newer_dir, _ = _write_checkpoint(tmp_path, 9)
    root = tmp_path

    reconcile_publication(root, newer_dir)
    latest_path = root / "latest.json"
    newer_latest_bytes = latest_path.read_bytes()
    catalog_path = publication_catalog_path(root)
    receipt_path = publication_receipt_path(root)
    older_ref = _ref_for(older_dir)
    newer_ref = _ref_for(newer_dir)
    older_files_before = {
        path: path.read_bytes()
        for relative in (*older_files.values(), "manifest.json", COMPLETE_MARKER)
        for path in (older_dir / relative,)
    }

    reconcile_publication(root, older_dir)

    assert read_publications(catalog_path) == (newer_ref, older_ref)
    assert has_publication_receipt(receipt_path, older_ref.content_id) is True
    assert latest_path.read_bytes() == newer_latest_bytes
    catalog_after_first = catalog_path.read_bytes()
    receipt_after_first = receipt_path.read_bytes()

    reconcile_publication(root, older_dir)

    assert catalog_path.read_bytes() == catalog_after_first
    assert receipt_path.read_bytes() == receipt_after_first
    assert latest_path.read_bytes() == newer_latest_bytes
    assert all(path.read_bytes() == content for path, content in older_files_before.items())


def test_reconcile_receipt_failure_preserves_catalog_latest_and_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    older_dir, older_files = _write_checkpoint(tmp_path, 7)
    newer_dir, _ = _write_checkpoint(tmp_path, 9)
    reconcile_publication(tmp_path, newer_dir)
    latest_path = tmp_path / "latest.json"
    newer_latest_bytes = latest_path.read_bytes()
    catalog_path = publication_catalog_path(tmp_path)
    receipt_path = publication_receipt_path(tmp_path)
    older_ref = _ref_for(older_dir)
    older_artifacts = {
        older_dir / relative: (older_dir / relative).read_bytes()
        for relative in (*older_files.values(), "manifest.json", COMPLETE_MARKER)
    }

    import tpen.checkpoint.receipt as receipt_module

    append_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fail_append(*args: object, **kwargs: object) -> None:
        append_calls.append((args, kwargs))
        raise OSError("receipt append failed")

    monkeypatch.setattr(receipt_module, "append_publication_receipt", fail_append)

    reconcile_publication(tmp_path, older_dir)

    assert len(append_calls) == 1
    assert older_ref in read_publications(catalog_path)
    assert latest_path.read_bytes() == newer_latest_bytes
    assert not has_publication_receipt(receipt_path, older_ref.content_id)
    assert all(path.read_bytes() == content for path, content in older_artifacts.items())
