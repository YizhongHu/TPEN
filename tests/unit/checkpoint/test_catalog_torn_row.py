"""Torch-free tests for how the publication catalog reads a torn row.

``publications.jsonl`` is load-bearing for restore, resume and reconcile, and
its reader is deliberately fail-loud: silently dropping a published checkpoint's
identity row is worse than refusing to read the file. These tests pin the reader
staying loud while gaining a *diagnosis* -- separating a row that was torn
mid-write, and is therefore repairable, from a row that is corrupt, and is not.

Parseability alone decides raise-versus-yield. The terminating newline selects
only WHICH exception is raised among rows that have already failed to parse, and
never causes a parseable row to be rejected. That matters in both directions: a
reader that skipped unparseable rows would lose a published checkpoint, and a
reader that rejected every unterminated line would refuse to read a catalog that
is entirely intact. Both directions are exercised below.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen.checkpoint.catalog import (
    CheckpointCatalog,
    IncompletePublicationRecordError,
    publication_catalog_path,
    reconcile_publication,
)
from tpen.checkpoint.artifact import read_latest
from tpen.checkpoint.receipt import publication_receipt_path
from tpen.checkpoint.reference import CheckpointRef


def _manifest(step: int) -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        "completed_updates": step - 1,
        "created_at_unix": 123.0,
        "files": {"model": "model.pt"},
        "hashes": {
            "model_config": "a" * 64,
            "hamiltonian_config": "b" * 64,
        },
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "run", "git_sha": "deadbeef"},
    }


def _write_checkpoint(root: Path, step: int = 7) -> Path:
    checkpoint_dir = root / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.pt").write_bytes(b"immutable-model")
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(_manifest(step), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return checkpoint_dir


def _catalog_with(tmp_path: Path, *, steps: tuple[int, ...] = (7,)) -> CheckpointCatalog:
    catalog = CheckpointCatalog(publication_catalog_path(tmp_path))
    for step in steps:
        catalog.publish(CheckpointRef.from_directory(_write_checkpoint(tmp_path, step)))
    return catalog


# --- the reader keeps refusing to read corruption ----------------------------


def test_a_terminated_malformed_row_still_raises(tmp_path: Path) -> None:
    """Fail-loud on committed corruption is the behaviour being preserved."""

    catalog = _catalog_with(tmp_path)
    with catalog.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all}\n")

    with pytest.raises(ValueError) as caught:
        catalog.records()

    assert not isinstance(caught.value, IncompletePublicationRecordError), (
        "a terminated row was committed; it must not be reported as repairable"
    )
    assert "invalid checkpoint publication" in str(caught.value)


def test_a_row_that_parses_but_has_the_wrong_schema_still_raises(tmp_path: Path) -> None:
    catalog = _catalog_with(tmp_path)
    with catalog.path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": "something.else/v1"}) + "\n")

    with pytest.raises(ValueError, match="unsupported checkpoint publication"):
        catalog.records()


# --- but it now distinguishes a torn final row -------------------------------


def test_an_unterminated_final_row_raises_the_recoverable_error(tmp_path: Path) -> None:
    catalog = _catalog_with(tmp_path)
    with catalog.path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema": "tpen.checkpoint-publica')  # torn mid-write

    with pytest.raises(IncompletePublicationRecordError) as caught:
        catalog.records()

    message = str(caught.value)
    assert "never committed" in message
    assert "reconcile_publication" in message, "the error must name the repair"
    assert "every earlier row is intact" in message
    assert "NEWEST complete step_* directory" in message
    assert "rewrites latest.json unconditionally" not in message
    assert "Do NOT reconcile older directories to be safe" not in message


def test_the_recoverable_error_is_still_a_value_error(tmp_path: Path) -> None:
    """Existing ``except ValueError`` callers must keep working unchanged."""

    assert issubclass(IncompletePublicationRecordError, ValueError)


# --- the over-restrictive direction ------------------------------------------


def test_an_unterminated_final_row_that_parses_is_yielded_not_rejected(
    tmp_path: Path,
) -> None:
    """A complete record that merely lost its terminator is not corruption.

    This is the mutation that closes too far: rejecting every unterminated line
    would pass every other test in this file and only surface when a real
    restore could not start on an intact catalog.
    """

    catalog = _catalog_with(tmp_path)
    expected = catalog.records()
    assert len(expected) == 1

    # Strip the final newline, leaving a complete but uncommitted row.
    text = catalog.path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    catalog.path.write_text(text[:-1], encoding="utf-8")

    assert catalog.records() == expected


def test_a_well_formed_catalog_reads_unchanged(tmp_path: Path) -> None:
    catalog = _catalog_with(tmp_path, steps=(7, 9))

    refs = catalog.records()

    assert [ref.next_iteration for ref in refs] == [7, 9]


def test_blank_lines_are_still_skipped(tmp_path: Path) -> None:
    catalog = _catalog_with(tmp_path)
    with catalog.path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")

    assert len(catalog.records()) == 1


# --- the falsifier, end to end -----------------------------------------------


def test_publishing_after_a_torn_row_never_merges_into_it(tmp_path: Path) -> None:
    """The core defect: an interrupted append destroying the NEXT record.

    ``publish`` reads before it writes, so the torn catalog blocks it -- which
    is the fail-loud behaviour. The bytes still must not merge when the row is
    appended through the shared primitive, so that is asserted directly.
    """

    from tpen.artifacts import append_jsonl

    catalog = _catalog_with(tmp_path)
    with catalog.path.open("a", encoding="utf-8") as handle:
        handle.write('{"schema": "tpen.checkpoint-publica')

    # A torn catalog blocks new publications rather than corrupting them.
    with pytest.raises(IncompletePublicationRecordError):
        catalog.publish(CheckpointRef.from_directory(_write_checkpoint(tmp_path, 9)))

    append_jsonl(catalog.path, {"schema": "tpen.checkpoint-publication/v1"})

    lines = catalog.path.read_text(encoding="utf-8").splitlines()
    assert lines[-2] == '{"schema": "tpen.checkpoint-publica'
    assert json.loads(lines[-1]) == {"schema": "tpen.checkpoint-publication/v1"}, (
        "the record written after the torn row must survive intact"
    )


def test_the_repair_named_in_the_error_message_actually_works(tmp_path: Path) -> None:
    """Follow the recipe the exception prints, and the catalog must come back.

    Written because the first draft of that message named
    ``reconcile_publication`` alone, which cannot work: it reads the catalog
    before it writes, so it raises on the very file it was meant to repair.
    """

    checkpoint_dir = _write_checkpoint(tmp_path, 7)
    catalog = CheckpointCatalog(publication_catalog_path(tmp_path))
    catalog.publish(CheckpointRef.from_directory(checkpoint_dir))
    expected = catalog.records()

    # Lose the catalog row to a torn append.
    text = catalog.path.read_text(encoding="utf-8")
    catalog.path.write_text(text + '{"schema": "tpen.checkpoint-publica', encoding="utf-8")
    with pytest.raises(IncompletePublicationRecordError):
        catalog.records()

    # Step 1 of the recipe: drop the unterminated final line.
    body = catalog.path.read_text(encoding="utf-8")
    catalog.path.write_text(body[: body.rindex("\n") + 1], encoding="utf-8")

    # Step 2: reconcile rebuilds any missing row from the committed checkpoint.
    reconcile_publication(tmp_path, checkpoint_dir)

    assert catalog.records() == expected


def test_reconcile_rebuilds_a_row_lost_entirely_to_a_torn_append(tmp_path: Path) -> None:
    """The recoverability that makes this error class 'recoverable' at all."""

    checkpoint_dir = _write_checkpoint(tmp_path, 7)
    catalog = CheckpointCatalog(publication_catalog_path(tmp_path))

    reconcile_publication(tmp_path, checkpoint_dir)

    refs = catalog.records()
    assert [ref.next_iteration for ref in refs] == [7]


def test_reconcile_older_checkpoint_preserves_newer_latest_target(tmp_path: Path) -> None:
    older = _write_checkpoint(tmp_path, 7)
    newer = _write_checkpoint(tmp_path, 9)

    reconcile_publication(tmp_path, newer)
    latest_path = tmp_path / "latest.json"
    newer_bytes = latest_path.read_bytes()

    reconcile_publication(tmp_path, older)

    assert latest_path.read_bytes() == newer_bytes


def test_reconcile_arbitrary_order_preserves_each_prefix_maximum(tmp_path: Path) -> None:
    checkpoints = {step: _write_checkpoint(tmp_path, step) for step in (7, 8, 9)}
    observed: list[str] = []
    expected: list[str] = []
    cursors: list[int] = []

    for step in (9, 7, 8):
        reconcile_publication(tmp_path, checkpoints[step])
        cursors.append(step)
        observed.append(read_latest(tmp_path)["checkpoint_dir"])
        expected.append(checkpoints[max(cursors)].name)

    assert observed == expected


def test_reconcile_newer_checkpoint_advances_latest(tmp_path: Path) -> None:
    older = _write_checkpoint(tmp_path, 7)
    newer = _write_checkpoint(tmp_path, 9)

    reconcile_publication(tmp_path, older)
    reconcile_publication(tmp_path, newer)

    assert read_latest(tmp_path) == {
        "checkpoint_dir": newer.name,
        "step": 9,
        "created_at_unix": 123.0,
    }


def test_reconcile_propagates_latest_read_failure_without_regressing_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    older = _write_checkpoint(tmp_path, 7)
    newer = _write_checkpoint(tmp_path, 9)
    reconcile_publication(tmp_path, newer)
    latest_path = tmp_path / "latest.json"
    before_latest = latest_path.read_bytes()

    import tpen.checkpoint.catalog as catalog_module

    def fail_read_latest(*args: object, **kwargs: object) -> dict[str, object]:
        raise PermissionError("latest read failed")

    monkeypatch.setattr(catalog_module, "read_latest", fail_read_latest)

    with pytest.raises(PermissionError, match="latest read failed"):
        reconcile_publication(tmp_path, older)

    assert latest_path.read_bytes() == before_latest


@pytest.mark.parametrize("representation", ["relative", "absolute", "parent_relative"])
def test_reconcile_preserves_bytes_for_absolute_and_relative_newer_pointer_targets(
    tmp_path: Path, representation: str
) -> None:
    older = _write_checkpoint(tmp_path, 7)
    if representation == "parent_relative":
        other_root = tmp_path.parent / f"{tmp_path.name}-other-checkpoints"
        other_root.mkdir()
        newer = _write_checkpoint(other_root, 9)
        pointer_value = f"../{other_root.name}/{newer.name}"
    else:
        newer = _write_checkpoint(tmp_path, 9)
        pointer_value = str(newer.resolve()) if representation == "absolute" else newer.name
    latest_path = tmp_path / "latest.json"
    latest_path.write_text(
        json.dumps(
            {
                "checkpoint_dir": pointer_value,
                "step": "not-a-number",
                "created_at_unix": "stale",
            }
        ),
        encoding="utf-8",
    )
    before = latest_path.read_bytes()

    reconcile_publication(tmp_path, older)

    assert latest_path.read_bytes() == before


def test_reconcile_equal_target_canonicalizes_and_repeats_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _write_checkpoint(tmp_path, 7)
    latest_path = tmp_path / "latest.json"
    latest_path.write_text(
        json.dumps(
            {
                "checkpoint_dir": checkpoint.name,
                "step": "stale",
                "created_at_unix": "stale",
            }
        ),
        encoding="utf-8",
    )

    expected = {
        "checkpoint_dir": checkpoint.name,
        "step": 7,
        "created_at_unix": 123.0,
    }
    import tpen.checkpoint.catalog as catalog_module

    write_calls: list[object] = []
    original_write_latest = catalog_module.write_latest

    def record_write_latest(*args: object, **kwargs: object) -> None:
        write_calls.append((args, kwargs))
        original_write_latest(*args, **kwargs)

    monkeypatch.setattr(catalog_module, "write_latest", record_write_latest)
    reconcile_publication(tmp_path, checkpoint)
    assert read_latest(tmp_path) == expected
    assert len(write_calls) == 1
    latest_after_first = latest_path.read_bytes()
    catalog_after_first = publication_catalog_path(tmp_path).read_bytes()
    receipt_path = publication_receipt_path(tmp_path)
    receipt_after_first = receipt_path.read_bytes()

    def fail_write_latest(*args: object, **kwargs: object) -> None:
        raise AssertionError("canonical reconciliation must not rewrite latest.json")

    monkeypatch.setattr(catalog_module, "write_latest", fail_write_latest)

    reconcile_publication(tmp_path, checkpoint)

    assert latest_path.read_bytes() == latest_after_first
    assert len(write_calls) == 1
    assert publication_catalog_path(tmp_path).read_bytes() == catalog_after_first
    assert receipt_path.read_bytes() == receipt_after_first


@pytest.mark.parametrize(
    "invalid_pointer",
    (
        "missing",
        "malformed_json",
        "missing_checkpoint_dir",
        "dangling",
        "incomplete",
        "malformed_manifest",
    ),
)
def test_reconcile_repairs_invalid_existing_target_and_rejects_invalid_candidate(
    tmp_path: Path, invalid_pointer: str
) -> None:
    candidate = _write_checkpoint(tmp_path, 7)
    target = _write_checkpoint(tmp_path, 9)
    candidate_manifest = json.loads(
        (candidate / "manifest.json").read_text(encoding="utf-8")
    )
    candidate_paths = tuple(
        candidate / name
        for name in (*candidate_manifest["files"].values(), "manifest.json", "COMPLETE")
    )
    candidate_bytes = {
        path: path.read_bytes()
        for path in candidate_paths
    }
    latest_path = tmp_path / "latest.json"
    if invalid_pointer == "missing":
        pass
    elif invalid_pointer == "malformed_json":
        latest_path.write_text("{not json", encoding="utf-8")
    elif invalid_pointer == "missing_checkpoint_dir":
        latest_path.write_text(json.dumps({"step": 9}), encoding="utf-8")
    elif invalid_pointer == "dangling":
        latest_path.write_text(
            json.dumps({"checkpoint_dir": "step_000099"}), encoding="utf-8"
        )
    elif invalid_pointer == "incomplete":
        (tmp_path / "step_000010").mkdir()
        latest_path.write_text(
            json.dumps({"checkpoint_dir": "step_000010"}), encoding="utf-8"
        )
    else:
        (target / "manifest.json").write_text("{not json", encoding="utf-8")
        latest_path.write_text(json.dumps({"checkpoint_dir": target.name}), encoding="utf-8")

    reconcile_publication(tmp_path, candidate)

    assert read_latest(tmp_path) == {
        "checkpoint_dir": candidate.name,
        "step": 7,
        "created_at_unix": 123.0,
    }
    assert all(path.read_bytes() == content for path, content in candidate_bytes.items())


@pytest.mark.parametrize(
    "bad_step", ["not-a-number", True, -1, 1.5, float("nan"), float("inf"), float("-inf")]
)
def test_reconcile_malformed_numeric_step_on_nonnewer_target_repairs_candidate(
    tmp_path: Path, bad_step: object
) -> None:
    target = _write_checkpoint(tmp_path, 5)
    candidate = _write_checkpoint(tmp_path, 7)
    (tmp_path / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint_dir": target.name,
                "step": bad_step,
                "created_at_unix": 123.0,
            }
        ),
        encoding="utf-8",
    )

    reconcile_publication(tmp_path, candidate)

    assert read_latest(tmp_path) == {
        "checkpoint_dir": candidate.name,
        "step": 7,
        "created_at_unix": 123.0,
    }


@pytest.mark.parametrize(
    "invalid_candidate", ["missing_complete", "missing_manifest", "malformed_manifest", "path_mismatch"]
)
def test_reconcile_invalid_candidate_cases_fail_closed_without_mutation(
    tmp_path: Path, invalid_candidate: str
) -> None:
    newer = _write_checkpoint(tmp_path, 9)
    reconcile_publication(tmp_path, newer)
    latest_path = tmp_path / "latest.json"
    before_latest = latest_path.read_bytes()
    catalog_path = publication_catalog_path(tmp_path)
    before_catalog = catalog_path.read_bytes()
    receipt_path = tmp_path / "publication_receipts.jsonl"
    before_receipt = receipt_path.read_bytes()
    candidate = _write_checkpoint(tmp_path, 7)
    valid_manifest = json.loads(
        (candidate / "manifest.json").read_text(encoding="utf-8")
    )
    candidate_paths = tuple(
        candidate / name
        for name in (*valid_manifest["files"].values(), "manifest.json", "COMPLETE")
    )
    if invalid_candidate == "missing_complete":
        (candidate / "COMPLETE").unlink()
    elif invalid_candidate == "missing_manifest":
        (candidate / "manifest.json").unlink()
    elif invalid_candidate == "path_mismatch":
        manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        manifest["next_iteration"] = 8
        (candidate / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    else:
        (candidate / "manifest.json").write_text("{not json", encoding="utf-8")
    before_candidate = {
        path: path.read_bytes() if path.exists() else None for path in candidate_paths
    }

    with pytest.raises((FileNotFoundError, ValueError)):
        reconcile_publication(tmp_path, candidate)

    assert latest_path.read_bytes() == before_latest
    assert catalog_path.read_bytes() == before_catalog
    assert receipt_path.read_bytes() == before_receipt
    assert all(
        (path.read_bytes() if path.exists() else None) == content
        for path, content in before_candidate.items()
    )


@pytest.mark.parametrize(
    ("malformed_field", "malformed_value", "expected_exception"),
    (
        ("missing_created_at", None, KeyError),
        ("null_next_iteration", None, TypeError),
        ("infinite_next_iteration", float("inf"), OverflowError),
        ("null_files", None, TypeError),
    ),
)
def test_reconcile_malformed_candidate_variants_fail_closed(
    tmp_path: Path,
    malformed_field: str,
    malformed_value: object,
    expected_exception: type[Exception],
) -> None:
    newer = _write_checkpoint(tmp_path, 9)
    reconcile_publication(tmp_path, newer)
    candidate = _write_checkpoint(tmp_path, 7)
    newer_manifest = json.loads(
        (newer / "manifest.json").read_text(encoding="utf-8")
    )
    target_artifacts = {
        path: path.read_bytes()
        for path in (
            *(newer / name for name in newer_manifest["files"].values()),
            newer / "manifest.json",
            newer / "COMPLETE",
        )
    }
    candidate_manifest_path = candidate / "manifest.json"
    candidate_manifest_before = candidate_manifest_path.read_bytes()
    candidate_manifest = json.loads(candidate_manifest_before)
    candidate_paths = tuple(
        candidate / name
        for name in (*candidate_manifest["files"].values(), "manifest.json", "COMPLETE")
    )
    if malformed_field == "missing_created_at":
        del candidate_manifest["created_at_unix"]
    elif malformed_field == "null_next_iteration":
        candidate_manifest["next_iteration"] = malformed_value
    elif malformed_field == "infinite_next_iteration":
        candidate_manifest["next_iteration"] = malformed_value
    else:
        candidate_manifest["files"] = malformed_value
    candidate_manifest_path.write_text(json.dumps(candidate_manifest), encoding="utf-8")
    assert candidate_manifest_path.read_bytes() != candidate_manifest_before
    post_fault_candidate = {
        path: path.read_bytes() for path in candidate_paths if path.exists()
    }
    latest_path = tmp_path / "latest.json"
    catalog_path = publication_catalog_path(tmp_path)
    receipt_path = publication_receipt_path(tmp_path)
    before_latest = latest_path.read_bytes()
    before_catalog = catalog_path.read_bytes()
    before_receipt = receipt_path.read_bytes()

    with pytest.raises(expected_exception):
        reconcile_publication(tmp_path, candidate)

    assert latest_path.read_bytes() == before_latest
    assert catalog_path.read_bytes() == before_catalog
    assert receipt_path.read_bytes() == before_receipt
    assert all(path.read_bytes() == content for path, content in target_artifacts.items())
    assert all(path.read_bytes() == content for path, content in post_fault_candidate.items())


def test_reconcile_pointer_repair_preserves_all_candidate_artifacts(tmp_path: Path) -> None:
    candidate = _write_checkpoint(tmp_path, 7)
    (tmp_path / "latest.json").write_text("{not json", encoding="utf-8")
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    candidate_artifacts = {
        path: path.read_bytes()
        for path in (
            *(candidate / name for name in manifest["files"].values()),
            candidate / "manifest.json",
            candidate / "COMPLETE",
        )
    }

    reconcile_publication(tmp_path, candidate)

    assert all(path.read_bytes() == content for path, content in candidate_artifacts.items())


def test_reconcile_conflicting_catalog_identity_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    other_root = tmp_path / "other-checkpoints"
    other_root.mkdir()
    existing = _write_checkpoint(root, 7)
    conflicting_location = _write_checkpoint(other_root, 7)
    reconcile_publication(root, existing)
    latest_path = root / "latest.json"
    catalog_path = publication_catalog_path(root)
    receipt_path = publication_receipt_path(root)
    before_latest = latest_path.read_bytes()
    before_catalog = catalog_path.read_bytes()
    before_receipt = receipt_path.read_bytes()
    conflicting_manifest = json.loads(
        (conflicting_location / "manifest.json").read_text(encoding="utf-8")
    )
    candidate_paths = tuple(
        conflicting_location / name
        for name in (
            *conflicting_manifest["files"].values(),
            "manifest.json",
            "COMPLETE",
        )
    )
    before_candidate = {
        path: path.read_bytes()
        for path in candidate_paths
    }

    with pytest.raises(ValueError, match="conflicting checkpoint publication"):
        reconcile_publication(root, conflicting_location)

    assert latest_path.read_bytes() == before_latest
    assert catalog_path.read_bytes() == before_catalog
    assert receipt_path.read_bytes() == before_receipt
    assert all(path.read_bytes() == content for path, content in before_candidate.items())


def test_publishing_still_works_when_the_final_row_lost_only_its_terminator(
    tmp_path: Path,
) -> None:
    """Row 3 stated as its production consequence, not as a reader property.

    ``test_an_unterminated_final_row_that_parses_is_yielded_not_rejected`` was
    the ONLY assertion in the suite that died when the reader closed too far --
    the mutation that rejects every unterminated tail killed exactly one test.
    The most load-bearing judgement in the slice therefore had a single point of
    failure.  This pins the same rule from the other end: a run must still be
    able to PUBLISH, which is what an over-restrictive reader would actually
    break, so an edit to either test alone cannot silently unpin the rule.
    """

    catalog = CheckpointCatalog(publication_catalog_path(tmp_path))
    catalog.publish(CheckpointRef.from_directory(_write_checkpoint(tmp_path, 7)))

    text = catalog.path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    catalog.path.write_text(text[:-1], encoding="utf-8")

    catalog.publish(CheckpointRef.from_directory(_write_checkpoint(tmp_path, 9)))

    assert [ref.next_iteration for ref in catalog.records()] == [7, 9]
