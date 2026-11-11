"""Torch-free tests for checkpoint identity and publication records."""

from __future__ import annotations

import json
from pathlib import Path

from tpen.checkpoint.catalog import CheckpointCatalog
from tpen.checkpoint.reference import CheckpointRef


def _manifest(step: int, *, model_name: str = "model.pt") -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": step,
        "completed_updates": step - 1,
        "created_at_unix": 123.0,
        "files": {"model": model_name},
        "hashes": {},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "run", "git_sha": "deadbeef"},
    }


def _write_checkpoint(root: Path, step: int = 7, *, model_name: str = "model.pt") -> Path:
    checkpoint_dir = root / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / model_name).write_bytes(b"immutable-model")
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(_manifest(step, model_name=model_name), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return checkpoint_dir


def _expect_raises(
    expected: type[BaseException] | tuple[type[BaseException], ...],
    function,
    *args,
    **kwargs,
) -> None:
    try:
        function(*args, **kwargs)
    except expected:
        return
    expected_names = (
        ", ".join(exc.__name__ for exc in expected)
        if isinstance(expected, tuple)
        else expected.__name__
    )
    raise AssertionError(f"expected {expected_names} from {function.__name__}")


def test_ref_json_round_trip_and_provenance_is_immutable(tmp_path: Path) -> None:
    ref = CheckpointRef.from_directory(_write_checkpoint(tmp_path))

    round_trip = CheckpointRef.from_mapping(ref.to_dict())

    assert round_trip == ref
    assert round_trip.content_id == ref.content_id

    def mutate_provenance() -> None:
        ref.provenance["run_id"] = "changed"  # type: ignore[index]

    _expect_raises(TypeError, mutate_provenance)
    _expect_raises(AttributeError, setattr, ref, "next_iteration", 8)


def test_content_identity_is_stable_when_checkpoint_is_moved(tmp_path: Path) -> None:
    first = CheckpointRef.from_directory(_write_checkpoint(tmp_path / "first"))
    second = CheckpointRef.from_directory(_write_checkpoint(tmp_path / "second"))

    assert first.checkpoint_dir != second.checkpoint_dir
    assert first.content_id == second.content_id
    assert first.identity == second.identity


def test_incomplete_tmp_and_latest_paths_are_rejected(tmp_path: Path) -> None:
    incomplete = tmp_path / "step_000007"
    incomplete.mkdir()
    _expect_raises(ValueError, CheckpointRef.from_directory, incomplete)

    temporary = tmp_path / "step_000007.tmp"
    temporary.mkdir()
    _expect_raises(ValueError, CheckpointRef.from_directory, temporary)

    complete = _write_checkpoint(tmp_path / "complete")
    (complete.parent / "latest.json").write_text(
        json.dumps({"checkpoint_dir": complete.name}), encoding="utf-8"
    )
    _expect_raises(ValueError, CheckpointRef.from_directory, complete.parent)
    _expect_raises(
        (ValueError, FileNotFoundError),
        CheckpointRef.from_directory,
        complete.parent / "latest.json",
    )


def test_path_and_manifest_step_disagreement_fails_loudly(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path, step=7)
    manifest_path = checkpoint_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["next_iteration"] = 8
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    _expect_raises(ValueError, CheckpointRef.from_directory, checkpoint_dir)


def test_manifest_model_binding_and_digest_tampering_fail_loudly(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path)
    ref = CheckpointRef.from_directory(checkpoint_dir)

    (checkpoint_dir / "model.pt").write_bytes(b"tampered-model")
    _expect_raises(ValueError, ref.validate)

    manifest_path = checkpoint_dir / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["files"]["model"] = "other.pt"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    (checkpoint_dir / "other.pt").write_bytes(b"tampered-model")
    _expect_raises(ValueError, CheckpointRef.from_directory, checkpoint_dir)


def test_catalog_is_append_only_and_has_no_lifecycle_state(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    first = CheckpointRef.from_directory(_write_checkpoint(root, step=7))
    second = CheckpointRef.from_directory(_write_checkpoint(root, step=8))
    catalog_path = tmp_path / "publications.jsonl"
    catalog = CheckpointCatalog(catalog_path)

    catalog.publish(first)
    first_bytes = catalog_path.read_bytes()
    catalog.publish(second)

    rows = catalog_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].encode() + b"\n" == first_bytes
    assert [ref.content_id for ref in catalog.records()] == [
        first.content_id,
        second.content_id,
    ]
    record = json.loads(rows[0])
    serialized = json.dumps(record, sort_keys=True)
    for lifecycle_key in ("status", "latest", "pinned", "deleted", "retention"):
        assert lifecycle_key not in record
        assert lifecycle_key not in serialized


def test_catalog_republishing_same_identity_is_idempotent(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path / "checkpoints")
    ref = CheckpointRef.from_directory(checkpoint_dir)
    catalog_path = tmp_path / "publications.jsonl"
    catalog = CheckpointCatalog(catalog_path)

    catalog.publish(ref)
    original_line = catalog_path.read_text(encoding="utf-8")
    record = json.loads(original_line)
    formatted_line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
    assert formatted_line != original_line
    assert json.loads(formatted_line) == record
    catalog_path.write_text(formatted_line, encoding="utf-8")
    before = catalog_path.read_bytes()

    # Parsed-equivalent JSON formatting is still the same publication.
    assert catalog.publish(ref) == ref
    assert catalog_path.read_bytes() == before
    assert catalog.records() == (ref,)


def test_catalog_rejects_conflicting_duplicate_identity(tmp_path: Path) -> None:
    first = CheckpointRef.from_directory(_write_checkpoint(tmp_path / "first"))
    second = CheckpointRef.from_directory(_write_checkpoint(tmp_path / "second"))
    assert second.content_id == first.content_id
    assert second.to_dict() != first.to_dict()

    catalog_path = tmp_path / "publications.jsonl"
    catalog = CheckpointCatalog(catalog_path)
    catalog.publish(first)
    before = catalog_path.read_bytes()

    _expect_raises(ValueError, catalog.publish, second)

    assert catalog_path.read_bytes() == before
    assert catalog.records() == (first,)


def test_catalog_rejects_republication_from_relocated_path(tmp_path: Path) -> None:
    original_dir = _write_checkpoint(tmp_path / "original")
    original_ref = CheckpointRef.from_directory(original_dir)
    catalog_path = tmp_path / "publications.jsonl"
    catalog = CheckpointCatalog(catalog_path)
    catalog.publish(original_ref)
    before = catalog_path.read_bytes()

    relocated_dir = tmp_path / "relocated" / original_dir.name
    relocated_dir.parent.mkdir()
    original_dir.rename(relocated_dir)
    relocated_ref = CheckpointRef.from_directory(relocated_dir)
    assert relocated_ref.content_id == original_ref.content_id

    _expect_raises(ValueError, catalog.publish, relocated_ref)

    assert catalog_path.read_bytes() == before
    assert catalog.records() == (original_ref,)


def test_catalog_rejects_multiline_jsonl_record(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path / "checkpoints")
    ref = CheckpointRef.from_directory(checkpoint_dir)
    catalog_path = tmp_path / "publications.jsonl"
    catalog = CheckpointCatalog(catalog_path)
    catalog.publish(ref)

    record = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    _expect_raises(ValueError, catalog.records)


def test_catalog_rejects_content_id_tampering(tmp_path: Path) -> None:
    checkpoint_dir = _write_checkpoint(tmp_path / "checkpoints")
    ref = CheckpointRef.from_directory(checkpoint_dir)
    catalog_path = tmp_path / "publications.jsonl"
    CheckpointCatalog(catalog_path).publish(ref)

    record = json.loads(catalog_path.read_text(encoding="utf-8"))
    record["ref"]["content_id"] = "0" * 64
    catalog_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    _expect_raises(ValueError, CheckpointCatalog(catalog_path).records)
