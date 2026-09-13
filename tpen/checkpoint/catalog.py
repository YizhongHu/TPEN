"""Append-only serialization and catalog for published checkpoint refs."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from tpen.artifacts import append_jsonl

from .artifact import LATEST_JSON, read_latest, resolve_checkpoint_dir, write_latest
from .receipt import backfill_publication_receipt, publication_receipt_path
from .hashing import file_sha256
from .reference import (
    CHECKPOINT_REF_SCHEMA,
    CheckpointRef,
    deserialize_checkpoint_ref,
    serialize_checkpoint_ref,
)
from .schema import read_manifest

PUBLICATION_CATALOG_FILENAME = "publications.jsonl"
PUBLICATION_RECORD_SCHEMA = "tpen.checkpoint-publication/v1"


class IncompletePublicationRecordError(ValueError):
    """A torn final row: unterminated, unparseable, and therefore recoverable.

    Subclasses :class:`ValueError` so that callers already written as
    ``except ValueError`` around catalog reads keep working unchanged; this
    narrows a diagnosis rather than introducing a new failure mode.

    Distinguished from a plain ``ValueError`` on a *terminated* malformed row,
    which signals corruption of committed content and has no automatic repair.
    A row this exception names was never committed -- the checkpoint directory
    it describes was renamed into place before the catalog append, so
    :func:`reconcile_publication` can rebuild the row from disk.
    """


class CheckpointCatalog:
    """Append-only JSONL catalog of immutable checkpoint publications.

    A record contains only the immutable checkpoint ref and its content ID.
    Lifecycle fields such as ``latest``, ``pinned``, ``deleted``, or ``status``
    intentionally do not exist here; those policies belong to later layers.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def publish(self, ref: CheckpointRef) -> CheckpointRef:
        """Publish ``ref`` while preserving append-only history.

        ``content_id`` is the publication identity.  A replay is idempotent
        when the parsed, canonical full CheckpointRef mapping is equal to the
        existing row; raw JSON formatting is intentionally irrelevant.  A
        same-content-id row with different serialized content is a conflict.
        Although ``content_id`` is path-independent, the compared mapping
        includes ``checkpoint_dir``.  A relocation therefore conflicts rather
        than silently creating two locations for one catalog identity: this
        append-only catalog cannot rewrite the original location, and accepting
        both would make the publication target ambiguous.  The duplicate check
        scans the append-only JSONL catalog, so its read cost is O(n) in the
        number of existing publications.
        """

        if not isinstance(ref, CheckpointRef):
            raise TypeError(f"expected CheckpointRef, got {type(ref).__name__}")
        ref.validate()
        serialized_ref = serialize_checkpoint_ref(ref)
        for existing in self.iter_publications():
            if existing.content_id != ref.content_id:
                continue
            if existing.to_dict() == serialized_ref:
                return ref
            raise ValueError(
                "conflicting checkpoint publication for content_id "
                f"{ref.content_id}"
            )
        append_jsonl(self.path, {"schema": PUBLICATION_RECORD_SCHEMA, "ref": serialized_ref})
        return ref

    append = publish

    def iter_publications(self) -> Iterator[CheckpointRef]:
        """Yield serialized refs in append order, rejecting malformed rows.

        Stays fail-loud.  Losing a published checkpoint's identity row silently
        is worse than refusing to read the catalog, so no row is ever skipped.
        What this does add is a *diagnosis*: a row that is malformed because it
        was torn mid-write is distinguishable from a row that is malformed
        because the catalog is corrupt, and only the first is repairable.

        A newline is what commits a record, so only the file's final line can
        lack one.  An unterminated final line that fails to parse was therefore
        torn mid-write, and raises
        :class:`IncompletePublicationRecordError`, whose message spells out the
        repair.  Any *terminated* row that fails to parse is ordinary corruption
        and raises :class:`ValueError` exactly as before.

        Note that this method is also on the *write* path: ``publish`` scans for
        duplicates before appending, so a torn catalog blocks new publications
        until it is repaired.  That is the intended fail-loud behaviour, and it
        is why the repair is an operator action rather than something
        ``reconcile_publication`` can do for itself.

        An unterminated final line that parses and validates is yielded, not
        rejected.  It is a complete record that merely lost its terminator --
        rejecting it would refuse to read a catalog that is entirely intact,
        which is the failure mode a stricter rule would hide until a real
        restore could not start.  The next append closes the line out rather
        than joining onto it.  A row that parses but fails schema validation is
        a different diagnosis and is left to ``_deserialize_record``.
        """

        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                # Read the terminator before stripping it off the line.  Note this
                # is universal-newline text mode, so a bare CR counts as a line
                # ending here while ``durable_append.ends_without_newline`` reads
                # bytes and would call the same tail unterminated.  Unreachable
                # from TPEN records -- ``json.dumps`` escapes CR as ``\\r`` -- so
                # no record's bytes can end in a literal CR.
                terminated = line.endswith("\n")
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if not terminated:
                        raise IncompletePublicationRecordError(
                            f"incomplete checkpoint publication at {self.path}:"
                            f"{line_number}: {exc}. This is the file's last line and it "
                            "has no terminating newline, so the record was torn mid-write "
                            "and never committed; every earlier row is intact. To repair: "
                            "drop this unterminated final line, which holds no committed "
                            "record, then re-run tpen.checkpoint.catalog.reconcile_publication "
                            "on the NEWEST complete step_* directory under the checkpoint root "
                            "that now has no catalog row -- a tear is always on the last "
                            "append. Start with the NEWEST complete directory missing from "
                            "the catalog; reconciliation orders latest.json by the validated "
                            "manifest step, so retrying an older complete directory is safe "
                            "for the latest pointer. Dropping the line is an operator action "
                            "by design -- "
                            "reconcile_publication reads the catalog before it writes, so it "
                            "cannot clear this itself, and TPEN does not truncate a "
                            "load-bearing file on its own."
                        ) from exc
                    raise ValueError(
                        f"invalid checkpoint publication at {self.path}:{line_number}: {exc}"
                    ) from exc
                yield _deserialize_record(record, path=self.path, line_number=line_number)

    def records(self) -> tuple[CheckpointRef, ...]:
        """Return all catalog refs in append order."""

        return tuple(self.iter_publications())

    read = records
    load = records


PublicationCatalog = CheckpointCatalog


def publication_catalog_path(checkpoint_root: str | Path) -> Path:
    """Return the default append-only publication catalog path."""

    return Path(checkpoint_root) / PUBLICATION_CATALOG_FILENAME


def append_publication(path: str | Path, ref: CheckpointRef) -> CheckpointRef:
    """Append one validated publication to a catalog path."""

    return CheckpointCatalog(path).publish(ref)


def read_publications(path: str | Path) -> tuple[CheckpointRef, ...]:
    """Read all publication refs from a catalog path."""

    return CheckpointCatalog(path).records()


def reconcile_publication(
    checkpoint_root: str | Path, checkpoint_dir: str | Path
) -> CheckpointRef:
    """Ensure one committed checkpoint has its catalog row, latest pointer, and receipt.

    The directory rename is the checkpoint commit, while catalog publication,
    ``latest.json``, and the publication receipt are separate durable
    operations.  A retry therefore needs to reconcile all three instead of
    treating a complete directory as proof that all of them succeeded.
    ``CheckpointCatalog.publish`` supplies the append-idempotent row
    operation: an existing identical ref is not appended again, while a
    conflicting row still fails closed.

    This helper only repairs non-destructive indexes. It must not rewrite the
    committed payload or remove any checkpoint directories; storage disposition
    is outside TPEN.

    The receipt backfill is last and best-effort, matching
    ``save_checkpoint``'s own ordering and failure handling: unlike the
    catalog row and ``latest.json`` above, a missing or unrecoverable receipt
    never fails this call. See
    ``tpen.checkpoint.receipt.backfill_publication_receipt``.
    """

    root = Path(checkpoint_root)
    directory = Path(checkpoint_dir)
    ref = CheckpointRef.from_directory(directory)
    CheckpointCatalog(publication_catalog_path(root)).publish(ref)

    manifest = read_manifest(directory / "manifest.json", mode="model_only")
    latest_state = _validated_latest_target(root)
    if latest_state is None:
        write_latest(
            root,
            directory,
            step=ref.next_iteration,
            created_at_unix=manifest.created_at_unix,
        )
    else:
        latest_target, latest_pointer = latest_state
        if latest_target.next_iteration < ref.next_iteration or (
            latest_target.next_iteration == ref.next_iteration
            and latest_pointer
            != {
                "checkpoint_dir": directory.name,
                "step": ref.next_iteration,
                "created_at_unix": manifest.created_at_unix,
            }
        ):
            write_latest(
                root,
                directory,
                step=ref.next_iteration,
                created_at_unix=manifest.created_at_unix,
            )
    backfill_publication_receipt(
        ref, directory, manifest.files, publication_receipt_path(root)
    )
    return ref


def _validated_latest_target(
    checkpoint_root: Path,
) -> tuple[CheckpointRef, dict[str, Any]] | None:
    """Return the validated checkpoint and raw pointer, if usable.

    ``latest.json`` is a convenience pointer: its ``step`` and timestamp are
    not ordering authorities.  The existing resolver deliberately accepts the
    same absolute and relative target representations as restore, while the
    checkpoint reference validates the complete directory and derives the
    authoritative step from its manifest.  A malformed pointer or target is
    treated as missing so reconciliation can repair it to the committed
    checkpoint it was given.
    """

    try:
        pointer = read_latest(checkpoint_root)
        target_value = pointer.get("checkpoint_dir")
        if not isinstance(target_value, str) or not target_value:
            return None
        target = resolve_checkpoint_dir(checkpoint_root / LATEST_JSON)
        target_ref = CheckpointRef.from_directory(target)
        target_manifest = read_manifest(target / "manifest.json", mode="model_only")
        declared_model_digest = target_manifest.hashes.get("model_sha256")
        if declared_model_digest is not None:
            actual_model_digest = file_sha256(target / target_manifest.files["model"])
            if actual_model_digest != declared_model_digest:
                raise ValueError(
                    f"{target}: model checkpoint file digest mismatch "
                    f"(manifest {declared_model_digest}, actual {actual_model_digest})"
                )
        return target_ref, pointer
    except (FileNotFoundError, ValueError, KeyError, TypeError, OverflowError):
        return None


def _deserialize_record(record: Any, *, path: Path, line_number: int) -> CheckpointRef:
    if not isinstance(record, Mapping) or record.get("schema") != PUBLICATION_RECORD_SCHEMA:
        raise ValueError(
            f"unsupported checkpoint publication at {path}:{line_number}; "
            f"expected schema {PUBLICATION_RECORD_SCHEMA!r}"
        )
    ref_data = record.get("ref")
    if not isinstance(ref_data, Mapping) or ref_data.get("schema") != CHECKPOINT_REF_SCHEMA:
        raise ValueError(f"publication at {path}:{line_number} lacks a valid CheckpointRef")
    return deserialize_checkpoint_ref(ref_data)


__all__ = [
    "PUBLICATION_CATALOG_FILENAME",
    "PUBLICATION_RECORD_SCHEMA",
    "CheckpointCatalog",
    "IncompletePublicationRecordError",
    "PublicationCatalog",
    "append_publication",
    "publication_catalog_path",
    "reconcile_publication",
    "read_publications",
]
