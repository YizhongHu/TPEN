"""Typed publication size and duration receipts for one checkpoint write.

A publication receipt is a second, append-only index alongside
``publications.jsonl`` (see :mod:`tpen.checkpoint.catalog`). Where the
publication catalog carries identity, the receipt carries the bytes and
durations of the write: a durable size fact that exists independently of
whether the ``ResourceUsage`` callback is configured for the run.

**Sequencing, for a Stage-T3 reader building fault-injection tests from this
module alone.** ``save_checkpoint`` (``tpen/checkpoint/save.py``) writes every
component file into a ``.tmp`` staging directory, writes ``manifest.json`` and
``COMPLETE``, then commits with ``tmp_dir.rename(final_dir)``. Only after that
rename does it construct the :class:`~tpen.checkpoint.reference.CheckpointRef`,
publish it to the catalog, and update ``latest.json``. This module's receipt is
built and appended as the LAST step of that sequence, strictly after
``write_latest`` returns. It never runs before the rename (the files it
measures would not exist yet at their final names) and its own append does not
change, reorder, or replace any earlier step -- it is purely additive at the
end of the existing publication sequence.

**The manifest self-size cycle.** ``manifest.json`` cannot record its own byte
size before it is written. This module resolves that by never trying:
:class:`CheckpointManifest` carries no size field for itself, and no size data
of any kind is written into ``manifest.json`` or ``COMPLETE``. All sizes,
including the manifest's and ``COMPLETE``'s own, live only in this module's
separate ``publication_receipts.jsonl`` record, read via ``stat()`` after both
files are fully written and closed. Building the receipt never reopens
``manifest.json`` or ``COMPLETE`` for writing, so neither file is mutated by
this module.

**No directory scan.** :func:`measure_checkpoint_files` looks up exactly the
files the manifest's own ``files`` mapping names, plus the two files that are
always present and load-bearing enough to have named constants (``manifest``
and ``COMPLETE``). It never calls ``Path.iterdir``, ``Path.glob``,
``Path.rglob``, or ``os.walk``. A file present in the checkpoint directory but
not named by the manifest (there should never be one) is invisible to the
receipt.

**The receipt is telemetry and must never become a new way to break a run.**
:func:`record_publication_receipt` -- the single function both
``save_checkpoint`` and :func:`backfill_publication_receipt` use to write a
row -- catches ``OSError`` narrowly around the build-and-append and logs a
WARNING naming the receipt path and checkpoint directory instead of raising.
``catalog.publish`` and ``write_latest`` are unaffected by this module and
keep their existing fail-loud behaviour; they are load-bearing and have
:func:`~tpen.checkpoint.catalog.reconcile_publication` as a repair path, which
is precisely why the receipt -- the least important of the three durable
writes -- is allowed the most forgiving failure mode. See
:func:`iter_valid_publication_receipts` for why reading a receipt log is
equally forgiving, and deliberately asymmetric with
:meth:`~tpen.checkpoint.catalog.CheckpointCatalog.iter_publications`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from tpen.artifacts import append_jsonl

from .artifact import COMPLETE_MARKER
from .reference import CheckpointRef

_LOGGER = logging.getLogger("tpen")

PUBLICATION_RECEIPT_FILENAME = "publication_receipts.jsonl"
PUBLICATION_RECEIPT_SCHEMA = "tpen.checkpoint-publication-receipt/v1"
MANIFEST_FILENAME = "manifest.json"

#: Components whose bytes are the restorable model/train-resume state, as
#: opposed to descriptive metadata about that state. Matches the component
#: names ``save_checkpoint`` uses as keys in its ``files`` mapping.
PAYLOAD_COMPONENT_NAMES = frozenset({"model", "optimizer", "trainer", "sampler", "rng"})

#: Pseudo-component names for the two files every checkpoint directory has
#: that are not listed in the manifest's own ``files`` mapping.
_MANIFEST_COMPONENT = "manifest"
_COMPLETE_COMPONENT = "complete"


@dataclass(frozen=True, slots=True)
class CheckpointFileSize:
    """Logical byte size of one file inside a published checkpoint directory.

    Parameters
    ----------
    component : str
        Manifest component name (e.g. ``"model"``), or one of the two fixed
        pseudo-component names ``"manifest"`` / ``"complete"`` for the files
        every checkpoint directory carries outside the manifest's own
        ``files`` mapping.
    relative_path : str
        File name relative to the checkpoint directory.
    size_bytes : int
        ``Path.stat().st_size`` of the file, read after the file was closed.
        Never derived from a directory scan or from block allocation.
    """

    component: str
    relative_path: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError(f"size_bytes must be nonnegative, got {self.size_bytes}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of this file size record."""

        return {
            "component": self.component,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class CheckpointPublished:
    """Scalar summary of one checkpoint's publication size and duration.

    Deliberately carries no per-file breakdown -- :class:`CheckpointPublicationReceipt`
    is the typed record for that -- so this stays small enough to log or hand
    to a metric sink on its own.

    Parameters
    ----------
    checkpoint_dir : str
        Name of the committed ``step_*`` directory (not a full path; stable
        across a run tree being moved or collected).
    content_id : str
        The published :class:`~tpen.checkpoint.reference.CheckpointRef`'s
        path-independent content identity.
    file_count : int
        Number of files this summary's totals were computed over. Always
        equal to ``len(files)`` on the sibling :class:`CheckpointPublicationReceipt`.
    payload_bytes : int
        Sum of :attr:`CheckpointFileSize.size_bytes` over files whose
        component is in :data:`PAYLOAD_COMPONENT_NAMES`.
    metadata_bytes : int
        Sum of :attr:`CheckpointFileSize.size_bytes` over the remaining files
        (``resolved_config``, ``manifest``, ``complete``).
    total_bytes : int
        ``payload_bytes + metadata_bytes``, equivalently the sum over every
        measured file. Never computed independently of the two components, so
        the identity cannot drift.
    write_duration_sec : float or None
        Wall-clock seconds from immediately before ``tmp_dir.mkdir`` to
        immediately after ``tmp_dir.rename(final_dir)`` in
        ``save_checkpoint``, measured with ``time.perf_counter()``. Covers
        every component write, hashing, and the manifest/``COMPLETE`` writes,
        but not catalog publication or the ``latest.json`` update. Present
        and non-negative on every receipt ``save_checkpoint`` writes directly.
        ``None`` ONLY on a receipt produced by
        :func:`backfill_publication_receipt` (via
        ``reconcile_publication``) for a checkpoint committed by an earlier,
        unrelated process invocation, where this duration cannot be
        recovered. Never fabricated and never zero-filled: zero is
        indistinguishable from a genuinely fast write and would corrupt any
        analysis or Stage-T3 consumer, so absence is represented explicitly
        with ``None`` (serialized as JSON ``null``) rather than omitted.
    publish_duration_sec : float or None
        Wall-clock seconds from immediately after ``tmp_dir.rename`` to
        immediately after ``write_latest`` returns, measured with
        ``time.perf_counter()``. Covers catalog-row publication and the
        ``latest.json`` update, not the receipt append itself. Present and
        non-negative, or ``None``, under exactly the same rule as
        ``write_duration_sec``.
    """

    checkpoint_dir: str
    content_id: str
    file_count: int
    payload_bytes: int
    metadata_bytes: int
    total_bytes: int
    write_duration_sec: float | None
    publish_duration_sec: float | None

    def __post_init__(self) -> None:
        if self.file_count < 0:
            raise ValueError(f"file_count must be nonnegative, got {self.file_count}")
        if self.payload_bytes < 0:
            raise ValueError(f"payload_bytes must be nonnegative, got {self.payload_bytes}")
        if self.metadata_bytes < 0:
            raise ValueError(f"metadata_bytes must be nonnegative, got {self.metadata_bytes}")
        if self.total_bytes < 0:
            raise ValueError(f"total_bytes must be nonnegative, got {self.total_bytes}")
        if self.payload_bytes + self.metadata_bytes != self.total_bytes:
            raise ValueError(
                "total_bytes must equal payload_bytes + metadata_bytes, got "
                f"{self.total_bytes} != {self.payload_bytes} + {self.metadata_bytes}"
            )
        if self.write_duration_sec is not None and self.write_duration_sec < 0:
            raise ValueError(
                "write_duration_sec must be nonnegative or None, got "
                f"{self.write_duration_sec}"
            )
        if self.publish_duration_sec is not None and self.publish_duration_sec < 0:
            raise ValueError(
                "publish_duration_sec must be nonnegative or None, got "
                f"{self.publish_duration_sec}"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of this scalar summary."""

        return {
            "checkpoint_dir": self.checkpoint_dir,
            "content_id": self.content_id,
            "file_count": self.file_count,
            "payload_bytes": self.payload_bytes,
            "metadata_bytes": self.metadata_bytes,
            "total_bytes": self.total_bytes,
            "write_duration_sec": self.write_duration_sec,
            "publish_duration_sec": self.publish_duration_sec,
        }


@dataclass(frozen=True, slots=True)
class CheckpointPublicationReceipt:
    """Full typed receipt: every measured file's size plus the scalar summary.

    Parameters
    ----------
    summary : CheckpointPublished
        Scalar totals and durations.
    files : tuple of CheckpointFileSize
        One entry per measured file, in the same order
        :func:`measure_checkpoint_files` returned them.
    """

    summary: CheckpointPublished
    files: tuple[CheckpointFileSize, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping of the full receipt."""

        return {
            "schema": PUBLICATION_RECEIPT_SCHEMA,
            "summary": self.summary.to_dict(),
            "files": [entry.to_dict() for entry in self.files],
        }


def measure_checkpoint_files(
    checkpoint_dir: Path, files: Mapping[str, str]
) -> tuple[CheckpointFileSize, ...]:
    """Return typed per-file logical sizes for one committed checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : pathlib.Path
        Committed (post-rename) ``step_*`` directory.
    files : Mapping[str, str]
        The manifest's own component-to-relative-filename mapping (e.g.
        ``{"model": "model.pt", ...}``), exactly as written to
        ``manifest.json``.

    Returns
    -------
    tuple of CheckpointFileSize
        One entry per file in ``files``, in mapping-iteration order, followed
        by the fixed ``manifest`` and ``complete`` entries. Sizes come from
        ``Path.stat().st_size``; no directory is ever listed or walked.
    """

    entries = [
        CheckpointFileSize(
            component=component,
            relative_path=relative_path,
            size_bytes=(checkpoint_dir / relative_path).stat().st_size,
        )
        for component, relative_path in files.items()
    ]
    entries.append(
        CheckpointFileSize(
            component=_MANIFEST_COMPONENT,
            relative_path=MANIFEST_FILENAME,
            size_bytes=(checkpoint_dir / MANIFEST_FILENAME).stat().st_size,
        )
    )
    entries.append(
        CheckpointFileSize(
            component=_COMPLETE_COMPONENT,
            relative_path=COMPLETE_MARKER,
            size_bytes=(checkpoint_dir / COMPLETE_MARKER).stat().st_size,
        )
    )
    return tuple(entries)


def build_publication_receipt(
    ref: CheckpointRef,
    checkpoint_dir: Path,
    files: Mapping[str, str],
    *,
    write_duration_sec: float | None,
    publish_duration_sec: float | None,
) -> CheckpointPublicationReceipt:
    """Measure and assemble the full typed receipt for one committed checkpoint.

    Parameters
    ----------
    ref : CheckpointRef
        The just-published ref this receipt describes.
    checkpoint_dir : pathlib.Path
        Committed (post-rename) ``step_*`` directory.
    files : Mapping[str, str]
        The manifest's own component-to-relative-filename mapping.
    write_duration_sec, publish_duration_sec : float or None
        Durations measured by the caller, or ``None`` to explicitly represent
        an unrecoverable duration (see :class:`CheckpointPublished`). Never
        pass ``0.0`` to mean "unknown".

    Returns
    -------
    CheckpointPublicationReceipt
        Full receipt whose summary totals are sums over ``files``, so the
        typed-sum identity holds by construction rather than by a separate
        check.
    """

    measured = measure_checkpoint_files(checkpoint_dir, files)
    payload_bytes = sum(
        entry.size_bytes for entry in measured if entry.component in PAYLOAD_COMPONENT_NAMES
    )
    metadata_bytes = sum(
        entry.size_bytes for entry in measured if entry.component not in PAYLOAD_COMPONENT_NAMES
    )
    summary = CheckpointPublished(
        checkpoint_dir=checkpoint_dir.name,
        content_id=ref.content_id,
        file_count=len(measured),
        payload_bytes=payload_bytes,
        metadata_bytes=metadata_bytes,
        total_bytes=payload_bytes + metadata_bytes,
        write_duration_sec=write_duration_sec,
        publish_duration_sec=publish_duration_sec,
    )
    return CheckpointPublicationReceipt(summary=summary, files=measured)


def publication_receipt_path(checkpoint_root: str | Path) -> Path:
    """Return the default append-only publication receipt log path."""

    return Path(checkpoint_root) / PUBLICATION_RECEIPT_FILENAME


def append_publication_receipt(
    path: str | Path, receipt: CheckpointPublicationReceipt
) -> None:
    """Append one publication receipt as a JSONL record.

    Guards against joining onto an unterminated last line before delegating
    to ``tpen.artifacts.append_jsonl``, which is otherwise untouched -- see
    :func:`_ensure_trailing_newline` for why this guard exists.
    """

    receipt_path = Path(path)
    _ensure_trailing_newline(receipt_path)
    append_jsonl(receipt_path, receipt.to_dict())


def _ensure_trailing_newline(path: Path) -> None:
    """Close out a possibly-unterminated last line before a new append.

    ``tpen.artifacts.append_jsonl`` writes a row's JSON body and its trailing
    newline as two separate writes and is out of scope to change in this
    slice (a pre-existing, inherited exposure shared with
    ``CheckpointCatalog.publish``; see :func:`iter_valid_publication_receipts`).
    Without this guard, appending after a row left unterminated by a
    mid-write ``OSError`` would land on the SAME physical line as that
    corrupt row rather than its own -- corrupting the NEW row too, not merely
    leaving the old one dead, which would defeat
    :func:`backfill_publication_receipt`'s entire purpose. This function only
    ever appends a single ``"\\n"`` when the file is non-empty and does not
    already end with one; it never reads, rewrites, or removes existing
    content, so it cannot itself lose data.
    """

    if not path.exists():
        return
    with path.open("rb") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            return
        handle.seek(-1, 2)
        last_byte = handle.read(1)
    if last_byte != b"\n":
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")


def iter_valid_publication_receipts(path: str | Path) -> Iterator[Mapping[str, object]]:
    """Yield parsed receipt rows from `path`, skipping any malformed row.

    Deliberately asymmetric with
    :meth:`~tpen.checkpoint.catalog.CheckpointCatalog.iter_publications`,
    which raises ``ValueError`` on a malformed row. Do not "fix" this
    function to match that one; the asymmetry is intentional and is the same
    telemetry-versus-load-bearing reasoning behind this module's whole
    failure-handling design. The catalog failing loud on corruption is
    defensible because silently losing a published checkpoint's identity row
    is worse than refusing to proceed. This log is telemetry: a malformed row
    here must never become a new way to break a run, including at read time,
    so it is skipped (with a logged warning) rather than raised.

    The most likely source of a malformed row is a partially written line
    left behind by an append that failed mid-write. ``tpen.artifacts.append_jsonl``
    writes the JSON body and the trailing newline as two separate writes, and
    on an NFS-mounted root (e.g. Netscratch) ``O_APPEND`` atomicity cannot be
    assumed even for a single write; a failure between the two writes leaves
    an unterminated line that the *next* append joins onto, corrupting one
    line where two records belonged. This exposure is pre-existing in
    ``append_jsonl`` and is inherited here, not introduced by this module --
    ``CheckpointCatalog.publish`` uses the identical primitive for
    ``publications.jsonl``, where it is strictly worse, since one such event
    makes ``iter_publications`` raise for the whole catalog file rather than
    for one row. Fixing ``append_jsonl`` itself, or moving to per-checkpoint
    receipt files, is out of scope for this slice and is tracked separately.
    Treating a malformed row as "no valid receipt for this checkpoint" is what
    lets :func:`backfill_publication_receipt` repair exactly this failure
    mode, bounding the damage to one recoverable telemetry row.

    Parameters
    ----------
    path : str or pathlib.Path
        Publication receipt log path.

    Returns
    -------
    Iterator[Mapping[str, object]]
        Parsed, schema-tagged rows only. A row is valid when it parses as
        JSON, is a mapping, has ``schema == PUBLICATION_RECEIPT_SCHEMA``, and
        has a ``summary`` mapping with a string ``content_id``.
    """

    receipt_path = Path(path)
    if not receipt_path.is_file():
        return
    with receipt_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                _LOGGER.warning(
                    "skipping malformed checkpoint publication receipt row at %s:%d",
                    receipt_path, line_number,
                )
                continue
            if not isinstance(record, Mapping) or record.get("schema") != PUBLICATION_RECEIPT_SCHEMA:
                _LOGGER.warning(
                    "skipping checkpoint publication receipt row with unexpected schema "
                    "at %s:%d",
                    receipt_path, line_number,
                )
                continue
            summary = record.get("summary")
            if not isinstance(summary, Mapping) or not isinstance(summary.get("content_id"), str):
                _LOGGER.warning(
                    "skipping checkpoint publication receipt row with no valid summary "
                    "at %s:%d",
                    receipt_path, line_number,
                )
                continue
            yield record


def has_publication_receipt(path: str | Path, content_id: str) -> bool:
    """Return whether a VALID receipt row already exists for `content_id`.

    Presence is decided from parsed content via
    :func:`iter_valid_publication_receipts`, never from file existence and
    never from "the file has any row for this checkpoint" without validating
    it. A checkpoint whose only row is malformed or truncated therefore counts
    as having NO receipt, which is what makes it eligible for
    :func:`backfill_publication_receipt` to repair.
    """

    return any(
        record["summary"]["content_id"] == content_id
        for record in iter_valid_publication_receipts(path)
    )


def record_publication_receipt(
    ref: CheckpointRef,
    checkpoint_dir: Path,
    files: Mapping[str, str],
    path: str | Path,
    *,
    write_duration_sec: float | None,
    publish_duration_sec: float | None,
) -> bool:
    """Build and append one receipt, converting an ``OSError`` into a warning.

    The receipt describes a publication that has already committed. An
    ``OSError`` while building or appending it -- in production, most often a
    quota/ENOSPC failure on the receipts log -- must not report a failed save
    for a checkpoint that is durably committed, published, and pointed at by
    ``latest.json``. Any OTHER exception type is NOT caught here: a bug in
    receipt construction should fail loudly rather than be swallowed
    alongside genuine I/O failures.

    This is the single function both ``save_checkpoint`` and
    :func:`backfill_publication_receipt` use to write a row, so both call
    sites share one failure-handling implementation rather than two that could
    drift.

    Returns
    -------
    bool
        ``True`` if the receipt was appended. ``False`` if an ``OSError``
        prevented it -- in which case a WARNING naming the receipt path and
        checkpoint directory has already been logged; the failure is never
        silent.
    """

    receipt_path = Path(path)
    try:
        receipt = build_publication_receipt(
            ref,
            checkpoint_dir,
            files,
            write_duration_sec=write_duration_sec,
            publish_duration_sec=publish_duration_sec,
        )
        append_publication_receipt(receipt_path, receipt)
    except OSError as error:
        _LOGGER.warning(
            "checkpoint publication receipt append failed for %s (receipt log %s): %s",
            checkpoint_dir, receipt_path, error,
        )
        return False
    return True


def backfill_publication_receipt(
    ref: CheckpointRef,
    checkpoint_dir: Path,
    files: Mapping[str, str],
    path: str | Path,
) -> bool:
    """Record a receipt for `ref` only if it has no valid row yet.

    Called by ``reconcile_publication`` for a checkpoint that is already
    committed and published but may have missed its receipt -- most often
    because an earlier ``save_checkpoint`` call's best-effort append failed.
    Durations are always recorded as ``None`` (explicit-absent): the write
    and publish durations of a checkpoint committed by a different process
    invocation cannot be recovered, and this function never fabricates or
    zero-fills them.

    Reading the existing log is itself wrapped in the same forgiving
    handling as the write path: an ``OSError`` while checking for an existing
    row (for example, the receipts path existing as something other than a
    regular file) is logged at WARNING and treated as "do not backfill this
    time" rather than propagated, so a repair helper can never turn a
    telemetry problem into a resume failure.

    Returns
    -------
    bool
        ``True`` if a new row was appended. ``False`` if a valid row already
        existed, or if an ``OSError`` prevented either the read or the write
        (logged in either case, never silent).
    """

    receipt_path = Path(path)
    try:
        if has_publication_receipt(receipt_path, ref.content_id):
            return False
    except OSError as error:
        _LOGGER.warning(
            "checkpoint publication receipt read failed for %s (receipt log %s); "
            "skipping backfill: %s",
            checkpoint_dir, receipt_path, error,
        )
        return False
    return record_publication_receipt(
        ref,
        checkpoint_dir,
        files,
        receipt_path,
        write_duration_sec=None,
        publish_duration_sec=None,
    )


__all__ = [
    "PAYLOAD_COMPONENT_NAMES",
    "PUBLICATION_RECEIPT_FILENAME",
    "PUBLICATION_RECEIPT_SCHEMA",
    "CheckpointFileSize",
    "CheckpointPublicationReceipt",
    "CheckpointPublished",
    "append_publication_receipt",
    "backfill_publication_receipt",
    "build_publication_receipt",
    "has_publication_receipt",
    "iter_valid_publication_receipts",
    "measure_checkpoint_files",
    "publication_receipt_path",
    "record_publication_receipt",
]
