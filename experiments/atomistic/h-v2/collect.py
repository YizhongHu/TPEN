"""Strict collector for complete H-v2 draw-by-walker records.

The collector is deliberately conservative: it validates identities and the
entire grid before calculating any statistic.  It never repairs, clips,
filters, replaces, or silently accepts a final-draw-only file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RecordIntegrityError(ValueError):
    """Raised when a record cannot be admitted to the complete-record set."""


@dataclass(frozen=True)
class RecordSpec:
    """Identity and shape requirements for one complete evaluation stream."""

    source_sha: str
    checkpoint_sha: str
    config_sha: str
    walkers: int = 1024
    draws: int = 64
    schema: str = "tpen.h-v2.records/v1"

    @property
    def expected_rows(self) -> int:
        """Return the complete draw-by-walker row count."""

        return self.walkers * self.draws


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_identity(record: dict[str, Any], key: str, expected: str) -> None:
    if record.get(key) != expected:
        raise RecordIntegrityError(f"wrong {key} identity")


def validate_manifest(manifest: dict[str, Any], spec: RecordSpec) -> None:
    """Validate the immutable publication envelope before reading rows."""

    if manifest.get("schema") != spec.schema:
        raise RecordIntegrityError("wrong record schema")
    for key, expected in (("source_sha", spec.source_sha), ("checkpoint_sha", spec.checkpoint_sha), ("config_sha", spec.config_sha)):
        _require_identity(manifest, key, expected)
    if manifest.get("commit_marker") != "committed":
        raise RecordIntegrityError("missing commit marker")
    if manifest.get("restore_mode", "model_only") != "model_only":
        raise RecordIntegrityError("collector requires a model-only restore")
    if manifest.get("row_count") != spec.expected_rows:
        raise RecordIntegrityError("record manifest is not complete")
    if manifest.get("walkers") != spec.walkers or manifest.get("draws") != spec.draws:
        raise RecordIntegrityError("record shape differs from the frozen evaluation")


def validate_rows(rows: list[dict[str, Any]], spec: RecordSpec, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate every row and return it in its original complete-grid order."""

    validate_manifest(manifest, spec)
    if len(rows) != spec.expected_rows:
        raise RecordIntegrityError("missing draw or uncommitted tail")
    seen: set[tuple[int, int]] = set()
    ordered: list[dict[str, Any] | None] = [None] * spec.expected_rows
    for row in rows:
        for key, expected in (("source_sha", spec.source_sha), ("checkpoint_sha", spec.checkpoint_sha), ("config_sha", spec.config_sha)):
            _require_identity(row, key, expected)
        if row.get("commit_marker") != "committed":
            raise RecordIntegrityError("row commit marker is not committed")
        try:
            draw = int(row["draw_index"])
            walker = int(row["walker_index"])
            energy = float(row["local_energy"])
            coordinates = [float(row[f"position_{axis}"]) for axis in "xyz"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordIntegrityError("malformed complete record") from exc
        if not 0 <= draw < spec.draws or not 0 <= walker < spec.walkers:
            raise RecordIntegrityError("record index outside the complete grid")
        key = (draw, walker)
        if key in seen:
            raise RecordIntegrityError("duplicate walker record")
        seen.add(key)
        if not math.isfinite(energy) or not all(math.isfinite(value) for value in coordinates):
            raise RecordIntegrityError("nonfinite record")
        flat_index = draw * spec.walkers + walker
        ordered[flat_index] = row
    if any(row is None for row in ordered):
        raise RecordIntegrityError("missing draw or uncommitted tail")
    return [row for row in ordered if row is not None]


def collect_records(record_path: Path, manifest_path: Path, spec: RecordSpec) -> dict[str, Any]:
    """Read and validate one complete CSV stream, returning a receipt."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with record_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ordered = validate_rows(rows, spec, manifest)
    return {
        "schema": "tpen.h-v2.collection/v1",
        "status": "complete",
        "source_sha": spec.source_sha,
        "checkpoint_sha": spec.checkpoint_sha,
        "config_sha": spec.config_sha,
        "row_count": len(ordered),
        "record_sha256": file_sha256(record_path),
        "statistic_input": "complete_draw_by_walker_grid",
    }


__all__ = ["RecordIntegrityError", "RecordSpec", "collect_records", "file_sha256", "validate_rows"]
