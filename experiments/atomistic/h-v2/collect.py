"""Strict collector for complete H-v2 draw-by-walker records.

Consumes TPEN's real per-evaluation-run artifacts directly: the trajectory
``sampled_eval_table.csv`` / ``sampled_eval_table.metadata.json`` envelope
written by ``tpen.evaluation.summaries.SampledRecordWriter`` for a complete
draw-by-walker grid (schema ``trajectory_records/v1``, see
``tpen.evaluation.trajectory_records``), the run's own ``resolved_config.yaml``
and ``metadata.json`` (written by ``tpen.callback.ResolvedConfigSnapshot`` and
``tpen.callback.Metadata``), and the restored checkpoint directory (written by
``tpen.checkpoint.save.save_checkpoint``).

Identity is reconciled at the envelope, not per row, following the he-v1
collector's precedent (resolved-config hash, csv hash, source commit): the
real per-row schema carries no per-row identity columns.

The collector is deliberately conservative: it validates every identity and
the entire grid before calculating any statistic.  It never repairs, clips,
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

import yaml

#: Schema and row-semantics constants mirrored from
#: ``tpen.evaluation.trajectory_records`` so this module never imports tpen
#: (see ``experiments/README.md``).
TRAJECTORY_RECORD_SCHEMA = "trajectory_records/v1"
ROW_SEMANTICS = "complete_draw_walker_grid"


class RecordIntegrityError(ValueError):
    """Raised when a record cannot be admitted to the complete-record set."""


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RecordSpec:
    """Expected identity and shape for one complete evaluation stream."""

    source_sha: str
    checkpoint_model_sha256: str
    resolved_config_sha256: str
    walkers: int = 1024
    draws: int = 64

    @property
    def expected_rows(self) -> int:
        """Return the complete draw-by-walker row count."""

        return self.walkers * self.draws


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordIntegrityError(message)


def validate_checkpoint_identity(checkpoint_dir: Path, spec: RecordSpec) -> None:
    """Validate the restored checkpoint's commit marker and content identity."""

    _require((checkpoint_dir / "COMPLETE").is_file(), "missing checkpoint commit marker")
    model_path = checkpoint_dir / "model.pt"
    _require(model_path.is_file(), "altered source/checkpoint/config identity")
    _require(
        file_sha256(model_path) == spec.checkpoint_model_sha256,
        "altered source/checkpoint/config identity",
    )


def validate_run_identity(
    run_metadata: dict[str, Any], resolved_config_path: Path, spec: RecordSpec
) -> None:
    """Validate the evaluation run's source commit and resolved-config identity."""

    _require(
        run_metadata.get("git_commit") == spec.source_sha,
        "altered source/checkpoint/config identity",
    )
    _require(
        file_sha256(resolved_config_path) == spec.resolved_config_sha256,
        "altered source/checkpoint/config identity",
    )


def validate_restore_mode(resolved_config: dict[str, Any]) -> None:
    """Refuse a restore whose mode is absent or is not ``model_only``.

    The collector is deliberately conservative: an absent ``load.mode`` is
    refused exactly like an explicit ``train_resume`` request, rather than
    defaulting to the safe value.
    """

    mode = (resolved_config.get("load") or {}).get("mode")
    _require(mode is not None, "collector requires an explicit restore_mode")
    _require(mode == "model_only", "collector requires a model-only restore_mode")


def validate_manifest(manifest: dict[str, Any], csv_path: Path, spec: RecordSpec) -> None:
    """Validate the immutable trajectory envelope before reading rows."""

    _require(manifest.get("schema") == TRAJECTORY_RECORD_SCHEMA, "wrong record schema")
    _require(
        manifest.get("row_semantics") == ROW_SEMANTICS,
        "record is not a complete draw-by-walker grid",
    )
    _require(manifest.get("walker_count") == spec.walkers, "record shape differs from the frozen evaluation")
    _require(manifest.get("draw_count") == spec.draws, "record shape differs from the frozen evaluation")
    _require(manifest.get("row_count") == spec.expected_rows, "record manifest is not complete")
    _require(manifest.get("nonfinite_count", 1) == 0, "nonfinite record")
    _require(manifest.get("csv_sha256") == file_sha256(csv_path), "uncommitted tail")


def validate_rows(rows: list[dict[str, Any]], spec: RecordSpec) -> list[dict[str, Any]]:
    """Validate every row and return it in its original complete-grid order."""

    _require(len(rows) == spec.expected_rows, "missing draw or uncommitted tail")
    seen: set[tuple[int, int]] = set()
    ordered: list[dict[str, Any] | None] = [None] * spec.expected_rows
    for row in rows:
        try:
            draw = int(row["draw_index"])
            walker = int(row["walker_index"])
            energy = float(row["local_energy"])
            logabs = float(row["logabs"])
            coordinates = [
                float(row[f"position/electron_0/axis_{axis}"]) for axis in range(3)
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordIntegrityError("malformed complete record") from exc
        if not 0 <= draw < spec.draws or not 0 <= walker < spec.walkers:
            raise RecordIntegrityError("record index outside the complete grid")
        key = (draw, walker)
        if key in seen:
            raise RecordIntegrityError("duplicate walker record")
        seen.add(key)
        # TPEN's own ``finite`` column covers only ``local_energy``
        # (tpen/evaluation/generators/trajectory.py); the contract's
        # complete-record rule also requires finite coordinates, so they are
        # checked explicitly here.
        if (
            row.get("finite") != "True"
            or not math.isfinite(energy)
            or not math.isfinite(logabs)
            or not all(math.isfinite(coordinate) for coordinate in coordinates)
        ):
            raise RecordIntegrityError("nonfinite record")
        ordered[draw * spec.walkers + walker] = row
    if any(row is None for row in ordered):
        raise RecordIntegrityError("missing draw or uncommitted tail")
    return [row for row in ordered if row is not None]


def collect_records(
    *,
    csv_path: Path,
    metadata_path: Path,
    resolved_config_path: Path,
    run_metadata_path: Path,
    checkpoint_dir: Path,
    spec: RecordSpec,
) -> dict[str, Any]:
    """Read and validate one complete real evaluation stream, returning a receipt."""

    validate_checkpoint_identity(checkpoint_dir, spec)
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    validate_run_identity(run_metadata, resolved_config_path, spec)
    resolved_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    validate_restore_mode(resolved_config)
    if not metadata_path.is_file():
        raise RecordIntegrityError("uncommitted tail")
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, csv_path, spec)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ordered = validate_rows(rows, spec)
    return {
        "schema": "tpen.h-v2.collection/v1",
        "status": "complete",
        "source_sha": spec.source_sha,
        "checkpoint_model_sha256": spec.checkpoint_model_sha256,
        "resolved_config_sha256": spec.resolved_config_sha256,
        "row_count": len(ordered),
        "csv_sha256": manifest["csv_sha256"],
        "statistic_input": "complete_draw_by_walker_grid",
    }


__all__ = [
    "RecordIntegrityError",
    "RecordSpec",
    "ROW_SEMANTICS",
    "TRAJECTORY_RECORD_SCHEMA",
    "collect_records",
    "file_sha256",
    "validate_checkpoint_identity",
    "validate_manifest",
    "validate_restore_mode",
    "validate_rows",
    "validate_run_identity",
]
