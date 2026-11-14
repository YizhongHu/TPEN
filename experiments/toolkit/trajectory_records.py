"""Small, plain-data readers for Stage-1 trajectory collection.

This module is the only experiments-layer import seam for core trajectory and
run inspection.  Core objects never cross this boundary.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class TrajectoryReaderError(ValueError):
    """A record or run could not be inspected or projected."""


@dataclass(frozen=True)
class InspectedTrajectory:
    """Plain projection of the public core inspection result."""

    record_schema: str
    path: Path
    metadata_path: Path
    draw_count: int
    walker_count: int
    row_count: int
    finite_count: int
    mean: float
    variance: float
    csv_sha256: str
    system_id: str | None
    execution_method_id: str | None
    resolved_config_identity: dict[str, str] | None
    model_provenance: str | None
    checkpoint_sha256: str | None
    record_content_id: str | None


def inspect_trajectory_record(metadata_path: str | Path, *,
                              expected_resolved_config_identity: Mapping[str, str] | None = None,
                              expected_checkpoint_sha256: str | None = None) -> InspectedTrajectory:
    """Inspect through :mod:`tpen.inspection`, returning only plain data."""

    from tpen.inspection import inspect_trajectory_record as inspect  # lazy boundary

    try:
        value = inspect(metadata_path,
                        expected_resolved_config_identity=expected_resolved_config_identity,
                        expected_checkpoint_sha256=expected_checkpoint_sha256)
    except Exception as exc:
        raise TrajectoryReaderError(str(exc)) from exc
    physical = getattr(value, "physical_context", None)
    try:
        finite_count = int(json.loads(Path(value.metadata_path).read_text(encoding="utf-8"))["finite_count"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TrajectoryReaderError("inspected trajectory sidecar lacks finite_count") from exc
    return InspectedTrajectory(
        record_schema=str(value.record_schema), path=Path(value.path),
        metadata_path=Path(value.metadata_path), draw_count=int(value.draw_count),
        walker_count=int(value.walker_count), row_count=int(value.row_count), finite_count=finite_count,
        mean=float(value.mean), variance=float(value.variance),
        csv_sha256=str(value.csv_sha256),
        system_id=(getattr(physical, "system_id", None)
                   if getattr(physical, "available", False) else None),
        execution_method_id=getattr(value, "execution_method_id", None),
        resolved_config_identity=(None if value.resolved_config_identity is None
                                  else dict(value.resolved_config_identity)),
        model_provenance=getattr(value, "model_provenance", None),
        checkpoint_sha256=getattr(value, "checkpoint_sha256", None),
        record_content_id=getattr(value, "record_content_id", None),
    )


def inspect_run(run_dir: str | Path) -> dict[str, Any]:
    """Inspect a run lazily and serialize its public evidence."""

    from tpen.inspection import inspect_run as inspect  # lazy boundary

    try:
        value = inspect(run_dir)
        evidence = value.to_dict()
        # inspect_run's typed evidence carries the run-start identity; source
        # provenance is the public run-start field, not the training checkpoint.
        start = Path(run_dir) / "run_start.json"
        if start.is_file():
            data = json.loads(start.read_text(encoding="utf-8"))
            git = data.get("git", {}) if isinstance(data, dict) else {}
            evidence["source_git_sha"] = git.get("sha") if isinstance(git, dict) else None
        else:
            evidence["source_git_sha"] = None
        return evidence
    except Exception as exc:
        raise TrajectoryReaderError(str(exc)) from exc


def read_energy_grid(record: InspectedTrajectory) -> list[list[float]]:
    """Read local-energy values only after successful record inspection."""

    actual = hashlib.sha256(record.path.read_bytes()).hexdigest()
    if actual != record.csv_sha256:
        raise TrajectoryReaderError("trajectory CSV changed after inspection")
    with record.path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0] or "local_energy" not in rows[0]:
        raise TrajectoryReaderError("trajectory CSV lacks local_energy")
    grid: dict[int, dict[int, float]] = {}
    for row in rows:
        try:
            draw, walker = int(row["draw_index"]), int(row["walker_index"])
            value = float(row["local_energy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise TrajectoryReaderError("trajectory CSV has invalid grid/value row") from exc
        if not math.isfinite(value):
            raise TrajectoryReaderError("complete Stage-1 grid must be finite")
        if draw in grid and walker in grid[draw]:
            raise TrajectoryReaderError("trajectory CSV contains duplicate grid coordinates")
        grid.setdefault(draw, {})[walker] = value
    if len(rows) != record.row_count or len(grid) != record.draw_count:
        raise TrajectoryReaderError("trajectory CSV shape disagrees with inspected record")
    if any(len(walkers) != record.walker_count for walkers in grid.values()):
        raise TrajectoryReaderError("trajectory CSV is not a complete draw-by-walker grid")
    return [[grid[d][w] for w in sorted(grid[d])] for d in sorted(grid)]


__all__ = ["InspectedTrajectory", "TrajectoryReaderError", "inspect_run",
           "inspect_trajectory_record", "read_energy_grid"]
