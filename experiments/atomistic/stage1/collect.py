"""Collect complete Stage-1 evaluation grids into a deterministic document."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from experiments.baselines.systems import load_registry
from experiments.toolkit.trajectory_records import (
    TrajectoryReaderError, inspect_run, inspect_trajectory_record, read_energy_grid,
)
from experiments.atomistic.stage1.protocol import manifest_hash


class CollectionError(ValueError):
    """The manifest and durable run evidence cannot be joined."""


def _load(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _record_path(run_dir: Path) -> Path:
    """Resolve the sole evaluator metadata sidecar.

    Parameters
    ----------
    run_dir
        Completed evaluation run directory containing the ``mcmc_energy``
        task output.

    Returns
    -------
    pathlib.Path
        The evaluator-owned ``sampled_eval_table.metadata.json`` path.

    Raises
    ------
    CollectionError
        If the fixed sidecar is missing or duplicate metadata sidecars are
        present.
    """
    task_dir = run_dir / "mcmc_energy"
    sidecars = sorted(task_dir.glob("*.metadata.json"))
    if len(sidecars) > 1:
        raise CollectionError(f"duplicate trajectory metadata sidecars in {task_dir}")
    path = run_dir / "mcmc_energy" / "sampled_eval_table.metadata.json"
    if not path.is_file():
        raise CollectionError(f"missing trajectory metadata sidecar {path}")
    return path


def _moments(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    return mean, sum((value - mean) ** 2 for value in values) / len(values)


def _reference(system_id: str, registry_path: Path | None) -> tuple[float, str]:
    systems = load_registry(registry_path).get("systems", [])
    for entry in systems:
        if entry.get("id") == system_id:
            if "reference_energy_hartree" not in entry or entry["reference_energy_hartree"] is None:
                raise CollectionError(f"reference energy missing for system {system_id}")
            if not isinstance(entry.get("confidence"), str):
                raise CollectionError(f"reference confidence missing for system {system_id}")
            return float(entry["reference_energy_hartree"]), entry["confidence"]
    raise CollectionError(f"unknown reference system id {system_id}")


def collect(manifest: Mapping[str, Any] | str | Path, *, output: str | Path | None = None,
            registry_path: Path | None = None) -> dict[str, Any]:
    """Collect all eval rows in a Stage-1 manifest.

    The public entry point is intentionally small: callers provide the A1
    manifest and receive a JSON-safe document, optionally written atomically
    by the caller's normal output handling.  The manifest carries the
    registry system ID, while records carry the physical system content ID;
    this collector cannot pin the expected physical hash because the manifest
    does not carry it.
    """
    document = _load(manifest)
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise CollectionError("manifest rows must be a list")
    expected_manifest_hash = document.get("manifest_hash")
    if expected_manifest_hash != manifest_hash(rows):
        raise CollectionError("manifest_hash does not match manifest rows")
    if any(not isinstance(row, Mapping) for row in rows):
        raise CollectionError("manifest contains a malformed eval row")
    eval_rows = [row for row in rows if row.get("kind") == "eval"]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    registry_to_physical: dict[str, str] = {}
    physical_to_registry: dict[str, str] = {}
    for row in eval_rows:
        row_id = str(row.get("row_id"))
        if row_id in seen:
            raise CollectionError(f"duplicate planned row {row_id}")
        seen.add(row_id)
        run_dir = Path(str(row["results_root"])) / str(row["run_id"])
        try:
            run = inspect_run(run_dir)
            if run.get("verdict") != "completed":
                raise CollectionError(f"run {run_dir} is not completed")
            identity = run.get("resolved_config_identity")
            if (not isinstance(identity, Mapping)
                    or set(identity) != {"schema", "sha256"}
                    or identity.get("schema") != "tpen-config-identity-v1"):
                raise CollectionError(f"row {row_id}: run identity is unavailable or malformed")
            metadata_path = _record_path(run_dir)
            record = inspect_trajectory_record(
                metadata_path, expected_resolved_config_identity=identity,
                expected_checkpoint_sha256=str(row["checkpoint_sha256"]),
            )
            if record.record_schema != "trajectory_records/v2":
                raise CollectionError(f"row {row_id}: identity unavailable for v1 record")
            if record.resolved_config_identity != dict(identity):
                raise CollectionError(f"row {row_id}: record identity differs from run identity")
            if record.draw_count != int(row["draws"]) or record.walker_count != int(row["walkers"]):
                raise CollectionError(f"row {row_id}: trajectory grid shape mismatch")
            if record.row_count != record.draw_count * record.walker_count:
                raise CollectionError(f"row {row_id}: incomplete trajectory grid")
            if record.model_provenance != "checkpoint":
                raise CollectionError(f"row {row_id}: model provenance is not checkpoint")
            physical_system_id = record.system_id
            if not isinstance(physical_system_id, str) or not physical_system_id.strip():
                raise CollectionError(f"row {row_id}: physical system identity is absent")
            registry_system_id = row.get("system_id")
            previous_physical = registry_to_physical.get(registry_system_id)
            if previous_physical is not None and previous_physical != physical_system_id:
                raise CollectionError(
                    f"row {row_id}: registry system {registry_system_id} maps to inconsistent "
                    "physical system identities"
                )
            previous_registry = physical_to_registry.get(physical_system_id)
            if previous_registry is not None and previous_registry != registry_system_id:
                raise CollectionError(
                    f"row {row_id}: distinct registry systems map to the same physical system"
                )
            registry_to_physical[registry_system_id] = physical_system_id
            physical_to_registry[physical_system_id] = registry_system_id
            if not record.execution_method_id:
                raise CollectionError(f"row {row_id}: execution method identity is absent")
            if run.get("source_git_sha") != row.get("source_sha"):
                raise CollectionError(f"row {row_id}: run_start.git.sha mismatches source_sha")
            grid = read_energy_grid(record)
        except (KeyError, TypeError, OSError, TrajectoryReaderError, CollectionError) as exc:
            if isinstance(exc, CollectionError):
                raise
            raise CollectionError(f"row {row_id}: {exc}") from exc
        if record.finite_count != record.row_count:
            raise CollectionError(f"row {row_id}: non-finite trajectory rows are not admissible")
        # Match tpen.evaluation.trajectory_records._scan_complete_grid exactly:
        # draw-ordered plain-float sums and E[x^2] - mean^2, avoiding a fixed
        # absolute tolerance that would be unjustified after cancellation.
        draw_sums = {index: sum(draw) for index, draw in enumerate(grid)}
        draw_square_sums = {index: sum(value * value for value in draw)
                            for index, draw in enumerate(grid)}
        total = 0.0
        total_square = 0.0
        for index in range(len(grid)):
            total += draw_sums[index]
            total_square += draw_square_sums[index]
        mean = total / record.row_count
        variance = total_square / record.row_count - mean * mean
        if math.isfinite(variance) and variance < 0.0:
            variance = 0.0
        if mean != record.mean or variance != record.variance:
            raise CollectionError(f"row {row_id}: pooled statistics do not reconcile with inspection")
        draw_means = [_moments(draw)[0] for draw in grid]
        _, draw_mean_variance = _moments(draw_means)
        reference, confidence = _reference(str(row["system_id"]), registry_path)
        records.append({
            "row_id": row_id, "system_id": row["system_id"], "lineage": row["lineage"],
            "checkpoint": row["checkpoint"], "stream": row["stream"],
            "run_dir": str(run_dir),
            "identity": {"record_content_id": record.record_content_id,
                         "physical_system_id": physical_system_id,
                         "csv_sha256": record.csv_sha256,
                         "checkpoint_sha256": record.checkpoint_sha256,
                         "resolved_config_identity": record.resolved_config_identity},
            "statistics": {"mean": mean, "variance": variance,
                           "draw_mean_variance": draw_mean_variance,
                           "draw_axis_uncertainty": math.sqrt(draw_mean_variance / len(draw_means)),
                           "n_draws": len(grid), "n_walkers": len(grid[0]),
                           "finite_count": record.finite_count,
                           "assumption": "draws treated as independent after recorded draw_stride and burn_in_draws; no autocorrelation correction"},
            "reference": {"energy_hartree": reference, "confidence": confidence,
                          "error": mean - reference, "absolute_error": abs(mean - reference)},
        })
    aggregates: dict[tuple[str, int, int], list[tuple[int, float]]] = defaultdict(list)
    for record in records:
        aggregates[(record["system_id"], record["lineage"], record["checkpoint"])].append(
            (record["stream"], record["statistics"]["mean"])
        )
    aggregate_rows = []
    for (system_id, lineage, checkpoint), stream_records in sorted(aggregates.items()):
        stream_indices = [stream for stream, _ in stream_records]
        # Aggregates are science outputs only when all planned streams are present.
        stream_count = len(stream_indices)
        if stream_count < 4 or set(stream_indices) != set(range(stream_count)):
            raise CollectionError(
                f"{system_id}/{lineage}/{checkpoint}: aggregate requires four streams "
                f"with contiguous unique indices, got {stream_indices}"
            )
        means = [mean for _, mean in stream_records]
        average = sum(means) / len(means)
        aggregate_rows.append({"system_id": system_id, "lineage": lineage, "checkpoint": checkpoint,
                               "kind": "stream_mean_aggregate", "stream_count": len(means),
                               "mean_of_stream_means": average,
                               "spread_of_stream_means": math.sqrt(sum((x - average) ** 2 for x in means) / len(means))})
    result = {"schema": "stage1-collection/v1", "manifest_hash": expected_manifest_hash,
              "records": records, "aggregates": aggregate_rows}
    if output is not None:
        Path(output).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["CollectionError", "collect"]

# Descriptive alias used by command-line and review tooling.
collect_manifest = collect
