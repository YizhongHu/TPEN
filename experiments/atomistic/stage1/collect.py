"""Collect complete Stage-1 evaluation artifacts.

The collector consumes TPEN's emitted ``sampled_eval_table.csv`` and its
``.metadata.json`` sidecar.  It deliberately treats a short draw axis as an
explicit floor condition instead of silently asking ``blocking_stderr`` for
an unblocked error bar.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from experiments.baselines.statistics import blocking_stderr
from experiments.atomistic.stage1.protocol import SYSTEM_IDS

TRAJECTORY_RECORD_SCHEMA = "trajectory_records/v1"
ROW_SEMANTICS = "complete_draw_walker_grid"


class CollectionError(ValueError):
    """Raised when an evaluation artifact is incomplete or has bad identity."""


@dataclass(frozen=True)
class RecordSpec:
    """Expected identity and complete rectangular shape of one stream."""

    walkers: int = 4096
    draws: int = 256
    source_sha: str | None = None
    checkpoint_model_sha256: str | None = None
    resolved_config_sha256: str | None = None

    @property
    def expected_rows(self) -> int:
        """Return the number of rows in the complete draw-by-walker grid."""

        return self.walkers * self.draws


def file_sha256(path: Path) -> str:
    """Hash a file in chunks so large trajectory tables stay bounded in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionError(message)


def reference_energy(system: str, registry: Mapping[str, Any]) -> float:
    """Read a required reference by system key; absence is never zero."""

    _require(system in SYSTEM_IDS, f"unknown Stage-1 system {system!r}")
    wanted = SYSTEM_IDS[system]
    for entry in registry.get("systems", []):
        if entry.get("id") == wanted:
            value = entry.get("reference_energy_hartree")
            _require(value is not None, f"reference key {wanted!r} is present but null")
            return float(value)
    raise CollectionError(f"no required key {wanted!r} in systems registry")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"resolved config is not a mapping: {path}")
    return value


def validate_identity(run_dir: Path, spec: RecordSpec) -> dict[str, Any]:
    """Validate source/config/checkpoint identity and strict model-only restore."""

    metadata_path = run_dir / "metadata.json"
    config_path = run_dir / "resolved_config.yaml"
    _require(metadata_path.is_file() and config_path.is_file(), "missing run identity artifacts")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = _load_yaml(config_path)
    if spec.source_sha is not None:
        _require(metadata.get("git_commit") == spec.source_sha, "source commit identity mismatch")
    if spec.resolved_config_sha256 is not None:
        _require(file_sha256(config_path) == spec.resolved_config_sha256, "resolved config identity mismatch")
    load = config.get("load") or {}
    _require(load.get("mode") == "model_only", "collector requires load.mode=model_only")
    _require(load.get("strict") is True, "collector requires strict model-only restore")
    checkpoint = load.get("path")
    _require(isinstance(checkpoint, str) and checkpoint not in {"", "???"}, "missing checkpoint path")
    checkpoint_dir = Path(checkpoint)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = run_dir / checkpoint_dir
    model = checkpoint_dir / "model.pt"
    _require((checkpoint_dir / "COMPLETE").is_file() and model.is_file(), "checkpoint is incomplete")
    if spec.checkpoint_model_sha256 is not None:
        _require(file_sha256(model) == spec.checkpoint_model_sha256, "checkpoint identity mismatch")
    return config


def _read_rows(csv_path: Path, spec: RecordSpec) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    _require(len(rows) == spec.expected_rows, "incomplete draw-by-walker grid")
    seen: set[tuple[int, int]] = set()
    for row in rows:
        try:
            draw = int(row["draw_index"])
            walker = int(row["walker_index"])
            energy = float(row["local_energy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CollectionError("malformed sampled_eval_table row") from exc
        _require(0 <= draw < spec.draws and 0 <= walker < spec.walkers, "record index outside grid")
        _require((draw, walker) not in seen, "duplicate draw-by-walker record")
        seen.add((draw, walker))
        _require(row.get("finite") == "True" and math.isfinite(energy), "nonfinite local-energy record")
    _require(len(seen) == spec.expected_rows, "incomplete draw-by-walker grid")
    return rows


def collect_records(
    trajectory_dir: Path,
    *,
    system: str,
    registry: Mapping[str, Any],
    spec: RecordSpec | None = None,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    """Collect mean, draw-axis MCSE, variance, identity, and wired task metrics.

    The MCSE series is the mean local energy of each draw across walkers.  A
    draw is one Markov-chain transition, so blocking this axis preserves the
    temporal correlation that a naive row-wise standard error would erase.
    """

    spec = spec or RecordSpec()
    csv_path = trajectory_dir / "sampled_eval_table.csv"
    metadata_path = trajectory_dir / "sampled_eval_table.metadata.json"
    _require(csv_path.is_file() and metadata_path.is_file(), "missing sampled evaluation table or metadata")
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == TRAJECTORY_RECORD_SCHEMA, "wrong record schema")
    _require(manifest.get("row_semantics") == ROW_SEMANTICS, "record is not a complete draw-by-walker grid")
    _require(manifest.get("walker_count") == spec.walkers and manifest.get("draw_count") == spec.draws,
             "record shape differs from requested evaluation")
    _require(manifest.get("row_count") == spec.expected_rows, "record manifest is incomplete")
    _require(manifest.get("nonfinite_count", 1) == 0, "nonfinite record")
    _require(manifest.get("csv_sha256") == file_sha256(csv_path), "sampled table hash mismatch")
    rows = _read_rows(csv_path, spec)
    if run_dir is not None:
        config = validate_identity(run_dir, spec)
    else:
        config = {}
    by_draw: dict[int, list[float]] = {draw: [] for draw in range(spec.draws)}
    for row in rows:
        by_draw[int(row["draw_index"])].append(float(row["local_energy"]))
    draw_means = [math.fsum(by_draw[draw]) / spec.walkers for draw in range(spec.draws)]
    mean = math.fsum(draw_means) / len(draw_means)
    variance = math.fsum((value - mean) ** 2 for value in draw_means) / len(draw_means)
    if len(draw_means) < 32:
        mcse, blocking_level, mcse_status = None, None, "insufficient_draws_for_blocking_floor_32"
    else:
        mcse, blocking_level = blocking_stderr(draw_means)
        mcse_status = "blocking"
    metrics: dict[str, dict[str, Any]] = {"equivariance": {}, "antisymmetry": {}}
    metrics_path = (run_dir / "metrics.jsonl") if run_dir is not None else None
    if metrics_path is not None and metrics_path.is_file():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            namespace = str(record.get("namespace", "")).lower()
            for task in metrics:
                if task in namespace:
                    metrics[task][namespace] = record.get("metrics", record)
    reference = reference_energy(system, registry)
    return {"schema": "stage1_collection/v2", "status": "complete", "system": system,
            "mean_local_energy": mean, "mcse": mcse, "mcse_status": mcse_status,
            "blocking_level": blocking_level, "reference_energy": reference,
            "reference_error": mean - reference, "local_energy_variance": variance,
            "finite": True, "row_count": len(rows), "equivariance": metrics["equivariance"],
            "antisymmetry": metrics["antisymmetry"], "resolved_config": bool(config)}


def collect_table(path: Path, *, system: str, registry: Mapping[str, Any], walkers: int, draws: int) -> dict[str, Any]:
    """Compatibility wrapper for callers holding the table path."""

    return collect_records(path.parent, system=system, registry=registry,
                           spec=RecordSpec(walkers=walkers, draws=draws))


def main() -> int:
    """Collect one artifact directory from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--system", choices=sorted(SYSTEM_IDS), required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--walkers", type=int, default=4096)
    parser.add_argument("--draws", type=int, default=256)
    args = parser.parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    print(json.dumps(collect_records(args.trajectory_dir, system=args.system, registry=registry,
                                     spec=RecordSpec(walkers=args.walkers, draws=args.draws),
                                     run_dir=args.run_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
