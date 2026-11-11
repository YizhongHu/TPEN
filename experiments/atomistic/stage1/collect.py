"""Collect complete Stage-1 evaluation tables without inventing values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import yaml
from experiments.baselines.statistics import blocking_stderr

from protocol import SYSTEM_IDS


class CollectionError(ValueError):
    """Raised when an evaluation artifact cannot support a reported value."""


def reference_energy(system: str, registry: Mapping[str, Any]) -> float:
    """Read one reference by canonical system key; missing keys are errors."""

    key = SYSTEM_IDS.get(system)
    if key is None:
        raise CollectionError(f"unknown Stage-1 system {system!r}")
    for record in registry.get("systems", []):
        if record.get("id") == key:
            value = record.get("reference_energy_hartree")
            if value is None:
                raise CollectionError(f"reference for {key!r} is absent")
            return float(value)
    raise CollectionError(f"systems.yaml has no required key {key!r}")


def require_grid(rows: list[Mapping[str, Any]], *, walkers: int, draws: int) -> None:
    """Require exactly one record for every walker/draw coordinate."""

    if not rows:
        raise CollectionError("evaluation table is empty")
    expected = {(walker, draw) for walker in range(walkers) for draw in range(draws)}
    observed = {(int(row["walker"]), int(row["draw"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise CollectionError(f"incomplete draw-by-walker grid: expected {len(expected)}, got {len(rows)}")
    if any(not isinstance(row.get("local_energy"), (int, float)) or row["local_energy"] != row["local_energy"]
           for row in rows):
        raise CollectionError("local-energy table contains a nonfinite value")


def collect_table(path: str | Path, *, system: str, registry: Mapping[str, Any]) -> dict[str, Any]:
    """Collect one CSV table, preserving absent optional task metrics."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require_grid(rows, walkers=max(int(row["walker"]) for row in rows) + 1,
                 draws=max(int(row["draw"]) for row in rows) + 1)
    values = [float(row["local_energy"]) for row in rows]
    mean = sum(values) / len(values)
    reference = reference_energy(system, registry)
    mcse, block_level = blocking_stderr(values, allow_below_floor=True)
    return {"system": system, "reference_energy": reference, "mean_local_energy": mean,
            "reference_error": mean - reference, "variance": sum((value - mean) ** 2 for value in values) / len(values),
            "mcse": mcse, "blocking_level": block_level, "finite": True,
            "equivariance": None, "antisymmetry": None}


def main() -> int:
    """Collect one table from the command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--system", choices=SYSTEM_IDS, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))
    print(json.dumps(collect_table(args.table, system=args.system, registry=registry), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
