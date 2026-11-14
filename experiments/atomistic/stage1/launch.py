"""Render guarded Stage-1 sbatch scripts; submission is always explicit."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Mapping

from experiments.atomistic.stage1.protocol import _require_sha, identity_expectations

ALLOWED_PARTITIONS = ("test", "gpu_test")
FACILITY_RESULTS_PREFIX = "/n/netscratch/kozinsky_lab/Lab/rhu/"


def command_for_row(row: Mapping[str, Any], *, python: str, driver: str) -> str:
    """Build the only driver command permitted by the launcher."""

    if row.get("kind") not in {"smoke", "train", "eval"}:
        raise ValueError("manifest row has an invalid kind")
    _require_sha(row.get("source_sha"), length=40, field="source_sha")
    _require_sha(row.get("config_sha256"), length=64, field="config_sha256")
    return shlex.join([python, driver, "--config", str(row["config"])])


def render(*, row: Mapping[str, Any], driver: str, python: str, head_sha: str,
           results_root: str, partition: str = "test") -> str:
    """Render a fail-closed script for one identity-bound row."""

    if not Path(python).is_absolute():
        raise ValueError("python must be an absolute path")
    _require_sha(head_sha, length=40, field="head_sha")
    if partition not in ALLOWED_PARTITIONS:
        raise ValueError(f"partition must be one of {ALLOWED_PARTITIONS}")
    if not results_root.startswith(FACILITY_RESULTS_PREFIX):
        raise ValueError("results_root must be below the approved Netscratch prefix")
    row_root = row.get("results_root")
    if not isinstance(row_root, str) or not row_root.startswith(results_root.rstrip("/") + "/"):
        raise ValueError("row results_root is not bound to the rendered root")
    if row.get("source_sha") != head_sha:
        raise ValueError("manifest source_sha does not match rendered head_sha")
    if row.get("kind") == "eval":
        identity_expectations(row)
    command = command_for_row(row, python=python, driver=driver)
    template_path = Path(__file__).with_name("templates") / "cannon.sbatch"
    template = template_path.read_text(encoding="utf-8")
    for marker, value in {
        "__PARTITION__": partition,
        "__RESULTS_ROOT__": results_root,
        "__HEAD_SHA__": head_sha,
        "__COMMAND__": command,
    }.items():
        template = template.replace(marker, value)
    return template


def main() -> int:
    """Write a script without submitting it."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=Path, required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--partition", default="test")
    args = parser.parse_args()
    row = json.loads(args.row.read_text(encoding="utf-8"))
    args.output.write_text(render(row=row, driver=args.driver, python=args.python,
                                  head_sha=args.head_sha, results_root=args.results_root,
                                  partition=args.partition), encoding="utf-8")
    args.output.chmod(0o750)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
