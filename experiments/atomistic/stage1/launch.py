"""Render, but do not implicitly submit, guarded Stage-1 sbatch scripts."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any, Mapping


def command_for_row(row: Mapping[str, Any], *, python: str, driver: str) -> list[str]:
    """Build the only execution command permitted for a manifest row.

    Protocol sizes and seeds are emitted as Hydra dotlist overrides, making
    the manifest and the checked-in complete config one auditable source of
    runtime truth.
    """

    config = row.get("config")
    overrides = row.get("overrides", {})
    if not isinstance(config, str) or not isinstance(overrides, Mapping):
        raise ValueError("manifest row must contain config and mapping overrides")
    return [python, driver, "--config", config] + [f"{key}={value}" for key, value in overrides.items()]


def shell_command_for_row(row: Mapping[str, Any], *, python: str, driver: str) -> str:
    """Return a shell-quoted row command for launch logs and sbatch scripts."""

    return shlex.join(command_for_row(row, python=python, driver=driver))


def render(*, row: Mapping[str, Any], driver: str, python: str, head_sha: str,
           results_root: str,
           partition: str = "test", allowlist: tuple[str, ...] = ("test", "gpu_test")) -> str:
    """Render one manifest row with explicit identity and results bindings."""

    if not Path(python).is_absolute():
        raise ValueError("--python must be an absolute interpreter path")
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValueError("--head-sha must be a full 40-character git SHA")
    if not results_root or results_root.startswith("experiments/"):
        raise ValueError("--results-root must be outside the checkout")
    if partition not in allowlist:
        raise ValueError(f"partition {partition!r} is not in the explicit allowlist {allowlist!r}")
    command = shell_command_for_row(row, python=python, driver=driver)
    row_root = row.get("overrides", {}).get("run.root")
    if not isinstance(row_root, str) or not row_root.startswith(results_root.rstrip("/") + "/"):
        raise ValueError("manifest row run.root must be below the explicit results root")
    return f'''#!/usr/bin/env bash
# Stage-1 generated script; no production partition is permitted.
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
set -euo pipefail
JOB_ROOT="/scratch/$USER/$SLURM_JOB_ID"
PYTHON={python!r}
RESULTS_ROOT={results_root!r}
REQUESTED_PARTITION={partition!r}
HEAD_SHA={head_sha!r}
trap 'rm -rf -- "$JOB_ROOT"' EXIT
if [[ "${{SLURM_JOB_PARTITION:-}}" != "$REQUESTED_PARTITION" ]]; then exit 2; fi
: "${{SLURM_JOB_ID:?must run inside Slurm}}"
export HEAD_SHA
export TPEN_SOURCE_SHA="$HEAD_SHA"
export STAGE1_RESULTS_ROOT="$RESULTS_ROOT"
[[ "$JOB_ROOT" == "/scratch/$USER/$SLURM_JOB_ID" ]]
mkdir -p "$JOB_ROOT"
set +e
{command}
rc=$?
set -e
printf 'job_id=%s host=%s partition=%s requested_partition=%s head=%s exit=%s\\n' \\
  "$SLURM_JOB_ID" "$(hostname)" "$SLURM_JOB_PARTITION" "$REQUESTED_PARTITION" "$HEAD_SHA" "$rc"
exit "$rc"
'''


def main() -> int:
    """Write one script; submission is a separate, explicit operator action."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--row", type=Path, required=True,
                        help="JSON manifest row")
    parser.add_argument("--python", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--partition", default="test")
    args = parser.parse_args()
    row = json.loads(args.row.read_text(encoding="utf-8"))
    args.output.write_text(render(driver=args.driver, row=row, python=args.python,
                                  head_sha=args.head_sha, results_root=args.results_root,
                                  partition=args.partition), encoding="utf-8")
    args.output.chmod(0o750)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
