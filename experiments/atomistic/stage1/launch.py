"""Render, but do not implicitly submit, guarded Stage-1 sbatch scripts."""

from __future__ import annotations

import argparse
from pathlib import Path


def render(*, driver: str, config: str, python: str, job_root: str,
           partition: str = "test", allowlist: tuple[str, ...] = ("test", "gpu_test")) -> str:
    """Render a test/gpu_test-only script with explicit runtime bindings."""

    if not Path(python).is_absolute():
        raise ValueError("--python must be an absolute interpreter path")
    if partition not in allowlist:
        raise ValueError(f"partition {partition!r} is not in the explicit allowlist {allowlist!r}")
    return f'''#!/usr/bin/env bash
# Stage-1 generated script; no production partition is permitted.
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
set -euo pipefail
JOB_ROOT={job_root!r}
PYTHON={python!r}
DRIVER={driver!r}
CONFIG={config!r}
REQUESTED_PARTITION={partition!r}
trap 'rm -rf -- "$JOB_ROOT"' EXIT
if [[ "${{SLURM_JOB_PARTITION:-}}" != "$REQUESTED_PARTITION" ]]; then exit 2; fi
: "${{SLURM_JOB_ID:?must run inside Slurm}}"
mkdir -p "$JOB_ROOT"
exec "$PYTHON" "$DRIVER" --config "$CONFIG"
'''


def main() -> int:
    """Write one script; submission is a separate, explicit operator action."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--partition", default="test")
    args = parser.parse_args()
    args.output.write_text(render(driver=args.driver, config=args.config, python=args.python,
                                  job_root=args.job_root, partition=args.partition), encoding="utf-8")
    args.output.chmod(0o750)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
