#!/usr/bin/env bash
# Generated Stage-1 verification/science template. Values are injected by launch.py.
set -euo pipefail
trap 'rm -rf -- "$JOB_ROOT"' EXIT
if [[ "${SLURM_JOB_PARTITION:-}" != "test" && "${SLURM_JOB_PARTITION:-}" != "gpu_test" ]]; then
  echo "Stage-1 jobs may run only on test or gpu_test" >&2; exit 2
fi
: "${SLURM_JOB_ID:?must run inside Slurm}"
: "${JOB_ROOT:?job-local scratch root is required}"
: "${REQUESTED_PARTITION:?requested partition is required}"
if [[ "${SLURM_JOB_PARTITION:-}" != "$REQUESTED_PARTITION" ]]; then
  echo "delivered partition does not match requested partition" >&2; exit 2
fi
mkdir -p "$JOB_ROOT"
exec "$PYTHON" "$DRIVER" --config "$CONFIG"
