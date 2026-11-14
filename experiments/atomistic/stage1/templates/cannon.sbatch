#!/usr/bin/env bash
# Canonical generated Stage-1 script template; launch.py fills the markers.
#SBATCH --partition=__PARTITION__
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --no-requeue
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "STAGE1_GUARD missing SLURM_JOB_ID" >&2
  exit 2
fi
REQUESTED_PARTITION="__PARTITION__"
if [[ "$SLURM_JOB_PARTITION" != "$REQUESTED_PARTITION" ]]; then
  echo "STAGE1_GUARD partition delivered=$SLURM_JOB_PARTITION requested=$REQUESTED_PARTITION" >&2
  exit 2
fi
SCRATCH_BASE='__SCRATCH_BASE__'
JOB_ROOT="$SCRATCH_BASE/$USER/$SLURM_JOB_ID"
RESULTS_ROOT='__RESULTS_ROOT__'
HEAD_SHA='__HEAD_SHA__'
if [[ "$RESULTS_ROOT" != "/n/netscratch/kozinsky_lab/Lab/rhu/"* ]]; then
  echo "STAGE1_GUARD results root outside /n/netscratch/kozinsky_lab/Lab/rhu/" >&2
  exit 2
fi
[[ "$JOB_ROOT" == "$SCRATCH_BASE/$USER/$SLURM_JOB_ID" ]] || {
  echo "STAGE1_GUARD invalid JOB_ROOT" >&2
  exit 2
}
mkdir -p -- "$SCRATCH_BASE/$USER"
if [[ -e "$JOB_ROOT" ]]; then
  echo "STAGE1_GUARD job root already exists" >&2
  exit 2
fi
mkdir -- "$JOB_ROOT"
trap 'rm -rf -- "$JOB_ROOT"' EXIT
printf 'job_id=%s host=%s partition=%s head_sha=%s results_root=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$SLURM_JOB_PARTITION" "$HEAD_SHA" "$RESULTS_ROOT"
__COMMAND__
