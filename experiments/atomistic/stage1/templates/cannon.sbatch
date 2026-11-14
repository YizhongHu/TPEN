#!/usr/bin/env bash
# Reference template; launch.py renders the concrete row command.
set -euo pipefail
: "${SLURM_JOB_ID:?must run inside Slurm}"
: "${SLURM_JOB_PARTITION:?Slurm did not provide a delivered partition}"
REQUESTED_PARTITION="__PARTITION__"
[[ "$SLURM_JOB_PARTITION" == "$REQUESTED_PARTITION" ]]
JOB_ROOT="/scratch/$USER/$SLURM_JOB_ID"
RESULTS_ROOT="__RESULTS_ROOT__"
NETSCRATCH_ROOT="__NETSCRATCH_ROOT__"
[[ "$RESULTS_ROOT" == "$NETSCRATCH_ROOT"/* ]]
trap 'rm -rf -- "$JOB_ROOT"' EXIT
mkdir -p "$JOB_ROOT"
printf 'job_id=%s host=%s partition=%s head_sha=%s\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$SLURM_JOB_PARTITION" "__HEAD_SHA__"
# __COMMAND__
