#!/usr/bin/env bash
set -u

test "$#" -eq 4 || { echo 'usage: corpus_baseline_job.sh RUN_ROOT SOURCE REVISION HARNESS' >&2; exit 2; }
RUN_ROOT=$1
SOURCE=$2
REVISION=$3
HARNESS=$4
test -n "${SLURM_JOB_ID:-}" || { echo 'SLURM_JOB_ID is required' >&2; exit 90; }
UV=${UV_BIN:-$HOME/.local/bin/uv}
test -x "$UV" || { echo "uv not executable: $UV" >&2; exit 91; }
export PATH="$RUN_ROOT/venv-trunk/bin:$PATH"
exec "$UV" run --no-project python "$HARNESS" \
  --revision "$REVISION" \
  --run-root "$RUN_ROOT" \
  --uv "$UV" \
  --source "$SOURCE"
