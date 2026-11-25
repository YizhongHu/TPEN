#!/bin/bash
# One complete CPU smoke control for Slice 1.  The caller supplies only the
# unique evidence root, staged source tar, and immutable shared environment.
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT must be an exact /n/netscratch/.../Lab/rhu/<tag> root}"
: "${STAGE_ROOT:?STAGE_ROOT must be an exact /n/netscratch/.../Lab/rhu/<tag> root}"
: "${SHARED_ENV_PREFIX:?SHARED_ENV_PREFIX must name the immutable shared env}"

case "$RUN_ROOT" in
  /n/netscratch/kozinsky_lab/Lab/rhu/*) ;;
  *) echo "REFUSE: RUN_ROOT outside approved Netscratch user root" >&2; exit 2 ;;
esac
case "$STAGE_ROOT" in
  /n/netscratch/kozinsky_lab/Lab/rhu/*) ;;
  *) echo "REFUSE: STAGE_ROOT outside approved Netscratch user root" >&2; exit 2 ;;
esac

WORK_ROOT="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$WORK_ROOT"
cleanup() { rm -rf "$WORK_ROOT"; }
trap cleanup EXIT

tar -xf "$STAGE_ROOT/source.tar" -C "$WORK_ROOT"
CHECKOUT="$WORK_ROOT/tpen"
cd "$CHECKOUT"

ENV_PYTHON="$SHARED_ENV_PREFIX/tpen-853c50ca90a2-cpu-py3.12/bin/python"
test -x "$ENV_PYTHON"
"$ENV_PYTHON" -m venv --without-pip "$WORK_ROOT/overlay"
OVERLAY_PYTHON="$WORK_ROOT/overlay/bin/python"
SHARED_SITE="$($ENV_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
OVERLAY_SITE="$($OVERLAY_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$SHARED_SITE" > "$OVERLAY_SITE/tpen-shared-env.pth"
printf '%s\n' "$CHECKOUT" > "$OVERLAY_SITE/tpen-checkout.pth"
export PATH="$WORK_ROOT/overlay/bin:$PATH"

echo "job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION host=$(hostname)"
echo "python=$(command -v python)"
python -c 'import sys, torch; print("python_version=" + sys.version.split()[0]); print("torch_version=" + torch.__version__); print("cuda_available=" + str(torch.cuda.is_available()))'
: "${TPEN_SOURCE_SHA:?TPEN_SOURCE_SHA must be the exact staged source head}"
python scripts/cannon/sr_replay_benchmark.py \
  --device cpu \
  --smoke \
  --output "$RUN_ROOT/sr-replay-benchmark.json"
