#!/usr/bin/env bash
# One short Polaris debug allocation for source clone, uv cache priming, and
# Stage-1 row materialization. Login nodes only submit and inspect this job.
#PBS -A HetRxnEnergy
#PBS -q debug
#PBS -l select=1:system=polaris:ncpus=4
#PBS -l walltime=00:20:00
#PBS -l filesystems=home:eagle
#PBS -k doe

set -Eeuo pipefail

RESULTS_PREFIX="/eagle/HetRxnEnergy/rhu/"
SOURCE_REPO="${SOURCE_REPO:-}"
SOURCE_SHA="${SOURCE_SHA:-}"
RESULTS_ROOT="${RESULTS_ROOT:-}"
SYSTEM="${SYSTEM:-}"
LINEAGE_SEEDS="${LINEAGE_SEEDS:-}"
PYBIN="${PYBIN:-/soft/applications/conda/2025-09-25/mconda3/bin/python}"
CHECKOUT_ROOT="${CHECKOUT_ROOT:-${RESULTS_ROOT}/checkout}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${RESULTS_ROOT}/uv-cache}"
PRIME_ENV="${PRIME_ENV:-${RESULTS_ROOT}/prime-env}"
PHASE1_MANIFEST="${PHASE1_MANIFEST:-${RESULTS_ROOT}/plan/phase1-manifest.json}"
SELECTED_ROW="${SELECTED_ROW:-${RESULTS_ROOT}/plan/selected-train.json}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${RESULTS_ROOT}/launch}"
WRAPPER_PATH="${WRAPPER_PATH:-${RESULTS_ROOT}/inputs/pbs_stage1.sh}"

die() {
  printf 'POLARIS_PREP FAIL %s\n' "$*" >&2
  exit 2
}

require_value() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "$name is required"
}

assert_results_path() {
  local path="$1" label="$2"
  [[ "$path" == "$RESULTS_PREFIX"* ]] || die "$label is outside $RESULTS_PREFIX: $path"
  [[ "$path" != "$RESULTS_PREFIX" ]] || die "$label must name a child path"
  [[ "$path" != */../* && "$path" != ../* && "$path" != */.. ]] ||
    die "$label contains a parent traversal: $path"
}

require_value SOURCE_REPO
require_value SOURCE_SHA
require_value RESULTS_ROOT
require_value SYSTEM
require_value LINEAGE_SEEDS
[[ -n "${PBS_JOBID:-}" ]] || die "PBS_JOBID is required inside the prep allocation"
[[ -z "${PBS_QUEUE:-}" || "$PBS_QUEUE" == debug ]] || die "delivered queue is not debug: ${PBS_QUEUE}"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || die "SOURCE_SHA must be a full lowercase SHA"
[[ "$SYSTEM" == he || "$SYSTEM" == h2 ]] || die "SYSTEM must be he or h2"
assert_results_path "$RESULTS_ROOT" RESULTS_ROOT
assert_results_path "$CHECKOUT_ROOT" CHECKOUT_ROOT
assert_results_path "$UV_CACHE_DIR" UV_CACHE_DIR
assert_results_path "$PRIME_ENV" PRIME_ENV
assert_results_path "$PHASE1_MANIFEST" PHASE1_MANIFEST
assert_results_path "$SELECTED_ROW" SELECTED_ROW
assert_results_path "$LAUNCH_ROOT" LAUNCH_ROOT
assert_results_path "$WRAPPER_PATH" WRAPPER_PATH

[[ ! -e "$CHECKOUT_ROOT" && ! -L "$CHECKOUT_ROOT" ]] || die "CHECKOUT_ROOT already exists: $CHECKOUT_ROOT"
[[ ! -e "$PRIME_ENV" && ! -L "$PRIME_ENV" ]] || die "PRIME_ENV already exists: $PRIME_ENV"
mkdir -p -- "$RESULTS_ROOT"/inputs "$RESULTS_ROOT"/logs "$RESULTS_ROOT"/plan "$LAUNCH_ROOT" "$UV_CACHE_DIR"
[[ -f "$WRAPPER_PATH" ]] || die "execution wrapper is missing: $WRAPPER_PATH"
sha256sum "$0" > "$RESULTS_ROOT/inputs/pbs_prepare.sh.sha256"
sha256sum "$WRAPPER_PATH" > "$RESULTS_ROOT/inputs/pbs_stage1.sh.sha256"

printf 'POLARIS_PREP PBS_JOBID=%s PBS_QUEUE=%s HOST=%s UTC=%s\n' \
  "${PBS_JOBID:-}" "${PBS_QUEUE:-}" "$(hostname -f 2>/dev/null || hostname)" "$(date -u +%FT%TZ)"
printf 'POLARIS_PREP SOURCE_SHA=%s RESULTS_ROOT=%s SYSTEM=%s\n' "$SOURCE_SHA" "$RESULTS_ROOT" "$SYSTEM"

git clone "$SOURCE_REPO" "$CHECKOUT_ROOT"
git -C "$CHECKOUT_ROOT" checkout --detach "$SOURCE_SHA"
[[ "$(git -C "$CHECKOUT_ROOT" rev-parse HEAD)" == "$SOURCE_SHA" ]] || die "detached checkout does not match SOURCE_SHA"
[[ -z "$(git -C "$CHECKOUT_ROOT" status --porcelain)" ]] || die "checkout is dirty"
[[ "$(git -C "$CHECKOUT_ROOT" rev-parse --is-shallow-repository)" == false ]] || die "checkout must be a full clone"
[[ -z "$(git -C "$CHECKOUT_ROOT" config --local --get-regexp '^remote\..*\.promisor$' || true)" ]] || die "checkout has a promisor remote"
[[ -z "$(git -C "$CHECKOUT_ROOT" config --local --get extensions.partialclone || true)" ]] || die "checkout is a partial clone"

export UV_CACHE_DIR
export UV_PYTHON="$PYBIN" UV_NO_MANAGED_PYTHON=1
export UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
"$PYBIN" -m uv lock --check --project "$CHECKOUT_ROOT"
UV_PROJECT_ENVIRONMENT="$PRIME_ENV" \
  "$PYBIN" -m uv sync --project "$CHECKOUT_ROOT" --extra cu128 --locked

cd "$CHECKOUT_ROOT"
LINEAGE_SEEDS="${LINEAGE_SEEDS//:/ }"
read -r -a lineage_seeds <<< "$LINEAGE_SEEDS"
"$PYBIN" -m experiments.atomistic.stage1.plan \
  --phase 1 --output "$PHASE1_MANIFEST" --source-sha "$SOURCE_SHA" \
  --results-root "$RESULTS_ROOT" --systems "$SYSTEM" --device cuda \
  --lineage-seeds "${lineage_seeds[@]}"

"$PYBIN" - "$PHASE1_MANIFEST" "$SELECTED_ROW" "$LAUNCH_ROOT" \
  "$CHECKOUT_ROOT" "$PRIME_ENV/bin/python" "$SOURCE_SHA" <<'PY'
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

from experiments.atomistic.stage1.launch import command_for_row

manifest_path, selected_path, launch_root, checkout_root, python, source_sha = map(Path, sys.argv[1:])
python = str(python)
source_sha = str(source_sha)
document = json.loads(manifest_path.read_text(encoding="utf-8"))
rows = document.get("rows")
if not isinstance(rows, list):
    raise SystemExit("phase-1 manifest rows must be a list")
train_rows = [row for row in rows if isinstance(row, dict) and row.get("kind") == "train"]
if len(train_rows) != 3:
    raise SystemExit(f"expected three phase-1 train rows, got {len(train_rows)}")
if any(row.get("source_sha") != source_sha for row in train_rows):
    raise SystemExit("phase-1 row source_sha does not match SOURCE_SHA")
launch_root.mkdir(parents=True, exist_ok=True)
selected_path.write_text(json.dumps(train_rows[0], sort_keys=True, indent=2) + "\n", encoding="utf-8")
receipt_rows = []
for row in train_rows:
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or re.fullmatch(r"[A-Za-z0-9_.-]+", row_id) is None:
        raise SystemExit(f"unsafe row_id: {row_id!r}")
    row_path = launch_root / f"row-{row_id}.json"
    command_path = launch_root / f"row-{row_id}.command"
    row_bytes = json.dumps(row, sort_keys=True, indent=2).encode() + b"\n"
    row_path.write_bytes(row_bytes)
    command = command_for_row(row, python=python, checkout_root=str(checkout_root))
    command_bytes = (shlex.join(shlex.split(command)) + "\n").encode()
    command_path.write_bytes(command_bytes)
    receipt_rows.append({"row_id": row_id, "row_json": str(row_path),
                         "row_sha256": hashlib.sha256(row_bytes).hexdigest(),
                         "command": str(command_path),
                         "command_sha256": hashlib.sha256(command_bytes).hexdigest()})
(launch_root / "render-receipt.json").write_text(
    json.dumps({"source_sha": source_sha, "rows": receipt_rows}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

printf 'POLARIS_PREP complete=PASS CHECKOUT_ROOT=%s UV_CACHE_DIR=%s PHASE1_MANIFEST=%s SELECTED_ROW=%s\n' \
  "$CHECKOUT_ROOT" "$UV_CACHE_DIR" "$PHASE1_MANIFEST" "$SELECTED_ROW"
