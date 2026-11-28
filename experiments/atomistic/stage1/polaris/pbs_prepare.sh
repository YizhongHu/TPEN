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
RESULTS_PREFIX_CANONICAL=""
STAGE1_PATHS_SH="${STAGE1_PATHS_SH:-}"
STAGE1_PATHS_SHA256="${STAGE1_PATHS_SHA256:-}"
STAGED_PREP_PATH="${STAGED_PREP_PATH:-}"
STAGED_WRAPPER_PATH="${STAGED_WRAPPER_PATH:-}"
STAGED_EVAL_PREP_PATH="${STAGED_EVAL_PREP_PATH:-}"
SOURCE_BUNDLE_PATH="${SOURCE_BUNDLE_PATH:-}"
SOURCE_BUNDLE_SHA256="${SOURCE_BUNDLE_SHA256:-}"
SOURCE_SHA="${SOURCE_SHA:-}"
RESULTS_ROOT="${RESULTS_ROOT:-}"
SYSTEM="${SYSTEM:-}"
LINEAGE_SEEDS="${LINEAGE_SEEDS:-}"
SMOKE_RESULTS_ROOT="${SMOKE_RESULTS_ROOT:-}"
SMOKE_SELECTED_ROW="${SMOKE_SELECTED_ROW:-}"
EVAL_PREP_PATH="${EVAL_PREP_PATH:-}"
PYBIN="${PYBIN:-/soft/applications/conda/2025-09-25/mconda3/bin/python}"
CHECKOUT_ROOT="${CHECKOUT_ROOT:-${RESULTS_ROOT}/checkout}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${RESULTS_ROOT}/uv-cache}"
PRIME_ENV="${PRIME_ENV:-${RESULTS_ROOT}/prime-env}"
BUNDLE_INPUT_PATH="${BUNDLE_INPUT_PATH:-${RESULTS_ROOT}/inputs/TPEN-${SOURCE_SHA}.bundle}"
PHASE1_MANIFEST="${PHASE1_MANIFEST:-${RESULTS_ROOT}/plan/phase1-manifest.json}"
SELECTED_ROW="${SELECTED_ROW:-${RESULTS_ROOT}/plan/selected-train.json}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${RESULTS_ROOT}/launch}"
WRAPPER_PATH="${WRAPPER_PATH:-${RESULTS_ROOT}/inputs/pbs_stage1.sh}"

die() {
  printf 'POLARIS_PREP FAIL %s\n' "$*" >&2
  exit 2
}

[[ -n "$STAGE1_PATHS_SH" ]] ||
  die "STAGE1_PATHS_SH is required; PBS may spool this script away from its inputs"
[[ "$STAGE1_PATHS_SH" == /* ]] ||
  die "STAGE1_PATHS_SH must be an absolute path: $STAGE1_PATHS_SH"
[[ -f "$STAGE1_PATHS_SH" && ! -L "$STAGE1_PATHS_SH" ]] ||
  die "STAGE1_PATHS_SH is not a regular file: $STAGE1_PATHS_SH"
[[ "$STAGE1_PATHS_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "STAGE1_PATHS_SHA256 must be a recorded lowercase SHA-256"
STAGE1_PATHS_ACTUAL_SHA256="$(sha256sum "$STAGE1_PATHS_SH")"
[[ "${STAGE1_PATHS_ACTUAL_SHA256%% *}" == "$STAGE1_PATHS_SHA256" ]] ||
  die "STAGE1_PATHS_SH SHA-256 mismatch: $STAGE1_PATHS_SH"
source "$STAGE1_PATHS_SH"

require_value() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "$name is required"
}

require_value SOURCE_BUNDLE_PATH
require_value SOURCE_BUNDLE_SHA256
require_value SOURCE_SHA
require_value RESULTS_ROOT
require_value SYSTEM
require_value LINEAGE_SEEDS
[[ -n "${PBS_JOBID:-}" ]] || die "PBS_JOBID is required inside the prep allocation"
[[ -z "${PBS_QUEUE:-}" || "$PBS_QUEUE" == debug ]] || die "delivered queue is not debug: ${PBS_QUEUE}"
[[ "$SOURCE_SHA" =~ ^[0-9a-f]{40}$ ]] || die "SOURCE_SHA must be a full lowercase SHA"
[[ "$SOURCE_BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  die "SOURCE_BUNDLE_SHA256 must be a recorded lowercase SHA-256"
[[ "$SYSTEM" == he || "$SYSTEM" == h2 ]] || die "SYSTEM must be he or h2"
RESULTS_PREFIX_CANONICAL="$(canonicalize_path "$RESULTS_PREFIX")" ||
  die "could not canonicalize results prefix: $RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$RESULTS_ROOT" RESULTS_ROOT "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$CHECKOUT_ROOT" CHECKOUT_ROOT "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$UV_CACHE_DIR" UV_CACHE_DIR "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$PRIME_ENV" PRIME_ENV "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$PHASE1_MANIFEST" PHASE1_MANIFEST "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$SELECTED_ROW" SELECTED_ROW "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$LAUNCH_ROOT" LAUNCH_ROOT "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$WRAPPER_PATH" WRAPPER_PATH "$RESULTS_PREFIX"
assert_results_path "$RESULTS_PREFIX_CANONICAL" "$BUNDLE_INPUT_PATH" BUNDLE_INPUT_PATH "$RESULTS_PREFIX"
if [[ -n "$SMOKE_RESULTS_ROOT" || -n "$SMOKE_SELECTED_ROW" ]]; then
  [[ -n "$SMOKE_RESULTS_ROOT" && -n "$SMOKE_SELECTED_ROW" ]] ||
    die "SMOKE_RESULTS_ROOT and SMOKE_SELECTED_ROW must be supplied together"
  assert_results_path "$RESULTS_PREFIX_CANONICAL" "$SMOKE_RESULTS_ROOT" SMOKE_RESULTS_ROOT "$RESULTS_PREFIX"
  assert_results_path "$RESULTS_PREFIX_CANONICAL" "$SMOKE_SELECTED_ROW" SMOKE_SELECTED_ROW "$RESULTS_PREFIX"
fi
if [[ -n "$EVAL_PREP_PATH" ]]; then
  assert_results_path "$RESULTS_PREFIX_CANONICAL" "$EVAL_PREP_PATH" EVAL_PREP_PATH "$RESULTS_PREFIX"
fi

[[ ! -e "$CHECKOUT_ROOT" && ! -L "$CHECKOUT_ROOT" ]] || die "CHECKOUT_ROOT already exists: $CHECKOUT_ROOT"
[[ ! -e "$PRIME_ENV" && ! -L "$PRIME_ENV" ]] || die "PRIME_ENV already exists: $PRIME_ENV"
[[ -f "$SOURCE_BUNDLE_PATH" && ! -L "$SOURCE_BUNDLE_PATH" ]] ||
  die "staged source bundle is missing or not a regular file: $SOURCE_BUNDLE_PATH"
mkdir -p -- "$RESULTS_ROOT"/inputs "$RESULTS_ROOT"/logs "$RESULTS_ROOT"/plan "$LAUNCH_ROOT" "$UV_CACHE_DIR"
if [[ -n "$SMOKE_RESULTS_ROOT" ]]; then
  mkdir -p -- "$SMOKE_RESULTS_ROOT"/logs "$SMOKE_RESULTS_ROOT"/plan
fi
if [[ -n "$STAGED_PREP_PATH" ]]; then
  [[ -f "$STAGED_PREP_PATH" ]] || die "staged preparation script is missing: $STAGED_PREP_PATH"
  cp -- "$STAGED_PREP_PATH" "$RESULTS_ROOT/inputs/pbs_prepare.sh"
fi
cp -- "$SOURCE_BUNDLE_PATH" "$BUNDLE_INPUT_PATH"
SOURCE_BUNDLE_ACTUAL_SHA256="$(sha256sum "$BUNDLE_INPUT_PATH")"
[[ "${SOURCE_BUNDLE_ACTUAL_SHA256%% *}" == "$SOURCE_BUNDLE_SHA256" ]] ||
  die "source bundle SHA-256 mismatch: $SOURCE_BUNDLE_PATH"
sha256sum "$BUNDLE_INPUT_PATH" > "$RESULTS_ROOT/inputs/TPEN-${SOURCE_SHA}.bundle.sha256"
git bundle verify "$BUNDLE_INPUT_PATH" > "$RESULTS_ROOT/inputs/TPEN-${SOURCE_SHA}.bundle.verify"
if [[ -n "$STAGED_WRAPPER_PATH" ]]; then
  [[ -f "$STAGED_WRAPPER_PATH" ]] || die "staged execution wrapper is missing: $STAGED_WRAPPER_PATH"
  cp -- "$STAGED_WRAPPER_PATH" "$WRAPPER_PATH"
fi
if [[ "$STAGE1_PATHS_SH" != "$RESULTS_ROOT/inputs/pbs_paths.sh" ]]; then
  cp -- "$STAGE1_PATHS_SH" "$RESULTS_ROOT/inputs/pbs_paths.sh"
fi
[[ -f "$WRAPPER_PATH" ]] || die "execution wrapper is missing: $WRAPPER_PATH"
sha256sum "$0" > "$RESULTS_ROOT/inputs/pbs_prepare.sh.sha256"
sha256sum "$STAGE1_PATHS_SH" > "$RESULTS_ROOT/inputs/pbs_paths.sh.sha256"
sha256sum "$WRAPPER_PATH" > "$RESULTS_ROOT/inputs/pbs_stage1.sh.sha256"
if [[ -n "$EVAL_PREP_PATH" ]]; then
  if [[ -n "$STAGED_EVAL_PREP_PATH" ]]; then
    [[ -f "$STAGED_EVAL_PREP_PATH" ]] ||
      die "staged evaluation preparation script is missing: $STAGED_EVAL_PREP_PATH"
    cp -- "$STAGED_EVAL_PREP_PATH" "$EVAL_PREP_PATH"
  fi
  [[ -f "$EVAL_PREP_PATH" ]] || die "evaluation preparation script is missing: $EVAL_PREP_PATH"
  sha256sum "$EVAL_PREP_PATH" > "$RESULTS_ROOT/inputs/pbs_select_eval.sh.sha256"
fi

printf 'POLARIS_PREP PBS_JOBID=%s PBS_QUEUE=%s HOST=%s UTC=%s\n' \
  "${PBS_JOBID:-}" "${PBS_QUEUE:-}" "$(hostname -f 2>/dev/null || hostname)" "$(date -u +%FT%TZ)"
printf 'POLARIS_PREP SOURCE_SHA=%s RESULTS_ROOT=%s SYSTEM=%s\n' "$SOURCE_SHA" "$RESULTS_ROOT" "$SYSTEM"

git clone --no-checkout "$BUNDLE_INPUT_PATH" "$CHECKOUT_ROOT"
git -C "$CHECKOUT_ROOT" checkout --detach "$SOURCE_SHA"
[[ "$(git -C "$CHECKOUT_ROOT" rev-parse HEAD)" == "$SOURCE_SHA" ]] || die "detached checkout does not match SOURCE_SHA"
[[ -z "$(git -C "$CHECKOUT_ROOT" status --porcelain)" ]] || die "checkout is dirty"
[[ "$(git -C "$CHECKOUT_ROOT" rev-parse --is-shallow-repository)" == false ]] || die "checkout must be a full clone"
[[ -z "$(git -C "$CHECKOUT_ROOT" config --local --get-regexp '^remote\..*\.promisor$' || true)" ]] || die "checkout has a promisor remote"
[[ -z "$(git -C "$CHECKOUT_ROOT" config --local --get extensions.partialclone || true)" ]] || die "checkout is a partial clone"

# ALCF Polaris Running Jobs guide, accessed 2026-11-28, documents this exact
# proxy set for compute-node internet access. SOAP E1 measured it working for
# the facility Python/uv path; do not inherit stale proxy values.
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"
printf 'HTTP_PROXY=%s\nHTTPS_PROXY=%s\nhttp_proxy=%s\nhttps_proxy=%s\nftp_proxy=%s\nno_proxy=%s\n' \
  "$HTTP_PROXY" "$HTTPS_PROXY" "$http_proxy" "$https_proxy" "$ftp_proxy" "$no_proxy" \
  > "$RESULTS_ROOT/inputs/proxy.env"
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
  "$CHECKOUT_ROOT" "$PRIME_ENV/bin/python" "$SOURCE_SHA" \
  "$SMOKE_RESULTS_ROOT" "$SMOKE_SELECTED_ROW" <<'PY'
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

from experiments.atomistic.stage1.launch import command_for_row

manifest_path, selected_path, launch_root, checkout_root, python, source_sha = map(Path, sys.argv[1:7])
smoke_results_root_arg, smoke_selected_arg = sys.argv[7:9]
python = str(python)
source_sha = str(source_sha)
if bool(smoke_results_root_arg) != bool(smoke_selected_arg):
    raise SystemExit("SMOKE_RESULTS_ROOT and SMOKE_SELECTED_ROW must be supplied together")
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
if smoke_results_root_arg:
    from experiments.atomistic.stage1.protocol import derive_smoke_request

    smoke_results_root = Path(smoke_results_root_arg)
    smoke_selected_path = Path(smoke_selected_arg)
    smoke_request = derive_smoke_request(
        train_rows[0], results_root=str(smoke_results_root), stop_at=50_000
    )
    smoke_bytes = json.dumps(smoke_request, sort_keys=True, indent=2).encode() + b"\n"
    smoke_selected_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_selected_path.write_bytes(smoke_bytes)
    (smoke_selected_path.parent / (smoke_selected_path.name + ".sha256")).write_text(
        f"{hashlib.sha256(smoke_bytes).hexdigest()}  {smoke_selected_path.name}\n",
        encoding="utf-8",
    )
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
