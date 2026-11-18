#!/usr/bin/env bash
# Polaris PBS wrapper for the direct Stage-1 He/H2 route.
#
# The wrapper owns PBS admission, Eagle storage, four-GPU/one-visible-GPU
# guards, and row-level CPU/GPU placement.  Stage-1 identity and argv remain
# owned by experiments.atomistic.stage1.launch.command_for_row.
#
# The submission driver supplies MODE=train|eval, SOURCE_SHA, SOURCE_REPO,
# RESULTS_ROOT, PLAN_MANIFEST, and EXPECTED_QUEUE.  The checked-in header is a
# debug/evaluation shape; the training runbook supplies the capacity queue and
# wall cap explicitly at qsub time.
#PBS -A HetRxnEnergy
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:eagle
#PBS -k doe

set -Eeuo pipefail

RESULTS_PREFIX="/eagle/HetRxnEnergy/rhu/"
MODE="${MODE:-${1:-}}"
SOURCE_SHA="${SOURCE_SHA:-}"
SOURCE_REPO="${SOURCE_REPO:-https://github.com/YizhongHu/TPEN.git}"
RESULTS_ROOT="${RESULTS_ROOT:-}"
PLAN_MANIFEST="${PLAN_MANIFEST:-}"
EXPECTED_QUEUE="${EXPECTED_QUEUE:-}"
PBS_JOBID="${PBS_JOBID:-}"
PBS_QUEUE="${PBS_QUEUE:-}"
DRY_EXEC="${DRY_EXEC:-0}"
PYBIN="${PYBIN:-/soft/applications/conda/2025-09-25/mconda3/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-}"
GUARD_PYTHON_BIN="${GUARD_PYTHON_BIN:-}"
NVIDIA_SMI_BIN="${NVIDIA_SMI_BIN:-nvidia-smi}"
BOOT_ROOT="${BOOT_ROOT:-}"

die() {
  printf '%s\n' "POLARIS_GUARD FAIL $*" >&2
  exit 2
}

require_value() {
  local name="$1" value="${!1:-}"
  [[ -n "$value" ]] || die "$name is required"
}

assert_results_root() {
  case "$RESULTS_ROOT" in
    "$RESULTS_PREFIX"*) ;;
    *) die "results root outside Eagle prefix: $RESULTS_ROOT" ;;
  esac
}

assert_pbs_context() {
  require_value PBS_JOBID
  require_value PBS_QUEUE
  require_value EXPECTED_QUEUE
  [[ "$PBS_QUEUE" == "$EXPECTED_QUEUE" ]] ||
    die "delivered queue=$PBS_QUEUE expected=$EXPECTED_QUEUE"
  printf 'POLARIS_GUARD PBS_JOBID=%s PBS_QUEUE=%s EXPECTED_QUEUE=%s\n' \
    "$PBS_JOBID" "$PBS_QUEUE" "$EXPECTED_QUEUE"
}

write_job_receipt() {
  [[ "$DRY_EXEC" == 1 ]] && return 0
  mkdir -p -- "$RESULTS_ROOT/logs" "$RESULTS_ROOT/launch" "$RESULTS_ROOT/plan"
  {
    printf 'PBS_JOBID=%s\n' "$PBS_JOBID"
    printf 'PBS_QUEUE=%s\n' "$PBS_QUEUE"
    printf 'HOST=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'UTC=%s\n' "$(date -u +%FT%TZ)"
    printf 'SOURCE_SHA=%s\n' "$SOURCE_SHA"
    printf 'MODE=%s\n' "$MODE"
    printf 'RESULTS_ROOT=%s\n' "$RESULTS_ROOT"
  } > "$RESULTS_ROOT/logs/polaris-${MODE}-${PBS_JOBID}.env"
}

guard_gpus() {
  require_value GUARD_PYTHON_BIN
  local gpu_info gpu_inventory mig_info line count i
  local -a gpu_lines inventory_lines mig_lines

  gpu_info="$($NVIDIA_SMI_BIN --query-gpu=name,memory.total --format=csv,noheader,nounits)" ||
    die "nvidia-smi name/memory query failed"
  printf 'POLARIS_GUARD nvidia-smi=%s\n' "$gpu_info"
  mapfile -t gpu_lines <<< "$gpu_info"
  [[ "${#gpu_lines[@]}" == 4 ]] || die "expected 4 GPUs, got ${#gpu_lines[@]}"
  for line in "${gpu_lines[@]}"; do
    [[ "$line" == "NVIDIA A100-SXM4-40GB, 40960" ]] ||
      die "unexpected GPU identity: $line"
  done

  gpu_inventory="$($NVIDIA_SMI_BIN -L)" || die "nvidia-smi inventory query failed"
  printf 'POLARIS_GUARD nvidia-smi-L=%s\n' "$gpu_inventory"
  mapfile -t inventory_lines <<< "$gpu_inventory"
  [[ "${#inventory_lines[@]}" == 4 ]] ||
    die "expected 4 GPU inventory records, got ${#inventory_lines[@]}"

  mig_info="$($NVIDIA_SMI_BIN --query-gpu=mig.mode.current --format=csv,noheader)" ||
    die "nvidia-smi MIG query failed"
  printf 'POLARIS_GUARD mig=%s\n' "$mig_info"
  mapfile -t mig_lines <<< "$mig_info"
  [[ "${#mig_lines[@]}" == 4 ]] || die "expected 4 MIG records, got ${#mig_lines[@]}"
  for line in "${mig_lines[@]}"; do
    line="${line//[[:space:]]/}"
    [[ "$line" == "Disabled" ]] || die "MIG mode is not Disabled: $line"
  done

  for i in 0 1 2 3; do
    count="$(CUDA_VISIBLE_DEVICES="$i" "$GUARD_PYTHON_BIN" -c \
      'import torch; print(torch.cuda.device_count())')" ||
      die "CUDA device-count probe failed for GPU $i"
    count="${count//$'\n'/}"
    printf 'POLARIS_GUARD binding=%s device_count=%s\n' "$i" "$count"
    [[ "$count" == 1 ]] || die "expected one visible GPU for binding $i, got $count"
  done
}

prepare_job() {
  if [[ "$DRY_EXEC" == 1 ]]; then
    BOOT_ROOT="${BOOT_ROOT:-${TMPDIR:-/tmp}/tpen-polaris-dry-${$}}"
    mkdir -p -- "$BOOT_ROOT"
    trap 'rm -rf -- "$BOOT_ROOT"' EXIT
    CHECKOUT="${CHECKOUT_ROOT:-$PWD}"
    PYTHON_BIN="${PYTHON_BIN:-python}"
    printf 'DRY_EXEC bootstrap=SKIPPED checkout=%s\n' "$CHECKOUT"
    return 0
  fi

  require_value SOURCE_SHA
  require_value SOURCE_REPO
  BOOT_ROOT="${BOOT_ROOT:-/local/scratch/${USER:?}/${PBS_JOBID}.stage1-${MODE}}"
  CHECKOUT="$BOOT_ROOT/checkout"
  JOB_ENV="$BOOT_ROOT/env"
  export TMPDIR="$BOOT_ROOT/tmp"
  export UV_CACHE_DIR="$BOOT_ROOT/uv-cache"
  mkdir -p -- "$BOOT_ROOT" "$TMPDIR" "$UV_CACHE_DIR"
  trap 'rm -rf -- "$BOOT_ROOT"' EXIT

  git clone --filter=blob:none --no-tags "$SOURCE_REPO" "$CHECKOUT"
  git -C "$CHECKOUT" checkout --detach "$SOURCE_SHA"
  [[ "$(git -C "$CHECKOUT" rev-parse HEAD)" == "$SOURCE_SHA" ]] ||
    die "detached checkout does not match SOURCE_SHA"
  [[ -z "$(git -C "$CHECKOUT" status --porcelain)" ]] || die "checkout is dirty"
  "$PYBIN" -m uv lock --check --project "$CHECKOUT"
  UV_PROJECT_ENVIRONMENT="$JOB_ENV" "$PYBIN" -m uv sync --offline \
    --project "$CHECKOUT" --extra cu128 --locked
  PYTHON_BIN="$JOB_ENV/bin/python"
  [[ -x "$PYTHON_BIN" ]] || die "job-local cu128 interpreter missing"
  printf 'POLARIS_ENV python=%s\n' "$PYTHON_BIN"
  printf 'POLARIS_ENV uv='; "$PYBIN" -m uv --version
}

extract_rows() {
  local manifest="$1" kind="$2" output_dir="$3"
  require_value SOURCE_SHA
  [[ -f "$manifest" ]] || die "manifest missing: $manifest"
  mkdir -p -- "$output_dir"
  "$PYTHON_BIN" - "$manifest" "$output_dir" "$kind" "$SOURCE_SHA" "$RESULTS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

manifest, output_dir, kind, source_sha, results_root = map(Path, sys.argv[1:])
kind = str(kind)
source_sha = str(source_sha)
results_root = str(results_root)
document = json.loads(manifest.read_text(encoding="utf-8"))
rows = [row for row in document.get("rows", []) if row.get("kind") == kind]
if kind == "train":
    rows.sort(key=lambda row: int(row["lineage"]))
    if [int(row["lineage"]) for row in rows] != [0, 1, 2]:
        raise SystemExit("expected train lineages 0,1,2")
else:
    rows.sort(key=lambda row: (int(row["lineage"]), int(row["stream"])))
    if len(rows) != 12:
        raise SystemExit(f"expected 12 eval rows, got {len(rows)}")
for row in rows:
    if row.get("source_sha") != source_sha:
        raise SystemExit("row source_sha does not match wrapper SOURCE_SHA")
    row_root = str(row.get("results_root", ""))
    if not row_root.startswith(results_root.rstrip("/") + "/") and row_root != results_root.rstrip("/"):
        raise SystemExit("row results_root is outside wrapper RESULTS_ROOT")
    row_id = str(row["row_id"])
    (output_dir / f"{row_id}.json").write_text(
        json.dumps(row, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    print(output_dir / f"{row_id}.json")
PY
}

run_command_for_row() {
  local row="$1" driver="$2" gpu="$3" cpu_range="$4" log="$5" status="$6"
  local rc=0 row_id
  row_id="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["row_id"])' "$row")"
  if [[ "$DRY_EXEC" == 1 ]]; then
    printf 'DRY_EXEC compute=SKIPPED row=%s gpu=%s cpu=%s\n' "$row_id" "$gpu" "$cpu_range" | tee "$log"
  else
    export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
    taskset --cpu-list "$cpu_range" "$PYTHON_BIN" - "$row" "$driver" "$CHECKOUT" \
      "$PYTHON_BIN" "$SOURCE_SHA" "$RESULTS_ROOT" > "$log" 2>&1 <<'PY' || rc=$?
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from experiments.atomistic.stage1.launch import command_for_row

row_path, driver, checkout, python, head_sha, results_root = sys.argv[1:]
row = json.loads(Path(row_path).read_text(encoding="utf-8"))
argv = shlex.split(command_for_row(row, python=python, driver=driver))
print("POLARIS_COMMAND " + shlex.join(argv), flush=True)
completed = subprocess.run(argv, cwd=checkout, env=os.environ)
raise SystemExit(completed.returncode)
PY
  fi
  printf 'row_id=%s gpu=%s cpu=%s exit_status=%s\n' "$row_id" "$gpu" "$cpu_range" "$rc" > "$status"
  return "$rc"
}

run_mode() {
  local row_dir="$BOOT_ROOT/rows/$MODE"
  local row_list row gpu cpu_range log status
  local rc=0
  row_list="$BOOT_ROOT/$MODE-row-list.txt"
  extract_rows "$PLAN_MANIFEST" "$MODE" "$row_dir" > "$row_list"
  if [[ "$MODE" == train ]]; then
    local -a pids=() rows=()
    mapfile -t rows < "$row_list"
    local index=0
    for row in "${rows[@]}"; do
      case "$index" in
        0) gpu=0; cpu_range=0-15 ;;
        1) gpu=1; cpu_range=16-31 ;;
        2) gpu=2; cpu_range=32-47 ;;
        *) die "unexpected train row index=$index" ;;
      esac
      log="$RESULTS_ROOT/logs/row-$(basename "$row" .json)-${PBS_JOBID}.log"
      status="$RESULTS_ROOT/logs/row-$(basename "$row" .json)-${PBS_JOBID}.status"
      if [[ "$DRY_EXEC" == 1 ]]; then
        log="$BOOT_ROOT/$(basename "$row" .json).log"
        status="$BOOT_ROOT/$(basename "$row" .json).status"
      fi
      run_command_for_row "$row" "$CHECKOUT/experiments/atomistic/stage1/train.py" \
        "$gpu" "$cpu_range" "$log" "$status" &
      pids+=("$!")
      index=$((index + 1))
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || rc=1
    done
  else
    while IFS= read -r row; do
      gpu=0
      cpu_range=0-15
      log="$RESULTS_ROOT/logs/row-$(basename "$row" .json)-${PBS_JOBID}.log"
      status="$RESULTS_ROOT/logs/row-$(basename "$row" .json)-${PBS_JOBID}.status"
      if [[ "$DRY_EXEC" == 1 ]]; then
        log="$BOOT_ROOT/$(basename "$row" .json).log"
        status="$BOOT_ROOT/$(basename "$row" .json).status"
      fi
      run_command_for_row "$row" "$CHECKOUT/experiments/atomistic/stage1/eval.py" \
        "$gpu" "$cpu_range" "$log" "$status" || rc=1
    done < "$row_list"
  fi
  return "$rc"
}

run_plan() {
  local -a command=("$PYTHON_BIN" -m experiments.atomistic.stage1.plan)
  local -a checkpoints lineage_seeds
  require_value PLAN_OUTPUT
  require_value PLAN_PHASE
  require_value CHECKPOINTS
  require_value LINEAGE_SEEDS
  [[ "$PLAN_PHASE" == 1 || "$PLAN_PHASE" == 2 ]] || die "PLAN_PHASE must be 1 or 2"
  CHECKPOINTS="${CHECKPOINTS//:/ }"
  LINEAGE_SEEDS="${LINEAGE_SEEDS//:/ }"
  case "$PLAN_OUTPUT" in
    "$RESULTS_ROOT"/*) ;;
    *) die "plan output is outside RESULTS_ROOT: $PLAN_OUTPUT" ;;
  esac
  read -r -a checkpoints <<< "$CHECKPOINTS"
  read -r -a lineage_seeds <<< "$LINEAGE_SEEDS"
  command+=(--phase "$PLAN_PHASE" --output "$PLAN_OUTPUT"
            --source-sha "$SOURCE_SHA" --results-root "$RESULTS_ROOT"
            --checkpoints "${checkpoints[@]}"
            --lineage-seeds "${lineage_seeds[@]}")
  if [[ "$PLAN_PHASE" == 1 ]]; then
    require_value SYSTEM
    command+=(--systems "$SYSTEM" --device cuda)
  else
    require_value PHASE1_MANIFEST
    require_value CHECKPOINT_BINDINGS
    command+=(--phase1-manifest "$PHASE1_MANIFEST"
              --checkpoint-bindings "$CHECKPOINT_BINDINGS")
  fi
  printf 'POLARIS_COMMAND %s\n' "${command[*]}"
  "${command[@]}"
}

run_collect() {
  require_value COLLECT_MANIFEST
  require_value COLLECT_OUTPUT
  case "$COLLECT_OUTPUT" in
    "$RESULTS_ROOT"/*) ;;
    *) die "collection output is outside RESULTS_ROOT: $COLLECT_OUTPUT" ;;
  esac
  "$PYTHON_BIN" -c \
    'import sys; from experiments.atomistic.stage1.collect import collect; collect(sys.argv[1], output=sys.argv[2])' \
    "$COLLECT_MANIFEST" "$COLLECT_OUTPUT"
}

main() {
  case "$MODE" in
    guard)
      EXPECTED_QUEUE="${EXPECTED_QUEUE:-debug}"
      require_value RESULTS_ROOT
      assert_results_root
      assert_pbs_context
      guard_gpus
      ;;
    dry-exec)
      DRY_EXEC=1
      MODE=train
      EXPECTED_QUEUE="${EXPECTED_QUEUE:-debug}"
      require_value RESULTS_ROOT
      require_value PLAN_MANIFEST
      assert_results_root
      assert_pbs_context
      prepare_job
      guard_gpus
      cd "$CHECKOUT"
      run_mode
      printf 'DRY_EXEC complete=PASS\n'
      ;;
    plan)
      EXPECTED_QUEUE="${EXPECTED_QUEUE:-debug}"
      require_value RESULTS_ROOT
      require_value SOURCE_SHA
      assert_results_root
      assert_pbs_context
      write_job_receipt
      prepare_job
      guard_gpus
      cd "$CHECKOUT"
      run_plan
      ;;
    collect)
      EXPECTED_QUEUE="${EXPECTED_QUEUE:-debug}"
      require_value RESULTS_ROOT
      require_value SOURCE_SHA
      assert_results_root
      assert_pbs_context
      write_job_receipt
      prepare_job
      cd "$CHECKOUT"
      run_collect
      ;;
    train|eval)
      [[ "$MODE" == train ]] && EXPECTED_QUEUE="${EXPECTED_QUEUE:-capacity}"
      [[ "$MODE" == eval ]] && EXPECTED_QUEUE="${EXPECTED_QUEUE:-debug}"
      require_value RESULTS_ROOT
      require_value PLAN_MANIFEST
      require_value SOURCE_SHA
      assert_results_root
      assert_pbs_context
      write_job_receipt
      prepare_job
      guard_gpus
      cd "$CHECKOUT"
      run_mode
      ;;
    *) die "usage: MODE=train|eval|guard|dry-exec" ;;
  esac
}

main "$@"
