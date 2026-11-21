#!/usr/bin/env bash
# Polaris PBS wrapper for the direct Stage-1 He/H2 route.
#
# The wrapper owns PBS admission, Eagle storage, four-GPU/one-visible-GPU
# guards, and row-level CPU/GPU placement.  Stage-1 identity and argv remain
# owned by experiments.atomistic.stage1.launch.command_for_row.
#
# The submission driver supplies MODE=train|eval, SOURCE_SHA, CHECKOUT_ROOT,
# UV_CACHE_DIR, RESULTS_ROOT, PLAN_MANIFEST, and EXPECTED_QUEUE.  The checked-in header is a
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
TPEN_POLARIS_TEST_RESULTS_PREFIX="${TPEN_POLARIS_TEST_RESULTS_PREFIX:-}"
MODE="${MODE:-${1:-}}"
SOURCE_SHA="${SOURCE_SHA:-}"
CHECKOUT_ROOT="${CHECKOUT_ROOT:-}"
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
TASKSET_BIN="${TASKSET_BIN:-taskset}"
POLARIS_LSCPU_BIN="${POLARIS_LSCPU_BIN:-lscpu}"
BOOT_ROOT="${BOOT_ROOT:-}"
BOOT_PARENT="${BOOT_PARENT:-/local/scratch/${USER:?}}"
OWNED_BOOT_ROOT=""
declare -a LINEAGE_CPU_SETS=()
declare -a LINEAGE_CORE_COUNTS=()

cleanup_owned_boot_root() {
  if [[ -n "$OWNED_BOOT_ROOT" && -d "$OWNED_BOOT_ROOT" ]]; then
    rm -rf -- "$OWNED_BOOT_ROOT"
  fi
}

die() {
  printf '%s\n' "POLARIS_GUARD FAIL $*" >&2
  exit 2
}

configure_results_prefix() {
  [[ -n "$TPEN_POLARIS_TEST_RESULTS_PREFIX" ]] || return 0
  if [[ "${PBS_ENVIRONMENT:-}" == PBS_BATCH || "${PBS_ENVIRONMENT:-}" == PBS_INTERACTIVE || -n "${PBS_JOBID:-}" ]]; then
    die "TPEN_POLARIS_TEST_RESULTS_PREFIX is forbidden in PBS contexts or when PBS_JOBID is set"
  fi
  RESULTS_PREFIX="$TPEN_POLARIS_TEST_RESULTS_PREFIX"
  printf 'POLARIS_TEST_PREFIX_OVERRIDE=%s\n' "$RESULTS_PREFIX" >&2
}

require_value() {
  local name="$1" value="${!1:-}"
  [[ -n "$value" ]] || die "$name is required"
}

assert_results_root() {
  require_value RESULTS_ROOT
  if [[ "$RESULTS_ROOT" == .. || "$RESULTS_ROOT" == ../* || "$RESULTS_ROOT" == */../* || "$RESULTS_ROOT" == */.. ]]; then
    die "RESULTS_ROOT is outside $RESULTS_PREFIX: lexical '..' component"
  fi
  assert_contained_path "$RESULTS_ROOT" "$RESULTS_PREFIX" RESULTS_ROOT
}

assert_contained_path() {
  local candidate="$1" root="$2" label="$3"
  local canonical
  if ! canonical="$("$PYBIN" - "$candidate" "$root" "$label" 2>&1 <<'PY'
import sys
from pathlib import Path

candidate, root, label = (Path(value) for value in sys.argv[1:])
if not candidate.is_absolute():
    raise SystemExit(f"{label} must be absolute: {candidate}")
if not root.is_absolute():
    raise SystemExit(f"containment root must be absolute: {root}")
canonical = candidate.resolve()
try:
    canonical.relative_to(root.resolve())
except ValueError as exc:
    raise SystemExit(f"{label} is outside {root}: {candidate}") from exc
print(canonical)
PY
  )"; then
    die "$canonical"
  fi
  printf -v "$label" '%s' "$canonical"
}

validate_mode_paths() {
  case "$MODE" in
    guard)
      ;;
    dry-exec|train|eval)
      require_value PLAN_MANIFEST
      ;;
    plan)
      require_value PLAN_OUTPUT
      assert_contained_path "$PLAN_OUTPUT" "$RESULTS_ROOT" PLAN_OUTPUT
      if [[ -n "${PHASE1_MANIFEST:-}" ]]; then
        assert_contained_path "$PHASE1_MANIFEST" "$RESULTS_ROOT" PHASE1_MANIFEST
      fi
      if [[ -n "${CHECKPOINT_BINDINGS:-}" ]]; then
        assert_contained_path "$CHECKPOINT_BINDINGS" "$RESULTS_ROOT" CHECKPOINT_BINDINGS
      fi
      ;;
    collect)
      require_value COLLECT_MANIFEST
      require_value COLLECT_OUTPUT
      assert_contained_path "$COLLECT_MANIFEST" "$RESULTS_ROOT" COLLECT_MANIFEST
      assert_contained_path "$COLLECT_OUTPUT" "$RESULTS_ROOT" COLLECT_OUTPUT
      ;;
    *)
      die "unsupported mode for path validation: $MODE"
      ;;
  esac
}

validate_selected_rows() {
  local manifest="$1" kind="$2"
  "$PYBIN" - "$manifest" "$kind" "$SOURCE_SHA" "$RESULTS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

manifest, kind, source_sha, results_root = sys.argv[1:]
manifest_path = Path(manifest)
root = Path(results_root)
document = json.loads(manifest_path.read_text(encoding="utf-8"))
rows = [row for row in document.get("rows", []) if row.get("kind") == kind]
if kind == "train" and [int(row["lineage"]) for row in sorted(rows, key=lambda row: int(row["lineage"]))] != [0, 1, 2]:
    raise SystemExit("expected train lineages 0,1,2")
if kind == "eval" and len(rows) != 12:
    raise SystemExit(f"expected 12 eval rows, got {len(rows)}")
for row in rows:
    row_root = Path(str(row.get("results_root", "")))
    if row.get("source_sha") != source_sha:
        raise SystemExit("row source_sha does not match wrapper SOURCE_SHA")
    # Replace launch.render's GPU-row admission before any guard or driver.
    if row.get("device") != "cuda":
        raise SystemExit(f"row {row.get('row_id', '<unknown>')} must have device=cuda")
    if any(part == ".." for part in row_root.parts):
        raise SystemExit(f"row results_root contains '..': {row_root}")
    try:
        row_root.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"row results_root is outside wrapper RESULTS_ROOT: {row_root}") from exc
PY
}

validate_selected_rows_or_die() {
  local error
  if ! error="$(validate_selected_rows "$1" "$2" 2>&1)"; then
    die "$error"
  fi
}

assert_pbs_context() {
  if [[ -n "$TPEN_POLARIS_TEST_RESULTS_PREFIX" && -z "$PBS_JOBID" ]]; then
    PBS_JOBID="polaris-test-local"
  fi
  require_value PBS_JOBID
  require_value PBS_QUEUE
  require_value EXPECTED_QUEUE
  [[ "$PBS_QUEUE" == "$EXPECTED_QUEUE" ]] ||
    die "delivered queue=$PBS_QUEUE expected=$EXPECTED_QUEUE"
  printf 'POLARIS_GUARD PBS_JOBID=%s PBS_QUEUE=%s EXPECTED_QUEUE=%s\n' \
    "$PBS_JOBID" "$PBS_QUEUE" "$EXPECTED_QUEUE"
}

write_job_receipt() {
  local receipt_path
  assert_contained_path "$RESULTS_ROOT/logs" "$RESULTS_ROOT" RESULTS_LOGS
  assert_contained_path "$RESULTS_ROOT/launch" "$RESULTS_ROOT" RESULTS_LAUNCH
  assert_contained_path "$RESULTS_ROOT/plan" "$RESULTS_ROOT" RESULTS_PLAN
  receipt_path="$RESULTS_LOGS/polaris-${MODE}-${PBS_JOBID}.env"
  assert_contained_path "$receipt_path" "$RESULTS_ROOT" receipt_path
  [[ "$DRY_EXEC" == 1 ]] && return 0
  mkdir -p -- "$RESULTS_LOGS" "$RESULTS_LAUNCH" "$RESULTS_PLAN"
  {
    printf 'PBS_JOBID=%s\n' "$PBS_JOBID"
    printf 'PBS_QUEUE=%s\n' "$PBS_QUEUE"
    printf 'HOST=%s\n' "$(hostname -f 2>/dev/null || hostname)"
    printf 'UTC=%s\n' "$(date -u +%FT%TZ)"
    printf 'SOURCE_SHA=%s\n' "$SOURCE_SHA"
    printf 'MODE=%s\n' "$MODE"
    printf 'RESULTS_ROOT=%s\n' "$RESULTS_ROOT"
    printf 'CHECKOUT_ROOT=%s\n' "$CHECKOUT_ROOT"
    printf 'UV_CACHE_DIR=%s\n' "$UV_CACHE_DIR"
    printf 'BASE_PYTHON=%s\n' "$PYBIN"
    printf 'BASE_PYTHON_VERSION='; "$PYBIN" --version 2>&1
    printf 'UV_PYTHON=%s\n' "$UV_PYTHON"
    printf 'UV_NO_MANAGED_PYTHON=%s\n' "$UV_NO_MANAGED_PYTHON"
    printf 'UV_PYTHON_DOWNLOADS=%s\n' "$UV_PYTHON_DOWNLOADS"
    printf 'UV_LINK_MODE=%s\n' "$UV_LINK_MODE"
    if [[ -n "$TPEN_POLARIS_TEST_RESULTS_PREFIX" ]]; then
      printf 'POLARIS_TEST_PREFIX_OVERRIDE=%s\n' "$RESULTS_PREFIX"
    fi
  } > "$receipt_path"
}

guard_gpus() {
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

derive_cpu_layout() {
  local report_only="${1:-0}" affinity_output layout_output line lineage core_count cpu_set
  local -a layout_lines

  affinity_output="$("$TASKSET_BIN" -pc $$ 2>&1)" ||
    die "could not determine process CPU affinity: $affinity_output"
  if ! layout_output="$($PYBIN - "$POLARIS_LSCPU_BIN" "$affinity_output" "$report_only" <<'PY'
import subprocess
import sys


lscpu_bin, affinity_text, report_only_text = sys.argv[1:]
report_only = report_only_text == "1"


def parse_cpu_list(text: str) -> set[int]:
    spec = text.rsplit(":", 1)[-1].strip()
    if not spec:
        raise ValueError(f"missing CPU list in taskset output: {text!r}")
    cpus: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            raise ValueError(f"empty CPU range in taskset output: {text!r}")
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"reversed CPU range in taskset output: {text!r}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(chunk))
    return cpus


try:
    allowed = parse_cpu_list(affinity_text)
    completed = subprocess.run(
        [lscpu_bin, "-p=CPU,CORE,NODE"],
        check=True,
        capture_output=True,
        text=True,
    )
    records = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            raise ValueError(f"malformed lscpu record: {raw_line!r}")
        cpu, core, node = (int(fields[index]) for index in range(3))
        if cpu in allowed:
            records.append((node, core, cpu))
except (OSError, ValueError, subprocess.CalledProcessError) as exc:
    raise SystemExit(f"could not derive CPU layout: {exc}") from exc

# Keep one logical CPU per physical core. Sorting by NUMA node then core keeps
# each group as NUMA-contiguous as the allocation permits; GPU-NUMA locality is
# intentionally not claimed because it is not derived from this topology.
selected = []
seen_cores = set()
for node, core, cpu in sorted(records):
    if core in seen_cores:
        continue
    seen_cores.add(core)
    selected.append((node, core, cpu))
ordered = sorted(selected, key=lambda record: (record[0], record[1], record[2]))
usable = len(ordered) - len(ordered) % 3
per_lineage = usable // 3
if per_lineage < 3 and report_only:
    cpus = ",".join(str(cpu) for _, _, cpu in ordered)
    print(f"report\t{len(ordered)}\t{cpus}")
    raise SystemExit(0)
if per_lineage < 3:
    raise SystemExit(
        f"fewer than 3 physical cores per lineage: allowed={len(ordered)}"
    )
groups = [ordered[index * per_lineage:(index + 1) * per_lineage] for index in range(3)]
core_sets = [{core for _, core, _ in group} for group in groups]
for left in range(3):
    for right in range(left + 1, 3):
        # Explicitly assert the invariant even though slicing currently makes
        # overlap impossible; this protects future layout changes.
        assert not (core_sets[left] & core_sets[right]), (
            f"lineage core sets overlap: {left} and {right}"
        )
for lineage, group in enumerate(groups):
    cpus = ",".join(str(cpu) for _, _, cpu in group)
    print(f"{lineage}\t{per_lineage}\t{cpus}")
PY
  )"; then
    die "$layout_output"
  fi
  mapfile -t layout_lines <<< "$layout_output"
  if [[ "$report_only" == 1 && "${#layout_lines[@]}" == 1 ]]; then
    IFS=$'\t' read -r lineage core_count cpu_set <<< "${layout_lines[0]}"
    [[ "$lineage" == report && "$core_count" =~ ^[0-9]+$ ]] ||
      die "malformed report-only CPU layout: ${layout_lines[0]}"
    printf 'POLARIS_LAYOUT report-only=1 cores=%s cpus=%s\n' "$core_count" "$cpu_set"
    return 0
  fi
  [[ "${#layout_lines[@]}" == 3 ]] ||
    die "expected three lineage CPU layouts, got ${#layout_lines[@]}"
  LINEAGE_CPU_SETS=()
  LINEAGE_CORE_COUNTS=()
  for line in "${layout_lines[@]}"; do
    IFS=$'\t' read -r lineage core_count cpu_set <<< "$line"
    [[ "$lineage" =~ ^[0-2]$ && "$core_count" =~ ^[0-9]+$ && -n "$cpu_set" ]] ||
      die "malformed derived CPU layout: $line"
    LINEAGE_CPU_SETS[$lineage]="$cpu_set"
    LINEAGE_CORE_COUNTS[$lineage]="$core_count"
    printf 'POLARIS_LAYOUT lineage=%s gpu=%s cores=%s cpus=%s\n' \
      "$lineage" "$lineage" "$core_count" "$cpu_set"
  done
}

guard_runtime() {
  case "$MODE" in
    train|eval)
      guard_gpus
      derive_cpu_layout
      ;;
    plan)
      derive_cpu_layout 1
      ;;
    collect)
      ;;
    *)
      guard_gpus
      derive_cpu_layout
      ;;
  esac
}

prepare_job() {
  require_value SOURCE_SHA
  require_value CHECKOUT_ROOT
  require_value UV_CACHE_DIR
  if [[ -e "$BOOT_ROOT" || -L "$BOOT_ROOT" ]]; then
    die "BOOT_ROOT already exists and is not owned: $BOOT_ROOT"
  fi
  mkdir -p -- "$BOOT_PARENT"
  BOOT_ROOT="$(mktemp -d -- "$BOOT_PARENT/tpen-polaris-${MODE}.XXXXXX")"
  OWNED_BOOT_ROOT="$BOOT_ROOT"
  trap cleanup_owned_boot_root EXIT
  CHECKOUT="$BOOT_ROOT/checkout"
  JOB_ENV="$BOOT_ROOT/env"
  export TMPDIR="$BOOT_ROOT/tmp"
  export UV_CACHE_DIR
  [[ -d "$UV_CACHE_DIR" ]] || die "UV_CACHE_DIR is missing: $UV_CACHE_DIR"
  if "$PYBIN" - "$UV_CACHE_DIR" "$BOOT_ROOT" <<'PY'
import sys
from pathlib import Path

cache, boot = (Path(value).resolve() for value in sys.argv[1:])
try:
    cache.relative_to(boot)
except ValueError:
    raise SystemExit(1)
raise SystemExit(0)
PY
  then
    die "UV_CACHE_DIR must be external to BOOT_ROOT: $UV_CACHE_DIR"
  fi
  export UV_PYTHON="$PYBIN" UV_NO_MANAGED_PYTHON=1 UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
  mkdir -p -- "$TMPDIR"

  if [[ "$DRY_EXEC" == 1 ]]; then
    CHECKOUT="$CHECKOUT_ROOT"
    PYTHON_BIN="${PYTHON_BIN:-$PYBIN}"
    GUARD_PYTHON_BIN="${GUARD_PYTHON_BIN:-$PYBIN}"
    printf 'DRY_EXEC bootstrap=SKIPPED checkout=%s\n' "$CHECKOUT"
    return 0
  fi

  [[ -d "$CHECKOUT_ROOT/.git" || -f "$CHECKOUT_ROOT/.git" ]] || die "CHECKOUT_ROOT is not a git checkout"
  CHECKOUT="$CHECKOUT_ROOT"
  [[ "$(git -C "$CHECKOUT" rev-parse HEAD)" == "$SOURCE_SHA" ]] ||
    die "detached checkout does not match SOURCE_SHA"
  [[ -z "$(git -C "$CHECKOUT" status --porcelain)" ]] || die "checkout is dirty"
  [[ "$(git -C "$CHECKOUT" rev-parse --is-shallow-repository)" == false ]] ||
    die "CHECKOUT_ROOT must be a full clone"
  [[ -z "$(git -C "$CHECKOUT" config --local --get-regexp '^remote\..*\.promisor$' || true)" ]] ||
    die "CHECKOUT_ROOT has a promisor remote"
  [[ -z "$(git -C "$CHECKOUT" config --local --get extensions.partialclone || true)" ]] ||
    die "CHECKOUT_ROOT is a partial clone"
  "$PYBIN" -m uv lock --check --project "$CHECKOUT"
  UV_PROJECT_ENVIRONMENT="$JOB_ENV" "$PYBIN" -m uv sync --offline \
    --project "$CHECKOUT" --extra cu128 --locked
  PYTHON_BIN="$JOB_ENV/bin/python"
  [[ -x "$PYTHON_BIN" ]] || die "job-local cu128 interpreter missing"
  GUARD_PYTHON_BIN="${GUARD_PYTHON_BIN:-$PYTHON_BIN}"
  printf 'POLARIS_ENV python=%s\n' "$PYTHON_BIN"
  printf 'POLARIS_ENV base_python=%s version=' "$PYBIN"; "$PYBIN" --version 2>&1
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
root = Path(results_root)
for row in rows:
    if row.get("source_sha") != source_sha:
        raise SystemExit("row source_sha does not match wrapper SOURCE_SHA")
    row_root = str(row.get("results_root", ""))
    row_root_path = Path(row_root)
    if any(part == ".." for part in row_root_path.parts):
        raise SystemExit("row results_root contains '..'")
    try:
        row_root_path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit("row results_root is outside wrapper RESULTS_ROOT") from exc
    if row.get("device") != "cuda":
        raise SystemExit(f"row {row.get('row_id', '<unknown>')} must have device=cuda")
    row_id = str(row["row_id"])
    (output_dir / f"{row_id}.json").write_text(
        json.dumps(row, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    print(output_dir / f"{row_id}.json")
PY
}

run_command_for_row() {
  local row="$1" driver="$2" gpu="$3" cpu_set="$4" core_count="$5" log="$6" status="$7"
  local rc=0 row_id
  row_id="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["row_id"])' "$row")"
  if [[ "$DRY_EXEC" == 1 ]]; then
    printf 'DRY_EXEC compute=SKIPPED row=%s gpu=%s cpu=%s\n' "$row_id" "$gpu" "$cpu_set" | tee "$log"
  else
    export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS="$core_count" MKL_NUM_THREADS="$core_count"
    "$TASKSET_BIN" --cpu-list "$cpu_set" "$PYTHON_BIN" - "$row" "$driver" "$CHECKOUT" \
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
  printf 'row_id=%s gpu=%s cpu=%s exit_status=%s\n' "$row_id" "$gpu" "$cpu_set" "$rc" > "$status"
  return "$rc"
}

run_mode() {
  local row_dir="$BOOT_ROOT/rows/$MODE"
  local row_list row gpu cpu_set core_count log status lineage destination_root
  local rc=0
  row_list="$BOOT_ROOT/$MODE-row-list.txt"
  extract_rows "$PLAN_MANIFEST" "$MODE" "$row_dir" > "$row_list"
  if [[ "$MODE" == train ]]; then
    local -a pids=() rows=()
    mapfile -t rows < "$row_list"
    local index=0
    for row in "${rows[@]}"; do
      case "$index" in
        0|1|2)
          lineage="$index"
          gpu="$lineage"
          cpu_set="${LINEAGE_CPU_SETS[$lineage]}"
          core_count="${LINEAGE_CORE_COUNTS[$lineage]}"
          ;;
        *) die "unexpected train row index=$index" ;;
      esac
      log="$RESULTS_ROOT/logs/row-$(basename "$row" .json)-${PBS_JOBID}.log"
      status="$RESULTS_ROOT/logs/row-$(basename "$row" .json)-${PBS_JOBID}.status"
      if [[ "$DRY_EXEC" == 1 ]]; then
        log="$BOOT_ROOT/$(basename "$row" .json).log"
        status="$BOOT_ROOT/$(basename "$row" .json).status"
        destination_root="$BOOT_ROOT"
      else
        destination_root="$RESULTS_ROOT"
      fi
      assert_contained_path "$log" "$destination_root" log
      assert_contained_path "$status" "$destination_root" status
      run_command_for_row "$row" "$CHECKOUT/experiments/atomistic/stage1/train.py" \
        "$gpu" "$cpu_set" "$core_count" "$log" "$status" &
      pids+=("$!")
      index=$((index + 1))
    done
    for pid in "${pids[@]}"; do
      wait "$pid" || rc=1
    done
  else
    while IFS= read -r row; do
      gpu=0
      lineage=0
      cpu_set="${LINEAGE_CPU_SETS[$lineage]}"
      core_count="${LINEAGE_CORE_COUNTS[$lineage]}"
      log="$RESULTS_ROOT/logs/row-$(basename "$row" .json)-${PBS_JOBID}.log"
      status="$RESULTS_ROOT/logs/row-$(basename "$row" .json)-${PBS_JOBID}.status"
      if [[ "$DRY_EXEC" == 1 ]]; then
        log="$BOOT_ROOT/$(basename "$row" .json).log"
        status="$BOOT_ROOT/$(basename "$row" .json).status"
        destination_root="$BOOT_ROOT"
      else
        destination_root="$RESULTS_ROOT"
      fi
      assert_contained_path "$log" "$destination_root" log
      assert_contained_path "$status" "$destination_root" status
      run_command_for_row "$row" "$CHECKOUT/experiments/atomistic/stage1/eval.py" \
        "$gpu" "$cpu_set" "$core_count" "$log" "$status" || rc=1
    done < "$row_list"
  fi
  return "$rc"
}

run_plan() {
  local -a command=("$PYTHON_BIN" -m experiments.atomistic.stage1.plan)
  local -a checkpoints lineage_seeds
  require_value PLAN_OUTPUT
  require_value PLAN_PHASE
  if [[ "$PLAN_PHASE" == 2 ]]; then
    require_value CHECKPOINTS
  fi
  require_value LINEAGE_SEEDS
  [[ "$PLAN_PHASE" == 1 || "$PLAN_PHASE" == 2 ]] || die "PLAN_PHASE must be 1 or 2"
  LINEAGE_SEEDS="${LINEAGE_SEEDS//:/ }"
  assert_contained_path "$PLAN_OUTPUT" "$RESULTS_ROOT" PLAN_OUTPUT
  read -r -a lineage_seeds <<< "$LINEAGE_SEEDS"
  command+=(--phase "$PLAN_PHASE" --output "$PLAN_OUTPUT"
            --source-sha "$SOURCE_SHA" --results-root "$RESULTS_ROOT"
            --lineage-seeds "${lineage_seeds[@]}")
  if [[ -n "${CHECKPOINTS:-}" ]]; then
    CHECKPOINTS="${CHECKPOINTS//:/ }"
    read -r -a checkpoints <<< "$CHECKPOINTS"
    command+=(--checkpoints "${checkpoints[@]}")
  fi
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
  assert_contained_path "$COLLECT_OUTPUT" "$RESULTS_ROOT" COLLECT_OUTPUT
  "$PYTHON_BIN" -c \
    'import sys; from experiments.atomistic.stage1.collect import collect; collect(sys.argv[1], output=sys.argv[2])' \
    "$COLLECT_MANIFEST" "$COLLECT_OUTPUT"
}

main() {
  configure_results_prefix
  case "$MODE" in
    guard)
      EXPECTED_QUEUE="${EXPECTED_QUEUE:-debug}"
      require_value RESULTS_ROOT
      assert_results_root
      assert_pbs_context
      guard_runtime
      ;;
    dry-exec)
      DRY_EXEC=1
      MODE=train
      EXPECTED_QUEUE="${EXPECTED_QUEUE:-debug}"
      require_value RESULTS_ROOT
      require_value PLAN_MANIFEST
      assert_results_root
      assert_pbs_context
      validate_mode_paths
      prepare_job
      validate_selected_rows_or_die "$PLAN_MANIFEST" train
      guard_runtime
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
      validate_mode_paths
      prepare_job
      write_job_receipt
      guard_runtime
      cd "$CHECKOUT"
      run_plan
      ;;
    collect)
      EXPECTED_QUEUE="${EXPECTED_QUEUE:-debug}"
      require_value RESULTS_ROOT
      require_value SOURCE_SHA
      assert_results_root
      assert_pbs_context
      validate_mode_paths
      prepare_job
      write_job_receipt
      guard_runtime
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
      validate_mode_paths
      prepare_job
      validate_selected_rows_or_die "$PLAN_MANIFEST" "$MODE"
      write_job_receipt
      guard_runtime
      cd "$CHECKOUT"
      run_mode
      ;;
    *) die "usage: MODE=train|eval|guard|dry-exec" ;;
  esac
}

main "$@"
