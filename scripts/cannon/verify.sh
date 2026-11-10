#!/usr/bin/env bash
# Versioned, parameterized Cannon verification runner.
# Facility-specific paths and the sbatch wrapper belong in the Task Orchestrator note.
set -Eeuo pipefail
if ! shopt -s inherit_errexit 2>/dev/null; then
  [[ "${LOCAL_FAKE:-0}" == 1 || "${1:-}" == --help || "${1:-}" == -h ]] ||
    { echo "PRECHECK_FAILED inherit_errexit unavailable" >&2; exit 2; }
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
MUTANT_HELPER="$SCRIPT_DIR/mutants.py"
HEAD_SHA="${HEAD_SHA:-}"
RUN_ROOT="${RUN_ROOT:-}"
WORK_ROOT="${WORK_ROOT:-}"
REPO_URL="${REPO_URL:-}"
SHARED_ENV_ROOT="${SHARED_ENV_ROOT:-}"
SHARED_ENV_PROJECT="${SHARED_ENV_PROJECT:-tpen}"
SHARED_ENV_EXTRA="${SHARED_ENV_EXTRA:-cpu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CANNON_XDIST_VERSION="${CANNON_XDIST_VERSION:-3.8.0}"
UV_BIN="${UV_BIN:-}"
FACILITY_RUN_PREFIX="${FACILITY_RUN_PREFIX:-}"
MUTANT_SPEC="${MUTANT_SPEC:-}"
MODE="${MODE:-quick}"
TYPECHECK="${TYPECHECK:-off}"
PARALLEL="${PARALLEL:-xdist}"
FULL_SUITE="${FULL_SUITE:-0}"
TEST_PATHS=()
MUTANT_IDS=()
MUTANT_STATUS=()

usage() {
  cat >&2 <<'USAGE'
usage: verify.sh --sha SHA --test PATH [--test PATH ...] [options]
required env: SLURM_JOB_ID, USER, RUN_ROOT, WORK_ROOT, REPO_URL,
SHARED_ENV_ROOT, UV_BIN, FACILITY_RUN_PREFIX
options: --sha SHA --test PATH --run-root PATH --mutants PATH
         --mode quick|measure --typecheck on|off --parallel serial|xdist
         --full-suite
USAGE
}
while (($#)); do
  case "$1" in
    --sha) HEAD_SHA="${2:?missing SHA}"; shift 2 ;;
    --test) TEST_PATHS+=("${2:?missing test path}"); shift 2 ;;
    --run-root) RUN_ROOT="${2:?missing run root}"; shift 2 ;;
    --mutants) MUTANT_SPEC="${2:?missing mutant spec}"; shift 2 ;;
    --mode) MODE="${2:?missing mode}"; shift 2 ;;
    --typecheck) TYPECHECK="${2:?missing mode}"; shift 2 ;;
    --parallel) PARALLEL="${2:?missing mode}"; shift 2 ;;
    --full-suite) FULL_SUITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "PRECHECK_FAILED unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
if ((${#TEST_PATHS[@]} == 0)) && [[ -n "${TEST_PATHS_ENV:-}" ]]; then
  read -r -a TEST_PATHS <<< "$TEST_PATHS_ENV"
fi
fail_precheck() { echo "PRECHECK_FAILED $*" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" ]] || fail_precheck "SLURM_JOB_ID required"
[[ -n "${USER:-}" ]] || fail_precheck "USER required"
[[ "$HEAD_SHA" =~ ^[0-9a-fA-F]{40}$ ]] || fail_precheck "HEAD_SHA must be 40 hex characters"
[[ "$MODE" == quick || "$MODE" == measure ]] || fail_precheck "invalid MODE"
[[ "$TYPECHECK" == on || "$TYPECHECK" == off ]] || fail_precheck "invalid TYPECHECK"
[[ "$PARALLEL" == serial || "$PARALLEL" == xdist ]] || fail_precheck "invalid PARALLEL"
[[ -n "$RUN_ROOT" && -n "$WORK_ROOT" && -n "$REPO_URL" ]] || fail_precheck "missing required paths"
[[ -n "$SHARED_ENV_ROOT" && -n "$FACILITY_RUN_PREFIX" ]] || fail_precheck "missing env prefix"
[[ "$WORK_ROOT" == "/scratch/$USER/$SLURM_JOB_ID" ]] || fail_precheck "WORK_ROOT must be exact job scratch"
[[ "$RUN_ROOT" == "$FACILITY_RUN_PREFIX"/* ]] || fail_precheck "RUN_ROOT outside facility prefix"
[[ "$SHARED_ENV_ROOT" == "$FACILITY_RUN_PREFIX"/* ]] || fail_precheck "SHARED_ENV_ROOT outside facility prefix"
[[ -x "$UV_BIN" && "$UV_BIN" == /* ]] || fail_precheck "UV_BIN must be executable absolute path"
[[ ${#TEST_PATHS[@]} -gt 0 || "$MODE" == measure ]] || fail_precheck "no test paths"
[[ -f "$MUTANT_HELPER" ]] || fail_precheck "mutant helper missing"
[[ -z "$MUTANT_SPEC" || -s "$MUTANT_SPEC" ]] || fail_precheck "mutant spec missing or empty"

assert_write_path() {
  case "$1" in
    "$WORK_ROOT"|"$WORK_ROOT"/*|"$RUN_ROOT"|"$RUN_ROOT"/*|"$SHARED_ENV_ROOT"|"$SHARED_ENV_ROOT"/*|"$SHARED_ENV_ROOT".lock|"$FACILITY_RUN_PREFIX"|"$FACILITY_RUN_PREFIX"/*) ;;
    *) echo "PREFIX_ASSERT_FAILED $1" >&2; return 1 ;;
  esac
}
safe_mkdir() { assert_write_path "$1"; mkdir -p -- "$1"; }

assert_write_path "$RUN_ROOT"; assert_write_path "$WORK_ROOT"
safe_mkdir "$RUN_ROOT"; safe_mkdir "$RUN_ROOT/evidence"; safe_mkdir "$WORK_ROOT"
assert_write_path "$RUN_ROOT/evidence/driver.log"
exec > "$RUN_ROOT/evidence/driver.log" 2>&1
cleanup() {
  local rc=$?
  set +e
  printf 'DRIVER_EXIT=%s\n' "$rc" > "$RUN_ROOT/evidence/driver-exit.txt"
  date -u +%FT%TZ > "$RUN_ROOT/evidence/driver-end-utc.txt"
  [[ "$WORK_ROOT" == "/scratch/$USER/$SLURM_JOB_ID" ]] && rm -rf -- "$WORK_ROOT"
  exit "$rc"
}
trap cleanup EXIT

printf 'JOB=%s HOST=%s PARTITION=%s UTC=%s SHA=%s MODE=%s\n' \
  "$SLURM_JOB_ID" "$(hostname -f)" "${SLURM_JOB_PARTITION:-unset}" "$(date -u +%FT%TZ)" "$HEAD_SHA" "$MODE"
CLONE="$WORK_ROOT/clone"
OVERLAY="$WORK_ROOT/overlay"
export UV_CACHE_DIR="$WORK_ROOT/uv-cache" TMPDIR="$WORK_ROOT/tmp" TZ=UTC PYTHONDONTWRITEBYTECODE=1 COPYFILE_DISABLE=1
unset PYTHONPATH
safe_mkdir "$UV_CACHE_DIR"; safe_mkdir "$TMPDIR"
assert_write_path "$RUN_ROOT/evidence/clone.log"
git clone --no-single-branch --tags "$REPO_URL" "$CLONE" > "$RUN_ROOT/evidence/clone.log" 2>&1
git -C "$CLONE" checkout --detach "$HEAD_SHA" >> "$RUN_ROOT/evidence/clone.log" 2>&1
git -C "$CLONE" fetch --tags --force >> "$RUN_ROOT/evidence/clone.log" 2>&1
[[ "$(git -C "$CLONE" rev-parse HEAD)" == "$HEAD_SHA" ]] || fail_precheck "exact SHA assertion failed"
[[ "$(git -C "$CLONE" rev-parse --is-shallow-repository)" == false ]] || fail_precheck "clone is shallow"
[[ -z "$(git -C "$CLONE" status --porcelain)" ]] || fail_precheck "clone is dirty"

LOCK_HASH="$(sha256sum "$CLONE/uv.lock" | cut -c1-12)"
SHARED_ENV_NAME="$(basename "$SHARED_ENV_ROOT")"
[[ "$SHARED_ENV_NAME" == *"$SHARED_ENV_PROJECT"*"$LOCK_HASH"*"$SHARED_ENV_EXTRA"*py"$PYTHON_VERSION"* ]] ||
  fail_precheck "shared env name must carry project, lock sha12, extra, and Python version"
SHARED_LOCK="$SHARED_ENV_ROOT.lock"
assert_write_path "$SHARED_LOCK"; assert_write_path "$RUN_ROOT/evidence/shared-env-build.log"
safe_mkdir "$(dirname -- "$SHARED_ENV_ROOT")"
flock -x "$SHARED_LOCK" bash -c '
  set -Eeuo pipefail
  root="$1"; clone="$2"; uv="$3"; extra="$4"; py="$5"; log="$6"; donefile="$7"
  if [[ ! -x "$root/bin/python" || ! -f "$donefile" ]]; then
    mkdir -p "$root"; export UV_PROJECT_ENVIRONMENT="$root"; cd "$clone"
    "$uv" sync --locked --no-install-project --extra "$extra" --python "$py" > "$log" 2>&1
    : > "$donefile"
  fi
  test -x "$root/bin/python"
' bash "$SHARED_ENV_ROOT" "$CLONE" "$UV_BIN" "$SHARED_ENV_EXTRA" "$PYTHON_VERSION" \
  "$RUN_ROOT/evidence/shared-env-build.log" "$SHARED_ENV_ROOT/.complete"

BASE_SITE="$($SHARED_ENV_ROOT/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
safe_mkdir "$OVERLAY"
$SHARED_ENV_ROOT/bin/python -m venv --without-pip "$OVERLAY" > "$RUN_ROOT/evidence/overlay-build.log" 2>&1
OVERLAY_SITE="$($OVERLAY/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
assert_write_path "$OVERLAY_SITE/_tpen_shared_env.pth"
printf '%s\n' "$BASE_SITE" > "$OVERLAY_SITE/_tpen_shared_env.pth"
$UV_BIN pip install --python "$OVERLAY/bin/python" --no-deps -e "$CLONE" >> "$RUN_ROOT/evidence/overlay-build.log" 2>&1
$UV_BIN pip install --python "$OVERLAY/bin/python" "pytest-xdist==$CANNON_XDIST_VERSION" >> "$RUN_ROOT/evidence/overlay-build.log" 2>&1
PYTHON="$OVERLAY/bin/python"
export PYTHON TPEN_PYTHON="$PYTHON" ROOT_PYTHON="$PYTHON"
env -u PYTHONPATH "$PYTHON" - "$CLONE" > "$RUN_ROOT/evidence/interpreter-identity.txt" 2>&1 <<'PYCHECK'
import importlib, os, sys
clone = os.path.realpath(sys.argv[1])
for name in ("tpen", "torch", "pytest"):
    module = importlib.import_module(name)
    print(f"{name}={os.path.realpath(module.__file__)}")
assert os.path.realpath(importlib.import_module("tpen").__file__).startswith(clone + os.sep)
PYCHECK
env -u PYTHONPATH "$PYTHON" -I - "$CLONE" >> "$RUN_ROOT/evidence/interpreter-identity.txt" 2>&1 <<'PYCHECK'
import importlib, os, sys
clone = os.path.realpath(sys.argv[1])
for name in ("tpen", "torch", "pytest"):
    module = importlib.import_module(name)
    print(f"isolated={sys.flags.isolated} {name}={os.path.realpath(module.__file__)}")
assert os.path.realpath(importlib.import_module("tpen").__file__).startswith(clone + os.sep)
PYCHECK

pytest_args() {
  local typecheck="$1" parallel="$2"; local -a args=(-q)
  [[ "$typecheck" == off ]] && args+=(-o addopts=)
  [[ "$parallel" == xdist ]] && args+=(-n "${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK required for xdist}")
  printf '%s\0' "${args[@]}"
}
run_pytest() {
  local name="$1" typecheck="$2" parallel="$3" mode="$4"; shift 4
  local log="$RUN_ROOT/evidence/$name.log" timing="$RUN_ROOT/evidence/$name.timing.tsv"
  local start end rc=0 item; local -a args=()
  while IFS= read -r -d '' item; do args+=("$item"); done < <(pytest_args "$typecheck" "$parallel")
  [[ "$mode" == targeted ]] && args+=("$@")
  assert_write_path "$log"; start="$(date +%s.%N)"
  (cd "$CLONE" && "$PYTHON" -m pytest "${args[@]}" > "$log" 2>&1) || rc=$?
  end="$(date +%s.%N)"; assert_write_path "$timing"
  printf 'name=%s\ttypecheck=%s\tparallel=%s\trc=%s\tseconds=%s\n' "$name" "$typecheck" "$parallel" "$rc" \
    "$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e-s }')" > "$timing"
  return "$rc"
}
purge_pycache() { assert_write_path "$CLONE"; find "$CLONE" -type d -name __pycache__ -prune -exec rm -rf -- {} +; }

if [[ "$MODE" == quick ]]; then
  baseline_rc=0; run_pytest baseline "$TYPECHECK" "$PARALLEL" targeted "${TEST_PATHS[@]}" || baseline_rc=$?
  mutant_fail=0
  if [[ -n "$MUTANT_SPEC" ]]; then
    spec_json="$($PYTHON "$MUTANT_HELPER" validate --spec "$MUTANT_SPEC")"
    mapfile -t MUTANT_IDS < <(printf '%s' "$spec_json" | "$PYTHON" -c 'import json,sys; print(*[x["id"] for x in json.load(sys.stdin)], sep="\n")')
    for mutant_id in "${MUTANT_IDS[@]}"; do
      backup="$WORK_ROOT/mutant-$mutant_id.original"; assert_write_path "$backup"
      apply_json=""; edit_rc=0
      apply_json="$($PYTHON "$MUTANT_HELPER" apply --spec "$MUTANT_SPEC" --source "$CLONE" --backup "$backup" --id "$mutant_id")" || edit_rc=$?
      assert_write_path "$RUN_ROOT/evidence/mutant-$mutant_id-edit.json"; printf '%s\n' "$apply_json" > "$RUN_ROOT/evidence/mutant-$mutant_id-edit.json"
      if [[ "$edit_rc" != 0 ]]; then MUTANT_STATUS+=("$mutant_id=EDIT_FAILED"); mutant_fail=1; continue; fi
      target="$(printf '%s' "$apply_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["target"])')"
      target_rel="${target#$CLONE/}"; assert_write_path "$CLONE/$target_rel"
      if git -C "$CLONE" diff --quiet -- "$target_rel"; then MUTANT_STATUS+=("$mutant_id=EMPTY_DIFF"); exit 1; fi
      mutant_rc=0; run_pytest "mutant-$mutant_id" "$TYPECHECK" "$PARALLEL" targeted "${TEST_PATHS[@]}" || mutant_rc=$?
      if [[ "$baseline_rc" == 0 && "$mutant_rc" != 0 ]]; then MUTANT_STATUS+=("$mutant_id=KILLED")
      elif [[ "$baseline_rc" == 0 ]]; then MUTANT_STATUS+=("$mutant_id=SURVIVED"); mutant_fail=1
      else MUTANT_STATUS+=("$mutant_id=NOT_BASELINE_GREEN"); mutant_fail=1; fi
      "$PYTHON" "$MUTANT_HELPER" restore --source "$CLONE" --backup "$backup" --target "$target_rel"
      [[ -z "$(git -C "$CLONE" status --porcelain)" ]] || fail_precheck "mutant restore left dirty clone"
      purge_pycache
    done
  fi
  result=PASS; [[ "$baseline_rc" == 0 && "$mutant_fail" == 0 ]] || result=FAIL
  printf '%s baseline_rc=%s mutants=%s sha=%s statuses=%s\n' "$result" "$baseline_rc" "${#MUTANT_IDS[@]}" "$HEAD_SHA" "${MUTANT_STATUS[*]:-none}"
  [[ "$result" == PASS ]]
  exit
fi

assert_write_path "$RUN_ROOT/evidence/measurements.tsv"
printf 'name\ttypecheck\tparallel\trc\tseconds\n' > "$RUN_ROOT/evidence/measurements.tsv"
measure_fail=0
for combo in on:serial on:xdist off:serial off:xdist; do
  IFS=: read -r typecheck parallel <<< "$combo"; rc=0
  run_pytest "targeted-$typecheck-$parallel" "$typecheck" "$parallel" targeted "${TEST_PATHS[@]}" || rc=$?
  timing="$RUN_ROOT/evidence/targeted-$typecheck-$parallel.timing.tsv"
  printf '%s\t%s\t%s\t%s\t%s\n' "$typecheck" "$parallel" "$rc" "$(awk -F $'\t' '{print $5}' "$timing")" "$HEAD_SHA" >> "$RUN_ROOT/evidence/measurements.tsv"
  [[ "$rc" == 0 ]] || measure_fail=1
done
if [[ "$FULL_SUITE" == 1 ]]; then
  rc=0; run_pytest full-suite on xdist all || rc=$?
  printf 'on\txdist\t%s\t%s\t%s\n' "$rc" "$(awk -F $'\t' '{print $5}' "$RUN_ROOT/evidence/full-suite.timing.tsv")" "$HEAD_SHA" >> "$RUN_ROOT/evidence/measurements.tsv"
  [[ "$rc" == 0 ]] || measure_fail=1
fi
result=PASS; [[ "$measure_fail" == 0 ]] || result=FAIL
printf '%s measurement_failures=%s sha=%s\n' "$result" "$measure_fail" "$HEAD_SHA"
exit "$measure_fail"
