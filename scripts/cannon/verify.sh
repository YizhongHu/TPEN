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
SHARED_ENV_PREFIX="${SHARED_ENV_PREFIX:-}"
SHARED_ENV_PROJECT="${SHARED_ENV_PROJECT:-tpen}"
SHARED_ENV_EXTRA="${SHARED_ENV_EXTRA:-cpu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CANNON_XDIST_VERSION="${CANNON_XDIST_VERSION:-3.8.0}"
UV_BIN="${UV_BIN:-}"
FACILITY_RUN_PREFIX="${FACILITY_RUN_PREFIX:-}"
MUTANT_SPEC="${MUTANT_SPEC:-}"
MODE="${MODE:-quick}"
TYPECHECK="${TYPECHECK:-on}"
PARALLEL="${PARALLEL:-serial}"
FULL_SUITE="${FULL_SUITE:-0}"
TEST_PATHS=()
MUTANT_IDS=()
MUTANT_STATUS=()

usage() {
  cat >&2 <<'USAGE'
usage: verify.sh --sha SHA --test PATH [--test PATH ...] [options]
required env: SLURM_JOB_ID, USER, RUN_ROOT, WORK_ROOT, REPO_URL,
SHARED_ENV_PREFIX, UV_BIN, FACILITY_RUN_PREFIX
options: --sha SHA --test PATH --run-root PATH --mutants PATH
         --mode quick|measure --typecheck on|off --parallel serial|xdist
         --full-suite
         type checking defaults to on; use --typecheck off explicitly when justified
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
[[ -n "$SHARED_ENV_PREFIX" && -n "$FACILITY_RUN_PREFIX" ]] || fail_precheck "missing env prefix"
[[ -x "$UV_BIN" && "$UV_BIN" == /* ]] || fail_precheck "UV_BIN must be executable absolute path"
[[ ${#TEST_PATHS[@]} -gt 0 || "$MODE" == measure ]] || fail_precheck "no test paths"
[[ -f "$MUTANT_HELPER" ]] || fail_precheck "mutant helper missing"
[[ -z "$MUTANT_SPEC" || -s "$MUTANT_SPEC" ]] || fail_precheck "mutant spec missing or empty"
command -v realpath >/dev/null 2>&1 || fail_precheck "realpath required for path containment checks"

# Canonicalize before any containment check: resolves "..", ".", duplicate
# slashes, and symlinks of existing ancestors, so a traversal or symlink
# escape cannot slip past a raw-string prefix match. `-m` tolerates a
# not-yet-created tail (e.g. WORK_ROOT/RUN_ROOT before mkdir).
canon() { realpath -m -- "$1"; }
CANON_WORK_ROOT="$(canon "$WORK_ROOT")"
CANON_RUN_ROOT="$(canon "$RUN_ROOT")"
CANON_FACILITY_RUN_PREFIX="$(canon "$FACILITY_RUN_PREFIX")"
CANON_JOB_SCRATCH="$(canon "/scratch/$USER/$SLURM_JOB_ID")"
[[ "$CANON_WORK_ROOT" == "$CANON_JOB_SCRATCH" ]] || fail_precheck "WORK_ROOT must be exact job scratch"
[[ "$CANON_RUN_ROOT" == "$CANON_FACILITY_RUN_PREFIX" || "$CANON_RUN_ROOT" == "$CANON_FACILITY_RUN_PREFIX"/* ]] ||
  fail_precheck "RUN_ROOT outside facility prefix"
# SHARED_ENV_PREFIX is a reuse-only, out-of-repo shared env root: it must sit
# entirely outside FACILITY_RUN_PREFIX, never inside it (this script never
# writes there -- see shared-env-precheck.sh, called once the clone's exact
# uv.lock sha256 is known, below).
"$SCRIPT_DIR/shared-env-precheck.sh" paths --shared-env-prefix "$SHARED_ENV_PREFIX" \
  --facility-run-prefix "$FACILITY_RUN_PREFIX" --run-root "$RUN_ROOT" --work-root "$WORK_ROOT"

assert_write_path() {
  local candidate canon_candidate root
  candidate="$1"
  canon_candidate="$(canon "$candidate")" || { echo "PREFIX_ASSERT_FAILED unresolvable: $candidate" >&2; return 1; }
  for root in "$CANON_WORK_ROOT" "$CANON_RUN_ROOT" "$CANON_FACILITY_RUN_PREFIX"; do
    if [[ "$canon_candidate" == "$root" || "$canon_candidate" == "$root"/* ]]; then
      return 0
    fi
  done
  echo "PREFIX_ASSERT_FAILED $candidate -> $canon_candidate" >&2
  return 1
}
safe_mkdir() { assert_write_path "$1"; mkdir -p -- "$1"; }

assert_write_path "$RUN_ROOT"; assert_write_path "$WORK_ROOT"
safe_mkdir "$RUN_ROOT"; safe_mkdir "$RUN_ROOT/evidence"; safe_mkdir "$WORK_ROOT"
assert_write_path "$RUN_ROOT/evidence/driver.log"
exec > "$RUN_ROOT/evidence/driver.log" 2>&1

# Full-suite runs are the only runs that retain job-local test artifacts.  The
# allowlist is intentionally explicit: source checkouts, overlays, dependency
# caches, and bytecode are never copied into RUN_ROOT.
artifact_is_excluded() {
  local rel="$1"
  case "/$rel/" in
    */.git/*|*/.git/|*/__pycache__/*|*/__pycache__/|*/.pytest_cache/*|*/.pytest_cache/|\
      */.mypy_cache/*|*/.mypy_cache/|*/.ruff_cache/*|*/.ruff_cache/|*/.venv/*|*/.venv/)
      return 0
      ;;
  esac
  return 1
}

collect_full_suite_artifacts() {
  local inventory="$RUN_ROOT/artifacts/.INVENTORY.sha256.tmp"
  local metadata="$RUN_ROOT/artifacts/.artifact-inventory.tsv.tmp"
  local status_file="$RUN_ROOT/evidence/artifacts-status.txt"
  local size_file="$RUN_ROOT/evidence/artifacts-size.txt"
  local collector_summary collector_rc collector_status collector_count collector_bytes collector_reason
  ARTIFACT_FILE_COUNT=0
  ARTIFACT_BYTES=0
  [[ ! -e "$RUN_ROOT/artifacts/INVENTORY.sha256" ]] || return 1
  [[ ! -e "$RUN_ROOT/artifacts/artifact-inventory.tsv" ]] || return 1
  assert_write_path "$RUN_ROOT/artifacts" || return 1
  mkdir -p -- "$RUN_ROOT/artifacts" || return 1
  assert_write_path "$inventory" || return 1
  assert_write_path "$metadata" || return 1
  if collector_summary="$(${PYTHON:-python3} - "$RUN_ROOT/artifacts" "$inventory" "$metadata" "$WORK_ROOT" "$TMPDIR" "$CLONE" "${ARTIFACT_COLLECTION_BUDGET_SECONDS:-900}" <<'PY'
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path


artifacts, inventory_path, metadata_path, work_root, tmpdir, clone, budget = sys.argv[1:]
started = time.monotonic()
deadline = time.monotonic() + float(budget)
delay = float(os.environ.get("ARTIFACT_TEST_COPY_DELAY_SECONDS", "0"))
enum_delay = float(os.environ.get("ARTIFACT_TEST_ENUM_DELAY_SECONDS", "0"))
seen = set()
count = 0
total = 0
partial = False
reason = ""

def budget_exceeded():
    return time.monotonic() >= deadline


def excluded(relative):
    parts = Path(relative).parts
    return any(part in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv"} for part in parts)


def copy_file(source, destination_rel, source_class, producer, inventory, metadata, allow_duplicate=False):
    global count, total, partial, reason
    if excluded(destination_rel):
        return True
    if destination_rel in seen:
        if allow_duplicate:
            return True
        raise RuntimeError(f"duplicate retained path: {destination_rel}")
    if budget_exceeded():
        partial = True
        reason = "collection budget exceeded"
        return False
    seen.add(destination_rel)
    destination = Path(artifacts, destination_rel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    size = 0
    with open(source, "rb") as src, destination.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
            if delay:
                time.sleep(delay)
    digest = hasher.hexdigest()
    escaped_destination = repr(destination_rel)[1:-1]
    escaped_source = repr(str(source))[1:-1]
    inventory.write(f"{digest}  {escaped_destination}\n")
    metadata.write(f"{source_class}\t{escaped_source}\ton\t{escaped_destination}\t{size}\t{digest}\t{producer}\n")
    count += 1
    total += size
    if budget_exceeded():
        partial = True
        reason = "collection budget exceeded"
        return False
    return True


def walk_tree(source, destination_prefix, source_class, producer, inventory, metadata, allow_duplicates=False):
    if not source.is_dir():
        return True
    stack = [(source, destination_prefix)]
    while stack:
        current, prefix = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            raise RuntimeError(f"cannot enumerate {current}: {exc}") from exc
        for entry in entries:
            if enum_delay:
                time.sleep(enum_delay)
            if budget_exceeded():
                return False
            relative = f"{prefix}/{entry.name}"
            if entry.is_dir(follow_symlinks=False):
                stack.append((Path(entry.path), relative))
            elif entry.is_file(follow_symlinks=False) and not copy_file(Path(entry.path), relative, source_class, producer, inventory, metadata, allow_duplicates):
                return False
    return True


def git_paths(*args):
    result = subprocess.run(["git", "-C", clone, *args], check=True, capture_output=True)
    return [os.fsdecode(item) for item in result.stdout.split(b"\0") if item]


if os.environ.get("ARTIFACTS_FORCE_FAILURE") == "1":
    raise RuntimeError("forced artifact collection failure")

with Path(inventory_path).open("w", encoding="utf-8") as inventory, Path(metadata_path).open("w", encoding="utf-8") as metadata:
    metadata.write("source_class\tsource_path\tarm\tretained_path\tbytes\tsha256\tproducer\n")
    if not walk_tree(Path(tmpdir), "tmpdir", "tmpdir", "pytest", inventory, metadata):
        partial = True
        reason = reason or "collection budget exceeded"
    if not partial:
        for entry in os.scandir(work_root):
            name = entry.name
            if name.startswith("basetemp-") and entry.is_dir(follow_symlinks=False):
                if not walk_tree(Path(entry.path), f"basetemp/{name}", "basetemp", "pytest", inventory, metadata):
                    partial = True
                    reason = reason or "collection budget exceeded"
                    break
            elif name in {"clone", "overlay", "uv-cache", "tmp"} or name.startswith(("pytest-", "pytest-of-")):
                continue
            elif entry.is_file(follow_symlinks=False):
                if not (name.startswith("mutant-") or name.endswith((".log", ".json", ".diff", ".spec", ".compile", ".xml", ".tsv"))):
                    raise RuntimeError(f"unclassified WORK_ROOT artifact: {name}")
                if not copy_file(Path(entry.path), f"harness/{name}", "harness", "harness", inventory, metadata):
                    partial = True
                    reason = reason or "collection budget exceeded"
                    break
            elif not entry.is_symlink():
                raise RuntimeError(f"unclassified WORK_ROOT artifact: {name}")
    if not partial:
        for relative in git_paths("diff", "--name-only", "-z") + git_paths("ls-files", "--others", "--exclude-standard", "-z"):
            source = Path(clone, relative)
            if source.is_file() and not copy_file(source, f"checkout/{relative}", "checkout", "pytest", inventory, metadata):
                partial = True
                reason = reason or "collection budget exceeded"
                break
    if not partial:
        for root in ("outputs", "results", "reports", "slurm"):
            if not walk_tree(Path(clone, root), f"checkout/{root}", "checkout", "pytest", inventory, metadata, True):
                partial = True
                reason = reason or "collection budget exceeded"
                break

if partial:
    print(f"PARTIAL\t{count}\t{total}\t{time.monotonic() - started:.3f}\t{reason}")
else:
    print(f"OK\t{count}\t{total}\t{time.monotonic() - started:.3f}")
PY
  )"; then
    collector_rc=0
  else
    collector_rc=$?
  fi
  ((collector_rc == 0)) || return "$collector_rc"
  IFS=$'\t' read -r collector_status collector_count collector_bytes collector_elapsed collector_reason <<< "$collector_summary"
  ARTIFACT_FILE_COUNT="$collector_count"
  ARTIFACT_BYTES="$collector_bytes"
  mv -- "$inventory" "$RUN_ROOT/artifacts/INVENTORY.sha256" || return 1
  mv -- "$metadata" "$RUN_ROOT/artifacts/artifact-inventory.tsv" || return 1
  assert_write_path "$RUN_ROOT/evidence/artifacts-timing.txt" || return 1
  printf 'artifact_collection_seconds=%s\n' "$collector_elapsed" > "$RUN_ROOT/evidence/artifacts-timing.txt" || return 1
  assert_write_path "$size_file" || return 1
  printf 'artifact_file_count=%s artifact_bytes=%s\n' "$ARTIFACT_FILE_COUNT" "$ARTIFACT_BYTES" > "$size_file" || return 1
  if [[ "$collector_status" == PARTIAL ]]; then
    printf 'ARTIFACTS_PARTIAL reason=%s artifact_file_count=%s artifact_bytes=%s\n' \
      "$collector_reason" "$ARTIFACT_FILE_COUNT" "$ARTIFACT_BYTES" > "$status_file" || return 1
  else
    printf 'ARTIFACTS_OK\n' > "$status_file" || return 1
  fi
}

cleanup() {
  local rc=$?
  set +e
  printf 'DRIVER_EXIT=%s\n' "$rc" > "$RUN_ROOT/evidence/driver-exit.txt"
  date -u +%FT%TZ > "$RUN_ROOT/evidence/driver-end-utc.txt"
  if [[ "$MODE" == measure && "$FULL_SUITE" == 1 ]]; then
    artifact_error=""
    collect_full_suite_artifacts || artifact_error="collection failed (rc=$?)"
    if [[ -n "$artifact_error" ]]; then
      printf 'ARTIFACTS_FAILED reason=%s\n' "$artifact_error" > "$RUN_ROOT/evidence/artifacts-status.txt"
    fi
  fi
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
assert_write_path "$CLONE"
assert_write_path "$RUN_ROOT/evidence/clone.log"
git clone --no-single-branch --tags "$REPO_URL" "$CLONE" > "$RUN_ROOT/evidence/clone.log" 2>&1
git -C "$CLONE" fetch origin "$HEAD_SHA" >> "$RUN_ROOT/evidence/clone.log" 2>&1
git -C "$CLONE" checkout --detach "$HEAD_SHA" >> "$RUN_ROOT/evidence/clone.log" 2>&1
git -C "$CLONE" fetch --tags --force >> "$RUN_ROOT/evidence/clone.log" 2>&1
[[ "$(git -C "$CLONE" rev-parse HEAD)" == "$HEAD_SHA" ]] || fail_precheck "exact SHA assertion failed"
[[ "$(git -C "$CLONE" rev-parse --is-shallow-repository)" == false ]] || fail_precheck "clone is shallow"
[[ -z "$(git -C "$CLONE" status --porcelain)" ]] || fail_precheck "clone is dirty"

LOCK_SHA256="$(sha256sum "$CLONE/uv.lock" | cut -d' ' -f1)"
# verify.sh never builds or writes a shared env (slice B1): it only resolves
# and validates an existing, complete one. shared-env-precheck.sh fails
# closed, naming the missing/invalid path, if build-shared-env.sh has not
# already produced it.
SHARED_ENV_PATH="$("$SCRIPT_DIR/shared-env-precheck.sh" resolve \
  --shared-env-prefix "$SHARED_ENV_PREFIX" --project "$SHARED_ENV_PROJECT" \
  --extra "$SHARED_ENV_EXTRA" --python "$PYTHON_VERSION" --lock-sha256 "$LOCK_SHA256")"
assert_write_path "$RUN_ROOT/evidence/shared-env-path.txt"
assert_write_path "$RUN_ROOT/evidence/shared-env-manifest.txt"
printf '%s\n' "$SHARED_ENV_PATH" > "$RUN_ROOT/evidence/shared-env-path.txt"
cp -- "$SHARED_ENV_PATH/.complete" "$RUN_ROOT/evidence/shared-env-manifest.txt"

BASE_SITE="$($SHARED_ENV_PATH/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
safe_mkdir "$OVERLAY"
assert_write_path "$RUN_ROOT/evidence/overlay-build.log"
$SHARED_ENV_PATH/bin/python -m venv --without-pip "$OVERLAY" > "$RUN_ROOT/evidence/overlay-build.log" 2>&1
OVERLAY_SITE="$($OVERLAY/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
assert_write_path "$OVERLAY_SITE/_tpen_shared_env.pth"
printf '%s\n' "$BASE_SITE" > "$OVERLAY_SITE/_tpen_shared_env.pth"
$UV_BIN pip install --python "$OVERLAY/bin/python" --no-deps -e "$CLONE" >> "$RUN_ROOT/evidence/overlay-build.log" 2>&1
$UV_BIN pip install --python "$OVERLAY/bin/python" --no-deps -e "$CLONE/executor" >> "$RUN_ROOT/evidence/overlay-build.log" 2>&1
$UV_BIN pip install --python "$OVERLAY/bin/python" "pytest-xdist==$CANNON_XDIST_VERSION" >> "$RUN_ROOT/evidence/overlay-build.log" 2>&1
PYTHON="$OVERLAY/bin/python"
export PYTHON TPEN_PYTHON="$PYTHON" ROOT_PYTHON="$PYTHON"
export PATH="$OVERLAY/bin:$PATH"
[[ -x "$PYTHON" ]] || fail_precheck "overlay interpreter missing after overlay build"
# Tests read per-caller interpreter selector vars ending in
# _EXECUTOR_PYTHON, _ROOT_PYTHON, or _SCIENCE_PYTHON (e.g. P8_EXECUTOR_PYTHON,
# P9C_ROOT_PYTHON, P10_ROOT_PYTHON, H_V2_SCIENCE_PYTHON) to launch
# subprocesses. Whole-word matching excludes compatibility runtimes,
# non-selector names, and prefixes such as XLA_PYTHON_CLIENT_PREALLOCATE.
# Bind only these selector names after the overlay exists; an unbound selector
# is read as "skip" by its test.
assert_write_path "$RUN_ROOT/evidence/executor-python-bindings.txt"
mapfile -t EXECUTOR_PYTHON_VARS < <(
  grep -rhoE -w '[A-Z][A-Z0-9_]*_(EXECUTOR|ROOT|SCIENCE)_PYTHON' "$CLONE" --include='*.py' 2>/dev/null | sort -u
)
: > "$RUN_ROOT/evidence/executor-python-bindings.txt"
for executor_var in "${EXECUTOR_PYTHON_VARS[@]:-}"; do
  [[ -n "$executor_var" ]] || continue
  case "$executor_var" in
    PYTHON|TPEN_PYTHON|ROOT_PYTHON) continue ;;
  esac
  export "${executor_var}=${PYTHON}"
  printf '%s=%s\n' "$executor_var" "$PYTHON" >> "$RUN_ROOT/evidence/executor-python-bindings.txt"
done
executor_resolution_precheck() {
  local executor_root="$1" resolution_file="$2"
  assert_write_path "$resolution_file"
  if ! {
    printf 'normal='
    env -u PYTHONPATH "$PYTHON" -B -c 'import os, sys, tpen_executor; path = os.path.realpath(tpen_executor.__file__); print(path); raise SystemExit(not path.startswith(os.path.realpath(sys.argv[1]) + os.sep))' "$executor_root"
    printf 'isolated='
    env -u PYTHONPATH "$PYTHON" -I -B -c 'import os, sys, tpen_executor; path = os.path.realpath(tpen_executor.__file__); print(path); raise SystemExit(not path.startswith(os.path.realpath(sys.argv[1]) + os.sep))' "$executor_root"
  } > "$resolution_file" 2>&1; then
    fail_precheck "tpen_executor resolution failed outside exact-SHA clone executor"
  fi
}
EXECUTOR_ROOT="$(canon "$CLONE/executor")"
executor_resolution_precheck "$EXECUTOR_ROOT" "$RUN_ROOT/evidence/executor-resolution.txt"
assert_write_path "$RUN_ROOT/evidence/interpreter-identity.txt"
env -u PYTHONPATH "$PYTHON" - "$CLONE" > "$RUN_ROOT/evidence/interpreter-identity.txt" 2>&1 <<'PYCHECK'
import importlib, os, sys
clone = os.path.realpath(sys.argv[1])
for name in ("tpen", "torch", "pytest"):
    module = importlib.import_module(name)
    print(f"{name}={os.path.realpath(module.__file__)}")
assert os.path.realpath(importlib.import_module("tpen").__file__).startswith(clone + os.sep)
PYCHECK
# -I ignores PYTHONDONTWRITEBYTECODE (exported above); -B is required on
# every isolated invocation that imports from the shared, reuse-only env, or
# a writable/stale shared env gets __pycache__/.pyc writes during this probe
# that assert_write_path cannot intercept (interpreter-internal writes, not
# shell redirects).
env -u PYTHONPATH "$PYTHON" -I -B - "$CLONE" >> "$RUN_ROOT/evidence/interpreter-identity.txt" 2>&1 <<'PYCHECK'
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
  local junit="$RUN_ROOT/evidence/$name.junit.xml"
  local start end rc=0 item; local -a args=()
  while IFS= read -r -d '' item; do args+=("$item"); done < <(pytest_args "$typecheck" "$parallel")
  args+=("--junit-xml=$junit")
  [[ "$mode" == targeted ]] && args+=("$@")
  assert_write_path "$log"; assert_write_path "$junit"; start="$(date +%s.%N)"
  (cd "$CLONE" && "$PYTHON" -m pytest "${args[@]}" > "$log" 2>&1) || rc=$?
  end="$(date +%s.%N)"; assert_write_path "$timing"
  printf 'name=%s\ttypecheck=%s\tparallel=%s\trc=%s\tseconds=%s\n' "$name" "$typecheck" "$parallel" "$rc" \
    "$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e-s }')" > "$timing"
  return "$rc"
}
purge_pycache() { assert_write_path "$CLONE"; find "$CLONE" -type d -name __pycache__ -prune -exec rm -rf -- {} +; }

if [[ "$MODE" == quick ]]; then
  purge_pycache
  baseline_rc=0; run_pytest baseline "$TYPECHECK" "$PARALLEL" targeted "${TEST_PATHS[@]}" || baseline_rc=$?
  # requires_typeguard tests assert a bare TypeCheckError with no manual
  # fallback; tests/conftest.py skips them whenever typeguard is not
  # instrumenting tpen (including this mode's --typecheck off), so the quick
  # check never silently drops their signal without naming the gap here.
  typeguard_skipped=0
  baseline_junit="$RUN_ROOT/evidence/baseline.junit.xml"
  [[ -f "$baseline_junit" ]] && typeguard_skipped="$(grep -c 'requires_typeguard' "$baseline_junit" || true)"
  mutant_fail=0 killed_count=0 survived_count=0 harness_error_count=0
  if [[ -n "$MUTANT_SPEC" ]]; then
    spec_json="$($PYTHON "$MUTANT_HELPER" validate --spec "$MUTANT_SPEC")"
    assert_write_path "$RUN_ROOT/evidence/mutant-spec.json"
    printf '%s\n' "$spec_json" > "$RUN_ROOT/evidence/mutant-spec.json"
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
      assert_write_path "$RUN_ROOT/evidence/mutant-$mutant_id.diff"
      git -C "$CLONE" diff -- "$target_rel" > "$RUN_ROOT/evidence/mutant-$mutant_id.diff"
      purge_pycache
      mutant_rc=0; run_pytest "mutant-$mutant_id" "$TYPECHECK" "$PARALLEL" targeted "${TEST_PATHS[@]}" || mutant_rc=$?
      # Fail closed on any harness condition: a dirty baseline, or a mutant
      # exit code outside pytest's normal {0=all passed, 1=some failed} range
      # (2/3/4=internal or usage error, 5=no tests collected, 137=killed)
      # can never be read as KILLED without inspecting the killer by name.
      if [[ "$baseline_rc" != 0 ]]; then
        MUTANT_STATUS+=("$mutant_id=HARNESS_ERROR:baseline_not_green"); mutant_fail=1; harness_error_count=$((harness_error_count + 1))
      elif [[ "$mutant_rc" != 0 && "$mutant_rc" != 1 ]]; then
        MUTANT_STATUS+=("$mutant_id=HARNESS_ERROR:mutant_rc_$mutant_rc"); mutant_fail=1; harness_error_count=$((harness_error_count + 1))
      else
        verdict_json="$($PYTHON "$MUTANT_HELPER" verdict --spec "$MUTANT_SPEC" --id "$mutant_id" \
          --baseline-junit "$RUN_ROOT/evidence/baseline.junit.xml" \
          --mutant-junit "$RUN_ROOT/evidence/mutant-$mutant_id.junit.xml")"
        assert_write_path "$RUN_ROOT/evidence/mutant-$mutant_id-verdict.json"
        printf '%s\n' "$verdict_json" > "$RUN_ROOT/evidence/mutant-$mutant_id-verdict.json"
        mutant_verdict="$(printf '%s' "$verdict_json" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["verdict"])')"
        case "$mutant_verdict" in
          KILLED) MUTANT_STATUS+=("$mutant_id=KILLED"); killed_count=$((killed_count + 1)) ;;
          SURVIVED) MUTANT_STATUS+=("$mutant_id=SURVIVED"); mutant_fail=1; survived_count=$((survived_count + 1)) ;;
          *) MUTANT_STATUS+=("$mutant_id=HARNESS_ERROR:$mutant_verdict"); mutant_fail=1; harness_error_count=$((harness_error_count + 1)) ;;
        esac
      fi
      "$PYTHON" "$MUTANT_HELPER" restore --spec "$MUTANT_SPEC" --source "$CLONE" --backup "$backup" --target "$target_rel"
      [[ -z "$(git -C "$CLONE" status --porcelain)" ]] || fail_precheck "mutant restore left dirty clone"
      purge_pycache
    done
  fi
  result=PASS; [[ "$baseline_rc" == 0 && "$mutant_fail" == 0 ]] || result=FAIL
  printf '%s baseline_rc=%s mutants=%s killed=%s survived=%s harness_error=%s typeguard_skipped=%s sha=%s statuses=%s\n' \
    "$result" "$baseline_rc" "${#MUTANT_IDS[@]}" "$killed_count" "$survived_count" "$harness_error_count" \
    "$typeguard_skipped" "$HEAD_SHA" "${MUTANT_STATUS[*]:-none}"
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
