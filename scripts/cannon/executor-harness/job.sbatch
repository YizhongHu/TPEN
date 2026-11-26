#!/usr/bin/env bash
# Cannon full-matrix verification harness v2. Submit with --job-name carrying HEAD_SHA.
# Throwaway work is node-local; only small evidence is retained under Lab/rhu/verify.
#SBATCH --partition=test
#SBATCH --account=kozinsky_lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
# FASRC Job Defense Shield rule (Cannon note cpu-cores-match-parallelism-jds-20261110): this harness runs
# plain pytest rows SERIALLY, but executor/tests/test_s1_slots.py pins 2 concurrent slot workers to
# distinct permitted CPUs, so 2 cores are required (1 core = false red, job 59519375; Cannon note
# cpu-cores-affinity-pinned-fixtures-20261110).
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -Eeuo pipefail
# Command substitutions ($(ensure_reused_env ...), $(make_overlay ...)) must abort on failure too.
# bash >= 4.4 (Cannon); only the local macOS bash 3.2 rehearsal may lack it.
if ! shopt -s inherit_errexit 2>/dev/null; then [[ "${LOCAL_FAKE:-0}" == 1 ]] || { echo "bash lacks inherit_errexit" >&2; exit 1; }; fi
: "${SLURM_JOB_ID:?scheduler allocation required}"
: "${TASK_ROOT:?explicit harness root required}"
: "${HEAD_SHA:?exact 40-character HEAD_SHA required}"
: "${JOB_NAME:?job name carrying HEAD_SHA required}"
: "${USER:?USER required}"

readonly UV="${UV:-/n/home10/rhu/.local/bin/uv}"
readonly HARNESS_ROOT="$TASK_ROOT"
readonly RUN_ROOT="${RUN_ROOT:-/n/netscratch/kozinsky_lab/Lab/rhu/verify/${JOB_NAME}}"
readonly NODE_TMP="${NODE_TMP:-${DRY_NODE_TMP:-/scratch/${USER}/${SLURM_JOB_ID}}}"
readonly MUTANT_SPEC="${MUTANT_SPEC:-${HARNESS_ROOT}/mutants.json}"
export UV HARNESS_ROOT RUN_ROOT NODE_TMP MUTANT_SPEC HEAD_SHA JOB_NAME TASK_ROOT
export TZ=UTC PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy COPYFILE_DISABLE=1
export DRY_RUN="${DRY_RUN:-0}" DRY_NODE_TMP="${DRY_NODE_TMP:-}"
export LOCAL_FAKE="${LOCAL_FAKE:-0}" LOCAL_FAKE_ROOT="${LOCAL_FAKE_ROOT:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export PATH="${PATH:-/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin}"

assert_prefix() {
  local candidate="$1"
  case "$candidate" in
    "$NODE_TMP"|"$NODE_TMP"/*|"$HARNESS_ROOT"|"$HARNESS_ROOT"/*|/n/netscratch/kozinsky_lab/Lab/rhu|/n/netscratch/kozinsky_lab/Lab/rhu/*) ;;
    *) printf 'PREFIX_ASSERT_FAILED=%s\n' "$candidate" >&2; return 1 ;;
  esac
}

safe_mkdir() {
  local target="$1"
  assert_prefix "$target"
  mkdir -p "$target"
}

validate_other_inputs() {
  # Pre-driver failures print to stderr: under sbatch stderr is /dev/null, but a login-node DRY_RUN shows it.
  [[ "$HEAD_SHA" =~ ^[0-9a-fA-F]{40}$ ]] || { echo "PRECHECK_FAILED HEAD_SHA not 40 hex: $HEAD_SHA" >&2; return 1; }
  [[ "$JOB_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "PRECHECK_FAILED JOB_NAME charset: $JOB_NAME" >&2; return 1; }
  [[ "$JOB_NAME" == *"${HEAD_SHA:0:8}"* ]] || { echo "PRECHECK_FAILED JOB_NAME must contain ${HEAD_SHA:0:8}: $JOB_NAME" >&2; return 1; }
  [[ "$COPYFILE_DISABLE" == 1 ]] || { echo "PRECHECK_FAILED COPYFILE_DISABLE!=1" >&2; return 1; }
  [[ "$PYTHONPATH" == "" ]] || { echo "PRECHECK_FAILED PYTHONPATH not empty: $PYTHONPATH" >&2; return 1; }
}

source "$HARNESS_ROOT/helpers/validate_inputs.sh"
validate_inputs || { echo "PRECHECK_FAILED validate_inputs (storage roots: NODE_TMP/DRY_NODE_TMP/HARNESS_ROOT/RUN_ROOT/TASK_ROOT must be under Lab/rhu or the exact /scratch/$USER/$SLURM_JOB_ID)" >&2; exit 1; }
validate_other_inputs || exit 1
assert_prefix "$HARNESS_ROOT"
assert_prefix "$RUN_ROOT"
assert_prefix "$NODE_TMP"
assert_prefix "$MUTANT_SPEC"
safe_mkdir "$RUN_ROOT"
safe_mkdir "$RUN_ROOT/evidence"
safe_mkdir "$NODE_TMP"
assert_prefix "$RUN_ROOT/evidence/driver.log"
exec > "$RUN_ROOT/evidence/driver.log" 2>&1

CURRENT_ARM=setup
cleanup() {
  local rc=$?
  set +e
  assert_prefix "$RUN_ROOT/evidence/driver-exit.txt"
  printf 'DRIVER_EXIT=%s\n' "$rc" > "$RUN_ROOT/evidence/driver-exit.txt"
  assert_prefix "$RUN_ROOT/evidence/driver-end-utc.txt"
  date -u +%FT%TZ > "$RUN_ROOT/evidence/driver-end-utc.txt"
  if [[ "$NODE_TMP" == "/scratch/${USER}/${SLURM_JOB_ID}" ]]; then assert_prefix "$NODE_TMP"; rm -rf -- "$NODE_TMP"; fi
  exit "$rc"
}
trap cleanup EXIT
assert_prefix "$RUN_ROOT/evidence/err-trap.log"
trap 'rc=$?; printf "arm=%s BASH_COMMAND=%q LINENO=%s rc=%s\n" "$CURRENT_ARM" "$BASH_COMMAND" "$LINENO" "$rc" >> "$RUN_ROOT/evidence/err-trap.log"' ERR

printf 'JOB=%s JOB_NAME=%s PARTITION=%s HOST=%s UTC=%s\n' "$SLURM_JOB_ID" "$JOB_NAME" "${SLURM_JOB_PARTITION:-unset}" "$(hostname -f)" "$(date -u +%FT%TZ)"
printf 'HEAD_SHA=%s\nTASK_ROOT=%s\nRUN_ROOT=%s\nNODE_TMP=%s\nMUTANT_SPEC=%s\n' "$HEAD_SHA" "$TASK_ROOT" "$RUN_ROOT" "$NODE_TMP" "$MUTANT_SPEC"

required_paths=(job.sbatch final_run.py final_verdict2.py run_pytest.py helpers/path_guard.py helpers/validate_inputs.sh helpers/env_audit.py helpers/reconcile_counts.py mutants.json repo.bundle)
for required_path in "${required_paths[@]}"; do test -f "$HARNESS_ROOT/$required_path"; done
test -s "$HARNESS_ROOT/repo.bundle"
test -s "$MUTANT_SPEC"
test -z "$(find "$HARNESS_ROOT" -type f -name '._*' -print -quit)"

export REVIEW_ARM=setup REVIEW_SOURCE="$HARNESS_ROOT" REVIEW_CAPTURE="$RUN_ROOT/evidence/setup/review-capture"
export ROOT_SOURCE="$NODE_TMP/clone-root"
export ROOT_PYTHON="${ROOT_PYTHON:-$NODE_TMP/fake-root/bin/python}"
export EXECUTOR_PYTHON="${EXECUTOR_PYTHON:-$NODE_TMP/fake-executor/bin/python}"
export P10_ROOT_PYTHON="$ROOT_PYTHON" P5B_REVIEW_SCIENCE_PYTHON="$ROOT_PYTHON"
export P8_EXECUTOR_PYTHON="$EXECUTOR_PYTHON" P9C_ROOT_PYTHON="$ROOT_PYTHON" P9C_EXECUTOR_PYTHON="$EXECUTOR_PYTHON"
export P10_REPO_ROOT="$ROOT_SOURCE" P10_EXECUTOR_REPO_ROOT="$NODE_TMP/clone-c1-py312-parsl/executor"
export C3_EXECUTOR_PYTHON="$NODE_TMP/venv-c3-py312-none/bin/python"
export UV_PROJECT_ENVIRONMENT="$NODE_TMP/unused-env" UV_CACHE_DIR="$NODE_TMP/uv-root" TMPDIR="$NODE_TMP/root-setup"

if [[ "$DRY_RUN" == 1 ]]; then
  command -v mkdir
  command -v tar
  command -v git
  command -v ssh
  command -v sha256sum
  command -v cut
  if command -v flock >/dev/null 2>&1; then
    command -v flock
  else
    [[ "$LOCAL_FAKE" == 1 ]] || { echo "flock is required outside LOCAL_FAKE" >&2; exit 1; }
    printf 'flock=missing (allowed only for LOCAL_FAKE rehearsal)\n'
  fi
  safe_mkdir "$RUN_ROOT/evidence/dry-run"
  assert_prefix "$RUN_ROOT/evidence/dry-run/env-audit.tsv"
  "$ROOT_PYTHON" "$HARNESS_ROOT/helpers/env_audit.py" "$HARNESS_ROOT" > "$RUN_ROOT/evidence/dry-run/env-audit.tsv"
  assert_prefix "$RUN_ROOT/evidence/dry-run/interpreters.tsv"
  : > "$RUN_ROOT/evidence/dry-run/interpreters.tsv"
  for interpreter in "$ROOT_PYTHON" "$EXECUTOR_PYTHON"; do
    test -x "$interpreter"
    "$interpreter" -c 'import sys; print(sys.executable, sys.version); assert sys.version_info[:2] >= (3, 12)' >> "$RUN_ROOT/evidence/dry-run/interpreters.tsv"
  done
  assert_prefix "$RUN_ROOT/evidence/dry-run/prefix-asserts.txt"
  {
    assert_prefix "$NODE_TMP"
    assert_prefix "$RUN_ROOT"
    assert_prefix "$RUN_ROOT/evidence/dry-run"
    assert_prefix "$MUTANT_SPEC"
    printf 'allowed-prefixes=PASS\nnode-tmp=%s\nrun-root=%s\n' "$NODE_TMP" "$RUN_ROOT"
  } > "$RUN_ROOT/evidence/dry-run/prefix-asserts.txt"
  printf 'DRY_RUN=1 PASS; command checks, env audit, interpreter checks, and prefix asserts completed before exit\n'
  exit 0
fi

test -n "${SLURM_JOB_PARTITION:-}"
test "${SLURM_JOB_NAME:-$JOB_NAME}" = "$JOB_NAME"

clone_one() {
  local arm="$1" clone="$NODE_TMP/clone-$1" log="$RUN_ROOT/evidence/$1-clone.log"
  assert_prefix "$clone"; assert_prefix "$log"
  git clone --no-hardlinks "$HARNESS_ROOT/repo.bundle" "$clone" > "$log" 2>&1
  CURRENT_ARM="$arm"
  git -C "$clone" checkout --detach "$HEAD_SHA" >> "$log" 2>&1
  # Bring every bundle ref (e.g. a tag for history-reading tests' commits such as dc99163e) into the clone.
  git -C "$clone" fetch -q "$HARNESS_ROOT/repo.bundle" 'refs/*:refs/bundle/*' >> "$log" 2>&1
  local required_commit
  for required_commit in ${HARNESS_REQUIRED_COMMITS:-dc99163e15bdb7f2b4cb482e570ff0e9a0e9a8ec}; do git -C "$clone" cat-file -e "${required_commit}^{commit}"; done
  git -C "$clone" bundle verify "$HARNESS_ROOT/repo.bundle" >> "$log" 2>&1
  test "$(git -C "$clone" rev-parse HEAD)" = "$HEAD_SHA"
  test "$(git -C "$clone" rev-parse --is-shallow-repository)" = false
  test -z "$(git -C "$clone" status --porcelain)"
}

for arm in c1-py312-parsl c2-py314-parsl c3-py312-none c4-py314-none; do clone_one "$arm"; done
clone_one root

ensure_reused_env() {
  local clone="$1" version="$2" extras="$3" arm="$4" project="$5" lockfile lock_hash env_root lock_path build_log
  # Separate statement: words of one `local` are all expanded before any assignment, so a sibling reference is unbound under set -u.
  lockfile="$clone/uv.lock"
  # Shared-env convention (manager 2026-11-09): envs/<project>-<uv.lock sha256[:12]>-<extras>-py<X.Y>, lock <dir>.lock.
  lock_hash="$(sha256sum "$lockfile" | cut -c1-12)"
  [[ "$lock_hash" =~ ^[0-9a-f]{12}$ ]] || { echo "bad lock hash for $lockfile" >&2; return 1; }
  env_root="/n/netscratch/kozinsky_lab/Lab/rhu/envs/${project}-${lock_hash}-${extras}-py${version}"
  lock_path="${env_root}.lock"; build_log="$RUN_ROOT/evidence/${arm}-env-build.log"
  assert_prefix "$env_root"; assert_prefix "$lock_path"; assert_prefix "$build_log"; assert_prefix "$RUN_ROOT/evidence/reused-envs.tsv"
  safe_mkdir "/n/netscratch/kozinsky_lab/Lab/rhu/envs"
  flock -x "$lock_path" bash -c 'env_root="$1"; clone="$2"; version="$3"; extras="$4"; log="$5"; node_tmp="$6"; if [[ ! -x "$env_root/bin/python" ]]; then export UV_PROJECT_ENVIRONMENT="$env_root" UV_CACHE_DIR="$node_tmp/uv-env-$extras-py$version" TMPDIR="$node_tmp/tmp-env-$extras-py$version"; case "$env_root" in /n/netscratch/kozinsky_lab/Lab/rhu/*) ;; *) exit 91 ;; esac; case "$UV_CACHE_DIR" in "$node_tmp"/*) ;; *) exit 92 ;; esac; case "$TMPDIR" in "$node_tmp"/*) ;; *) exit 93 ;; esac; mkdir -p "$UV_CACHE_DIR" "$TMPDIR"; cd "$clone"; if [[ "$extras" == none ]]; then /n/home10/rhu/.local/bin/uv sync --locked --no-install-project --python "$version" > "$log" 2>&1; else /n/home10/rhu/.local/bin/uv sync --locked --no-install-project --extra "$extras" --python "$version" > "$log" 2>&1; fi; fi; test -x "$env_root/bin/python"' bash "$env_root" "$clone" "$version" "$extras" "$build_log" "$NODE_TMP"
  printf '%s\t%s\t%s\t%s\n' "$arm" "$lock_hash" "$env_root" "reused-read-only-after-flock" >> "$RUN_ROOT/evidence/reused-envs.tsv"
  printf '%s\n' "$env_root"
}

make_overlay() {
  # H-v2 convention r3 (TO 8a67671f h-v2-cannon-harness-convention-20261109-r3): --system-site-packages
  # inherits the BASE CPython site, not the shared env's. Use --without-pip + a .pth to the shared env's
  # site-packages (honoured even under -I) + an EDITABLE install into this job-local overlay only.
  local base_env="$1" clone="$2" arm="$3" import_names="$4" overlay log base_site overlay_site
  overlay="$NODE_TMP/overlay-$arm"; log="$RUN_ROOT/evidence/$arm-overlay-build.log"
  assert_prefix "$overlay"; assert_prefix "$log"; assert_prefix "$RUN_ROOT/evidence/$arm-overlay-identity.txt"
  "$base_env/bin/python" -m venv --without-pip "$overlay" > "$log" 2>&1 || return 1
  base_site="$("$base_env/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  overlay_site="$("$overlay/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  [[ -d "$base_site" && -d "$overlay_site" && "$overlay_site" == "$overlay"/* ]] || return 1
  printf '%s\n' "$base_site" > "$overlay_site/_tpen_shared_env.pth"
  export UV_CACHE_DIR="$NODE_TMP/uv-overlay-$arm" TMPDIR="$NODE_TMP/tmp-overlay-$arm"
  assert_prefix "$UV_CACHE_DIR"; assert_prefix "$TMPDIR"
  mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
  /n/home10/rhu/.local/bin/uv pip install --python "$overlay/bin/python" --no-deps -e "$clone" >> "$log" 2>&1 || return 1
  test -x "$overlay/bin/python" || return 1
  # Identity check under normal AND isolated mode: every module imports, and the project module resolves into this checkout.
  local mode
  for mode in normal isolated; do
    local flags=(); [[ "$mode" == isolated ]] && flags=(-I)
    env -u PYTHONPATH "$overlay/bin/python" ${flags[@]+"${flags[@]}"} - "$clone" "$import_names" >> "$RUN_ROOT/evidence/$arm-overlay-identity.txt" 2>&1 <<'PYCHECK' || return 1
import importlib, os, sys
clone, names = sys.argv[1], sys.argv[2].split(",")
for index, name in enumerate(names):
    module = importlib.import_module(name)
    print(f"mode_flags={sys.flags.isolated} {name} {module.__file__}")
    if index == 0:
        assert os.path.realpath(module.__file__).startswith(os.path.realpath(clone) + os.sep), (name, module.__file__, clone)
PYCHECK
  done
  printf '%s\n' "$overlay"
}

root_env="$(ensure_reused_env "$ROOT_SOURCE" 3.12 cpu root tpen)"
root_overlay="$(make_overlay "$root_env" "$ROOT_SOURCE" root tpen,torch,omegaconf,pytest)"
ROOT_PYTHON="$root_overlay/bin/python"
for arm in c1-py312-parsl c2-py314-parsl c3-py312-none c4-py314-none; do
  version=3.12; [[ "$arm" == c2-* || "$arm" == c4-* ]] && version=3.14
  extras=none; [[ "$arm" == *-parsl ]] && extras=parsl
  env_path="$(ensure_reused_env "$NODE_TMP/clone-$arm/executor" "$version" "$extras" "$arm" tpen-executor)"
  overlay_path="$(make_overlay "$env_path" "$NODE_TMP/clone-$arm/executor" "$arm" tpen_executor,pytest)"
  export "${arm//-/_}_PYTHON=$overlay_path/bin/python"
done

# Re-point every interpreter variable exported before the overlays existed (Cannon job 59359049: a stale
# P9C_ROOT_PYTHON from the submit line made executor-arm tests spawn a yaml-less interpreter).
export P10_ROOT_PYTHON="$ROOT_PYTHON" P5B_REVIEW_SCIENCE_PYTHON="$ROOT_PYTHON" P9C_ROOT_PYTHON="$ROOT_PYTHON"
export EXECUTOR_PYTHON="$c1_py312_parsl_PYTHON" P8_EXECUTOR_PYTHON="$c1_py312_parsl_PYTHON" P9C_EXECUTOR_PYTHON="$c1_py312_parsl_PYTHON"
# H-v2 review tests (pilot_payloads_r1, science_config_r1) require these; same interpreters as P5B/P8 (dev full suite 59600070: 11 setup errors without them).
export H_V2_SCIENCE_PYTHON="$ROOT_PYTHON" H_V2_EXECUTOR_PYTHON="$c1_py312_parsl_PYTHON"
export ROOT_PYTHON ROOT_SOURCE UV_PROJECT_ENVIRONMENT="$root_env" UV_CACHE_DIR="$NODE_TMP/uv-root" TMPDIR="$NODE_TMP/root-setup"
assert_prefix "$RUN_ROOT/evidence/env-audit.tsv"
"$ROOT_PYTHON" "$HARNESS_ROOT/helpers/env_audit.py" "$ROOT_SOURCE" "$HARNESS_ROOT" > "$RUN_ROOT/evidence/env-audit.tsv"
CURRENT_ARM=final-matrix
set +e
"$ROOT_PYTHON" "$HARNESS_ROOT/final_run.py"
run_rc=$?
"$ROOT_PYTHON" "$HARNESS_ROOT/final_verdict2.py" "$RUN_ROOT"
verdict_rc=$?
set -e
assert_prefix "$RUN_ROOT/evidence/final-exit.rc"
printf 'final_run_rc=%s\nfinal_verdict_rc=%s\n' "$run_rc" "$verdict_rc" > "$RUN_ROOT/evidence/final-exit.rc"
exit "$((run_rc != 0 || verdict_rc != 0))"
