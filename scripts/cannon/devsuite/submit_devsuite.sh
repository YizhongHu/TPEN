#!/usr/bin/env bash
# Build, stage, and rehearse the TPEN dev full-suite packet.
# Real submissions require both --submit and DEVSUITE_ALLOW_SUBMIT=1.
set -Eeuo pipefail
export COPYFILE_DISABLE=1 LC_ALL=C

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
FACILITY_PREFIX="/n/netscratch/kozinsky_lab/Lab/rhu"
VERIFY_PREFIX="${FACILITY_PREFIX}/verify"
EXECUTOR_REQUIRED_COMMIT="dc99163e15bdb7f2b4cb482e570ff0e9a0e9a8ec"
DRY_RUN=0
SUBMIT=0
EXECUTOR_HARNESS="${EXECUTOR_HARNESS_DIR:-}"
EXECUTOR_MUTANTS=""
ALLOW_EXECUTOR_MUTANTS=0
EXECUTOR_ONLY=0
STAGE_BASE="${DEVSUITE_STAGE_BASE:-$VERIFY_PREFIX}"
BUNDLE_HEAD_REF=""
BUNDLE_REQUIRED_REF=""

usage() {
  cat <<'USAGE'
usage: submit_devsuite.sh [--dry-run|--submit] [--executor-harness DIR] [--executor-mutants SPEC.json] [--executor-only] SHA

The SHA must be a full 40-hex commit present in the current checkout.  The
dry-run stages both arms, reads back every staged-file sha256, runs bash -n and
bare-SBATCH checks, runs wrapper/harness prechecks on Cannon, and invokes only
sbatch --test-only.  A real submit additionally requires
DEVSUITE_ALLOW_SUBMIT=1.
USAGE
}

fail() {
  printf 'DEVSUITE_FAILED %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --submit) SUBMIT=1; shift ;;
    --executor-harness) EXECUTOR_HARNESS="${2:?missing harness directory}"; shift 2 ;;
    --executor-mutants) EXECUTOR_MUTANTS="${2:?missing mutant spec}"; ALLOW_EXECUTOR_MUTANTS=1; shift 2 ;;
    --executor-only) EXECUTOR_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) fail "unknown option: $1" ;;
    *) break ;;
  esac
done

[[ $# == 1 ]] || { usage >&2; fail "exactly one SHA is required"; }
HEAD_SHA="$1"
[[ "$HEAD_SHA" =~ ^[0-9a-fA-F]{40}$ ]] || fail "SHA must be a full 40-hex commit"
[[ "$DRY_RUN" == 1 || "$SUBMIT" == 1 ]] || fail "choose --dry-run or --submit"
[[ "$DRY_RUN" == 0 || "$SUBMIT" == 0 ]] || fail "--dry-run and --submit are mutually exclusive"
if [[ "$SUBMIT" == 1 && "${DEVSUITE_ALLOW_SUBMIT:-0}" != 1 ]]; then
  fail "real submission is safety-gated; set DEVSUITE_ALLOW_SUBMIT=1 explicitly"
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || fail "not inside a git checkout"
git -C "$REPO_ROOT" cat-file -e "${HEAD_SHA}^{commit}" || fail "SHA is not present in this checkout"
[[ "$(git -C "$REPO_ROOT" rev-parse "${HEAD_SHA}^{commit}")" == "$HEAD_SHA" ]] || fail "SHA did not resolve exactly"
REPO_URL="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null)" || fail "origin remote is unavailable"
[[ -n "$REPO_URL" ]] || fail "origin remote is empty"

[[ "$STAGE_BASE" == "$VERIFY_PREFIX" || "$STAGE_BASE" == "$VERIFY_PREFIX"/* ]] || fail "stage base escapes Cannon verify prefix"
[[ "$STAGE_BASE" != */ ]] || fail "stage base must not have a trailing slash"
[[ -n "$EXECUTOR_HARNESS" ]] || fail "--executor-harness or EXECUTOR_HARNESS_DIR is required"
[[ "$EXECUTOR_HARNESS" == /* ]] || fail "executor harness path must be absolute"
[[ -d "$EXECUTOR_HARNESS" ]] || fail "executor harness directory does not exist"
if [[ -n "$EXECUTOR_MUTANTS" ]]; then
  [[ "$EXECUTOR_MUTANTS" == /* ]] || fail "executor mutant spec path must be absolute"
  [[ -f "$EXECUTOR_MUTANTS" ]] || fail "executor mutant spec does not exist"
fi
for required in job.sbatch mutants.json repo.bundle helpers/validate_inputs.sh helpers/env_audit.py; do
  [[ -f "$EXECUTOR_HARNESS/$required" ]] || fail "executor harness missing $required"
done
grep -q 'H_V2_SCIENCE_PYTHON' "$EXECUTOR_HARNESS/job.sbatch" || fail "executor harness lacks H_V2_SCIENCE_PYTHON"
grep -q 'H_V2_EXECUTOR_PYTHON' "$EXECUTOR_HARNESS/job.sbatch" || fail "executor harness lacks H_V2_EXECUTOR_PYTHON"
if [[ "$ALLOW_EXECUTOR_MUTANTS" == 0 ]]; then
  grep -Eq '"mutants"[[:space:]]*:[[:space:]]*\[[[:space:]]*\]' "$EXECUTOR_HARNESS/mutants.json" || fail "executor harness mutant spec is not empty"
fi
[[ -s "$EXECUTOR_HARNESS/repo.bundle" ]] || fail "executor harness repo.bundle is empty"
[[ "$(git -C "$REPO_ROOT" cat-file -t "$EXECUTOR_REQUIRED_COMMIT" 2>/dev/null || true)" == commit ]] ||
  fail "checkout lacks required executor historical commit $EXECUTOR_REQUIRED_COMMIT"

hash_local() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -- "$1" | cut -d' ' -f1
  else
    fail "sha256sum or shasum is required locally"
  fi
}

quote_for_shell() {
  printf '%q' "$1"
}

TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/devsuite-template.XXXXXX")"
cleanup() {
  local rc=$?
  set +e
  if [[ -n "$BUNDLE_HEAD_REF" ]]; then git -C "$REPO_ROOT" update-ref -d "$BUNDLE_HEAD_REF"; fi
  if [[ -n "$BUNDLE_REQUIRED_REF" ]]; then git -C "$REPO_ROOT" update-ref -d "$BUNDLE_REQUIRED_REF"; fi
  rm -rf -- "$TMP_ROOT"
  exit "$rc"
}
trap cleanup EXIT

SHA8="${HEAD_SHA:0:8}"
SHA12="${HEAD_SHA:0:12}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REMOTE_STAGE="${STAGE_BASE}/dev-fullsuite-${SHA8}-${STAMP}"
ROOT_RUN_ROOT="${REMOTE_STAGE}/root-run"
ROOT_TASK_ROOT="${REMOTE_STAGE}/root"
EXECUTOR_TASK_ROOT="${REMOTE_STAGE}/executor-harness"
EXECUTOR_RUN_ROOT="${REMOTE_STAGE}/executor-run"
ROOT_JOB_NAME="dev-root-${SHA12}-${STAMP}"
EXECUTOR_JOB_NAME="dev-executor-${SHA12}-${STAMP}"

ROOT_LOCAL="$TMP_ROOT/root"
EXECUTOR_LOCAL="$TMP_ROOT/executor"
mkdir -p "$ROOT_LOCAL" "$EXECUTOR_LOCAL"

# The root packet contains the complete scripts/cannon tree from the exact
# SHA; verify.sh is intentionally not edited by this launcher.
git -C "$REPO_ROOT" archive "$HEAD_SHA" scripts/cannon | tar -x -C "$ROOT_LOCAL"

# Copy the approved versioned executor harness as data, then replace only its
# repo.bundle with an exact-head bundle. No harness source is edited here.
tar -C "$EXECUTOR_HARNESS" -cf - . | tar -C "$EXECUTOR_LOCAL" -xf -
if [[ -n "$EXECUTOR_MUTANTS" ]]; then
  cp -- "$EXECUTOR_MUTANTS" "$EXECUTOR_LOCAL/mutants.json"
fi
chmod 0755 "$EXECUTOR_LOCAL/job.sbatch"
# Bundle refs must be branch refs so `git clone bundle` has a default head;
# non-branch refs make clone report an empty repository even when objects exist.
BUNDLE_HEAD_REF="refs/heads/devsuite-bundle-head-$$-${SHA8}"
BUNDLE_REQUIRED_REF="refs/heads/devsuite-bundle-required-$$-${SHA8}"
git -C "$REPO_ROOT" show-ref --verify --quiet "$BUNDLE_HEAD_REF" && fail "temporary bundle ref already exists: $BUNDLE_HEAD_REF"
git -C "$REPO_ROOT" show-ref --verify --quiet "$BUNDLE_REQUIRED_REF" && fail "temporary bundle ref already exists: $BUNDLE_REQUIRED_REF"
git -C "$REPO_ROOT" update-ref "$BUNDLE_HEAD_REF" "$HEAD_SHA"
git -C "$REPO_ROOT" update-ref "$BUNDLE_REQUIRED_REF" "$EXECUTOR_REQUIRED_COMMIT"
git -C "$REPO_ROOT" bundle create "$EXECUTOR_LOCAL/repo.bundle" \
  "$BUNDLE_HEAD_REF" "$BUNDLE_REQUIRED_REF" >/dev/null
git -C "$REPO_ROOT" bundle verify "$EXECUTOR_LOCAL/repo.bundle" >/dev/null

REPO_URL_DEFAULT="$(quote_for_shell "$REPO_URL")"
render() {
  local source="$1" target="$2"
  sed \
    -e "s|__HEAD_SHA__|$HEAD_SHA|g" \
    -e "s|__ROOT_JOB_NAME__|$ROOT_JOB_NAME|g" \
    -e "s|__ROOT_TASK_ROOT__|$ROOT_TASK_ROOT|g" \
    -e "s|__ROOT_RUN_ROOT__|$ROOT_RUN_ROOT|g" \
    -e "s|__REPO_URL_DEFAULT__|$REPO_URL_DEFAULT|g" \
    -e "s|__EXECUTOR_JOB_NAME__|$EXECUTOR_JOB_NAME|g" \
    -e "s|__EXECUTOR_TASK_ROOT__|$EXECUTOR_TASK_ROOT|g" \
    -e "s|__EXECUTOR_RUN_ROOT__|$EXECUTOR_RUN_ROOT|g" \
    -e "s|__ALLOW_EXECUTOR_MUTANTS__|$ALLOW_EXECUTOR_MUTANTS|g" \
    "$source" > "$target"
  chmod 0755 "$target"
}
render "$SCRIPT_DIR/root.sbatch.in" "$ROOT_LOCAL/root.sbatch"
render "$SCRIPT_DIR/executor.sbatch.in" "$EXECUTOR_LOCAL/executor.sbatch"

[[ -f "$ROOT_LOCAL/scripts/cannon/verify.sh" ]] || fail "root packet missing verify.sh"
[[ -f "$ROOT_LOCAL/scripts/cannon/shared-env-manifest.sh" ]] || fail "root packet missing shared-env sidecar"
[[ -f "$ROOT_LOCAL/scripts/cannon/shared-env-policy.json" ]] || fail "root packet missing policy sidecar"
[[ -f "$EXECUTOR_LOCAL/executor.sbatch" ]] || fail "executor packet missing wrapper"

write_local_manifest() {
  local root="$1" manifest="$2" path rel
  : > "$manifest"
  while IFS= read -r path; do
    rel="${path#${root}/}"
    printf '%s %s\n' "$(hash_local "$path")" "$rel" >> "$manifest"
  done < <(find "$root" -type f -print | LC_ALL=C sort)
  [[ -s "$manifest" ]] || fail "empty packet manifest: $root"
}

write_local_manifest "$ROOT_LOCAL" "$TMP_ROOT/root.sha256"
write_local_manifest "$EXECUTOR_LOCAL" "$TMP_ROOT/executor.sha256"
if [[ -n "$EXECUTOR_MUTANTS" ]]; then
  printf 'EXECUTOR_MUTANTS_SHA256 PASS spec=%s sha256=%s\n' "$EXECUTOR_MUTANTS" "$(hash_local "$EXECUTOR_MUTANTS")"
fi

ssh_mux_check() {
  ssh -o BatchMode=yes -O check cannon >/dev/null 2>&1 ||
    fail "Cannon ControlMaster check failed; stopped without attempting login"
}

remote_hash_manifest() {
  local root="$1" local_manifest="$2" remote_manifest="$3" hash_script="$4"
  {
    printf '#!/usr/bin/env bash\nset -Eeuo pipefail\nstage=%q\n' "$root"
    while read -r hash rel; do
      printf 'printf "%%s %%s\\n" "$(sha256sum -- "$stage/%s" | cut -d" " -f1)" %q\n' "$rel" "$rel"
    done < "$local_manifest"
  } > "$hash_script"
  ssh -o BatchMode=yes cannon bash -s < "$hash_script" > "$remote_manifest"
  diff -u "$local_manifest" "$remote_manifest" || fail "remote sha256 readback differs for $root"
}

stage_packet() {
  local local_root="$1" remote_root="$2"
  ssh -o BatchMode=yes cannon mkdir -p -- "$remote_root"
  tar -C "$local_root" -cf - . | ssh -o BatchMode=yes cannon tar -C "$remote_root" -xf -
}

remote_user="$(ssh -o BatchMode=yes cannon id -un)" || fail "could not read Cannon user"
[[ -n "$remote_user" ]] || fail "Cannon user is empty"
ssh_mux_check
stage_packet "$ROOT_LOCAL" "${REMOTE_STAGE}/root"
stage_packet "$EXECUTOR_LOCAL" "$EXECUTOR_TASK_ROOT"
remote_hash_manifest "${REMOTE_STAGE}/root" "$TMP_ROOT/root.sha256" "$TMP_ROOT/root.remote.sha256" "$TMP_ROOT/root-hash.sh"
remote_hash_manifest "$EXECUTOR_TASK_ROOT" "$TMP_ROOT/executor.sha256" "$TMP_ROOT/executor.remote.sha256" "$TMP_ROOT/executor-hash.sh"
printf 'STAGE PASS remote=%s root_files=%s executor_files=%s\n' \
  "$REMOTE_STAGE" "$(wc -l < "$TMP_ROOT/root.sha256" | tr -d ' ')" "$(wc -l < "$TMP_ROOT/executor.sha256" | tr -d ' ')"

remote_bash_n() {
  local path="$1"
  ssh -o BatchMode=yes cannon bash -n "$path"
  printf 'BASH_N PASS %s\n' "$path"
}
remote_exec_check() {
  local path="$1"
  ssh -o BatchMode=yes cannon test -x "$path" || fail "remote exec target is not executable: $path"
  printf 'EXECUTABLE PASS %s\n' "$path"
}
remote_bare_sbatch_check() {
  local path="$1"
  local output rc
  output="$(ssh -o BatchMode=yes cannon "grep -nE '^[[:space:]]*SBATCH([[:space:]]|$)' -- '$path'" 2>&1)" || rc=$?
  rc="${rc:-0}"
  case "$rc" in
    0) fail "bare SBATCH directive found in $path: $output" ;;
    1) : ;;
    *) fail "remote bare-SBATCH check failed for $path (rc=$rc): $output" ;;
  esac
  printf 'SBATCH_PREFIX PASS %s\n' "$path"
}
remote_bash_n "${REMOTE_STAGE}/root/root.sbatch"
remote_bash_n "${EXECUTOR_TASK_ROOT}/executor.sbatch"
remote_bash_n "${EXECUTOR_TASK_ROOT}/job.sbatch"
remote_exec_check "${REMOTE_STAGE}/root/scripts/cannon/verify.sh"
remote_exec_check "${EXECUTOR_TASK_ROOT}/job.sbatch"
remote_bare_sbatch_check "${REMOTE_STAGE}/root/root.sbatch"
remote_bare_sbatch_check "${EXECUTOR_TASK_ROOT}/executor.sbatch"
remote_bare_sbatch_check "${EXECUTOR_TASK_ROOT}/job.sbatch"

ROOT_SUBMIT_JOB_NAME="$(ssh -o BatchMode=yes cannon "sed -n 's/^#SBATCH --job-name=//p' -- '${REMOTE_STAGE}/root/root.sbatch'" | head -n 1)"
EXECUTOR_SUBMIT_JOB_NAME="$(ssh -o BatchMode=yes cannon "sed -n 's/^#SBATCH --job-name=//p' -- '${EXECUTOR_TASK_ROOT}/executor.sbatch'" | head -n 1)"
[[ -n "$ROOT_SUBMIT_JOB_NAME" ]] || fail "root wrapper has no #SBATCH --job-name"
[[ -n "$EXECUTOR_SUBMIT_JOB_NAME" ]] || fail "executor wrapper has no #SBATCH --job-name"
printf 'SBATCH_JOB_NAME PASS root=%s executor=%s\n' "$ROOT_SUBMIT_JOB_NAME" "$EXECUTOR_SUBMIT_JOB_NAME"

BUNDLE_CLONE_CHECK_ROOT="${REMOTE_STAGE}/bundle-clone-check"
bundle_clone_cleanup() {
  local attempt
  for attempt in 1 2 3; do
    if ssh -o BatchMode=yes cannon rm -rf -- "$BUNDLE_CLONE_CHECK_ROOT" &&
       ! ssh -o BatchMode=yes cannon test -e "$BUNDLE_CLONE_CHECK_ROOT"; then
      return 0
    fi
    sleep 1
  done
  return 1
}
bundle_clone_cleanup || fail "bundle clone check temp cleanup failed before check"
ssh -o BatchMode=yes cannon mkdir -p -- "$BUNDLE_CLONE_CHECK_ROOT"
bundle_clone_fail() {
  bundle_clone_cleanup || true
  fail "$1"
}
ssh -o BatchMode=yes cannon git clone --no-checkout --no-hardlinks \
  "${EXECUTOR_TASK_ROOT}/repo.bundle" "$BUNDLE_CLONE_CHECK_ROOT/clone" >/dev/null ||
  bundle_clone_fail "bundle clone check failed"
ssh -o BatchMode=yes cannon git -C "$BUNDLE_CLONE_CHECK_ROOT/clone" bundle verify \
  "${EXECUTOR_TASK_ROOT}/repo.bundle" >/dev/null ||
  bundle_clone_fail "bundle verify check failed"
ssh -o BatchMode=yes cannon git -C "$BUNDLE_CLONE_CHECK_ROOT/clone" checkout --detach \
  "$HEAD_SHA" >/dev/null 2>&1 || bundle_clone_fail "bundle exact-SHA checkout failed"
BUNDLE_CLONE_HEAD="$(ssh -o BatchMode=yes cannon git -C "$BUNDLE_CLONE_CHECK_ROOT/clone" rev-parse HEAD)"
[[ "$BUNDLE_CLONE_HEAD" == "$HEAD_SHA" ]] || bundle_clone_fail "bundle clone HEAD mismatch: $BUNDLE_CLONE_HEAD"
bundle_clone_cleanup || fail "bundle clone check temp cleanup failed"
printf 'BUNDLE_CLONE_CHECK PASS head=%s\n' "$BUNDLE_CLONE_HEAD"

ssh -o BatchMode=yes cannon env \
  SLURM_JOB_ID=99999999 \
  SLURM_JOB_NAME="$ROOT_SUBMIT_JOB_NAME" \
  SLURM_JOB_PARTITION=test \
  USER="$remote_user" \
  HEAD_SHA="$HEAD_SHA" \
  TASK_ROOT="$ROOT_TASK_ROOT" \
  RUN_ROOT="$ROOT_RUN_ROOT" \
  REPO_URL="$REPO_URL" \
  DEVSUITE_PRECHECK=1 \
  bash "${REMOTE_STAGE}/root/root.sbatch"
printf 'ROOT_PRECHECK PASS\n'

ssh -o BatchMode=yes cannon env \
  SLURM_JOB_ID=99999999 \
  SLURM_JOB_NAME="$EXECUTOR_SUBMIT_JOB_NAME" \
  SLURM_JOB_PARTITION=test \
  USER="$remote_user" \
  HEAD_SHA="$HEAD_SHA" \
  JOB_NAME="$EXECUTOR_JOB_NAME" \
  TASK_ROOT="$EXECUTOR_TASK_ROOT" \
  RUN_ROOT="$EXECUTOR_RUN_ROOT" \
  DEVSUITE_PRECHECK=1 \
  bash "${EXECUTOR_TASK_ROOT}/executor.sbatch"
printf 'EXECUTOR_PRECHECK PASS\n'

# Rehearse the harness's own DRY_RUN path. This performs its command, env-audit,
# interpreter, and prefix checks but never clones, syncs, runs pytest, or submits.
REMOTE_PYBIN="$(ssh -o BatchMode=yes cannon bash -s <<'REMOTE_PYBIN_PROBE'
for p in /n/home10/rhu/.local/bin/python3.12 /usr/bin/python3.12 /usr/bin/python3; do
  if [ -x "$p" ] && "$p" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 12) else 1)' >/dev/null 2>&1; then
    printf '%s' "$p"
    exit 0
  fi
done
exit 1
REMOTE_PYBIN_PROBE
)" ||
  fail "no Cannon login-node Python >=3.12 available for harness DRY_RUN"
ssh -o BatchMode=yes cannon env \
  SLURM_JOB_ID=99999999 \
  SLURM_JOB_NAME="$EXECUTOR_SUBMIT_JOB_NAME" \
  SLURM_JOB_PARTITION=test \
  USER="$remote_user" \
  HEAD_SHA="$HEAD_SHA" \
  JOB_NAME="$EXECUTOR_JOB_NAME" \
  TASK_ROOT="$EXECUTOR_TASK_ROOT" \
  HARNESS_ROOT="$EXECUTOR_TASK_ROOT" \
  RUN_ROOT="$EXECUTOR_RUN_ROOT" \
  MUTANT_SPEC="$EXECUTOR_TASK_ROOT/mutants.json" \
  DRY_RUN=1 \
  DRY_NODE_TMP="$EXECUTOR_RUN_ROOT/dry-node" \
  LOCAL_FAKE=1 \
  LOCAL_FAKE_ROOT="$EXECUTOR_TASK_ROOT" \
  ROOT_PYTHON="$REMOTE_PYBIN" \
  EXECUTOR_PYTHON="$REMOTE_PYBIN" \
  COPYFILE_DISABLE=1 \
  PYTHONPATH= \
  bash "${EXECUTOR_TASK_ROOT}/job.sbatch"
printf 'EXECUTOR_HARNESS_DRY_RUN PASS python=%s\n' "$REMOTE_PYBIN"

test_only() {
  local path="$1"
  local output
  output="$(ssh -o BatchMode=yes cannon sbatch --test-only "$path" 2>&1)" || fail "sbatch --test-only rejected $path: $output"
  printf 'SBATCH_TEST_ONLY PASS path=%s output=%s\n' "$path" "$output"
}
test_only "${REMOTE_STAGE}/root/root.sbatch"
test_only "${EXECUTOR_TASK_ROOT}/executor.sbatch"

if [[ "$SUBMIT" == 1 ]]; then
  # This branch is intentionally unreachable in this rehearsal. It exists for
  # the manager's later 12-hourly use after explicit authorization.
  ssh_mux_check
  if [[ "$EXECUTOR_ONLY" == 1 || "$ALLOW_EXECUTOR_MUTANTS" == 1 ]]; then
    ssh -o BatchMode=yes cannon sbatch "${EXECUTOR_TASK_ROOT}/executor.sbatch"
  else
    ssh -o BatchMode=yes cannon sbatch "${REMOTE_STAGE}/root/root.sbatch"
    ssh -o BatchMode=yes cannon sbatch "${EXECUTOR_TASK_ROOT}/executor.sbatch"
  fi
else
  printf 'NO_REAL_SUBMISSION PASS (dry-run only; no sbatch without --test-only)\n'
fi

printf 'DEVSUITE DRY-RUN PASS sha=%s stage=%s root_job=%s executor_job=%s\n' \
  "$HEAD_SHA" "$REMOTE_STAGE" "$ROOT_JOB_NAME" "$EXECUTOR_JOB_NAME"
