#!/usr/bin/env bash
# Local, Slurm-free regression probes for submit-lint.sh's submission
# contract. bash/stdlib only; no pytest, no cluster access, no installs.
# Builds one positive and one negative fixture per check in a throwaway
# tmpdir, runs submit-lint.sh against each, and asserts the exit code.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LINTER="$SCRIPT_DIR/submit-lint.sh"
command -v realpath >/dev/null 2>&1 || { echo "SELFTEST_FAILED realpath required" >&2; exit 2; }

TMP="$(mktemp -d)"
cleanup() { rm -rf -- "$TMP"; }
trap cleanup EXIT
BASE="$(realpath -- "$TMP")"
mkdir -p "$BASE/stage" "$BASE/other-root" "$BASE/home" "$BASE/shared-env/bin"
touch "$BASE/shared-env/bin/python"; chmod +x "$BASE/shared-env/bin/python"

SHA="1111111111111111111111111111111111111111"
SHORT="${SHA:0:12}"

write_manifest() {
  local path="$1" sha="$2" job_name="$3" task_root="$4" run_root="$5" dry_run="$6" python="$7"
  cat > "$path" <<EOF
HEAD_SHA=$sha
JOB_NAME=$job_name
TASK_ROOT=$task_root
RUN_ROOT=$run_root
DRY_RUN=$dry_run
PYTHON=$python
EOF
}

write_sbatch() {
  local path="$1" sha="$2" job_name="$3" task_root="$4" run_root="$5" dry_run="$6" python="$7"
  cat > "$path" <<EOF
#!/bin/bash
#SBATCH --job-name=$job_name
export HEAD_SHA=$sha
export JOB_NAME=$job_name
export TASK_ROOT=$task_root
export RUN_ROOT=$run_root
export DRY_RUN=$dry_run
export PYTHON=$python
EOF
}

GOOD_TASK_ROOT="$BASE/stage/task"
GOOD_RUN_ROOT="$BASE/stage/run"
GOOD_PYTHON="$BASE/shared-env/bin/python"
GOOD_SBATCH="$BASE/stage/good.sbatch"
GOOD_MANIFEST="$BASE/stage/good.manifest"
write_manifest "$GOOD_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
write_sbatch "$GOOD_SBATCH" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"

PASS=0 FAIL=0
expect() {
  local label="$1" want="$2"; shift 2
  local out rc=0
  out="$("$@" 2>&1)" || rc=$?
  if [[ "$want" == pass && "$rc" == 0 ]] || [[ "$want" == fail && "$rc" != 0 ]]; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    printf 'SELFTEST FAILED: %s (expected %s, rc=%s)\n%s\n' "$label" "$want" "$rc" "$out" >&2
  fi
}

lint() { "$LINTER" --sbatch "$1" --manifest "$2" --allowed-root "$BASE/stage" --shared-env-root "$BASE/shared-env" --home "$BASE/home" "${@:3}"; }

# --- Baseline: the shared good fixture must pass on its own. ---
expect "good fixture" pass lint "$GOOD_SBATCH" "$GOOD_MANIFEST"

# --- Check 1: HEAD_SHA must be 40 hex chars. ---
BAD1_MANIFEST="$BASE/stage/bad1.manifest"
write_manifest "$BAD1_MANIFEST" "not-a-sha" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
expect "non-hex HEAD_SHA" fail lint "$GOOD_SBATCH" "$BAD1_MANIFEST"

# --- Check 2: JOB_NAME must contain SHA[:12]. ---
BAD2_MANIFEST="$BASE/stage/bad2.manifest"
BAD2_SBATCH="$BASE/stage/bad2.sbatch"
write_manifest "$BAD2_MANIFEST" "$SHA" "tpen-verify-nosha" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
write_sbatch "$BAD2_SBATCH" "$SHA" "tpen-verify-nosha" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
expect "JOB_NAME missing SHA[:12]" fail lint "$BAD2_SBATCH" "$BAD2_MANIFEST"

# --- Check 3: sbatch path must be absolute, canonical, inside stage root. ---
expect "relative sbatch path" fail "$LINTER" --sbatch "stage/good.sbatch" --manifest "$GOOD_MANIFEST" --allowed-root "$BASE/stage" --shared-env-root "$BASE/shared-env" --home "$BASE/home"
mkdir -p "$BASE/outside"
OUTSIDE_SBATCH="$BASE/outside/good.sbatch"
cp "$GOOD_SBATCH" "$OUTSIDE_SBATCH"
expect "sbatch path outside allowed root" fail lint "$OUTSIDE_SBATCH" "$GOOD_MANIFEST"

# --- Check 4: critical exports must match the manifest (stale/missing/extra). ---
BAD4_SBATCH="$BASE/stage/bad4-stale.sbatch"
write_sbatch "$BAD4_SBATCH" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT-stale" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
expect "stale critical export" fail lint "$BAD4_SBATCH" "$GOOD_MANIFEST"
BAD4B_SBATCH="$BASE/stage/bad4-missing.sbatch"
cat > "$BAD4B_SBATCH" <<EOF
#!/bin/bash
export HEAD_SHA=$SHA
export JOB_NAME=tpen-verify-$SHORT
export RUN_ROOT=$GOOD_RUN_ROOT
export DRY_RUN=0
export PYTHON=$GOOD_PYTHON
EOF
expect "missing critical export" fail lint "$BAD4B_SBATCH" "$GOOD_MANIFEST"
BAD4C_MANIFEST="$BASE/stage/bad4-extra.manifest"
cat > "$BAD4C_MANIFEST" <<EOF
HEAD_SHA=$SHA
JOB_NAME=tpen-verify-$SHORT
RUN_ROOT=$GOOD_RUN_ROOT
DRY_RUN=0
PYTHON=$GOOD_PYTHON
EOF
expect "extra critical export" fail lint "$GOOD_SBATCH" "$BAD4C_MANIFEST"

# --- Check 5: no $HOME staging, literal or resolved. ---
BAD5_MANIFEST="$BASE/stage/bad5.manifest"
BAD5_SBATCH="$BASE/stage/bad5.sbatch"
HOME_TASK="$BASE/home/staged-task"
write_manifest "$BAD5_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$HOME_TASK" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
write_sbatch "$BAD5_SBATCH" "$SHA" "tpen-verify-$SHORT" "$HOME_TASK" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
expect "literal \$HOME staging" fail "$LINTER" --sbatch "$BAD5_SBATCH" --manifest "$BAD5_MANIFEST" --allowed-root "$BASE/stage" --allowed-root "$BASE/home" --shared-env-root "$BASE/shared-env" --home "$BASE/home"
mkdir -p "$BASE/home/real-staged"
ln -s "$BASE/home/real-staged" "$BASE/stage/home-symlink"
SYMLINK_TASK="$BASE/stage/home-symlink/sub"
BAD5B_MANIFEST="$BASE/stage/bad5b.manifest"
BAD5B_SBATCH="$BASE/stage/bad5b.sbatch"
write_manifest "$BAD5B_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$SYMLINK_TASK" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
write_sbatch "$BAD5B_SBATCH" "$SHA" "tpen-verify-$SHORT" "$SYMLINK_TASK" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
expect "resolved \$HOME staging via symlink" fail "$LINTER" --sbatch "$BAD5B_SBATCH" --manifest "$BAD5B_MANIFEST" --allowed-root "$BASE/stage" --allowed-root "$BASE/home" --shared-env-root "$BASE/shared-env" --home "$BASE/home"

# --- Check 6: facility-root escape for TASK_ROOT/RUN_ROOT. ---
BAD6_MANIFEST="$BASE/stage/bad6.manifest"
BAD6_SBATCH="$BASE/stage/bad6.sbatch"
ESCAPED="$BASE/other-root/task"
write_manifest "$BAD6_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$ESCAPED" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
write_sbatch "$BAD6_SBATCH" "$SHA" "tpen-verify-$SHORT" "$ESCAPED" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
expect "TASK_ROOT escapes allowed root" fail lint "$BAD6_SBATCH" "$BAD6_MANIFEST"

# --- Check 7: dry-run interpreter must be under the shared-env root. ---
BAD7_MANIFEST="$BASE/stage/bad7.manifest"
BAD7_SBATCH="$BASE/stage/bad7.sbatch"
write_manifest "$BAD7_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 1 "/usr/bin/python3"
write_sbatch "$BAD7_SBATCH" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 1 "/usr/bin/python3"
expect "dry-run python outside shared-env root" fail lint "$BAD7_SBATCH" "$BAD7_MANIFEST"
GOOD7_MANIFEST="$BASE/stage/good7.manifest"
GOOD7_SBATCH="$BASE/stage/good7.sbatch"
write_manifest "$GOOD7_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 1 "$GOOD_PYTHON"
write_sbatch "$GOOD7_SBATCH" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 1 "$GOOD_PYTHON"
expect "dry-run python under shared-env root" pass lint "$GOOD7_SBATCH" "$GOOD7_MANIFEST"

# --- Check 8: ssh in the submission helper needs -o BatchMode=yes. ---
BAD8_HELPER="$BASE/stage/bad8-helper.sh"
printf '#!/bin/bash\nssh cannon.example "sbatch x"\n' > "$BAD8_HELPER"
expect "ssh without BatchMode=yes" fail lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$BAD8_HELPER"
GOOD8_HELPER="$BASE/stage/good8-helper.sh"
printf '#!/bin/bash\nssh -o BatchMode=yes cannon.example "sbatch x"\n' > "$GOOD8_HELPER"
expect "ssh with BatchMode=yes" pass lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$GOOD8_HELPER"

# --- Check 9: real-vs-dry export delta beyond documented dry-only keys. ---
REAL_ENV="$BASE/stage/real.env"
cat > "$REAL_ENV" <<EOF
HEAD_SHA=$SHA
DRY_RUN=0
TASK_ROOT=$GOOD_TASK_ROOT
EOF
DRY_ENV_GOOD="$BASE/stage/dry-good.env"
cat > "$DRY_ENV_GOOD" <<EOF
HEAD_SHA=$SHA
DRY_RUN=1
TASK_ROOT=$GOOD_TASK_ROOT
EOF
expect "real-vs-dry delta limited to documented keys" pass lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --real-env "$REAL_ENV" --dry-env "$DRY_ENV_GOOD" --dry-only-keys "DRY_RUN"
DRY_ENV_BAD="$BASE/stage/dry-bad.env"
cat > "$DRY_ENV_BAD" <<EOF
HEAD_SHA=$SHA
DRY_RUN=1
TASK_ROOT=$GOOD_TASK_ROOT-different
EOF
expect "real-vs-dry delta on undocumented key" fail lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --real-env "$REAL_ENV" --dry-env "$DRY_ENV_BAD" --dry-only-keys "DRY_RUN"

printf 'SELFTEST pass=%s fail=%s\n' "$PASS" "$FAIL"
[[ "$FAIL" == 0 ]]
