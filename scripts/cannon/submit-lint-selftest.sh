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

# Like expect(), but also asserts the output contains a specific substring,
# so a test can't pass for the wrong reason (e.g. the wrong check rejecting).
expect_contains() {
  local label="$1" want="$2" needle="$3"; shift 3
  local out rc=0
  out="$("$@" 2>&1)" || rc=$?
  if { [[ "$want" == pass && "$rc" == 0 ]] || [[ "$want" == fail && "$rc" != 0 ]]; } && [[ "$out" == *"$needle"* ]]; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    printf 'SELFTEST FAILED: %s (expected %s containing %s, rc=%s)\n%s\n' "$label" "$want" "$needle" "$rc" "$out" >&2
  fi
}

lint() { "$LINTER" --sbatch "$1" --manifest "$2" --allowed-root "$BASE/stage" --shared-env-prefix "$BASE/shared-env" --home "$BASE/home" "${@:3}"; }

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
# Run with cwd=BASE (not wherever this selftest happens to be invoked from),
# so the relative path actually resolves to a real file and the rejection is
# the absolute-path check itself, not an earlier "file not found" exit.
run_relative_sbatch() (
  cd "$BASE" && "$LINTER" --sbatch "stage/good.sbatch" --manifest "$GOOD_MANIFEST" --allowed-root "$BASE/stage" --shared-env-prefix "$BASE/shared-env" --home "$BASE/home"
)
expect_contains "relative sbatch path (cwd=BASE)" fail "not absolute" run_relative_sbatch
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
expect "literal \$HOME staging" fail "$LINTER" --sbatch "$BAD5_SBATCH" --manifest "$BAD5_MANIFEST" --allowed-root "$BASE/stage" --allowed-root "$BASE/home" --shared-env-prefix "$BASE/shared-env" --home "$BASE/home"
mkdir -p "$BASE/home/real-staged"
ln -s "$BASE/home/real-staged" "$BASE/stage/home-symlink"
SYMLINK_TASK="$BASE/stage/home-symlink/sub"
BAD5B_MANIFEST="$BASE/stage/bad5b.manifest"
BAD5B_SBATCH="$BASE/stage/bad5b.sbatch"
write_manifest "$BAD5B_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$SYMLINK_TASK" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
write_sbatch "$BAD5B_SBATCH" "$SHA" "tpen-verify-$SHORT" "$SYMLINK_TASK" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
expect "resolved \$HOME staging via symlink" fail "$LINTER" --sbatch "$BAD5B_SBATCH" --manifest "$BAD5B_MANIFEST" --allowed-root "$BASE/stage" --allowed-root "$BASE/home" --shared-env-prefix "$BASE/shared-env" --home "$BASE/home"

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

# --- Check 9: a real-only optional value equal to the absence placeholder
# must not read as "matches the other side's absence". ---
REAL9_ENV="$BASE/stage/real9.env"
cat > "$REAL9_ENV" <<EOF
HEAD_SHA=$SHA
DRY_RUN=0
OPTIONAL=__unset__
EOF
DRY9_ENV="$BASE/stage/dry9.env"
cat > "$DRY9_ENV" <<EOF
HEAD_SHA=$SHA
DRY_RUN=1
EOF
expect_contains "OPTIONAL=__unset__ vs absent is still a delta" fail "present on only one side" lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --real-env "$REAL9_ENV" --dry-env "$DRY9_ENV" --dry-only-keys "DRY_RUN"
REAL9B_ENV="$BASE/stage/real9b.env"
cat > "$REAL9B_ENV" <<EOF
HEAD_SHA=$SHA
DRY_RUN=0
OPTIONAL=
EOF
DRY9B_ENV="$BASE/stage/dry9b.env"
cat > "$DRY9B_ENV" <<EOF
HEAD_SHA=$SHA
DRY_RUN=1
OPTIONAL=
EOF
expect "empty optional value present on both sides still passes" pass lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --real-env "$REAL9B_ENV" --dry-env "$DRY9B_ENV" --dry-only-keys "DRY_RUN"

# --- Check 4b: an always-required critical field absent from BOTH files
# must fail (check 4 alone has nothing to compare and would pass it). ---
MIN_MANIFEST="$BASE/stage/min.manifest"
MIN_SBATCH="$BASE/stage/min.sbatch"
cat > "$MIN_MANIFEST" <<EOF
HEAD_SHA=$SHA
JOB_NAME=tpen-verify-$SHORT
EOF
cat > "$MIN_SBATCH" <<EOF
#!/bin/bash
export HEAD_SHA=$SHA
export JOB_NAME=tpen-verify-$SHORT
EOF
expect_contains "TASK_ROOT missing from both files is rejected" fail "required critical field 'TASK_ROOT'" lint "$MIN_SBATCH" "$MIN_MANIFEST"

MIN_DRY_MANIFEST="$BASE/stage/min-dry.manifest"
MIN_DRY_SBATCH="$BASE/stage/min-dry.sbatch"
write_manifest "$MIN_DRY_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 1 ""
cat > "$MIN_DRY_SBATCH" <<EOF
#!/bin/bash
export HEAD_SHA=$SHA
export JOB_NAME=tpen-verify-$SHORT
export TASK_ROOT=$GOOD_TASK_ROOT
export RUN_ROOT=$GOOD_RUN_ROOT
export DRY_RUN=1
EOF
expect_contains "PYTHON missing from both files under DRY_RUN=1 is rejected" fail "required critical field 'PYTHON'" lint "$MIN_DRY_SBATCH" "$MIN_DRY_MANIFEST"

# --- Check 8: ssh BatchMode must be checked per logical invocation, not per
# physical line or whole-file substring (the 4 reviewer-reported cases). ---
COMMENT_TRICK_HELPER="$BASE/stage/helper-comment-trick.sh"
printf '#!/bin/bash\nssh cannon.example true # -o BatchMode=yes\n' > "$COMMENT_TRICK_HELPER"
expect "ssh BatchMode mentioned only in a trailing comment is still rejected" fail lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$COMMENT_TRICK_HELPER"

SECOND_CMD_HELPER="$BASE/stage/helper-second-cmd.sh"
printf '#!/bin/bash\nssh -o BatchMode=yes a true; ssh b true\n' > "$SECOND_CMD_HELPER"
expect "second ssh command on the same line without BatchMode is rejected" fail lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$SECOND_CMD_HELPER"

COMMENT_MENTION_HELPER="$BASE/stage/helper-comment-mention.sh"
printf '#!/bin/bash\n# ssh invocations below always carry -o BatchMode=yes\ntrue\n' > "$COMMENT_MENTION_HELPER"
expect "a comment merely mentioning ssh is not flagged" pass lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$COMMENT_MENTION_HELPER"

CONTINUATION_HELPER="$BASE/stage/helper-continuation.sh"
printf '#!/bin/bash\nssh \\\n  -o BatchMode=yes cannon.example "sbatch x"\n' > "$CONTINUATION_HELPER"
expect "ssh continued onto the next line with BatchMode is accepted" pass lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$CONTINUATION_HELPER"

# --- ssh/scp/rsync must be caught as a command WORD anywhere in the
# segment, not only when the segment literally starts with it; words that
# merely contain "ssh"/"rsync" as a substring (sshd_config, rsync_log) must
# NOT be mistaken for the command. ---
WRAPPED1_HELPER="$BASE/stage/helper-wrapped1.sh"
printf '#!/bin/bash\ntimeout 30 ssh host true\n' > "$WRAPPED1_HELPER"
expect "'timeout 30 ssh ...' without BatchMode is rejected" fail lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$WRAPPED1_HELPER"

WRAPPED2_HELPER="$BASE/stage/helper-wrapped2.sh"
printf '#!/bin/bash\nenv X=1 ssh host\n' > "$WRAPPED2_HELPER"
expect "'env X=1 ssh ...' without BatchMode is rejected" fail lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$WRAPPED2_HELPER"

WRAPPED3_HELPER="$BASE/stage/helper-wrapped3.sh"
printf '#!/bin/bash\ncommand ssh host\n' > "$WRAPPED3_HELPER"
expect "'command ssh ...' without BatchMode is rejected" fail lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$WRAPPED3_HELPER"

RSYNC_HELPER="$BASE/stage/helper-rsync.sh"
printf '#!/bin/bash\nrsync -a -e ssh src host:dst\n' > "$RSYNC_HELPER"
expect "'rsync -e ssh ...' without BatchMode is rejected" fail lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$RSYNC_HELPER"

SUBSHELL_HELPER="$BASE/stage/helper-subshell.sh"
printf '#!/bin/bash\nx=$(ssh host true)\n' > "$SUBSHELL_HELPER"
expect "'x=\$(ssh ...)' without BatchMode is rejected" fail lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$SUBSHELL_HELPER"

SCP_HELPER="$BASE/stage/helper-scp.sh"
printf '#!/bin/bash\nscp host:a b\n' > "$SCP_HELPER"
expect "scp without BatchMode is rejected" fail lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$SCP_HELPER"

NO_MATCH_HELPER="$BASE/stage/helper-no-match.sh"
printf '#!/bin/bash\ncat /etc/sshd_config\nrotate rsync_log\n' > "$NO_MATCH_HELPER"
expect "sshd_config/rsync_log substrings are not mistaken for the commands" pass lint "$GOOD_SBATCH" "$GOOD_MANIFEST" --submit-helper "$NO_MATCH_HELPER"

# --- --real-env/--dry-env must be given together or not at all. ---
expect_contains "orphan --real-env (no --dry-env) is LINT_FAIL" fail "must be given together" "$LINTER" --sbatch "$GOOD_SBATCH" --manifest "$GOOD_MANIFEST" --allowed-root "$BASE/stage" --shared-env-prefix "$BASE/shared-env" --home "$BASE/home" --real-env "$REAL_ENV"
expect_contains "orphan --dry-env (no --real-env) is LINT_FAIL" fail "must be given together" "$LINTER" --sbatch "$GOOD_SBATCH" --manifest "$GOOD_MANIFEST" --allowed-root "$BASE/stage" --shared-env-prefix "$BASE/shared-env" --home "$BASE/home" --dry-env "$DRY_ENV_GOOD"

# --- realpath -m support is probed up front with a clear LINT_FAIL. ---
# This machine's normal PATH already carries GNU coreutils' realpath (the
# selftest and pytest contract both run under it); to also demonstrate the
# failure a stock macOS PATH (BSD realpath, no -m) would hit, shim a
# BSD-realpath-shaped binary named 'realpath' ahead of PATH and confirm the
# linter refuses it with the GNU-coreutils message instead of misbehaving.
ORIG_REALPATH="$(command -v realpath)"
BSD_REALPATH_SHIM_DIR="$BASE/bsd-realpath-shim"
mkdir -p "$BSD_REALPATH_SHIM_DIR"
cat > "$BSD_REALPATH_SHIM_DIR/realpath" <<SHIM
#!/bin/bash
# Mimics BSD/macOS realpath: no -m support, same error shape as the real one.
for a in "\$@"; do
  if [[ "\$a" == "-m" ]]; then
    echo "realpath: illegal option -- m" >&2
    echo "usage: realpath [-q] path ..." >&2
    exit 1
  fi
done
exec "$ORIG_REALPATH" "\$@"
SHIM
chmod +x "$BSD_REALPATH_SHIM_DIR/realpath"
run_under_bsd_realpath_shim() (
  PATH="$BSD_REALPATH_SHIM_DIR:$PATH"
  lint "$GOOD_SBATCH" "$GOOD_MANIFEST"
)
expect_contains "stock-macOS-shaped realpath without -m fails clearly" fail "GNU coreutils realpath is required" run_under_bsd_realpath_shim

# --- Check 4 (ef480b65 linter-trial-log false blocks): critical values
# passed via `sbatch --export=` and only bare-re-exported in the body
# (`export A B C`, no '=') must count as present, from the --export list. ---
REEXPORT_MANIFEST="$BASE/stage/reexport.manifest"
REEXPORT_SBATCH="$BASE/stage/reexport.sbatch"
write_manifest "$REEXPORT_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
cat > "$REEXPORT_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=tpen-verify-$SHORT
#SBATCH --export=HEAD_SHA=$SHA,JOB_NAME=tpen-verify-$SHORT,TASK_ROOT=$GOOD_TASK_ROOT,RUN_ROOT=$GOOD_RUN_ROOT,DRY_RUN=0,PYTHON=$GOOD_PYTHON
export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT DRY_RUN PYTHON
EOF
expect "bare re-export after sbatch --export= is accepted" pass lint "$REEXPORT_SBATCH" "$REEXPORT_MANIFEST"

# --- A multi-assignment export line must parse each key independently: one
# key's value must not swallow a sibling assignment on the same line. ---
MULTI_SBATCH="$BASE/stage/multiassign.sbatch"
cat > "$MULTI_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=tpen-verify-$SHORT
#SBATCH --export=HEAD_SHA=$SHA,JOB_NAME=tpen-verify-$SHORT,TASK_ROOT=$GOOD_TASK_ROOT,RUN_ROOT=$GOOD_RUN_ROOT,DRY_RUN=0,PYTHON=$GOOD_PYTHON
export DRY_RUN="\${DRY_RUN:-0}" DRY_NODE_TMP="\${DRY_NODE_TMP:-}"
export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT PYTHON
EOF
expect "multi-assignment export line parsed per key" pass lint "$MULTI_SBATCH" "$REEXPORT_MANIFEST"

# --- --staging-root LOCAL=FINAL: a sbatch script linted in place at a local
# staging root is validated for containment at its mapped final root. ---
mkdir -p "$BASE/local-stage" "$BASE/final-stage/task" "$BASE/final-stage/run"
STAGING_MANIFEST="$BASE/local-stage/staged.manifest"
STAGING_SBATCH="$BASE/local-stage/staged.sbatch"
write_manifest "$STAGING_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$BASE/final-stage/task" "$BASE/final-stage/run" 0 "$GOOD_PYTHON"
write_sbatch "$STAGING_SBATCH" "$SHA" "tpen-verify-$SHORT" "$BASE/final-stage/task" "$BASE/final-stage/run" 0 "$GOOD_PYTHON"
expect "sbatch linted at a mapped local staging root is accepted" pass \
  "$LINTER" --sbatch "$STAGING_SBATCH" --manifest "$STAGING_MANIFEST" \
  --allowed-root "$BASE/final-stage" --shared-env-prefix "$BASE/shared-env" --home "$BASE/home" \
  --staging-root "$BASE/local-stage=$BASE/final-stage"

# --- Round 2 (T1 review of PR #653). Each job below starts from a
# `#SBATCH --export=` directive supplying all six values with DRY_RUN=0 and
# is linted against REEXPORT_MANIFEST (DRY_RUN=0); only the body differs. ---
R2_SPEC="HEAD_SHA=$SHA,JOB_NAME=tpen-verify-$SHORT,TASK_ROOT=$GOOD_TASK_ROOT,RUN_ROOT=$GOOD_RUN_ROOT,DRY_RUN=0,PYTHON=$GOOD_PYTHON"
R2_REST="export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT PYTHON"
r2_job() {  # NAME BODY: writes $BASE/stage/r2-NAME.sbatch
  printf '#!/bin/bash\n#SBATCH --export=%s\n%s\n%s\n' "$R2_SPEC" "$2" "$R2_REST" > "$BASE/stage/r2-$1.sbatch"
}
STALE_DRY="critical export 'DRY_RUN' is stale: manifest='0' sbatch='1'"

# Finding 1: re-exports and self-defaults resolve against the CURRENT value.
r2_job bare $'export DRY_RUN=1\nexport DRY_RUN'
expect_contains "bare re-export keeps the current value, not the directive's" fail "$STALE_DRY" lint "$BASE/stage/r2-bare.sbatch" "$REEXPORT_MANIFEST"
r2_job default $'export DRY_RUN=1\nexport DRY_RUN="${DRY_RUN:-0}"'
expect_contains "self-default uses the current value, not the directive's" fail "$STALE_DRY" lint "$BASE/stage/r2-default.sbatch" "$REEXPORT_MANIFEST"
r2_job plain $'export DRY_RUN\nDRY_RUN=1'
expect_contains "plain assignment after export updates the value" fail "$STALE_DRY" lint "$BASE/stage/r2-plain.sbatch" "$REEXPORT_MANIFEST"
r2_job prefix $'DRY_RUN=1 python run.py | tee log\nexport DRY_RUN'
expect "command-prefix assignment leaves the job environment alone" pass lint "$BASE/stage/r2-prefix.sbatch" "$REEXPORT_MANIFEST"

# Finding 2: quote provenance and comments.
r2_job squote "export DRY_RUN='\${DRY_RUN:-0}' NOTE='x y=z'"
expect_contains "single-quoted value is never expanded" fail "sbatch='\${DRY_RUN:-0}'" lint "$BASE/stage/r2-squote.sbatch" "$REEXPORT_MANIFEST"
r2_job comment "export DRY_RUN=1 # DRY_RUN=0"
expect_contains "unquoted comment ends the export statement" fail "$STALE_DRY" lint "$BASE/stage/r2-comment.sbatch" "$REEXPORT_MANIFEST"
r2_job literal "export NOTE='x y=z' OTHER=\"a b=c\" DRY_RUN=0"
expect "literal quoted spaces and '=' are accepted" pass lint "$BASE/stage/r2-literal.sbatch" "$REEXPORT_MANIFEST"
r2_job cmdsub 'export DRY_RUN="$(echo 0)"'
expect_contains "command substitution in a critical value is rejected" fail "will not guess its value" lint "$BASE/stage/r2-cmdsub.sbatch" "$REEXPORT_MANIFEST"
r2_job cmdsubok $'export PATH="$(pwd)/bin:$PATH"\nexport DRY_RUN'
expect "command substitution in a non-critical value is tolerated" pass lint "$BASE/stage/r2-cmdsubok.sbatch" "$REEXPORT_MANIFEST"

# Finding 3: the WHOLE mapped destination is canonicalized (FINAL/link -> HOME).
mkdir -p "$BASE/r2-local/link" "$BASE/r2-final"
ln -s "$BASE/home" "$BASE/r2-final/link"
r2_staged() {  # SBATCH TASK_ROOT RUN_ROOT
  write_manifest "$1.manifest" "$SHA" "tpen-verify-$SHORT" "$2" "$3" 0 "$GOOD_PYTHON"
  write_sbatch "$1" "$SHA" "tpen-verify-$SHORT" "$2" "$3" 0 "$GOOD_PYTHON"
  "$LINTER" --sbatch "$1" --manifest "$1.manifest" --allowed-root "$BASE/r2-final" \
    --shared-env-prefix "$BASE/shared-env" --home "$BASE/home" --staging-root "$BASE/r2-local=$BASE/r2-final"
}
expect_contains "mapped TASK_ROOT through FINAL/link -> HOME is rejected" fail "TASK_ROOT resolves under" r2_staged "$BASE/r2-local/t.sbatch" "$BASE/r2-local/link/x" "$BASE/r2-final/run"
expect_contains "mapped RUN_ROOT through FINAL/link -> HOME is rejected" fail "RUN_ROOT resolves under" r2_staged "$BASE/r2-local/r.sbatch" "$BASE/r2-final/task" "$BASE/r2-local/link/x"
expect_contains "mapped sbatch path through FINAL/link -> HOME is rejected" fail "sbatch path escapes" r2_staged "$BASE/r2-local/link/s.sbatch" "$BASE/r2-final/task" "$BASE/r2-final/run"

# Finding 4: the submit helper's own `sbatch --export=... <job>` line.
CLI_SBATCH="$BASE/stage/cli-job.sbatch"
printf '#!/bin/bash\n#SBATCH --job-name=tpen-verify-%s\nexport HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT DRY_RUN PYTHON\n' "$SHORT" > "$CLI_SBATCH"
CLI_HELPER="$BASE/stage/cli-submit.sh"
printf '#!/bin/bash\njobid=$(sbatch --parsable --export=%s %s)\n' "$R2_SPEC" "$CLI_SBATCH" > "$CLI_HELPER"
expect "helper sbatch --export= line supplies bare re-exports" pass lint "$CLI_SBATCH" "$REEXPORT_MANIFEST" --submit-helper "$CLI_HELPER"
CLI_HELPER_BAD="$BASE/stage/cli-submit-bad.sh"
printf '#!/bin/bash\njobid=$(sbatch --parsable --export=%s %s)\n' "${R2_SPEC/DRY_RUN=0/DRY_RUN=1}" "$CLI_SBATCH" > "$CLI_HELPER_BAD"
expect_contains "helper sbatch --export= line with a wrong value is rejected" fail "$STALE_DRY" lint "$CLI_SBATCH" "$REEXPORT_MANIFEST" --submit-helper "$CLI_HELPER_BAD"
CLI_HELPER_EXP="$BASE/stage/cli-submit-exp.sh"
printf '#!/bin/bash\nsbatch --export="HEAD_SHA=$SHA,DRY_RUN=0" %s\n' "$CLI_SBATCH" > "$CLI_HELPER_EXP"
expect_contains "helper --export with shell expansion is rejected" fail "unsupported --export value" lint "$CLI_SBATCH" "$REEXPORT_MANIFEST" --submit-helper "$CLI_HELPER_EXP"

# --shared-env-prefix must be disjoint from --allowed-root / --staging-root
# FINAL (the prefix split).
NESTED_SHARED="$BASE/stage/shared-env-nested"
mkdir -p "$NESTED_SHARED/bin"
touch "$NESTED_SHARED/bin/python"; chmod +x "$NESTED_SHARED/bin/python"
expect_contains "--shared-env-prefix nested inside --allowed-root is rejected" fail "disjoint" \
  "$LINTER" --sbatch "$GOOD_SBATCH" --manifest "$GOOD_MANIFEST" --allowed-root "$BASE/stage" \
  --shared-env-prefix "$NESTED_SHARED" --home "$BASE/home"
expect_contains "--shared-env-prefix containing --allowed-root is rejected" fail "disjoint" \
  "$LINTER" --sbatch "$GOOD_SBATCH" --manifest "$GOOD_MANIFEST" --allowed-root "$NESTED_SHARED/bin" \
  --shared-env-prefix "$NESTED_SHARED" --home "$BASE/home"
mkdir -p "$BASE/staging-final-shared/bin"
expect_contains "--shared-env-prefix nested inside a --staging-root FINAL is rejected" fail "disjoint" \
  "$LINTER" --sbatch "$STAGING_SBATCH" --manifest "$STAGING_MANIFEST" --allowed-root "$BASE/final-stage" \
  --shared-env-prefix "$BASE/final-stage/shared-sub" --home "$BASE/home" \
  --staging-root "$BASE/local-stage=$BASE/final-stage"

# Check 10: manifest SHARED_ENV_PREFIX/SHARED_ENV_PATH keys must stay
# consistent with --shared-env-prefix and never resolve under an allowed root.
# Since SHARED_ENV_PREFIX is now a critical key (Check 4, T1/F1), its own
# sbatch fixture must export a matching value too.
SENV10_MANIFEST="$BASE/stage/senv10.manifest"
SENV10_SBATCH="$BASE/stage/senv10.sbatch"
cat > "$SENV10_MANIFEST" <<EOF
HEAD_SHA=$SHA
JOB_NAME=tpen-verify-$SHORT
TASK_ROOT=$GOOD_TASK_ROOT
RUN_ROOT=$GOOD_RUN_ROOT
DRY_RUN=0
PYTHON=$GOOD_PYTHON
SHARED_ENV_PREFIX=$BASE/shared-env
EOF
cat > "$SENV10_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=tpen-verify-$SHORT
export HEAD_SHA=$SHA
export JOB_NAME=tpen-verify-$SHORT
export TASK_ROOT=$GOOD_TASK_ROOT
export RUN_ROOT=$GOOD_RUN_ROOT
export DRY_RUN=0
export PYTHON=$GOOD_PYTHON
export SHARED_ENV_PREFIX=$BASE/shared-env
EOF
expect "manifest SHARED_ENV_PREFIX matching the CLI value is accepted" pass lint "$SENV10_SBATCH" "$SENV10_MANIFEST"
SENV10B_MANIFEST="$BASE/stage/senv10b.manifest"
SENV10B_SBATCH="$BASE/stage/senv10b.sbatch"
cat > "$SENV10B_MANIFEST" <<EOF
HEAD_SHA=$SHA
JOB_NAME=tpen-verify-$SHORT
TASK_ROOT=$GOOD_TASK_ROOT
RUN_ROOT=$GOOD_RUN_ROOT
DRY_RUN=0
PYTHON=$GOOD_PYTHON
SHARED_ENV_PREFIX=$BASE/other-root
EOF
cat > "$SENV10B_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=tpen-verify-$SHORT
export HEAD_SHA=$SHA
export JOB_NAME=tpen-verify-$SHORT
export TASK_ROOT=$GOOD_TASK_ROOT
export RUN_ROOT=$GOOD_RUN_ROOT
export DRY_RUN=0
export PYTHON=$GOOD_PYTHON
export SHARED_ENV_PREFIX=$BASE/other-root
EOF
expect_contains "manifest SHARED_ENV_PREFIX not matching the CLI value is rejected" fail "does not match" lint "$SENV10B_SBATCH" "$SENV10B_MANIFEST"
SENV10C_MANIFEST="$BASE/stage/senv10c.manifest"
cat > "$SENV10C_MANIFEST" <<EOF
HEAD_SHA=$SHA
JOB_NAME=tpen-verify-$SHORT
TASK_ROOT=$GOOD_TASK_ROOT
RUN_ROOT=$GOOD_RUN_ROOT
DRY_RUN=0
PYTHON=$GOOD_PYTHON
SHARED_ENV_PATH=$BASE/shared-env/tpen-abc123-cpu-py3.12
EOF
expect "manifest SHARED_ENV_PATH as a direct child of the CLI prefix is accepted" pass lint "$GOOD_SBATCH" "$SENV10C_MANIFEST"
SENV10D_MANIFEST="$BASE/stage/senv10d.manifest"
cat > "$SENV10D_MANIFEST" <<EOF
HEAD_SHA=$SHA
JOB_NAME=tpen-verify-$SHORT
TASK_ROOT=$GOOD_TASK_ROOT
RUN_ROOT=$GOOD_RUN_ROOT
DRY_RUN=0
PYTHON=$GOOD_PYTHON
SHARED_ENV_PATH=$BASE/shared-env/sub/tpen-abc123-cpu-py3.12
EOF
expect_contains "manifest SHARED_ENV_PATH two levels deep is rejected" fail "not a direct child" lint "$GOOD_SBATCH" "$SENV10D_MANIFEST"

# Check 11: TASK_ROOT/RUN_ROOT must not resolve under --shared-env-prefix.
SENV11_MANIFEST="$BASE/stage/senv11.manifest"
SENV11_SBATCH="$BASE/stage/senv11.sbatch"
SENV11_TASK="$BASE/shared-env/task"
write_manifest "$SENV11_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$SENV11_TASK" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
write_sbatch "$SENV11_SBATCH" "$SHA" "tpen-verify-$SHORT" "$SENV11_TASK" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
expect_contains "TASK_ROOT under --shared-env-prefix is rejected" fail "resolves under --shared-env-prefix" \
  "$LINTER" --sbatch "$SENV11_SBATCH" --manifest "$SENV11_MANIFEST" --allowed-root "$BASE/stage" --allowed-root "$BASE/shared-env" \
  --shared-env-prefix "$BASE/shared-env" --home "$BASE/home"

# DT2 (T2 delta review): Check 11's staging-root mapping must also catch a
# symlink escape into --shared-env-prefix, not only the $HOME case the
# finding-3 r2-staged tests above already cover; a disjoint mapped
# counterpart through the same staging machinery still passes.
mkdir -p "$BASE/senv-local/link" "$BASE/senv-final"
ln -s "$BASE/shared-env" "$BASE/senv-final/link"
senv_staged() {  # SBATCH TASK_ROOT RUN_ROOT
  write_manifest "$1.manifest" "$SHA" "tpen-verify-$SHORT" "$2" "$3" 0 "$GOOD_PYTHON"
  write_sbatch "$1" "$SHA" "tpen-verify-$SHORT" "$2" "$3" 0 "$GOOD_PYTHON"
  "$LINTER" --sbatch "$1" --manifest "$1.manifest" --allowed-root "$BASE/senv-final" \
    --shared-env-prefix "$BASE/shared-env" --home "$BASE/home" --staging-root "$BASE/senv-local=$BASE/senv-final"
}
expect_contains "mapped TASK_ROOT through FINAL/link -> shared-env-prefix is rejected" fail "TASK_ROOT resolves under --shared-env-prefix" \
  senv_staged "$BASE/senv-local/t.sbatch" "$BASE/senv-local/link/x" "$BASE/senv-final/run"
expect_contains "mapped RUN_ROOT through FINAL/link -> shared-env-prefix is rejected" fail "RUN_ROOT resolves under --shared-env-prefix" \
  senv_staged "$BASE/senv-local/r.sbatch" "$BASE/senv-final/task" "$BASE/senv-local/link/x"
mkdir -p "$BASE/senv-local/elsewhere" "$BASE/senv-final/elsewhere"
expect "mapped root disjoint from --shared-env-prefix (same staging machinery) is accepted" pass \
  senv_staged "$BASE/senv-local/d.sbatch" "$BASE/senv-local/elsewhere/task" "$BASE/senv-final/run"

# DT2's real `shared-env-precheck.sh resolve` producer/consumer control lives
# in tests/test_cannon_submit_lint.py only, not here: shared-env-precheck.sh
# itself requires bash >= 4.4 (`inherit_errexit`) and refuses to run at all
# under bash 3.2, and this selftest script must keep running top-to-bottom
# under bash 3.2 (a mid-script PRECHECK_FAILED would abort every later
# case). pytest isolates one bash4-only test's failure from the rest of the
# suite the way this single `set -euo pipefail` script cannot.

# Check 7 (T3, review-round-1 F3): a dry-run PYTHON under an allowed root but
# NOT under --shared-env-prefix (e.g. a previous job's overlay) must be
# rejected -- the allowed-root exception is removed; only the shared-env
# prefix itself identifies a trusted interpreter.
OVERLAY_PYTHON="$BASE/stage/run/overlay/bin/python"
mkdir -p "$(dirname -- "$OVERLAY_PYTHON")"; touch "$OVERLAY_PYTHON"; chmod +x "$OVERLAY_PYTHON"
BAD7B_MANIFEST="$BASE/stage/bad7b.manifest"
BAD7B_SBATCH="$BASE/stage/bad7b.sbatch"
write_manifest "$BAD7B_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 1 "$OVERLAY_PYTHON"
write_sbatch "$BAD7B_SBATCH" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 1 "$OVERLAY_PYTHON"
expect_contains "dry-run python under an allowed root but outside --shared-env-prefix is rejected" fail "is not under --shared-env-prefix" lint "$BAD7B_SBATCH" "$BAD7B_MANIFEST"

# Check 4 (T1, review-round-1 F1): job-exported SHARED_ENV_PREFIX must
# reconcile with the manifest's (and transitively, via Check 10, the CLI
# prefix) -- stale, missing, or extra values are rejected like any other
# critical export.
SENV_PREFIX_MANIFEST="$BASE/stage/senv-prefix-match.manifest"
SENV_PREFIX_SBATCH_GOOD="$BASE/stage/senv-prefix-match-good.sbatch"
SENV_PREFIX_SBATCH_STALE="$BASE/stage/senv-prefix-match-stale.sbatch"
SENV_PREFIX_SBATCH_MISSING="$BASE/stage/senv-prefix-match-missing.sbatch"
cat > "$SENV_PREFIX_MANIFEST" <<EOF
HEAD_SHA=$SHA
JOB_NAME=tpen-verify-$SHORT
TASK_ROOT=$GOOD_TASK_ROOT
RUN_ROOT=$GOOD_RUN_ROOT
DRY_RUN=0
PYTHON=$GOOD_PYTHON
SHARED_ENV_PREFIX=$BASE/shared-env
EOF
cat > "$SENV_PREFIX_SBATCH_GOOD" <<EOF
#!/bin/bash
#SBATCH --job-name=tpen-verify-$SHORT
export HEAD_SHA=$SHA
export JOB_NAME=tpen-verify-$SHORT
export TASK_ROOT=$GOOD_TASK_ROOT
export RUN_ROOT=$GOOD_RUN_ROOT
export DRY_RUN=0
export PYTHON=$GOOD_PYTHON
export SHARED_ENV_PREFIX=$BASE/shared-env
EOF
expect "job-exported SHARED_ENV_PREFIX matching the manifest/CLI is accepted" pass lint "$SENV_PREFIX_SBATCH_GOOD" "$SENV_PREFIX_MANIFEST"
cat > "$SENV_PREFIX_SBATCH_STALE" <<EOF
#!/bin/bash
#SBATCH --job-name=tpen-verify-$SHORT
export HEAD_SHA=$SHA
export JOB_NAME=tpen-verify-$SHORT
export TASK_ROOT=$GOOD_TASK_ROOT
export RUN_ROOT=$GOOD_RUN_ROOT
export DRY_RUN=0
export PYTHON=$GOOD_PYTHON
export SHARED_ENV_PREFIX=$BASE/other-root
EOF
expect_contains "job-exported SHARED_ENV_PREFIX disagreeing with the manifest is rejected" fail "critical export 'SHARED_ENV_PREFIX' is stale" lint "$SENV_PREFIX_SBATCH_STALE" "$SENV_PREFIX_MANIFEST"
cat > "$SENV_PREFIX_SBATCH_MISSING" <<EOF
#!/bin/bash
#SBATCH --job-name=tpen-verify-$SHORT
export HEAD_SHA=$SHA
export JOB_NAME=tpen-verify-$SHORT
export TASK_ROOT=$GOOD_TASK_ROOT
export RUN_ROOT=$GOOD_RUN_ROOT
export DRY_RUN=0
export PYTHON=$GOOD_PYTHON
EOF
expect_contains "manifest SHARED_ENV_PREFIX missing from the job export is rejected" fail "critical export 'SHARED_ENV_PREFIX' missing from sbatch script" lint "$SENV_PREFIX_SBATCH_MISSING" "$SENV_PREFIX_MANIFEST"
expect_contains "job-exported SHARED_ENV_PREFIX with no manifest counterpart is rejected" fail "critical export 'SHARED_ENV_PREFIX' present in sbatch script but not in manifest" lint "$SENV_PREFIX_SBATCH_GOOD" "$GOOD_MANIFEST"

# DT1 (T2 delta review DF1): a bare `export SHARED_ENV_PREFIX` with no prior
# assignment leaves it exported with an undetermined value ("unset"), which
# must not read as "absent from both sides" the way a truly unmentioned name
# does (the "good fixture" baseline above is that both-omit green control).
DF1_MANIFEST="$BASE/stage/df1-bare.manifest"
DF1_SBATCH="$BASE/stage/df1-bare.sbatch"
write_manifest "$DF1_MANIFEST" "$SHA" "tpen-verify-$SHORT" "$GOOD_TASK_ROOT" "$GOOD_RUN_ROOT" 0 "$GOOD_PYTHON"
cat > "$DF1_SBATCH" <<EOF
#!/bin/bash
#SBATCH --export=HEAD_SHA=$SHA,JOB_NAME=tpen-verify-$SHORT,TASK_ROOT=$GOOD_TASK_ROOT,RUN_ROOT=$GOOD_RUN_ROOT,DRY_RUN=0,PYTHON=$GOOD_PYTHON
export SHARED_ENV_PREFIX
export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT PYTHON
EOF
expect_contains "bare-exported SHARED_ENV_PREFIX with no manifest counterpart is rejected" fail \
  "critical export 'SHARED_ENV_PREFIX' is exported with no determined value" lint "$DF1_SBATCH" "$DF1_MANIFEST"

# The linter is static/syntax-only; an inherited, mismatching real value in
# the linter's OWN process environment must not change that verdict.
DF1_ENV_OUT="$BASE/df1-env.out"
DF1_ENV_RC=0
SHARED_ENV_PREFIX="$BASE/other-root" "$LINTER" --sbatch "$DF1_SBATCH" --manifest "$DF1_MANIFEST" \
  --allowed-root "$BASE/stage" --shared-env-prefix "$BASE/shared-env" --home "$BASE/home" \
  >"$DF1_ENV_OUT" 2>&1 || DF1_ENV_RC=$?
if [[ "$DF1_ENV_RC" != 0 ]] && grep -q "critical export 'SHARED_ENV_PREFIX' is exported with no determined value" "$DF1_ENV_OUT"; then
  PASS=$((PASS + 1))
else
  FAIL=$((FAIL + 1))
  printf 'SELFTEST FAILED: %s (rc=%s)\n%s\n' "inherited real SHARED_ENV_PREFIX in the linter's own environment does not change the DF1 verdict" "$DF1_ENV_RC" "$(cat "$DF1_ENV_OUT")" >&2
fi

# Green control: a literal assignment followed by a bare re-export keeps the
# already-known value (state stays "set", not "unset"), so it still
# reconciles normally against a matching manifest.
DF1_LITBARE_MANIFEST="$BASE/stage/df1-litbare.manifest"
DF1_LITBARE_SBATCH="$BASE/stage/df1-litbare.sbatch"
cat > "$DF1_LITBARE_MANIFEST" <<EOF
HEAD_SHA=$SHA
JOB_NAME=tpen-verify-$SHORT
TASK_ROOT=$GOOD_TASK_ROOT
RUN_ROOT=$GOOD_RUN_ROOT
DRY_RUN=0
PYTHON=$GOOD_PYTHON
SHARED_ENV_PREFIX=$BASE/shared-env
EOF
cat > "$DF1_LITBARE_SBATCH" <<EOF
#!/bin/bash
#SBATCH --job-name=tpen-verify-$SHORT
export HEAD_SHA=$SHA
export JOB_NAME=tpen-verify-$SHORT
export TASK_ROOT=$GOOD_TASK_ROOT
export RUN_ROOT=$GOOD_RUN_ROOT
export DRY_RUN=0
export PYTHON=$GOOD_PYTHON
export SHARED_ENV_PREFIX=$BASE/shared-env
export SHARED_ENV_PREFIX
EOF
expect "literal assignment plus a bare re-export still reconciles (state stays known)" pass lint "$DF1_LITBARE_SBATCH" "$DF1_LITBARE_MANIFEST"

# Already-accepted T1 case, pinned under the SHARED_ENV_PREFIX name: a bare
# NAME in the `#SBATCH --export=` directive itself is "unknown" (sbatch would
# copy it from the submitting shell), not "unset", and already rejected by
# the pre-existing generic unknown-export loop -- confirm DF1's new check
# doesn't change or duplicate that outcome.
DF1_UNKNOWN_SBATCH="$BASE/stage/df1-unknown.sbatch"
cat > "$DF1_UNKNOWN_SBATCH" <<EOF
#!/bin/bash
#SBATCH --export=HEAD_SHA=$SHA,JOB_NAME=tpen-verify-$SHORT,TASK_ROOT=$GOOD_TASK_ROOT,RUN_ROOT=$GOOD_RUN_ROOT,DRY_RUN=0,PYTHON=$GOOD_PYTHON,SHARED_ENV_PREFIX
export HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT PYTHON
EOF
expect_contains "directive-bare SHARED_ENV_PREFIX (unknown, not unset) is already rejected" fail \
  "has a value the linter cannot determine" lint "$DF1_UNKNOWN_SBATCH" "$GOOD_MANIFEST"

# Check 10 (T2, review-round-1 F2): --shared-env-prefix and the manifest's
# mirrored keys must already be spelled canonically (no trailing slash,
# `.`/`..`, relative path, or unresolved symlink) -- comparing only the
# canonicalized values is not enough, since a deterministic submission
# failure can still occur downstream on the raw spelling.
expect_contains "--shared-env-prefix with a trailing slash is rejected" fail "--shared-env-prefix is not canonical" \
  "$LINTER" --sbatch "$GOOD_SBATCH" --manifest "$GOOD_MANIFEST" --allowed-root "$BASE/stage" \
  --shared-env-prefix "$BASE/shared-env/" --home "$BASE/home"
expect_contains "relative --shared-env-prefix is rejected" fail "--shared-env-prefix is not absolute" \
  "$LINTER" --sbatch "$GOOD_SBATCH" --manifest "$GOOD_MANIFEST" --allowed-root "$BASE/stage" \
  --shared-env-prefix "shared-env" --home "$BASE/home"
SENV_SYMLINK="$BASE/shared-env-alias"
ln -s "$BASE/shared-env" "$SENV_SYMLINK"
expect_contains "a symlink alias for --shared-env-prefix is rejected (resolves to a different spelling)" fail "--shared-env-prefix is not canonical" \
  "$LINTER" --sbatch "$GOOD_SBATCH" --manifest "$GOOD_MANIFEST" --allowed-root "$BASE/stage" \
  --shared-env-prefix "$SENV_SYMLINK" --home "$BASE/home"
SENV10E_MANIFEST="$BASE/stage/senv10e.manifest"
cat > "$SENV10E_MANIFEST" <<EOF
HEAD_SHA=$SHA
JOB_NAME=tpen-verify-$SHORT
TASK_ROOT=$GOOD_TASK_ROOT
RUN_ROOT=$GOOD_RUN_ROOT
DRY_RUN=0
PYTHON=$GOOD_PYTHON
SHARED_ENV_PREFIX=$BASE/shared-env/
EOF
expect_contains "manifest SHARED_ENV_PREFIX with a trailing slash is rejected" fail "manifest SHARED_ENV_PREFIX is not canonical" lint "$GOOD_SBATCH" "$SENV10E_MANIFEST"

# Cannon job 59624204: `((i++))` with i=0 aborts under set -e on bash >= 4
# but not on macOS bash 3.2. Statically forbid ((...)) increments and
# assignments outside `for ((...))` headers in the linter.
arith_offenders() {
  grep -nE '(^|[^$[:alnum:]_])\(\([^()]*(\+\+|--|[^<>=!]=[^=])[^()]*\)\)' "$LINTER" | grep -vE '^[0-9]+:[[:space:]]*(#|for[[:space:]]*\(\()' || true
}
expect_contains "no errexit-unsafe ((...)) statements in the linter" pass "" bash -c '[[ -z "$1" ]] || { printf "%s\n" "$1"; exit 1; }' _ "$(arith_offenders)"

printf 'SELFTEST pass=%s fail=%s\n' "$PASS" "$FAIL"
[[ "$FAIL" == 0 ]]
