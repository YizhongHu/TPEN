#!/bin/bash
# Complete verifier-only submitter. Usage:
#   h-atom-r1-verify-submit.sh INPUTS_FILE success-mini|fail-deepqmc-default
set -euo pipefail

test "$#" = 2 || test "$#" = 3
INPUTS_FILE="$1"
CONTROL="$2"
DRY_RUN="${3:-}"
test -z "$DRY_RUN" || test "$DRY_RUN" = --dry-run
case "$CONTROL" in
  success-mini) INJECT_FAIL= ;;
  fail-deepqmc-default) INJECT_FAIL=deepqmc-default-seed0 ;;
  *) printf 'unknown verification control: %s\n' "$CONTROL" >&2; exit 2 ;;
esac

# A verifier must not accidentally inherit a prior mode.  The script itself,
# not its caller, constructs the complete verified qsub environment below.
test -z "${BL_R1_MODE+x}"
test -z "${BL_R1_VERIFY_STEPS+x}"
test -z "${BL_R1_VERIFY_INJECT_FAIL+x}"
source "$INPUTS_FILE"
test "$SCIENCE" = true || test "$SCIENCE" = false
test "$(sha256sum "$PRE_FREEZE_RECEIPT" | awk '{print $1}')" = "$PRE_FREEZE_RECEIPT_SHA256"
git -C "$SOURCE_CHECKOUT" rev-parse --verify "$RUNBOOK_COMMIT^{commit}"
test "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" = "$RUNBOOK_COMMIT"
test -z "$(git -C "$SOURCE_CHECKOUT" status --porcelain=v1)"
while read -r var path; do
  test -n "${!var}"
  test "$(git -C "$SOURCE_CHECKOUT" ls-tree "$RUNBOOK_COMMIT" -- "$path" | awk '{print $3}')" = "${!var}"
done <<'BLOBS'
RUNBOOK_BLOB experiments/baselines/runbooks/h-atom-r1.md
PBS_BLOB experiments/baselines/runbooks/h-atom-r1-ramp.pbs
INPUT_SPEC_BLOB experiments/baselines/runbooks/configs/h-atom-r1.yaml
DQ_CONFIG_BLOB experiments/baselines/runbooks/configs/deepqmc_h_atom.yaml
FN_CONFIG_BLOB experiments/baselines/runbooks/configs/ferminet_h_atom.py
RECEIPT_HELPER_BLOB experiments/baselines/runbooks/ramp_receipt.py
VERIFY_SUBMIT_BLOB experiments/baselines/runbooks/h-atom-r1-verify-submit.sh
BLOBS

mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/logs" "$RUN_ROOT/ramp"
cp "$INPUTS_FILE" "$RUN_ROOT/ramp-inputs.env"
cp "$SOURCE_CHECKOUT/experiments/baselines/runbooks/h-atom-r1-ramp.pbs" "$RUN_ROOT/ramp.pbs"
sha256sum "$RUN_ROOT/ramp-inputs.env" "$PRE_FREEZE_RECEIPT" "$RUN_ROOT/ramp.pbs" > "$RUN_ROOT/input-inventory.sha256"

export RUN_ROOT SOURCE_CHECKOUT RUNBOOK_COMMIT SCIENCE PRE_FREEZE_RECEIPT PRE_FREEZE_RECEIPT_SHA256
export BL_R1_MODE=verify BL_R1_VERIFY_STEPS=10
if test -n "$INJECT_FAIL"; then export BL_R1_VERIFY_INJECT_FAIL="$INJECT_FAIL"; fi
printf '{"control":"%s","mode":"%s","steps":%s,"inject_fail":"%s"}\n' "$CONTROL" "$BL_R1_MODE" "$BL_R1_VERIFY_STEPS" "$INJECT_FAIL" > "$RUN_ROOT/verify-submit-mode.json"

write_submit_partial() {
  set +e
  /home/rhu/.venvs/deepqmc-jax083-edf373e7/bin/python "$SOURCE_CHECKOUT/experiments/baselines/runbooks/ramp_receipt.py" inventory --root "$RUN_ROOT" --output "$RUN_ROOT/artifact-inventory.submit-partial.jsonl" --ramp-required --required ramp-inputs.env --required ramp.pbs --required input-inventory.sha256 --required verify-submit-mode.json --required qsub.stdout --required qsub.stderr --required qsub.status --required jobid.txt > "$RUN_ROOT/artifact-inventory.submit-partial.stdout" 2> "$RUN_ROOT/artifact-inventory.submit-partial.stderr"
  return 0
}
on_submit_exit() { rc=$?; write_submit_partial; exit "$rc"; }
trap on_submit_exit EXIT

QSUB_VARIABLES=RUN_ROOT,SOURCE_CHECKOUT,RUNBOOK_COMMIT,SCIENCE,PRE_FREEZE_RECEIPT,PRE_FREEZE_RECEIPT_SHA256,BL_R1_MODE,BL_R1_VERIFY_STEPS
if test -n "$INJECT_FAIL"; then QSUB_VARIABLES="$QSUB_VARIABLES,BL_R1_VERIFY_INJECT_FAIL"; fi
if test "$DRY_RUN" = --dry-run; then
  env | grep -E '^(RUN_ROOT|SOURCE_CHECKOUT|RUNBOOK_COMMIT|SCIENCE|PRE_FREEZE_RECEIPT|PRE_FREEZE_RECEIPT_SHA256|BL_R1_MODE|BL_R1_VERIFY_STEPS|BL_R1_VERIFY_INJECT_FAIL)=' | sort > "$RUN_ROOT/verify-submit-dry-run.env"
  grep -Fx 'BL_R1_MODE=verify' "$RUN_ROOT/verify-submit-dry-run.env"
  grep -Fx 'BL_R1_VERIFY_STEPS=10' "$RUN_ROOT/verify-submit-dry-run.env"
  printf '%s\n' "$QSUB_VARIABLES" > "$RUN_ROOT/verify-submit-qsub-variables.txt"
  exit 0
fi

set +e
qsub -v "$QSUB_VARIABLES" -o "$RUN_ROOT/logs/ramp.stdout" -e "$RUN_ROOT/logs/ramp.stderr" "$RUN_ROOT/ramp.pbs" > "$RUN_ROOT/qsub.stdout" 2> "$RUN_ROOT/qsub.stderr"
QSUB_STATUS=$?
set -e
printf '%s\n' "$QSUB_STATUS" > "$RUN_ROOT/qsub.status"
test "$QSUB_STATUS" = 0
test "$(wc -l < "$RUN_ROOT/qsub.stdout" | tr -d ' ')" = 1
JOBID="$(cat "$RUN_ROOT/qsub.stdout")"
test -n "$JOBID"
printf '%s\n' "$JOBID" > "$RUN_ROOT/jobid.txt"

for poll in $(seq 1 120); do
  stamp="$(date --iso-8601=seconds)"
  query="$RUN_ROOT/qstat-$poll.txt"
  set +e
  qstat -xf "$JOBID" > "$query" 2>&1
  QUERY_STATUS=$?
  set -e
  printf '%s status=%s file=%s\n' "$stamp" "$QUERY_STATUS" "$(basename "$query")" >> "$RUN_ROOT/qstat-observations.txt"
  test "$QUERY_STATUS" = 0
  cp "$query" "$RUN_ROOT/qstat-last-success.txt"
  STATE="$(awk -F= '/^[[:space:]]*job_state[[:space:]]*=/{gsub(/[[:space:]]/,"",$2);print $2;exit}' "$query")"
  test "$(awk -F= '/^[[:space:]]*queue[[:space:]]*=/{gsub(/[[:space:]]/,"",$2);print $2;exit}' "$query")" = debug
  test "$(awk -F= '/^[[:space:]]*run_count[[:space:]]*=/{gsub(/[[:space:]]/,"",$2);print $2;exit}' "$query")" = 1
  grep -Eq 'Resource_List.select = 1:system=polaris' "$query"
  case "$STATE" in F) break ;; Q|R|B|E|W|T|S) sleep 30 ;; *) exit 1 ;; esac
done
test "$STATE" = F
test "$(awk -F= '/^[[:space:]]*Exit_status[[:space:]]*=/{gsub(/[[:space:]]/,"",$2);print $2;exit}' "$query")" = 0
/home/rhu/.venvs/deepqmc-jax083-edf373e7/bin/python "$SOURCE_CHECKOUT/experiments/baselines/runbooks/ramp_receipt.py" inventory --root "$RUN_ROOT" --output "$RUN_ROOT/artifact-inventory.jsonl" --ramp-required --required ramp-inputs.env --required ramp.pbs --required input-inventory.sha256 --required verify-submit-mode.json --required qsub.stdout --required qsub.stderr --required qsub.status --required jobid.txt --required qstat-observations.txt --required qstat-last-success.txt --required logs/ramp.stdout --required logs/ramp.stderr
