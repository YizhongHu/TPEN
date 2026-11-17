---
item_id: "dc036f76-7ddb-4e2f-baf8-b33a1161b539"
runbook_revision: 7
runbook_commit: "external: runbook-index-r5 and submission receipt"
science: "PENDING_OPERATOR"
---

# BL-R1 — atomic-H baseline ramp

This is the complete one-shot ramp, not authority for full runs. The runner
executes commands, preserves evidence, and stops on every failure: no patch,
retry, resubmission, cancellation, or scientific interpretation.

## Fixed surface and source evidence

The operator selects science true/false: non-science uses frozen dev and
temporary data; science uses main, human submission approval, a science report,
and Holystore backup. The containing TPEN commit/blob/config/helper/PBS blobs
are bound externally in runbook-index-r5 and the submission receipt.

Two seed-0 rows run: FermiNet-codebase ferminet and psiformer; batch 4096;
1000 real-ramp steps; zero pretraining. The H input is one electron,
spins (1,0), one Z=1 origin nucleus. The DeepQMC rows are removed because
the pinned producer rejects one-electron H at `hamil.py:144` (`assert n_elec >
1`), as recorded in TO note `plan-revision-1-deepqmc-cannot-run-h-20261116`.
Each H-atom record and receipt is labelled `ferminet f4b1846c + logging patch
<patch sha256>`. The patch adds `walker_variance`, sourced from FermiNet
`aux_data.variance` (unclipped batch local-energy variance, `loss.py:227`);
`ewvar` remains time variance. The trajectory reader
(`adapters/ferminet.py`) does not support this new column in this slice; that
is a follow-up item.

Pinned producer: FermiNet
f4b1846cd6f2776321b09b027d4d5506ae98e19b at
/home/rhu/src/ferminet-psiformer-60c9fab3 (JAX 0.9.2).
The PBS job asserts that shared tree is at this commit and clean, copies it
to a new `$RUN_ROOT/src/ferminet-patched`, applies the committed logging patch
there, asserts the diff touches only `ferminet/train.py`, and records the patch
SHA-256, patched `train.py` SHA-256, and full diff under the run root. It never
modifies the shared tree; `FN_SRC`/`PYTHONPATH` point to the patched copy.
FermiNet batch/step/NaN/pretraining defaults: ferminet/base_config.py:47,61,
85,305-318; required trace schema: train.py:648; loop/NaN: train.py:965-1033.
Each row asserts one GPU using its own interpreter.

The PBS template passes every FermiNet boolean override with an equals form,
for example `--config.optim.reset_if_nan=False`. This is the syntax consumed
by the pinned `absl.flags`/`ml_collections.config_flags` path in
`ferminet/main.py`; a space-separated boolean value can set the flag by
presence and leave the value token ineffective. FermiNet logs the resolved
`FLAGS.config` immediately before training. The row receipt parses that exact
`stderr.log` dump into `producer-config.json` and stops before the row receipt
if any required value differs: `reset_if_nan=false`, `check_nan=true`,
`deterministic=true`, `seed=0`, `batch_size=4096`, `iterations=STEPS`, and
`pretrain.iterations=0`. A separately re-resolved config is intent evidence,
not a producer receipt, and is not used by this runbook.

Use Polaris debug only, one node, one hour, no chain. The orchestrator must
re-plan from seconds/step before any full run, inference, or trajectory export.

## Pre-freeze verifier checks (read-only; runner does not do them)

1. Confirm clean pinned producer heads, named Python/JAX/package origins,
   flags/custom FermiNet config, Eagle capacity, and terminal qstat -xf.
2. Resolve all rows and retain real training/.hydra plus training/result.h5
   fixtures; check H identity and settings/native finite outputs.
3. Prepare clean exact TPEN checkout and immutable data-only ramp-inputs.env:
   RUNBOOK_COMMIT, SCIENCE, SOURCE_CHECKOUT, fresh RUN_ROOT, pre-freeze
   receipt/SHA, and RUNBOOK_BLOB, PBS_BLOB, INPUT_SPEC_BLOB,
   FN_CONFIG_BLOB, RECEIPT_HELPER_BLOB, VERIFY_SUBMIT_BLOB, PATCH_BLOB.

Any failure stops before runner execution.

## Normal runner submission

The verifier provides one ramp-inputs.env path as $1. This template takes only
that argument. mkdir has no -p, so existing root stops before qsub.
QUEUE_POLL_LIMIT=120 and RUN_POLL_LIMIT=120 are separate 60-minute bounds at
30 seconds per observation, not retries. Each exhausted bound records the
last observation and stops/reports without qdel.

~~~bash
set -euo pipefail
test "$#" = 1; INPUTS_FILE="$1"; source "$INPUTS_FILE"
test "$SCIENCE" = true || test "$SCIENCE" = false
test -z "${BL_R1_MODE+x}" && test -z "${BL_R1_VERIFY_STEPS+x}" && test -z "${BL_R1_VERIFY_INJECT_FAIL+x}"
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
FN_CONFIG_BLOB experiments/baselines/runbooks/configs/ferminet_h_atom.py
RECEIPT_HELPER_BLOB experiments/baselines/runbooks/ramp_receipt.py
VERIFY_SUBMIT_BLOB experiments/baselines/runbooks/h-atom-r1-verify-submit.sh
PATCH_BLOB experiments/baselines/runbooks/patches/ferminet-f4b1846c-walker-variance-logging.patch
BLOBS
mkdir "$RUN_ROOT"; mkdir "$RUN_ROOT/logs" "$RUN_ROOT/ramp"
cp "$INPUTS_FILE" "$RUN_ROOT/ramp-inputs.env"
cp "$SOURCE_CHECKOUT/experiments/baselines/runbooks/h-atom-r1-ramp.pbs" "$RUN_ROOT/ramp.pbs"
sha256sum "$RUN_ROOT/ramp-inputs.env" "$PRE_FREEZE_RECEIPT" "$RUN_ROOT/ramp.pbs" > "$RUN_ROOT/input-inventory.sha256"
export RUN_ROOT SOURCE_CHECKOUT RUNBOOK_COMMIT SCIENCE PRE_FREEZE_RECEIPT PRE_FREEZE_RECEIPT_SHA256 PATCH_BLOB
write_submit_partial() {
  set +e
  /home/rhu/.venvs/ferminet-jax092-60c9fab3/bin/python "$SOURCE_CHECKOUT/experiments/baselines/runbooks/ramp_receipt.py" inventory --root "$RUN_ROOT" --output "$RUN_ROOT/artifact-inventory.submit-partial.jsonl" --ramp-required --required ramp-inputs.env --required ramp.pbs --required input-inventory.sha256 --required qsub.stdout --required qsub.stderr --required qsub.status --required jobid.txt > "$RUN_ROOT/artifact-inventory.submit-partial.stdout" 2> "$RUN_ROOT/artifact-inventory.submit-partial.stderr"
  return 0
}
on_submit_exit() { rc=$?; write_submit_partial; exit "$rc"; }
trap on_submit_exit EXIT
set +e
qsub -v RUN_ROOT,SOURCE_CHECKOUT,RUNBOOK_COMMIT,SCIENCE,PRE_FREEZE_RECEIPT,PRE_FREEZE_RECEIPT_SHA256,PATCH_BLOB -o "$RUN_ROOT/logs/ramp.stdout" -e "$RUN_ROOT/logs/ramp.stderr" "$RUN_ROOT/ramp.pbs" > "$RUN_ROOT/qsub.stdout" 2> "$RUN_ROOT/qsub.stderr"
QSUB_STATUS=$?
set -e
printf '%s\n' "$QSUB_STATUS" > "$RUN_ROOT/qsub.status"
test "$QSUB_STATUS" = 0
test "$(wc -l < "$RUN_ROOT/qsub.stdout" | tr -d ' ')" = 1
JOBID="$(cat "$RUN_ROOT/qsub.stdout")"; test -n "$JOBID"; printf '%s\n' "$JOBID" > "$RUN_ROOT/jobid.txt"
QUEUE_POLL_LIMIT=120; RUN_POLL_LIMIT=120; QUEUE_POLLS=0; RUN_POLLS=0; poll=0
while :; do
  poll=$((poll + 1))
  stamp="$(date --iso-8601=seconds)"; query="$RUN_ROOT/qstat-$poll.txt"
  set +e; qstat -xf "$JOBID" > "$query" 2>&1; QUERY_STATUS=$?; set -e
  printf '%s status=%s file=%s\n' "$stamp" "$QUERY_STATUS" "$(basename "$query")" >> "$RUN_ROOT/qstat-observations.txt"
  test "$QUERY_STATUS" = 0
  cp "$query" "$RUN_ROOT/qstat-last-success.txt"
  STATE="$(/home/rhu/.venvs/ferminet-jax092-60c9fab3/bin/python "$SOURCE_CHECKOUT/experiments/baselines/runbooks/ramp_receipt.py" poll-check --path "$query" --field state)"
  PHASE="$(/home/rhu/.venvs/ferminet-jax092-60c9fab3/bin/python "$SOURCE_CHECKOUT/experiments/baselines/runbooks/ramp_receipt.py" poll-check --path "$query" --field phase)"
  test "$(awk -F= '/^[[:space:]]*queue[[:space:]]*=/{gsub(/[[:space:]]/,"",$2);print $2;exit}' "$query")" = debug
  grep -Eq 'Resource_List.select = 1:system=polaris' "$query"
  case "$PHASE" in
    queue) QUEUE_POLLS=$((QUEUE_POLLS + 1)); if test "$QUEUE_POLLS" -gt "$QUEUE_POLL_LIMIT"; then printf 'queue wait bound exceeded; preserve last observation and report, do not qdel\n' >&2; exit 1; fi ;;
    run) RUN_POLLS=$((RUN_POLLS + 1)); if test "$RUN_POLLS" -gt "$RUN_POLL_LIMIT"; then printf 'run wait bound exceeded; preserve last observation and report, do not qdel\n' >&2; exit 1; fi ;;
    terminal) break ;;
    *) exit 1 ;;
  esac
  sleep 30
done
/home/rhu/.venvs/ferminet-jax092-60c9fab3/bin/python "$SOURCE_CHECKOUT/experiments/baselines/runbooks/ramp_receipt.py" inventory --root "$RUN_ROOT" --output "$RUN_ROOT/artifact-inventory.jsonl" --ramp-required --required ramp-inputs.env --required ramp.pbs --required input-inventory.sha256 --required qsub.stdout --required qsub.stderr --required qsub.status --required jobid.txt --required qstat-observations.txt --required qstat-last-success.txt --required logs/ramp.stdout --required logs/ramp.stderr
~~~

PBS writes artifact-inventory.partial.jsonl on every exit. The final command
writes the complete deterministic inventory: fixed relative paths, size,
SHA-256, missing status; it excludes itself and fails on a gap. Handoff is
both inventories, raw qsub output/status, append-only qstat responses, inputs,
PBS bytes/logs, all row logs/receipts/device bindings/timings/producer configs/native
artifacts, and ramp.complete only after mechanical success.

## Verification mode and required controls

Verification uses the complete committed script
experiments/baselines/runbooks/h-atom-r1-verify-submit.sh. It takes exactly an
inputs file and one control name; no setup/qsub fragment may be spliced by
hand. It repeats the normal setup/blob/fresh-root/inventory/poll logic, asserts
no inherited verification variables, then exports every value named in qsub
-v. PBS accepts only BL_R1_MODE=verify, decimal BL_R1_VERIFY_STEPS from 1 to
100, and optional exact BL_R1_VERIFY_INJECT_FAIL. It changes only producer
step count and a post-producer injected row failure. Normal submission still
asserts all three variables unset; PBS records mode real or verify.

The verifier runs exactly one of these complete submitting commands, with a
separate verifier-prepared fresh inputs file for each command:

~~~bash
bash "$SOURCE_CHECKOUT/experiments/baselines/runbooks/h-atom-r1-verify-submit.sh" "$INPUTS_FILE" success-mini
bash "$SOURCE_CHECKOUT/experiments/baselines/runbooks/h-atom-r1-verify-submit.sh" "$INPUTS_FILE" fail-ferminet-ferminet
~~~

The success control runs both real producer rows for ten steps. The failure
control injects a post-producer FermiNet-codebase ferminet-row failure; expected
evidence is a failed receipt, no ramp.complete, and preserved
partial inventory. The script's optional --dry-run performs all pre-qsub setup,
records the exact exported variables and qsub -v set, then stops before qsub.

The three non-submitting controls are literal commands. Existing-root control
uses an inputs file naming an existing RUN_ROOT and must fail before qsub:

~~~bash
set -euo pipefail; source "$INPUTS_FILE"; mkdir "$RUN_ROOT"
~~~

Wrong-producer-SHA control is read-only, submits nothing, and must fail:

~~~bash
test "$(git -C /home/rhu/src/ferminet-psiformer-60c9fab3 rev-parse HEAD)" = 0000000000000000000000000000000000000000
~~~

Malformed-FermiNet-trace control must fail:

~~~bash
printf 'step,energy,ewmean,ewvar,pmove,walker_variance\n0,-0.5,-0.5,0.1,0.5,0.2\n1,,,,,\n' > "$RUN_ROOT/malformed-train_stats.csv"
! /home/rhu/.venvs/ferminet-jax092-60c9fab3/bin/python "$SOURCE_CHECKOUT/experiments/baselines/runbooks/ramp_receipt.py" validate-ferminet-trace --path "$RUN_ROOT/malformed-train_stats.csv" --steps 2
~~~

After success and all four controls, the verifier records commands, statuses,
head, cluster artifacts, and resolved FermiNet fixtures before merge review.
