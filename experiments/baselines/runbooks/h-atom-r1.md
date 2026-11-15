---
item_id: "dc036f76-7ddb-4e2f-baf8-b33a1161b539"
runbook_revision: 2
runbook_commit: "external: runbook-index-r2 and the submission receipt"
science: "PENDING_OPERATOR"
---

# BL-R1 — atomic-H baseline ramp

This is the complete, one-shot **ramp** procedure.  It is not authority for a
full run.  The runner executes commands exactly, records mechanics, and stops
on every stated failure; it does not retry, modify an input, cancel work,
interpret physics, or decide whether to continue.  The orchestrator owns the
post-ramp plan decision.

## Contract and fixed surface

The operator must select `science: true|false` before freeze.  For `false`,
use the reviewed frozen `dev` commit and disposable/temporary data; for
`true`, use a reviewed `main` commit, human submission approval, a science
report, and Holystore backup.  `runbook_commit` is deliberately external: a
file cannot name the Git commit that will contain it.  The verifier records
that exact TPEN commit, this file's blob, the PBS-template blob, and every
config/helper blob in `runbook-index-r2`; the runner copies those values into
its submission receipt.  A pending/missing value is a stop.

Only these five seed-0 rows run: DeepQMC `default`, `ferminet`, `psiformer`;
FermiNet-codebase `ferminet`, `psiformer`.  Each has 1,000 optimizer steps and
batch 4,096.  DeepQMC's selected task is respectively `train`,
`train_ferminet`, and `train_psiformer`; all set `task.pretrain_steps=0` so
ramp timings exclude pretraining.  This is explicit because the pinned task
defaults differ: `train.yaml:7-12`, `train_ferminet.yaml:7-12`, and
`train_psiformer.yaml:7-20` in
`/tmp/bl-r1-pinned-src/deepqmc-edf373e7/src/deepqmc/conf/task/`.

The immutable TPEN physical input is
`configs/h-atom-r1.yaml`: one electron, spins `(1,0)`, and a `Z=1` nucleus at
the origin.  DeepQMC's native H group declares the origin/charge/one-electron
state at `src/deepqmc/conf/hamil/mol/H.yaml:1-5` in its pinned producer.  The
FermiNet input fixes `atom="H"` and `(1,0)` in
`configs/ferminet_h_atom.py`; its atom config constructs the molecule/spins at
`/tmp/bl-r1-pinned-src/ferminet-f4b1846c/ferminet/configs/atom.py:36-60`.

The fixed producer identities are DeepQMC
`edf373e7c26545616f87b3b10c3e48b25c70f627`
(`/home/rhu/src/deepqmc-edf373e7`, Python 3.12/JAX 0.8.3) and FermiNet
`f4b1846cd6f2776321b09b027d4d5506ae98e19b`
(`/home/rhu/src/ferminet-psiformer-60c9fab3`, JAX 0.9.2).  Do not substitute a
package path, producer revision, task default, or config.  `ramp.pbs` sets
DeepQMC `task.max_restarts=1`: its native retry loop is
`src/deepqmc/train.py:303-368` and its default is three at `:48-65`, so this
permits one training pass only.  FermiNet explicitly enables `debug.check_nan`
and disables `optim.reset_if_nan`; their defaults are false at
`ferminet/base_config.py:85,305-318`.  Its step/batch flags are native config
fields (`base_config.py:47,61`), seed is consumed in `train.py:400-444`, and
the main optimization loop and CSV writes are at `train.py:965-1020`.
DeepQMC's Hydra defaults/working-directory handoff are in
`src/deepqmc/conf/config.yaml:1-17` and `src/deepqmc/app.py:314-328`; FermiNet
accepts/resolves its config at `ferminet/main.py:28-35`.  Those references are
all to the two pinned source snapshots named above.

Use Polaris `debug` only: one node, one hour, no chain.  The ramp measures
seconds/step before an orchestrator-owned re-plan determines full steps,
seeds, budget, and whether pretraining is in scope.  No inference evaluation,
trajectory export, or full run is authorized here.

## Pre-freeze verifier checks (read-only; runner does **not** do these)

1. Confirm both producer paths are clean at the stated commits; their named
   interpreters import the stated JAX versions; the listed CLI overrides and
   custom FermiNet config resolve on Polaris.
2. Resolve every row and confirm the stated physical identity, ansatz, batch,
   seed, 1,000 steps, zero pretraining, no-restart/NaN settings, and artifact
   layout; confirm DeepQMC `result.h5` exposes finite local-energy mean/std
   (the producer logger writes `result.h5` at `deepqmc/log.py:204-253` and
   produces these statistics at `fit.py:146-160`) and FermiNet emits a finite
   0--999 `train_stats.csv` trace.
3. Prepare a clean exact TPEN checkout and a shell-quoted `ramp-inputs.env`,
   validate its blob/commit values, Eagle capacity, and that `qstat -xf` can
   read terminal jobs.  Record the verifier receipt path and SHA-256.

If any check fails, no runner command is authorized; the verifier reports the
failure to the orchestrator.  These are cluster facts, not assumptions for the
runner.

## Runner input and submission

The verifier supplies exactly one immutable `ramp-inputs.env`, with shell-safe
single-line values: `RUNBOOK_COMMIT`, `SCIENCE`, `SOURCE_CHECKOUT`, `RUN_ROOT`,
`PRE_FREEZE_RECEIPT`, `PRE_FREEZE_RECEIPT_SHA256`, and the six Git blob IDs
`RUNBOOK_BLOB`, `PBS_BLOB`, `INPUT_SPEC_BLOB`, `DQ_CONFIG_BLOB`,
`FN_CONFIG_BLOB`, and `RECEIPT_HELPER_BLOB`.  `SOURCE_CHECKOUT` is the
prepared exact TPEN checkout, and `RUN_ROOT` is a never-before-used Eagle
path.  It must not contain shell code.  The runner does not create or edit it.

The verifier handoff gives the runner one literal path to that file.  The
following shell template takes it as its sole positional argument; no other
substitution is allowed.  Any failing command stops; preserve the existing
path and report it.  `mkdir` deliberately has no `-p`, so a reused run
tag/root cannot overwrite or blend results.

```bash
set -euo pipefail
test "$#" = 1
INPUTS_FILE="$1"
source "$INPUTS_FILE"
test "$SCIENCE" = true || test "$SCIENCE" = false
test -n "$RUNBOOK_COMMIT" && test -n "$SOURCE_CHECKOUT" && test -n "$RUN_ROOT"
test -f "$PRE_FREEZE_RECEIPT"
test "$(sha256sum "$PRE_FREEZE_RECEIPT" | awk '{print $1}')" = "$PRE_FREEZE_RECEIPT_SHA256"
git -C "$SOURCE_CHECKOUT" rev-parse --verify "$RUNBOOK_COMMIT^{commit}"
test "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" = "$RUNBOOK_COMMIT"
test -z "$(git -C "$SOURCE_CHECKOUT" status --porcelain=v1)"
while read -r variable path; do
  expected="${!variable}"
  actual="$(git -C "$SOURCE_CHECKOUT" ls-tree "$RUNBOOK_COMMIT" -- "$path" | awk '{print $3}')"
  test -n "$expected" && test "$actual" = "$expected"
done <<'BLOBS'
RUNBOOK_BLOB experiments/baselines/runbooks/h-atom-r1.md
PBS_BLOB experiments/baselines/runbooks/h-atom-r1-ramp.pbs
INPUT_SPEC_BLOB experiments/baselines/runbooks/configs/h-atom-r1.yaml
DQ_CONFIG_BLOB experiments/baselines/runbooks/configs/deepqmc_h_atom.yaml
FN_CONFIG_BLOB experiments/baselines/runbooks/configs/ferminet_h_atom.py
RECEIPT_HELPER_BLOB experiments/baselines/runbooks/ramp_receipt.py
BLOBS
mkdir "$RUN_ROOT"
mkdir "$RUN_ROOT/logs" "$RUN_ROOT/ramp"
cp "$INPUTS_FILE" "$RUN_ROOT/ramp-inputs.env"
sha256sum "$RUN_ROOT/ramp-inputs.env" "$PRE_FREEZE_RECEIPT" > "$RUN_ROOT/input-inventory.sha256"
cp "$SOURCE_CHECKOUT/experiments/baselines/runbooks/h-atom-r1-ramp.pbs" "$RUN_ROOT/ramp.pbs"
export RUN_ROOT SOURCE_CHECKOUT RUNBOOK_COMMIT SCIENCE PRE_FREEZE_RECEIPT PRE_FREEZE_RECEIPT_SHA256
JOBID="$(qsub -v RUN_ROOT,SOURCE_CHECKOUT,RUNBOOK_COMMIT,SCIENCE,PRE_FREEZE_RECEIPT,PRE_FREEZE_RECEIPT_SHA256 -o "$RUN_ROOT/logs/ramp.stdout" -e "$RUN_ROOT/logs/ramp.stderr" "$RUN_ROOT/ramp.pbs")"
test -n "$JOBID"
printf '%s\n' "$JOBID" > "$RUN_ROOT/jobid.txt"
while :; do
  qstat -xf "$JOBID" > "$RUN_ROOT/qstat-final.txt"
  STATE="$(awk -F= '/^[[:space:]]*job_state[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2; exit}' "$RUN_ROOT/qstat-final.txt")"
  case "$STATE" in
    F) break ;;
    Q|R|B|E|W|T|S) sleep 30 ;;
    *) printf 'unexpected PBS state: %s\n' "$STATE" >&2; exit 1 ;;
  esac
done
EXIT_STATUS="$(awk -F= '/^[[:space:]]*Exit_status[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2; exit}' "$RUN_ROOT/qstat-final.txt")"
test "$EXIT_STATUS" = 0
```

`qsub` must produce one nonempty PBS job identifier; otherwise do not submit
again.  The runner performs no SSH.  The PBS script asserts
`PBS_QUEUE=debug`, makes all producer/interpreter/source checks, and uses a
pre-existing `-o/-e` directory.  It records each child exit status, waits for
already-launched siblings without cancellation, and prevents FermiNet launch
when the DeepQMC wave fails.

## PBS behavior and handoff

`h-atom-r1-ramp.pbs` owns the zero-judgement execution.  It creates exactly
one row directory once; verifies the resolved/native artifacts after each row;
records `receipt/receipt.json` using `ramp_receipt.py`; then calls its gate.
The gate refuses missing, mismatched, or nonzero row receipts.  This helper is
pure standard library and its offline tests cover missing/failed rows and
overwrite refusal; it makes no scientific claim.

The runner hands the analyst and orchestrator, without grading them:

- `ramp-inputs.env`, `input-inventory.sha256`, `jobid.txt`, `qstat-final.txt`,
  PBS stdout/stderr, PBS script, and the pre-freeze receipt;
- the five row stdout/stderr, timing files, receipts, resolved configs,
  DeepQMC `result.h5`, and FermiNet `train_stats.csv`; and
- `ramp.complete` only if every mechanical gate passed.

The analyst may calculate seconds/step from each timing file.  The
orchestrator must issue a new plan before any later run.  On every missing
artifact, nonfinite value, nonzero child status, failed gate, scheduler
deviation, or missing terminal `qstat` record: stop and report; do not repair,
retry, resubmit, cancel, or delete.  After PR #725 merges, a separate approved
collection stage may use `trajectory.py`; it is outside this ramp.
