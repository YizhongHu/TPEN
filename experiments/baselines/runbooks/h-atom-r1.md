---
item_id: "dc036f76-7ddb-4e2f-baf8-b33a1161b539"
source_repo: "TPEN"
source_sha: "PENDING_OPERATOR_TRAIT_AND_REVIEWED_SOURCE"
science: "PENDING_OPERATOR"
runbook_revision: 1
---

# BL-R1 — Hydrogen baseline runbook

This is a **draft runbook**. It is intentionally non-runnable until the
operator selects `science: true|false`, a reviewer pins `source_sha`, and the
corresponding source commit is recorded in Task Orchestrator. The runner does
not resolve either value. If either remains `PENDING_*`, stop before cloning,
submitting, or allocating resources and report it to the orchestrator.

## Scope, ownership, and hard stops

Run only `h_atom` with the physical identity in
`experiments/baselines/runbooks/configs/h-atom-r1.yaml`: one electron,
spins `(1, 0)`, one `Z=1` nucleus at the origin in bohr. The matrix is five
code/ansatz rows at seeds `0,1,2`: DeepQMC `default`, `ferminet`, `psiformer`;
FermiNet-codebase `ferminet`, `psiformer`. The runner collects operational
evidence and hands it to the analyst; the runner does not decide completion,
convergence, significance, science status, archive scope, or a retry.

Run only on Polaris queue `debug`, one exclusive node with four A100s and one
shared per-user running slot. Before *each* access or submission reread
`cluster-access`, Polaris `00-read-first-polaris`, `usage-conventions`,
`polaris-node-shape-gpu-binding-guardrails-2026-08-12`, and Clusters
`free-submit-queues`. Every SSH command uses `ssh -o BatchMode=yes`. On any
access, environment, assertion, scheduler, output, or collection deviation:
stop, preserve the named paths and scheduler response, and report; do not
patch, retry, resubmit, switch queue, chain a job, cancel, or delete.

The first and only authorized execution stage is the ramp: one PBS job with
five 1,000-step rows. It records elapsed seconds/step separately for every
code/ansatz. After it, the **orchestrator** writes a new plan decision from the
observed measurements. No 50,000-step row, inference evaluation, or trajectory
export may start without that revision. The candidate 50,000-step length is a
planning hypothesis intended to expose a late training-history rise while
remaining inside the one-hour debug wall; the ramp, not this file, decides
whether it fits. Never split a row across jobs.

## Fixed source and inputs

At authorization, replace the three `PENDING_*` header values in a reviewed
revision; use the same full source SHA for the checkout, runbook, configs, and
submission receipt. The runner shall first read that file from the pinned
commit and verify its Git blob ID against `runbook-index`. A changed SHA, blob,
lockfile, config file, or runbook is a stop condition requiring a new reviewed
revision and renewed applicable approval.

The only allowed installed environments are:

```sh
DQMC_VENV=/home/rhu/.venvs/deepqmc-jax083-edf373e7
FN_VENV=/home/rhu/.venvs/ferminet-jax092-60c9fab3
FERMINET_SRC=/home/rhu/src/ferminet-psiformer-60c9fab3
EAGLE_RUNS=/eagle/HetRxnEnergy/rhu/runs
```

Derive `DQMC_SRC` from the installed `deepqmc` package at runtime; do not guess
it from a historical path. Both source trees must be clean and their commit
must be written to the operational receipt. `h-atom-r1.yaml` is the immutable
input specification. The native commands below deliberately use the upstream
Hydra `hamil/mol=H` (specified in `configs/deepqmc_h_atom.yaml`) and the
FermiNet atom configuration (`configs/ferminet_h_atom.py`), then verify their
resolved artifacts against that specification; they do not infer physics from
a run-directory name.

## Pre-submit command

Run this command on the Polaris login node only after the header has been
resolved. It prepares a unique root and proves PBS output paths exist before
`qsub`; it does not submit.

```sh
set -euo pipefail
test "$SOURCE_SHA" != PENDING_OPERATOR_TRAIT_AND_REVIEWED_SOURCE
test "$SCIENCE" = true || test "$SCIENCE" = false
test -n "$RUN_TAG"
RUN_ROOT="$EAGLE_RUNS/bl-r1-h-atom-${RUN_TAG}-${SOURCE_SHA:0:12}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"
test -d "$LOG_DIR"
git -C "$SOURCE_CHECKOUT" rev-parse --verify "$SOURCE_SHA^{commit}"
test "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" = "$SOURCE_SHA"
test -z "$(git -C "$SOURCE_CHECKOUT" status --porcelain=v1)"
git -C "$SOURCE_CHECKOUT" ls-tree "$SOURCE_SHA" \
  experiments/baselines/runbooks/h-atom-r1.md \
  experiments/baselines/runbooks/configs/h-atom-r1.yaml \
  experiments/baselines/runbooks/configs/deepqmc_h_atom.yaml \
  experiments/baselines/runbooks/configs/ferminet_h_atom.py
```

Expected final command output: four `100644 blob ...` lines. Any other result
stops the run. Write `RUN_ROOT`, `SOURCE_SHA`, runbook blob ID, config blob ID,
and the operator's selected science branch into `pre_submit.json` under
`RUN_ROOT`; the runner hands this file to the analyst.

## Ramp PBS template — the only job before re-planning

Save the following exact text as `$RUN_ROOT/ramp.pbs`, substituting only the
already-approved `RUN_ROOT`, `SOURCE_CHECKOUT`, `SOURCE_SHA`, and selected
`SCIENCE`. The runner must compare the saved file byte-for-byte with the
reviewed template after substitution and retain both its SHA-256 and `qsub`
response. The `mkdir -p` step above is mandatory before `qsub` because PBS may
otherwise hold the job before a process starts.

```bash
#!/bin/bash
#PBS -A HetRxnEnergy
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -k doe
set -euo pipefail

test "${PBS_QUEUE:?missing PBS_QUEUE}" = debug
test -n "${PBS_JOBID:?missing PBS_JOBID}"
test -d "$RUN_ROOT/logs"
test "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" = "$SOURCE_SHA"
test -z "$(git -C "$SOURCE_CHECKOUT" status --porcelain=v1)"
cd "$SOURCE_CHECKOUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
DQMC_PY=/home/rhu/.venvs/deepqmc-jax083-edf373e7/bin/python
DQMC=/home/rhu/.venvs/deepqmc-jax083-edf373e7/bin/deepqmc
FN_PY=/home/rhu/.venvs/ferminet-jax092-60c9fab3/bin/python
FERMINET_SRC=/home/rhu/src/ferminet-psiformer-60c9fab3
for executable in "$DQMC_PY" "$DQMC" "$FN_PY" "$FERMINET_SRC/ferminet/main.py"; do
  test -x "$executable" || test -f "$executable"
done
test "$("$DQMC_PY" -c 'import sys; print(sys.prefix)')" = /home/rhu/.venvs/deepqmc-jax083-edf373e7
test "$("$FN_PY" -c 'import sys; print(sys.prefix)')" = /home/rhu/.venvs/ferminet-jax092-60c9fab3
"$DQMC_PY" -m experiments.baselines.check_polaris_deepqmc_env env \
  --expect-prefix /home/rhu/.venvs/deepqmc-jax083-edf373e7 --expect-jax 0.8.3
"$FN_PY" - <<'PY'
import jax, sys
assert sys.prefix == '/home/rhu/.venvs/ferminet-jax092-60c9fab3'
assert any(device.platform == 'gpu' for device in jax.devices())
print({'python': sys.executable, 'prefix': sys.prefix, 'jax': jax.__version__,
       'devices': [str(device) for device in jax.devices()]})
PY
nvidia-smi --query-gpu=name,uuid --format=csv,noheader
date --iso-8601=seconds

run_dqmc () {
  local ansatz="$1" gpu="$2" row="$RUN_ROOT/ramp/deepqmc-${ansatz}-seed0"
  mkdir -p "$row"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    start=$(date --iso-8601=seconds)
    "$DQMC" hydra.run.dir="$row" hamil/mol=H ansatz="$ansatz" \
      task.seed=0 task.steps=1000 task.electron_batch_size=4096
    end=$(date --iso-8601=seconds)
    printf 'start=%s\nend=%s\nsteps=1000\n' "$start" "$end" > "$row/timing.txt"
    "$DQMC_PY" -m experiments.baselines.check_polaris_deepqmc_env seed \
      --run-dir "$row" --expect-seed 0
  ) >"$row/stdout.log" 2>"$row/stderr.log"
}
run_dqmc default 0 &
run_dqmc ferminet 1 &
run_dqmc psiformer 2 &
wait

run_fn () {
  local ansatz="$1" gpu="$2" row="$RUN_ROOT/ramp/ferminet-codebase-${ansatz}-seed0"
  mkdir -p "$row/run"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    start=$(date --iso-8601=seconds)
    "$FN_PY" "$FERMINET_SRC/ferminet/main.py" \
      --config "$SOURCE_CHECKOUT/experiments/baselines/runbooks/configs/ferminet_h_atom.py" \
      --config.batch_size 4096 --config.optim.iterations 1000 \
      --config.network.network_type "$ansatz" --config.network.determinants 16 \
      --config.debug.deterministic True --config.debug.seed 0 \
      --config.log.save_path "$row/run"
    end=$(date --iso-8601=seconds)
    printf 'start=%s\nend=%s\nsteps=1000\n' "$start" "$end" > "$row/timing.txt"
    test -s "$row/run/train_stats.csv"
  ) >"$row/stdout.log" 2>"$row/stderr.log"
}
run_fn ferminet 0 &
run_fn psiformer 1 &
wait
printf 'RAMP_COMPLETE job=%s queue=%s\n' "$PBS_JOBID" "$PBS_QUEUE" \
  | tee "$RUN_ROOT/ramp/ramp.complete"
```

Submit only after confirming the exact `-o` and `-e` directories exist:

```sh
test -d "$LOG_DIR"
qsub -o "$LOG_DIR/ramp.stdout" -e "$LOG_DIR/ramp.stderr" "$RUN_ROOT/ramp.pbs"
```

Expected response is one PBS job identifier. Record it verbatim. If `qsub`
does not return exactly one identifier, do not issue another `qsub`.

## Ramp handoff and re-plan gate

After terminal PBS observation, the runner verifies only these mechanical
facts and hands the cited files to the orchestrator and analyst: `qstat -f`
job state/comment/run count; `ramp.complete`; five `timing.txt`; five stdout
and stderr logs; three DeepQMC resolved Hydra configs; two FermiNet
`train_stats.csv`; `nvidia-smi` banner; source/config/runbook blobs; and a
SHA-256 inventory. The runner calculates seconds/step as elapsed seconds
divided by 1,000 and records the raw values, not a pass/fail grade.

Stop unless every row has its named artifact, all required process return codes
are zero, all DeepQMC seed snapshots say seed 0, and the logs show the asserted
interpreter/GPU provenance. The runner must not interpret energy or variance.
The orchestrator alone compares the observed five rates with the one-hour
debug wall, writes a plan revision stating the full-run geometry and budget, and
obtains any further necessary human authority. Missing #725 merge or missing
real-producer DeepQMC adapter round trip also stops before full runs.

## Full-run template after an explicit revised contract

This section is a locked recipe, not current authorization. The revised
contract must set `FULL_STEPS` (initial candidate `50000`), exact seeds,
job shape, and per-row wall fit from the ramp. It must keep the producer flags,
batch size, system identity, source, queue, and no-retry rule above. Launch
each row with `task.steps="$FULL_STEPS"` or
`--config.optim.iterations "$FULL_STEPS"`, its explicit seed, a unique row
root, and a single `CUDA_VISIBLE_DEVICES` set before Python import. Use the
same preflight, PBS queue assertion, output-directory check, timing stamps,
resolved-config check, and device banner. If any row would exceed the wall,
stop; do not chain it. The runner emits DeepQMC records only with
`experiments.baselines.adapters.deepqmc` and FermiNet records only with
`experiments.baselines.adapters.ferminet`, using their respective documented
training-tail windows. `training_tail` and a later `inference` estimator must
remain separate datasets and labels.

After PR #725 merges, run its `trajectory.py` against each completed native
artifact, preserving its per-step table and summary. Its export must label
FermiNet `ewvar` separately from walker local-energy variance. The collector
must validate every intended `baseline_record.json`; any missing or invalid
record stops collection handoff rather than producing a partial comparison.

## Retention and final handoff

Keep the exact Eagle run root, PBS scripts/logs, source and config snapshots,
native `result.h5`/`train_stats.csv`, adapter records, trajectory exports,
collection output, and SHA-256 inventory. Never delete or archive them under
this runbook. For a non-science selection, their lifecycle/disposition remains
with the manager and human approval; for science, the analyst must record this
slice's scientific report and Holystore backup coverage before terminal review.
The runner hands only operational evidence and missing-artifact facts to the
separate analyst, who is responsible for answering the question.
