# Stage 1 He/H2 Polaris direct-venue runbook

This is the optional direct PBS route for the He and H2 Stage-1 rows.  It is
independent of the Cannon runbooks and does not authorize a science
allocation.  The implementation is the pair
`experiments/atomistic/stage1/polaris/pbs_prepare.sh` and
`experiments/atomistic/stage1/polaris/pbs_stage1.sh`.  One short debug
preparation job owns source checkout, uv cache priming, and row rendering;
later submissions use the complete execution wrapper with an explicit mode.

## Scope and approval boundary

The route is one direct Polaris allocation for one already-approved training
lineage, followed by any explicitly selected evaluation row and collection on
Eagle.  The wrapper consumes a frozen row object and calls
`launch.command_for_row`; it does not reconstruct a recipe from `qsub -v`, fan
out to successor lineages, or use a first-class PBS path in `launch.py`.

- `debug` and `debug-scaling` are free queues for bounded smoke, guard, plan,
  evaluation, and collection work.  The PBS queue name is `debug-scaling`,
  with a hyphen.  The preparation job uses only `debug`, one node, four CPUs,
  and a 20-minute walltime.
- `capacity` is required for the 50,000-update training allocation and needs
  fresh operator approval relayed by the manager.  Do not submit it from this
  document without that approval.
- The wrapper's checked-in PBS header is only a safe debug default.  The
  command lines below state the actual queue and resource shape explicitly.
- Never `qdel`, `qhold`, or `qrls`.  A guard, identity, checkpoint, wall-cap,
  or OOM failure is a terminal invalid measurement for that row; resubmission
  requires a new approval boundary.
- `APPROVAL_ID` is a required operational input for every train/eval/dry-exec
  control.  It identifies the manager-relayed queue/allocation approval; it is
  never part of the science row.  No proxy hostname, port, or environment
  recipe is copied from an older receipt.  If network access is needed, stop
  at the current official ALCF documentation gate and record the verified
  host note before proceeding.

Before any Polaris action, read the `cluster-access` skill and the current TO
chain: Clusters `cluster-access-read-first`, `00-orientation`,
`free-submit-queues`, ALCF `provider-and-hpss-orientation-2026-08-07`, and
root `cluster-rule-no-reductions-on-login-nodes-20261127`, plus ALCF's current
`operator-notice-alcf-in-node-proxy-change-20261124`, and
Polaris `00-read-first-polaris`, `usage-conventions`,
`polaris-facility-toolchain-and-gotchas-2026-08-12`,
`polaris-node-shape-gpu-binding-guardrails-2026-08-12`,
`polaris-node-topology-and-cpu-gpu-affinity-2026-09-03`,
`polaris-artifact-inventory-2026-08-12`,
`polaris-submission-rules-systematic-2026-08-27`,
`missing-output-dir-holds-a-job-and-eats-the-debug-slot-2026-09-18`,
`two-uv-binaries-on-polaris-bare-uv-shadows-facility-since-2026-09-08`, and
`cp-a-eagle-preserve-ownership-permission-denied-20261116`.  Gate scripted
access on `ssh -O check polaris`; every SSH command uses
`ssh -o BatchMode=yes`.  If the gate fails, stop.

## Fixed identities and paths

Set these once for one retained run root.  `SOURCE_SHA` is a full released
`main` SHA, not `dev` and not a short SHA.

```bash
SOURCE_SHA=<released-main-sha>
SOURCE_REPO=https://github.com/YizhongHu/TPEN.git   # passed only to the prep job
SYSTEM=he                         # repeat the complete sequence with h2
RESULTS_ROOT=/eagle/HetRxnEnergy/rhu/runs/stage1-${SYSTEM}-${SOURCE_SHA}-<utc>
WRAPPER=/eagle/HetRxnEnergy/rhu/runs/stage1-${SYSTEM}-${SOURCE_SHA}-<utc>/inputs/pbs_stage1.sh
PREP_SCRIPT=/eagle/HetRxnEnergy/rhu/runs/stage1-${SYSTEM}-${SOURCE_SHA}-<utc>/inputs/pbs_prepare.sh
CHECKOUT_ROOT=/eagle/HetRxnEnergy/rhu/runs/stage1-${SYSTEM}-${SOURCE_SHA}-<utc>/checkout
UV_CACHE_DIR=/eagle/HetRxnEnergy/rhu/runs/stage1-${SYSTEM}-${SOURCE_SHA}-<utc>/uv-cache
PYBIN=/soft/applications/conda/2025-09-25/mconda3/bin/python
```

The run root must be created before any `qsub` so PBS can open its `-o` and
`-e` files.  Assert it is below `/eagle/HetRxnEnergy/rhu/`, create `logs/`,
`plan/`, `launch/`, and `inputs/`, copy both scripts there, and record both
SHA-256 values.  Do not copy a git tree with `cp -a` from `/home` to Eagle;
the current Polaris note records that this fails while preserving ownership.

The login node does not clone, run Python, or invoke uv.  Submit exactly one
short debug preparation job.  Inside that allocation the script performs a
full clone (no `--filter`), detaches at the exact SHA, verifies a clean
non-promisor/non-partial checkout, primes the cache with the facility
`$PYBIN -m uv` binary, and renders the phase-1 rows and command receipts under
the Eagle results root:

```bash
mkdir -p "$RESULTS_ROOT"/{inputs,logs,plan}
scp -p experiments/atomistic/stage1/polaris/pbs_stage1.sh "$WRAPPER"
scp -p experiments/atomistic/stage1/polaris/pbs_prepare.sh "$PREP_SCRIPT"
PREP_JOB_ID="$(qsub -N "stage1-prep-${SYSTEM}-${SOURCE_SHA:0:8}" \
  -A HetRxnEnergy -q debug \
  -l select=1:system=polaris:ncpus=4,walltime=00:20:00,filesystems=home:eagle,place=scatter \
  -o "$RESULTS_ROOT/logs/prep.pbs.out" -e "$RESULTS_ROOT/logs/prep.pbs.err" \
  -v SOURCE_REPO="$SOURCE_REPO",SOURCE_SHA="$SOURCE_SHA",RESULTS_ROOT="$RESULTS_ROOT",SYSTEM="$SYSTEM",LINEAGE_SEEDS="1:2:3",PYBIN="$PYBIN",WRAPPER_PATH="$WRAPPER" \
  "$PREP_SCRIPT")"
printf 'submitted prep job %s\n' "$PREP_JOB_ID"
qstat -xf "$PREP_JOB_ID"
```

After PBS reaches a terminal state, inspect only the named `qstat` record and
the named preparation outputs.  A nonzero job or missing output is
`HARNESS_INVALID`/`UNMEASURED`; preserve it and stop.  Do not resubmit without
a new approval.  The preparation job's `$PYBIN -m uv` invocation sets
`UV_PYTHON=$PYBIN`, `UV_NO_MANAGED_PYTHON=1`, `UV_PYTHON_DOWNLOADS=never`, and
`UV_LINK_MODE=copy`; it primes `UV_CACHE_DIR` and writes `prime-env` only as a
reproducible preparation artifact.  The run wrapper later creates its own
job-local environment from that cache.  Never use bare `uv`.

## Wrapper boundary

The wrapper reuses the Stage-1 command contract at
`experiments/atomistic/stage1/launch.py:42-80` through
`launch.command_for_row` in `pbs_stage1.sh:203-226`.  That preserves row
identity, `runtime.seed`, results root, run identity, checkpoint binding,
strict restore, and `load.mode=model_only`.

The PBS wrapper replaces only the Slurm execution shell around that contract:

- `pbs_prepare.sh` owns the one short debug allocation that clones the exact
  source, primes the facility uv cache, runs phase-1 planning, and emits the
  immutable row/command receipts;
- `launch.py:193-277` remains the Cannon renderer and its Slurm/Netscratch
  `FACILITY_RESULTS_PREFIX` and A7 80-GB guard are untouched;
- `pbs_stage1.sh:54-110` supplies the PBS queue/job/GPU guards for Polaris;
- `pbs_stage1.sh:123-185` verifies the pre-staged detached checkout, provisions
  the job-local cu128 environment from the external cache, and cleans only a
  unique owned temporary child;
- `pbs_stage1.sh:203-269` records PBS/facility inputs and runs one selected
  row with its lineage's `CUDA_VISIBLE_DEVICES` and `taskset` CPU range.

The wrapper never silently routes a Polaris job through the Cannon Slurm
template and never edits `launch.py`.

Because the wrapper calls `launch.command_for_row` directly, it explicitly
replaces `launch.render`'s GPU-row admission with a `device=cuda` check before
the Polaris GPU guards.

## Exact command sequence

All commands below are run from a Polaris login shell for light preparation,
submission, and monitoring only.  `qstat -xf <jobid>` is the authoritative
terminal scheduler observation.  Preserve qsub stdout/stderr, qstat output,
wrapper logs, row status files, rendered command receipts, and every failure.

### 1. Consume the prep job's frozen rows

Planning and full roster/grid reconciliation remain S0-B/S1-RC-owned.  The
preparation job writes `plan/phase1-manifest.json`, the exact first
`plan/selected-train.json`, one JSON file per train row under `launch/`, and a
`launch/render-receipt.json`.  Inspect those named files only; do not recreate
or edit rows on the login node.  The same row shape is used for an unbounded
route and for a bounded qualification; only the latter adds the
already-approved `execution.stop_at` datum in its separate request file.

```bash
mkdir -p "$RESULTS_ROOT"/{inputs,logs,plan,launch}
ls -l "$RESULTS_ROOT/plan/phase1-manifest.json" "$RESULTS_ROOT/plan/selected-train.json"
ls -l "$RESULTS_ROOT/launch/render-receipt.json"
```

The selected file is immutable evidence: record its SHA-256, source/config/
overlay identities, lineage, results root, and the manager-relayed
`APPROVAL_ID`.  Do not pass a phase manifest to the wrapper's train/eval
consumer.

### 2. Bounded ramp-up control on free `debug-scaling`

Run the bounded ramp-up through the same wrapper and selected-lineage route
before requesting any capacity approval.  `derive_smoke_request` changes only
the isolated output root and the retained `execution.stop_at`; it does not
invent a second recipe, seed, walker budget, or checkpoint schedule.  The
example below uses the retained 50,000-update boundary and must be adapted
only to the approved selected row and run root.

```bash
SMOKE_RESULTS_ROOT="$RESULTS_ROOT/ramp-up-r1"
SMOKE_SELECTED_ROW="$SMOKE_RESULTS_ROOT/plan/selected-train.json"
mkdir -p "$SMOKE_RESULTS_ROOT"/plan "$SMOKE_RESULTS_ROOT"/logs
test -d "$SMOKE_RESULTS_ROOT/plan" -a -d "$SMOKE_RESULTS_ROOT/logs" || {
  echo "bounded PBS output directories are not ready" >&2
  exit 2
}
"$PYBIN" - "$RESULTS_ROOT/plan/selected-train.json" "$SMOKE_SELECTED_ROW" "$SMOKE_RESULTS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

from experiments.atomistic.stage1.protocol import derive_smoke_request

source, target, results_root = map(Path, sys.argv[1:])
row = json.loads(source.read_text(encoding="utf-8"))
request = derive_smoke_request(row, results_root=str(results_root), stop_at=50_000)
target.write_text(json.dumps(request, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
sha256sum "$SMOKE_SELECTED_ROW" > "$SMOKE_SELECTED_ROW.sha256"
APPROVAL_ID="${MANAGER_RELAYED_APPROVAL_ID:-}"
: "${APPROVAL_ID:?APPROVAL_ID must be set from the explicit manager-relayed value}"
qsub -A HetRxnEnergy -q debug-scaling \
  -l select=1:system=polaris:ngpus=4:ncpus=64,walltime=00:55:00,filesystems=home:eagle,place=scatter \
  -o "$SMOKE_RESULTS_ROOT/logs/train-ramp.pbs.out" -e "$SMOKE_RESULTS_ROOT/logs/train-ramp.pbs.err" \
  -v MODE=train,SOURCE_SHA="$SOURCE_SHA",CHECKOUT_ROOT="$CHECKOUT_ROOT",UV_CACHE_DIR="$UV_CACHE_DIR",RESULTS_ROOT="$SMOKE_RESULTS_ROOT",EXPECTED_QUEUE=debug-scaling,PYBIN="$PYBIN",SELECTED_ROW="$SMOKE_SELECTED_ROW",APPROVAL_ID="$APPROVAL_ID" \
  "$RESULTS_ROOT/inputs/pbs_stage1.sh"
```

The bounded control is compute-skipped only when explicitly run in
`DRY_EXEC=1`; a submitted ramp-up must retain the PBS stdout/stderr, qstat
terminal record, wrapper receipts, row log/status, and the selected-row hash.
It is a ladder rung, not a science result or capacity approval.

### 3. Capacity-approval gate

Before requesting any `capacity` approval, the operator must have all three
non-empty, retained artifacts from the planned and passed ramp-up ladder and
the smoke/science comparison.  These are evidence gates, not proxy values for
the later science run:

```bash
RAMP_UP_PLAN_RECEIPT="$RESULTS_ROOT/plan/ramp-up-planned.receipt"
RAMP_UP_PASS_RECEIPT="$RESULTS_ROOT/plan/ramp-up-passed.receipt"
SMOKE_SCIENCE_ARTIFACT_DIFF="$RESULTS_ROOT/plan/smoke-science-artifact-diff.txt"
test -s "$RAMP_UP_PLAN_RECEIPT" || { echo "missing planned ramp-up receipt" >&2; exit 2; }
test -s "$RAMP_UP_PASS_RECEIPT" || { echo "missing passed ramp-up receipt" >&2; exit 2; }
test -s "$SMOKE_SCIENCE_ARTIFACT_DIFF" || { echo "missing smoke/science artifact diff" >&2; exit 2; }
```

Do not request or submit the paid run until those checks pass and a fresh
manager-relayed approval names the exact source SHA, selected row, Eagle root,
queue, node/GPU shape, and wall cap.  Paid runs remain an operator decision.

### 4. One selected-lineage train control

This is the first nonfree action and needs fresh approval naming the exact
source SHA, selected row, Eagle root, queue, one-node/four-GPU shape, and wall
cap.  Before qsub, verify the queue/account live and the `-o`/`-e` directory.
The command below is complete and carries only facility inputs plus the frozen
row path; the recipe is reconstructed nowhere in PBS.

```bash
APPROVAL_ID=<manager-relayed-approval-id>
qsub -A HetRxnEnergy -q capacity \
  -l select=1:system=polaris:ngpus=4:ncpus=64,walltime=11:00:00,filesystems=home:eagle,place=scatter \
  -o "$RESULTS_ROOT/logs/train.pbs.out" -e "$RESULTS_ROOT/logs/train.pbs.err" \
  -v MODE=train,SOURCE_SHA="$SOURCE_SHA",CHECKOUT_ROOT="$CHECKOUT_ROOT",UV_CACHE_DIR="$UV_CACHE_DIR",RESULTS_ROOT="$RESULTS_ROOT",EXPECTED_QUEUE=capacity,PYBIN="$PYBIN",SELECTED_ROW="$RESULTS_ROOT/plan/selected-train.json",APPROVAL_ID="$APPROVAL_ID" \
  "$RESULTS_ROOT/inputs/pbs_stage1.sh"
```

The wrapper derives the affinity-constrained layout from `taskset -pc $$` and
`lscpu -p=CPU,CORE,NODE`, then runs only the selected lineage on its matching
GPU/CPU slice.  The unbounded train row and a bounded request use this exact
same train -> S1-R restart -> public gateway route; they differ only in the
row's `execution.stop_at` and isolated output root.

### 5. One selected eval row on free `debug`

After the checkpoint owner publishes an eval manifest, materialize one
approved eval row in the same way.  This is a subset consumer only; it does not
replace the full roster/grid reconciliation.

```bash
"$PYBIN" -m experiments.atomistic.stage1.select_eval \
  --manifest "$RESULTS_ROOT/plan/phase2-manifest.json" \
  --system he --lineage 0 --checkpoint 100000 --stream 0 \
  --output "$RESULTS_ROOT/plan/selected-eval.json" || exit 1
qsub -A HetRxnEnergy -q debug \
  -l select=1:system=polaris:ngpus=4:ncpus=64,walltime=00:55:00,filesystems=home:eagle,place=scatter \
  -o "$RESULTS_ROOT/logs/eval.pbs.out" -e "$RESULTS_ROOT/logs/eval.pbs.err" \
  -v MODE=eval,SOURCE_SHA="$SOURCE_SHA",CHECKOUT_ROOT="$CHECKOUT_ROOT",UV_CACHE_DIR="$UV_CACHE_DIR",RESULTS_ROOT="$RESULTS_ROOT",EXPECTED_QUEUE=debug,PYBIN="$PYBIN",SELECTED_ROW="$RESULTS_ROOT/plan/selected-eval.json",APPROVAL_ID="$APPROVAL_ID" \
  "$RESULTS_ROOT/inputs/pbs_stage1.sh"
```

### 6. Collection on free `debug`

Collection is a CPU-only command inside the same exact-SHA job-local
environment.  It writes only the named collection output beneath Eagle.

```bash
qsub -A HetRxnEnergy -q debug \
  -l select=1:system=polaris:ncpus=4,walltime=00:30:00,filesystems=home:eagle,place=scatter \
  -o "$RESULTS_ROOT/logs/collect.pbs.out" -e "$RESULTS_ROOT/logs/collect.pbs.err" \
  -v MODE=collect,SOURCE_SHA="$SOURCE_SHA",CHECKOUT_ROOT="$CHECKOUT_ROOT",UV_CACHE_DIR="$UV_CACHE_DIR",RESULTS_ROOT="$RESULTS_ROOT",EXPECTED_QUEUE=debug,PYBIN="$PYBIN",COLLECT_MANIFEST="$RESULTS_ROOT/plan/phase2-manifest.json",COLLECT_OUTPUT="$RESULTS_ROOT/collection.json" \
  "$RESULTS_ROOT/inputs/pbs_stage1.sh"
```

Collection is a product/result verdict only if the manifest, row statuses,
checkpoint evidence, per-row assertions, and collection JSON reconcile.  PBS
completion alone is not a scientific result.

## Dry-exec and guard evidence

The wrapper's `guard` mode runs only the queue/job/GPU guards; `dry-exec` runs
the full plan/train control flow with checkout/environment/driver work marked
explicitly skipped.  On Cannon, the S1-P2 verification packet supplies stubs
that replay the captured P1 R2 files under
`experiments/atomistic/stage1/tests/fixtures/polaris_p1_r2/` and runs the
test suite on the free `test` partition.  The fixtures preserve the P1 R2
identity, memory, MIG, UUID, and single-visible-GPU observations from
`/eagle/HetRxnEnergy/rhu/runs/stage1-p1-r2-f92eb901-20261118T145856Z`.

Required guard cases are: the captured four-A100 pass, wrong GPU count,
MIG enabled, a Cannon 80-GB identity, missing GPU output, device count other
than one, wrong queue, and an outside-Eagle results root.  The base-on arm
must include this test file with type checking on.  A mutation arm drops each
of the GPU-count, MIG, single-visible-GPU, and Eagle-prefix checks; every
mutant must be `KILLED` by its named test, with no harness-invalid arm.

## Receipts and stop rules

For every job preserve the qsub command/output, full PBS job ID, qstat
terminal record, queue, account, delivered host, walltime, source SHA,
wrapper SHA-256, lock SHA-256, exact command, environment/import output,
GPU names/memory/MIG/UUIDs, row logs/status files, checkpoint hashes, phase
manifest hashes, collection hash, and a complete named-file inventory.

Do not interpret a scheduler `F` or `Exit_status=0` as a product verdict if
the wrapper did not reach its assertions.  Missing PBS output directories,
failed bootstrap, wrong queue, failed guards, or missing artifacts are
`HARNESS_INVALID`/`UNMEASURED`; retain the evidence and stop.
