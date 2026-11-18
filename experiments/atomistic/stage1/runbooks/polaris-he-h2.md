# Stage 1 He/H2 Polaris direct-venue runbook

This is the optional direct PBS route for the He and H2 Stage-1 rows.  It is
independent of the Cannon runbooks and does not authorize a science
allocation.  The implementation is
`experiments/atomistic/stage1/polaris/pbs_stage1.sh`; every submission below
uses that one complete wrapper with an explicit mode.

## Scope and approval boundary

The route is one direct Polaris allocation: three independent 50,000-update
training lineages run concurrently on one four-GPU Polaris node, followed by
strict model-only evaluation rows and collection on Eagle.  It does not use
Balsam, debug-job chaining, or a first-class PBS path in `launch.py`.

- `debug` and `debug-scaling` are free queues for bounded smoke, guard, plan,
  evaluation, and collection work.  The PBS queue name is `debug-scaling`,
  with a hyphen.
- `capacity` is required for the 50,000-update training allocation and needs
  fresh operator approval relayed by the manager.  Do not submit it from this
  document without that approval.
- The wrapper's checked-in PBS header is only a safe debug default.  The
  command lines below state the actual queue and resource shape explicitly.
- Never `qdel`, `qhold`, or `qrls`.  A guard, identity, checkpoint, wall-cap,
  or OOM failure is a terminal invalid measurement for that row; resubmission
  requires a new approval boundary.

Before any Polaris action, read the `cluster-access` skill and the current TO
chain: Clusters `cluster-access-read-first`, `00-orientation`,
`free-submit-queues`, ALCF `provider-and-hpss-orientation-2026-08-07`, and
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
SOURCE_REPO=https://github.com/YizhongHu/TPEN.git
SYSTEM=he                         # repeat the complete sequence with h2
RESULTS_ROOT=/eagle/HetRxnEnergy/rhu/runs/stage1-${SYSTEM}-${SOURCE_SHA}-<utc>
WRAPPER=/eagle/HetRxnEnergy/rhu/runs/stage1-${SYSTEM}-${SOURCE_SHA}-<utc>/inputs/pbs_stage1.sh
PYBIN=/soft/applications/conda/2025-09-25/mconda3/bin/python
```

The run root must be created before any `qsub` so PBS can open its `-o` and
`-e` files.  Assert it is below `/eagle/HetRxnEnergy/rhu/`, create `logs/`,
`plan/`, `launch/`, and `inputs/`, copy the wrapper there, and record its
SHA-256.  Do not copy a git tree with `cp -a` from `/home` to Eagle; the
current Polaris note records that this fails while preserving ownership.

The wrapper clones `SOURCE_REPO` into node-local `/local/scratch/$USER/$PBS_JOBID.stage1-<mode>`, detaches
`SOURCE_SHA`, checks a clean tree, runs the facility interpreter as
`$PYBIN -m uv`, and builds the cu128 environment with `uv sync --offline
--extra cu128 --locked`.  The cache and `TMPDIR` are job-local.  The P1 R2
receipt measured the same lock-bound operation at 25 seconds and proved
Torch `2.11.0+cu128`, four `NVIDIA A100-SXM4-40GB` devices, 40,960 MiB, and
MIG `Disabled`.  Prime the cache in the same login session with the same
`$PYBIN -m uv` binary before a real submission; never use bare `uv`.

## Wrapper boundary

The wrapper reuses the Stage-1 command contract at
`experiments/atomistic/stage1/launch.py:42-80` through
`launch.command_for_row` in `pbs_stage1.sh:203-226`.  That preserves row
identity, `runtime.seed`, results root, run identity, checkpoint binding,
strict restore, and `load.mode=model_only`.

The PBS wrapper replaces only the Slurm execution shell around that contract:

- `launch.py:193-277` remains the Cannon renderer and its Slurm/Netscratch
  `FACILITY_RESULTS_PREFIX` and A7 80-GB guard are untouched;
- `pbs_stage1.sh:54-110` supplies the PBS queue/job/GPU guards for Polaris;
- `pbs_stage1.sh:123-153` supplies detached checkout, job-local cu128
  provisioning, and cleanup;
- `pbs_stage1.sh:203-269` runs three train rows in parallel or eval rows
  sequentially, with `CUDA_VISIBLE_DEVICES` and `taskset` CPU ranges.

The wrapper never silently routes a Polaris job through the Cannon Slurm
template and never edits `launch.py`.

## Exact command sequence

All commands below are run from a Polaris login shell for light preparation,
submission, and monitoring only.  `qstat -xf <jobid>` is the authoritative
terminal scheduler observation.  Preserve qsub stdout/stderr, qstat output,
wrapper logs, row status files, rendered command receipts, and every failure.

### 1. Phase-1 plan on free `debug`

Create the output directory before submitting.  The wrapper records the PBS
job ID and queue, checks all four GPU identities and MIG modes, and runs the
plan through `python -m` from the exact detached checkout.

```bash
mkdir -p "$RESULTS_ROOT"/{inputs,logs,plan,launch}
cp experiments/atomistic/stage1/polaris/pbs_stage1.sh "$RESULTS_ROOT/inputs/pbs_stage1.sh"
chmod 750 "$RESULTS_ROOT/inputs/pbs_stage1.sh"
sha256sum "$RESULTS_ROOT/inputs/pbs_stage1.sh" > "$RESULTS_ROOT/inputs/pbs_stage1.sh.sha256"
qsub -A HetRxnEnergy -q debug \
  -l select=1:system=polaris:ngpus=4:ncpus=64,walltime=00:55:00,filesystems=home:eagle,place=scatter \
  -o "$RESULTS_ROOT/logs/plan.pbs.out" -e "$RESULTS_ROOT/logs/plan.pbs.err" \
  -v MODE=plan,SOURCE_SHA="$SOURCE_SHA",SOURCE_REPO="$SOURCE_REPO",RESULTS_ROOT="$RESULTS_ROOT",EXPECTED_QUEUE=debug,PYBIN="$PYBIN",PLAN_PHASE=1,SYSTEM="$SYSTEM",PLAN_OUTPUT="$RESULTS_ROOT/plan/phase1-manifest.json",CHECKPOINTS=50000,LINEAGE_SEEDS=1:2:3 \
  "$RESULTS_ROOT/inputs/pbs_stage1.sh"
```

Require `qstat -xf` to show the delivered queue `debug`, `Exit_status=0`,
and `run_count=1` before reading the phase-1 manifest.  The plan must contain
one smoke row and three train rows for the selected system, with lineages
`0,1,2`, walkers `4096`, updates `50000`, terminal checkpoint `50000`,
`device=cuda`, and the exact `SOURCE_SHA`.

### 2. Three train rows in one `capacity` allocation

This is the first nonfree action and needs a fresh approval naming the exact
source SHA, Eagle root, queue, one-node/four-GPU shape, and wall cap.  Before
the qsub, verify the `capacity` queue and account live, and verify the `-o`/`-e`
directory exists.

```bash
qsub -A HetRxnEnergy -q capacity \
  -l select=1:system=polaris:ngpus=4:ncpus=64,walltime=11:00:00,filesystems=home:eagle,place=scatter \
  -o "$RESULTS_ROOT/logs/train.pbs.out" -e "$RESULTS_ROOT/logs/train.pbs.err" \
  -v MODE=train,SOURCE_SHA="$SOURCE_SHA",SOURCE_REPO="$SOURCE_REPO",RESULTS_ROOT="$RESULTS_ROOT",EXPECTED_QUEUE=capacity,PYBIN="$PYBIN",PLAN_MANIFEST="$RESULTS_ROOT/plan/phase1-manifest.json" \
  "$RESULTS_ROOT/inputs/pbs_stage1.sh"
```

The wrapper maps lineages `0,1,2` to GPU/CPU ranges `(0,0-15)`, `(1,16-31)`,
and `(2,32-47)`, exports `OMP_NUM_THREADS=16` and `MKL_NUM_THREADS=16`, and
starts all three rows concurrently.  GPU 3 is left unused.  Each row writes
a separate log and `.status` receipt under `RESULTS_ROOT/logs/`; the wrapper
also captures the exact argv returned by `launch.command_for_row`.

### 3. Phase-2 checkpoint binding on free `debug`

After training, build `plan/checkpoint-evidence.json` from the three named run
directories only.  Require `status.json` to say `completed`, require the
complete `checkpoints/step_050000` directory, and record each `model.pt`
SHA-256.  Do not search the Eagle tree recursively.  Then submit:

```bash
qsub -A HetRxnEnergy -q debug \
  -l select=1:system=polaris:ngpus=4:ncpus=64,walltime=00:55:00,filesystems=home:eagle,place=scatter \
  -o "$RESULTS_ROOT/logs/phase2.pbs.out" -e "$RESULTS_ROOT/logs/phase2.pbs.err" \
  -v MODE=plan,SOURCE_SHA="$SOURCE_SHA",SOURCE_REPO="$SOURCE_REPO",RESULTS_ROOT="$RESULTS_ROOT",EXPECTED_QUEUE=debug,PYBIN="$PYBIN",PLAN_PHASE=2,PLAN_OUTPUT="$RESULTS_ROOT/plan/phase2-manifest.json",PHASE1_MANIFEST="$RESULTS_ROOT/plan/phase1-manifest.json",CHECKPOINT_BINDINGS="$RESULTS_ROOT/plan/checkpoint-evidence.json",CHECKPOINTS=50000,LINEAGE_SEEDS=1:2:3 \
  "$RESULTS_ROOT/inputs/pbs_stage1.sh"
```

Require the phase-2 manifest to contain 12 eval rows (four streams for each
lineage), each strictly bound to the phase-1 manifest, checkpoint digest,
absolute `model.pt`, and `device=cuda`.

### 4. Evaluation on free `debug`

The wrapper runs the 12 eval rows sequentially on GPU 0, one fresh strict
`model_only` restore per row.  The debug one-hour limit is suitable for this
bounded evaluation shape; stop on the first nonzero row status.

```bash
qsub -A HetRxnEnergy -q debug \
  -l select=1:system=polaris:ngpus=4:ncpus=64,walltime=00:55:00,filesystems=home:eagle,place=scatter \
  -o "$RESULTS_ROOT/logs/eval.pbs.out" -e "$RESULTS_ROOT/logs/eval.pbs.err" \
  -v MODE=eval,SOURCE_SHA="$SOURCE_SHA",SOURCE_REPO="$SOURCE_REPO",RESULTS_ROOT="$RESULTS_ROOT",EXPECTED_QUEUE=debug,PYBIN="$PYBIN",PLAN_MANIFEST="$RESULTS_ROOT/plan/phase2-manifest.json" \
  "$RESULTS_ROOT/inputs/pbs_stage1.sh"
```

### 5. Collection on free `debug`

Collection is a CPU-only command inside the same exact-SHA job-local
environment.  It writes only the named collection output beneath Eagle.

```bash
qsub -A HetRxnEnergy -q debug \
  -l select=1:system=polaris:ncpus=4,walltime=00:30:00,filesystems=home:eagle,place=scatter \
  -o "$RESULTS_ROOT/logs/collect.pbs.out" -e "$RESULTS_ROOT/logs/collect.pbs.err" \
  -v MODE=collect,SOURCE_SHA="$SOURCE_SHA",SOURCE_REPO="$SOURCE_REPO",RESULTS_ROOT="$RESULTS_ROOT",EXPECTED_QUEUE=debug,PYBIN="$PYBIN",COLLECT_MANIFEST="$RESULTS_ROOT/plan/phase2-manifest.json",COLLECT_OUTPUT="$RESULTS_ROOT/collection.json" \
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
