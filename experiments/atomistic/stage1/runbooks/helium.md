# Stage 1 helium science benchmark runbook (Cannon, r2)

Status: reviewed-runbook candidate.  Do not submit from a `dev` commit.  The
science source is one released `main` commit, `SOURCE_SHA`, that contains this
reviewed file, #537 (`4f8ebb0f`), and the Stage-1 launch layers S1-A6 #742
(`1c15b144`), S1-A7 #746 (`59e57433`), S1-A8 #747 (`4ec84737`) and S1-A9 #750
(`a5ad17d7`).  Until that release exists `SOURCE_SHA` is a placeholder; the
expected release is v0.5.4.  The runner records `SOURCE_SHA` and this file's
blob SHA at `SOURCE_SHA` in the submission receipt.

Facility: FASRC Cannon (operator decision 2026-11-18, TO note
`operator-he-facility-cannon-20261118` on c6fc9dc3).  Deadline: 2026-11-24.

## Scientific question and fixed analysis

For the fixed two-electron helium Hamiltonian (one `Z=2` nucleus at the origin,
one up and one down electron), what energy and local-energy variance are
obtained by three independently seeded 50,000-update TPEN lineages when each
terminal checkpoint is evaluated by four fresh model-only streams?

The reference is `-2.903724377034119598 Ha`, the infinite-nuclear-mass,
nonrelativistic value in `experiments/baselines/systems.yaml` (`he_atom`),
transcribed there from Aznabaev, Bekbaev and Korobov, *Nonrelativistic energy
levels of helium atom*, arXiv:1810.11288, Table 3.  It is a comparison target,
not a training input.  The planned statistic for each lineage is the mean of
the four stream means at update 50,000, with their spread reported.  The
collector's draw-axis uncertainty assumes retained draws are independent after
the configured burn-in/stride; it is not a replacement for between-lineage
variation.

This run can show only the behavior of this model, optimizer, finite update
budget, and configured sampler.  It cannot prove convergence to the exact
energy, an architecture-wide claim, or absence of autocorrelation/systematic
bias.  A non-finite row, incomplete grid, wrong identity, or fewer than four
streams is evidence of an invalid measurement, not a scientific result.

## Contract and selection rule

- Train exactly lineages `0,1,2` with frozen lineage seeds `1 2 3` (the
  planner maps each to a distinct Torch seed), 4,096 walkers, 50,000 updates,
  and one checkpoint at `50000`.
- Select only the terminal checkpoint, mechanically: a lineage is selected iff
  its run `status.json` says `completed` and
  `checkpoints/step_050000/` is a complete checkpoint (`COMPLETE` marker and
  `manifest.json`) whose `model.pt` SHA-256 is recorded.  No energy-based or
  visually chosen checkpoint is allowed.
- Run streams `0,1,2,3` separately for every selected checkpoint: each is a
  fresh 4,096-walker, 256-draw strict `model_only` restore.  This is 12 eval
  rows and 12,582,912 retained local-energy rows across the experiment.
- Collection rejects any source/config/checkpoint/physical-system identity
  mismatch, non-finite value, incomplete `4096 x 256` grid, non-checkpoint
  provenance, a phase-2 row not bound to its phase-1 train row, or an
  incomplete four-stream set.  Preserve all logs, manifests, rows, rendered
  scripts, status/config snapshots, records, and the collection JSON.

## Identities and paths

All paths below are fixed by `SOURCE_SHA`; set them once and never edit them
in place.  `SOURCE_SHA` is the full 40-hex released main SHA.

```bash
SOURCE_SHA=<released-main-sha>          # placeholder until v0.5.4 is released
SYSTEM=he
RESULTS_ROOT=/n/netscratch/kozinsky_lab/Lab/rhu/stage1-$SYSTEM/$SOURCE_SHA
BUNDLE=$RESULTS_ROOT/inputs/TPEN-$SOURCE_SHA.bundle
ENV_PREFIX=/n/holystore01/LABS/kozinsky_lab/Lab/User/rhu/envs
LOCK12=<first 12 hex of sha256(uv.lock at SOURCE_SHA)>
SHARED_ENV=$ENV_PREFIX/tpen-$LOCK12-cu128-py3.12
UV_BIN=/n/home10/rhu/.local/bin/uv
```

- `RESULTS_ROOT` is the only external write root.  Every script asserts the
  prefix `/n/netscratch/kozinsky_lab/Lab/rhu/` before any write (the rendered
  launcher already refuses any other root).  Layout under it:
  `inputs/` (bundle), `plan/` (manifests, evidence, row JSON), `jobs/`
  (submitted wrappers), `launch/` (in-job rendered scripts), `logs/` (Slurm
  stdout/stderr), `stage1/he/<run_id>/` (TPEN run directories, written by the
  `nested` artifact layout), and `collection.json`.
- The shared CUDA env is reused read-only; it is never built in a job.  The
  env name must equal `tpen-<sha256(uv.lock)[:12]>-cu128-py3.12` for the
  `uv.lock` at `SOURCE_SHA`.  `tpen-e6110e5b3363-cu128-py3.12` matches v0.5.3
  and current `dev`.  A release version bump edits the `tpen` version line in
  `uv.lock` (v0.5.3 did, commit `0cc1de6d`), so v0.5.4 is expected to need a
  new env name; building it is an orchestrator task, not a runner task.
- Per-job work is on node-local scratch only.  A wrapper job uses
  `BOOT_ROOT=/scratch/$USER/${SLURM_JOB_ID}.stage1-he` for the exact-SHA
  checkout, its overlay venv, `TMPDIR` and `UV_CACHE_DIR`, removed by an EXIT
  trap.  The rendered launcher separately creates and removes
  `/scratch/$USER/$SLURM_JOB_ID`; the wrapper must not create that path.

## Resources and measured cost

Target: one full A100-80GB per GPU job, partitions
`seas_gpu,kozinsky_gpu,gpu`, `--gres=gpu:1 --constraint=a100`.  On these three
partitions every `a100` node is `nvidia_a100-sxm4-80gb` (sinfo check
2026-11-16, TO note `hooke2-30k-gpu-stratum-check-20261116`); the in-job guard
still asserts it.  These are not free-submit queues: every submission needs a
per-run operator approval relayed by the manager (request draft:
TO note `he-cannon-gpu-approval-request-20261118` on c6fc9dc3).

Measured on Cannon `gpu_test` (A100 MIG 3g.20gb slice, dev `1c15b144`; TO
item 6f4c431e notes `gpu-calibration-he-result-20261118` and
`eval-calibration-he-result-20261118`):

- train: median 0.425 s/update on the MIG slice (p90 0.480); full-A100
  estimate about 0.39 s/update (0.925 slice-to-full factor from one H-v2
  reference pair, so treat it as approximate);
- eval: 187 s per 4,096 x 256 stream on the MIG slice (0.052 GPU-h), plus
  about 26 s model build/restore; full-A100 estimate about 0.048 GPU-h.

| Item | Jobs | Expected per job | Wall cap per job | Expected GPU-h | GPU-h at cap |
| --- | --- | --- | --- | --- | --- |
| Train lineage | 3 | 5.4 h (5.9 h at MIG rate, 6.7 h at MIG p90) | `12:00:00` | 16.3 | 36.0 |
| Eval, 12 streams in one job | 1 | about 50 min | `02:00:00` | 0.7 | 2.0 |
| Total | 4 | | | 17.0; with x1.2 margin about 20 (22 at MIG rate) | 38.0 |

The three train jobs run in parallel, so training wall time is about 5.4 h
after the jobs start.  Queue wait dominates: earlier `sbatch --test-only`
start estimates on these partitions were days out, so submit the train jobs
as soon as the source and approval gates pass.

Resources per GPU job: `--cpus-per-task=4 --mem=32G` (calibration peak GPU
memory was 1.4 to 1.7 GiB; host memory has not been measured, so 32G is a
conservative request, not a measurement).

Eval batching.  Default: one eval job runs the 12 rendered stream scripts
sequentially (about 4 min each), stopping at the first failure.  Alternative
if queue placement favours short jobs: three jobs, one per lineage, each
running its four streams, cap `01:00:00`.  Each stream is its own run
directory and its own rendered script either way; the choice changes only the
Slurm job count, and the approval must name the chosen shape.

## Gates before any GPU submission

Abort before submission unless all of these hold, each with a receipt:

1. `SOURCE_SHA` is on `origin/main`, contains the commits listed in the
   status paragraph and this reviewed file, and `git rev-parse` of the bundle
   checkout equals it.
2. The shared env exists at `$SHARED_ENV` for `LOCK12` computed from
   `uv.lock` at `SOURCE_SHA`, and `uv lock --check` passes in the job.
3. The manager has relayed the operator's per-run approval for the exact
   shape in this file (partitions, GPU, caps, job count).
4. A Netscratch quota/access check for `Lab/rhu` is satisfactory.
5. The staging job (step 1) passed, including the dry-execute of the train
   wrapper body with the GPU steps stubbed.

## Command sequence

Every Python step runs inside an allocation from the exact-SHA checkout with
the overlay venv, never on a login node.  Every cluster `ssh`/`rsync` uses
`-o BatchMode=yes`; gate scripted access on `ssh -O check cannon`.  Never use
`python -m tpen.run`.

### Step 0: stage the source bundle (local, then rsync)

```bash
git bundle create TPEN-$SOURCE_SHA.bundle $SOURCE_SHA
rsync -e 'ssh -o BatchMode=yes' TPEN-$SOURCE_SHA.bundle cannon:$BUNDLE
```

Record the bundle SHA-256 on both sides.

### Step 1: staging job on `test` (phase-1 plan, row files, dry-execute)

A 1-CPU `test` job (free queue, `--time=00:30:00`) does the following inside
`BOOT_ROOT`:

```bash
git clone --no-checkout "$BUNDLE" "$BOOT_ROOT/checkout"
git -C "$BOOT_ROOT/checkout" checkout --detach "$SOURCE_SHA"
[[ "$(git -C "$BOOT_ROOT/checkout" rev-parse HEAD)" == "$SOURCE_SHA" ]] || exit 2
"$UV_BIN" lock --check --directory "$BOOT_ROOT/checkout" >/dev/null || exit 2
# overlay: shared env site-packages + editable tpen from this checkout only
"$SHARED_ENV/bin/python" -m venv --without-pip "$BOOT_ROOT/overlay"
SITE_PACKAGES="$SHARED_ENV/lib/python3.12/site-packages"
[[ -d "$SITE_PACKAGES" ]] || exit 2
printf '%s\n' "$SITE_PACKAGES" > "$BOOT_ROOT/overlay/lib/python3.12/site-packages/tpen-shared.pth"
"$UV_BIN" pip install --python "$BOOT_ROOT/overlay/bin/python" \
  --no-deps --no-build-isolation -e "$BOOT_ROOT/checkout"
PY=$BOOT_ROOT/overlay/bin/python
cd "$BOOT_ROOT/checkout"

"$PY" experiments/atomistic/stage1/plan.py --phase 1 --systems he --device cuda \
  --output "$RESULTS_ROOT/plan/phase1-manifest.json" \
  --source-sha "$SOURCE_SHA" --results-root "$RESULTS_ROOT" \
  --checkpoints 50000 --lineage-seeds 1 2 3
```

Phase 1 computes the `stage1-config-v1` config identity from this checkout
(no `--config-sha256`, no config-root argument) and writes one smoke row and
three train rows, each with `device: cuda`.  The smoke row is not launched
under this runbook: its only shape on these partitions is a production GPU
job, and the train rows run the same config, launch path and guard.  Then:

- write each train row to `$RESULTS_ROOT/plan/rows/<row_id>.json` (exact JSON
  object from the manifest, nothing added);
- assert three train rows, lineages `0,1,2`, `updates == 50000`,
  `walkers == 4096`, `checkpoint_steps == [50000]`, `device == "cuda"`, one
  `config_sha256`, `config_identity_rule == "stage1-config-v1"`, and
  `source_sha == SOURCE_SHA`;
- render each train row once here (the step-2 command with this job's
  `BOOT_ROOT`) and record the `#SBATCH` block; it must be exactly
  `--partition=seas_gpu,kozinsky_gpu,gpu --nodes=1 --ntasks=1
  --cpus-per-task=4 --mem=32G --time=12:00:00 --gres=gpu:1 --constraint=a100
  --no-requeue`;
- dry-execute the full train wrapper body (step 2) with only the GPU guard
  and driver run stubbed, so a literal `\${VAR}` or a wrong path fails here
  rather than after a multi-day queue wait (`bash -n` and
  `sbatch --test-only` alone do not catch that);
- run both drivers with `--help` and no `PYTHONPATH`.

Record the phase-1 `manifest_hash`, the manifest file SHA-256, the three row
IDs, and the staging job id.

### Step 2: launch the three train rows

One wrapper per train row, `jobs/train-lineage-<n>.sbatch`.  Its `#SBATCH`
block is the rendered block recorded in step 1 plus only `--job-name`,
`--output=$RESULTS_ROOT/logs/%x-%j.out` and
`--error=$RESULTS_ROOT/logs/%x-%j.err`.  Its body:

1. assert `RESULTS_ROOT` prefix, create `BOOT_ROOT`, set `TMPDIR` and
   `UV_CACHE_DIR` under it, `trap 'rm -rf -- "$BOOT_ROOT"' EXIT`;
2. clone, check SHA, `uv lock --check`, build the overlay (as in step 1);
3. render the row from this checkout:

   ```bash
   "$PY" "$BOOT_ROOT/checkout/experiments/atomistic/stage1/launch.py" \
     --output "$BOOT_ROOT/train.sbatch" \
     --row "$RESULTS_ROOT/plan/rows/<row_id>.json" \
     --driver experiments/atomistic/stage1/train.py \
     --checkout-root "$BOOT_ROOT/checkout" --python "$PY" \
     --head-sha "$SOURCE_SHA" --results-root "$RESULTS_ROOT" \
     --partition seas_gpu,kozinsky_gpu,gpu \
     --cpus-per-task 4 --mem 32G --time 12:00:00
   ```

4. assert the rendered `#SBATCH` lines equal this wrapper's own (read `$0`,
   ignoring `--job-name/--output/--error`), assert no `PYTHONPATH=` in it,
   copy it to `$RESULTS_ROOT/launch/<row_id>.$SLURM_JOB_ID.sbatch`, record its
   SHA-256 in the log;
5. `bash "$BOOT_ROOT/train.sbatch"`; exit with its status.

Submit the three wrappers once each, only after all gates pass:

```bash
ssh -o BatchMode=yes cannon sbatch "$RESULTS_ROOT/jobs/train-lineage-0.sbatch"   # and -1, -2
```

The rendered script (S1-A7 #746) enforces, inside the allocation and before
the driver starts: a Slurm job id; the delivered partition is one of the
requested three; exactly one management GPU whose name contains `A100`, is not
a MIG device, and has 81920 MiB; MIG mode `Disabled`;
`torch.cuda.device_count() == 1`; and the results-root prefix.  The driver
then validates the base64 row binding against its config and overrides
(`runtime.device=cuda`, seed, run identity, walkers, updates,
`checkpoint_steps=[50000]`).  Any failure prints a `STAGE1_GUARD` line and
exits 2.

### Step 3: checkpoint evidence document

After all three train jobs end, a 1-CPU `test` job with the same checkout and
overlay builds `plan/checkpoint-evidence.json`.  It reads only the three
named run directories (no recursive walk of Netscratch):

```python
import hashlib, json, sys
from pathlib import Path
from experiments.atomistic.stage1.protocol import CHECKPOINT_EVIDENCE_SCHEMA, manifest_hash
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.checkpoint.artifact import require_complete_checkpoint_dir

phase1 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert phase1["phase"] == 1 and phase1["manifest_hash"] == manifest_hash(phase1["rows"])
train = [row for row in phase1["rows"] if row["kind"] == "train"]
assert sorted(row["lineage"] for row in train) == [0, 1, 2]
checkpoints = {}
for row in train:
    run_dir = resolve_run_dir(row)
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed", (run_dir, status["status"])
    step_dir = require_complete_checkpoint_dir(run_dir / "checkpoints" / "step_050000")
    digest = hashlib.sha256((step_dir / "model.pt").read_bytes()).hexdigest()
    checkpoints[f"{row['system']}:{row['lineage']}:50000"] = digest
evidence = {
    "schema": CHECKPOINT_EVIDENCE_SCHEMA,  # "stage1-checkpoint-evidence/v1"
    "phase1_manifest_hash": phase1["manifest_hash"],
    "phase1_train_row_hashes": {
        f"{row['system']}:{row['lineage']}": manifest_hash([row]) for row in train
    },
    "checkpoints": checkpoints,
}
Path(sys.argv[2]).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
```

Run it as `"$PY" build_evidence.py "$RESULTS_ROOT/plan/phase1-manifest.json"
"$RESULTS_ROOT/plan/checkpoint-evidence.json"` with the script staged under
`$RESULTS_ROOT/plan/`.  If `status.json` is not at the top of a run directory,
stop and report; do not search for it.  A lineage that fails this step is not
selected, and this runbook then has no complete science grid: stop.

### Step 4: phase-2 plan (same staging job as step 3)

```bash
"$PY" experiments/atomistic/stage1/plan.py --phase 2 \
  --phase1-manifest "$RESULTS_ROOT/plan/phase1-manifest.json" \
  --checkpoint-bindings "$RESULTS_ROOT/plan/checkpoint-evidence.json" \
  --output "$RESULTS_ROOT/plan/phase2-manifest.json" \
  --source-sha "$SOURCE_SHA" --results-root "$RESULTS_ROOT" \
  --checkpoints 50000 --lineage-seeds 1 2 3
```

Phase 2 recomputes the config identity from the checkout and refuses a
mismatch, a wrong phase-1 hash, a train-row hash mismatch, or a missing
digest.  `--device` is omitted: eval rows inherit `cuda` from phase 1.
`--config-sha256` is omitted (it is only a legacy comparison value).
`--results-root` and `--lineage-seeds` are required by the CLI but the
phase-2 path takes both from the phase-1 rows.  Assert 16 rows (1 smoke,
3 train, 12 eval), every eval row carries `phase1_manifest_hash`,
`phase1_train_row_hash`, `checkpoint_sha256` equal to the evidence digest,
`load`-target paths under `$RESULTS_ROOT/stage1/he/train-lineage-<n>/
checkpoints/step_050000`, and `device == "cuda"`.  Write each eval row to
`plan/rows/<row_id>.json`.  Record the phase-2 `manifest_hash`.

### Step 5: launch eval rows

Same wrapper pattern as step 2 with `--driver experiments/atomistic/stage1/
eval.py`, `--time` set to the approved per-job cap, and a loop over the
stream rows in the job: render each to `$BOOT_ROOT/eval-<row_id>.sbatch`,
copy it to `launch/`, run it with `bash`, and stop at the first nonzero exit.
Each rendered script creates and removes `/scratch/$USER/$SLURM_JOB_ID` in
turn, so sequential streams in one job do not collide.  The eval driver
refuses a checkpoint whose SHA-256 differs from `--expected-checkpoint-sha256`
and restores with `load.mode=model_only`, `load.strict=true`.

### Step 6: collect (1-CPU `test` job)

```bash
"$PY" -c 'import sys; from experiments.atomistic.stage1.collect import collect; collect(sys.argv[1], output=sys.argv[2])' \
  "$RESULTS_ROOT/plan/phase2-manifest.json" "$RESULTS_ROOT/collection.json"
```

The collector (S1-A9 #750 statistics) verifies the manifest hash, the
phase-1/phase-2 binding, config identity, physical-system ID, checkpoint
provenance, and finite complete grids, and refuses on any mismatch.  It does
not change the manifest.

## Receipts

On the run slice, per step: job id, `sbatch` response, delivered partition and
node, the `STAGE1_GUARD nvidia-smi=` line, start/end time and exit state
(`sacct`), `SOURCE_SHA`, this runbook's blob SHA, bundle SHA-256, env name,
phase-1 and phase-2 `manifest_hash`, config SHA-256, the three checkpoint
digests, rendered-script SHA-256 per row, run `status.json` and
`run-outcome.json` per row, collection SHA-256, and an explicit file inventory
of the named directories.  Scheduler completion alone is not a result.

## Abort and stop criteria

- Stop before submission if any gate is unmet.  A `dev` SHA is never a
  science source.
- Stop a row, without retry or reshape, on any `STAGE1_GUARD` exit, identity
  or binding refusal, checkpoint-digest mismatch, or out-of-memory; report the
  receipt to the orchestrator.  A resubmission needs a new manager-relayed
  approval naming it.
- A train job that hits its wall cap is a failed lineage, not a partial
  result; there is no resume path in this runbook.
- Never `scancel` a job without explicit operator approval.  Never delete or
  overwrite anything under `RESULTS_ROOT`; preserve failure evidence.
- If any lineage is not selected or any stream fails, collection is either
  refused or incomplete: report it as an invalid measurement, not a result.

## Resolved implementation blockers (r1 draft)

1. He-only phase-1 planning before checkpoints exist: resolved by S1-A8 #747
   (`--phase 1 --systems he`, phase-2 evidence binding).
2. Production A100 render shape: resolved by S1-A7 #746 (GPU partitions,
   explicit cpus/mem/time, `--gres=gpu:1 --constraint=a100`, in-job guard).
3. GPU device policy bound to rows: resolved by S1-A8 #747 (`--device cuda`
   row field) and S1-A7 #746 (renderer requires `cuda` on GPU partitions;
   override `runtime.device=cuda`).
4. Versioned multi-file config identity: resolved by S1-A8 #747
   (`stage1-config-v1`, computed by the planner).
5. Rendered drivers bound to the staged checkout: resolved by S1-A6 #742
   (`--checkout-root` required).
6. Collector statistics: resolved by S1-A9 #750.

All six are merged on `dev`; none is on `main` until the v0.5.4 release.
