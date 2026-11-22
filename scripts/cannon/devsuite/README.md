# TPEN dev full-suite launcher template

This packet is a laptop-side launcher for the 12-hourly Cannon `dev` full
suite. It stages one exact SHA, one root arm, and one executor matrix arm. The
root arm uses the exact SHA's complete `scripts/cannon/` tree and invokes
`verify.sh --mode measure --typecheck on --parallel serial --full-suite` with
`PYTEST_ADDOPTS=--durations=60`. Because the current unmodified `verify.sh`
measure mode also runs four targeted combinations before its full-suite arm,
the wrapper supplies only the small staged `scripts/cannon/tests` directory as
that preflight; it does not accidentally repeat the whole repository four
times. The executor arm uses an explicitly supplied,
versioned `executor-harness-v3` directory with an empty mutant list by
default; the variant must contain both H-v2 interpreter bindings. An explicit
`--executor-mutants /absolute/spec.json` enables the harness's supported
non-empty mutant path and stages that spec without modifying the source
harness.

The current expected wall times are approximately 31 minutes for the root
suite and 12–50 minutes for the executor matrix. The root wrapper requests one
CPU/16 GiB/2 hours; the executor wrapper requests two CPUs/32 GiB/8 hours
because the executor slot fixture uses two concurrent workers.

## Usage

Run the safe rehearsal from a TPEN checkout. The SHA is always a full 40-hex
commit, normally `git rev-parse origin/dev`:

```bash
HARNESS=/Users/yizhonghu/tpen-manager-receipts/executor-harness-v3-release-rc2-ca0f93b3-20261111T092513Z
./submit_devsuite.sh --dry-run --executor-harness "$HARNESS" "$(git rev-parse origin/dev)"
```

For an executor mutation rehearsal, pass the approved spec explicitly:

```bash
./submit_devsuite.sh --dry-run --executor-harness "$HARNESS" \
  --executor-mutants /absolute/path/to/mutants.json "$(git rev-parse origin/dev)"
```

Use `--executor-only` when a root arm is already running and only an executor
baseline is authorized for submission:

```bash
DEVSUITE_ALLOW_SUBMIT=1 ./submit_devsuite.sh --submit --executor-only \
  --executor-harness "$HARNESS" "$(git rev-parse origin/dev)"
```

The dry run performs all of the following on Cannon's login node, without
running tests or allocating compute:

- stages the exact-SHA `scripts/cannon/` tree and the versioned executor
  harness under `/n/netscratch/kozinsky_lab/Lab/rhu/verify/…`;
- reads back and compares a SHA-256 for every staged file;
- runs `bash -n` on both generated wrappers and the executor harness;
- derives the synthetic precheck `SLURM_JOB_NAME` from each wrapper's rendered
  `#SBATCH --job-name` directive;
- checks with `test -x` on Cannon that the root `verify.sh` and inner executor
  `job.sbatch` are executable;
- clones the staged executor bundle on the Cannon login node into an ephemeral
  stage-root directory, verifies the bundle, and asserts the clone HEAD is the
  requested exact SHA; the temporary directory is removed before continuing;
- rejects a line beginning with `SBATCH` unless it begins with `#SBATCH`;
- runs both wrapper precheck functions with job id `99999999`, name containing
  the 12-character SHA prefix, and partition `test`;
- runs the executor harness `DRY_RUN=1` path, which stops before clone, uv,
  pytest, and scheduler work; and
- runs `sbatch --test-only` for both submitted wrappers.

It gates every Cannon operation on `ssh -o BatchMode=yes -O check cannon`. A
failed mux check stops immediately; it never retries with plain SSH. The
launcher never uses `cannon-login` or `cannon-keeper`.

The real branch is deliberately double-gated for later operator use:

```bash
DEVSUITE_ALLOW_SUBMIT=1 ./submit_devsuite.sh --submit \
  --executor-harness /absolute/path/to/approved/executor-harness-v3 \
  "$SHA"
```

Do not use that branch during rehearsal. The current task permits only
`sbatch --test-only`.

## Safety and scope

- No repository files are changed, committed, or pushed by this template.
- Every job name contains `SHA[:12]`; every path is absolute and under the
  Cannon `Lab/rhu/verify` prefix, with node-local `/scratch/$USER/$SLURM_JOB_ID`
  reserved for job work.
- The root arm stages the whole `scripts/cannon/` tree because the current
  `verify.sh` depends on sidecars such as `shared-env-manifest.sh` and
  `shared-env-policy.json`.
- The executor harness is copied as an immutable input. The launcher replaces
  only `repo.bundle` with a bundle containing the target SHA and the required
  historical commit used by the executor's bit-identity test.
- `mutants.json` must contain `"mutants": []` for the default full-suite
  packet. The explicit `--executor-mutants` path records the supplied spec's
  SHA-256 and permits a non-empty mutation run. With `--submit` plus this
  option, only the executor mutation job is submitted.
- The generated executor wrapper owns the actual Slurm `--output` and
  `--error` paths. The inner versioned harness's `/dev/null` directives are
  inert because that file is executed as the job body, not submitted directly.

## Receipt template

Record this in Task Orchestrator note
`devsuite-launcher-template-20261115` on the Stage-1 container, using `body`
and reading it back non-empty:

```text
DEVSUITE TEMPLATE RECEIPT
Template: /Users/yizhonghu/.paseo/stage1-evidence/devsuite-template/
SHA: <full 40-hex>
Executor harness: <absolute versioned path>
Files + sha256:
  <relative path> <sha256>
Dry-run UTC: <timestamp>
Checks: stage/readback PASS; bash -n PASS; bare-SBATCH grep PASS;
  root precheck PASS; executor precheck PASS; executor harness DRY_RUN PASS;
  sbatch --test-only root PASS; sbatch --test-only executor PASS;
  real submission NOT PERFORMED.
Open caveats: <record exact current verify.sh or facility limitation>
```

