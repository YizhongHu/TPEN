# Cannon executor harness v3

## v3 changes

v3 keeps the v2 storage, environment, overlay, submission, path-guard, and
reconciliation contract, but makes mutation execution suite-aware.

- Every mutant has `suite: "root"` or `suite: "executor"`. A killer under
  `executor/tests/` must be `executor`; every other killer must be `root`.
  The loader rejects missing/invalid/mismatched declarations before any run.
- Executor mutants still mutate each executor arm, run that arm's full
  executor suite, and restore the clone byte-for-byte while clearing
  `__pycache__`.
- Root mutants mutate the root clone once and run the killer's complete test
  module plus optional `extra_selectors`, using the root overlay. The edit
  return code and exact old-needle count are recorded; pytest is never run
  after an unsuccessful edit. The root baseline must show the named killer
  passed in the same job, and only a mutant failure/error counts as KILLED;
  skips never kill.
- The legacy `reader_mutant` slot is migrated into the root mutant collection
  at load time, so old specs remain readable. New specs may put it directly in
  `mutants`.
- Root rows set `PYTHONPATH` to the root checkout only. Executor rows retain
  the executor source path. This keeps executor sources out of root processes
  and preserves the project-boundary check.
- `final_verdict2.py` reports each mutant by id and suite. Executor mutants
  have four arm results; root mutants have one root result and use the root
  baseline JUnit/collection receipt.

This directory is a parameterized, one-job full-matrix verifier adapted from
the S2b-1 final2 harness. It is a receipt/staging artifact, not a repository
change. The current hold forbids Cannon writes and submissions; the only
verification performed for this delivery is local syntax, compilation, and the
`DRY_RUN` rehearsal.

Slurm's default stdout/stderr sinks are `/dev/null`; the driver redirects its
own output to the live evidence root before doing work, so an accidental submit
directory cannot receive run artifacts.

## Parameters

- `HEAD_SHA` is the exact 40-character commit checked out from `repo.bundle`.
- `JOB_NAME` is supplied with `--job-name`; it must contain the first 12 SHA
  characters. The Slurm job name and evidence directory therefore identify the
  exact head.
- `MUTANT_SPEC` names the JSON file defining the mutants. Executor killer
  selectors are rooted at `executor/tests/...` in the spec and are normalized
  to `tests/...` when pytest runs from the executor directory. Root selectors
  are rooted at `tests/...` or `experiments/...`. The baseline-green rule is
  enforced by `final_verdict2.py`.
- `TASK_ROOT` is the staged harness input. `RUN_ROOT` is the live evidence
  root. `submission-line.template` shows the complete `--export=NONE,...`
  shape; `submission-line.txt` is a concrete rehearsal line.
- `LOCAL_FAKE=1` plus `LOCAL_FAKE_ROOT` is accepted only by local `DRY_RUN`
  rehearsal tooling; production runs have no local-root exception.

The matrix is four executor arms (`c1-py312-parsl`, `c2-py314-parsl`,
`c3-py312-none`, `c4-py314-none`), the full root suite, three real-core rows,
seven named mutants on every arm, and the toolkit reader mutant. Collection,
JUnit, terminal summaries, skips, subtests, and call outcomes are reconciled
before the native subtest-aware verdict is accepted.

## Storage contract

The job enforces the current Cannon amendment as follows.

- `helpers/validate_inputs.sh` performs fixed-literal root validation before
  `job.sbatch` calls the descendant-aware `assert_prefix`: non-DRY runs require
  `NODE_TMP=/scratch/$USER/$SLURM_JOB_ID` exactly; DRY_RUN requires
  `DRY_NODE_TMP` under `Lab/rhu` unless the explicit local-fake gate is set;
  `HARNESS_ROOT`, `RUN_ROOT`, and `TASK_ROOT` must be under `Lab/rhu`.
- `job.sbatch:28-62` accepts writes only below node-local
  `/scratch/$USER/$SLURM_JOB_ID` or the approved `Lab/rhu` subtree, and validates
  the fixed roots, SHA, job name, empty `PYTHONPATH`, and `COPYFILE_DISABLE=1`
  before writes.
- `job.sbatch:64-82` creates the run root and node-local root, streams driver
  evidence live, and removes only the exact job-owned node-local directory in
  the exit trap. `/tmp` is never selected.
- `job.sbatch:93-97` rejects AppleDouble `._*` files before execution.
- `job.sbatch:153-165` clones the bundle and mutant sources only into node
  local storage.
- `job.sbatch:166-195` hashes the relevant `uv.lock`, builds one shared
  dependency environment per `(lock hash, extras, Python)` under `flock`, uses
  `uv sync --locked --no-install-project`, and then uses the environment
  read-only. A fresh node-local overlay (`venv --without-pip` plus a `.pth` naming the
  shared env's site-packages; `--system-site-packages` would inherit the BASE CPython, not the shared env,
  and failed Cannon job 59347894 with `ModuleNotFoundError: pytest`) receives the exact checkout with
  `uv pip install --no-deps -e`. The install is EDITABLE into the job-local overlay only, so `python -I`
  subprocess tests import the checkout. `make_overlay` asserts imports under normal and `-I` modes and that
  the project module resolves inside the checkout (evidence `<arm>-overlay-identity.txt`). Follows H-v2
  convention r3 (TO 8a67671f). Build caches and `TMPDIR` stay below the node-local
  root.
- `job.sbatch:116-139` performs the DRY_RUN command audit, environment audit,
  interpreter checks, and prefix assertions before exiting without clone, uv,
  pytest, or shared-environment work.
- `helpers/path_guard.py:9-66`, `final_run.py:48-77`,
  `run_pytest.py:39-56`, and `final_verdict2.py:193-195` repeat the prefix
  check before Python-side artifact writes. Evidence is written as each arm
  runs under `Lab/rhu/verify/<JOB_NAME>`.

The one deliberate exception is the local rehearsal helper's temporary copy;
it is local test scaffolding, not a Cannon job path. On Cannon, the job's
default `NODE_TMP` is always the Slurm node-local scratch path.

## Reused-environment protocol

For each arm, hash the lockfile from the exact checkout. Construct
`Lab/rhu/envs/<project>-<uv.lock sha256[:12]>-<extras>-py<ver>/` and a sibling `<dir>.lock` path (manager shared-env convention, 2026-11-09). Hold
the lock while checking for `bin/python` and, only when absent, run the exact
locked dependency sync with a node-local `UV_CACHE_DIR` and `TMPDIR`. Later
arms/jobs wait on the same lock and use the completed environment read-only.
Each job creates a node-local system-site-packages overlay and installs the
exact checkout there with `uv pip install --no-deps`; no shared environment
points at one job's checkout.

## Local checks and next gate

Permitted local checks are:

```text
bash -n job.sbatch
python3 -m py_compile final_run.py final_verdict2.py run_pytest.py helpers/*.py
python3 helpers/negative_storage_checks.py
python3 helpers/rehearse_submit_line.py .
```

The final command parses the literal `--export` list, copies this directory to
a fake local layout, and invokes `job.sbatch` with `DRY_RUN=1`. It must report
the command checks, environment audit, interpreter checks, and prefix asserts
as passing. After the manager clears the hold, the remaining verification is
one login-node `DRY_RUN` using the current exact head and current storage paths;
it must still stop before any clone, uv, pytest, or submission action.

The negative test prints `FAIL` for both gates for each of these cases:
`NODE_TMP=/tmp/x`, `RUN_ROOT=/n/netscratch/kozinsky_lab/Lab/x`, and
`HARNESS_ROOT=/var/x`.

## Submission-line rule (added 2026-11-09 after Cannon job 59344332)

Failures before `exec > driver.log` (input validation, `source helpers/validate_inputs.sh`) leave no
trace: Slurm stdout/stderr are `/dev/null` by design. Job 59344332 died in 1 s that way, most likely
because `TASK_ROOT` pointed at the stage root rather than the harness directory. Therefore:

- The real `--export` line must differ from the DRY_RUN line ONLY in `DRY_RUN=0` and in dropping
  `DRY_NODE_TMP`, `LOCAL_FAKE*` and any fake `SLURM_JOB_ID`. Record the diff of the two lines.
- `TASK_ROOT` is the directory containing `job.sbatch`, `helpers/`, `mutants.json` and `repo.bundle`.
- From the login node, `test -f` every exported path before `sbatch`.
- `RUN_ROOT` for the real job is a new directory, distinct from the DRY_RUN run root.
- Submit with the ABSOLUTE path of the staged `job.sbatch`, e.g. `sbatch ... "$TASK_ROOT/job.sbatch"`, and assert
  `test "$(realpath "$TASK_ROOT/job.sbatch")" = "$TASK_ROOT/job.sbatch"` first. A bare `job.sbatch` resolves
  against the submit cwd: Cannon job 59351097 was submitted from `$HOME` and ran an unrelated
  `/n/home10/rhu/job.sbatch` (an H-v2 script) that cloned the repo into `Lab/rhu/h-v2-r4-59351097`.
- After `sbatch`, confirm `scontrol show job <id> | grep Command=` names the staged harness `job.sbatch`.
- `repo.bundle` must contain `HEAD_SHA` AND every historical commit a test reads with `git show`
  (currently `dc99163e15bdb7f2b4cb482e570ff0e9a0e9a8ec`, used by
  `tests/unit/nn/test_wavefunction_atom_migration_bit_identity.py`; it is not an ancestor of dev heads
  after a squash merge). Build it as: `git tag -f harness-pre-a5 dc99163e...` (local tag, never pushed) then
  `git bundle create repo.bundle <scratch-branch> refs/tags/harness-pre-a5`. `clone_one` fetches all bundle
  refs and asserts every commit in `HARNESS_REQUIRED_COMMITS` exists.
- PATH inside each pytest row is prefixed with the row's overlay `bin/` (tests spawn bare `python`).
- `run_pytest.py` must never import the harness `helpers` package (it runs pytest in-process; Cannon job
  59350515 lost every executor arm to a shadowed `helpers` test package).

## Targeted mode (operator decision 2026-11-10, root note operator-decisions-speedup-20261110)

Per-PR finals no longer run the full root suite. In `mutants.json` set:
- `"root_selectors": ["<dir>", ...]`: whole test directories of the changed root modules plus their direct
  consumers (e.g. `experiments/toolkit`). The root baseline and every root mutant run exactly these
  directories (plus the killer's module if it is outside them). An empty or absent list means the full
  root suite (legacy behaviour).
- `"executor_selectors"`: default `["tests"]` (the whole executor suite) for the four executor arms and
  executor mutants.
- 3-5 contract mutants per PR.
A tree-identical rebased head inherits verification (`git rev-parse <old>^{tree}` == `<new>^{tree}`; record
both SHAs and the tree hash). The full suite runs once on dev per operator merge batch (the manager owns
that) and at qualification. Receipt key per verified head: `tip-verification-<sha8>`.

## Cores (FASRC Job Defense Shield, 2026-11-10)
Rows run serially with plain pytest, but `executor/tests/test_s1_slots.py` pins 2 concurrent slot workers to
distinct permitted CPUs, so `job.sbatch` requests `--cpus-per-task=2` (Cannon notes
`cpu-cores-match-parallelism-jds-20261110` and `cpu-cores-affinity-pinned-fixtures-20261110`; 1 core gave 18
false failures per parsl arm in job 59519375). Earlier v3 jobs requested 4 cores and used about 1. After a job,
run `jobstats <jobid>` and record CPU-Util in the receipt.

## H-v2 interpreter exports (2026-11-11)
The root arm and root rows now also export `H_V2_SCIENCE_PYTHON` (= `ROOT_PYTHON`) and `H_V2_EXECUTOR_PYTHON` (= the c1 py312-parsl executor interpreter, the same one as `P8_EXECUTOR_PYTHON`). `experiments/atomistic/h-v2/test_review_pilot_payloads_r1.py` and `test_review_science_config_r1.py` (added by #637 and #629) require them and ERROR at setup without them. That gave 11 setup errors in dev full-suite job 59600070. This mirrors the earlier `P8_EXECUTOR_PYTHON` fix. Backups: `final_run.py.pre-hv2env-20261111`, `job.sbatch.pre-hv2env-20261111`.
