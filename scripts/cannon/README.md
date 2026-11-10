# Cannon verification runner

verify.sh is the versioned, facility-neutral entry point for exact-head CPU
verification. The Cannon submission wrapper supplies facility paths and
resources; the repository receives only CLI and environment inputs.

Quick mode requires a Slurm allocation, clones the exact SHA with tags, builds
or reuses the lock-scoped dependency environment under a flock, creates a
job-local overlay with venv --without-pip, adds a .pth for the shared
environment, and installs this checkout editable into the overlay. It runs the
requested whole test paths with pytest -q, type checking disabled by an addopts
override. `--parallel` defaults to `serial`: measurement on three heads showed
xdist gives no speed-up on the small targeted sets this runner is meant for
(~20.8s serial vs ~21.3s xdist with type checking on); pass `--parallel xdist`
only where a measured speed-up justifies it (e.g. the full suite).

`PYTHON`'s overlay interpreter is also exported as `TPEN_PYTHON` and
`ROOT_PYTHON`, and `$OVERLAY/bin` is prepended to `PATH` so tests that spawn a
bare `python` subprocess resolve the overlay interpreter. In addition, every
`*_EXECUTOR_PYTHON` / `*_ROOT_PYTHON` environment variable name referenced
anywhere in the exact-SHA clone's `.py` files (e.g. `P8_EXECUTOR_PYTHON`,
`P9C_EXECUTOR_PYTHON`, `P9C_ROOT_PYTHON`, `P10_ROOT_PYTHON`,
`H_V2_EXECUTOR_PYTHON` as of this writing) is discovered by grep after the
overlay is built and bound to the same job's interpreter; the resolved list is
recorded in `executor-python-bindings.txt`. An inherited or unbound one is
read by these tests as "skip this check", so binding by discovery (not a
hardcoded name) matters whenever a new `*_EXECUTOR_PYTHON`/`*_ROOT_PYTHON`
caller is added.

Use --mutants SPEC for a JSON object or list containing 3-5 rows with id,
target, old, new, and **killer** — the exact pytest node id (or bare id with
parametrizations matched as `killer[...]`) that is expected to fail under the
mutation. A mutant is KILLED only when: the unmutated targeted run is green in
the same job (baseline_rc == 0), the mutated run exits in pytest's normal
{0,1} range, AND the named killer's JUnit outcome (parsed from
`--junit-xml`, never the bare process exit code) actually failed or errored
while having passed cleanly in the baseline. Any other outcome — a harness-level
mutant exit code (collection/usage/internal error, no tests collected, killed
process), a killer missing from either report, or a killer whose
parametrization set changed — is recorded as `HARNESS_ERROR` and fails the job
closed; it is never read as KILLED. Every mutation is restored, `__pycache__`
is purged immediately before the baseline run and before and after every
mutant run (not only after restore, since a same-tick bytecode cache can
otherwise mask an edit), and the clone must be clean before the next mutant.
The resolved spec, each applied mutant's diff, and each mutant's verdict JSON
are copied into the evidence directory (mutant-spec.json, mutant-<id>.diff,
mutant-<id>-verdict.json) before restore, since the spec path and job scratch
are both ephemeral — SURVIVED needs that diff to diagnose without re-running on
the cluster. A SURVIVED verdict with typeguard off is not necessarily a
quick-mode artifact: check the diff against the killer test's actual
assertions first. A mutant that only cancels inside a self-referential
equality (e.g. comparing two calls of the mutated function to each other) will
survive under typeguard-on too; that is a test-coverage gap, not a runner
defect, and is out of this runner's scope to fix.

A small number of tests assert a bare `TypeCheckError` with no
manual-validation fallback (e.g.
`tests/unit/nn/test_readout.py::test_pfaffian_readout_rejects_malformed_kernel_inputs`)
and are marked `@pytest.mark.requires_typeguard`. `tests/conftest.py` skips
these whenever typeguard is not instrumenting `tpen` — which is always true in
quick mode's `--typecheck off` — so they are never silently reported as a
baseline failure or a false SURVIVED/HARNESS_ERROR. **Quick mode does not
exercise these tests; only the full suite (type checking on) does.** The
one-line summary reports the skip count explicitly as
`typeguard_skipped=N`; a nonzero count is the documented gap, not a defect.



Write-destination containment (RUN_ROOT, WORK_ROOT, SHARED_ENV_ROOT, and every
path passed through `assert_write_path`) is checked against canonicalized
roots (`realpath -m`, resolving `..`, `.`, duplicate slashes, and symlinks of
existing ancestors) before any write, so a `..`-traversal or symlinked escape
outside the facility prefix is rejected rather than silently satisfied by a
raw-string prefix match.

Run `scripts/cannon/selftest.sh` to exercise these guarantees locally with
bash/stdlib only — no Slurm, no pytest, no cluster access. Unlike `verify.sh`
(which clones its own exact-SHA tree), `selftest.sh` reads the repo it is
physically sitting inside to find the `*_EXECUTOR_PYTHON`/`*_ROOT_PYTHON`
test family, so it must be run from inside a full repo checkout (a git clone,
or a `git archive`/`git worktree` extraction at the exact head) with
`scripts/cannon/selftest.sh` at its normal repo-relative depth — **not** a
bare file-level stage of just the `scripts/cannon/*` files. Staging only those
files at the same depth elsewhere (e.g. directly under a shared directory)
makes `../..` resolve outside the repo entirely; `selftest.sh` now asserts the
resolved root looks like a real `tpen` checkout (`pyproject.toml` with
`name = "tpen"`) and fails loudly instead of silently reporting an empty,
vacuously-passing F3 section.

Use --mode measure --full-suite to record targeted type-checking/parallelism
combinations plus one full-suite timing in the evidence root. All evidence is
written live below RUN_ROOT; throwaway checkout, cache, temporary files,
overlays, and mutant backups are removed from the exact job scratch directory
by the exit trap.

## Layout

Facility-specific code and its tests live under `scripts/cannon/`, never under
the core `tpen` package or core `tests/` tree: core may provide generic
structure (markers, hooks) but never facility specifics. Tests for this
runner (and any other facility-specific test that previously lived under
`tests/unit/infra`) belong in `scripts/cannon/tests/`, with their own
`conftest.py`/pytest config local to that directory. The root `pyproject.toml`
excludes `scripts` via `norecursedirs`, so a bare `pytest`/`uv run pytest` from
the repository root never collects facility-specific tests; run them
explicitly with `uv run pytest scripts/cannon/tests`.
