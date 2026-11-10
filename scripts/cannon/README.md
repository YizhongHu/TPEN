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

`PYTHON`'s overlay interpreter is also exported as `TPEN_PYTHON`, `ROOT_PYTHON`,
and `P8_EXECUTOR_PYTHON` (required by `experiments/toolkit/test_review_p8_*`),
and `$OVERLAY/bin` is prepended to `PATH` so tests that spawn a bare `python`
subprocess resolve the overlay interpreter.

Use --mutants SPEC for a JSON object or list containing 3-5 rows with id,
target, old, and new. A mutant is KILLED only when the unmutated targeted run
was green in the same job and the mutated run is non-zero. Every mutation is
restored and the clone must be clean before the next one. The resolved spec and
each applied mutant's diff are copied into the evidence directory
(mutant-spec.json, mutant-<id>.diff) before restore, since the spec path and
job scratch are both ephemeral — SURVIVED needs that diff to diagnose without
re-running on the cluster. A SURVIVED verdict with typeguard off is not
necessarily a quick-mode artifact: check the diff against the killer test's
actual assertions first. A mutant that only cancels inside a self-referential
equality (e.g. comparing two calls of the mutated function to each other) will
survive under typeguard-on too; that is a test-coverage gap, not a runner
defect, and is out of this runner's scope to fix.

Use --mode measure --full-suite to record targeted type-checking/parallelism
combinations plus one full-suite timing in the evidence root. All evidence is
written live below RUN_ROOT; throwaway checkout, cache, temporary files,
overlays, and mutant backups are removed from the exact job scratch directory
by the exit trap.
