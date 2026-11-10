# Cannon verification runner

verify.sh is the versioned, facility-neutral entry point for exact-head CPU
verification. The Cannon submission wrapper supplies facility paths and
resources; the repository receives only CLI and environment inputs.

Quick mode requires a Slurm allocation, clones the exact SHA with tags, builds
or reuses the lock-scoped dependency environment under a flock, creates a
job-local overlay with venv --without-pip, adds a .pth for the shared
environment, and installs this checkout editable into the overlay. It runs the
requested whole test paths with pytest -q, type checking disabled by an addopts
override and xdist sized from SLURM_CPUS_PER_TASK.

Use --mutants SPEC for a JSON object or list containing 3-5 rows with id,
target, old, and new. A mutant is KILLED only when the unmutated targeted run
was green in the same job and the mutated run is non-zero. Every mutation is
restored and the clone must be clean before the next one.

Use --mode measure --full-suite to record targeted type-checking/parallelism
combinations plus one full-suite timing in the evidence root. All evidence is
written live below RUN_ROOT; throwaway checkout, cache, temporary files,
overlays, and mutant backups are removed from the exact job scratch directory
by the exit trap.
