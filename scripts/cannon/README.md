# Cannon verification runner

verify.sh is the versioned, facility-neutral entry point for exact-head CPU
verification. The Cannon submission wrapper supplies facility paths and
resources; the repository receives only CLI and environment inputs.

Quick mode requires a Slurm allocation, clones the exact SHA with tags,
resolves and validates a pre-built, reuse-only shared dependency environment
under `SHARED_ENV_PREFIX` (see "Shared environment: reuse-only" below), creates
a job-local overlay with venv --without-pip, adds a .pth for the shared
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



Write-destination containment (RUN_ROOT, WORK_ROOT, FACILITY_RUN_PREFIX, and every
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

## Shared environment: reuse-only

`verify.sh` never builds, writes, or mutates a shared uv environment. It takes
`SHARED_ENV_PREFIX` (the exact shared-envs root, e.g. a Holystore envs
directory) and `FACILITY_RUN_PREFIX` (this run's own, disjoint, writable
prefix) and delegates resolution/validation to
`scripts/cannon/shared-env-precheck.sh`, in two steps:

1. **`paths`** (before any write): refuses if `SHARED_ENV_PREFIX` and
   `FACILITY_RUN_PREFIX` are equal or either contains the other, if
   `RUN_ROOT` is not inside `FACILITY_RUN_PREFIX`, if `RUN_ROOT` is inside
   `SHARED_ENV_PREFIX`, or if `WORK_ROOT` (the job's own writable/deleted
   scratch root) is equal to or nested either way with `SHARED_ENV_PREFIX`.
   A shared, reuse-only env root must never overlap a run's own writable
   area, in either direction, including the area the job's `EXIT` trap
   recursively deletes. `SHARED_ENV_PREFIX` is also refused outright if it
   canonicalizes to the filesystem root (`/`), and every containment check
   here uses one root-aware helper so a literal `/` on the other side of a
   comparison can't defeat it either.
2. **`resolve`** (once the clone's exact `uv.lock` sha256 is known): computes
   the expected env path
   `SHARED_ENV_PREFIX/<project>-<sha256(uv.lock)[:12]>-<extra>-py<X.Y>` — the
   same deterministic naming `build-shared-env.sh` uses, via a helper shared
   between the two scripts (`shared-env-manifest.sh`) — and refuses unless
   it is a real directory (not a symlink) whose canonical parent is exactly
   `SHARED_ENV_PREFIX`, carries a `.complete` manifest matching the same
   six-field contract `build-shared-env.sh` writes (not just a matching
   `uv_lock_sha256` line) whose hash equals the clone's full sha256, and has
   an executable `bin/python`. On any refusal the message names the
   missing/invalid path and tells the operator to build it first with
   `build-shared-env.sh`.

On success, the resolved path and a copy of its `.complete` manifest are
recorded into evidence (`shared-env-path.txt`, `shared-env-manifest.txt`); the
job-local overlay's `.pth` then points at that path's `site-packages`, same as
before. `SHARED_ENV_PREFIX` is never added to `verify.sh`'s own write-path
containment roots — this script has no write access there at all. The one
isolated (`-I`) Python invocation that imports from the shared env also
passes `-B`: `-I` ignores `PYTHONDONTWRITEBYTECODE`, so without `-B` a
writable or cold shared env would get `__pycache__`/`.pyc` writes during that
probe that `assert_write_path` cannot see (they are interpreter-internal, not
shell redirects).

`shared-env-manifest.sh` holds the naming (`compute_shared_env_name`) and
manifest-validation (`manifest_is_valid`) logic both `build-shared-env.sh` and
`shared-env-precheck.sh` need; each sources it from its own `SCRIPT_DIR`
(never from the target clone under verification), so the two scripts cannot
drift into accepting or naming environments differently.

## build-shared-env.sh — capped builder for shared immutable uv environments

`build-shared-env.sh` builds, or reuses if already complete, one immutable
shared `uv` environment under a facility env root (all facility paths arrive
by `--envs-root`/`--checkout`, never hardcoded here), named deterministically:

```
<project>-<sha256(uv.lock)[:12]>-<extra>-py<X.Y>
```

```
scripts/cannon/build-shared-env.sh \
  --envs-root /absolute/canonical/facility/env/root \
  --checkout /absolute/path/to/an/exact-SHA/clean/checkout \
  --project tpen --extra cpu --python 3.12 \
  --uv-bin /absolute/path/to/uv
```

`UV_CACHE_DIR` must be set by the caller. The `--checkout` must be a clean
git work tree (its `HEAD` is recorded in the manifest below). `--project` and
`--extra` must each be a single path component (`^[A-Za-z0-9][A-Za-z0-9._+]*$`,
no `/`, no leading `.` so `..` is rejected) — the computed target's canonical
parent is also asserted to equal the canonical `--envs-root` under the lock,
so neither flag, nor a weakened name check, can make the target escape the
counted root.

The committed cap in `scripts/cannon/shared-env-policy.json`
(`MAX_SHARED_ENV_DIRS`) is read only from that file via a whole-document
match (one top-level object, one key, a plain non-leading-zero integer up to
3 digits) in the exact verified checkout, and only from there — no CLI flag,
environment variable, or default can override it. Truncated, duplicated,
nested, or trailing-garbage policy documents refuse rather than having a
substring accepted. Under one root-level `flock`, with pathname expansion
explicitly re-enabled (`set +f`, `GLOBIGNORE` unset, `failglob` disabled) so
an inherited `SHELLOPTS=noglob`/`BASHOPTS=failglob` cannot silently empty the
census or break an empty-root first build, the builder counts **every**
one-level directory entry beneath `--envs-root` (dot-dirs, malformed or
partial names, symlinked directories, even a directory literally named
`*` — not only directories carrying a `.complete` manifest), so a cap
cannot be evaded by an unfinished or oddly-named directory. A directory
census is only ever treated as a glob-expansion failure (rather than a real
entry) when the one and only result is the still-literal, unexpanded `*`
pattern itself. Decision order, all before any `mkdir`/`uv sync`:

1. the target (`<env-name>`) is a symlink -> hard refuse, never reused or
   created through;
2. the target already exists and carries a `.complete` manifest whose
   recorded `uv.lock` sha256 matches the current checkout's (the build
   `HEAD` may differ — dependency identity, not history, governs reuse) ->
   reuse, exit 0 (checked even if the root is otherwise at cap, since the
   target itself is one of the counted entries);
3. the target exists without a valid `.complete` -> hard refuse: partial or
   invalid build, needs an explicit disposition, never auto-cleaned;
4. the one-level directory count is at or above the committed cap -> hard
   refuse, printing the observed count, the requested path, the cap, the
   policy file path, and the checkout's exact `HEAD`;
5. otherwise, `mkdir`, then `UV_PROJECT_ENVIRONMENT=<env> uv sync --locked
   --no-install-project --extra <extra> --python <X.Y>` under the checkout,
   then query `<env>/bin/python --version`, `uv --version`, and the UTC
   build time into checked variables (any query failure refuses and leaves
   the build partial, counted, undeleted — never masked by an otherwise
   successful write), and write a `.complete` manifest — via a temp file in
   the same directory, strictly re-validated (exactly six fields, in order,
   each format-checked — a 40-hex `head`, a 64-hex `uv_lock_sha256` equal to
   the checkout's, non-empty `uv_version`, `X.Y.Z` `python_version`, `X.Y`
   `python_request`, full `YYYY-MM-DDTHH:MM:SSZ` `built_utc`) before
   publishing by one same-directory atomic `mv`, so a late-truncated or
   malformed write is never published. The same strict check governs reuse:
   a producer/consumer manifest truncated even past every nominal field is
   rejected, not just an empty or early-truncated one.

Raising the cap requires a reviewed PR that changes the committed policy
file and its hard-refusal test together, visible in Git/GitHub — there is
no automatic GC, override flag, or separate environment-management project.
This builder only builds/reuses one environment per invocation; it does not
itself decide which jobs consume which shared env, migrate consumers, or
dispose of any existing environment (see the Holystore envs plan,
Task Orchestrator item `05803194`, for those later slices).

## submit-lint.sh — pre-submit submission-contract linter

`submit-lint.sh` is a generic, facility-neutral linter run on the generated
sbatch script and manifest before a submission wrapper calls `sbatch`, so a
recurring staging/argument mistake fails before an allocation is consumed. It
never runs `verify.sh`'s own semantics (exact-SHA clone, mutant testing,
JUnit verdicts) — it only validates the generated sbatch script and manifest
that a wrapper is about to submit. All facility paths (stage root, shared-env
root, `$HOME`) arrive by CLI; the script hardcodes none.

**Soft requirement (operator ruling `operator-ruling-q1-lint-20261110` on TO
item `ef480b65`, after #639 merged):** a submitting wrapper runs
`submit-lint.sh` on the generated sbatch script and manifest, records the
result (`LINT_PASS`/`LINT_FAIL`) in its receipt, and may still submit on a
`LINT_FAIL` only by recording an explicit bypass reason alongside that
result. The linter itself never softens its own verdict — "soft" describes
whether the *caller* blocks on `LINT_FAIL`, not what the linter reports. A
formal mandatory-vs-soft decision is deferred to an operator review 3 days
after the #639 merge (trial evidence: TO note `linter-trial-log` on
`ef480b65`). `scripts/cannon/verify.sh` (this directory's sole exact-SHA
executor) does not itself call `sbatch` — it runs inside an already-granted
Slurm allocation (`SLURM_JOB_ID` is a required precondition) — so it is not
the call site for this linter; the call site is whatever Cannon submission
wrapper generates the sbatch script and manifest and invokes `sbatch` on
them. No such wrapper exists in this repo as of this writing; this section
documents the call contract that wrapper is expected to use.

```
scripts/cannon/submit-lint.sh \
  --sbatch /absolute/canonical/path/to/generated.sbatch \
  --manifest /absolute/path/to/generated.manifest \
  --allowed-root <facility stage/run prefix> [--allowed-root <prefix> ...] \
  --shared-env-root <facility shared-env root> \
  [--home "$HOME"] \
  [--submit-helper /path/to/the/sbatch-invoking/helper] \
  [--staging-root <local staging prefix>=<final facility prefix> ...] \
  [--real-env /path/to/real-run.env --dry-env /path/to/dry-run.env \
   --dry-only-keys DRY_RUN]
```

`--staging-root LOCAL=FINAL` declares that a path canonically under `LOCAL`
is, for containment purposes only, the same path it will be once staged
(e.g. rsynced) under `FINAL`: checks 3, 5 and 6 substitute the `LOCAL`
prefix for `FINAL` and canonicalize the WHOLE mapped destination (so a
symlink such as `FINAL/link -> $HOME` that exists where the linter runs is
resolved; a symlink that exists only on the remote side cannot be seen)
before testing `$HOME` and containment against `--allowed-root`, so a
sbatch script linted in place at its local staging location (before rsync to
the facility) is validated as if it already sat at its eventual facility
path. The substitution never widens scope: a mapped path still must land
inside a declared `--allowed-root`, and a path outside every declared
`LOCAL` prefix is checked unchanged. Omit this flag entirely when linting a
sbatch script already staged at its final root (the common case); use it
only for the local-staging-then-rsync workflow.

The manifest is a `KEY=VALUE` file the wrapper generates alongside the
sbatch script, carrying the same critical values it exported into the
script: `HEAD_SHA`, `JOB_NAME`, `TASK_ROOT`, `RUN_ROOT`, `DRY_RUN`, `PYTHON`.
Checks performed (exit 0 on `LINT_PASS`, non-zero on `LINT_FAIL`):

| # | Check |
|---|---|
| 1 | `HEAD_SHA` is exactly 40 hex characters |
| 2 | `JOB_NAME` contains `HEAD_SHA[:12]` |
| 3 | sbatch path is absolute, canonical, and inside a declared `--allowed-root` |
| 4 | critical exports (`HEAD_SHA`, `JOB_NAME`, `TASK_ROOT`, `RUN_ROOT`, `DRY_RUN`, `PYTHON`) match exactly between the manifest and the job's effective exported environment — no stale, missing, or extra values; `HEAD_SHA`/`JOB_NAME`/`TASK_ROOT`/`RUN_ROOT`/`DRY_RUN` must additionally be present and nonempty in both files (a field absent from both passes the match test but is still rejected), and `PYTHON` is required the same way whenever `DRY_RUN=1`. The effective environment is computed by a deliberately narrow shell subset (documented in full at the top of the parser in `submit-lint.sh`), never by running anything and never from the manifest: it is SEEDED from the submit helper's own `sbatch ... --export=SPEC <job>` line when `--submit-helper` has one (a command-line option overrides the directive, as in Slurm), else from the last `#SBATCH --export=SPEC` directive in the leading comment block; SPEC must be literal `NAME=VALUE,...` (a bare `NAME` there is unknown). Then `export` and plain `NAME=VALUE` statements apply in file order: a bare `export NAME` keeps NAME's CURRENT value; `$NAME`/`${NAME}`/`${NAME:-DEFAULT}` expand against the current value; single quotes are literal; an unquoted `#` starts a comment; a `NAME=VALUE cmd` prefix does not change the job environment. Unsupported syntax (command substitution, other expansions, unquoted globs/tilde/operators, an unterminated quote, `unset`/`declare` on a critical key) is a LINT_FAIL when the statement could set a critical key, and a critical key exported with a value the linter cannot determine is a LINT_FAIL; it never guesses. Not modelled: control flow, functions, heredoc bodies, `source` |
| 5 | `TASK_ROOT`/`RUN_ROOT` are not staged under `$HOME`, literally or after canonicalization (catches a symlink into `$HOME`) |
| 6 | `TASK_ROOT`/`RUN_ROOT` stay inside a declared `--allowed-root` (facility-root escape) |
| 7 | when `DRY_RUN=1`, `PYTHON` resolves under `--shared-env-root` |
| 8 | every `ssh`/`scp`/`rsync` invocation in `--submit-helper` passes `-o BatchMode=yes` (for `rsync`, via `-e "ssh -o BatchMode=yes"`), checked per logical invocation (backslash continuations joined, comments dropped, split on `;`/`&&`/`\|\|`/`\|`) rather than per physical line or whole-file substring; the command is matched as a WORD anywhere in the segment (`timeout 30 ssh ...`, `x=$(ssh ...)`, `rsync -e ssh ...`), not only when the segment starts with it, while still excluding substrings like `sshd_config`/`rsync_log` |
| 9 | with `--real-env`/`--dry-env`, every exported key matches between a real and a dry run except the `--dry-only-keys` list; a key present on only one side is a delta even if the other side's value happens to look like an "absent" placeholder |

`--real-env` and `--dry-env` must be given together or not at all; supplying
only one is `LINT_FAIL`. The linter also probes up front that `realpath`
supports GNU's `-m` flag (resolve a path that need not exist) and fails
clearly, naming the GNU coreutils requirement, if it doesn't — a BSD/macOS
`realpath` accepts no such option.

Run `scripts/cannon/submit-lint-selftest.sh` to exercise all nine checks
locally with bash/stdlib fixtures only — no Slurm, no pytest, no cluster
access. `scripts/cannon/tests/test_cannon_submit_lint.py` is the pytest-run
contract for the same checks, with its own local `conftest.py`/pytest config
(see Layout above), run on Cannon by the verification runner (not locally,
since it uses the repo's `uv`-managed interpreter) via
`uv run pytest scripts/cannon/tests`; its directory also carries
`mutant-spec.json`, deliberate linter mutants (including
`accept-relative-sbatch`, `omit-job-sha-check`, `allow-home-staging`,
`ignore-sbatch-export-directive`, and `ignore-multiassign-self-default`)
for `verify.sh --mutants`.
