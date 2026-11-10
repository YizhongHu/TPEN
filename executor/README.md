# tpen-executor

`tpen-executor` is the independent project boundary for TPEN execution
orchestration.  Its staged Hydra configuration preserves future-process
programs as unresolved specifications; owner processes materialize those
programs with their local environment.

Runtime dependencies are `hydra-core==1.3.2` and `omegaconf==2.3.0`.  The
development dependency is the same pinned pytest release used by the root
project.

The worker launches the selected science process through the `-c` shim
`<prepared-python> -c 'import sys; from tpen.run import main; raise SystemExit(main(sys.argv[1:]))' --config <path> [dotlist overrides...]`
and collects evidence with `<prepared-python> -m tpen.inspection <run_dir>`.
The core module guard for direct module execution is deferred to a core
follow-up. The executor never imports
`tpen` or `experiments`; all scientific behavior and artifact reads remain in
the child process, while the executor decodes only the inspection JSON output.

Program authoring rules are intentionally explicit. A program marker may have
several named concrete target slots; `materialize()` returns a frozen mapping
and attribute view of those slots. Nested program markers remain handles until
their owning process materializes them. Marker subtrees must not use key-taking
resolver arguments, references from a parent marker into a nested marker's
interior, or submitter-interpolated intermediate nodes on keys referenced by a
program. `${oc.env:NAME}` without a key-taking argument remains fine. Pass
values that exist only in the owner process through the `materialize()`
environment. Descriptor `env:` values must be strings; unquoted YAML integers
are rejected. The packaged config demonstrates independent
control and science environment slots; they may point at the same prepared
environment, but no descriptor provisions, writes, or existence-checks paths.

From this directory, install the locked environment and run the executor
tests with:

```text
cd executor && uv sync --locked && uv run pytest -q tests
```

Operational outcome receipts use the closed vocabulary `completed`, `failed`,
`timed_out`, and `launch_failed`. These statuses describe execution only;
scientific validity remains the responsibility of core evidence.

## GPU placement qualification

The v2 placement receipt reports an explicit reason class: `qualified`,
`mismatch`, `telemetry-unavailable`, or `refused`. A GPU segment is never
qualified without live science-PID telemetry matched to the canonical physical
UUID entitlement. Missing or unavailable telemetry, and post-launch mismatch,
are operationally advisory: the segment may remain accepted and dispatch may
return `0`, but the receipt and manifest mark the GPU placement unqualified.
Prelaunch refusal is fail-closed (`launch_failed`, no accepted segment, and
dispatch return code `3`).

H-v2/science qualification must treat every unqualified GPU segment as *not
accepted for science*, even when its operational segment was accepted. Host
comparison uses the physical scheduler mapping only; a logical slot label by
itself cannot cause refusal or unqualification.

`DirectoryStore` requires a POSIX filesystem with advisory `fcntl.lockf`
support. It holds one task lock across pointer advancement, fence validation,
and outcome publication. Receipt publication writes and fsyncs a temporary
file, links it into the final name without replacement, fsyncs the containing
directory, and removes the temporary name. Newly created store directories and
pointer parents are fsynced before `begin_attempt` returns.

Store-owned directories and receipt files must not be pre-existing symlinks;
the store refuses them before use. The task namespace is deliberately limited
to reversible `[A-Za-z0-9._:-]+` path components. Resistance to a concurrent
adversary swapping a link after these checks is outside this layer's contract.

## CLI usage and exit codes

Dispatch one producer input with a fresh store and its allocation, worker, and
configured-runs records:

```text
python -m tpen_executor dispatch \
  --input <executor_input.json> \
  --store <fresh-store-directory> \
  --allocation <allocation.json> \
  --worker-config <worker.yaml> \
  --runs <configured-runs.json> \
  [--dispatcher local|parsl] [--timeout SECONDS]
```

The command writes a JSON manifest to stdout on dispatch and a typed
`tpen-executor/cli-error-v1` envelope to stderr on refusal or failure. Exit
codes are:

* `0`: all admitted members completed operationally;
* `2`: usage or admission refusal; no store is reserved;
* `3`: dispatch completed with a failed, timed-out, or launch-failed member;
* `4`: unexpected dispatch or publication failure after admission;
* `130`: cancellation after an attempt began.

## Local dispatch

`LocalDispatcher` is the attached-allocation baseline. It admits an exact
mapping from `ReadyGroup` member IDs to `ConfiguredRun` records, then launches
members sequentially through the materialized worker slots. Each member gets a
fresh store fence, and its child receives the configured command arguments plus
`run.root=<fence output>`, `run.layout=flat`, and `run.run_id=core`. With the
flat layout, core therefore writes its run directory at `<fence output>/core`.
Child and inspection logs are kept in the fence directory. Operational status
is independent of decoded core evidence; missing evidence leaves
`result_digest` unset. This dispatcher does not provide concurrency, retries,
recovery, Parsl, Balsam, scheduler integration, or facility qualification.

Admission also rejects a concrete store path containing the literal `${` before
the first lifecycle `begin`, because unresolved OmegaConf syntax must not be
allowed to redirect a core run. Any supplied timeout must be finite and
strictly positive. If cancellation interrupts a begun member, the dispatcher
attempts exactly one `finish`: it preserves an already-established
`timed_out` status and otherwise records `failed`, then propagates the original
cancellation. The subprocess worker kills its owned process group and keeps
waiting until the leader is reaped; further cleanup interruptions are attached
to the original exception rather than silently discarded.

Member `begin` failures are independent: later members are still visited and
their fences are finished. After processing, the first `begin` exception is
raised as the primary error. A later `finish` exception is retained as a note
on that same exception; when there was no `begin` failure, a `finish` exception
is raised normally.
