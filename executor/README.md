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

The L1a v2 placement receipt reports an explicit reason class: `not-sampled`,
`mismatch`, or `refused`. L1a deliberately does not run an in-child GPU
sampler: declared-device evidence is `device=unavailable` with the loud
reason `not-sampled`. This is operationally advisory, so a completed segment
may remain accepted and dispatch may return `0`; the manifest records that H-v2
must perform its own delivered-GPU check. H-v2 does not wait for L1b.

L1b adds an `nvidia-smi` PID-to-UUID sampler and a separate closed wire,
`science-placement-receipt-v3`; `v2` keeps its L1a meaning and `v1` is
unchanged. The sampler lives in the worker that launches and owns the science
child (`SubprocessWorker`), not in the child, so `RUN_SHIM` is unchanged from
L1a. For a declared-GPU v2-placement slot the controller assigns a
worker-only evidence path; the worker claims it with an attempted,
unfinalized record before spawn, samples the exact child PID immediately and
then every cadence (default 2 s, bounded to 0.01-60 s) for the child's whole
lifetime, and takes a final sample once it has observed the exit. At most 16
samples are retained; a closed lifetime aggregate records every GPU the
exact PID ever used and whether out-of-entitlement use or an unresolved
entitlement was ever seen, including in samples that were not retained.
Only the explicit prepared descriptor opts a launch into sampling; an
inherited owner-environment value of the evidence key is ignored and the key
is never passed to the child. The worker holds the sampler before starting
it, so an interruption during or right after startup cannot lose its cleanup.
`nvidia-smi` is resolved once through the `PATH` of the worker's prepared
child environment (the descriptor's setup script is not applied to it); the
resolved path and its reported version are recorded.
Each command runs in its own session. On every path, including a worker
timeout or cancellation of the science child, the worker SIGKILLs, drains,
reaps, and confirms every command group empty (ESRCH from `killpg(pgid, 0)`)
under one shared deadline before it returns. The deadline is derived from the
command bounds: at most one attempt in progress plus one final attempt, each
of at most three commands bounded by a 5 s read plus a 2 s cleanup, plus a
3 s cancel reserve (`2 * 3 * (5 + 2) + 3 = 45` s). If a launch or a group
cleanup is still outstanding at the deadline, the record is not quiescent and
keeps the surviving group ids.

Slot quarantine. Before it spawns a sampled child, the worker quarantines the
slot: an in-process lease plus a durable marker in the attempt store under
`<store root>/slot-quarantine/<sha256 of allocation id, backend job id,
physical host, slot id>/`. The lease is settled on every exit, including an
interrupted or failed evidence write, and released (marker removed) only once
the child is reaped and every sampler command group is confirmed empty.
Until then every science launch on that slot fails with `SlotQuarantined`
before spawning, sampled or not: the worker checks the in-process lease by
`TPEN_SLOT_ID`, and dispatch admission also checks the durable markers, so a
restarted controller that reuses the same attempt store refuses the slot too.
Only the member's own inspection is exempt. The exemption is the typed worker
argument `purpose=LaunchPurpose.INSPECTION`, passed only by
`TPENEvidence.read`; no environment value grants it, and the retired
`TPEN_EXECUTOR_LAUNCH_ROLE` key is scrubbed from every launch environment. An
unsampled launch, including the legacy/no-unique-slot science fallback, is
never treated as an inspection. This argument is a correctness guard against
accidental slot reuse, not a security boundary: any caller that controls the
worker invocation can pass it.

Quarantine release (pending a manager decision). A marker whose owner died is
not released by another process, and no operator release command exists. The
conservative default holds until the manager decides otherwise: the slot stays
quarantined for the rest of the allocation, and allocation teardown is the
release (a new allocation has a new allocation and backend job identity, so it
does not see the old marker). The marker is kept as evidence until a separate
disposition authorizes removing it.

Assumption: one controller per allocation. Every dispatcher runs its workers
in the controller process (Parsl is ThreadPool-only), so the in-process lease
sees every launch, and the durable marker covers a controller restart on the
same store. Markers in a different attempt store, and concurrent independent
controllers on one allocation, are not detected; cross-store leases are
deferred to follow-up item `0aa73edd`.

Telemetry quality is kept separate from the operational outcome: a worker
timeout or cancellation of the science child still finishes the sampler, so
usable, quiescent observations taken before it survive. Only a sampler that
is itself cancelled (a command still in flight at the deadline) or never
finalized is `telemetry-failed`. Transient command failures (non-zero exit,
read timeout, read error) are bounded by a recorded gap budget: one isolated
failed attempt is skipped, while two in a row or a fourth over the lifetime
is the terminal `gap-budget-exceeded` failure. Every gap is recorded (count,
longest consecutive run, total seconds, and each event) together with the
budget in force. Outside use, malformed or contradictory output, and
unconfirmed cleanup are never forgiven. MIG device identifiers are
unsupported and fail with the distinct `mig-unsupported` code. The timeout
and gap-budget values are unmeasured defaults to be calibrated on a real-GPU
run.

The v3 verdict is derived, never trusted: out-of-entitlement use by the exact
science PID at any time is `mismatch`; an attempted sampler failure, an
unresolved entitlement, or no row for the exact PID during the whole
lifetime is `telemetry-failed`; `qualified` requires quiescent evidence with
every active UUID inside the slot entitlement. A GPU child that never creates
a CUDA context is therefore `telemetry-failed` and science-blocking.
`not-sampled` (intentionally unattempted, for example a CPU-only slot) is not
science-blocking; `telemetry-failed` is. Both stay operationally advisory.

Post-launch host or affinity mismatch is also advisory: the segment remains
accepted, dispatch returns `0`, and the receipt is unqualified for placement.
Prelaunch refusal is fail-closed (`launch_failed`, a typed placement receipt,
no accepted segment, and dispatch return code `3`). Host comparison uses the
physical scheduler mapping only; a logical slot label is retained as metadata
and cannot be attributed to the scheduler-node map.

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
