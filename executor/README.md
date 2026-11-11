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

Quarantine release (decided: no in-allocation release; manager ruling
2026-11-10, Task Orchestrator item 08eeff51 note
`manager-ruling-quarantine-release-20261110`). A marker whose owner died is
not released by another process, and no operator release command exists. The
slot stays quarantined for the rest of the allocation, and allocation teardown
is the release (a new allocation has a new allocation and backend job
identity, so it does not see the old marker). The marker is kept as evidence
until a separate disposition authorizes removing it. A supervised release
command, which must prove that all possibly-launched work is gone, is a future
follow-up.

Assumption: one controller per allocation. The local and ThreadPool
dispatchers run their workers in the controller process, so the in-process
lease sees every launch, and the durable marker covers a controller restart on
the same store. A leaf on the attached remote route (S3 Layer B) is a fresh
process on a worker: it is admitted against the durable markers in the shared
attempt store only. Markers in a different attempt store, and concurrent independent
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

## Remote-leaf wire (S3 Layer A)

`tpen_executor.dispatch.transport` defines the versioned wire between a
dispatch controller and a leaf that may run in another process or node.
`LeafRequest` (`tpen-executor/leaf-request-v1`) is built by the controller
for one begun attempt. It binds the group, task, attempt, allocation,
backend job and slot identities, the configured payload and its digest, the
fenced output paths, and the prepared science environment. It holds only
exact plain values at every depth, so a lock, future, process handle,
controller callback or builtin/`SourceScript` subclass is refused at
construction. The configured run must belong to the fenced task, and the
output directory must be the store directory of exactly this group, task and
attempt, whether or not the environment carries a placement identity. Its
payload digest and any placement identity in the environment must name the
same attempt, and a carried slot quarantine directory must be the one derived
from the request's store root. Decoding accepts only bytes equal to the
canonical re-encoding.

`run_leaf(payload: bytes) -> bytes` reconstructs the prepared environment,
refuses a fence that is not visible or not fresh (a replay), applies local
dispatch slot admission against the durable quarantine markers under the
request's store root before any spawn, and runs science
once through `SubprocessWorker` with the exact `RUN_SHIM` arguments. It then
runs public inspection in the same leaf, under the same conditions as a
local completion. It returns `LeafResult` (`tpen-executor/leaf-result-v1`)
facts: the reaped process outcome, a launch error or pre-core refusal, the
inspection state and decoded evidence, and `{"leaders_reaped": true}`. The
leaf never begins, finishes, accepts or publishes anything. The controller
checks every echoed field with `validate_leaf_result`, refuses evidence for
another run directory and any non-refused science outcome that skipped
inspection, and publishes from
`LeafResult.status()`, `result_digest()` and `core_evidence()`. The GPU
sampler and worker slot quarantine are unchanged because the leaf launches
through the same worker. Layer A wires no dispatcher, so local and ThreadPool dispatch
are unchanged. Remote submission and cancellation acknowledgement belong to
later S3 layers.

## Attached remote transport (S3 Layer B)

`ParslDispatcher(worker, transport=...)` selects the attached remote route.
`transport` is an `AttachedTransportSpec`, or its strict mapping with
`schema: tpen-executor/attached-transport-v1`. Only one form is admitted:
`executor: htex`, `provider: local` and `launcher: mpiexec`, which is one
HighThroughputExecutor whose LocalProvider starts exactly one worker block
through MpiExecLauncher inside the allocation the controller already holds.
Any other executor, provider or launcher is refused, and `transport` cannot be
combined with `executor_config`. Without `transport`, Parsl dispatch is the
unchanged in-process ThreadPool route.

The spec holds generic values only: `label`, `nodes_per_block`,
`max_workers_per_node`, `worker_init`, `address`, `mpiexec_bind_cmd`,
`mpiexec_overrides` and `preflight_timeout`. The executor ships no facility
value. An operator-owned profile supplies the worker environment setup, the
interchange address and launcher arguments. The interchange and the worker
pool are started as `python -m parsl.executors.high_throughput.interchange`
and `python -m parsl.executors.high_throughput.process_worker_pool` with the
controller's own interpreter (`sys.executable`), so neither depends on
`PATH`; that interpreter path must therefore also be valid on every worker
node, as it is for a shared environment. The route never
acquires, replaces or retries an allocation: Parsl retries are 0 and the
scaling strategy only starts the initial block.

Before any lifecycle write, admission refuses:

* a lifecycle that is not `AttachedAllocation`;
* a worker program other than the standard `SubprocessWorker`,
  `TPENCommand` and `TPENEvidence` with a `PreparedPythonEnvironment`,
  because the leaf would silently run the standard program instead;
* more slots than `nodes_per_block * max_workers_per_node` workers;
* an attempt-store root that is not an existing absolute POSIX directory.

Each run then builds a fresh config and DataFlowKernel and preflights the
shared store. The controller writes a nonce token under
`<store>/parsl/store-preflight-<nonce>/` and checks POSIX `lockf` on it. It
then submits one probe per worker. Each probe must resolve the store root to
the controller's canonical path and read the token. It then takes and
releases `lockf` on a lock-capability file private to that probe, never on the
shared token, so probes never contend with each other and a healthy store is
never refused for ordinary contention. Finally it waits at a file barrier until
every worker holds a probe at once. A store-blind, lock-incapable, slow or
missing worker refuses the whole run (`SharedStorePreflightError`) before the
first `begin`, so no attempt exists. No probe or other Parsl app is retried.

Each begun member is submitted as the Layer A leaf with its canonical
`LeafRequest` bytes as the only argument. The request environment is the
controller's prepared environment after the slot binding, context-key and
placement overlays. The leaf applies it over the worker's ambient
environment, so neither controller nor worker ambient `CUDA_VISIBLE_DEVICES`
or thread values can replace a bound key. The controller validates the
returned bytes against the request and publishes through the shared
completion path; the leaf's inspection facts replace controller inspection.

A remote slot's CPUs are checked where the slot runs. Local and ThreadPool
dispatch still refuse a declared CPU outside the controller's affinity
(`placement-cpu-outside-controller-affinity`). The attached route skips that
controller check, and the leaf instead refuses, on its worker and before any
child is spawned, a declared CPU outside the worker's affinity
(`placement-cpu-outside-worker-affinity`) or an unavailable worker affinity
(`placement-worker-affinity-unavailable`). The controller publishes that
refusal as the attempt's placement receipt and the attempt is `launch_failed`.

These cases are durable `launch_failed` attempts and never accepted segments:

* a returned result that does not answer its request;
* a leaf launch error or refusal;
* a failed remote future.

On cancellation, the controller finishes every in-flight remote attempt
`failed` without waiting for it or signalling a remote PID. Bounded remote
teardown acknowledgement is S3 Layer C.

Node routing is not implemented in Layer B. HTEX sends a task to any idle
worker, so a declared slot node is not a routing constraint, and a leaf may
run science on a host other than its slot's declared node. Under the
inherited S2b policy such a host mismatch is ADVISORY, not blocking: the
child records `child-placement-mismatch` in its placement sidecar but science
still runs, and the attempt can complete and be accepted with an unqualified
placement receipt. Layer B therefore does NOT qualify placement, and it does
NOT qualify R5 (two-node) evidence. Deterministic routing is deferred to
Layer D: a node-to-executor-label map with one worker pool constrained to
each named node by operator-supplied launcher data, each slot submitted only
to its node's label, and observed node identity preflighted, which revises
Layer B's one-executor validator.

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
