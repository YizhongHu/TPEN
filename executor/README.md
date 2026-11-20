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

## Allocation-group reconciliation (B1a)

The executor exposes a fake-driven, pure Layer A contract for reconciling an
immutable allocation group. `ExecutionPlanV2` binds opaque plan content to an
ordered task list and explicit continuation/reconciliation protocol tokens;
`ReadyGroupV2` binds that plan to an opaque `GroupId`. An
`AllocationAttemptV2` separately identifies one controller/allocation
generation. `GroupId`, `AllocationAttemptId`, and `InvocationId` are
deliberately non-interchangeable with `AttemptId`, `AllocationId`, or
`BackendJobId`.

`reconcile_group()` consumes typed executor start/outcome receipt references
and a typed `CoreAdmissionPort` verdict. It derives one of
`preserve_completed`, `admit_successor`, `start_fresh`, `preserve_terminal`,
or `park` for every member. Parked evidence dominates the group aggregate;
otherwise a successor or fresh member yields `replacement_required`, and a
fully preserved group is `settled` with completed/terminal counts. The
reconciler publishes no ledger, starts no process, replaces no allocation,
scans no checkpoint, and maps no result to a scheduler or Balsam status.

The profile/admission port is intentionally a narrow typed fake seam. A
missing or refused native profile parks a member before an optional restore
callback can run. Real-core compatibility, durable publication, successor
fencing, Balsam lifecycle, and facility integration are later layers and are
not claimed by B1a.

### B1a-C durable successor and multi-generation publication

`DirectoryStore.begin_admitted_successor()` is the only executor start path
for a CX-04 admitted successor. It locks the task, rereads the canonical
successor-admission ledger, requires the predecessor to remain current, and
consumes the recorded `AttemptId` and `InvocationId` exactly. The start and
invocation files are write-once; an identical replay returns the same fence,
while a missing, crossed, symlinked, or differing record is refused.

`publish_reconciliation()` is keyed by the new controller's
`AllocationAttemptV2`, but each existing member is validated against the
allocation attempt recorded in that member's durable `invocation.json`. This
allows recovered groups to combine evidence from several prior allocation
generations. Evidence from another group, from the new controller lease, or
whose durable invocation was tampered with parks the publication. Existing
single-generation `ReconciliationPublicationV2` bytes remain unchanged.

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

### Start-receipt-v1 migration

Historical `start-receipt-v1` attempts carry no transport kind, so after an
upgrade they read as unknown and are not eligible for continuation or replay
without a positive remote teardown receipt. Their transport is never inferred
or rewritten as local.

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
  [--dispatcher local|parsl] [--transport <attached-transport.json>] \
  [--timeout SECONDS]
```

`--transport` is optional and is accepted only with `--dispatcher parsl`. Its
file must be a strict `tpen-executor/attached-transport-v1` mapping accepted by
`AttachedTransportSpec`; without it, Parsl uses the unchanged in-process
ThreadPool route. The transport profile supplies only generic attached-route
values; the executor does not provide facility defaults.

The command writes a JSON manifest to stdout on dispatch and a typed
`tpen-executor/cli-error-v1` envelope to stderr on refusal or failure. Exit
codes are:

* `0`: all admitted members completed operationally;
* `2`: usage or admission refusal; no store is reserved;
* `3`: dispatch completed with a failed, timed-out, or launch-failed member;
* `4`: unexpected dispatch or publication failure after admission;
* `130`: cancellation after an attempt began.

Attached transport capacity and standard-worker shape refusals are reported as
typed `dispatcher-admission` errors with exit code `2`, before the store is
reserved. Shared-store reachability checks remain runtime dispatch failures.

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

### Routed attached transport v2 (Layers D2a/D2b)

The strict `tpen-executor/attached-routed-transport-v2` profile describes an
ordered pool for each captured allocation node. Each pool carries its node,
label, worker count, worker setup, interchange address, and opaque
`MpiExecLauncher` bind/override values. The executor supplies no facility
values and does not interpret those launcher strings. Its matching
`tpen-executor/allocation-slots-v2` record may contain several slots per node;
admission requires the exact node set from the captured scheduler host map and
requires `pool.workers == slot_count` for every node. Host strings are compared
exactly at this boundary; aliases and DNS normalization are not inferred.

Admission constructs one fresh `HighThroughputExecutor` with a
`LocalProvider(MpiExecLauncher(...))` for each pool. Every pool is a single
initial block with its own label and worker count, and the route retains the
no-retry, no-scaling, interpreter-module launch settings. Before any
`lifecycle.begin`, the controller submits exactly the pool's worker count of
shared-store probes through that pool's label. All probes join one aggregate
barrier, and each returned hostname must equal the pool node exactly. A typed
`RoutedTransportPreflightError` refuses wrong, missing, extra, or late facts.

At each member launch, the controller resolves the binding's captured
physical node to exactly one pool label and submits the leaf with only
`executors=[that_label]`. An absent or ambiguous node-to-label mapping is an
admission refusal before `begin`; the v1 attached profile remains the
one-executor, single-label path described above.

Routed children carry an internal strict-host flag. `RUN_SHIM` still writes
the identity-bound placement sidecar, but a declared-host mismatch under that
flag aborts before importing core. The controller recognizes only that
routed sidecar case as `PlacementRefusal(code="routed-child-node-mismatch")`,
publishes the refusal receipt, finishes the attempt as `launch_failed`, and
accepts no segment. v1 and non-routed children retain the advisory
`child-placement-mismatch` behavior and may complete.

All routed normal and refusal replies continue through the CX-06 remote
completion lifecycle. Cancellation or remote loss sweeps every active label
under one bounded teardown deadline; missing or unacknowledged replies remain
`remote-quiescence-unconfirmed` and close admission. D2b adds no retry,
continuation, successor, or resubmission behavior.

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

## Balsam group registration and application boundary (B1b)

`tpen_executor.lifecycle.balsam` provides an optional, group-keyed registration
port and application factory. It does not import Balsam until
`make_group_application(api=None, ...)` is called. Tests inject a fake API;
no SDK dependency or installation is added by this slice.

`register_group(intent, store, client)` binds one persistent `BackendJobId` to
one `GroupId` in the configured `DirectoryStore`. The intent includes the
ready-group digest and the digest of the complete backend request (including
application identity and deployment options). The client owns constructing
that request and must honor those digests. Registration uses the store's POSIX
durable no-replace publication primitives: an exclusive intent is persisted
before the sole backend call. Identical acknowledged intent reuses its receipt;
changed intent raises `BalsamGroupRefusal`. A missing/ambiguous acknowledgement,
including failure to persist it, raises `BalsamGroupParked`. Reopening an intent
without acknowledgement never calls registration again, even if the original
process died before making the call. Permanent client refusal is retained.
Do not remove these records to request a retry. The one-Job guarantee is scoped
to this shared store; independent stores are not coordinated.

`GroupExitContract(backend_retry_codes=...)` requires an explicit deployed retry
set and rejects overlap with exit 2 (or success 0). Its version-one no-retry
code is fixed at 2. The historical set `(-11, 1, 255, 12345)` is test input,
not an installed-version assertion or automatic default. `group_exit_code`
accepts typed B1a reconciliation or a `GroupFailure`:

| Authoritative outcome | Process exit |
| --- | --- |
| Every member `preserve_completed` | 0 |
| Settled with any terminal member | 2 |
| Parked | 2 |
| Pre-start failure | 2 |
| Uncaught exception | 2 |
| Voluntary exit with replacement required | 2 |

Backend observations, `JOB_FINISHED`, and integer return values cannot establish
completion. `make_group_application(api, controller=..., contract=...,
record_failure=...)` builds an `ApplicationDefinition` at call time. Its
`run()` passes the Job to the injected controller exactly once. Non-success
raises `SystemExit(2)` instead of returning an integer that a backend Python
runner might serialize while exiting zero. Exceptions (including an injected
`SystemExit(1)`) go through the failure recorder and also exit 2. If recording
itself fails, the original exception retains that fact and the process still
exits 2; storage failure cannot guarantee a durable receipt.

The shell preamble is empty and `environment_variables` is empty, so this
boundary preserves inherited environment. The injected controller owns
prepending configured paths, recording the effective environment, durable
start/pre-start receipts before work, allocation and invocation identities,
reconciliation, successor admission and dispatch. Those controller mechanics
are **not implemented or qualified by B1b**. Per the scope split, B1a-C supplies
the storage seams and B1b2 supplies `run_group_controller`. No scheduler,
reaper, retry loop, allocation replacement, walltime/signal behavior, or real
core admission is supplied here. Deployment exit behavior and durability need
independent B1c qualification; fake tests are not facility evidence.
