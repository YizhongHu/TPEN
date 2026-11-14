# Core ↔ executor boundary contract v1

Status: normative for the current `origin/dev` implementation at
`704d1c46c00cf19a4d55c28237abc423421e8f61`. This document is the shared
boundary contract for CX-00. “Current” means implemented and tested by the
checked-in code; “PROPOSED” is design only and is not an accepted wire format.
Core owns scientific run artifacts, public inspection, and checkpoint facts.
The executor owns process launch, attempt fencing, operational receipts, and
its immutable mirror of core evidence.

## 1. Scope and authority

The public core evidence record is produced by `tpen.inspection.inspect_run`
(`tpen/inspection.py:158-224`) and serialized by `RunEvidence.to_dict`
(`tpen/inspection.py:87-103`). The executor consumes that JSON through
`CoreRunEvidence.from_mapping` (`executor/src/tpen_executor/runners.py:143-196`)
and does not import `tpen` or read the run directory. The canonical v2 fixture
is therefore core-owned at
`tests/contracts/fixtures/core-run-evidence-v2.json`; both projects test that
same file.

This contract does not make executor operational status a scientific result.
`CoreRunEvidence` is an observed core record; `OutcomeReceipt` and segment
acceptance remain executor records. No facility, scheduler, or study-specific
fact belongs in the core evidence schema.

## 2. Launch v1

The supported launch surface is the prepared Python `-c` shim that imports
`tpen.run.main`, passes `--config`, and then passes positional OmegaConf
overrides. Core parses exactly that shape in `tpen/run.py:58-103`; `main`
delegates to `run_from_config` at `tpen/run.py:58-80`. The executor constructs
the command in `executor/src/tpen_executor/runners.py:257-291`.

The command configuration must be an absolute path. `TPENCommand` rejects a
relative or non-string `config` before the child is launched
(`executor/src/tpen_executor/runners.py:278-291`). This precondition is part of
the launch contract; an equivalent launcher must resolve and validate the
configuration path before constructing the command.

The executor must append the fenced overrides
`run.root=<attempt-output> run.layout=flat run.run_id=core`, as implemented at
`executor/src/tpen_executor/dispatch/contract.py:1070-1076`. Direct
`python -m tpen.run` execution is prohibited: it has no module guard on this
head and is not a launch contract. Future launchers use the shim, the
installed `tpen` console entrypoint, or an equivalent call to `tpen.run.main`.

## 3. Physical/layout v1

The executor-owned physical fence is `<attempt>/core`. Segment acceptance
requires a real `core` directory and compares the inspected evidence path to
that exact canonical directory (`executor/src/tpen_executor/storage.py:443-453`).
The executor owns this leaf convention; core owns the logical run layout.

Core's `ArtifactManager` owns standard artifact layout
(`tpen/artifacts.py:178-285`). `ArtifactDestination` is an optional,
launcher-supplied physical destination (`tpen/artifacts.py:83-175`) and is
validated before sinks are constructed (`tpen/run.py:115-180`). A logical
`run.dir` may therefore differ from the physical leaf while the core config
identity remains stable. The current evidence record reports the resolved
run directory and checkpoint directory observed by inspection; the canonical
fixture normalizes both absolute paths to `/normalized/core` for byte-stable
parity tests. Timestamps and host-specific values are not part of the v2
evidence record.

## 4. Core evidence v2 (current)

The current closed records are `tpen.run-evidence/v1` and
`tpen.run-evidence/v2` (`tpen/inspection.py:23-27`). V1 contains exactly:

```json
{
  "schema": "tpen.run-evidence/v1",
  "run_dir": "...",
  "verdict": "completed|incomplete|failed|refused",
  "run_status": "...|null",
  "checkpoint": {"checkpoint_dir":"...","schema_version":2,"next_iteration":7,"completed_updates":5}|null,
  "refusal": {"code":"...","reason":"..."}|null
}
```

V2 adds the required non-null `resolved_config_identity` object with exactly
`schema=tpen-config-identity-v1` and a lower-case 64-character SHA-256 digest.
The exact v2 selection rule is: a successfully read, non-null
`resolved_config_identity` from the run-start sidecar, on a result that is not
a refusal. The producer selects the schema only after reading and validating
that identity (`tpen/inspection.py:227-254`). Any refusal always returns v1
without identity because the refusal branch calls `_evidence_schema(None)`
(`tpen/inspection.py:216-224`); a readable sidecar that lacks the identity
also leaves the identity null and therefore does not select v2
(`tpen/inspection.py:233-253`). `RunEvidence.from_mapping` enforces the
version-specific top-level and nested key sets (`tpen/inspection.py:104-143`).
The executor mirrors the same closed shape and invariants
(`executor/src/tpen_executor/runners.py:163-196,211-253`). Unknown schema
versions and unknown/extra fields are refused. A completed record requires a
checkpoint; a refused record requires a refusal and cannot carry a checkpoint.
The refusal `reason` embeds the Python JSON parser message, so fixture parity is pinned to project Python 3.12 (a later slice may compare the reason by prefix if this wording proves version-fragile).

## 4.1 Core evidence v3

CX-01 defines `tpen.run-evidence/v3` as the v2 object plus `outcome` and
`checkpoint_ref`. The producer selects v3 when the run has the S4
`run-outcome.json` artifact and decodes it through `tpen.run_events.read_run_outcome`;
inspection never infers the typed result from `status.json` or the process exit
code. The S4 artifact is the closed `tpen.run-outcome/v1` mapping emitted from
`RunResult.to_dict`; the decoder also accepts the pre-CX-01 byte shape as a
read-only implicit-v1 record.

`outcome` is exactly one of `completed`, `planned_yield`, `failed`, `unknown`,
or `refused`. `run_status` remains the operational status written by core and
must equal the typed result status. A mismatch, malformed artifact, missing
ref, stale ref, unpublished ref, or unsupported payload profile produces a
refused v3 evidence record and can never produce `planned_yield` evidence.

When present, `checkpoint_ref` is a complete `tpen.checkpoint-ref/v1` identity:
the schema, concrete checkpoint directory, manifest schema/kind, both progress
counters, manifest and model SHA-256 values, immutable manifest provenance,
and path-independent `content_id`, plus the manifest-declared payload
`profile`. A planned yield requires the ref's live bytes and content ID to
validate, the ref to appear in the publication catalog, profile
`train_resume`, and both result counters to equal the ref counters. A
`model_only` or missing ref is therefore non-eligible even when a checkpoint
directory exists.

Continuation eligibility is an explicit rule, not an operational status:

```
eligible iff outcome == planned_yield
           and the typed RunResult decoded successfully
           and checkpoint_ref is a live, published train_resume ref
           and checkpoint_ref.next_iteration/completed_updates equal the result
```

`completed`, `failed`, `unknown`, and `refused` are never continuation-eligible.
The v1/v2 producer and decoder remain read-only and byte unchanged when no
typed outcome artifact is present.

## 5. Checkpoint publication and recovery

Core publishes checkpoint directories, catalog rows, pointers, and receipts
through the checkpoint package (`tpen/checkpoint/save.py:87-106,209-241`,
`tpen/checkpoint/catalog.py:25-100`, `tpen/checkpoint/artifact.py:50-89`).
Publication is ordered: component files are written to a temporary directory,
the manifest and `COMPLETE` marker are written last, and rename commits the
checkpoint. Core then appends the catalog row, atomically updates
`latest.json`, and finally appends the publication receipt
(`tpen/checkpoint/save.py:87-110`). Catalog and pointer failures are load
bearing; the receipt append is best effort and can be reconciled later. A
receipt therefore proves publication bookkeeping only when present and must
not be inferred from the checkpoint summary exposed at this boundary.
Inspection selects the newest valid train-resume checkpoint and exposes only
the v2 summary fields (`tpen/inspection.py:276-356`). The checkpoint directory
is a location; the underlying `CheckpointRef` is the core-owned content
identity (`tpen/checkpoint/reference.py:1-80`).

The current executor mirror intentionally retains only
`checkpoint_dir`, `schema_version`, `next_iteration`, and
`completed_updates`. New accepted segments additionally require the v2 config
identity (`executor/src/tpen_executor/storage.py:454-485`). A consumer must
not infer a full checkpoint reference, publication receipt, or resumability
from this summary alone.

## 6. Segment continuation (PROPOSED)

Current `AcceptedSegment` is a fresh-start operational record. It requires
core evidence with a checkpoint and an executor outcome whose status is
`completed` (`executor/src/tpen_executor/storage.py:434-498`). Current
invocation/recovery code can discover a published predecessor read-only, but
only the allocation that still owns the run leaf may reconcile a missing
publication index (`tpen/execution/invocation.py:685-705`). There is no
end-to-end executor successor launch contract.

Continuation must consume a typed v3 planned-yield result and a full,
publication-bound checkpoint reference. It must not reuse `AcceptedSegment`
while pretending the operational attempt completed, and it must reject failed,
unknown, or refused outcomes. The exact successor record, fork policy, and
cross-allocation result bundle remain future CX-02/HI-S5 decisions.

## 7. Teardown and placement

The executor owns process groups, placement receipts, attempt fences, and
operational outcome receipts. The dispatch path launches the fenced command
and collects inspection evidence after the child (`executor/src/tpen_executor/dispatch/contract.py:1062-1079`).
An existing slot is quarantined before a sampled launch and remains quarantined
until sampler cleanup is confirmed; any non-inspection launch carrying that
slot id is refused while the quarantine exists
(`executor/src/tpen_executor/workers.py:120-137`). This is the current local
slot-quarantine contract, not a remote-quiescence guarantee.
Core does not consume executor teardown or placement records. A scheduler or
process exit alone is not core completion; completion is the durable core
status plus admissible checkpoint evidence (`tpen/run.py:475-497`,
`tpen/inspection.py:185-224`).

**PROPOSED:** a future remote continuation wire may carry an identity-bound
teardown acknowledgement and typed placement result, but it must remain
operational metadata and cannot replace the core typed outcome or checkpoint
publication proof.

## 8. Evolution rules and compatibility

Schema objects are closed: producers and consumers must reject unknown
versions and unknown fields. Additive changes require a new schema version,
canonical core-owned fixtures, producer and consumer parity tests, and an
updated matrix. Existing v1/v2 readers remain read-only compatible with their
own records; no consumer may silently reinterpret a newer record as an older
one. The fixture generator is `tests/contracts/_regen_fixtures.py`; it creates
a real status/checkpoint/run-start artifact set, calls `inspect_run`, then
normalizes only absolute paths under the temporary producer root. Planned-yield
runs execute with that temporary root as cwd so checkpoint provenance is
normalized by the same prefix-only rule. The canonical planned-yield fixture
replaces exactly `checkpoint_ref.content_id` with `<sha256:content_id>` and
`checkpoint_ref.manifest_sha256` with `<sha256:manifest>`, after asserting each
producer value is a 64-hex SHA-256 string. Live tests compare those fields with
`CheckpointRef` computed from the checkpoint bytes; the placeholders are only
for cross-clone fixture stability and do not change product hashing.

| Producer record | Consumer v1 | Consumer v2 | Consumer v3 |
|---|---|---|---|
| v1 | decode | decode read-only (identity absent) | refuse/unsupported until v3 policy |
| v2 | refuse (closed v1 key set; no operation-dependent downgrade) | decode | refuse/unsupported until v3 policy |
| v3 | refuse | refuse | decode and apply the CX-01 eligibility rule |

“Historical decode” means a v2-capable consumer may read a v1 record for
historical records; it is separate from identity-required acceptance and does
not upgrade the record. A v1 consumer must refuse v2 because its closed key set
does not admit the newer record; there is no operation-dependent downgrade.
The current executor implementation supports v1 and v2 only and
rejects all other versions and extra fields at
`executor/src/tpen_executor/runners.py:163-173`.

The proposed remote-quiescence contract tracked by #668 remains PROPOSED and
is not implemented by this head.

Naming is reserved as follows: `HI-S<n>` identifies the HI-resume execution
lane, `EX-S3.<letter>` identifies executor transport layers, and `CX-nn`
identifies cross-boundary contract slices. This document is `CX-00` and does
not rename or collapse those workstreams.
