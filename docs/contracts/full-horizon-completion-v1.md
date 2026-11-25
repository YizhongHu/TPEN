# Full-horizon completion v1

Status: normative for the executor's task-level completion writer and HI
consumer. This is a wire contract. A consumer MUST parse the record with the
Python standard library (`json.loads`) and MUST NOT import `tpen_executor` to
discover its shape. The record is the only executor evidence from which HI may
derive science `COMPLETE` for a task.

## Durable placement and discovery

For store root `<store>`, group digest `<group>`, and task `<task>`, the one
write-once record is:

```text
<store>/groups/<group>/<task>/full-horizon-completion.json
```

The reader derives `task_dir` from the final attempt's `attempt_core_dir` as:

```text
task_dir = dirname(dirname(attempt_core_dir))
```

This is exact: `AttemptDelta.attempt_core_dir` is an absolute normalized path
ending in `<attempt>/core`, and the store layout is
`<store>/groups/<group>/<task>/<attempt>`. The record is task-level, beside the
attempt leaves, not inside the final attempt leaf. `final_attempt_id` identifies
which leaf supplied the terminal facts.

The source definitions are `AttemptDelta`
(`executor/src/tpen_executor/configured_program.py:69-85`), the layout parser
(`executor/src/tpen_executor/storage.py:128-151`), and the
fence-to-task-directory validation
(`executor/src/tpen_executor/storage.py:674-683`).

The task directory also contains a write-once canonical `task-spec.json`
snapshot, published when the first attempt is started. The final writer takes
the task's `TaskSpec.science_digest` verbatim from that snapshot. A missing or crossed snapshot is a refusal; recovery
checkpoint and grant lineage are not substitutes.

Absence of `full-horizon-completion.json` means the task is not complete. There
is no parked completion record.

Each attempt leaf also contains the executor-internal, write-once
`start-provenance.json` record at
`<store>/groups/<group>/<task>/<attempt>/start-provenance.json`. HI does not
read this file. Its schema is `tpen-executor/start-provenance-v1` and its
common fields are `kind`, `task_id`, `attempt_id`, and `group_digest`. A fresh
attempt has `kind: "fresh"`; an admitted successor has `kind: "admitted"` plus
`predecessor_attempt_id` and the canonical successor-admission `request_digest`.
The executor writes it atomically under the task lock when the corresponding
begin operation succeeds. The completion writer requires the root attempt to
be fresh and every later attempt to be admitted with a matching durable v3
ledger link; missing, crossed, unreadable, or mismatched provenance refuses
publication.

## Closed record shape

Required top-level fields are exactly:

| Field | JSON type | Rule |
| --- | --- | --- |
| `schema` | string | Exactly `tpen-executor/full-horizon-completion-v1`. |
| `group_digest` | string | Lowercase 64-hex SHA-256 group identity. |
| `task_id` | string | Non-empty logical task identity. |
| `science_digest` | string | Verbatim final-task `TaskSpec.science_digest`, lowercase 64-hex. HI compares it with the row id. |
| `final_attempt_id` | string | Canonical attempt identity of the final durable leaf. |
| `completed_updates` | integer | `final_attempt.core_outcome.evidence.checkpoint.completed_updates`. |
| `target_iterations` | integer | Full task horizon in iteration units; it equals `TaskSpec.target_horizon` when the task declares `target_horizon_units == "iterations"`. |
| `final_checkpoint` | object | Exactly `path` and `content_id`; both strings. |
| `completed_at` | string | Canonical UTC rendering of the final outcome receipt timestamp, formatted `YYYY-MM-DDTHH:MM:SS.ffffffZ`. |
| `core_acceptance_digest` | string | `digest(verdict.to_dict())` for the typed final admission-port verdict. |
| `attempt_chain` | array | Ordered successor-admission-v3 identity links. |

`program_digest` is optional audit data. It may be present only when a typed
durable configured-program source exists; it is never a completion authority.
No other top-level field is permitted.

Each `attempt_chain` entry has exactly:

```json
{"predecessor_attempt_id":"...","successor_attempt_id":"...","request_digest":"..."}
```

The entry identifies an existing canonical
`tpen-executor/successor-admission-v3` record in the predecessor ledger.
Links are continuous, acyclic, unique, and the final link's successor is
`final_attempt_id`. A legacy single-attempt task has an empty chain.

The canonical encoding is UTF-8 JSON with sorted keys, compact separators, and
no duplicate keys. Example:

```json
{"attempt_chain":[],"completed_at":"2026-11-25T14:00:00.000000Z","completed_updates":100,"core_acceptance_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","final_attempt_id":"attempt-S","final_checkpoint":{"content_id":"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff","path":"/store/groups/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb/task-7/attempt-S/core/checkpoints/final"},"group_digest":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","schema":"tpen-executor/full-horizon-completion-v1","science_digest":"abababababababababababababababababababababababababababababababab","target_iterations":100,"task_id":"task-7"}
```

## Producer guards

The writer is called only after the final `CoreAdmissionPort.admit` call
returns a typed `CoreAdmissionVerdictV2` with `outcome == "completed"`. It
reads `completed_updates` from the final durable v3
`CoreOutcomeRecord.evidence["checkpoint"]["completed_updates"]`, never from an
exit code or an `OutcomeReceipt` alone. The final `CoreRunEvidence` checkpoint
must be present and its `checkpoint_ref` must be present. The final checkpoint
is:

```text
final_checkpoint.path       = checkpoint_ref["checkpoint_dir"]
final_checkpoint.content_id = checkpoint_ref["content_id"]
```

`checkpoint["checkpoint_dir"]` must equal `checkpoint_ref["checkpoint_dir"]`.
If a typed checkpoint source also carries a content id, it must equal the
reference content id.

`target_iterations` is the full task horizon in iteration units. The writer
requires `TaskSpec.target_horizon_units == "iterations"` and exact equality
with `TaskSpec.target_horizon`; non-iteration units are an explicit typed
refusal. Every admitted successor request also carries that same fixed total
horizon, so every request copy must equal `TaskSpec.target_horizon` and agree
with every other request. The writer performs no silent unit conversion.

`science_digest` is read verbatim from the durable final-task `TaskSpec` and
must equal any science lineage carried by the admitted successor chain. The
recovery predicate's checkpoint/grant equality does not authorize copying
either opaque value into this field.

`TaskSpec.science_digest` is the typed field defined and validated at
`executor/src/tpen_executor/contracts.py:137-180`; the writer's durable
snapshot is the final-path source of truth.

`completed_at` is derived only from the durable final `OutcomeReceipt`'s
`finished_at_utc`; the writer MUST NOT call the clock. It parses an explicit
timezone-bearing timestamp, requires a zero UTC offset, converts it to UTC, and
renders the canonical form `%Y-%m-%dT%H:%M:%S.%fZ` (six microsecond digits,
including `.000000` when the input has no fractional seconds). A `Z` or
`+00:00` input is accepted; a naive or non-UTC offset is refused. The same
durable input therefore produces identical bytes on replay.

The timestamp field is defined by `OutcomeReceipt` at
`executor/src/tpen_executor/results.py:151-199`; the final path reads the
receipt from the attempt output namespace.

The first canonical bytes at the task-level path win. An identical replay
returns the existing record. Different facts produce a typed fork/refusal and
never overwrite the file.

This is an upgrade/no-backfill boundary: a pre-PR completed attempt without a
durable task-spec snapshot, start provenance, or required successor ledger is
not backfilled into a completion record. A later fresh start may create these
facts for its own attempt; absence of trusted final-path facts remains absence
of completion.

Write-once serialization is process-level: the writer holds the task's POSIX
`lockf` advisory lock while reading, comparing, and publishing. Threads in one
process are not serialized by this protocol; they must treat a losing
same-fact publication as a replay/fork outcome rather than as evidence of
thread-level mutual exclusion.

## Rejections

Readers and writers fail closed on malformed/noncanonical JSON, missing or extra
fields, wrong schema/types, invalid identities or digests, a crossed task/group
or final attempt, missing/untyped/unknown/failed core evidence, a non-completed
typed verdict, under-target progress, missing/crossed TaskSpec or science
digest, non-iteration target units, disagreement between successor request
copies or request digests, crossed successor group/lineage, missing or mismatched checkpoint
path/content identity, missing/naive/non-UTC durable timestamp, disconnected/crossed
successor chain, checkpoint counter disagreement, cyclic chain, noncanonical
completion bytes, missing/crossed/unreadable start provenance, a non-fresh
chain root, an admitted attempt without its matching ledger link, and
conflicting bytes at the task-level path. A missing file is simply not
complete; it is not a park result.

