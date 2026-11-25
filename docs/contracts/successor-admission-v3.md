# Successor admission linked result v3

Status: normative for the executor's B1-R2B linked-result reader and writer.
This is a wire contract. A consumer MUST be able to parse the document's JSON
shape with the Python standard library (`json.loads`) and MUST NOT import
`tpen_executor` merely to discover the shape. The record is immutable and is
written only after a real R2 admission has returned `status: "admitted"`.

Full-horizon completion is outside this contract.

## Schema and closed top-level shape

The schema id is `tpen-executor/successor-admission-v3`.

The decoded JSON value MUST be an object with exactly these fields:

| Field | JSON type | Meaning |
| --- | --- | --- |
| `schema` | string | Exactly `tpen-executor/successor-admission-v3`. |
| `predecessor` | object | Typed predecessor identity and the selected checkpoint path. Its exact fields are `task_id`, `attempt_id`, `content_id`, and `checkpoint_path`. |
| `successor` | object | Reserved successor identity. Its exact fields are `task_id` and `attempt_id`. |
| `request_digest` | string | Lowercase 64-hex SHA-256 of the canonical `successor_request` mapping. |
| `core_acceptance_digest` | string | Lowercase 64-hex SHA-256 supplied by the caller for core's accepted request. The executor stores and compares it; it does not manufacture core acceptance. |
| `successor_binding` | object | Canonical `MemberInvocationV2` binding for the launched leaf. |
| `recovery_admission` | object | Canonical `RecoveryAdmissionV1` decision that produced this link. It MUST be admitted. |
| `successor_request` | object or `null` | The canonical executor-facing successor request used for this admission. R2 v3 publications MUST carry the object so a reader does not need a fabricated core outcome. |

The object is canonical JSON: a writer serializes the closed mapping with
sorted keys and no additional fields. A reader MUST reject missing, extra, or
wrong-typed fields rather than guessing.

## Typed fields

### `predecessor`

`predecessor` MUST have exactly the following string fields:

| Field | Type and rule |
| --- | --- |
| `task_id` | Non-empty logical task identifier. |
| `attempt_id` | The canonical textual mapping of the typed `AttemptId` (`AttemptId.value`). It is an identity, not a path and not a value parsed from a path. |
| `content_id` | Lowercase 64-hex SHA-256 content identity for the selected predecessor checkpoint. |
| `checkpoint_path` | Absolute, normalized checkpoint path selected by R2. It MUST equal `recovery_admission.checkpoint_path` byte-for-byte. |

The outer linked-result representation uses the canonical `AttemptId.value`
string for `predecessor.attempt_id`. The embedded typed identity records use
their own canonical identity mapping:

```json
{"schema":"tpen-executor/attempt-id-v1","value":"attempt-0001"}
```

Thus `recovery_admission.predecessor_attempt_id` and
`recovery_admission.successor_attempt_id` are objects with exactly `schema`
and `value`; the outer predecessor and successor identity fields remain
strings. A consumer MUST compare these typed identities by their canonical
`value`, not by directory-name parsing.

### `successor`

`successor` MUST have exactly two non-empty string fields:

| Field | Type and rule |
| --- | --- |
| `task_id` | Logical task identifier; it MUST match `successor_binding.task_id`. |
| `attempt_id` | Canonical textual `AttemptId.value`; it MUST match both the launched leaf identity and `successor_binding.attempt_id.value`. |

### `successor_binding`

`successor_binding` is the `MemberInvocationV2` JSON object with exactly:

| Field | JSON type |
| --- | --- |
| `schema` | string, exactly `tpen-executor/member-invocation-v2` |
| `allocation_attempt` | `AllocationAttemptV2` object |
| `group_digest` | lowercase 64-hex SHA-256 string |
| `task_id` | non-empty string |
| `lineage_id` | non-empty string |
| `attempt_id` | canonical `AttemptId` object (`schema`, `value`) |
| `invocation_id` | canonical `InvocationId` object (`schema`, `value`) |

`allocation_attempt` is the typed object with `schema`, `group_id`,
`allocation_attempt_id`, `allocation_id`, `backend_job_id`,
`controller_generation`, and `observed_at_utc`; the four `*_id` values are
canonical identity objects. This binding identifies the configured leaf and
the controller generation; it is not permission to launch by itself.

### `recovery_admission`

`recovery_admission` is a `RecoveryAdmissionV1` object with exactly:

| Field | JSON type and rule |
| --- | --- |
| `schema` | string, exactly `tpen-executor/recovery-admission-v1` |
| `status` | string, exactly `admitted` for v3 |
| `reason` | string, exactly `recovery-evidence-valid` for an admitted record |
| `evidence_digest` | lowercase 64-hex SHA-256 string |
| `predecessor_attempt_id` | canonical `AttemptId` object |
| `checkpoint_path` | absolute, normalized string selected by R2 |
| `successor_attempt_id` | canonical `AttemptId` object; required when admitted |
| `grant_id` | non-empty string for an admitted decision |

The embedded predecessor attempt and successor attempt are the typed source of
the link. The outer `predecessor.attempt_id` and `successor.attempt_id` MUST
equal their respective embedded `value` fields.

### `successor_request`

When present, `successor_request` is a JSON object with exactly the executor
request shape:

| Field | JSON type |
| --- | --- |
| `schema` | string, exactly `tpen.execution/successor-admission-request-v1` |
| `predecessor` | object with `task_id`, `attempt_id`, `core_outcome_digest`, and `resolved_config_identity` |
| `successor` | object with string `task_id` and string `attempt_id` |
| `science_input_digest` | lowercase 64-hex SHA-256 string |
| `target_iterations` | non-negative integer |
| `core_evidence` | JSON object containing the caller-supplied evidence mapping |

The nested request's `predecessor` and `successor` identity fields are strings;
`resolved_config_identity` is the object with `schema:
"tpen-config-identity-v1"` and lowercase 64-hex `sha256`. The request is
frozen into the record and its canonical digest MUST equal `request_digest`.

For an R2 record, the writer MUST bind this embedded request to the independent
`recovery_admission` before writing the ledger: both request task ids equal the
record task, the request predecessor and successor `attempt_id` values equal
the admission's typed `value` fields, and
`core_evidence.checkpoint_ref.content_id` and `checkpoint_dir` equal the
admission's selected `content_id` and `checkpoint_path`. The reader MUST
validate the same closed shape and equalities after recomputing
`request_digest`; a request that is individually well-formed but crossed with
the outer admission is unreadable.

## Crossed-link rejection

A decoder or store reader MUST fail closed on any crossed link. In particular,
it MUST reject a record when:

1. `predecessor.attempt_id` differs from
   `recovery_admission.predecessor_attempt_id.value`.
2. `predecessor.checkpoint_path` differs from
   `recovery_admission.checkpoint_path`.
3. `successor.attempt_id` differs from
   `recovery_admission.successor_attempt_id.value`.
4. `successor.task_id` or `successor.attempt_id` differs from the corresponding
   `successor_binding` identity.
5. `recovery_admission.status` is `parked`, the admission names no successor,
   or its reason is not `recovery-evidence-valid`.
6. The ledger task, predecessor start receipt, group, or launched output leaf
   disagrees with the record's identities.
7. The embedded request has extra/missing fields, its canonical request digest
   does not match `request_digest`, its predecessor/successor identities do not
   equal the outer/admitted identities, or its checkpoint content/path does
   not equal the admitted checkpoint facts.
8. The caller science/iteration intent does not match `successor_request`.

The linked result is not a core outcome. If the predecessor was hard-killed and
no core outcome exists, the reader uses the embedded admitted request and
records observed absence; it MUST NOT fabricate a core result to satisfy the
legacy continuation reader.

## Durable placement

For store root `<store>`, group digest `<group>`, task `<task>`, and typed
predecessor attempt `<predecessor-attempt>`, the record lives at:

```text
<store>/groups/<group>/<task>/successor-admissions/<predecessor-attempt>/successor-admission.json
```

The record is therefore in the predecessor task's successor-admission ledger,
not inside the successor attempt leaf. The predecessor start receipt remains
at:

```text
<store>/groups/<group>/<task>/<predecessor-attempt>/start.json
```

The consumer is invoked with the configured successor leaf
`<store>/groups/<group>/<task>/<successor-attempt>`. It selects the one ledger
record whose `successor.attempt_id` equals that leaf's typed attempt identity,
then validates the predecessor receipt and all crossed-link rules above. The
trusted store root comes from the live fence/store; it MUST NOT be inferred
from `checkpoint_path`.

