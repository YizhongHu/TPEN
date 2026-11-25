# Executor recovery admission v1

Status: normative for the B1-R2A executor-owned recovery evidence and
admission predicate. This contract is separate from the frozen
`attempt-delta-v1` transport shape. The transport is only cross-referenced by
name here and is unchanged.

The executor records operational facts and authorization. It does not create a
core acceptance, a `CoreAdmissionVerdict`, or a raw scheduler resume. The
legacy planned-yield selector remains unchanged; this predicate covers both a
planned yield and an operator-authorized unplanned end.

## Versioned records

All records are immutable, closed mappings. Nested identity records use their
own typed identity schemas. Program, group, H, content, and evidence identities
are lower-case SHA-256 strings. `science_digest` is different: it is an opaque,
non-empty value issued by HI and carried unchanged by the configured program,
for example `manifest_sha256:<manifest-sha256>:row:<row-id>`. The executor
does not derive or hash this value; admission compares the caller-supplied
checkpoint and grant values for equality only.

The HI caller obtains the configured-program inputs from
`executor/src/tpen_executor/configured_program.py` (`manifest_sha256` and
`row`) and constructs this token before creating the checkpoint facts and
finite grant. The same token is carried into both records; the executor only
validates that it is non-empty and compares the two values for equality.

### `RecoverySchedulerEndV1`

Schema id: `tpen-executor/recovery-scheduler-end-v1`.

| Field | Type and rule |
| --- | --- |
| `schema` | string, exactly the schema id above |
| `task_id` | non-empty string |
| `group_digest` | lower-case SHA-256 string |
| `attempt_id` | typed `AttemptId` mapping |
| `scheduler` | non-empty scheduler/backend name |
| `allocation_id` | non-empty scheduler allocation identity |
| `backend_job_id` | non-empty scheduler job identity |
| `terminal_state` | non-empty string; admission accepts only `completed`, `failed`, `timeout`, `cancelled`, `killed`, or `node_lost` |
| `terminal_at_utc` | timezone-aware ISO-8601 timestamp |
| `writer_end_identity` | non-empty writer/owner identity |
| `writer_end_at_utc` | timezone-aware ISO-8601 timestamp not before `terminal_at_utc` |

The scheduler identity is bound to the task, group, current attempt,
allocation, and backend job in the evidence record. A running, unknown, or
crossed observation is not terminal evidence.

### `RecoveryCheckpointFactsV1`

Schema id: `tpen-executor/recovery-checkpoint-facts-v1`.

| Field | Type and rule |
| --- | --- |
| `schema` | string, exactly the schema id above |
| `checkpoint_path` | absolute, normalized string; concrete checkpoint location |
| `content_id` | lower-case SHA-256 content identity |
| `next_iteration` | non-negative integer, retained independently |
| `completed_updates` | non-negative integer, retained independently |
| `program_digest` | lower-case SHA-256 program identity |
| `science_digest` | HI-issued science-lineage token, validated as a lowercase 64-hex SHA-256 (`TaskSpec.science_digest` via `sha256_text`, `contracts.py:166`) and carried unchanged from the configured program |
| `h_digest` | lower-case SHA-256 H identity |
| `committed` | boolean; true only for an independently validated committed checkpoint |

The facts are supplied only after an independent concrete checkpoint read and
validation. A changed path or `next_iteration` alone is not committed
progress. Real committed progress requires `committed == true`, a strictly
larger `completed_updates`, and a different `content_id` from the prior
checkpoint.

### `RecoveryGrantV1`

Schema id: `tpen-executor/recovery-grant-v1`.

| Field | Type and rule |
| --- | --- |
| `schema` | string, exactly the schema id above |
| `grant_id` | non-empty finite-campaign grant identity |
| `expires_at_utc` | timezone-aware ISO-8601 expiry timestamp |
| `group_digest` | lower-case SHA-256 group identity |
| `task_id` | non-empty frozen task identity |
| `program_digest` | lower-case SHA-256 program identity |
| `science_digest` | HI-issued science-lineage token, validated as a lowercase 64-hex SHA-256 (`TaskSpec.science_digest` via `sha256_text`, `contracts.py:166`); the finite grant carries the same value for equality comparison |
| `h_digest` | lower-case SHA-256 H identity |
| `successor_limit` | positive finite non-negative integer bound |

### `RecoveryEvidenceV1`

Schema id: `tpen-executor/recovery-evidence-v1`.

| Field | Type and rule |
| --- | --- |
| `schema` | string, exactly the schema id above |
| `mode` | exactly `planned_yield` or `operator_authorized_unplanned_end` |
| `task_id` | non-empty task identity |
| `group_digest` | lower-case SHA-256 group identity |
| `attempt_id` | typed current `AttemptId` |
| `predecessor_attempt_id` | typed predecessor `AttemptId`, different from `attempt_id` |
| `allocation_id` | expected non-empty allocation identity |
| `backend_job_id` | expected non-empty backend job identity |
| `expected_scheduler` | non-empty expected launch backend/scheduler namespace; the observed scheduler end must match it |
| `checkpoint` | typed `RecoveryCheckpointFactsV1` |
| `scheduler_end` | typed `RecoverySchedulerEndV1` |
| `grant` | typed `RecoveryGrantV1` or `null`; `null` is default-OFF |
| `budget_remaining` | non-negative integer; zero parks |
| `no_progress_threshold` | positive integer |
| `no_progress_count` | non-negative integer count before this decision |
| `prior_content_id` | lower-case SHA-256 string or `null` |
| `prior_completed_updates` | non-negative integer |
| `prior_next_iteration` | non-negative integer, retained separately |
| `external_submit_state` | exactly `not_submitted`, `acknowledged`, or `ambiguous`; `ambiguous` parks and is never retried |

### `RecoveryAdmissionV1`

Schema id: `tpen-executor/recovery-admission-v1`.

| Field | Type and rule |
| --- | --- |
| `schema` | string, exactly the schema id above |
| `status` | exactly `admitted` or `parked` |
| `reason` | `recovery-evidence-valid` for admission, otherwise one park code below |
| `evidence_digest` | lower-case SHA-256 digest of canonical v1 evidence |
| `predecessor_attempt_id` | typed predecessor `AttemptId` |
| `checkpoint_path` | absolute, normalized typed predecessor checkpoint path |
| `successor_attempt_id` | typed successor `AttemptId`, required when admitted; otherwise `null` |
| `grant_id` | non-empty grant id when present; `null` only without a grant |

The admission record exposes `predecessor_attempt_id` and `checkpoint_path` as
typed fields. B1-R2B publishes these same fields through the versioned linked
result `tpen-executor/successor-admission-v3`; consumers read the typed fields
and do not parse path strings to recover lineage. The v3 record also carries
the admitted R2 decision and the canonical successor request so C1 can read an
unplanned-end admission without fabricating a core outcome.

## Admission predicate

`admit_recovery` requires the caller's trusted `trusted_store_root` fence/store
root and enforces all of the following before returning `admitted`:

1. The scheduler end binds to the exact task, group, current attempt,
   allocation, and backend job, has an accepted terminal state, and includes a
   writer-end identity and timestamp.
2. The checkpoint is committed and independently validated.
3. The checkpoint path is strictly below the exact component path
   `<trusted_store_root>/groups/<group>/<task>/<predecessor_attempt>/core`.
   The root is supplied by the caller's trusted fence/store, never derived
   from the checkpoint path. The current attempt leaf is excluded
   independently. A path in an unrelated group/task, nested store layout, or
   different store parks.
4. Program and H digests match the finite grant. The HI-issued opaque
   `science_digest` carried by the configured program matches the checkpoint
   and finite grant by equality; the executor does not derive this value.
5. A grant exists, is unexpired, and has remaining successor capacity; budget
   remaining must be positive. No grant means default-OFF.
6. A repeated identical progress observation parks when
   `no_progress_count >= no_progress_threshold`. Only the explicit committed
   progress rule above resets that accounting.
7. An ambiguous external submit parks and cannot trigger a retry.

The existing task lock, predecessor-to-successor uniqueness, attempt fence,
controller generation, launch intent, and ambiguous-ack handling remain the
operational mechanisms around this pure predicate.

## Park reason codes

The closed park codes are:

`unknown-scheduler-end`, `scheduler-identity-mismatch`, `writers-not-ended`,
`checkpoint-unavailable`, `checkpoint-corrupt-or-ambiguous`,
`checkpoint-lineage-mismatch`, `checkpoint-is-current-attempt`,
`program-lineage-mismatch`, `science-lineage-mismatch`, `h-lineage-mismatch`,
`grant-missing`, `grant-expired`, `budget-exhausted`, `no-progress-exhausted`,
`ambiguous-external-submit`, and `invalid-recovery-evidence`.

Unknown, mismatched, unavailable, corrupt, or ambiguous facts are parks, not
product success and not permission to retry.
