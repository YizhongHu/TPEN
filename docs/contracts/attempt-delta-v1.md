# Configured attempt delta v1

Status: normative for the C2.2 configured-program boundary. The executor
constructs this value after placement and passes it as one closed argument:

```text
--attempt-delta <canonical JSON>
```

The delta is not a recovery decision. It carries only the already-prepared
load and the physical destination for this attempt. Every attempt has a new
leaf and core directory. Recovery admission (B1-R2A/B) establishes the input
precondition that a `train_resume` path names a predecessor attempt's core.
The executor does not select or establish that lineage; it only requires the
path to be outside the current attempt leaf.

## Closed shape

The JSON object has exactly these fields:

```json
{"attempt_core_dir":"/store/groups/g/t/a/core","attempt_id":"a","group_digest":"g","load":{"mode":"fresh","path":null},"schema":"tpen-executor/attempt-delta-v1","task_id":"t"}
```

| Field | Type and rule |
| --- | --- |
| `schema` | string, exactly `tpen-executor/attempt-delta-v1` |
| `attempt_core_dir` | non-empty string; absolute, normalized path ending in `/<attempt>/core` |
| `load` | object with exactly `mode` and `path` |
| `load.mode` | string, exactly `fresh` or `train_resume`; absent mode is invalid |
| `load.path` | `null` iff mode is `fresh`; otherwise a non-empty absolute, normalized checkpoint path outside the current attempt leaf. The predecessor-core relationship is an input precondition from recovery admission, not an executor predicate. |
| `group_digest`, `task_id`, `attempt_id` | non-empty strings, each one path component; they must match the placement fence |

The executor derives `attempt_core_dir` from the fence's
`<store>/groups/<group>/<task>/<attempt>` leaf. Before any child is created,
all local, Parsl, and serialized carriers compare the delta's directory and
identities to that fence. A configured launch is admitted only when allocation
resolution supplies exactly one physical slot with a slot ID, host, and CPUs;
otherwise it is refused before child creation. A crossed output, program, or
fence is refused.

## Canonical bytes and digest

Canonical bytes are UTF-8 JSON with object keys sorted lexicographically at
every depth, separators `,` and `:`, no insignificant whitespace, and no ASCII
escaping (`ensure_ascii=false`). The example above is therefore exactly the
single-line byte sequence shown in the code block. Its SHA-256 is:

```text
95f504768be45378d0d6a9396a88fc85b6524bbfda6ced9b9723b2701f775159
```

The run/request payload digest is the SHA-256 of canonical JSON with exactly
`program_digest` (the configured-program digest) and `attempt_delta_digest`
(the SHA-256 above). The delta digest is not included in the configured-program
digest itself because the physical attempt identities are known only after
placement. The legacy TPEN route does not use this argument and retains its
byte-identical three `run.*` overrides.
