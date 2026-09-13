# Cannon trunk corpus baseline

`tools/corpus_baseline.py` is a reusable, allocation-only measurement harness.
The source revision is supplied at runtime; the harness does not embed a
baseline count or node set.

Run it from a Cannon `test` allocation with a unique run root, per-arm
`UV_PROJECT_ENVIRONMENT`/`UV_CACHE_DIR`, and a full clone source. The command
must include `--source` pointing to the staged source repository and
`--revision` set to the trunk SHA being measured. The resulting
`corpus-result.json`, evidence directory, XML, logs, identity block, and
command provenance are the measurement record. Keep the raw evidence.

The harness performs collect-only and full pytest instruments without pytest
path arguments, derives its floor from the collected set in the allocation,
checks JUnit `(classname, name)` identities without requiring a `file`
attribute, records independent inner statuses, and runs a deliberate red
control in the same allocation. It asserts the allocation, exact revision,
tag visibility, interpreter/Torch identity, and explicit flushes every
machine-readable result before returning.
