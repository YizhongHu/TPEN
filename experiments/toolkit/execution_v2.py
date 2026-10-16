"""Experiment-side v2 ready-batch data adapter.

This module is intentionally data-only.  It serializes existing experiment
plans and dispatch rows without importing :mod:`tpen` or any core execution
module.  The core receives the resulting bytes through a launcher boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

# This literal is intentionally duplicated rather than imported from the core:
# experiments are the producer side of the launcher boundary and must remain
# importable without core execution dependencies.
READY_BATCH_ADAPTER_SCHEMA = "tpen.execution/ready-batch-v1"


def _canonical(payload: Any) -> bytes:
    """Return canonical JSON bytes for adapter-owned data."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(payload: Any) -> str:
    """Return the adapter payload digest."""

    return sha256(_canonical(payload)).hexdigest()


def ready_batch_v2(
    *,
    lineage_id: str,
    batch_id: str,
    source_config_digest: str,
    science_input_digest: str,
    target_iterations: int,
    backend: str,
    recovery: str,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    backend_options: Mapping[str, Any] | None = None,
    recovery_options: Mapping[str, Any] | None = None,
    recovery_selection_digest: str | None = None,
) -> dict[str, Any]:
    """Build a plain v2 ready-batch mapping from experiment data.

    Parameters
    ----------
    lineage_id, batch_id : str
        Opaque experiment identities; the adapter does not parse them.
    source_config_digest : str
        Digest of the source syntax captured by the experiment.
    science_input_digest : str
        Digest of the immutable science input captured by the experiment.
    target_iterations : int
        Fixed total target horizon for the execution lineage.
    backend, recovery : str
        Explicit operational lego selections, separate from ``payload``.
    payload : mapping
        Exact serialized science intent.  It remains data and is not resolved.
    rows : sequence of mappings
        Already materialized v2 task/dispatch rows.
    """

    if not isinstance(lineage_id, str) or not lineage_id:
        raise ValueError("lineage_id must be non-empty")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("batch_id must be non-empty")
    if not isinstance(source_config_digest, str) or not source_config_digest:
        raise ValueError("source_config_digest must be non-empty")
    if not isinstance(science_input_digest, str) or not science_input_digest:
        raise ValueError("science_input_digest must be non-empty")
    if type(target_iterations) is not int or target_iterations < 0:
        raise ValueError("target_iterations must be a nonnegative integer")
    if not isinstance(backend, str) or not backend:
        raise ValueError("backend must be non-empty")
    if not isinstance(recovery, str) or not recovery:
        raise ValueError("recovery must be non-empty")
    copied_rows = [dict(row) for row in rows]
    identities = [row.get("invocation_id") for row in copied_rows]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ValueError("every v2 row needs an opaque invocation_id")
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate invocation_id in v2 ready batch")
    payload_copy = json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False))
    items = []
    for row in copied_rows:
        for name in ("logical_task_id", "fresh_output_id", "result_dir"):
            value = row.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"v2 row {name} must be a non-empty string")
        argv = row["argv"]
        if not isinstance(argv, (list, tuple)) or not argv or any(
            not isinstance(part, str) or not part for part in argv
        ):
            raise ValueError("v2 row argv must be a non-empty sequence of strings")
        items.append({
            "invocation_id": row["invocation_id"],
            "logical_task_id": row["logical_task_id"],
            "fresh_output_id": row["fresh_output_id"],
            "argv": list(argv),
            "result_dir": row["result_dir"],
            "payload": row.get("payload", {}),
        })
    digest = _digest(payload_copy)
    return {
        "schema": READY_BATCH_ADAPTER_SCHEMA,
        "lineage_id": lineage_id,
        "batch_id": batch_id,
        "source_config_digest": source_config_digest,
        "payload_digest": digest,
        "science_input_digest": science_input_digest,
        "target_iterations": target_iterations,
        "execution": {
            "schema": "tpen.execution/config-v1",
            "lineage_id": lineage_id,
            "backend": backend,
            "recovery": recovery,
            "source_config_digest": source_config_digest,
            "payload_digest": digest,
            "recovery_selection_digest": recovery_selection_digest,
            "backend_options": dict(backend_options or {}),
            "recovery_options": dict(recovery_options or {}),
        },
        "payload": payload_copy,
        "items": items,
    }


def encode_ready_batch_v2(batch: Mapping[str, Any]) -> bytes:
    """Serialize an adapter mapping deterministically after validation."""

    if batch.get("schema") != READY_BATCH_ADAPTER_SCHEMA:
        raise ValueError("unsupported v2 ready-batch adapter schema")
    expected = _digest(batch.get("payload"))
    if batch.get("payload_digest") != expected:
        raise ValueError("v2 payload digest does not match payload")
    return _canonical(batch)


def decode_ready_batch_v2(data: bytes | str | Mapping[str, Any]) -> dict[str, Any]:
    """Deserialize and validate adapter-owned v2 data without core imports."""

    batch = dict(data) if isinstance(data, Mapping) else json.loads(data)
    if not isinstance(batch, dict):
        raise ValueError("v2 ready batch must be an object")
    encoded = encode_ready_batch_v2(batch)
    return json.loads(encoded)


__all__ = [
    "READY_BATCH_ADAPTER_SCHEMA",
    "decode_ready_batch_v2",
    "encode_ready_batch_v2",
    "ready_batch_v2",
]

