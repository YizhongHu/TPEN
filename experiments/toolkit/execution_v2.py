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


def _text(value: Any, name: str) -> str:
    """Validate a required non-whitespace-only text field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest_field(value: Any, name: str) -> str:
    """Validate a lowercase SHA-256 digest without importing the core."""

    value = _text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _closed_mapping(
    data: Any, required: Mapping[str, type], *,
    optional: Mapping[str, type | tuple[type, ...]] | None = None, name: str,
) -> Mapping[str, Any]:
    """Validate adapter wire fields without crossing the core import boundary."""

    if not isinstance(data, Mapping):
        raise ValueError(f"{name} must be a mapping")
    fields = {**required, **(optional or {})}
    if set(data) - fields.keys() or set(required) - data.keys():
        raise ValueError(f"{name} has unknown or missing fields")
    for key, value in data.items():
        expected = fields[key]
        valid = isinstance(value, expected)
        if expected is int:
            valid = type(value) is int
        elif expected is Sequence:
            valid = valid and not isinstance(value, (str, bytes, bytearray))
        if not valid:
            raise ValueError(f"{name}.{key} has an invalid type")
    return data


def _row(data: Any) -> Mapping[str, Any]:
    """Read one closed adapter row before any field lookup."""

    return _closed_mapping(data, {
        "invocation_id": str, "logical_task_id": str, "fresh_output_id": str,
        "argv": Sequence, "result_dir": str,
    }, optional={"payload": Mapping}, name="v2 row")


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

    lineage_id = _text(lineage_id, "lineage_id")
    batch_id = _text(batch_id, "batch_id")
    source_config_digest = _digest_field(source_config_digest, "source_config_digest")
    science_input_digest = _digest_field(science_input_digest, "science_input_digest")
    if type(target_iterations) is not int or target_iterations < 0:
        raise ValueError("target_iterations must be a nonnegative integer")
    backend = _text(backend, "backend")
    recovery = _text(recovery, "recovery")
    if recovery_selection_digest is not None:
        recovery_selection_digest = _digest_field(
            recovery_selection_digest, "recovery_selection_digest"
        )
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise ValueError("v2 rows must be a sequence")
    copied_rows = [dict(_row(row)) for row in rows]
    identities = [row.get("invocation_id") for row in copied_rows]
    if any(not isinstance(identity, str) or not identity.strip() for identity in identities):
        raise ValueError("every v2 row needs an opaque invocation_id")
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate invocation_id in v2 ready batch")
    outputs = [row.get("fresh_output_id") for row in copied_rows]
    if len(outputs) != len(set(outputs)):
        raise ValueError("duplicate fresh_output_id in v2 ready batch")
    payload_copy = json.loads(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False))
    items = []
    for row in copied_rows:
        for name in ("logical_task_id", "fresh_output_id", "result_dir"):
            value = row.get(name)
            if not isinstance(value, str) or not value.strip():
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

    _closed_mapping(batch, {
        "schema": str, "lineage_id": str, "batch_id": str,
        "source_config_digest": str, "science_input_digest": str,
        "payload_digest": str, "target_iterations": int,
        "execution": Mapping, "payload": Mapping, "items": Sequence,
    }, name="v2 ready batch")
    if batch["schema"] != READY_BATCH_ADAPTER_SCHEMA:
        raise ValueError("unsupported v2 ready-batch adapter schema")
    expected = _digest(batch["payload"])
    if batch["payload_digest"] != expected:
        raise ValueError("v2 payload digest does not match payload")
    execution = _closed_mapping(batch["execution"], {
        "schema": str, "lineage_id": str, "backend": str, "recovery": str,
        "source_config_digest": str, "payload_digest": str,
    }, optional={
        "recovery_selection_digest": (str, type(None)),
        "backend_options": Mapping, "recovery_options": Mapping,
    }, name="v2 execution")
    if execution["schema"] != "tpen.execution/config-v1":
        raise ValueError("unsupported v2 execution schema")
    rebuilt = ready_batch_v2(
        lineage_id=batch["lineage_id"], batch_id=batch["batch_id"],
        source_config_digest=batch["source_config_digest"],
        science_input_digest=batch["science_input_digest"],
        target_iterations=batch["target_iterations"], backend=execution["backend"],
        recovery=execution["recovery"], payload=batch["payload"], rows=batch["items"],
        recovery_selection_digest=execution.get("recovery_selection_digest"),
        backend_options=execution.get("backend_options"), recovery_options=execution.get("recovery_options"),
    )
    if _canonical(rebuilt) != _canonical(batch):
        raise ValueError("v2 execution binding does not match ready batch")
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

