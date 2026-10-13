"""Immutable ready-batch data and its versioned codec.

This module carries already-materialized execution data.  It does not submit
work, import a backend, or instantiate a scientific object.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import EXECUTION_CONFIG_SCHEMA, ExecutionConfig, canonical_payload_bytes, payload_digest
from .recovery import _freeze, _thaw

READY_BATCH_SCHEMA = "tpen.execution/ready-batch-v1"


def _text(value: Any, name: str) -> str:
    """Return a required non-empty text field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest(value: Any, name: str) -> str:
    """Validate a lowercase SHA-256 digest."""

    value = _text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ReadyItem:
    """One immutable invocation-ready row in a batch."""

    invocation_id: str
    logical_task_id: str
    fresh_output_id: str
    argv: tuple[str, ...]
    result_dir: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("invocation_id", "logical_task_id", "fresh_output_id", "result_dir"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        argv = tuple(self.argv)
        if not argv or any(not isinstance(part, str) or not part for part in argv):
            raise ValueError("argv must be a non-empty sequence of non-empty strings")
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "payload", _freeze(self.payload, path="item.payload"))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe item mapping."""

        return {
            "invocation_id": self.invocation_id,
            "logical_task_id": self.logical_task_id,
            "fresh_output_id": self.fresh_output_id,
            "argv": list(self.argv),
            "result_dir": self.result_dir,
            "payload": _thaw(self.payload),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReadyItem":
        """Construct an item from its serialized mapping."""

        if not isinstance(data, Mapping):
            raise TypeError("ready-batch item must be a mapping")
        return cls(
            invocation_id=data.get("invocation_id", ""),
            logical_task_id=data.get("logical_task_id", ""),
            fresh_output_id=data.get("fresh_output_id", ""),
            argv=tuple(data.get("argv", ())),
            result_dir=data.get("result_dir", ""),
            payload=data.get("payload", {}),
        )


@dataclass(frozen=True, slots=True)
class ReadyBatch:
    """Immutable, versioned batch accepted by a later execution layer.

    ``payload`` is the exact science-independent serialized intent handed to
    the execution layer.  Its digest is checked on every construction so a
    changed payload cannot reuse a lineage identity silently.
    """

    lineage_id: str
    batch_id: str
    source_config_digest: str
    payload_digest: str
    science_input_digest: str
    target_iterations: int
    execution: ExecutionConfig
    items: tuple[ReadyItem, ...]
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema: str = READY_BATCH_SCHEMA

    def __post_init__(self) -> None:
        for name in ("lineage_id", "batch_id", "science_input_digest", "source_config_digest", "payload_digest"):
            value = getattr(self, name)
            if name.endswith("digest"):
                value = _digest(value, name)
            else:
                value = _text(value, name)
            object.__setattr__(self, name, value)
        if type(self.target_iterations) is not int or self.target_iterations < 0:
            raise ValueError("target_iterations must be a nonnegative integer")
        if not isinstance(self.execution, ExecutionConfig):
            raise TypeError("execution must be an ExecutionConfig")
        if self.execution.lineage_id != self.lineage_id:
            raise ValueError("execution lineage_id does not match ready batch")
        if self.execution.source_config_digest != self.source_config_digest:
            raise ValueError("execution source_config_digest does not match ready batch")
        if self.execution.payload_digest != self.payload_digest:
            raise ValueError("execution payload_digest does not match ready batch")
        if self.schema != READY_BATCH_SCHEMA:
            raise ValueError(f"unsupported ready-batch schema: {self.schema!r}")
        payload = _freeze(self.payload, path="payload")
        if payload_digest(payload) != self.payload_digest:
            raise ValueError("payload_digest does not match the ready-batch payload")
        object.__setattr__(self, "payload", payload)
        items = tuple(item if isinstance(item, ReadyItem) else ReadyItem.from_mapping(item) for item in self.items)
        if not items:
            raise ValueError("ready batch must contain at least one item")
        identities = [item.invocation_id for item in items]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate invocation_id in ready batch")
        outputs = [item.fresh_output_id for item in items]
        if len(outputs) != len(set(outputs)):
            raise ValueError("duplicate fresh_output_id in ready batch")
        object.__setattr__(self, "items", items)

    @property
    def identity(self) -> str:
        """Return the stable digest of the serialized batch."""

        return payload_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic JSON-safe batch mapping."""

        return {
            "schema": self.schema,
            "lineage_id": self.lineage_id,
            "batch_id": self.batch_id,
            "source_config_digest": self.source_config_digest,
            "payload_digest": self.payload_digest,
            "science_input_digest": self.science_input_digest,
            "target_iterations": self.target_iterations,
            "execution": self.execution.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "payload": _thaw(self.payload),
        }

    def encode(self) -> bytes:
        """Return canonical UTF-8 JSON bytes for durable transport."""

        return canonical_payload_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReadyBatch":
        """Construct a batch and verify all identity bindings."""

        if not isinstance(data, Mapping):
            raise TypeError("ready batch must be a mapping")
        execution = ExecutionConfig.from_mapping(data.get("execution", {}))
        return cls(
            schema=data.get("schema", ""),
            lineage_id=data.get("lineage_id", ""),
            batch_id=data.get("batch_id", ""),
            source_config_digest=data.get("source_config_digest", ""),
            payload_digest=data.get("payload_digest", ""),
            science_input_digest=data.get("science_input_digest", ""),
            target_iterations=data.get("target_iterations", -1),
            execution=execution,
            items=tuple(ReadyItem.from_mapping(row) for row in data.get("items", ())),
            payload=data.get("payload", {}),
        )

    @classmethod
    def decode(cls, data: bytes | str | Mapping[str, Any]) -> "ReadyBatch":
        """Decode canonical JSON or a mapping into a ready batch."""

        if isinstance(data, Mapping):
            payload = data
        else:
            payload = json.loads(data)
        return cls.from_mapping(payload)


def encode_ready_batch(batch: ReadyBatch) -> bytes:
    """Encode one validated batch."""

    return batch.encode()


def decode_ready_batch(data: bytes | str | Mapping[str, Any]) -> ReadyBatch:
    """Decode one validated batch."""

    return ReadyBatch.decode(data)


def decode_ready_batch_wire(data: bytes | str | Mapping[str, Any]) -> ReadyBatch:
    """Read the versioned launcher wire format into a validated batch.

    The experiment adapter emits the same ``READY_BATCH_SCHEMA`` bytes
    without importing this package.  The named core reader makes that
    launcher boundary explicit while retaining one validation path.
    """

    return decode_ready_batch(data)


def require_lineage_payload(
    lineage_payloads: Mapping[str, str], *, lineage_id: str, payload: Any
) -> str:
    """Bind a payload to a lineage, refusing a changed reused payload.

    The caller owns the durable mapping.  This helper only applies the
    fail-closed comparison and never mutates that mapping.
    """

    lineage_id = _text(lineage_id, "lineage_id")
    digest = payload_digest(payload)
    previous = lineage_payloads.get(lineage_id)
    if previous is not None and previous != digest:
        raise ValueError(f"lineage {lineage_id!r} is already bound to a different payload")
    return digest


__all__ = [
    "READY_BATCH_SCHEMA",
    "ReadyBatch",
    "ReadyItem",
    "decode_ready_batch",
    "decode_ready_batch_wire",
    "encode_ready_batch",
    "require_lineage_payload",
]

