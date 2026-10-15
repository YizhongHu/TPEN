"""Backend-neutral operational receipt contracts.

Receipts attest observations and bindings; they do not start workers, call a
backend API, or infer scientific success from scheduler state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from .batch import ReadyBatch
from .config import canonical_payload_bytes, payload_digest
from .recovery import _freeze, _thaw


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
class BackendJobReceipt:
    """Receipt that a backend accepted one invocation."""

    schema: ClassVar[str] = "tpen.execution/receipt-backend-job-v1"
    lineage_id: str
    batch_id: str
    invocation_id: str
    backend: str
    job_id: str
    ready_batch_digest: str
    source_config_digest: str
    payload_digest: str

    def __post_init__(self) -> None:
        for name in ("lineage_id", "batch_id", "invocation_id", "backend", "job_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("ready_batch_digest", "source_config_digest", "payload_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialized receipt mapping."""

        return {"schema": self.schema, **{name: getattr(self, name) for name in (
            "lineage_id", "batch_id", "invocation_id", "backend", "job_id",
            "ready_batch_digest", "source_config_digest", "payload_digest",
        )}}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BackendJobReceipt":
        """Deserialize and validate a backend-job receipt."""

        _check_schema(data, cls.schema)
        return cls(**{name: data.get(name, "") for name in (
            "lineage_id", "batch_id", "invocation_id", "backend", "job_id",
            "ready_batch_digest", "source_config_digest", "payload_digest",
        )})


@dataclass(frozen=True, slots=True)
class AllocationObservation:
    """Receipt of an observed allocation, independent of a backend API."""

    schema: ClassVar[str] = "tpen.execution/receipt-allocation-v1"
    lineage_id: str
    allocation_id: str
    observed_at_utc: str
    host: str
    partition: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("lineage_id", "allocation_id", "observed_at_utc", "host", "partition"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "metadata", _freeze(self.metadata, path="allocation.metadata"))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialized allocation observation."""

        return {"schema": self.schema, "lineage_id": self.lineage_id, "allocation_id": self.allocation_id,
                "observed_at_utc": self.observed_at_utc, "host": self.host, "partition": self.partition,
                "metadata": _thaw(self.metadata)}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AllocationObservation":
        """Deserialize and validate an allocation observation."""

        _check_schema(data, cls.schema)
        return cls(
            lineage_id=data.get("lineage_id", ""), allocation_id=data.get("allocation_id", ""),
            observed_at_utc=data.get("observed_at_utc", ""), host=data.get("host", ""),
            partition=data.get("partition", ""), metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class InvocationReceipt:
    """Receipt tying an invocation to one ready batch and backend job."""

    schema: ClassVar[str] = "tpen.execution/receipt-invocation-v1"
    lineage_id: str
    batch_id: str
    invocation_id: str
    job_id: str
    fresh_output_id: str
    ready_batch_digest: str
    status: str

    def __post_init__(self) -> None:
        for name in ("lineage_id", "batch_id", "invocation_id", "job_id", "fresh_output_id", "status"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "ready_batch_digest", _digest(self.ready_batch_digest, "ready_batch_digest"))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialized invocation receipt."""

        return {"schema": self.schema, **{name: getattr(self, name) for name in (
            "lineage_id", "batch_id", "invocation_id", "job_id", "fresh_output_id",
            "ready_batch_digest", "status",
        )}}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "InvocationReceipt":
        """Deserialize and validate an invocation receipt."""

        _check_schema(data, cls.schema)
        return cls(**{name: data.get(name, "") for name in (
            "lineage_id", "batch_id", "invocation_id", "job_id", "fresh_output_id",
            "ready_batch_digest", "status",
        )})


@dataclass(frozen=True, slots=True)
class FreshOutputReceipt:
    """Receipt naming the fresh output namespace for an invocation."""

    schema: ClassVar[str] = "tpen.execution/receipt-fresh-output-v1"
    lineage_id: str
    batch_id: str
    invocation_id: str
    fresh_output_id: str
    output_path: str
    payload_digest: str

    def __post_init__(self) -> None:
        for name in ("lineage_id", "batch_id", "invocation_id", "fresh_output_id", "output_path"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "payload_digest", _digest(self.payload_digest, "payload_digest"))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialized fresh-output receipt."""

        return {"schema": self.schema, **{name: getattr(self, name) for name in (
            "lineage_id", "batch_id", "invocation_id", "fresh_output_id", "output_path", "payload_digest",
        )}}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FreshOutputReceipt":
        """Deserialize and validate a fresh-output receipt."""

        _check_schema(data, cls.schema)
        return cls(**{name: data.get(name, "") for name in (
            "lineage_id", "batch_id", "invocation_id", "fresh_output_id", "output_path", "payload_digest",
        )})


@dataclass(frozen=True, slots=True)
class ResultReceipt:
    """Receipt of a result bound to one fresh output and ready payload."""

    schema: ClassVar[str] = "tpen.execution/receipt-result-v1"
    lineage_id: str
    batch_id: str
    invocation_id: str
    fresh_output_id: str
    result_receipt_id: str
    payload_digest: str
    result_digest: str
    status: str

    def __post_init__(self) -> None:
        for name in ("lineage_id", "batch_id", "invocation_id", "fresh_output_id", "result_receipt_id", "status"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("payload_digest", "result_digest"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialized result receipt."""

        return {"schema": self.schema, **{name: getattr(self, name) for name in (
            "lineage_id", "batch_id", "invocation_id", "fresh_output_id", "result_receipt_id",
            "payload_digest", "result_digest", "status",
        )}}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ResultReceipt":
        """Deserialize and validate a result receipt."""

        _check_schema(data, cls.schema)
        return cls(**{name: data.get(name, "") for name in (
            "lineage_id", "batch_id", "invocation_id", "fresh_output_id", "result_receipt_id",
            "payload_digest", "result_digest", "status",
        )})


def _check_schema(data: Mapping[str, Any], expected: str) -> None:
    """Reject a receipt serialized under another schema."""

    if not isinstance(data, Mapping) or data.get("schema") != expected:
        raise ValueError(f"receipt schema is not {expected!r}")


def validate_receipt_binding(
    receipt: BackendJobReceipt | InvocationReceipt | FreshOutputReceipt | ResultReceipt,
    batch: ReadyBatch,
) -> None:
    """Verify receipt identities and payload bindings against a ready batch."""

    if receipt.lineage_id != batch.lineage_id or receipt.batch_id != batch.batch_id:
        raise ValueError("receipt is bound to a different lineage or batch")
    if isinstance(receipt, (BackendJobReceipt, InvocationReceipt)) and receipt.ready_batch_digest != batch.identity:
        raise ValueError("receipt is bound to a different ready batch")
    if isinstance(receipt, BackendJobReceipt) and receipt.source_config_digest != batch.source_config_digest:
        raise ValueError("receipt is bound to a different source configuration")
    if isinstance(receipt, BackendJobReceipt) and receipt.backend != batch.execution.backend:
        raise ValueError("receipt names a different backend")
    if isinstance(receipt, (BackendJobReceipt, FreshOutputReceipt, ResultReceipt)) and receipt.payload_digest != batch.payload_digest:
        raise ValueError("receipt is bound to a different payload")
    if hasattr(receipt, "invocation_id"):
        try:
            item = next(item for item in batch.items if item.invocation_id == receipt.invocation_id)
        except StopIteration as exc:
            raise ValueError("receipt names an invocation absent from the ready batch") from exc
        if hasattr(receipt, "fresh_output_id") and receipt.fresh_output_id != item.fresh_output_id:
            raise ValueError("receipt pairs an invocation with a different fresh output")


def encode_receipt(receipt: Any) -> bytes:
    """Encode one receipt mapping canonically."""

    if not hasattr(receipt, "to_dict"):
        raise TypeError("receipt must provide to_dict")
    return canonical_payload_bytes(receipt.to_dict())


__all__ = [
    "AllocationObservation",
    "BackendJobReceipt",
    "FreshOutputReceipt",
    "InvocationReceipt",
    "ResultReceipt",
    "encode_receipt",
    "validate_receipt_binding",
]

