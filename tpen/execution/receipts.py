"""Backend-neutral operational receipt contracts.

Receipts attest observations and bindings; they do not start workers, call a
backend API, or infer scientific success from scheduler state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ._codec import decode_fields
from .batch import ReadyBatch
from .config import canonical_payload_bytes
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
    def from_mapping(cls, data: Any) -> "BackendJobReceipt":
        """Deserialize a closed, typed receipt record."""

        fields = (
            "lineage_id",
            "batch_id",
            "invocation_id",
            "backend",
            "job_id",
            "ready_batch_digest",
            "source_config_digest",
            "payload_digest",
        )
        decode_fields(data, dict.fromkeys(fields, str), schema=cls.schema, name="BackendJobReceipt")
        return cls(**{name: data[name] for name in fields})


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
    def from_mapping(cls, data: Any) -> "AllocationObservation":
        """Deserialize a closed, typed receipt record."""

        fields = (
            "lineage_id",
            "allocation_id",
            "observed_at_utc",
            "host",
            "partition",
        )
        decode_fields(data, dict.fromkeys(fields, str), optional={"metadata": Mapping},
                      schema=cls.schema, name="AllocationObservation")
        return cls(**{name: data[name] for name in fields}, metadata=data.get("metadata", {}))


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
    def from_mapping(cls, data: Any) -> "InvocationReceipt":
        """Deserialize a closed, typed receipt record."""

        fields = (
            "lineage_id",
            "batch_id",
            "invocation_id",
            "job_id",
            "fresh_output_id",
            "ready_batch_digest",
            "status",
        )
        decode_fields(data, dict.fromkeys(fields, str), schema=cls.schema, name="InvocationReceipt")
        return cls(**{name: data[name] for name in fields})


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
    def from_mapping(cls, data: Any) -> "ResultReceipt":
        """Deserialize a closed, typed receipt record."""

        fields = (
            "lineage_id",
            "batch_id",
            "invocation_id",
            "fresh_output_id",
            "result_receipt_id",
            "payload_digest",
            "result_digest",
            "status",
        )
        decode_fields(data, dict.fromkeys(fields, str), schema=cls.schema, name="ResultReceipt")
        return cls(**{name: data[name] for name in fields})


def validate_receipt_binding(
    receipt: BackendJobReceipt | InvocationReceipt | ResultReceipt,
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
    if isinstance(receipt, (BackendJobReceipt, ResultReceipt)) and receipt.payload_digest != batch.payload_digest:
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
    "InvocationReceipt",
    "ResultReceipt",
    "encode_receipt",
    "validate_receipt_binding",
]

