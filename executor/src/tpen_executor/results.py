"""Operational receipts and the executor run manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ._codec import canonical_bytes, decode_json, digest, fields, sequence, sha256_text, text, thaw
from .contracts import AllocationId, AttemptId, BackendJobId, ReadyGroup


def _identity(value: Any, cls: type, name: str) -> Any:
    if not isinstance(value, cls):
        raise TypeError(f"{name} must be {cls.__name__}")
    return value


@dataclass(frozen=True, slots=True)
class StartReceipt:
    """Durable observation that an attempt was started."""

    schema: ClassVar[str] = "tpen-executor/start-receipt-v1"
    task_id: str
    attempt_id: AttemptId
    allocation_id: AllocationId
    backend_job_id: BackendJobId
    started_at_utc: str
    output_dir: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", text(self.task_id, "task_id"))
        _identity(self.attempt_id, AttemptId, "attempt_id"); _identity(self.allocation_id, AllocationId, "allocation_id"); _identity(self.backend_job_id, BackendJobId, "backend_job_id")
        object.__setattr__(self, "started_at_utc", text(self.started_at_utc, "started_at_utc")); object.__setattr__(self, "output_dir", text(self.output_dir, "output_dir"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "task_id": self.task_id, "attempt_id": self.attempt_id.value, "allocation_id": self.allocation_id.value, "backend_job_id": self.backend_job_id.value, "started_at_utc": self.started_at_utc, "output_dir": self.output_dir}

    @classmethod
    def from_mapping(cls, data: Any) -> "StartReceipt":
        fields(data, {"schema", "task_id", "attempt_id", "allocation_id", "backend_job_id", "started_at_utc", "output_dir"}, cls.schema, "StartReceipt")
        if any(type(data[key]) is not str for key in ("task_id", "attempt_id", "allocation_id", "backend_job_id", "started_at_utc", "output_dir")): raise TypeError("StartReceipt has an invalid field type")
        return cls(data["task_id"], AttemptId(data["attempt_id"]), AllocationId(data["allocation_id"]), BackendJobId(data["backend_job_id"]), data["started_at_utc"], data["output_dir"])


@dataclass(frozen=True, slots=True)
class OutcomeReceipt:
    """Durable observation of an attempt's operational outcome."""

    schema: ClassVar[str] = "tpen-executor/outcome-receipt-v1"
    task_id: str
    attempt_id: AttemptId
    finished_at_utc: str
    status: str
    output_dir: str
    result_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", text(self.task_id, "task_id")); _identity(self.attempt_id, AttemptId, "attempt_id")
        object.__setattr__(self, "finished_at_utc", text(self.finished_at_utc, "finished_at_utc")); object.__setattr__(self, "status", text(self.status, "status")); object.__setattr__(self, "output_dir", text(self.output_dir, "output_dir"))
        if self.result_digest is not None: object.__setattr__(self, "result_digest", sha256_text(self.result_digest, "result_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "task_id": self.task_id, "attempt_id": self.attempt_id.value, "finished_at_utc": self.finished_at_utc, "status": self.status, "output_dir": self.output_dir, "result_digest": self.result_digest}

    @classmethod
    def from_mapping(cls, data: Any) -> "OutcomeReceipt":
        fields(data, {"schema", "task_id", "attempt_id", "finished_at_utc", "status", "output_dir", "result_digest"}, cls.schema, "OutcomeReceipt")
        if any(type(data[key]) is not str for key in ("task_id", "attempt_id", "finished_at_utc", "status", "output_dir")) or (data["result_digest"] is not None and type(data["result_digest"]) is not str): raise TypeError("OutcomeReceipt has an invalid field type")
        return cls(data["task_id"], AttemptId(data["attempt_id"]), data["finished_at_utc"], data["status"], data["output_dir"], data["result_digest"])


@dataclass(frozen=True, slots=True)
class AllocationObservation:
    """Backend-neutral observation of an allocation."""

    schema: ClassVar[str] = "tpen-executor/allocation-observation-v1"
    allocation_id: AllocationId
    observed_at_utc: str
    state: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _identity(self.allocation_id, AllocationId, "allocation_id"); object.__setattr__(self, "observed_at_utc", text(self.observed_at_utc, "observed_at_utc")); object.__setattr__(self, "state", text(self.state, "state"))
        from ._codec import _freeze
        object.__setattr__(self, "metadata", _freeze(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "allocation_id": self.allocation_id.value, "observed_at_utc": self.observed_at_utc, "state": self.state, "metadata": thaw(self.metadata)}

    @classmethod
    def from_mapping(cls, data: Any) -> "AllocationObservation":
        fields(data, {"schema", "allocation_id", "observed_at_utc", "state", "metadata"}, cls.schema, "AllocationObservation")
        if any(type(data[key]) is not str for key in ("allocation_id", "observed_at_utc", "state")): raise TypeError("AllocationObservation scalar fields have invalid types")
        if not isinstance(data["metadata"], dict): raise TypeError("metadata must be a mapping")
        return cls(AllocationId(data["allocation_id"]), data["observed_at_utc"], data["state"], data["metadata"])


@dataclass(frozen=True, slots=True)
class ManifestAttempt:
    """One group member's attempt and receipt references."""

    schema: ClassVar[str] = "tpen-executor/manifest-attempt-v1"
    task_id: str
    attempt_id: AttemptId
    start_receipt: StartReceipt
    outcome_receipt: OutcomeReceipt | None
    output_dir: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", text(self.task_id, "task_id")); _identity(self.attempt_id, AttemptId, "attempt_id"); _identity(self.start_receipt.attempt_id, AttemptId, "start_receipt.attempt_id")
        if self.outcome_receipt is not None and not isinstance(self.outcome_receipt, OutcomeReceipt): raise TypeError("outcome_receipt must be OutcomeReceipt or None")
        object.__setattr__(self, "output_dir", text(self.output_dir, "output_dir"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "task_id": self.task_id, "attempt_id": self.attempt_id.value, "start_receipt": self.start_receipt.to_dict(), "outcome_receipt": None if self.outcome_receipt is None else self.outcome_receipt.to_dict(), "output_dir": self.output_dir}

    @classmethod
    def from_mapping(cls, data: Any) -> "ManifestAttempt":
        fields(data, {"schema", "task_id", "attempt_id", "start_receipt", "outcome_receipt", "output_dir"}, cls.schema, "ManifestAttempt")
        if any(type(data[key]) is not str for key in ("task_id", "attempt_id", "output_dir")): raise TypeError("ManifestAttempt has an invalid scalar field")
        outcome = None if data["outcome_receipt"] is None else OutcomeReceipt.from_mapping(data["outcome_receipt"])
        return cls(data["task_id"], AttemptId(data["attempt_id"]), StartReceipt.from_mapping(data["start_receipt"]), outcome, data["output_dir"])


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Versioned group manifest that points at operational receipts."""

    schema: ClassVar[str] = "tpen-executor/run-manifest-v1"
    group_id: str
    group_digest: str
    attempts: tuple[ManifestAttempt, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", text(self.group_id, "group_id")); object.__setattr__(self, "group_digest", sha256_text(self.group_digest, "group_digest"))
        attempts = sequence(self.attempts, "attempts")
        if any(not isinstance(attempt, ManifestAttempt) for attempt in attempts): raise TypeError("attempts must contain ManifestAttempt records")
        object.__setattr__(self, "attempts", attempts)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "group_id": self.group_id, "group_digest": self.group_digest, "attempts": [attempt.to_dict() for attempt in self.attempts]}

    def encode(self) -> bytes: return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "RunManifest":
        fields(data, {"schema", "group_id", "group_digest", "attempts"}, cls.schema, "RunManifest")
        if type(data["group_id"]) is not str or type(data["group_digest"]) is not str: raise TypeError("RunManifest scalar fields have invalid types")
        return cls(data["group_id"], data["group_digest"], tuple(ManifestAttempt.from_mapping(item) for item in sequence(data["attempts"], "attempts")))

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "RunManifest":
        return cls.from_mapping(decode_json(data))


__all__ = ["AllocationObservation", "ManifestAttempt", "OutcomeReceipt", "RunManifest", "StartReceipt"]


def encode_manifest(manifest: RunManifest) -> bytes:
    """Encode one validated run manifest."""

    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be RunManifest")
    return manifest.encode()


def decode_manifest(data: bytes | str | dict[str, Any]) -> RunManifest:
    """Decode one run manifest."""

    return RunManifest.decode(data)
