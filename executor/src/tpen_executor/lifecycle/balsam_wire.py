"""Closed, versioned records for the executor-side Balsam boundary.

The control process owns the Balsam SDK.  The executor exchanges only these
small JSON records, so a Balsam object, callback, or arbitrary command cannot
cross the process boundary accidentally.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from .._codec import canonical_bytes, decode_json, fields, sha256_text, text
from .._records import frozen_record
from ..contracts import BackendJobId, BatchJobId, GroupId


WIRE_VERSION = "tpen-executor/balsam-wire-v2"
Operation = Literal["register", "observe", "submit", "rearm", "readback"]
Status = Literal["ack", "refused", "ambiguous"]
_OPERATIONS = frozenset(("register", "observe", "submit", "rearm", "readback"))
_STATUSES = frozenset(("ack", "refused", "ambiguous"))


class BalsamWireError(ValueError):
    """Base class for malformed or semantically invalid wire data."""


def _operation(value: Any) -> str:
    if type(value) is not str or value not in _OPERATIONS:
        raise BalsamWireError("unknown Balsam wire operation")
    return value


def _status(value: Any) -> str:
    if type(value) is not str or value not in _STATUSES:
        raise BalsamWireError("unknown Balsam wire status")
    return value


@frozen_record
class BalsamWireRequest:
    """One closed request for a single Balsam operation.

    Optional fields are operation-specific and are checked by the constructor;
    they are not an escape hatch for arbitrary SDK kwargs.
    """

    schema: ClassVar[str] = WIRE_VERSION + "/request"
    operation: str
    group_id: GroupId
    backend_job_id: BackendJobId | None = None
    group_digest: str | None = None
    deployment_digest: str | None = None
    batch_digest: str | None = None
    recovery_digest: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    num_nodes: int | None = None
    wall_time_min: int | None = None
    job_mode: str | None = None
    site_id: int | None = None
    project: str | None = None
    queue: str | None = None

    def __post_init__(self) -> None:
        op = _operation(self.operation)
        if not isinstance(self.group_id, GroupId):
            raise TypeError("group_id must be GroupId")
        if self.backend_job_id is not None and not isinstance(self.backend_job_id, BackendJobId):
            raise TypeError("backend_job_id must be BackendJobId")
        for name in ("group_digest", "deployment_digest", "batch_digest", "recovery_digest"):
            value = getattr(self, name)
            if value is not None:
                sha256_text(value, name)
        if self.stdout_path is not None or self.stderr_path is not None:
            if op != "submit" or self.stdout_path is None or self.stderr_path is None:
                raise BalsamWireError("output paths are submit-only and must be paired")
            for name in ("stdout_path", "stderr_path"):
                value = getattr(self, name)
                if type(value) is not str or not Path(value).is_absolute():
                    raise BalsamWireError(f"{name} must be absolute")
        allocation_fields = (self.num_nodes, self.wall_time_min, self.job_mode,
                             self.site_id, self.project, self.queue)
        if op != "submit" and any(value is not None for value in allocation_fields):
            raise BalsamWireError("allocation fields are submit-only")
        if any(value is not None for value in allocation_fields) and any(value is None for value in allocation_fields):
            raise BalsamWireError("submit allocation fields must be complete")
        if self.num_nodes is not None:
            if self.stdout_path is None or self.stderr_path is None:
                raise BalsamWireError("submit allocation fields require output paths")
            for name in ("num_nodes", "wall_time_min", "site_id"):
                value = getattr(self, name)
                if type(value) is not int or value <= 0:
                    raise BalsamWireError(f"{name} must be a positive integer")
            for name in ("job_mode", "project", "queue"):
                if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                    raise BalsamWireError(f"{name} must be non-empty text")
        required = {
            "register": (self.backend_job_id is None and self.group_digest is not None and self.deployment_digest is not None
                         and self.batch_digest is None and self.recovery_digest is None),
            "observe": (self.backend_job_id is not None and self.group_digest is None and self.deployment_digest is None
                        and self.batch_digest is None and self.recovery_digest is None),
            "submit": (self.backend_job_id is not None and self.group_digest is None and self.deployment_digest is None
                        and self.batch_digest is not None and self.recovery_digest is None),
            "rearm": (self.backend_job_id is not None and self.group_digest is None and self.deployment_digest is None
                       and self.batch_digest is None and self.recovery_digest is not None),
            "readback": (self.backend_job_id is not None and self.group_digest is None and self.deployment_digest is None
                          and self.batch_digest is None and self.recovery_digest is None),
        }[op]
        if not required:
            raise BalsamWireError(f"fields do not match {op} request")

    @classmethod
    def register(cls, group_id: GroupId, group_digest: str, deployment_digest: str) -> "BalsamWireRequest":
        """Build the one-shot registration request."""
        return cls("register", group_id, group_digest=group_digest, deployment_digest=deployment_digest)

    @classmethod
    def observe(cls, group_id: GroupId, backend_job_id: BackendJobId) -> "BalsamWireRequest":
        """Build a non-mutating observation request."""
        return cls("observe", group_id, backend_job_id=backend_job_id)

    @classmethod
    def submit(cls, group_id: GroupId, backend_job_id: BackendJobId, batch_digest: str,
               *, stdout_path: str | None = None, stderr_path: str | None = None,
               num_nodes: int | None = None, wall_time_min: int | None = None,
               job_mode: str | None = None, site_id: int | None = None,
               project: str | None = None, queue: str | None = None) -> "BalsamWireRequest":
        """Build the explicitly authorized batch-submission request."""
        return cls("submit", group_id, backend_job_id=backend_job_id, batch_digest=batch_digest,
                   stdout_path=stdout_path, stderr_path=stderr_path,
                   num_nodes=num_nodes, wall_time_min=wall_time_min, job_mode=job_mode,
                   site_id=site_id, project=project, queue=queue)

    @classmethod
    def rearm(cls, group_id: GroupId, backend_job_id: BackendJobId, recovery_digest: str) -> "BalsamWireRequest":
        """Build the explicitly authorized same-job rearm request."""
        return cls("rearm", group_id, backend_job_id=backend_job_id, recovery_digest=recovery_digest)

    @classmethod
    def readback(cls, group_id: GroupId, backend_job_id: BackendJobId) -> "BalsamWireRequest":
        """Build a non-mutating deployment/readback request."""
        return cls("readback", group_id, backend_job_id=backend_job_id)

    def to_dict(self) -> dict[str, Any]:
        """Return the exact closed request mapping."""
        return {"schema": self.schema, "operation": self.operation, "group_id": self.group_id.to_dict(),
                "backend_job_id": None if self.backend_job_id is None else self.backend_job_id.to_dict(),
                "group_digest": self.group_digest, "deployment_digest": self.deployment_digest,
                "batch_digest": self.batch_digest, "recovery_digest": self.recovery_digest,
                "stdout_path": self.stdout_path, "stderr_path": self.stderr_path,
                "num_nodes": self.num_nodes, "wall_time_min": self.wall_time_min,
                "job_mode": self.job_mode, "site_id": self.site_id,
                "project": self.project, "queue": self.queue}

    def encode(self) -> bytes:
        """Return canonical request bytes."""
        return canonical_bytes(self.to_dict())

    @classmethod
    def decode(cls, payload: bytes | str | Mapping[str, Any]) -> "BalsamWireRequest":
        """Decode and validate a closed request."""
        data = decode_json(payload)
        fields(data, {"schema", "operation", "group_id", "backend_job_id", "group_digest", "deployment_digest",
                      "batch_digest", "recovery_digest", "stdout_path", "stderr_path", "num_nodes", "wall_time_min",
                      "job_mode", "site_id", "project", "queue"}, cls.schema, cls.__name__)
        backend = data["backend_job_id"]
        if backend is not None:
            backend = BackendJobId.from_mapping(backend)
        return cls(data["operation"], GroupId.from_mapping(data["group_id"]), backend,
                   data["group_digest"], data["deployment_digest"], data["batch_digest"], data["recovery_digest"],
                   data.get("stdout_path"), data.get("stderr_path"), data.get("num_nodes"),
                   data.get("wall_time_min"), data.get("job_mode"), data.get("site_id"),
                   data.get("project"), data.get("queue"))


@frozen_record
class BalsamWireResult:
    """Closed response distinguishing acknowledgement, refusal, and uncertainty."""

    schema: ClassVar[str] = WIRE_VERSION + "/result"
    operation: str
    status: str
    group_id: GroupId
    backend_job_id: BackendJobId | None = None
    batch_job_id: BatchJobId | None = None
    state: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _operation(self.operation)
        status = _status(self.status)
        if not isinstance(self.group_id, GroupId):
            raise TypeError("group_id must be GroupId")
        if self.backend_job_id is not None and not isinstance(self.backend_job_id, BackendJobId):
            raise TypeError("backend_job_id must be BackendJobId")
        if self.batch_job_id is not None and not isinstance(self.batch_job_id, BatchJobId):
            raise TypeError("batch_job_id must be BatchJobId")
        if self.state is not None:
            text(self.state, "state")
        if self.reason is not None and type(self.reason) is not str:
            raise TypeError("reason must be a string or None")
        if status == "ack" and self.reason is not None:
            raise BalsamWireError("acknowledgement cannot carry a refusal reason")
        if status != "ack" and (self.reason is None or not self.reason.strip()):
            raise BalsamWireError("refusal or ambiguity requires a reason")
        if status == "ack":
            expected = {
                "register": (True, False),
                "observe": (True, False),
                "submit": (True, True),
                "rearm": (True, False),
                "readback": (True, False),
            }[self.operation]
            if (self.backend_job_id is not None, self.batch_job_id is not None) != expected:
                raise BalsamWireError("acknowledgement identities do not match operation")
        elif self.backend_job_id is not None or self.batch_job_id is not None:
            raise BalsamWireError("non-acknowledgement cannot carry backend handles")
        if status == "ambiguous" and self.operation not in {"register", "submit", "rearm"}:
            raise BalsamWireError("only mutating operations may be ambiguous")

    @classmethod
    def ack(cls, request: BalsamWireRequest, *, backend_job_id: BackendJobId | None = None,
            batch_job_id: BatchJobId | None = None, state: str | None = None) -> "BalsamWireResult":
        """Build a typed successful response bound to one request."""
        return cls(request.operation, "ack", request.group_id, backend_job_id, batch_job_id, state)

    @classmethod
    def refused(cls, request: BalsamWireRequest, reason: str) -> "BalsamWireResult":
        """Build a refusal that proves the effect was not accepted."""
        return cls(request.operation, "refused", request.group_id, reason=reason)

    @classmethod
    def ambiguous(cls, request: BalsamWireRequest, reason: str) -> "BalsamWireResult":
        """Build an unknown-effect result; callers must park and reconcile."""
        return cls(request.operation, "ambiguous", request.group_id, reason=reason)

    def to_dict(self) -> dict[str, Any]:
        """Return the exact closed response mapping."""
        return {"schema": self.schema, "operation": self.operation, "status": self.status,
                "group_id": self.group_id.to_dict(),
                "backend_job_id": None if self.backend_job_id is None else self.backend_job_id.to_dict(),
                "batch_job_id": None if self.batch_job_id is None else self.batch_job_id.to_dict(),
                "state": self.state, "reason": self.reason}

    def encode(self) -> bytes:
        """Return canonical response bytes."""
        return canonical_bytes(self.to_dict())

    @classmethod
    def decode(cls, payload: bytes | str | Mapping[str, Any]) -> "BalsamWireResult":
        """Decode and validate a closed response."""
        data = decode_json(payload)
        fields(data, {"schema", "operation", "status", "group_id", "backend_job_id", "batch_job_id", "state", "reason"},
               cls.schema, cls.__name__)
        backend = data["backend_job_id"]
        batch = data["batch_job_id"]
        return cls(data["operation"], data["status"], GroupId.from_mapping(data["group_id"]),
                   None if backend is None else BackendJobId.from_mapping(backend),
                   None if batch is None else BatchJobId.from_mapping(batch), data["state"], data["reason"])


def encode_request(request: BalsamWireRequest) -> bytes:
    """Encode one typed request, rejecting foreign objects."""
    if not isinstance(request, BalsamWireRequest):
        raise TypeError("request must be BalsamWireRequest")
    return request.encode()


def decode_request(payload: bytes | str | Mapping[str, Any]) -> BalsamWireRequest:
    """Decode one request using the opposite codec entry point."""
    return BalsamWireRequest.decode(payload)


def encode_result(result: BalsamWireResult) -> bytes:
    """Encode one typed response, rejecting foreign objects."""
    if not isinstance(result, BalsamWireResult):
        raise TypeError("result must be BalsamWireResult")
    return result.encode()


def decode_result(payload: bytes | str | Mapping[str, Any]) -> BalsamWireResult:
    """Decode one response using the opposite codec entry point."""
    return BalsamWireResult.decode(payload)


__all__ = ["BalsamWireError", "BalsamWireRequest", "BalsamWireResult", "WIRE_VERSION",
           "decode_request", "decode_result", "encode_request", "encode_result"]
