"""Explicit backend allocation configuration and observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from ._codec import _freeze, canonical_bytes, decode_json, digest, fields, sha256_text, thaw, text
from .contracts import AllocationAttemptId, AllocationId, AttemptId, BackendJobId, GroupId, InvocationId
from ._records import frozen_record
from .results import AllocationObservation


def _hashable(value: Any) -> Any:
    """Convert a deeply frozen JSON value to Python's equality/hash shape."""
    if isinstance(value, Mapping):
        return tuple(sorted((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, tuple):
        return tuple(_hashable(item) for item in value)
    return value


@frozen_record
class ConfiguredAllocation:
    """An explicitly configured allocation; no scheduler discovery occurs.

    Parameters
    ----------
    allocation_id, backend_job_id : AllocationId, BackendJobId
        Operational identities.
    facts : mapping
        JSON-shaped backend facts copied and deeply frozen at construction.
    """

    schema: ClassVar[str] = "tpen-executor/configured-allocation-v1"
    allocation_id: AllocationId
    backend_job_id: BackendJobId
    facts: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.allocation_id, AllocationId) or not isinstance(self.backend_job_id, BackendJobId):
            raise TypeError("allocation identities have invalid types")
        if not isinstance(self.facts, Mapping):
            raise TypeError("facts must be a mapping")
        facts = dict(self.facts)
        scheduler = facts.get("scheduler", facts.get("scheduler_host_map"))
        if isinstance(scheduler, Mapping):
            scheduler = dict(scheduler)
            slots = facts.get("slots")
            routed_slots = isinstance(slots, Mapping) and slots.get("schema") == "tpen-executor/allocation-slots-v2"
            path_keys = {"pbs_nodefile_path", "PBS_NODEFILE_PATH"}
            scalar_pbs = scheduler.get("pbs_nodefile", scheduler.get("PBS_NODEFILE"))
            has_path = any(scheduler.get(key) is not None for key in path_keys) or (
                isinstance(scalar_pbs, str) and "\n" not in scalar_pbs
            )
            if has_path:
                # Scheduler paths are an input-time concern. Capture their
                # contents before this allocation is frozen so later binding
                # resolution cannot change an immutable allocation's host map.
                from .placement import PlacementRefusal, captured_scheduler_host_map, scheduler_host_map

                try:
                    hosts = (
                        captured_scheduler_host_map(scheduler)
                        if routed_slots
                        else scheduler_host_map(scheduler, environ={})
                    )
                except PlacementRefusal as error:
                    # Preserve a typed capture failure so dispatch can publish
                    # launch_failed/rc=3 placement evidence instead of turning
                    # an unreadable PBS path into admission rc=2.
                    scheduler["pbs_nodefile_error"] = error.to_dict()
                    hosts = ()
                scheduler["pbs_nodefile_text"] = "\n".join(hosts)
                for key in path_keys | {"pbs_nodefile", "PBS_NODEFILE"}:
                    scheduler.pop(key, None)
                if "scheduler" in facts:
                    facts["scheduler"] = scheduler
                else:
                    facts["scheduler_host_map"] = scheduler
        object.__setattr__(self, "facts", _freeze(facts, "facts"))

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON representation."""
        return {"schema": self.schema, "allocation_id": self.allocation_id.value, "backend_job_id": self.backend_job_id.value, "facts": thaw(self.facts)}

    def encode(self) -> bytes:
        """Encode this allocation canonically."""
        return canonical_bytes(self.to_dict())

    def __hash__(self) -> int:
        """Hash the immutable Python equality representation.

        Python numeric equality is intentional here: nested ``1``, ``True``,
        and ``1.0`` values, as well as ``0.0`` and ``-0.0``, hash alike even
        though canonical JSON encoding preserves their distinct wire forms.
        Equal records may therefore encode differently.
        """
        return hash((self.allocation_id, self.backend_job_id, _hashable(self.facts)))

    @classmethod
    def from_mapping(cls, data: Any) -> "ConfiguredAllocation":
        """Decode one strict allocation mapping."""
        fields(data, {"schema", "allocation_id", "backend_job_id", "facts"}, cls.schema, "ConfiguredAllocation")
        if any(type(data[key]) is not str for key in ("allocation_id", "backend_job_id")):
            raise TypeError("allocation identity fields must be strings")
        return cls(AllocationId(data["allocation_id"]), BackendJobId(data["backend_job_id"]), data["facts"])

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "ConfiguredAllocation":
        """Decode canonical allocation bytes or a mapping."""
        return cls.from_mapping(decode_json(data))

    def observe(self, state: str, at_utc: str) -> AllocationObservation:
        """Create a backend-neutral observation for this allocation."""
        return AllocationObservation(self.allocation_id, at_utc, state, self.facts)


@frozen_record
class AllocationAttemptV2:
    """Typed binding for one controller generation on one allocation.

    This record deliberately carries no member lineage or member attempt.  A
    controller may use it to scope a group reconciliation, but it cannot be
    substituted for ``AttemptId`` or for the backend allocation identity.
    """

    schema: ClassVar[str] = "tpen-executor/allocation-attempt-v2"
    group_id: GroupId
    allocation_attempt_id: AllocationAttemptId
    allocation_id: AllocationId
    backend_job_id: BackendJobId
    controller_generation: str
    observed_at_utc: str

    def __post_init__(self) -> None:
        for value, name, cls in (
            (self.group_id, "group_id", GroupId),
            (self.allocation_attempt_id, "allocation_attempt_id", AllocationAttemptId),
            (self.allocation_id, "allocation_id", AllocationId),
            (self.backend_job_id, "backend_job_id", BackendJobId),
        ):
            if not isinstance(value, cls):
                raise TypeError(f"{name} must be {cls.__name__}")
        object.__setattr__(self, "controller_generation", text(self.controller_generation, "controller_generation"))
        object.__setattr__(self, "observed_at_utc", text(self.observed_at_utc, "observed_at_utc"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "group_id": self.group_id.to_dict(),
            "allocation_attempt_id": self.allocation_attempt_id.to_dict(),
            "allocation_id": self.allocation_id.to_dict(),
            "backend_job_id": self.backend_job_id.to_dict(),
            "controller_generation": self.controller_generation,
            "observed_at_utc": self.observed_at_utc,
        }

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "AllocationAttemptV2":
        fields(
            data,
            {
                "schema", "group_id", "allocation_attempt_id", "allocation_id",
                "backend_job_id", "controller_generation", "observed_at_utc",
            },
            cls.schema,
            cls.__name__,
        )
        if type(data["controller_generation"]) is not str or type(data["observed_at_utc"]) is not str:
            raise TypeError("AllocationAttemptV2 observation fields must be strings")
        return cls(
            GroupId.from_mapping(data["group_id"]),
            AllocationAttemptId.from_mapping(data["allocation_attempt_id"]),
            AllocationId.from_mapping(data["allocation_id"]),
            BackendJobId.from_mapping(data["backend_job_id"]),
            data["controller_generation"],
            data["observed_at_utc"],
        )

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "AllocationAttemptV2":
        return cls.from_mapping(decode_json(data))


@frozen_record
class MemberInvocationV2:
    """Controller-issued invocation, echoed by the leaf with all identities.

    A binding is an identity record, not proof that a process is quiescent or
    that a successor has been admitted. Only the store may publish it.
    """

    schema: ClassVar[str] = "tpen-executor/member-invocation-v2"
    allocation_attempt: AllocationAttemptV2
    group_digest: str
    task_id: str
    lineage_id: str
    attempt_id: AttemptId
    invocation_id: InvocationId

    def __post_init__(self) -> None:
        if not isinstance(self.allocation_attempt, AllocationAttemptV2):
            raise TypeError("allocation_attempt must be AllocationAttemptV2")
        if not isinstance(self.attempt_id, AttemptId) or not isinstance(self.invocation_id, InvocationId):
            raise TypeError("member attempt and invocation identities have invalid types")
        object.__setattr__(self, "group_digest", sha256_text(self.group_digest, "group_digest"))
        for name in ("task_id", "lineage_id"):
            object.__setattr__(self, name, text(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "allocation_attempt": self.allocation_attempt.to_dict(),
            "group_digest": self.group_digest,
            "task_id": self.task_id,
            "lineage_id": self.lineage_id,
            "attempt_id": self.attempt_id.to_dict(),
            "invocation_id": self.invocation_id.to_dict(),
        }

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "MemberInvocationV2":
        fields(data, {"schema", "allocation_attempt", "group_digest", "task_id", "lineage_id", "attempt_id", "invocation_id"}, cls.schema, cls.__name__)
        return cls(
            AllocationAttemptV2.from_mapping(data["allocation_attempt"]),
            data["group_digest"], data["task_id"], data["lineage_id"],
            AttemptId.from_mapping(data["attempt_id"]), InvocationId.from_mapping(data["invocation_id"]),
        )

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "MemberInvocationV2":
        return cls.from_mapping(decode_json(data))


@frozen_record
class ControllerGenerationV1:
    """Store-owned, strictly ordered authority for one allocation group.

    ``claimant_nonce`` must be freshly generated and unique for every
    controller incarnation; it is not a retry counter or a reusable name.
    """

    schema: ClassVar[str] = "tpen-executor/controller-generation-v1"
    group_id: GroupId
    generation: int
    allocation_id: AllocationId
    backend_job_id: BackendJobId
    registration_receipt_digest: str
    claimant_nonce: str
    predecessor_record_digest: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, GroupId):
            raise TypeError("group_id must be GroupId")
        if type(self.generation) is not int or self.generation < 1:
            raise TypeError("generation must be a positive integer")
        for value, name, cls in ((self.allocation_id, "allocation_id", AllocationId), (self.backend_job_id, "backend_job_id", BackendJobId)):
            if not isinstance(value, cls):
                raise TypeError(f"{name} must be {cls.__name__}")
        object.__setattr__(self, "registration_receipt_digest", sha256_text(self.registration_receipt_digest, "registration_receipt_digest"))
        object.__setattr__(self, "claimant_nonce", text(self.claimant_nonce, "claimant_nonce"))
        if self.predecessor_record_digest is not None:
            object.__setattr__(self, "predecessor_record_digest", sha256_text(self.predecessor_record_digest, "predecessor_record_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "group_id": self.group_id.to_dict(), "generation": self.generation,
                "allocation_id": self.allocation_id.to_dict(), "backend_job_id": self.backend_job_id.to_dict(),
                "registration_receipt_digest": self.registration_receipt_digest, "claimant_nonce": self.claimant_nonce,
                "predecessor_record_digest": self.predecessor_record_digest}

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def record_digest(self) -> str:
        """Digest used to link the next generation to this record."""
        return digest(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "ControllerGenerationV1":
        fields(data, {"schema", "group_id", "generation", "allocation_id", "backend_job_id",
                      "registration_receipt_digest", "claimant_nonce", "predecessor_record_digest"}, cls.schema, cls.__name__)
        if type(data["generation"]) is not int:
            raise TypeError("generation must be an integer")
        allocation_id = AllocationId.from_mapping(data["allocation_id"])
        backend_job_id = BackendJobId.from_mapping(data["backend_job_id"])
        return cls(GroupId.from_mapping(data["group_id"]), data["generation"], allocation_id, backend_job_id,
                   data["registration_receipt_digest"], data["claimant_nonce"], data["predecessor_record_digest"])

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "ControllerGenerationV1":
        return cls.from_mapping(decode_json(data))


@frozen_record
class LaunchIntentV1:
    """Write-once proof that one member launch was claimed by one controller."""

    schema: ClassVar[str] = "tpen-executor/launch-intent-v1"
    group_id: GroupId
    task_id: str
    attempt_id: AttemptId
    invocation_id: InvocationId
    controller_generation: int
    claimant_nonce: str
    allocation_id: AllocationId
    backend_job_id: BackendJobId

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, GroupId) or not isinstance(self.attempt_id, AttemptId) or not isinstance(self.invocation_id, InvocationId):
            raise TypeError("launch intent identities have invalid types")
        object.__setattr__(self, "task_id", text(self.task_id, "task_id"))
        if type(self.controller_generation) is not int or self.controller_generation < 1:
            raise TypeError("controller_generation must be a positive integer")
        object.__setattr__(self, "claimant_nonce", text(self.claimant_nonce, "claimant_nonce"))
        if not isinstance(self.allocation_id, AllocationId) or not isinstance(self.backend_job_id, BackendJobId):
            raise TypeError("launch intent allocation identities have invalid types")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "group_id": self.group_id.to_dict(), "task_id": self.task_id,
                "attempt_id": self.attempt_id.to_dict(), "invocation_id": self.invocation_id.to_dict(),
                "controller_generation": self.controller_generation, "claimant_nonce": self.claimant_nonce,
                "allocation_id": self.allocation_id.to_dict(), "backend_job_id": self.backend_job_id.to_dict()}

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "LaunchIntentV1":
        fields(data, {"schema", "group_id", "task_id", "attempt_id", "invocation_id", "controller_generation",
                      "claimant_nonce", "allocation_id", "backend_job_id"}, cls.schema, cls.__name__)
        allocation_id = AllocationId.from_mapping(data["allocation_id"])
        backend_job_id = BackendJobId.from_mapping(data["backend_job_id"])
        return cls(GroupId.from_mapping(data["group_id"]), data["task_id"], AttemptId.from_mapping(data["attempt_id"]),
                   InvocationId.from_mapping(data["invocation_id"]), data["controller_generation"], data["claimant_nonce"],
                   allocation_id, backend_job_id)

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "LaunchIntentV1":
        return cls.from_mapping(decode_json(data))


@frozen_record
class ControllerOutcomeReceiptV1:
    """Write-once operational acknowledgement for one fenced launch."""

    schema: ClassVar[str] = "tpen-executor/controller-outcome-v1"
    task_id: str
    attempt_id: AttemptId
    invocation_id: InvocationId
    controller_generation: int
    claimant_nonce: str
    status: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.attempt_id, AttemptId) or not isinstance(self.invocation_id, InvocationId):
            raise TypeError("controller outcome identities have invalid types")
        object.__setattr__(self, "task_id", text(self.task_id, "task_id"))
        if type(self.controller_generation) is not int or self.controller_generation < 1:
            raise TypeError("controller_generation must be positive")
        object.__setattr__(self, "claimant_nonce", text(self.claimant_nonce, "claimant_nonce"))
        object.__setattr__(self, "status", text(self.status, "status"))
        object.__setattr__(self, "reason", text(self.reason, "reason"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "task_id": self.task_id, "attempt_id": self.attempt_id.to_dict(),
                "invocation_id": self.invocation_id.to_dict(), "controller_generation": self.controller_generation,
                "claimant_nonce": self.claimant_nonce, "status": self.status, "reason": self.reason}

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "ControllerOutcomeReceiptV1":
        fields(data, {"schema", "task_id", "attempt_id", "invocation_id", "controller_generation", "claimant_nonce", "status", "reason"}, cls.schema, cls.__name__)
        return cls(data["task_id"], AttemptId.from_mapping(data["attempt_id"]), InvocationId.from_mapping(data["invocation_id"]), data["controller_generation"], data["claimant_nonce"], data["status"], data["reason"])

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "ControllerOutcomeReceiptV1":
        return cls.from_mapping(decode_json(data))


__all__ = ["AllocationAttemptV2", "ConfiguredAllocation", "ControllerGenerationV1", "ControllerOutcomeReceiptV1", "LaunchIntentV1", "MemberInvocationV2"]
