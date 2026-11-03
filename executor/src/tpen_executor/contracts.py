"""Immutable executor admission contracts and identity records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

from ._codec import _freeze, canonical_bytes, digest, fields, sequence, sha256_text, text, thaw


@dataclass(frozen=True, slots=True)
class _Identity:
    """Base for deliberately non-interchangeable opaque identities."""

    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", text(self.value, "value"))

    def __str__(self) -> str:
        return self.value


class AttemptId(_Identity):
    """Identity of one fresh execution attempt."""


class AllocationId(_Identity):
    """Identity of one backend allocation."""


class BackendJobId(_Identity):
    """Identity assigned by a backend to one submitted job."""


@dataclass(frozen=True, slots=True)
class ProgramSpecRef:
    """Opaque unresolved program configuration and its content digest."""

    schema: ClassVar[str] = "tpen-executor/program-spec-ref-v1"
    config_ref: Any
    content_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_ref", _freeze(self.config_ref, "config_ref"))
        if digest(self.config_ref) != sha256_text(self.content_digest, "content_digest"):
            raise ValueError("content_digest does not match config_ref")

    @property
    def config(self) -> Any:
        """Compatibility name for the unresolved configuration payload."""

        return self.config_ref

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "config_ref": thaw(self.config_ref), "content_digest": self.content_digest}

    @classmethod
    def from_mapping(cls, data: Any) -> "ProgramSpecRef":
        fields(data, {"schema", "config_ref", "content_digest"}, cls.schema, "ProgramSpecRef")
        return cls(data["config_ref"], data["content_digest"])


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """One logical task admitted to an executor group."""

    schema: ClassVar[str] = "tpen-executor/task-spec-v1"
    task_id: str
    lineage_id: str
    program: ProgramSpecRef
    science_digest: str
    target_horizon: int
    target_horizon_units: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", text(self.task_id, "task_id"))
        object.__setattr__(self, "lineage_id", text(self.lineage_id, "lineage_id"))
        if not isinstance(self.program, ProgramSpecRef): raise TypeError("program must be ProgramSpecRef")
        object.__setattr__(self, "science_digest", sha256_text(self.science_digest, "science_digest"))
        if type(self.target_horizon) is not int or self.target_horizon < 0: raise TypeError("target_horizon must be a nonnegative integer")
        object.__setattr__(self, "target_horizon_units", text(self.target_horizon_units, "target_horizon_units"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "task_id": self.task_id, "lineage_id": self.lineage_id, "program": self.program.to_dict(), "science_digest": self.science_digest, "target_horizon": self.target_horizon, "target_horizon_units": self.target_horizon_units}

    @classmethod
    def from_mapping(cls, data: Any) -> "TaskSpec":
        fields(data, {"schema", "task_id", "lineage_id", "program", "science_digest", "target_horizon", "target_horizon_units"}, cls.schema, "TaskSpec")
        if type(data["task_id"]) is not str or type(data["lineage_id"]) is not str or type(data["science_digest"]) is not str or type(data["target_horizon"]) is not int or type(data["target_horizon_units"]) is not str: raise TypeError("TaskSpec has an invalid field type")
        return cls(data["task_id"], data["lineage_id"], ProgramSpecRef.from_mapping(data["program"]), data["science_digest"], data["target_horizon"], data["target_horizon_units"])


@dataclass(frozen=True, slots=True)
class ReadyGroup:
    """Ordered, heterogeneous group of tasks ready for one allocation plan."""

    schema: ClassVar[str] = "tpen-executor/ready-group-v1"
    group_id: str
    members: tuple[TaskSpec, ...]
    execution_plan_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", text(self.group_id, "group_id"))
        members = sequence(self.members, "members")
        if not members or any(not isinstance(member, TaskSpec) for member in members): raise TypeError("members must be a non-empty sequence of TaskSpec")
        if len({member.task_id for member in members}) != len(members): raise ValueError("duplicate task_id in group")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "execution_plan_digest", sha256_text(self.execution_plan_digest, "execution_plan_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "group_id": self.group_id, "members": [member.to_dict() for member in self.members], "execution_plan_digest": self.execution_plan_digest}

    def encode(self) -> bytes: return canonical_bytes(self.to_dict())

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "ReadyGroup":
        from ._codec import decode_json
        return cls.from_mapping(decode_json(data))

    @classmethod
    def from_mapping(cls, data: Any) -> "ReadyGroup":
        fields(data, {"schema", "group_id", "members", "execution_plan_digest"}, cls.schema, "ReadyGroup")
        if type(data["group_id"]) is not str or type(data["execution_plan_digest"]) is not str: raise TypeError("ReadyGroup scalar fields have invalid types")
        members = sequence(data["members"], "members")
        return cls(data["group_id"], tuple(TaskSpec.from_mapping(member) for member in members), data["execution_plan_digest"])

__all__ = ["AllocationId", "AttemptId", "BackendJobId", "ProgramSpecRef", "ReadyGroup", "TaskSpec"]
