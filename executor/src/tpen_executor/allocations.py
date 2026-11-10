"""Explicit backend allocation configuration and observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from ._codec import _freeze, canonical_bytes, decode_json, fields, thaw, text
from .contracts import AllocationId, BackendJobId
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
            path_keys = {"pbs_nodefile_path", "PBS_NODEFILE_PATH"}
            scalar_pbs = scheduler.get("pbs_nodefile", scheduler.get("PBS_NODEFILE"))
            has_path = any(scheduler.get(key) is not None for key in path_keys) or (
                isinstance(scalar_pbs, str) and "\n" not in scalar_pbs
            )
            if has_path:
                # Scheduler paths are an input-time concern. Capture their
                # contents before this allocation is frozen so later binding
                # resolution cannot change an immutable allocation's host map.
                from .placement import PlacementRefusal, scheduler_host_map

                try:
                    hosts = scheduler_host_map(scheduler, environ={})
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


__all__ = ["ConfiguredAllocation"]
