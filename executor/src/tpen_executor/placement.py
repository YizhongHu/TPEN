"""Explicit allocation-to-worker binding records for attached dispatch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar

from .allocations import ConfiguredAllocation
from ._records import frozen_record


class BindingError(ValueError):
    """Raised when a worker binding is absent or cannot be trusted."""


def _normalize_nodes(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one generic node shape or refuse ambiguous geometry."""

    if "nodes" not in value:
        return dict(value)
    nodes = value["nodes"]
    if isinstance(nodes, (str, bytes, bytearray)) or not isinstance(nodes, Sequence):
        raise BindingError("worker binding nodes must be a sequence")
    if len(nodes) != 1 or type(nodes[0]) is not str or not nodes[0]:
        raise BindingError("worker binding must identify exactly one worker node")
    normalized = dict(value)
    existing = normalized.get("node")
    if existing is not None and existing != nodes[0]:
        raise BindingError("worker binding node and nodes disagree")
    normalized["node"] = nodes[0]
    return normalized


@frozen_record
class WorkerBinding:
    """Immutable environment binding for one Parsl worker.

    Parameters
    ----------
    environment : mapping[str, str]
        Environment entries applied after the prepared worker environment.
    worker_id, node : str or None, optional
        Optional operator-visible binding identities. They are carried as
        metadata by callers and are not inferred from scheduler state.
    """

    environment: Mapping[str, str] | tuple[tuple[str, str], ...]
    worker_id: str | None = None
    node: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.environment, Mapping):
            raw_entries = tuple(self.environment.items())
        elif isinstance(self.environment, tuple):
            raw_entries = self.environment
        else:
            raise BindingError("worker binding environment must be a mapping or pairs")
        if not raw_entries:
            raise BindingError("worker binding environment must not be empty")
        entries: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in raw_entries:
            if not isinstance(item, tuple) or len(item) != 2:
                raise BindingError("worker binding environment entries must be pairs")
            key, value = item
            if type(key) is not str or not key or type(value) is not str:
                raise BindingError("worker binding environment keys and values must be strings")
            if key in seen:
                raise BindingError(f"worker binding contains duplicate environment key {key!r}")
            seen.add(key)
            entries.append((key, value))
        object.__setattr__(self, "environment", tuple(entries))
        for name in ("worker_id", "node"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise BindingError(f"{name} must be a non-empty string or None")

    @property
    def env(self) -> Mapping[str, str]:
        """Return the immutable environment mapping for this worker."""

        return MappingProxyType(dict(self.environment))

    @classmethod
    def from_mapping(cls, value: Any) -> "WorkerBinding":
        """Decode one explicit binding mapping."""

        if isinstance(value, WorkerBinding):
            return value
        if not isinstance(value, Mapping):
            raise BindingError("worker binding must be a mapping")
        value = _normalize_nodes(value)
        environment = value.get("environment", value.get("env"))
        if environment is None:
            environment = {
                key: item
                for key, item in value.items()
                if key not in {"worker_id", "node", "nodes", "visibility", "accelerator_visibility"}
            }
        if not isinstance(environment, Mapping):
            raise BindingError("worker binding environment must be a mapping")
        normalized: dict[str, str] = {}
        for key, item in environment.items():
            if type(key) is not str or not key:
                raise BindingError("worker binding environment keys must be strings")
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                item = ",".join(str(part) for part in item)
            if type(item) is not str or not item:
                raise BindingError("worker binding environment values must be non-empty strings")
            normalized[key] = item
        visibility = value.get("visibility", value.get("accelerator_visibility"))
        if visibility is not None and "CUDA_VISIBLE_DEVICES" not in normalized:
            if isinstance(visibility, Sequence) and not isinstance(visibility, (str, bytes, bytearray)):
                visibility = ",".join(str(part) for part in visibility)
            if type(visibility) is not str or not visibility:
                raise BindingError("accelerator visibility must be a non-empty string or sequence")
            normalized["CUDA_VISIBLE_DEVICES"] = visibility
        if not normalized:
            raise BindingError("worker binding must declare a worker environment")
        return cls(
            tuple(normalized.items()),
            value.get("worker_id"),
            value.get("node"),
        )


_SLOT_SCHEMA = "tpen-executor/allocation-slots-v1"
_THREAD_NAMES = frozenset({"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"})


@frozen_record
class AllocationSlot:
    """Immutable generic entitlement for one attached allocation slot.

    Parameters
    ----------
    slot_id, node : str
        Stable slot and worker-node identities.
    devices : sequence of str
        Device identifiers entitled to this slot.  Empty is valid for a
        CPU-only attached allocation.
    cpus : sequence of int
        Disjoint logical CPU identifiers declared for this slot.
    threads : mapping[str, int]
        Positive thread settings propagated to the child environment.
    environment : mapping[str, str]
        Additional child environment entries.  Reserved derived entries must
        agree with the slot entitlement when supplied.
    """

    schema: ClassVar[str] = _SLOT_SCHEMA
    slot_id: str
    node: str
    devices: tuple[str, ...]
    cpus: tuple[int, ...]
    threads: tuple[tuple[str, int], ...]
    environment: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name, value in (("slot_id", self.slot_id), ("node", self.node)):
            if type(value) is not str or not value.strip():
                raise BindingError(f"slot {name} must be a non-empty string")
        devices = _string_tuple(self.devices, "slot devices")
        if len(set(devices)) != len(devices):
            raise BindingError("slot devices must be unique")
        cpus = _int_tuple(self.cpus, "slot cpus")
        if not cpus or any(cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus):
            raise BindingError("slot cpus must be a non-empty set of unique non-negative integers")
        if isinstance(self.threads, Mapping):
            raw_threads = tuple(self.threads.items())
        else:
            raw_threads = self.threads
        normalized_threads: list[tuple[str, int]] = []
        seen_threads: set[str] = set()
        for item in raw_threads:
            if not isinstance(item, tuple) or len(item) != 2:
                raise BindingError("slot thread settings must be pairs")
            key, value = item
            if type(key) is not str or key not in _THREAD_NAMES or key in seen_threads:
                raise BindingError("slot thread settings contain an invalid or duplicate name")
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise BindingError("slot thread settings must be positive integers")
            seen_threads.add(key)
            normalized_threads.append((key, value))
        if isinstance(self.environment, Mapping):
            raw_environment = tuple(self.environment.items())
        else:
            raw_environment = self.environment
        normalized_environment: list[tuple[str, str]] = []
        seen_environment: set[str] = set()
        for item in raw_environment:
            if not isinstance(item, tuple) or len(item) != 2:
                raise BindingError("slot environment entries must be pairs")
            key, value = item
            if type(key) is not str or not key or key in seen_environment:
                raise BindingError("slot environment contains an invalid or duplicate name")
            if type(value) is not str or not value:
                raise BindingError("slot environment values must be non-empty strings")
            seen_environment.add(key)
            normalized_environment.append((key, value))
        env = dict(normalized_environment)
        if devices:
            visible = ",".join(devices)
            if "CUDA_VISIBLE_DEVICES" in env and env["CUDA_VISIBLE_DEVICES"] != visible:
                raise BindingError("slot CUDA_VISIBLE_DEVICES disagrees with device entitlement")
        for key, value in normalized_threads:
            if key in env and env[key] != str(value):
                raise BindingError(f"slot environment {key!r} disagrees with thread setting")
        derived = {
            "TPEN_SLOT_ID": self.slot_id,
            "TPEN_SLOT_NODE": self.node,
            "TPEN_SLOT_CPUS": ",".join(str(cpu) for cpu in cpus),
        }
        if devices:
            derived["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        derived.update({key: str(value) for key, value in normalized_threads})
        for key, value in derived.items():
            if key in env and env[key] != value:
                raise BindingError(f"slot environment {key!r} disagrees with derived binding")
        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "cpus", cpus)
        object.__setattr__(self, "threads", tuple(normalized_threads))
        object.__setattr__(self, "environment", tuple(normalized_environment))

    @property
    def env(self) -> Mapping[str, str]:
        """Return the complete child overlay for this slot."""

        values = dict(self.environment)
        values.update({"TPEN_SLOT_ID": self.slot_id, "TPEN_SLOT_NODE": self.node, "TPEN_SLOT_CPUS": ",".join(str(cpu) for cpu in self.cpus)})
        if self.devices:
            values["CUDA_VISIBLE_DEVICES"] = ",".join(self.devices)
        values.update({key: str(value) for key, value in self.threads})
        return MappingProxyType(values)

    def binding(self) -> WorkerBinding:
        """Return the worker binding derived from this immutable slot."""

        return WorkerBinding(self.env, self.slot_id, self.node)

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-shaped slot payload."""

        return {
            "slot_id": self.slot_id,
            "node": self.node,
            "devices": list(self.devices),
            "cpus": list(self.cpus),
            "threads": dict(self.threads),
            "environment": dict(self.environment),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "AllocationSlot":
        """Decode one strict slot payload."""

        if not isinstance(value, Mapping):
            raise BindingError("allocation slot must be a mapping")
        expected = {"slot_id", "node", "devices", "cpus", "threads", "environment"}
        if set(value) != expected:
            missing, extra = expected - set(value), set(value) - expected
            raise BindingError(f"allocation slot keys differ (missing={sorted(missing)!r}, extra={sorted(extra)!r})")
        devices = _string_tuple(value["devices"], "slot devices")
        cpus = _int_tuple(value["cpus"], "slot cpus")
        threads = value["threads"]
        environment = value["environment"]
        if not isinstance(threads, Mapping) or not isinstance(environment, Mapping):
            raise BindingError("slot threads and environment must be mappings")
        return cls(value["slot_id"], value["node"], devices, cpus, tuple(threads.items()), tuple(environment.items()))


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Validate one non-string sequence of non-empty strings."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BindingError(f"{name} must be a sequence")
    result = tuple(value)
    if any(type(item) is not str or not item for item in result):
        raise BindingError(f"{name} must contain non-empty strings")
    return result


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    """Validate one non-string sequence of integer entitlements."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BindingError(f"{name} must be a sequence")
    result = tuple(value)
    if any(type(item) is not int or isinstance(item, bool) for item in result):
        raise BindingError(f"{name} must contain integers")
    return result


def slots_from_allocation(allocation: ConfiguredAllocation) -> tuple[AllocationSlot, ...]:
    """Decode and strictly validate the optional versioned slot record."""

    if not isinstance(allocation, ConfiguredAllocation):
        raise BindingError("slot resolution requires ConfiguredAllocation")
    raw = allocation.facts.get("slots")
    if raw is None:
        return ()
    if not isinstance(raw, Mapping) or set(raw) != {"schema", "slots"} or raw.get("schema") != _SLOT_SCHEMA:
        raise BindingError(f"allocation slots must use schema {_SLOT_SCHEMA!r}")
    entries = raw["slots"]
    if isinstance(entries, (str, bytes, bytearray)) or not isinstance(entries, Sequence) or not entries:
        raise BindingError("allocation slots must be a non-empty sequence")
    decoded = tuple(AllocationSlot.from_mapping(entry) for entry in entries)
    slot_ids = [slot.slot_id for slot in decoded]
    if len(set(slot_ids)) != len(slot_ids):
        raise BindingError("allocation slots must have unique slot_id values")
    devices = [device for slot in decoded for device in slot.devices]
    if len(set(devices)) != len(devices):
        raise BindingError("allocation slots must have disjoint device entitlements")
    cpus = [cpu for slot in decoded for cpu in slot.cpus]
    if len(set(cpus)) != len(cpus):
        raise BindingError("allocation slots must have disjoint CPU entitlements")
    nodes = [slot.node for slot in decoded]
    if any(type(node) is not str or not node.strip() for node in nodes):
        raise BindingError("allocation slots must identify unambiguous nodes")
    return decoded


def bindings_from_allocation(allocation: ConfiguredAllocation) -> tuple[WorkerBinding, ...]:
    """Return per-slot bindings, or the legacy one-binding fallback."""

    slots = slots_from_allocation(allocation)
    if slots:
        return tuple(slot.binding() for slot in slots)
    return (binding_from_allocation(allocation),)


SlotRecord = AllocationSlot
allocation_slots = slots_from_allocation


def binding_from_allocation(allocation: ConfiguredAllocation) -> WorkerBinding:
    """Resolve an explicit worker binding from configured allocation facts.

    Accepted fact shapes are ``binding``/``worker_binding`` mappings or a
    mapping under ``environment`` plus optional ``visibility`` fields. A
    missing shape is a typed refusal; scheduler discovery and facility-shaped
    nodefile parsing remain outside this executor module.
    """

    if not isinstance(allocation, ConfiguredAllocation):
        raise BindingError("binding resolution requires ConfiguredAllocation")
    facts = allocation.facts
    raw = facts.get("binding", facts.get("worker_binding"))
    if raw is None:
        raw = {
            key: facts[key]
            for key in ("environment", "env", "visibility", "accelerator_visibility", "worker_id", "node", "nodes")
            if key in facts
        }
    if not raw:
        raise BindingError("configured allocation has no worker binding")
    return WorkerBinding.from_mapping(raw)


__all__ = [
    "AllocationSlot",
    "SlotRecord",
    "BindingError",
    "WorkerBinding",
    "binding_from_allocation",
    "bindings_from_allocation",
    "allocation_slots",
    "slots_from_allocation",
]
