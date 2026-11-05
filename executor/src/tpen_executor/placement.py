"""Explicit allocation-to-worker binding records for attached dispatch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .allocations import ConfiguredAllocation


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


@dataclass(frozen=True, slots=True)
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


__all__ = ["BindingError", "WorkerBinding", "binding_from_allocation"]
