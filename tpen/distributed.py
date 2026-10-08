"""Explicit execution topology and rank-local resource profile artifacts.

Topology is supplied by the launcher.  This module deliberately does not read
launcher environment variables or infer a device from a rank.  Resource
records are rank-local by construction: this layer has no reduction path, and
``project_scalars`` projects one typed record without summing readings.  The
NODE and JOB labels are reserved for a future explicit aggregate record type
and are deliberately refused until that type exists.
"""

from __future__ import annotations

from collections.abc import Buffer, Mapping, Sequence
import json
import math
import os
import socket
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Union

from tpen.accelerator import (
    AcceleratorIdentity,
    AcceleratorKind,
    AllocatorUnavailable,
    AllocatorUsage,
)
from tpen.durable_append import append_record
from tpen.process_resources import ProcessResourceResult, ResourceUnavailable


class ProfileScope(Enum):
    """Ownership scope of one profile record."""

    PROCESS = "process"
    DEVICE = "device"
    NODE = "node"
    JOB = "job"


@dataclass(frozen=True)
class ExecutionTopology:
    """Immutable identity supplied by a launcher for one Python process.

    Rank fields may be ``None`` when the launcher has no value for that rank.
    ``None`` is preserved; it is never replaced by a guessed rank or a local
    rank used as a device index.
    """

    global_rank: int | None
    global_size: int
    local_rank: int | None
    local_size: int
    node_rank: int | None
    node_size: int
    host: str
    pid: int
    device: str
    job_id: str | None = None
    device_identity: AcceleratorIdentity | None = None

    def __post_init__(self) -> None:
        for name, size in (
            ("global_size", self.global_size),
            ("local_size", self.local_size),
            ("node_size", self.node_size),
        ):
            if type(size) is not int:
                raise TypeError(f"{name} must be an int, got {type(size).__name__}")
            if size < 1:
                raise ValueError(f"{name} must be positive, got {size}")
        for rank_name, rank, size in (
            ("global_rank", self.global_rank, self.global_size),
            ("local_rank", self.local_rank, self.local_size),
            ("node_rank", self.node_rank, self.node_size),
        ):
            if rank is not None and type(rank) is not int:
                raise TypeError(f"{rank_name} must be an int or None, got {type(rank).__name__}")
            if rank is not None and not 0 <= rank < size:
                raise ValueError(f"{rank_name} must be in [0, {size})")
        if not self.host:
            raise ValueError("host must be nonempty")
        if self.pid < 1:
            raise ValueError("pid must be positive")
        if not self.device:
            raise ValueError("device must be nonempty")

    @classmethod
    def single_process(
        cls,
        *,
        device: str,
        device_identity: AcceleratorIdentity | None = None,
        job_id: str | None = None,
    ) -> "ExecutionTopology":
        """Create an explicit one-process topology for the local launcher."""

        return cls(
            global_rank=0,
            global_size=1,
            local_rank=0,
            local_size=1,
            node_rank=0,
            node_size=1,
            host=socket.gethostname(),
            pid=os.getpid(),
            device=device,
            job_id=job_id,
            device_identity=device_identity,
        )

    @property
    def rank_path_component(self) -> str:
        """Return the collision-free global rank component."""

        if self.global_rank is None:
            raise ValueError("global rank is unavailable; cannot create a rank-local path")
        return f"rank-{self.global_rank:05d}"


_RUNNER_TOPOLOGY_FACT_KEYS = frozenset(
    {
        "global_rank",
        "global_size",
        "local_rank",
        "local_size",
        "node_rank",
        "node_size",
        "host",
        "pid",
        "device",
        "job_id",
        "device_identity",
    }
)


def _detach_topology_facts(value: Any) -> Any:
    """Materialize the closed mapping-facts value schema for TPEN ownership."""

    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, Buffer) or hasattr(type(value), "__buffer__"):
        raise ValueError(f"binary buffer refused: {type(value).__name__}")
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("topology fact mapping keys must be exact str")
            copied[key] = _detach_topology_facts(item)
        return copied
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_detach_topology_facts(item) for item in value]
    raise ValueError(f"unsupported topology fact value type: {type(value).__name__}")


def execution_topology_from_facts(
    facts: ExecutionTopology | Mapping[str, Any],
) -> ExecutionTopology:
    """Normalize launcher facts into the typed topology consumed by TPEN.

    The mapping form is the launcher-facing boundary representation. It keeps
    callers that do not own TPEN's topology type independent of this module,
    while ensuring the production consumer receives the same validated typed
    object as callers that already have one.
    """

    if isinstance(facts, Buffer) or hasattr(type(facts), "__buffer__"):
        raise ValueError(f"binary buffer refused: {type(facts).__name__}")
    if isinstance(facts, ExecutionTopology):
        return facts
    if not isinstance(facts, Mapping):
        raise TypeError("topology must be an ExecutionTopology or mapping")

    facts = _detach_topology_facts(facts)

    unsupported = tuple(key for key in facts if key not in _RUNNER_TOPOLOGY_FACT_KEYS)
    if unsupported:
        names = ", ".join(repr(key) for key in unsupported)
        raise ValueError("unsupported production runner topology facts: " + names)
    required = (
        "global_rank",
        "global_size",
        "local_rank",
        "local_size",
        "node_rank",
        "node_size",
        "host",
        "pid",
        "device",
    )
    missing = tuple(name for name in required if name not in facts)
    if missing:
        raise ValueError(
            "production runner topology is missing required facts: " + ", ".join(missing)
        )

    identity_value = facts.get("device_identity")
    if identity_value is None:
        identity = identity_value
    elif isinstance(identity_value, Mapping):
        unsupported_identity = set(identity_value) - {"kind", "index", "uuid"}
        if unsupported_identity:
            names = ", ".join(repr(key) for key in sorted(unsupported_identity, key=str))
            raise ValueError("unsupported topology.device_identity keys: " + names)
        try:
            identity = AcceleratorIdentity(
                kind=AcceleratorKind(str(identity_value["kind"])),
                index=identity_value.get("index"),
                uuid=identity_value.get("uuid"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("topology.device_identity is malformed") from error
    else:
        raise ValueError("topology.device_identity must be a mapping or null")

    try:
        return ExecutionTopology(
            global_rank=facts["global_rank"],
            global_size=facts["global_size"],
            local_rank=facts["local_rank"],
            local_size=facts["local_size"],
            node_rank=facts["node_rank"],
            node_size=facts["node_size"],
            host=facts["host"],
            pid=facts["pid"],
            device=facts["device"],
            job_id=facts.get("job_id"),
            device_identity=identity,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid production runner topology: {error}") from error


Scalar = Union[bool, float, int]


@dataclass(frozen=True)
class ScalarMetric:
    """One JSON-safe scalar metric at the serialization boundary."""

    key: str
    value: Scalar


@dataclass(frozen=True)
class ProfileRecord:
    """Typed resource readings owned by one process and one topology."""

    scope: ProfileScope
    monotonic_time: float
    topology: ExecutionTopology
    process: ProcessResourceResult | None = None
    device: AllocatorUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scope, ProfileScope):
            raise TypeError(f"scope must be a ProfileScope, got {type(self.scope).__name__}")
        if not math.isfinite(self.monotonic_time) or self.monotonic_time < 0:
            raise ValueError("monotonic_time must be finite and nonnegative")
        if self.scope is ProfileScope.PROCESS:
            if self.process is None:
                raise ValueError("PROCESS profile requires process readings")
            if self.device is not None:
                raise ValueError("PROCESS profile must not carry device readings")
        if self.scope is ProfileScope.DEVICE:
            if self.device is None:
                raise ValueError("DEVICE profile requires device readings")
            if self.process is not None:
                raise ValueError("DEVICE profile must not carry process readings")
            if (
                self.topology.device_identity is not None
                and self.device.identity != self.topology.device_identity
            ):
                raise ValueError(
                    "device identity mismatch between topology and allocator usage"
                )
        if self.scope in (ProfileScope.NODE, ProfileScope.JOB):
            raise ValueError("NODE and JOB profiles require an explicit aggregate record")


def project_scalars(record: ProfileRecord) -> tuple[ScalarMetric, ...]:
    """Project one typed record deterministically without aggregating readings."""

    if record.scope is ProfileScope.PROCESS:
        if record.process is None:
            raise ValueError("PROCESS profile requires process readings")
        readings = (
            ("user_cpu_seconds", record.process.user_cpu_seconds),
            ("system_cpu_seconds", record.process.system_cpu_seconds),
            ("read_block_operations", record.process.read_block_operations),
            ("write_block_operations", record.process.write_block_operations),
            ("voluntary_context_switches", record.process.voluntary_context_switches),
            ("involuntary_context_switches", record.process.involuntary_context_switches),
            ("peak_rss_mb", record.process.peak_rss_mb),
        )
        return _project_readings(readings)
    if record.device is None:
        raise ValueError("DEVICE profile requires device readings")
    readings = (
        ("allocated_mb", record.device.allocated_mb),
        ("reserved_mb", record.device.reserved_mb),
        ("device_count", record.device.device_count),
    )
    return _project_readings(readings)


def _project_readings(readings: tuple[tuple[str, object], ...]) -> tuple[ScalarMetric, ...]:
    metrics: list[ScalarMetric] = []
    for key, value in readings:
        if isinstance(value, (ResourceUnavailable, AllocatorUnavailable)):
            metrics.append(ScalarMetric(f"{key}_unavailable", True))
        elif value is None:
            metrics.append(ScalarMetric(f"{key}_unavailable", True))
        elif isinstance(value, float) and not math.isfinite(value):
            # A backend can report a non-finite scalar (e.g. a driver glitch);
            # degrading only this field keeps the finite readings intact
            # instead of losing the whole record at the strict-JSON boundary.
            metrics.append(ScalarMetric(f"{key}_unavailable", True))
        else:
            if not isinstance(value, (bool, float, int)):
                raise TypeError(f"{key} is not a scalar: {type(value).__name__}")
            metrics.append(ScalarMetric(key, value))
    return tuple(metrics)


class RankLocalJSONLWriter:
    """Append profile records to ``profiles/rank-00000/resources.jsonl``."""

    def __init__(self, run_dir: Path | str, topology: ExecutionTopology) -> None:
        self.topology = topology
        self.path = Path(run_dir) / "profiles" / topology.rank_path_component / "resources.jsonl"

    def write(self, record: ProfileRecord) -> None:
        """Append one scalar-projected, topology-stamped JSON record."""

        if record.topology != self.topology:
            raise ValueError("record topology does not belong to this rank-local writer")
        payload = {
            "scope": record.scope.value,
            "time_monotonic": record.monotonic_time,
            "global_rank": record.topology.global_rank,
            "global_size": record.topology.global_size,
            "local_rank": record.topology.local_rank,
            "local_size": record.topology.local_size,
            "node_rank": record.topology.node_rank,
            "node_size": record.topology.node_size,
            "host": record.topology.host,
            "pid": record.topology.pid,
            "device": record.topology.device,
            "job_id": record.topology.job_id,
            "device_identity": None
            if record.topology.device_identity is None
            else {
                "kind": record.topology.device_identity.kind.value,
                "index": record.topology.device_identity.index,
                "uuid": record.topology.device_identity.uuid,
            },
            "metrics": {metric.key: metric.value for metric in project_scalars(record)},
        }
        try:
            serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            payload = {
                "scope": record.scope.value,
                "time_monotonic": record.monotonic_time,
                "global_rank": record.topology.global_rank,
                "serialization_error": {
                    "fields": _unserializable_fields(payload),
                },
            }
            serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
        append_record(self.path, serialized)


def _unserializable_fields(payload: dict[str, object]) -> tuple[str, ...]:
    """Identify top-level fields that cannot cross the JSON boundary."""

    failed: list[str] = []
    for field, value in payload.items():
        try:
            json.dumps(value, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            failed.append(field)
    return tuple(failed)


__all__ = [
    "ExecutionTopology",
    "execution_topology_from_facts",
    "ProfileRecord",
    "ProfileScope",
    "RankLocalJSONLWriter",
    "ScalarMetric",
    "project_scalars",
]
