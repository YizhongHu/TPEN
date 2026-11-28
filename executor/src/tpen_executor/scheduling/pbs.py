"""Facility-neutral typed records for a bounded PBS carrier request.

The records describe machine data and a finite request.  They never contact a
scheduler and do not grant permission to submit work.
"""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any, ClassVar

from .._codec import canonical_bytes, decode_json, fields, text
from .._records import frozen_record


class PbsSchedulingRefusal(ValueError):
    """Raised when scheduler data or a carrier request is invalid."""


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z", re.ASCII)


def _identifier(value: Any, name: str) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None:
        raise PbsSchedulingRefusal(f"{name} must be an ASCII identifier-like string")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        raise PbsSchedulingRefusal(f"{name} must be a positive integer")
    return value


def _string_tuple(value: Any, name: str, *, nonempty: bool = True) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PbsSchedulingRefusal(f"{name} must be a sequence")
    result = tuple(value)
    if any(type(item) is not str or (nonempty and not item.strip()) for item in result):
        raise PbsSchedulingRefusal(f"{name} must contain strings")
    if len(result) != len(set(result)):
        raise PbsSchedulingRefusal(f"{name} must not contain duplicates")
    return result


def _cpu_set(value: Any, name: str) -> tuple[str, tuple[int, ...]]:
    if type(value) is not str or not value.strip():
        raise PbsSchedulingRefusal(f"{name} must be non-empty text")
    cpus: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            raise PbsSchedulingRefusal(f"{name} contains an empty CPU token")
        if "-" in token:
            pieces = token.split("-")
            if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
                raise PbsSchedulingRefusal(f"{name} contains an invalid CPU range")
            start, stop = (int(piece) for piece in pieces)
            if start > stop:
                raise PbsSchedulingRefusal(f"{name} contains a reversed CPU range")
            cpus.update(range(start, stop + 1))
        elif token.isdigit():
            cpus.add(int(token))
        else:
            raise PbsSchedulingRefusal(f"{name} contains an invalid CPU token")
    if not cpus:
        raise PbsSchedulingRefusal(f"{name} must contain a CPU")
    return value.strip(), tuple(sorted(cpus))


@frozen_record
class PbsQueueLimit:
    """One named queue's finite resource ceilings."""

    schema: ClassVar[str] = "tpen-executor/pbs-queue-limit-v1"
    name: str
    max_nodes: int
    max_walltime_minutes: int
    max_queued_jobs: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "queue name"))
        for name in ("max_nodes", "max_walltime_minutes", "max_queued_jobs"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation."""
        return {"schema": self.schema, "name": self.name, "max_nodes": self.max_nodes,
                "max_walltime_minutes": self.max_walltime_minutes,
                "max_queued_jobs": self.max_queued_jobs}

    def encode(self) -> bytes:
        """Encode the canonical queue record."""
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Any) -> "PbsQueueLimit":
        """Decode one strict queue record."""
        data = fields(value, {"schema", "name", "max_nodes", "max_walltime_minutes", "max_queued_jobs"},
                      cls.schema, cls.__name__)
        return cls(data["name"], data["max_nodes"], data["max_walltime_minutes"], data["max_queued_jobs"])

    @classmethod
    def decode(cls, value: bytes | str | dict[str, Any]) -> "PbsQueueLimit":
        """Decode canonical JSON bytes or a mapping."""
        return cls.from_mapping(decode_json(value))


@frozen_record
class PackedNodeLayout:
    """Typed single-node CPU/GPU topology used by placement capture."""

    schema: ClassVar[str] = "tpen-executor/packed-node-layout-v1"
    physical_cores: int
    hardware_threads: int
    gpus_per_node: int
    cpu_sets_by_gpu_pci_order: tuple[str, ...]

    def __post_init__(self) -> None:
        physical = _positive_int(self.physical_cores, "physical_cores")
        threads = _positive_int(self.hardware_threads, "hardware_threads")
        gpus = _positive_int(self.gpus_per_node, "gpus_per_node")
        if threads < physical:
            raise PbsSchedulingRefusal("hardware_threads cannot be below physical_cores")
        if isinstance(self.cpu_sets_by_gpu_pci_order, (str, bytes, bytearray)):
            raise PbsSchedulingRefusal("cpu_sets_by_gpu_pci_order must be a sequence")
        try:
            raw = tuple(self.cpu_sets_by_gpu_pci_order)
        except TypeError as error:
            raise PbsSchedulingRefusal("cpu_sets_by_gpu_pci_order must be a sequence") from error
        if len(raw) != gpus:
            raise PbsSchedulingRefusal("one CPU set is required for each GPU")
        normalized: list[str] = []
        seen: set[int] = set()
        for index, value in enumerate(raw):
            display, cpus = _cpu_set(value, f"CPU set {index}")
            if set(cpus) & seen:
                raise PbsSchedulingRefusal("GPU CPU sets must be disjoint")
            if any(cpu >= threads for cpu in cpus):
                raise PbsSchedulingRefusal("CPU set exceeds hardware_threads")
            seen.update(cpus)
            normalized.append(display)
        object.__setattr__(self, "physical_cores", physical)
        object.__setattr__(self, "hardware_threads", threads)
        object.__setattr__(self, "gpus_per_node", gpus)
        object.__setattr__(self, "cpu_sets_by_gpu_pci_order", tuple(normalized))

    def cpu_ids(self, gpu_index: int) -> tuple[int, ...]:
        """Return the canonical CPU ids assigned to one GPU index."""
        if type(gpu_index) is not int or not 0 <= gpu_index < self.gpus_per_node:
            raise PbsSchedulingRefusal("GPU index is outside the declared layout")
        return _cpu_set(self.cpu_sets_by_gpu_pci_order[gpu_index], "CPU set")[1]

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation."""
        return {"schema": self.schema, "physical_cores": self.physical_cores,
                "hardware_threads": self.hardware_threads, "gpus_per_node": self.gpus_per_node,
                "cpu_sets_by_gpu_pci_order": list(self.cpu_sets_by_gpu_pci_order)}

    def encode(self) -> bytes:
        """Encode the canonical layout record."""
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Any) -> "PackedNodeLayout":
        """Decode one strict layout record."""
        data = fields(value, {"schema", "physical_cores", "hardware_threads", "gpus_per_node",
                              "cpu_sets_by_gpu_pci_order"}, cls.schema, cls.__name__)
        return cls(data["physical_cores"], data["hardware_threads"], data["gpus_per_node"],
                   data["cpu_sets_by_gpu_pci_order"])

    @classmethod
    def decode(cls, value: bytes | str | dict[str, Any]) -> "PackedNodeLayout":
        """Decode canonical JSON bytes or a mapping."""
        return cls.from_mapping(decode_json(value))


@frozen_record
class PbsVenue:
    """Machine-only scheduler data; loading it grants no submission authority."""

    schema: ClassVar[str] = "tpen-executor/pbs-venue-v1"
    system: str
    filesystems: tuple[str, ...]
    queues: tuple[PbsQueueLimit, ...]
    node: PackedNodeLayout

    def __post_init__(self) -> None:
        object.__setattr__(self, "system", text(self.system, "system"))
        filesystems = _string_tuple(self.filesystems, "filesystems")
        if not filesystems:
            raise PbsSchedulingRefusal("filesystems must not be empty")
        if type(self.queues) is not tuple:
            try:
                queues = tuple(self.queues)
            except TypeError as error:
                raise PbsSchedulingRefusal("queues must be a sequence") from error
        else:
            queues = self.queues
        if not queues or any(type(queue) is not PbsQueueLimit for queue in queues):
            raise PbsSchedulingRefusal("queues must contain PbsQueueLimit records")
        if len({queue.name for queue in queues}) != len(queues):
            raise PbsSchedulingRefusal("queue names must be unique")
        if not isinstance(self.node, PackedNodeLayout):
            raise TypeError("node must be PackedNodeLayout")
        object.__setattr__(self, "filesystems", filesystems)
        object.__setattr__(self, "queues", queues)

    def queue(self, name: str) -> PbsQueueLimit:
        """Return a declared queue or refuse an undeclared one."""
        for queue in self.queues:
            if queue.name == name:
                return queue
        raise PbsSchedulingRefusal(f"queue {name!r} is not declared by the venue")

    def validate_resources(self, resources: "PbsBatchResources") -> None:
        """Check a bounded carrier request against this venue's ceilings."""
        if not isinstance(resources, PbsBatchResources):
            raise TypeError("resources must be PbsBatchResources")
        queue = self.queue(resources.queue)
        if resources.nodes > queue.max_nodes:
            raise PbsSchedulingRefusal("requested nodes exceed queue limit")
        if resources.walltime_minutes > queue.max_walltime_minutes:
            raise PbsSchedulingRefusal("requested walltime exceeds queue limit")

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation."""
        return {"schema": self.schema, "system": self.system, "filesystems": list(self.filesystems),
                "queues": [queue.to_dict() for queue in self.queues], "node": self.node.to_dict()}

    def encode(self) -> bytes:
        """Encode the canonical venue record."""
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Any) -> "PbsVenue":
        """Decode one strict venue record."""
        data = fields(value, {"schema", "system", "filesystems", "queues", "node"}, cls.schema, cls.__name__)
        return cls(data["system"], data["filesystems"],
                   tuple(PbsQueueLimit.from_mapping(item) for item in data["queues"]),
                   PackedNodeLayout.from_mapping(data["node"]))

    @classmethod
    def decode(cls, value: bytes | str | dict[str, Any]) -> "PbsVenue":
        """Decode canonical JSON bytes or a mapping."""
        return cls.from_mapping(decode_json(value))


@frozen_record
class PbsBatchResources:
    """Finite user/project carrier request, independent of deployment effects."""

    schema: ClassVar[str] = "tpen-executor/pbs-batch-resources-v1"
    account: str
    queue: str
    nodes: int
    walltime_minutes: int
    job_mode: str = "serial"

    def __post_init__(self) -> None:
        object.__setattr__(self, "account", text(self.account, "account"))
        object.__setattr__(self, "queue", _identifier(self.queue, "queue"))
        object.__setattr__(self, "nodes", _positive_int(self.nodes, "nodes"))
        object.__setattr__(self, "walltime_minutes", _positive_int(self.walltime_minutes, "walltime_minutes"))
        object.__setattr__(self, "job_mode", _identifier(self.job_mode, "job_mode"))

    def validate(self, venue: PbsVenue) -> None:
        """Validate this request against a typed venue."""
        venue.validate_resources(self)

    @property
    def num_nodes(self) -> int:
        """Return the Balsam BatchJob spelling of ``nodes``."""
        return self.nodes

    @property
    def wall_time_min(self) -> int:
        """Return the Balsam BatchJob spelling of ``walltime_minutes``."""
        return self.walltime_minutes

    @property
    def project(self) -> str:
        """Return the Balsam project/allocation spelling of ``account``."""
        return self.account

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation."""
        return {"schema": self.schema, "account": self.account, "queue": self.queue,
                "nodes": self.nodes, "walltime_minutes": self.walltime_minutes,
                "job_mode": self.job_mode}

    def encode(self) -> bytes:
        """Encode the canonical request."""
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Any) -> "PbsBatchResources":
        """Decode one strict resource record."""
        data = fields(value, {"schema", "account", "queue", "nodes", "walltime_minutes", "job_mode"},
                      cls.schema, cls.__name__)
        return cls(data["account"], data["queue"], data["nodes"], data["walltime_minutes"], data["job_mode"])

    @classmethod
    def decode(cls, value: bytes | str | dict[str, Any]) -> "PbsBatchResources":
        """Decode canonical JSON bytes or a mapping."""
        return cls.from_mapping(decode_json(value))


__all__ = ["PackedNodeLayout", "PbsBatchResources", "PbsQueueLimit", "PbsSchedulingRefusal", "PbsVenue"]
