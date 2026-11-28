"""Capture one assigned full-node scope from parent scheduler evidence.

The parent allocation is provenance.  The effective allocation given to the
executor contains only the assigned host, while the complete parent nodefile
and its digest remain attached as immutable evidence.  This module performs
no allocation, submission, recovery, or scheduler discovery.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any, ClassVar

from .._codec import canonical_bytes, decode_json, fields, sha256_text, text
from .._records import frozen_record
from ..allocations import ConfiguredAllocation
from ..contracts import AllocationId, BackendJobId, BatchJobId
from ..placement import (
    ALLOCATION_SLOTS_V2_SCHEMA,
    AllocationSlot,
    PlacementRefusal,
    captured_scheduler_host_map,
    validate_physical_device_entitlements,
)
from ..scheduling.pbs import PackedNodeLayout, PbsVenue


_SNAPSHOT_KEYS = {
    "schema", "parent_allocation_id", "parent_batch_job_id", "parent_job_id", "parent_nodefile_text",
    "parent_nodefile_digest", "parent_hosts", "requested_nodes", "requested_cpus_per_node",
    "requested_gpus_per_node", "delivered_nodes", "delivered_cpus_per_node", "delivered_gpus_per_node",
}
_ASSIGNMENT_KEYS = {
    "schema", "parent", "host", "allowed_cpus", "device_uuids", "node_layout", "slots",
    "outer_packing_count", "controller_rank_count",
}


class AllocationCaptureRefusal(ValueError):
    """Raised when parent or assigned-node evidence is not admissible."""

    def __init__(self, code: str, reason: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("allocation refusal code must be non-empty text")
        if type(reason) is not str or not reason:
            raise TypeError("allocation refusal reason must be non-empty text")
        self.code = code
        self.reason = reason
        super().__init__(reason)

    def to_dict(self) -> dict[str, str]:
        """Return the stable refusal envelope."""

        return {"code": self.code, "reason": self.reason}


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        raise AllocationCaptureRefusal("invalid-resource-evidence", f"{name} must be a positive integer")
    return value


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AllocationCaptureRefusal("affinity-unavailable", f"{name} must be a sequence")
    result = tuple(value)
    if not result or any(type(item) is not int or isinstance(item, bool) or item < 0 for item in result):
        raise AllocationCaptureRefusal("affinity-unavailable", f"{name} must contain non-negative CPU ids")
    if len(set(result)) != len(result):
        raise AllocationCaptureRefusal("affinity-ambiguous", f"{name} contains duplicate CPU ids")
    return tuple(sorted(result))


def _uuid_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise AllocationCaptureRefusal("device-inventory-unavailable", "device UUID inventory must be a sequence")
    result = tuple(value)
    if not result or any(type(item) is not str or not item.strip() for item in result):
        raise AllocationCaptureRefusal("device-inventory-unavailable", "device UUID inventory is incomplete")
    if len(set(result)) != len(result):
        raise AllocationCaptureRefusal("duplicate-device-inventory", "device UUID inventory contains duplicates")
    return result


def _digest_nodefile(nodefile_text: str) -> str:
    return sha256(nodefile_text.encode("utf-8")).hexdigest()


@frozen_record
class SchedulerAllocationSnapshot:
    """Immutable parent allocation and carrier provenance.

    Parameters
    ----------
    parent_allocation_id : AllocationId
        Parent scheduler allocation identity.
    parent_batch_job_id : BatchJobId
        Outer carrier identity associated with the parent allocation.
    parent_job_id : BackendJobId
        Persistent backend job identity associated with the node group.
    parent_nodefile_text : str
        Complete captured parent host text, retained for provenance.
    requested_*_per_node, delivered_*_per_node : int
        Resource evidence for the parent request and effective delivery.
    """

    schema: ClassVar[str] = "tpen-executor/scheduler-allocation-snapshot-v1"
    parent_allocation_id: AllocationId
    parent_batch_job_id: BatchJobId
    parent_job_id: BackendJobId
    parent_nodefile_text: str
    parent_nodefile_digest: str
    parent_hosts: tuple[str, ...]
    requested_nodes: int
    requested_cpus_per_node: int
    requested_gpus_per_node: int
    delivered_nodes: int
    delivered_cpus_per_node: int
    delivered_gpus_per_node: int

    def __post_init__(self) -> None:
        if not isinstance(self.parent_allocation_id, AllocationId):
            raise TypeError("parent_allocation_id must be AllocationId")
        if not isinstance(self.parent_batch_job_id, BatchJobId):
            raise TypeError("parent_batch_job_id must be BatchJobId")
        if not isinstance(self.parent_job_id, BackendJobId):
            raise TypeError("parent_job_id must be BackendJobId")
        nodefile = text(self.parent_nodefile_text, "parent_nodefile_text")
        try:
            hosts = captured_scheduler_host_map({"pbs_nodefile_text": nodefile})
        except PlacementRefusal as error:
            raise AllocationCaptureRefusal(error.code, error.reason) from error
        expected_digest = _digest_nodefile(nodefile)
        if self.parent_nodefile_digest != expected_digest:
            raise AllocationCaptureRefusal("parent-provenance-mismatch", "parent nodefile digest does not match captured text")
        sha256_text(self.parent_nodefile_digest, "parent_nodefile_digest")
        if tuple(self.parent_hosts) != hosts:
            raise AllocationCaptureRefusal("parent-provenance-mismatch", "parent host set does not match captured nodefile")
        for name in (
            "requested_nodes", "requested_cpus_per_node", "requested_gpus_per_node",
            "delivered_nodes", "delivered_cpus_per_node", "delivered_gpus_per_node",
        ):
            _positive_int(getattr(self, name), name)
        object.__setattr__(self, "parent_nodefile_text", nodefile)
        object.__setattr__(self, "parent_hosts", hosts)

    @classmethod
    def capture(
        cls,
        parent_allocation_id: AllocationId,
        parent_batch_job_id: BatchJobId,
        parent_job_id: BackendJobId,
        parent_nodefile_text: str,
        *,
        requested_nodes: int,
        requested_cpus_per_node: int,
        requested_gpus_per_node: int,
        delivered_nodes: int,
        delivered_cpus_per_node: int,
        delivered_gpus_per_node: int,
    ) -> "SchedulerAllocationSnapshot":
        """Capture parent host text exactly once and build its snapshot."""

        return cls(
            parent_allocation_id,
            parent_batch_job_id,
            parent_job_id,
            parent_nodefile_text,
            _digest_nodefile(parent_nodefile_text),
            captured_scheduler_host_map({"pbs_nodefile_text": parent_nodefile_text}),
            requested_nodes,
            requested_cpus_per_node,
            requested_gpus_per_node,
            delivered_nodes,
            delivered_cpus_per_node,
            delivered_gpus_per_node,
        )

    @property
    def balsam_job_id(self) -> BackendJobId:
        """Return the persistent backend job identity under its common name."""

        return self.parent_job_id

    @property
    def batch_job_id(self) -> BatchJobId:
        """Return the outer carrier identity under its common name."""

        return self.parent_batch_job_id

    @property
    def parent_nodefile(self) -> str:
        """Return the complete captured parent nodefile text."""

        return self.parent_nodefile_text

    @property
    def nodefile_sha256(self) -> str:
        """Return the captured nodefile digest."""

        return self.parent_nodefile_digest

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical provenance mapping."""

        return {
            "schema": self.schema,
            "parent_allocation_id": self.parent_allocation_id.to_dict(),
            "parent_batch_job_id": self.parent_batch_job_id.to_dict(),
            "parent_job_id": self.parent_job_id.to_dict(),
            "parent_nodefile_text": self.parent_nodefile_text,
            "parent_nodefile_digest": self.parent_nodefile_digest,
            "parent_hosts": list(self.parent_hosts),
            "requested_nodes": self.requested_nodes,
            "requested_cpus_per_node": self.requested_cpus_per_node,
            "requested_gpus_per_node": self.requested_gpus_per_node,
            "delivered_nodes": self.delivered_nodes,
            "delivered_cpus_per_node": self.delivered_cpus_per_node,
            "delivered_gpus_per_node": self.delivered_gpus_per_node,
        }

    def encode(self) -> bytes:
        """Encode canonical provenance bytes."""

        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Any) -> "SchedulerAllocationSnapshot":
        """Decode one strict parent snapshot."""

        data = fields(value, _SNAPSHOT_KEYS, cls.schema, cls.__name__)
        return cls(
            AllocationId.from_mapping(data["parent_allocation_id"]),
            BatchJobId.from_mapping(data["parent_batch_job_id"]),
            BackendJobId.from_mapping(data["parent_job_id"]),
            data["parent_nodefile_text"],
            data["parent_nodefile_digest"],
            tuple(data["parent_hosts"]),
            data["requested_nodes"], data["requested_cpus_per_node"], data["requested_gpus_per_node"],
            data["delivered_nodes"], data["delivered_cpus_per_node"], data["delivered_gpus_per_node"],
        )

    @classmethod
    def decode(cls, value: bytes | str | Mapping[str, Any]) -> "SchedulerAllocationSnapshot":
        """Decode canonical parent snapshot bytes or a mapping."""

        return cls.from_mapping(decode_json(value))


@frozen_record
class FullNodeAssignment:
    """Validated one-host assignment and its physical row slots.

    Parameters
    ----------
    parent : SchedulerAllocationSnapshot
        Immutable parent scheduler provenance.
    host : str
        Assigned host, which must be present in ``parent``.
    allowed_cpus, device_uuids, node_layout, slots
        Caller-supplied host-local evidence and the declared topology.
    outer_packing_count, controller_rank_count : int
        Node-group cardinality evidence; both must equal one.

    Raises
    ------
    AllocationCaptureRefusal
        If caller-supplied host-local evidence does not establish the declared
        full-node topology. Identity cross-checks are optional.
    """

    schema: ClassVar[str] = "tpen-executor/full-node-assignment-v1"
    parent: SchedulerAllocationSnapshot
    host: str
    allowed_cpus: tuple[int, ...]
    device_uuids: tuple[str, ...]
    node_layout: PackedNodeLayout
    slots: tuple[AllocationSlot, ...]
    outer_packing_count: int
    controller_rank_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.parent, SchedulerAllocationSnapshot):
            raise TypeError("parent must be SchedulerAllocationSnapshot")
        if not isinstance(self.node_layout, PackedNodeLayout):
            raise TypeError("node_layout must be PackedNodeLayout")
        host = text(self.host, "host")
        if host not in self.parent.parent_hosts:
            raise AllocationCaptureRefusal("assigned-host-not-in-parent", "assigned host is absent from parent provenance")
        if self.parent.requested_nodes < 1 or self.parent.delivered_nodes < 1:
            raise AllocationCaptureRefusal("multi-node-assignment", "parent allocation has no delivered node")
        if self.parent.requested_cpus_per_node < self.node_layout.hardware_threads:
            raise AllocationCaptureRefusal("resource-undergrant", "parent CPU request is below the full-node layout")
        if self.parent.requested_gpus_per_node < self.node_layout.gpus_per_node:
            raise AllocationCaptureRefusal("resource-undergrant", "parent GPU request is below the full-node layout")
        if self.parent.delivered_cpus_per_node < self.node_layout.hardware_threads:
            raise AllocationCaptureRefusal("resource-undergrant", "delivered CPUs do not cover the full-node layout")
        if self.parent.delivered_gpus_per_node < self.node_layout.gpus_per_node:
            raise AllocationCaptureRefusal("resource-undergrant", "delivered GPUs do not cover the full-node layout")
        allowed = _int_tuple(self.allowed_cpus, "allowed_cpus")
        devices = _uuid_tuple(self.device_uuids)
        if len(devices) != self.node_layout.gpus_per_node:
            raise AllocationCaptureRefusal("device-count-mismatch", "device inventory count differs from node layout")
        if len(self.slots) != self.node_layout.gpus_per_node:
            raise AllocationCaptureRefusal("slot-count-mismatch", "slot count differs from node layout")
        if isinstance(self.outer_packing_count, bool) or not isinstance(self.outer_packing_count, int) or self.outer_packing_count != 1:
            raise AllocationCaptureRefusal("packing-mismatch", "node-group outer packing count must be one")
        if isinstance(self.controller_rank_count, bool) or not isinstance(self.controller_rank_count, int) or self.controller_rank_count != 1:
            raise AllocationCaptureRefusal("rank-mismatch", "node-group controller rank count must be one")
        expected_cpus = set(allowed)
        if not set(range(self.node_layout.hardware_threads)).issubset(expected_cpus):
            raise AllocationCaptureRefusal("incomplete-node-affinity", "controller affinity does not cover the declared full-node CPU domain")
        seen_cpus: set[int] = set()
        for index, slot in enumerate(self.slots):
            if not isinstance(slot, AllocationSlot) or slot.node != host:
                raise AllocationCaptureRefusal("slot-scope-mismatch", "every slot must name the assigned host")
            if slot.devices != (devices[index],):
                raise AllocationCaptureRefusal("device-topology-mismatch", "slot device order differs from observed inventory")
            expected_slot_cpus = self.node_layout.cpu_ids(index)
            if slot.cpus != expected_slot_cpus:
                raise AllocationCaptureRefusal("cpu-topology-mismatch", "slot CPU set differs from declared node topology")
            if not set(slot.cpus).issubset(expected_cpus):
                raise AllocationCaptureRefusal("affinity-outside-entitlement", "declared slot CPUs exceed controller affinity")
            if seen_cpus & set(slot.cpus):
                raise AllocationCaptureRefusal("overlapping-cpu-entitlement", "slot CPU entitlements overlap")
            seen_cpus.update(slot.cpus)
        try:
            validate_physical_device_entitlements(self.slots)
        except PlacementRefusal as error:
            raise AllocationCaptureRefusal(error.code, error.reason) from error
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "allowed_cpus", allowed)
        object.__setattr__(self, "device_uuids", devices)

    @property
    def effective_hosts(self) -> tuple[str, ...]:
        """Return the one-host scope supplied to routed placement."""

        return (self.host,)

    @property
    def job_id(self) -> BackendJobId:
        """Return the persistent backend job identity."""

        return self.parent.parent_job_id

    @property
    def batch_job_id(self) -> BatchJobId:
        """Return the outer carrier identity."""

        return self.parent.parent_batch_job_id

    @property
    def allocation(self) -> ConfiguredAllocation:
        """Return the effective one-host allocation record."""

        return self.to_configured_allocation()

    def to_configured_allocation(self) -> ConfiguredAllocation:
        """Build an allocation whose effective scheduler scope is one host."""

        facts = {
            "scheduler": {"pbs_nodefile_text": self.host},
            "slots": {"schema": ALLOCATION_SLOTS_V2_SCHEMA, "slots": [slot.to_dict() for slot in self.slots]},
            "parent_provenance": self.parent.to_dict(),
            "batch_job_id": self.parent.parent_batch_job_id.to_dict(),
            "job_id": self.parent.parent_job_id.to_dict(),
            "outer_packing_count": self.outer_packing_count,
            "controller_rank_count": self.controller_rank_count,
        }
        return ConfiguredAllocation(self.parent.parent_allocation_id, self.parent.parent_job_id, facts)

    def configured_allocation(self) -> ConfiguredAllocation:
        """Compatibility alias for :meth:`to_configured_allocation`."""

        return self.to_configured_allocation()

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical assignment mapping."""

        return {
            "schema": self.schema,
            "parent": self.parent.to_dict(),
            "host": self.host,
            "allowed_cpus": list(self.allowed_cpus),
            "device_uuids": list(self.device_uuids),
            "node_layout": self.node_layout.to_dict(),
            "slots": [slot.to_dict() for slot in self.slots],
            "outer_packing_count": self.outer_packing_count,
            "controller_rank_count": self.controller_rank_count,
        }

    def encode(self) -> bytes:
        """Encode canonical assignment bytes."""

        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Any) -> "FullNodeAssignment":
        """Decode one strict assignment."""

        data = fields(value, _ASSIGNMENT_KEYS, cls.schema, cls.__name__)
        return cls(
            SchedulerAllocationSnapshot.from_mapping(data["parent"]), data["host"], tuple(data["allowed_cpus"]),
            tuple(data["device_uuids"]), PackedNodeLayout.from_mapping(data["node_layout"]),
            tuple(AllocationSlot.from_mapping(slot) for slot in data["slots"]),
            data["outer_packing_count"], data["controller_rank_count"],
        )

    @classmethod
    def decode(cls, value: bytes | str | Mapping[str, Any]) -> "FullNodeAssignment":
        """Decode canonical assignment bytes or a mapping."""

        return cls.from_mapping(decode_json(value))


def assign_full_node(
    parent: SchedulerAllocationSnapshot,
    *,
    host: str,
    allowed_cpus: Sequence[int],
    device_uuids: Sequence[str],
    node_layout: PackedNodeLayout,
    venue: PbsVenue | None = None,
    observed_allocation_id: AllocationId | None = None,
    observed_batch_job_id: BatchJobId | None = None,
    observed_job_id: BackendJobId | None = None,
    outer_packing_count: int = 1,
    controller_rank_count: int = 1,
) -> FullNodeAssignment:
    """Validate observed full-node evidence and derive physical row slots.

    Parameters
    ----------
    parent : SchedulerAllocationSnapshot
        Immutable parent scheduler provenance.
    host, allowed_cpus, device_uuids, node_layout
        Assigned host and host-local observed evidence plus declared topology.
        The caller must supply this evidence; identity cross-checks are optional.

    Returns
    -------
    FullNodeAssignment
        Validated one-host assignment with derived physical row slots.

    Raises
    ------
    AllocationCaptureRefusal
        If host-local affinity, topology, inventory, cardinality, or optional
        identity evidence is not admissible.
    """

    if venue is not None:
        if not isinstance(venue, PbsVenue):
            raise TypeError("venue must be PbsVenue")
        if venue.node != node_layout:
            raise AllocationCaptureRefusal("venue-topology-mismatch", "observed layout differs from declared venue data")
    observed = (
        (observed_allocation_id, parent.parent_allocation_id, "allocation"),
        (observed_batch_job_id, parent.parent_batch_job_id, "batch job"),
        (observed_job_id, parent.parent_job_id, "backend job"),
    )
    for actual, expected, name in observed:
        if actual is not None and actual != expected:
            raise AllocationCaptureRefusal("identity-association-mismatch", f"observed {name} differs from parent provenance")
    allowed = _int_tuple(allowed_cpus, "allowed_cpus")
    devices = _uuid_tuple(device_uuids)
    if len(devices) != node_layout.gpus_per_node:
        raise AllocationCaptureRefusal("device-count-mismatch", "device inventory count differs from node layout")
    slots = tuple(
        AllocationSlot(
            f"{parent.parent_job_id.value}-{host}-slot-{index}",
            host,
            (device,),
            node_layout.cpu_ids(index),
            (),
            (),
        )
        for index, device in enumerate(devices)
    )
    return FullNodeAssignment(
        parent, host, allowed, devices, node_layout, slots, outer_packing_count, controller_rank_count
    )


def capture_parent_allocation(*args: Any, **kwargs: Any) -> SchedulerAllocationSnapshot:
    """Alias for the explicit parent snapshot capture operation."""

    if "balsam_job_id" in kwargs and "parent_job_id" not in kwargs:
        kwargs["parent_job_id"] = kwargs.pop("balsam_job_id")
    if "nodefile_text" in kwargs and "parent_nodefile_text" not in kwargs:
        kwargs["parent_nodefile_text"] = kwargs.pop("nodefile_text")
    return SchedulerAllocationSnapshot.capture(*args, **kwargs)


capture_full_node_assignment = assign_full_node
capture_node_assignment = assign_full_node


__all__ = [
    "AllocationCaptureRefusal",
    "FullNodeAssignment",
    "SchedulerAllocationSnapshot",
    "assign_full_node",
    "capture_full_node_assignment",
    "capture_node_assignment",
    "capture_parent_allocation",
]
