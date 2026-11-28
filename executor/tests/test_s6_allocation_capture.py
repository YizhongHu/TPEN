"""Focused S6 parent-provenance and full-node assignment checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen_executor.lifecycle.allocation_capture import (
    AllocationCaptureRefusal,
    FullNodeAssignment,
    SchedulerAllocationSnapshot,
    assign_full_node,
)
from tpen_executor.placement import routed_slots_from_allocation
from tpen_executor.contracts import AllocationId, BackendJobId, BatchJobId
from tpen_executor.scheduling.pbs import PackedNodeLayout, PbsQueueLimit, PbsVenue


MODULE = Path(__file__).parents[1] / "src" / "tpen_executor" / "lifecycle" / "allocation_capture.py"
MODULE_NAME = "executor/src/tpen_executor/lifecycle/allocation_capture.py"
MUTANTS = Path(__file__).parents[2] / "scripts" / ("c" "annon") / "tests" / "mutants-s6-packed-allocator.json"


def _snapshot(*, nodes: str = "node-a\n", requested_nodes: int = 1, delivered_nodes: int = 1) -> SchedulerAllocationSnapshot:
    """Build a parent fixture with complete carrier provenance."""

    return SchedulerAllocationSnapshot.capture(
        AllocationId("parent-allocation"),
        BatchJobId("parent-batch"),
        BackendJobId("group-job"),
        nodes,
        requested_nodes=requested_nodes,
        requested_cpus_per_node=16,
        requested_gpus_per_node=4,
        delivered_nodes=delivered_nodes,
        delivered_cpus_per_node=16,
        delivered_gpus_per_node=4,
    )


def _layout() -> PackedNodeLayout:
    return PackedNodeLayout(8, 16, 4, ("0-3", "4-7", "8-11", "12-15"))


def _devices(seed: str) -> tuple[str, ...]:
    return tuple(f"GPU-{seed}{index:01X}000000-0000-0000-0000-000000000000" for index in range(4))


def test_capture_preserves_parent_and_routes_only_assigned_host() -> None:
    """A two-host parent yields one-host effective scopes with shared provenance."""

    parent = _snapshot(nodes="node-a\nnode-b\n", requested_nodes=2, delivered_nodes=2)
    layout = _layout()
    first = assign_full_node(parent, host="node-a", allowed_cpus=range(16), device_uuids=_devices("A"), node_layout=layout)
    second = assign_full_node(parent, host="node-b", allowed_cpus=range(16), device_uuids=_devices("B"), node_layout=layout)

    for assignment in (first, second):
        allocation = assignment.to_configured_allocation()
        assert allocation.facts["parent_provenance"]["parent_hosts"] == ("node-a", "node-b")
        assert routed_slots_from_allocation(allocation)
        assert {slot.node for slot in routed_slots_from_allocation(allocation)} == {assignment.host}
        assert len(assignment.slots) == 4


def test_capture_requires_full_node_and_exact_association() -> None:
    """Under-granted resources, crossed hosts, and wrong topology refuse."""

    undergrant = SchedulerAllocationSnapshot.capture(
        AllocationId("parent-allocation"), BatchJobId("parent-batch"), BackendJobId("group-job"), "node-a\n",
        requested_nodes=1, requested_cpus_per_node=8, requested_gpus_per_node=4,
        delivered_nodes=1, delivered_cpus_per_node=16, delivered_gpus_per_node=4,
    )
    with pytest.raises(AllocationCaptureRefusal, match="full-node"):
        assign_full_node(
            undergrant, host="node-a", allowed_cpus=range(16), device_uuids=_devices("A"),
            node_layout=PackedNodeLayout(8, 16, 4, ("0-3", "4-7", "8-11", "12-15")),
        )

    delivered_undergrant = SchedulerAllocationSnapshot.capture(
        AllocationId("parent-allocation"), BatchJobId("parent-batch"), BackendJobId("group-job"), "node-a\n",
        requested_nodes=1, requested_cpus_per_node=16, requested_gpus_per_node=4,
        delivered_nodes=1, delivered_cpus_per_node=8, delivered_gpus_per_node=4,
    )
    with pytest.raises(AllocationCaptureRefusal, match="delivered CPUs"):
        assign_full_node(
            delivered_undergrant, host="node-a", allowed_cpus=range(16), device_uuids=_devices("A"), node_layout=_layout(),
        )

    with pytest.raises(AllocationCaptureRefusal, match="absent from parent"):
        assign_full_node(_snapshot(), host="node-b", allowed_cpus=range(16), device_uuids=_devices("A"), node_layout=_layout())

    with pytest.raises(AllocationCaptureRefusal, match="differs from parent"):
        assign_full_node(
            _snapshot(), host="node-a", allowed_cpus=range(16), device_uuids=_devices("A"), node_layout=_layout(),
            observed_batch_job_id=BatchJobId("foreign-batch"),
        )

    with pytest.raises(AllocationCaptureRefusal, match="physical UUID"):
        assign_full_node(
            _snapshot(), host="node-a", allowed_cpus=range(16),
            device_uuids=("not-a-physical-id", *_devices("A")[1:]), node_layout=_layout(),
        )

    venue = PbsVenue("machine", ("shared",), (PbsQueueLimit("short", 1, 30, 1),), _layout())
    with pytest.raises(AllocationCaptureRefusal, match="declared venue"):
        assign_full_node(
            _snapshot(), host="node-a", allowed_cpus=range(16), device_uuids=_devices("A"),
            node_layout=PackedNodeLayout(8, 16, 2, ("0-3", "4-7")), venue=venue,
        )


def test_capture_round_trips_and_keeps_venue_descriptive() -> None:
    """Venue data validates topology but exposes no submission operation."""

    parent = _snapshot()
    assignment = assign_full_node(parent, host="node-a", allowed_cpus=range(16), device_uuids=_devices("A"), node_layout=_layout())
    assert SchedulerAllocationSnapshot.decode(parent.encode()) == parent
    assert FullNodeAssignment.decode(assignment.encode()) == assignment
    assert not any(name.startswith("submit") for name in dir(PbsVenue))


def test_s6_mutants_are_exact_once_and_compile_valid() -> None:
    """Every bounded S6 mutant has one source anchor and compiles in memory."""

    source = MODULE.read_text(encoding="utf-8")
    manifest = json.loads(MUTANTS.read_text(encoding="utf-8"))
    for mutant in manifest["mutants"]:
        assert mutant["target"] == MODULE_NAME
        assert mutant["killer"].startswith("executor/tests/")
        assert "::" in mutant["killer"]
        assert source.count(mutant["old"]) == 1
        compile(source.replace(mutant["old"], mutant["new"], 1), MODULE_NAME, "exec")
