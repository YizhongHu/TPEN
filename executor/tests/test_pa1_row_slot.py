"""P-a1 explicit per-row physical-slot binding checks."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from tpen_executor.dispatch.contract import _require_configured_physical_slot, _with_child_placement
from tpen_executor.dispatch.transport import LeafRequest
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle.local_executor import CapacityExceededError
from tpen_executor.lifecycle.local_executor import LocalLeafExecutor
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.placement import AllocationSlot, PlacementRefusal

from test_b1e1_local_executor import (
    _allocation,
    _environment,
    _failure_result,
    _group,
    _invocation,
)
from test_s3_transport_wire import _begin
from test_c22_output_binding import _program


MUTANTS = [
    {
        "id": "PA1-foreign-slot",
        "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": "        if len(matching) != 1 or matching[0] != selected_slot:\n",
        "new": "        if False:\n",
        "killer": "test_foreign_selected_slot_is_refused",
    },
    {
        "id": "PA1-legacy-sole-slot",
        "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": "        if len(slots) != 1:\n            raise PlacementRefusal(\n                \"configured-physical-slot-required\",\n",
        "new": "        if False:\n            raise PlacementRefusal(\n                \"configured-physical-slot-required\",\n",
        "killer": "test_absent_selected_slot_keeps_legacy_refusal",
    },
    {
        "id": "PA1-overlay-selected-slot",
        "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": "        slot = _require_configured_physical_slot(allocation, selected_slot)\n",
        "new": "        slot = _require_configured_physical_slot(allocation)\n",
        "killer": "test_env_forgery_cannot_change_selected_slot",
    },
    {
        "id": "PA1-request-selected-slot",
        "target": "executor/src/tpen_executor/dispatch/transport.py",
        "old": "            slot = _require_configured_physical_slot(allocation, selected_slot)\n",
        "new": "            slot = _require_configured_physical_slot(allocation)\n",
        "killer": "test_serialized_leaf_request_carries_selected_slot",
    },
]


def test_mutants_have_unique_executable_anchors_and_killers() -> None:
    module = Path(__file__).read_text(encoding="utf-8")
    assert len({row["id"] for row in MUTANTS}) == len(MUTANTS)
    assert len({row["killer"] for row in MUTANTS}) == len(MUTANTS)
    for row in MUTANTS:
        source = (Path(__file__).parents[2] / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1
        ast.parse(source.replace(row["old"], row["new"]))
        assert f"def {row['killer']}(" in module


def test_serialized_leaf_request_carries_selected_slot(tmp_path: Path) -> None:
    """The producer overlay and serialized request consume the same slot."""

    slot = _allocation(2)
    group, allocation, lifecycle, configured, fence = _begin(
        tmp_path, task_id="member", facts=slot.facts
    )
    configured = replace(configured, config="", overrides=(), program=_program(tmp_path))
    selected = _require_configured_physical_slot(allocation, _allocation_slot(allocation, 0))
    environment = _with_child_placement(
        _environment(tmp_path), allocation, fence, group, group.members[0], configured,
        selected_slot=selected,
    )
    request = LeafRequest.for_attempt(
        fence, allocation, configured, environment, selected_slot=selected,
    )
    decoded = LeafRequest.decode(request.encode())
    assert decoded.slot_id == selected.slot_id
    assert decoded.environment.env["TPEN_SLOT_NODE"] == selected.node
    assert decoded.environment.env["TPEN_SLOT_CPUS"] == ",".join(map(str, selected.cpus))


def _allocation_slot(allocation, index: int) -> AllocationSlot:
    from tpen_executor.placement import physical_slots_from_allocation

    return physical_slots_from_allocation(allocation)[index]


def test_two_slots_admit_two_rows_and_refuse_third_until_collection(tmp_path: Path) -> None:
    """The local producer leases one distinct slot per concurrent row."""

    group = _group(("one", "two", "three"))
    allocation = _allocation(2)
    started = {name: Event() for name in ("one", "two")}
    release = Event()

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        started[request.task_id].set()
        release.wait(5)
        return _failure_result(request).encode()

    def materializer(program):
        return ConfiguredRun(program.config_ref["task"], "", (), _program(tmp_path))

    executor = LocalLeafExecutor(group, allocation, materializer, _environment, leaf_runner=runner)
    try:
        first = _invocation(group, allocation, "one", tmp_path)
        second = _invocation(group, allocation, "two", tmp_path)
        _, handle_a = executor.submit(*first)
        _, handle_b = executor.submit(*second)
        assert started["one"].wait(2) and started["two"].wait(2)
        with pytest.raises(CapacityExceededError, match="capacity"):
            executor.submit(*_invocation(group, allocation, "three", tmp_path))
        release.set()
        executor.collect(handle_a)
        executor.collect(handle_b)
    finally:
        release.set()
        executor.close()


def test_duplicate_concurrent_slot_use_is_refused(tmp_path: Path) -> None:
    """A one-slot pool refuses a second row before it can reuse the slot."""

    group = _group(("one", "two"))
    allocation = _allocation(1)
    release = Event()

    def runner(payload: bytes) -> bytes:
        release.wait(5)
        return _failure_result(LeafRequest.decode(payload)).encode()

    def materializer(program):
        return ConfiguredRun(program.config_ref["task"], "", (), _program(tmp_path))

    executor = LocalLeafExecutor(group, allocation, materializer, _environment, leaf_runner=runner)
    try:
        _, handle = executor.submit(*_invocation(group, allocation, "one", tmp_path))
        with pytest.raises(CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
        release.set()
        executor.collect(handle)
    finally:
        release.set()
        executor.close()


def test_foreign_selected_slot_is_refused() -> None:
    allocation = _allocation(1)
    foreign = AllocationSlot("foreign", "localhost", (), (999999,), (), ())
    with pytest.raises(PlacementRefusal, match="not an exact entitlement"):
        _require_configured_physical_slot(allocation, foreign)


def test_absent_selected_slot_keeps_legacy_refusal() -> None:
    with pytest.raises(PlacementRefusal, match="exactly one"):
        _require_configured_physical_slot(_allocation(2))


def test_one_slot_configured_regression() -> None:
    selected = _allocation_slot(_allocation(1), 0)
    assert _require_configured_physical_slot(_allocation(1), selected) == selected


def test_env_forgery_cannot_change_selected_slot(tmp_path: Path) -> None:
    group, allocation, lifecycle, configured, fence = _begin(
        tmp_path, task_id="member", facts=_allocation(2).facts
    )
    configured = replace(configured, config="", overrides=(), program=_program(tmp_path))
    selected = _allocation_slot(allocation, 0)
    forged = PreparedPythonEnvironment(
        _environment(tmp_path).python,
        env={"TPEN_SLOT_ID": _allocation_slot(allocation, 1).slot_id,
             "TPEN_SLOT_NODE": "forged-node", "TPEN_SLOT_CPUS": "999"},
    )
    result = _with_child_placement(
        forged, allocation, fence, group, group.members[0], configured,
        selected_slot=selected,
    )
    assert result.env["TPEN_SLOT_ID"] == selected.slot_id
    assert result.env["TPEN_SLOT_NODE"] == selected.node


def test_named_crossed_slot_detector() -> None:
    """A serialized request rejects a placement identity crossed after production."""

    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        group, allocation, lifecycle, configured, fence = _begin(
            tmp_path, task_id="member", facts=_allocation(1).facts
        )
        configured = replace(configured, config="", overrides=(), program=_program(tmp_path))
        selected = _allocation_slot(allocation, 0)
        environment = _with_child_placement(
            _environment(tmp_path), allocation, fence, group, group.members[0], configured,
            selected_slot=selected,
        )
        request = LeafRequest.for_attempt(fence, allocation, configured, environment, selected_slot=selected)
        crossed = PreparedPythonEnvironment(
            request.environment.python,
            request.environment.setup,
            {**request.environment.env, "TPEN_PLACEMENT_SLOT_ID": "crossed-slot"},
        )
        with pytest.raises(ValueError, match="placement identity crosses"):
            replace(request, environment=crossed)
