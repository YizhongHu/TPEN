"""S2-L3 generic group-runtime composition tests."""

from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation, ControllerGenerationV1
from tpen_executor.contracts import AllocationId, BackendJobId, ExecutionPlanV2, GroupId, ProgramSpecRef, ReadyGroupV2, TaskSpec
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle.group_runtime import GroupRuntimeDispatcher, PackedGroupRequest, PreparedGroupInput
from tpen_executor.recovery import MemberRecoveryInputV2


class _Materialized:
    def __init__(self, slots):
        self.slots = slots
        for key, value in slots.items():
            setattr(self, key, value)

    def __contains__(self, key):
        return key in self.slots


class _Program:
    def __init__(self, materialized):
        self.materialized = materialized
        self.envs = []

    def materialize(self, *, env):
        self.envs.append(dict(env))
        return self.materialized


def _prepared(tmp_path: Path, *, allocation_program=None, worker_program=None):
    config = {"program": "fixture"}
    task = TaskSpec("task", "lineage", ProgramSpecRef(config, digest(config)), "a" * 64, 1, "steps")
    plan = ExecutionPlanV2(config, digest(config), ("task",), "profile", "protocol")
    group = ReadyGroupV2(GroupId("group"), (task,), plan)
    allocation = ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})
    controller = ControllerGenerationV1(group.group_id, 1, allocation.allocation_id, allocation.backend_job_id, "b" * 64, "nonce", None)
    env = {"paths": {"control_python": "/prepared/control", "science_python": "/prepared/science"}}
    allocation_program = allocation_program or _Program(_Materialized({"dispatcher": _Materialized({"run": lambda: None})}))
    worker_program = worker_program or _Program(_Materialized({"environment": PreparedPythonEnvironment("/prepared/science"), "process": object(), "runner": object(), "evidence": object()}))
    return PreparedGroupInput(
        group, allocation, controller, (MemberRecoveryInputV2("task", "lineage"),),
        allocation_program, worker_program, env, str(tmp_path / "store"), "c" * 64,
        "deployment", "d" * 64, "claimant", "2026-11-26T00:00:00Z",
    )


def test_materialize_forwards_explicit_env_to_both_owner_processes(tmp_path):
    allocation = _Program(_Materialized({"dispatcher": _Materialized({"run": lambda: None})}))
    worker = _Program(_Materialized({"environment": PreparedPythonEnvironment("/prepared/science"), "process": object(), "runner": object(), "evidence": object()}))
    prepared = _prepared(tmp_path, allocation_program=allocation, worker_program=worker)
    prepared.materialize()
    assert allocation.envs == [prepared.env]
    assert worker.envs == [prepared.env]


def test_packed_request_is_immutable_and_rejects_partial_members(tmp_path):
    prepared = _prepared(tmp_path)
    request = PackedGroupRequest(prepared)
    with pytest.raises((AttributeError, TypeError)):
        request.reentry = True
    with pytest.raises(ValueError):
        PreparedGroupInput(
            prepared.group, prepared.allocation, prepared.controller, (), prepared.allocation_program,
            prepared.worker_program, prepared.env, prepared.store_root, prepared.execution_bundle_digest,
            prepared.deployment_lineage, prepared.registration_receipt_digest, prepared.claimant_nonce,
            prepared.observed_at_utc,
        )


def test_dispatcher_has_finite_deadlines():
    dispatcher = GroupRuntimeDispatcher(timeout=3.0, core_timeout=2.0)
    assert dispatcher.timeout == 3.0
    assert dispatcher.core_timeout == 2.0
    with pytest.raises(ValueError):
        GroupRuntimeDispatcher(timeout=0)
