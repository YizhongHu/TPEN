"""B1a v2 contract and identity conformance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.allocations import AllocationAttemptV2
from tpen_executor.contracts import (
    AllocationAttemptId,
    AllocationId,
    AttemptId,
    BackendJobId,
    ExecutionPlanV2,
    GroupId,
    InvocationId,
    ProgramSpecRef,
    ReadyGroup,
    ReadyGroupV2,
    TaskSpec,
)


def _task(task_id: str, lineage_id: str | None = None) -> TaskSpec:
    config = {"entrypoint": "opaque", "task": task_id}
    program = ProgramSpecRef(config, digest(config))
    return TaskSpec(task_id, lineage_id or f"lineage-{task_id}", program, digest({"task": task_id}), 1, "steps")


def _group() -> ReadyGroupV2:
    members = (_task("task-a"), _task("task-b"))
    plan_ref = {"members": [member.task_id for member in members], "mode": "attached"}
    plan = ExecutionPlanV2(
        plan_ref,
        digest(plan_ref),
        tuple(member.task_id for member in members),
        "native-recovery/v2",
        "preserve-or-park/v1",
    )
    return ReadyGroupV2(GroupId("group-v2"), members, plan)


def test_v2_contract_round_trips_as_canonical_bytes() -> None:
    group = _group()
    assert ReadyGroupV2.decode(group.encode()) == group
    assert group.encode() == canonical_bytes(group.to_dict())
    assert group.identity == digest(group.to_dict())


def test_v2_decoder_refuses_v1_unknown_missing_and_extra_shapes() -> None:
    group = _group()
    legacy = ReadyGroup("legacy", group.members, digest({"legacy": True}))
    with pytest.raises((TypeError, ValueError)):
        ReadyGroupV2.from_mapping(legacy.to_dict())
    for mutation in ("schema", "extra"):
        mapping = group.to_dict()
        if mutation == "schema":
            mapping["schema"] = "tpen-executor/ready-group-v1"
        else:
            mapping["unexpected"] = True
        with pytest.raises((TypeError, ValueError)):
            ReadyGroupV2.from_mapping(mapping)
    missing = group.to_dict()
    missing.pop("execution_plan")
    with pytest.raises((TypeError, ValueError)):
        ReadyGroupV2.from_mapping(missing)


def test_v2_identities_are_nominally_distinct() -> None:
    assert GroupId("same") != AllocationAttemptId("same")
    assert AllocationAttemptId("same") != InvocationId("same")
    with pytest.raises(TypeError):
        AllocationAttemptV2(
            GroupId("g"),
            AllocationAttemptId("aa"),
            AllocationAttemptId("wrong"),  # type: ignore[arg-type]
            BackendJobId("job"),
            "generation-1",
            "2026-11-18T00:00:00Z",
        )


def test_allocation_attempt_v2_binds_all_operational_identities() -> None:
    record = AllocationAttemptV2(
        GroupId("g"),
        AllocationAttemptId("aa"),
        AllocationId("allocation"),
        BackendJobId("job"),
        "generation-1",
        "2026-11-18T00:00:00Z",
    )
    assert AllocationAttemptV2.decode(record.encode()) == record
    for key in ("group_id", "allocation_attempt_id", "allocation_id", "backend_job_id"):
        mapping = record.to_dict()
        mapping[key] = {"schema": "tpen-executor/attempt-id-v1", "value": "same"}
        with pytest.raises((TypeError, ValueError)):
            AllocationAttemptV2.from_mapping(mapping)

