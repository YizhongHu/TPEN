"""Conformance tests for immutable executor admission contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.contracts import (
    AllocationId,
    AttemptId,
    BackendJobId,
    ProgramSpecRef,
    ReadyGroup,
    TaskSpec,
)
from tpen_executor.results import StartReceipt


FIXTURES = Path(__file__).parent / "fixtures"


def _fixture_bytes(name: str) -> bytes:
    """Read canonical JSON text while ignoring the file's terminal newline."""

    return (FIXTURES / f"{name}.json").read_text(encoding="utf-8").rstrip("\n").encode("utf-8")


def _program() -> ProgramSpecRef:
    config = {"entrypoint": "tpen.run", "options": {"mode": "attached", "workers": 2}}
    return ProgramSpecRef(config, digest(config))


def _tasks() -> tuple[TaskSpec, TaskSpec]:
    program = _program()
    return (
        TaskSpec("task-a", "lineage-a", program, digest({"seed": 1, "system": "he"}), 100, "steps"),
        TaskSpec("task-b", "lineage-b", program, digest({"seed": 2, "system": "he"}), 250, "steps"),
    )


def _group() -> ReadyGroup:
    return ReadyGroup("group-1", _tasks(), digest({"members": ["task-a", "task-b"], "backend": "local"}))


@pytest.mark.parametrize(
    ("name", "record", "decoder"),
    [
        ("program-spec-ref", _program(), ProgramSpecRef),
        ("task-spec", _tasks()[0], TaskSpec),
        ("ready-group", _group(), ReadyGroup),
    ],
)
def test_contract_round_trip_and_golden_bytes(name: str, record: object, decoder: type) -> None:
    """Every contract has a stable mapping and canonical JSON representation."""

    mapping = record.to_dict()
    expected = _fixture_bytes(name)
    assert decoder.from_mapping(mapping) == record
    assert record.encode() == expected
    assert decoder.decode(record.encode()) == record
    assert canonical_bytes(mapping) == expected


@pytest.mark.parametrize("identity_cls", [AttemptId, AllocationId, BackendJobId])
def test_identity_round_trip_and_golden_bytes(identity_cls: type) -> None:
    """Identity classes have distinct strict wire schemas."""

    identity = identity_cls({AttemptId: "attempt-1", AllocationId: "allocation-1", BackendJobId: "job-1"}[identity_cls])
    assert identity_cls.from_mapping(identity.to_dict()) == identity
    name = {AttemptId: "attempt-id", AllocationId: "allocation-id", BackendJobId: "backend-job-id"}[identity_cls]
    expected = _fixture_bytes(name)
    assert identity.encode() == expected
    assert identity_cls.decode(identity.encode()) == identity
    assert canonical_bytes(identity.to_dict()) == expected


@pytest.mark.parametrize("identity_cls", [AttemptId, AllocationId, BackendJobId])
def test_identity_decoders_are_strict(identity_cls: type) -> None:
    """Identity wire records reject unknown schema, missing, extra, and wrong values."""

    identity = identity_cls("identity")
    cases = []
    wrong_schema = identity.to_dict(); wrong_schema["schema"] = "unknown"; cases.append(wrong_schema)
    missing = identity.to_dict(); missing.pop("value"); cases.append(missing)
    extra = identity.to_dict(); extra["unexpected"] = True; cases.append(extra)
    wrong_type = identity.to_dict(); wrong_type["value"] = 7; cases.append(wrong_type)
    for case in cases:
        with pytest.raises((ValueError, TypeError)):
            identity_cls.from_mapping(case)


@pytest.mark.parametrize("identity_cls", [AttemptId, AllocationId, BackendJobId])
def test_identities_are_frozen(identity_cls: type) -> None:
    """Identity values cannot be reassigned."""

    identity = identity_cls("identity")
    with pytest.raises((AttributeError, TypeError)):
        identity.value = "changed"


@pytest.mark.parametrize("record", [_program(), _tasks()[0], _group()])
def test_contracts_are_frozen_and_nested_sequences_are_immutable(record: object) -> None:
    """Frozen records expose tuples and reject attribute mutation."""

    with pytest.raises((AttributeError, TypeError)):
        record.task_id = "changed"  # type: ignore[attr-defined]
    if isinstance(record, ReadyGroup):
        assert isinstance(record.members, tuple)
        with pytest.raises((AttributeError, TypeError)):
            record.members[0] = record.members[0]  # type: ignore[index]
    if isinstance(record, ProgramSpecRef):
        with pytest.raises((TypeError, AttributeError)):
            record.config_ref["entrypoint"] = "changed"


@pytest.mark.parametrize("decoder, record", [(ProgramSpecRef, _program()), (TaskSpec, _tasks()[0]), (ReadyGroup, _group())])
def test_contract_decoders_are_closed_and_typed(decoder: type, record: object) -> None:
    """Unknown schema, missing/extra keys, and wrong scalar types are refused."""

    original = record.to_dict()
    cases = []
    wrong = dict(original)
    wrong["schema"] = 7
    cases.append(wrong)
    missing = dict(original)
    missing.pop("schema")
    cases.append(missing)
    extra = dict(original)
    extra["unexpected"] = True
    cases.append(extra)
    wrong_type = dict(original)
    key = next(key for key in original if key != "schema" and isinstance(original[key], str))
    wrong_type[key] = 7
    cases.append(wrong_type)
    for case in cases:
        with pytest.raises((ValueError, TypeError)):
            decoder.from_mapping(case)


def test_group_digest_changes_for_member_changes_and_order() -> None:
    """Member content and member order both affect the canonical group digest."""

    first, second = _tasks()
    original = ReadyGroup("group-1", (first, second), digest({"members": ["task-a", "task-b"], "backend": "local"}))
    changed = TaskSpec(first.task_id, first.lineage_id, first.program, digest({"seed": 99}), first.target_horizon, first.target_horizon_units)
    changed_group = ReadyGroup("group-1", (changed, second), original.execution_plan_digest)
    reversed_group = ReadyGroup("group-1", (second, first), original.execution_plan_digest)
    assert digest(changed_group.to_dict()) != digest(original.to_dict())
    assert digest(reversed_group.to_dict()) != digest(original.to_dict())


def test_group_accepts_heterogeneous_science_digests_and_horizons() -> None:
    """A group does not impose the legacy single-science-digest constraint."""

    first, second = _tasks()
    group = ReadyGroup("group-1", (first, second), digest({"members": ["task-a", "task-b"]}))
    restored = ReadyGroup.from_mapping(group.to_dict())
    assert [member.science_digest for member in restored.members] == [first.science_digest, second.science_digest]
    assert [member.target_horizon for member in restored.members] == [100, 250]


def test_identity_types_are_not_interchangeable() -> None:
    """Attempt, allocation, and backend-job identities remain nominally distinct."""

    with pytest.raises(TypeError):
        StartReceipt("task-a", AllocationId("wrong"), AllocationId("allocation-1"), BackendJobId("job-1"), digest({"group": 1}), "now", "/out")
    with pytest.raises(TypeError):
        StartReceipt("task-a", AttemptId("attempt-1"), AttemptId("wrong"), BackendJobId("job-1"), digest({"group": 1}), "now", "/out")
