"""Independent R1 contract tests for executor admission records."""

from __future__ import annotations

import operator
from dataclasses import replace

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
    "mutator",
    [
        lambda mapping: operator.ior(mapping, {"injected": True}),
        lambda mapping: dict.__setitem__(mapping, "injected", True),
        lambda mapping: dict.update(mapping, injected=True),
        lambda mapping: dict.__init__(mapping, {"injected": True}),
    ],
)
def test_frozen_mapping_rejects_dict_level_mutation(mutator) -> None:
    """R1-C1 (F4) — expected RED at 777700d6: frozen mappings reject every dict-level mutation."""

    program = _program()
    with pytest.raises(TypeError):
        mutator(program.config_ref)
    assert "injected" not in program.config_ref
    assert digest(program.config_ref) == program.content_digest


@pytest.mark.parametrize("identity_cls", [AttemptId, AllocationId, BackendJobId])
def test_identity_instances_reject_new_attributes(identity_cls: type) -> None:
    """R1-C2 (F6) — expected RED at 777700d6: identity instances have no extension attributes."""

    identity = identity_cls("identity")
    with pytest.raises((AttributeError, TypeError)):
        identity.extra = 1


def test_ready_group_exposes_canonical_identity() -> None:
    """R1-C3 (F3) — expected RED at 777700d6: ReadyGroup exposes its canonical identity."""

    group = _group()
    assert group.identity == digest(group.to_dict())
    first, second = _tasks()
    reversed_group = ReadyGroup(group.group_id, (second, first), group.execution_plan_digest)
    assert reversed_group.identity != group.identity
    assert isinstance(group.identity, str)
    assert len(group.identity) == 64
    assert group.identity == group.identity.lower()


def test_canonical_bytes_are_order_and_container_independent() -> None:
    """R1-C5 — expected GREEN: canonical JSON normalizes mapping order and sequence containers."""

    assert canonical_bytes({"b": 1, "a": [1, 2]}) == canonical_bytes({"a": (1, 2), "b": 1}) == b'{"a":[1,2],"b":1}'


def test_canonical_bytes_distinguish_scalar_types() -> None:
    """R1-C6 — expected GREEN: canonical digests distinguish JSON scalar types."""

    assert digest(1) != digest(True)
    assert digest(1) != digest(1.0)
    assert digest("1") != digest(1)
    assert digest(None) != digest(0)


def test_canonical_bytes_preserve_unicode_without_escaping() -> None:
    """R1-C7 — expected GREEN: canonical JSON preserves UTF-8 Unicode characters."""

    assert canonical_bytes({"k": "é"}) == '{"k":"é"}'.encode("utf-8")


@pytest.mark.parametrize(
    ("value", "exc"),
    [
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        ({"k": float("-inf")}, ValueError),
        ({1: "x"}, TypeError),
        (b"bytes", TypeError),
        (object(), TypeError),
        ({"k": {1, 2}}, TypeError),
    ],
)
def test_canonical_bytes_reject_non_json_shapes(value, exc: type[Exception]) -> None:
    """R1-C8 — expected GREEN: canonical JSON rejects non-finite and non-JSON values."""

    with pytest.raises(exc):
        canonical_bytes(value)


def test_program_spec_ref_rejects_mismatched_or_malformed_digest() -> None:
    """R1-C9 — expected GREEN: ProgramSpecRef validates exact lowercase content digests."""

    cfg = {"entrypoint": "tpen.run"}
    with pytest.raises(ValueError):
        ProgramSpecRef(cfg, digest({"other": 1}))
    with pytest.raises(ValueError):
        ProgramSpecRef(cfg, "abc")
    with pytest.raises(TypeError):
        ProgramSpecRef(cfg, 7)
    with pytest.raises(ValueError):
        ProgramSpecRef(cfg, digest(cfg).upper())
    config = ["a", {"b": 1}]
    program = ProgramSpecRef(config, digest(config))
    assert isinstance(program.config_ref, tuple)
    assert program.config is program.config_ref


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_horizon": -1},
        {"target_horizon": True},
        {"target_horizon": "100"},
        {"task_id": ""},
        {"program": _program().to_dict()},
        {"science_digest": digest({}).upper()},
        {"target_horizon_units": ""},
    ],
)
def test_task_spec_rejects_invalid_scalars(overrides: dict[str, object]) -> None:
    """R1-C10 — expected GREEN: TaskSpec rejects invalid scalar and nested-record values."""

    task = _tasks()[0]
    with pytest.raises((TypeError, ValueError)):
        replace(task, **overrides)


def test_ready_group_rejects_empty_duplicate_and_foreign_members() -> None:
    """R1-C11 — expected GREEN: ReadyGroup requires distinct TaskSpec members and accepts lists."""

    first, second = _tasks()
    digest_value = digest({"members": ["task-a", "task-b"]})
    with pytest.raises(TypeError):
        ReadyGroup("g", (), digest_value)
    with pytest.raises(ValueError):
        ReadyGroup("g", (first, first), digest_value)
    with pytest.raises(TypeError):
        ReadyGroup("g", (first, first.to_dict()), digest_value)
    group = ReadyGroup("g", [first, second], digest_value)
    assert isinstance(group.members, tuple)


def test_ready_group_bytes_round_trip() -> None:
    """R1-C12 — expected GREEN: ReadyGroup decodes bytes, text, and mappings identically."""

    group = _group()
    assert ReadyGroup.decode(group.encode()) == group
    assert ReadyGroup.decode(group.encode().decode("utf-8")) == group
    assert ReadyGroup.decode(group.to_dict()) == group
    assert group.encode() == canonical_bytes(group.to_dict())


def test_ready_group_decode_rejects_non_object_and_non_finite() -> None:
    """R1-C13 — expected GREEN: ReadyGroup decoding rejects non-objects and non-finite JSON."""

    group = _group()
    with pytest.raises(ValueError):
        ReadyGroup.decode(b"[]")
    with pytest.raises(ValueError):
        ReadyGroup.decode(b'"x"')
    malformed = group.encode().replace(b'"workers":2', b'"workers":NaN')
    assert malformed != group.encode()
    with pytest.raises(ValueError):
        ReadyGroup.decode(malformed)


def test_identity_types_are_nominal_on_wire_and_in_memory() -> None:
    """R1-C14 — expected GREEN: identity classes remain nominally distinct in memory and on wire."""

    assert AttemptId("x") != AllocationId("x")
    with pytest.raises(ValueError):
        AttemptId.from_mapping(AllocationId("x").to_dict())
    assert str(AttemptId("x")) == "x"
    with pytest.raises(TypeError):
        AttemptId("")
    with pytest.raises(TypeError):
        AttemptId(7)
    assert len({AttemptId("x"), AllocationId("x"), BackendJobId("x")}) == 3
