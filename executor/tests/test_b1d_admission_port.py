"""B1d behavior tests for the executor-owned core admission boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import (
    AllocationId,
    AttemptId,
    BackendJobId,
    ExecutionPlanV2,
    GroupId,
    ProgramSpecRef,
    ReadyGroupV2,
    TaskSpec,
)
from tpen_executor.recovery import MappingCoreAdmissionPort, MemberRecoveryInputV2, ProfileCheck, reconcile_group
from tpen_executor.results import OutcomeReceipt, StartReceipt


def _group(*task_ids: str) -> ReadyGroupV2:
    members = tuple(
        _task(task_id)
        for task_id in task_ids
    )
    plan_ref = {"members": list(task_ids), "plan": "b1d-exec"}
    plan = ExecutionPlanV2(plan_ref, digest(plan_ref), task_ids, "native-recovery/v2", "preserve-or-park/v1")
    return ReadyGroupV2(GroupId("group-b1d"), members, plan)


def _task(task_id: str) -> TaskSpec:
    config_ref = {"entrypoint": "opaque", "task": task_id}
    return TaskSpec(
        task_id,
        f"lineage-{task_id}",
        ProgramSpecRef(config_ref, digest(config_ref)),
        digest({"task": task_id}),
        1,
        "steps",
    )


def _member(group: ReadyGroupV2, task_id: str, *, status: str = "failed") -> MemberRecoveryInputV2:
    attempt = AttemptId(f"attempt-{task_id}")
    start = StartReceipt(
        task_id,
        attempt,
        AllocationId(f"allocation-{task_id}"),
        BackendJobId(f"job-{task_id}"),
        group.identity,
        "2026-11-20T00:00:00Z",
        f"/runs/{task_id}",
    )
    outcome = OutcomeReceipt(task_id, attempt, group.identity, "2026-11-20T00:01:00Z", status, start.output_dir)
    return MemberRecoveryInputV2(task_id, f"lineage-{task_id}", attempt, start, outcome)


def _profile(*, verdict: str = "supported") -> dict[str, object]:
    return ProfileCheck(
        "native-recovery/v2",
        verdict,
        frozenset({"model", "optimizer"}),
        None if verdict == "supported" else {"code": "unsupported-callback", "reason": "refused"},
    ).to_dict()


def _wire(
    *,
    outcome: str = "successor",
    admission: str = "admitted",
    checkpoint: str = "valid",
    quiescence: str = "unknown",
    profile: dict[str, object] | None = None,
    reason: str = "wire-result",
) -> dict[str, object]:
    return {
        "schema": "tpen-core-admission-v1",
        "outcome": outcome,
        "admission": admission,
        "checkpoint": checkpoint,
        "quiescence": quiescence,
        "profile": _profile() if profile is None and outcome == "successor" else profile,
        "reason": reason,
    }


def _result(group: ReadyGroupV2, payload: object, *, status: str = "failed"):
    member = _member(group, "task", status=status)
    port = MappingCoreAdmissionPort(lambda _member: payload)  # type: ignore[arg-type]
    return reconcile_group(group, (member,), admission_port=port).members[0]


@pytest.mark.parametrize(
    ("wire", "status", "action", "reason"),
    [
        (_wire(outcome="completed", admission="unknown", checkpoint="absent", profile=None), "completed", "preserve_completed", "completed-and-settled"),
        (_wire(outcome="terminal", admission="refused", checkpoint="absent", profile=None), "failed", "preserve_terminal", "core-terminal"),
        (_wire(outcome="terminal", admission="unknown", checkpoint="unknown", profile=None), "failed", "preserve_terminal", "core-terminal"),
        (_wire(outcome="unknown", admission="unknown", checkpoint="unknown", profile=None), "failed", "park", "unknown-core-result"),
        (_wire(), "failed", "admit_successor", "valid-successor-admission"),
        (_wire(admission="refused", checkpoint="invalid"), "failed", "park", "core-admission-refused"),
        (_wire(profile=_profile(verdict="refused")), "failed", "park", "unsupported-profile"),
    ],
)
def test_wire_verdict_classes_reconcile_through_group(wire: dict[str, object], status: str, action: str, reason: str) -> None:
    """Every positive and negative verdict class has the B1a group outcome."""

    member = _member(_group("task"), "task", status=status)
    result = reconcile_group(
        _group("task"),
        (member,),
        admission_port=MappingCoreAdmissionPort(lambda _member: wire),
    ).members[0]
    assert (result.action, result.reason_code) == (action, reason)


@pytest.mark.parametrize("mutation", ["unknown-schema", "extra-field", "missing-field", "non-string"])
def test_malformed_wire_mapping_parks_core_admission_unavailable(mutation: str) -> None:
    """Unknown, open, incomplete, and wrongly typed mappings never admit."""

    wire = _wire()
    if mutation == "unknown-schema":
        wire["schema"] = "unknown"
    elif mutation == "extra-field":
        wire["unexpected"] = "reject"
    elif mutation == "missing-field":
        del wire["reason"]
    else:
        wire["admission"] = 1
    result = _result(_group("task"), wire)
    assert (result.action, result.reason_code) == ("park", "core-admission-unavailable")


def test_core_quiescence_must_be_unknown_and_is_filled_from_settled_outcome() -> None:
    """Core cannot self-certify quiescence; a settled receipt supplies it."""

    malformed = _result(_group("task"), _wire(quiescence="valid"))
    assert (malformed.action, malformed.reason_code) == ("park", "core-admission-unavailable")
    admitted = _result(_group("task"), _wire(), status="failed")
    assert (admitted.action, admitted.reason_code) == ("admit_successor", "valid-successor-admission")
    remote_group = _group("task")
    remote_member = _member(remote_group, "task", status="remote-quiescence-unconfirmed")
    decoded = MappingCoreAdmissionPort(lambda _member: _wire()).admit(remote_member)
    assert decoded is not None and decoded.quiescence == "unknown"


@pytest.mark.parametrize("error", [OSError("os"), KeyError("key"), RuntimeError("runtime")])
def test_evaluator_exceptions_are_unavailable(error: Exception) -> None:
    """Any ordinary evaluator exception is fail-closed and never admits."""

    def evaluate(_member: MemberRecoveryInputV2) -> object:
        raise error

    group = _group("task")
    result = reconcile_group(
        group,
        (_member(group, "task"),),
        admission_port=MappingCoreAdmissionPort(evaluate),  # type: ignore[arg-type]
    ).members[0]
    assert (result.action, result.reason_code) == ("park", "core-admission-unavailable")


def test_none_evidence_is_unavailable_and_never_admitted() -> None:
    """An absent core mapping is different from a terminal core verdict."""

    result = _result(_group("task"), None)
    assert (result.action, result.reason_code) == ("park", "core-admission-unavailable")


def test_remote_quiescence_unconfirmed_never_reaches_port() -> None:
    """Unconfirmed remote teardown is parked before the port is invoked."""

    calls: list[str] = []
    group = _group("task")
    result = reconcile_group(
        group,
        (_member(group, "task", status="remote-quiescence-unconfirmed"),),
        admission_port=MappingCoreAdmissionPort(lambda member: calls.append(member.task_id) or _wire()),
    ).members[0]
    assert (result.action, result.reason_code) == ("park", "remote-quiescence-unconfirmed")
    assert calls == []


def test_evaluator_called_exactly_once_per_member() -> None:
    """The in-process evaluator has one call for each eligible group member."""

    group = _group("first", "second", "third")
    calls: list[str] = []

    def evaluate(member: MemberRecoveryInputV2) -> dict[str, object]:
        calls.append(member.task_id)
        return _wire(outcome="terminal", admission="unknown", checkpoint="absent", profile=None)

    result = reconcile_group(
        group,
        tuple(_member(group, task_id) for task_id in ("first", "second", "third")),
        admission_port=MappingCoreAdmissionPort(evaluate),
    )
    assert [member.action for member in result.members] == ["preserve_terminal"] * 3
    assert calls == ["first", "second", "third"]


def test_mutant_table_anchors_are_unique_and_killers_exist() -> None:
    """Every reviewed mutation has one source anchor and a named behavior killer."""

    source = Path(__file__).parents[1] / "src" / "tpen_executor" / "recovery.py"
    names: set[str] = set()
    anchors: set[tuple[str, str]] = set()
    for name, filename, old, _new, killer in MUTANTS:
        assert name not in names
        names.add(name)
        assert (filename, old) not in anchors
        anchors.add((filename, old))
        assert (source.parent / filename).read_text(encoding="utf-8").count(old) == 1
        assert any(killer in path.read_text(encoding="utf-8") for path in Path(__file__).parent.glob("test_*.py"))


MUTANTS = (
    (
        "accept-any-schema",
        "recovery.py",
        'if type(payload["schema"]) is not str or payload["schema"] != CORE_ADMISSION_WIRE_SCHEMA:',
        'if False:',
        "test_malformed_wire_mapping_parks_core_admission_unavailable",
    ),
    (
        "skip-extra-field-check",
        "recovery.py",
        'if set(payload) != _CORE_ADMISSION_WIRE_FIELDS:',
        'if False:',
        "test_malformed_wire_mapping_parks_core_admission_unavailable",
    ),
    (
        "pass-through-core-quiescence",
        "recovery.py",
        'quiescence = (\n                "valid"\n                if member.outcome_receipt is not None\n                and member.outcome_receipt.status not in NON_SETTLED_OPERATIONAL_OUTCOMES\n                else "unknown"\n            )\n            if payload["quiescence"] != "unknown":\n                raise ValueError("core admission quiescence must be unknown")',
        'quiescence = payload["quiescence"]',
        "test_core_quiescence_must_be_unknown_and_is_filled_from_settled_outcome",
    ),
    (
        "quiescence-valid-for-non-settled",
        "recovery.py",
        'quiescence = (\n                "valid"\n                if member.outcome_receipt is not None\n                and member.outcome_receipt.status not in NON_SETTLED_OPERATIONAL_OUTCOMES\n                else "unknown"\n            )',
        'quiescence = "valid"',
        "test_core_quiescence_must_be_unknown_and_is_filled_from_settled_outcome",
    ),
    (
        "catch-only-runtime-error",
        "recovery.py",
        'except Exception:\n            # Any evaluator or decoder failure is unavailable evidence.\n            return None  # unavailable',
        'except RuntimeError:\n            # Any evaluator or decoder failure is unavailable evidence.\n            return None  # unavailable',
        "test_evaluator_exceptions_are_unavailable",
    ),
    (
        "admit-on-exception",
        "recovery.py",
        'return None  # unavailable',
        'return CoreAdmissionVerdictV2("successor", "admitted", "valid", "valid", ProfileCheck("native-recovery/v2", "supported", frozenset({"model"}), None), "exception")',
        "test_evaluator_exceptions_are_unavailable",
    ),
)
