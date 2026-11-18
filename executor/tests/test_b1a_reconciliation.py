"""Pure fake-driven B1a allocation-group reconciliation tests."""

from __future__ import annotations

from dataclasses import dataclass

from tpen_executor._codec import digest
from tpen_executor.contracts import GroupId, ProgramSpecRef, ReadyGroupV2, TaskSpec, ExecutionPlanV2, AttemptId, AllocationId, BackendJobId
from tpen_executor.recovery import (
    CoreAdmissionVerdictV2,
    GroupReconciliationV2,
    MemberRecoveryInputV2,
    ProfileCheck,
    reconcile_group,
)
from tpen_executor.results import OutcomeReceipt, StartReceipt


def _task(task_id: str) -> TaskSpec:
    config = {"entrypoint": "opaque", "task": task_id}
    return TaskSpec(task_id, f"lineage-{task_id}", ProgramSpecRef(config, digest(config)), digest({"task": task_id}), 1, "steps")


def _group(task_ids: tuple[str, ...] = ("completed", "successor", "fresh", "terminal", "ambiguous")) -> ReadyGroupV2:
    members = tuple(_task(task_id) for task_id in task_ids)
    plan_ref = {"members": list(task_ids), "plan": "b1a"}
    plan = ExecutionPlanV2(plan_ref, digest(plan_ref), task_ids, "native-recovery/v2", "preserve-or-park/v1")
    return ReadyGroupV2(GroupId("group-b1a"), members, plan)


def _profile(verdict: str = "supported") -> ProfileCheck:
    return ProfileCheck(
        "native-recovery/v2",
        verdict,
        frozenset({"model", "optimizer"}),
        None if verdict == "supported" else {"code": "unsupported-callback", "reason": "fake refusal"},
    )


def _core(outcome: str, *, profile: ProfileCheck | None = None, admission: str = "unknown", checkpoint: str = "absent", quiescence: str = "unknown") -> CoreAdmissionVerdictV2:
    return CoreAdmissionVerdictV2(outcome, admission, checkpoint, quiescence, profile, f"fake-{outcome}")


def _receipt(group: ReadyGroupV2, task_id: str, status: str = "failed") -> tuple[StartReceipt, OutcomeReceipt]:
    attempt = AttemptId(f"attempt-{task_id}")
    allocation = AllocationId("allocation-1")
    backend_job = BackendJobId("job-1")
    start = StartReceipt(task_id, attempt, allocation, backend_job, group.identity, "2026-11-18T00:00:00Z", f"/runs/{task_id}")
    outcome = OutcomeReceipt(task_id, attempt, group.identity, "2026-11-18T00:01:00Z", status, start.output_dir)
    return start, outcome


def _input(group: ReadyGroupV2, task_id: str, verdict: CoreAdmissionVerdictV2 | None) -> MemberRecoveryInputV2:
    start, outcome = _receipt(group, task_id)
    return MemberRecoveryInputV2(task_id, f"lineage-{task_id}", start.attempt_id, start, outcome, verdict)


def test_mixed_group_emits_all_five_actions_and_park_dominates() -> None:
    group = _group()
    inputs = (
        _input(group, "completed", _core("completed")),
        _input(group, "successor", _core("successor", profile=_profile(), admission="admitted", checkpoint="valid", quiescence="valid")),
        MemberRecoveryInputV2("fresh", "lineage-fresh"),
        _input(group, "terminal", _core("terminal")),
        _input(group, "ambiguous", _core("unknown")),
    )
    result = reconcile_group(group, inputs)
    assert [member.action for member in result.members] == [
        "preserve_completed", "admit_successor", "start_fresh", "preserve_terminal", "park"
    ]
    assert result.aggregate == "parked"
    assert result.completed_count == 1
    assert result.terminal_count == 1
    assert GroupReconciliationV2.from_mapping(result.to_dict()) == result


def test_without_ambiguity_aggregate_requires_replacement() -> None:
    group = _group(("completed", "successor", "fresh", "terminal"))
    inputs = (
        _input(group, "completed", _core("completed")),
        _input(group, "successor", _core("successor", profile=_profile(), admission="admitted", checkpoint="valid", quiescence="valid")),
        MemberRecoveryInputV2("fresh", "lineage-fresh"),
        _input(group, "terminal", _core("terminal")),
    )
    result = reconcile_group(group, inputs)
    assert result.aggregate == "replacement_required"


def test_unsupported_profile_is_parked_before_restore_callback() -> None:
    group = _group(("successor",))
    called: list[str] = []
    result = reconcile_group(
        group,
        (_input(group, "successor", _core("successor", profile=None, admission="admitted", checkpoint="valid", quiescence="valid")),),
        restore_callback=lambda member: called.append(member.task_id),
    )
    assert result.members[0].action == "park"
    assert result.members[0].reason_code == "unsupported-profile"
    assert called == []


def test_start_receipt_never_becomes_start_fresh_and_completed_is_not_relaunched() -> None:
    group = _group(("started", "completed"))
    called: list[str] = []
    started = _input(group, "started", _core("unknown"))
    completed = _input(group, "completed", _core("completed"))
    result = reconcile_group(group, (started, completed), restore_callback=lambda member: called.append(member.task_id))
    assert result.members[0].action == "park"
    assert result.members[0].reason_code == "unknown-core-result"
    assert result.members[1].action == "preserve_completed"
    assert called == []


def test_injected_port_is_used_and_foreign_or_missing_evidence_parks() -> None:
    group = _group(("successor", "missing"))
    @dataclass
    class RefusingPort:
        calls: int = 0

        def admit(self, member: MemberRecoveryInputV2) -> CoreAdmissionVerdictV2:
            self.calls += 1
            return _core("successor", profile=_profile("refused"), admission="admitted", checkpoint="valid", quiescence="valid")

    port = RefusingPort()
    result = reconcile_group(group, (_input(group, "successor", None),), admission_port=port)
    assert port.calls == 1
    assert result.members[0].reason_code == "unsupported-profile"
    assert result.members[1].reason_code == "missing-member-evidence"


def test_crossed_group_receipt_is_parked() -> None:
    group = _group(("completed",))
    attempt = AttemptId("attempt-foreign")
    start = StartReceipt("completed", attempt, AllocationId("allocation-1"), BackendJobId("job-1"), "a" * 64, "2026-11-18T00:00:00Z", "/runs/foreign")
    outcome = OutcomeReceipt("completed", attempt, "a" * 64, "2026-11-18T00:01:00Z", "completed", "/runs/foreign")
    result = reconcile_group(
        group,
        (MemberRecoveryInputV2("completed", "lineage-completed", attempt, start, outcome, _core("completed")),),
    )
    assert result.members[0].action == "park"
    assert result.members[0].reason_code == "crossed-group-receipt"

