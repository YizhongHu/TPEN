"""Reviewer regression proposals for B1a allocation-group reconciliation."""

from __future__ import annotations

from dataclasses import replace

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
from tpen_executor.recovery import (
    CoreAdmissionVerdictV2,
    MemberRecoveryInputV2,
    ProfileCheck,
    reconcile_group,
)
from tpen_executor.results import OutcomeReceipt, StartReceipt


def _group(*task_ids: str) -> ReadyGroupV2:
    members = tuple(
        TaskSpec(
            task_id,
            f"lineage-{task_id}",
            ProgramSpecRef({"entrypoint": "opaque", "task": task_id}, digest({"entrypoint": "opaque", "task": task_id})),
            digest({"task": task_id}),
            1,
            "steps",
        )
        for task_id in task_ids
    )
    plan_ref = {"members": list(task_ids), "plan": "b1a-review-r1"}
    return ReadyGroupV2(
        GroupId("group-b1a-review-r1"),
        members,
        ExecutionPlanV2(plan_ref, digest(plan_ref), task_ids, "native-recovery/v2", "preserve-or-park/v1"),
    )


def _successor_verdict(*, quiescence: str = "valid") -> CoreAdmissionVerdictV2:
    profile = ProfileCheck("native-recovery/v2", "supported", frozenset({"model", "optimizer"}), None)
    return CoreAdmissionVerdictV2("successor", "admitted", "valid", quiescence, profile, "review-r1")


def _input(
    group: ReadyGroupV2,
    task_id: str,
    *,
    attempt_id: AttemptId | None = None,
    start_transport: str | None = None,
    outcome_status: str = "failed",
    verdict: CoreAdmissionVerdictV2 | None = None,
) -> MemberRecoveryInputV2:
    receipt_attempt = attempt_id or AttemptId(f"attempt-{task_id}")
    start = StartReceipt(
        task_id,
        receipt_attempt,
        AllocationId(f"allocation-{task_id}"),
        BackendJobId(f"job-{task_id}"),
        group.identity,
        "2026-11-18T00:00:00Z",
        f"/runs/{task_id}",
        start_transport,
    )
    outcome = OutcomeReceipt(
        task_id,
        receipt_attempt,
        group.identity,
        "2026-11-18T00:01:00Z",
        outcome_status,
        start.output_dir,
    )
    return MemberRecoveryInputV2(
        task_id,
        f"lineage-{task_id}",
        attempt_id,
        start,
        outcome,
        verdict,
    )


def test_remote_quiescence_unconfirmed_parks_before_restore() -> None:
    group = _group("successor")
    restored: list[str] = []
    result = reconcile_group(
        group,
        (
            _input(
                group,
                "successor",
                attempt_id=AttemptId("attempt-remote"),
                start_transport="remote",
                outcome_status="remote-quiescence-unconfirmed",
                verdict=_successor_verdict(),
            ),
        ),
        restore_callback=lambda member: restored.append(member.task_id),
    )

    assert result.members[0].action == "park"
    assert restored == []


def test_foreign_input_parks_every_member_without_restore_callbacks() -> None:
    group = _group("first", "second")
    restored: list[str] = []
    result = reconcile_group(
        group,
        (
            _input(group, "first", attempt_id=AttemptId("attempt-first"), verdict=_successor_verdict()),
            _input(group, "foreign", attempt_id=AttemptId("attempt-foreign"), verdict=_successor_verdict()),
        ),
        restore_callback=lambda member: restored.append(member.task_id),
    )

    assert [member.task_id for member in result.members] == ["first", "second"]
    assert [member.action for member in result.members] == ["park", "park"]
    assert restored == []


def test_foreign_input_preserves_group_order_under_input_permutation() -> None:
    group = _group("first", "second", "third")
    inputs = {
        task_id: _input(group, task_id, attempt_id=AttemptId(f"attempt-{task_id}"), verdict=_successor_verdict())
        for task_id in ("first", "second", "third")
    }
    foreign = _input(group, "foreign", attempt_id=AttemptId("attempt-foreign"), verdict=_successor_verdict())

    canonical = reconcile_group(group, (inputs["first"], inputs["second"], inputs["third"], foreign))
    result = reconcile_group(group, (inputs["third"], foreign, inputs["first"], inputs["second"]))

    assert [member.task_id for member in result.members] == ["first", "second", "third"]
    assert all(member.action == "park" for member in result.members)
    assert result.encode() == canonical.encode()


def test_missing_input_attempt_id_is_recovered_from_matching_receipts() -> None:
    group = _group("successor")
    receipt_attempt = AttemptId("attempt-from-receipts")
    member = _input(
        group,
        "successor",
        attempt_id=receipt_attempt,
        verdict=_successor_verdict(),
    )
    member = replace(member, attempt_id=None)
    result = reconcile_group(
        group,
        (member,),
    )

    assert result.members[0].action == "admit_successor"
    assert result.members[0].attempt_id == receipt_attempt


def test_valid_successor_with_supplied_attempt_is_admitted() -> None:
    group = _group("successor")
    supplied_attempt = AttemptId("attempt-supplied")
    result = reconcile_group(
        group,
        (_input(group, "successor", attempt_id=supplied_attempt, verdict=_successor_verdict()),),
    )

    assert result.members[0].action == "admit_successor"
    assert result.members[0].attempt_id == supplied_attempt


def test_missing_or_refused_profile_from_injected_port_parks_without_restore_callback() -> None:
    group = _group("successor")
    profiles = (
        None,
        ProfileCheck(
            "native-recovery/v2",
            "refused",
            frozenset(),
            {"code": "unsupported-callback", "reason": "fake refusal"},
        ),
    )
    for index, profile in enumerate(profiles):
        verdict = CoreAdmissionVerdictV2("successor", "admitted", "valid", "valid", profile, "review-r1")

        class Port:
            def admit(self, member: MemberRecoveryInputV2) -> CoreAdmissionVerdictV2:
                return verdict

        restored: list[str] = []
        result = reconcile_group(
            group,
            (_input(group, "successor", attempt_id=AttemptId(f"attempt-profile-{index}")),),
            admission_port=Port(),
            restore_callback=lambda member: restored.append(member.task_id),
        )

        assert result.members[0].action == "park"
        assert restored == []
