"""B1d integration coverage for the real core producer and executor decoder."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen.execution.core_admission import evaluate_core_admission
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
from tpen_executor.lifecycle.controller import (
    GroupControllerFactsV1,
    MemberLaunchResult,
    run_group_controller,
)
from tpen_executor.recovery import (
    CoreAdmissionUnavailable,
    MappingCoreAdmissionPort,
    MemberRecoveryInputV2,
    reconcile_group,
)
from tpen_executor.results import OutcomeReceipt, StartReceipt

from tests.helpers.core_admission import (
    admission_context,
    checkpoint,
    planned_yield_run,
    request,
    write_outcome,
    write_run_start,
)


MUTANTS = (
    (
        "core-quiescence-claims-valid",
        "tpen/execution/core_admission.py",
        '"quiescence": "unknown",',
        '"quiescence": "valid",',
        "test_real_mapping_does_not_claim_quiescence",
    ),
    (
        "executor-accepts-core-quiescence",
        "executor/src/tpen_executor/recovery.py",
        'if payload["quiescence"] != "unknown":',
        "if False:",
        "test_executor_rejects_core_quiescence_claim",
    ),
)


def _group(task_ids: tuple[str, ...] = ("task",)) -> ReadyGroupV2:
    members = tuple(
        TaskSpec(
            task_id,
            f"lineage-{task_id}",
            ProgramSpecRef({"task": task_id}, digest({"task": task_id})),
            digest({"task": task_id, "plan": "b1d-int"}),
            1,
            "steps",
        )
        for task_id in task_ids
    )
    plan_ref = {"members": list(task_ids), "plan": "b1d-int"}
    plan = ExecutionPlanV2(
        plan_ref,
        digest(plan_ref),
        task_ids,
        "native-recovery/v2",
        "preserve-or-park/v1",
    )
    return ReadyGroupV2(GroupId("group-b1d-int"), members, plan)


def _member(
    group: ReadyGroupV2,
    task_id: str = "task",
    *,
    status: str = "failed",
    root: Path,
) -> MemberRecoveryInputV2:
    attempt = AttemptId(f"attempt-{task_id}")
    start = StartReceipt(
        task_id,
        attempt,
        AllocationId(f"allocation-{task_id}"),
        BackendJobId(f"job-{task_id}"),
        group.identity,
        "2026-11-21T00:00:00Z",
        str(root / task_id),
    )
    outcome = OutcomeReceipt(
        task_id,
        attempt,
        group.identity,
        "2026-11-21T00:01:00Z",
        status,
        start.output_dir,
    )
    return MemberRecoveryInputV2(
        task_id,
        f"lineage-{task_id}",
        attempt,
        start,
        outcome,
    )


def _reconcile_real_mapping(
    tmp_path: Path,
    context,
    *,
    task_id: str = "task",
    status: str = "failed",
) -> tuple[dict[str, object] | None, object]:
    """Run the real producer through the real executor decoder and reducer."""

    mapping = evaluate_core_admission(context)
    group = _group((task_id,))
    member = _member(group, task_id, status=status, root=tmp_path)
    result = reconcile_group(
        group,
        (member,),
        admission_port=MappingCoreAdmissionPort(lambda _member: mapping),
    )
    return mapping, result.members[0]


def test_real_producer_and_decoder_cover_every_admission_class(tmp_path: Path) -> None:
    """Each typed producer class reaches its closed executor action."""

    completed = tmp_path / "completed"
    completed.mkdir()
    completed_ref = checkpoint(completed)
    write_run_start(completed)
    (completed / "status.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    write_outcome(completed, "completed", completed_ref)

    failed = tmp_path / "failed"
    failed.mkdir()
    write_run_start(failed)
    (failed / "status.json").write_text('{"status":"failed"}\n', encoding="utf-8")
    write_outcome(failed, "failed")

    refused = tmp_path / "refused"
    refused.mkdir()
    write_run_start(refused)
    (refused / "status.json").write_text('{"status":"refused"}\n', encoding="utf-8")
    write_outcome(refused, "refused")

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    write_run_start(unknown)
    (unknown / "status.json").write_text('{"status":"unknown"}\n', encoding="utf-8")
    write_outcome(unknown, "unknown")

    planned = tmp_path / "planned"
    planned_ref = planned_yield_run(planned)
    planned_request = request(planned_ref)

    profile_refused = tmp_path / "profile-refused"
    profile_refused_ref = planned_yield_run(profile_refused)
    profile_request = request(profile_refused_ref)

    stale = tmp_path / "stale"
    stale_ref = planned_yield_run(stale)
    stale_request = request(stale_ref)
    (stale / "checkpoints" / "publications.jsonl").unlink()

    cases = (
        ("completed", admission_context(completed), "preserve_completed", "completed-and-settled"),
        ("failed", admission_context(failed), "preserve_terminal", "core-terminal"),
        ("refused", admission_context(refused), "preserve_terminal", "core-terminal"),
        ("unknown", admission_context(unknown), "park", "unknown-core-result"),
        ("planned-yield", admission_context(planned, planned_request), "admit_successor", "valid-successor-admission"),
        (
            "profile-refusal",
            admission_context(
                profile_refused,
                profile_request,
                config={"callbacks": [{"_target_": "not.allowed"}]},
            ),
            "park",
            "unsupported-profile",
        ),
        ("stale-checkpoint", admission_context(stale, stale_request), "park", "core-admission-refused"),
        ("missing-leaf", admission_context(None), "park", "core-admission-unavailable"),
    )
    for name, context, action, reason in cases:
        mapping, result = _reconcile_real_mapping(
            tmp_path / name,
            context,
            status="completed" if name == "completed" else "failed",
        )
        assert (result.action, result.reason_code) == (action, reason), name
        if name == "missing-leaf":
            assert mapping is None
        else:
            assert mapping is not None

    _, contradictory = _reconcile_real_mapping(
        tmp_path / "contradictory-completion",
        admission_context(completed),
        status="failed",
    )
    assert (contradictory.action, contradictory.reason_code) == (
        "park",
        "contradictory-completion",
    )


def test_real_mapping_does_not_claim_quiescence(tmp_path: Path) -> None:
    """Core leaves operational quiescence unknown for the executor."""

    ref = planned_yield_run(tmp_path)
    mapping = evaluate_core_admission(
        admission_context(tmp_path, request(ref))
    )
    assert mapping is not None
    assert mapping["quiescence"] == "unknown"


def test_executor_rejects_core_quiescence_claim(tmp_path: Path) -> None:
    """The decoder rejects a core mapping that claims settled quiescence."""

    group = _group()
    member = _member(group, root=tmp_path)
    mapping = {
        "schema": "tpen-core-admission-v1",
        "outcome": "successor",
        "admission": "admitted",
        "checkpoint": "valid",
        "quiescence": "valid",
        "profile": None,
        "reason": "claimed",
    }
    with pytest.raises(CoreAdmissionUnavailable):
        MappingCoreAdmissionPort(lambda _member: mapping).admit(member)


def test_real_controller_launches_only_the_admitted_successor(tmp_path: Path) -> None:
    """The real port drives one successor launch and parks an unknown member."""

    # Reuse the durable controller fixture that creates valid start/outcome and
    # invocation records; only the core-owned mapping is replaced below.
    from test_b1a_b_publication import _fixture
    from test_review_b1b2_t2_r1 import _binding

    store, group, lease, inputs, _bindings, successor_rows, fences = _fixture(
        tmp_path, ("ambiguous", "successor"), remote=False
    )
    ambiguous_dir = Path(fences["ambiguous"].output_dir) / "core"
    write_run_start(ambiguous_dir)
    (ambiguous_dir / "status.json").write_text('{"status":"unknown"}\n', encoding="utf-8")
    write_outcome(ambiguous_dir, "unknown")

    successor_dir = Path(fences["successor"].output_dir) / "core"
    successor_ref = planned_yield_run(successor_dir)
    predecessor = inputs[1]
    proposed = successor_rows["successor"][2]
    successor_request = request(
        successor_ref,
        task_id="successor",
        predecessor_attempt_id=predecessor.attempt_id.value,
        successor_attempt_id=proposed.attempt_id.value,
    )
    successor_rows["successor"] = (
        successor_request.to_dict(),
        "b" * 64,
        proposed,
    )
    contexts = {
        "ambiguous": admission_context(ambiguous_dir, task_id="ambiguous"),
        "successor": admission_context(
            successor_dir, successor_request, task_id="successor"
        ),
    }

    class RecordingExecutor:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def launch(self, fence, invocation) -> MemberLaunchResult:
            self.calls.append(fence.task_id)
            return MemberLaunchResult(True)

    from tpen_executor.allocations import ConfiguredAllocation

    configured = ConfiguredAllocation(lease.allocation_id, lease.backend_job_id, {})
    facts = GroupControllerFactsV1(
        group,
        configured,
        "a" * 64,
        "b1d-controller",
        "2026-11-21T00:02:00Z",
        inputs,
        successor_rows,
    )
    executor = RecordingExecutor()
    result = run_group_controller(
        _binding(group, configured.backend_job_id),
        configured.backend_job_id,
        facts,
        store,
        MappingCoreAdmissionPort(
            lambda member: evaluate_core_admission(contexts[member.task_id])
        ),
        executor,
    )
    assert dict(result.member_outcomes) == {
        "ambiguous": "parked",
        "successor": "launched",
    }
    assert executor.calls == ["successor"]
    assert result.reconciliation.members[0].action == "park"
    assert result.reconciliation.members[1].action == "admit_successor"


def test_mutant_rows_are_well_formed_and_have_named_killers() -> None:
    """Mutation rows select one compilable source anchor and a real killer."""

    root = Path(__file__).parents[2]
    target_sources = {
        "tpen/execution/core_admission.py": (root / "tpen/execution/core_admission.py").read_text(),
        "executor/src/tpen_executor/recovery.py": (root / "executor/src/tpen_executor/recovery.py").read_text(),
    }
    source = Path(__file__).read_text()
    for name, target, before, after, killer in MUTANTS:
        target_source = target_sources[target]
        assert target_source.count(before) == 1, name
        compile(target_source.replace(before, after, 1), target, "exec")
        assert f"def {killer}(" in source, killer
        assert "wait(" not in after
