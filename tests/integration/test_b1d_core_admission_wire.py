"""B1d integration coverage for the real core producer and executor decoder."""

from __future__ import annotations

from pathlib import Path

from tpen.execution.core_admission import evaluate_core_admission

from tests.helpers.core_admission import (
    admission_context,
    checkpoint,
    planned_yield_run,
    request,
    write_outcome,
    write_run_start,
)
from tests.helpers.executor_boundary import executor_call


_EXECUTOR_CHILD = r'''
import builtins
import json
import sys
from pathlib import Path

_real_import = builtins.__import__
_tpen_imports = []


def _block_tpen_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tpen" or name.startswith("tpen."):
        _tpen_imports.append(name)
        raise ModuleNotFoundError("executor child must not import tpen")
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = _block_tpen_import


def _emit(value):
    if not isinstance(value, dict):
        raise TypeError("executor child result must be a mapping")
    tpen_modules = sorted(
        name for name in sys.modules if name == "tpen" or name.startswith("tpen.")
    )
    value["_executor_boundary"] = {
        "tpen_imports": list(_tpen_imports),
        "tpen_modules": tpen_modules,
    }
    print(json.dumps(value, sort_keys=True))


from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
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
from tpen_executor.lifecycle.balsam import (
    BalsamGroupRegistrationIntent,
    BalsamGroupRegistrationReceipt,
)
from tpen_executor.lifecycle.controller import (
    GroupControllerFactsV1,
    MemberLaunchResult,
    run_group_controller,
)
from tpen_executor.recovery import (
    MappingCoreAdmissionPort,
    MemberRecoveryInputV2,
    reconcile_group,
)
from tpen_executor.results import OutcomeReceipt, StartReceipt

if any(name == "tpen" or name.startswith("tpen.") for name in sys.modules):
    raise RuntimeError("executor child started with tpen already imported")


def _group(task_ids=("task",)):
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


def _member(group, task_id="task", status="failed", root=Path(".")):
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


def _binding(group, job):
    return BalsamGroupRegistrationReceipt(
        BalsamGroupRegistrationIntent(group.group_id, group.identity, digest({"b1d": True})),
        job,
    )


def _reconcile(payload):
    mapping = payload["mapping"]
    task_id = payload["task_id"]
    group = _group((task_id,))
    member = _member(
        group,
        task_id,
        status=payload["status"],
        root=Path(payload["root"]),
    )
    result = reconcile_group(
        group,
        (member,),
        admission_port=MappingCoreAdmissionPort(lambda _member: mapping),
    )
    member_result = result.members[0]
    return {
        "action": member_result.action,
        "reason_code": member_result.reason_code,
    }


def _controller(payload):
    # Use the existing executor-side durable fixture; only the core mapping is
    # supplied over the wire from the parent process.
    from test_b1a_b_publication import _fixture

    store, group, lease, inputs, _bindings, successor_rows, _fences = _fixture(
        Path(payload["root"]), ("ambiguous", "successor"), remote=False
    )

    class RecordingExecutor:
        def __init__(self):
            self.calls = []

        def launch(self, fence, invocation):
            self.calls.append(fence.task_id)
            return MemberLaunchResult(True)

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
    mappings = payload["mappings"]
    executor = RecordingExecutor()
    result = run_group_controller(
        _binding(group, configured.backend_job_id),
        configured.backend_job_id,
        facts,
        store,
        MappingCoreAdmissionPort(lambda member: mappings[member.task_id]),
        executor,
    )
    _emit({
        "member_outcomes": dict(result.member_outcomes),
        "launch_calls": list(executor.calls),
        "launch_count": len(executor.calls),
    })


action = sys.argv[1]
payload = json.loads(sys.argv[2])
if action == "reconcile":
    _emit(_reconcile(payload))
elif action == "controller":
    _controller(payload)
else:
    raise SystemExit(f"unknown child action: {action}")
'''


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


def _reconcile_real_mapping(
    tmp_path: Path,
    context,
    *,
    task_id: str = "task",
    status: str = "failed",
) -> tuple[dict[str, object] | None, object]:
    """Run the real producer through the executor-only child decoder/reducer."""

    mapping = evaluate_core_admission(context)
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = executor_call(
        _EXECUTOR_CHILD,
        tmp_path,
        "reconcile",
        {
            "mapping": mapping,
            "root": str(tmp_path),
            "status": status,
            "task_id": task_id,
        },
    )
    return mapping, result


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
        assert (result["action"], result["reason_code"]) == (action, reason), name
        if name == "missing-leaf":
            assert mapping is None
        else:
            assert mapping is not None

    _, contradictory = _reconcile_real_mapping(
        tmp_path / "contradictory-completion",
        admission_context(completed),
        status="failed",
    )
    assert (contradictory["action"], contradictory["reason_code"]) == (
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

    mapping = {
        "schema": "tpen-core-admission-v1",
        "outcome": "successor",
        "admission": "admitted",
        "checkpoint": "valid",
        "quiescence": "valid",
        "profile": None,
        "reason": "claimed",
    }
    result = executor_call(
        _EXECUTOR_CHILD,
        tmp_path,
        "reconcile",
        {"mapping": mapping, "root": str(tmp_path), "status": "failed", "task_id": "task"},
    )
    assert (result["action"], result["reason_code"]) == (
        "park",
        "core-admission-unavailable",
    )


def test_real_controller_launches_only_the_admitted_successor(tmp_path: Path) -> None:
    """The real port drives one successor launch and parks an unknown member."""

    # Build both producer contexts in the parent from the real CPU fixtures.
    # The executor child creates its own durable controller fixture and only
    # receives these closed mappings over the wire.
    controller_root = tmp_path / "controller-parent"
    ambiguous_dir = controller_root / "ambiguous"
    ambiguous_dir.mkdir(parents=True)
    write_run_start(ambiguous_dir)
    (ambiguous_dir / "status.json").write_text('{"status":"unknown"}\n', encoding="utf-8")
    write_outcome(ambiguous_dir, "unknown")

    successor_dir = controller_root / "successor"
    successor_ref = planned_yield_run(successor_dir)
    successor_request = request(
        successor_ref,
        task_id="successor",
    )
    mappings = {
        "ambiguous": evaluate_core_admission(
            admission_context(ambiguous_dir, task_id="ambiguous")
        ),
        "successor": evaluate_core_admission(
            admission_context(successor_dir, successor_request, task_id="successor")
        ),
    }
    result = executor_call(
        _EXECUTOR_CHILD,
        tmp_path,
        "controller",
        {"mappings": mappings, "root": str(tmp_path / "controller-child")},
    )
    assert result["member_outcomes"] == {
        "ambiguous": "parked",
        "successor": "launched",
    }
    assert result["launch_calls"] == ["successor"]
    assert result["launch_count"] == 1


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
