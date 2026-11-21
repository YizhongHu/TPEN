"""Behaviour and mutant checks for the B1e-1 local submit/collect seam."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import os
import sys
from threading import Event
from typing import Any

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import AllocationAttemptV2, ConfiguredAllocation, MemberInvocationV2
from tpen_executor.contracts import (
    AllocationAttemptId,
    AllocationId,
    AttemptId,
    BackendJobId,
    ExecutionPlanV2,
    GroupId,
    InvocationId,
    ProgramSpecRef,
    ReadyGroupV2,
    TaskSpec,
)
from tpen_executor.dispatch.transport import LeafRequest, LeafResult, LeafTeardownReceipt
from tpen_executor.runners import TPENCommand
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle.local_executor import (
    CapacityExceededError,
    InvocationMismatchError,
    LeafSubmissionError,
    LocalLeafExecutor,
    UnknownLaunchHandleError,
)
from tpen_executor.placement import AllocationSlot
from tpen_executor.storage import AttemptFence
from tpen_executor.tasks import ConfiguredRun


MODULE = Path(__file__).parents[1] / "src" / "tpen_executor" / "lifecycle" / "local_executor.py"
MODULE_NAME = "executor/src/tpen_executor/lifecycle/local_executor.py"
MUTANTS = (
    (
        "submit-blocks-on-future",
        MODULE_NAME,
        "return MemberLaunchResult(True, status=\"submitted\", detail=\"local leaf accepted\"), handle",
        "handle._future.result(timeout=6); return MemberLaunchResult(True, status=\"submitted\", detail=\"local leaf accepted\"), handle",
        "test_submit_returns_before_leaf_completion",
    ),
    (
        "submit-retries-after-start-error",
        MODULE_NAME,
        "future = self._pool.submit(self._run_leaf, request)",
        "try:\n                future = self._pool.submit(self._run_leaf, request)\n            except Exception:\n                future = self._pool.submit(self._run_leaf, request)",
        "test_submit_start_exception_is_not_retried",
    ),
    (
        "slot-by-active-count",
        MODULE_NAME,
        "slot = self._free_slots.pop() if self._free_slots else None",
        "slot = self._slots[self._active] if self._active < len(self._slots) else None",
        "test_reuses_collected_slot_only",
    ),
    (
        "collect-skips-result-validation",
        MODULE_NAME,
        "result = validate_leaf_result(handle._request, result)",
        "result = result",
        "test_collect_rejects_crossed_result",
    ),
    (
        "submit-skips-cross-check",
        MODULE_NAME,
        "task = self._validate_invocation(fence, invocation)",
        "task = self._task(invocation.task_id)",
        "test_crossed_invocation_is_rejected_before_launch",
    ),
    (
        "submit-ignores-capacity",
        MODULE_NAME,
        "if self._active >= self.capacity:",
        "if False:",
        "test_capacity_refuses_without_launch",
    ),
    (
        "consume-on-later-done",
        MODULE_NAME,
        "if not terminal:\n                    # Preserve the handle and its slot lease for a later\n                    # collection attempt.\n                    raise",
        "if not terminal:\n                    # Preserve the handle and its slot lease for a later\n                    # collection attempt.\n                    terminal = handle._future.done()\n                    raise",
        "test_timeout_completion_race_preserves_uncollected_result",
    ),
    (
        "no-collect-reservation",
        MODULE_NAME,
        'if key in self._collecting:\n                raise HandleCollectionInProgressError("launch handle is already being collected")',
        'if False:\n                raise AssertionError("unreachable")',
        "test_overlapping_collect_returns_result_to_only_one_caller",
    ),
)


def _group(task_ids: tuple[str, ...]) -> ReadyGroupV2:
    members = tuple(
        TaskSpec(
            task_id,
            f"lineage-{task_id}",
            ProgramSpecRef({"task": task_id}, digest({"task": task_id})),
            digest({"science": task_id}),
            1,
            "step",
        )
        for task_id in task_ids
    )
    plan_ref = {"name": "b1e1", "members": list(task_ids)}
    plan = ExecutionPlanV2(plan_ref, digest(plan_ref), task_ids, "native", "preserve")
    return ReadyGroupV2(GroupId("b1e1-group"), members, plan)


def _allocation(count: int) -> ConfiguredAllocation:
    available = tuple(sorted(os.sched_getaffinity(0)))
    assert len(available) >= count
    slots = tuple(
        AllocationSlot(f"slot-{index}", "localhost", (), (available[index],), (), ())
        for index in range(count)
    )
    return ConfiguredAllocation(
        AllocationId("b1e1-allocation"),
        BackendJobId("b1e1-job"),
        {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot.to_dict() for slot in slots]}},
    )


def _materializer(program: ProgramSpecRef) -> ConfiguredRun:
    task_id = program.config_ref["task"]
    return ConfiguredRun(task_id, f"/tmp/{task_id}.yaml", ("run.seed=7",))


def _invocation(group: ReadyGroupV2, allocation: ConfiguredAllocation, task_id: str, root: Path) -> tuple[AttemptFence, MemberInvocationV2]:
    task = next(member for member in group.members if member.task_id == task_id)
    attempt = AttemptId(f"attempt-{task_id}")
    allocation_attempt = AllocationAttemptV2(
        group.group_id,
        AllocationAttemptId(f"allocation-attempt-{task_id}"),
        allocation.allocation_id,
        allocation.backend_job_id,
        "1",
        "2026-11-21T00:00:00Z",
    )
    fence = AttemptFence(
        str(root),
        group.identity,
        task_id,
        allocation.allocation_id.value,
        attempt,
        str(root / "groups" / group.identity / task_id / attempt.value),
    )
    invocation = MemberInvocationV2(
        allocation_attempt,
        group.identity,
        task_id,
        task.lineage_id,
        attempt,
        InvocationId(f"invocation-{task_id}"),
    )
    return fence, invocation


def _environment(*_args: Any) -> PreparedPythonEnvironment:
    return PreparedPythonEnvironment(sys.executable, env={"B1E1_BASE": "yes"})


def _failure_result(request: LeafRequest) -> LeafResult:
    return LeafResult(
        request.digest(),
        request.group_digest,
        request.task_id,
        request.attempt_id,
        request.allocation_id,
        request.backend_job_id,
        request.slot_id,
        request.payload_digest,
        "localhost",
        1,
        request.environment.python,
        (),
        None,
        {"type": "fake", "message": "controlled"},
        None,
        "not-run",
        None,
        None,
        LeafTeardownReceipt.for_request(request, inspection="not-run", child_group_reaped=True).to_dict(),
    )


def _executor(group: ReadyGroupV2, allocation: ConfiguredAllocation, runner: Any) -> LocalLeafExecutor:
    return LocalLeafExecutor(group, allocation, _materializer, _environment, leaf_runner=runner)


def test_submit_returns_before_leaf_completion(tmp_path: Path) -> None:
    group = _group(("one", "two"))
    allocation = _allocation(2)
    started = {task_id: Event() for task_id in ("one", "two")}
    release = Event()

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        started[request.task_id].set()
        release.wait(5)
        return _failure_result(request).encode()

    executor = _executor(group, allocation, runner)
    try:
        fence, invocation = _invocation(group, allocation, "one", tmp_path)
        acknowledgement, handle = executor.submit(fence, invocation)
        assert acknowledgement.acknowledged and not handle._future.done()
        assert started["one"].wait(2)
        fence2, invocation2 = _invocation(group, allocation, "two", tmp_path)
        acknowledgement2, handle2 = executor.submit(fence2, invocation2)
        assert acknowledgement2.acknowledged and started["two"].wait(2)
        release.set()
        executor.collect(handle)
        executor.collect(handle2)
    finally:
        release.set()
        executor.close()


def test_reuses_collected_slot_only(tmp_path: Path) -> None:
    group = _group(("one", "two", "three"))
    allocation = _allocation(2)
    captured: dict[str, LeafRequest] = {}
    captured_events = {task_id: Event() for task_id in ("one", "two", "three")}

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        captured[request.task_id] = request
        captured_events[request.task_id].set()
        return _failure_result(request).encode()

    executor = _executor(group, allocation, runner)
    try:
        fence_a, invocation_a = _invocation(group, allocation, "one", tmp_path)
        _, handle_a = executor.submit(fence_a, invocation_a)
        fence_b, invocation_b = _invocation(group, allocation, "two", tmp_path)
        _, handle_b = executor.submit(fence_b, invocation_b)
        executor.collect(handle_a)
        fence_c, invocation_c = _invocation(group, allocation, "three", tmp_path)
        _, handle_c = executor.submit(fence_c, invocation_c)
        assert all(event.wait(2) for event in captured_events.values())
        assert captured["three"].slot_id == captured["one"].slot_id
        assert captured["three"].slot_id != captured["two"].slot_id
        executor.collect(handle_b)
        executor.collect(handle_c)
    finally:
        executor.close()


def test_submit_start_exception_is_not_retried(tmp_path: Path) -> None:
    group = _group(("one",))
    allocation = _allocation(1)
    calls: list[bytes] = []
    executor = _executor(group, allocation, lambda payload: calls.append(payload) or b"")
    submit_calls = 0

    def fail_submit(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal submit_calls
        submit_calls += 1
        raise RuntimeError("start failed")

    executor._pool.submit = fail_submit  # type: ignore[method-assign]
    fence, invocation = _invocation(group, allocation, "one", tmp_path)
    try:
        with pytest.raises(LeafSubmissionError, match="submission failed"):
            executor.submit(fence, invocation)
        assert submit_calls == 1
        assert calls == []
    finally:
        executor.close()


def test_crossed_invocation_is_rejected_before_launch(tmp_path: Path) -> None:
    group = _group(("one",))
    allocation = _allocation(1)
    calls: list[bytes] = []
    executor = _executor(group, allocation, lambda payload: calls.append(payload) or b"")
    fence, invocation = _invocation(group, allocation, "one", tmp_path)
    crossed = replace(invocation, group_digest=digest({"crossed": True}))
    try:
        with pytest.raises(InvocationMismatchError):
            executor.submit(fence, crossed)
        assert calls == []
    finally:
        executor.close()


def test_collect_rejects_crossed_result(tmp_path: Path) -> None:
    group = _group(("one",))
    allocation = _allocation(1)

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        return replace(_failure_result(request), task_id="foreign").encode()

    executor = _executor(group, allocation, runner)
    fence, invocation = _invocation(group, allocation, "one", tmp_path)
    try:
        _, handle = executor.submit(fence, invocation)
        with pytest.raises(Exception, match="task"):
            executor.collect(handle)
        with pytest.raises(UnknownLaunchHandleError):
            executor.collect(handle)
    finally:
        executor.close()


def test_capacity_refuses_without_launch(tmp_path: Path) -> None:
    group = _group(("one", "two"))
    allocation = _allocation(1)
    started = Event()
    release = Event()
    calls: list[str] = []

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        calls.append(request.task_id)
        started.set()
        release.wait(5)
        return _failure_result(request).encode()

    executor = _executor(group, allocation, runner)
    try:
        fence, invocation = _invocation(group, allocation, "one", tmp_path)
        _, handle = executor.submit(fence, invocation)
        assert started.wait(2)
        assert calls == ["one"]
        fence2, invocation2 = _invocation(group, allocation, "two", tmp_path)
        with pytest.raises(CapacityExceededError):
            executor.submit(fence2, invocation2)
        assert calls == ["one"]
        release.set()
        executor.collect(handle)
    finally:
        release.set()
        executor.close()


def test_exact_request_args_and_environment_have_no_executor_additions(tmp_path: Path) -> None:
    group = _group(("one",))
    allocation = _allocation(1)
    captured: dict[str, Any] = {}

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        captured["args"] = request.science_args()
        captured["env"] = dict(request.environment.env)
        return _failure_result(request).encode()

    executor = _executor(group, allocation, runner)
    fence, invocation = _invocation(group, allocation, "one", tmp_path)
    try:
        _, handle = executor.submit(fence, invocation)
        executor.collect(handle)
        configured = _materializer(group.members[0].program)
        assert captured["args"] == (
            *TPENCommand(configured.config, configured.overrides).args,
            f"run.root={fence.output_dir}",
            "run.layout=flat",
            "run.run_id=core",
        )
        assert captured["env"]["B1E1_BASE"] == "yes"
        assert "TPEN_EXECUTOR_LAUNCH_ROLE" not in captured["env"]
    finally:
        executor.close()


def test_real_local_run_leaf_subprocess(tmp_path: Path) -> None:
    """Run one real ``run_leaf`` subprocess with a minimal core shim."""

    fake_core = tmp_path / "fake-core" / "tpen"
    fake_core.mkdir(parents=True)
    (fake_core / "__init__.py").write_text("", encoding="utf-8")
    (fake_core / "run.py").write_text(
        "from pathlib import Path\n"
        "def main(argv):\n"
        "    root = Path(next(item.split('=', 1)[1] for item in argv if item.startswith('run.root=')))\n"
        "    root.mkdir(parents=True, exist_ok=True)\n"
        "    return 0\n",
        encoding="utf-8",
    )
    group = _group(("one",))
    allocation = ConfiguredAllocation(AllocationId("real-allocation"), BackendJobId("real-job"), {})
    environment = PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(fake_core.parent)})
    executor = LocalLeafExecutor(group, allocation, _materializer, environment)
    fence, invocation = _invocation(group, allocation, "one", tmp_path)
    Path(fence.output_dir).mkdir(parents=True)
    try:
        acknowledgement, handle = executor.submit(fence, invocation)
        assert acknowledgement.acknowledged
        result = executor.collect(handle, timeout=60)
        assert result.status() == "completed"
        assert result.phases == ("science", "inspection")
        assert result.science is not None and result.science["returncode"] == 0
        assert result.inspection == "unavailable"
    finally:
        executor.close()


def test_mutant_rows_are_unique_and_well_formed() -> None:
    source = MODULE.read_text(encoding="utf-8")
    review_source = Path(__file__).with_name("test_review_b1e1_r1.py").read_text(encoding="utf-8")
    test_source = Path(__file__).read_text(encoding="utf-8") + review_source
    assert len({row[2] for row in MUTANTS}) == len(MUTANTS)
    assert len(MUTANTS) == 8
    for _name, filename, old, new, killer in MUTANTS:
        assert filename == MODULE_NAME
        assert source.count(old) == 1
        compile(source.replace(old, new), str(MODULE), "exec")
        assert f"def {killer}(" in test_source

