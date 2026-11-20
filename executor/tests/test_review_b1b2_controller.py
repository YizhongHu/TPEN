"""Behavioral fake-only B1b2 controller tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest

import tpen_executor.lifecycle.controller as controller_module
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation, ControllerGenerationV1
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId
from tpen_executor.lifecycle.balsam import BalsamGroupRegistrationIntent, BalsamGroupRegistrationReceipt, GroupFailure
from tpen_executor.recovery import GroupReconciliationV2, MemberReconciliationV2, MemberRecoveryInputV2
from tpen_executor.storage import DirectoryStore, LaunchIntentRejected
from test_b1a_reconciliation import _group

MODULE = Path(__file__).parents[1] / "src" / "tpen_executor" / "lifecycle" / "controller.py"
MODULE_NAME = "executor/src/tpen_executor/lifecycle/controller.py"
MUTANTS = (
    ("drop-job-id-check", MODULE_NAME, "if observed_job_id != binding.backend_job_id", "if False", "test_job_id_mismatch_is_pre_start"),
    ("drop-acquire-refusal-park", MODULE_NAME, "controller = store.acquire_controller_generation(group, controller_request)", "controller = controller_request", "test_stale_controller_parks_before_publish"),
    ("drop-pre-launch-authority-check", MODULE_NAME, "store.require_current_controller(group, controller.generation, controller.claimant_nonce)", "pass", "test_superseded_after_intent_does_not_launch"),
    ("begin-attempt-for-successor", MODULE_NAME, "store.begin_admitted_successor(", "store.begin_attempt(", "test_successor_uses_admitted_start_only"),
    ("relaunch-existing-intent", MODULE_NAME, "existing = store.read_launch_intent(fence)", "existing = None", "test_recovered_intent_never_relaunches"),
    ("park-is-not-success", MODULE_NAME, "else 2)", "else 0)", "test_terminal_and_park_exit_two"),
    ("publish-equality-off", MODULE_NAME, "if published.encode() != checked.encode():", "if False:", "test_publication_conflict_parks"),
)


class FakeAdmission:
    def __init__(self, error=None): self.error, self.calls = error, []
    def admit(self, member):
        self.calls.append(member.task_id)
        if self.error: raise self.error
        return None


class FakeExecutor:
    def __init__(self, error=None): self.error, self.calls = error, []
    def launch(self, fence, invocation):
        self.calls.append(fence.task_id)
        if self.error: raise self.error
        return True


class FakeStore:
    def __init__(self, real, controller, published, *, recovered=False, supersede=False, reject=False):
        self.real, self.controller, self.published = real, controller, published
        self.recovered, self.supersede, self.reject, self.calls = recovered, supersede, reject, []
    def record_pre_start_receipt(self, reason): self.real.record_pre_start_receipt(reason)
    def acquire_controller_generation(self, group, record):
        self.calls.append("acquire")
        if self.reject:
            from tpen_executor.storage import ControllerGenerationRejected
            raise ControllerGenerationRejected("stale")
        return self.controller
    def require_current_controller(self, group, generation, nonce):
        self.calls.append("require")
        if self.supersede and self.calls.count("require") >= 2: raise RuntimeError("superseded")
        return self.controller
    def publish_reconciliation(self, group, attempt, inputs, *, bindings, successors):
        self.calls.append("publish"); return self.published
    def begin_attempt(self, group, task_id, allocation, *args, **kwargs):
        self.calls.append("begin_attempt:" + task_id)
        return SimpleNamespace(task_id=task_id, attempt_id=AttemptId("attempt-" + task_id), output_dir=str(self.real.root))
    def begin_admitted_successor(self, group, task_id, allocation, *args, **kwargs):
        self.calls.append("begin_admitted_successor:" + task_id)
        return SimpleNamespace(task_id=task_id, attempt_id=AttemptId("successor-" + task_id), output_dir=str(self.real.root))
    def read_launch_intent(self, fence):
        self.calls.append("read_intent:" + fence.task_id)
        if self.recovered: return object()
        raise LaunchIntentRejected("none")
    def create_launch_intent(self, fence, controller, invocation): self.calls.append("intent:" + fence.task_id); return object()


def _setup(tmp_path, actions=("start_fresh",), **kwargs):
    group = _group(tuple("task-" + str(i) for i in range(len(actions))))
    allocation = ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})
    controller = ControllerGenerationV1(group.group_id, 1, allocation.allocation_id, allocation.backend_job_id, "r" * 64, "nonce", None)
    published = GroupReconciliationV2(group.group_id, group.identity, tuple(MemberReconciliationV2(task.task_id, action, "fake", None) for task, action in zip(group.members, actions)))
    real = DirectoryStore(tmp_path / "store")
    store = FakeStore(real, controller, published, **kwargs)
    facts = controller_module.GroupControllerFactsV1(group, allocation, "r" * 64, "nonce", "utc", tuple(MemberRecoveryInputV2(task.task_id, task.lineage_id) for task in group.members))
    binding = BalsamGroupRegistrationReceipt(BalsamGroupRegistrationIntent(group.group_id, group.identity, digest({"backend": "fake"})), BackendJobId("job"))
    return facts, binding, store


def _run(tmp_path, actions=("start_fresh",), **kwargs):
    facts, binding, store = _setup(tmp_path, actions, **kwargs)
    executor, admission = FakeExecutor(), FakeAdmission()
    original = controller_module.reconcile_group
    controller_module.reconcile_group = lambda group, inputs: store.published
    try: result = controller_module.run_group_controller(binding, BackendJobId("job"), facts, store, admission, executor)
    finally: controller_module.reconcile_group = original
    return result, store, executor


def test_ordered_happy_mixed_group_and_exit_contract(tmp_path):
    result, store, executor = _run(tmp_path, ("preserve_completed", "preserve_terminal", "start_fresh", "park"))
    assert result.exit_code == 2 and result.reconciliation.aggregate == "parked"
    assert store.calls[:3] == ["acquire", "require", "publish"] and executor.calls == ["task-2"]

def test_job_id_mismatch_is_pre_start(tmp_path):
    facts, binding, store = _setup(tmp_path)
    result = controller_module.run_group_controller(binding, BackendJobId("other"), facts, store, FakeAdmission(), FakeExecutor())
    assert result == GroupFailure.PRE_START and list(Path(store.real.root).rglob("pre-start-*.json")) and not store.calls

def test_stale_controller_parks_before_publish(tmp_path):
    result, store, executor = _run(tmp_path, reject=True)
    assert result.reconciliation.aggregate == "parked" and "publish" not in store.calls and not executor.calls

def test_superseded_after_intent_does_not_launch(tmp_path):
    result, store, executor = _run(tmp_path, supersede=True)
    assert result.reconciliation.aggregate == "replacement_required" and not executor.calls

def test_recovered_intent_never_relaunches(tmp_path):
    result, store, executor = _run(tmp_path, recovered=True)
    assert result.member_outcomes[0][1] == "parked" and not executor.calls

def test_successor_uses_admitted_start_only(tmp_path):
    result, store, executor = _run(tmp_path, ("admit_successor",))
    assert "begin_admitted_successor:task-0" in store.calls and "begin_attempt:task-0" not in store.calls

def test_executor_exception_is_one_call_and_parked(tmp_path):
    facts, binding, store = _setup(tmp_path)
    executor = FakeExecutor(error=RuntimeError("lost acknowledgement"))
    original = controller_module.reconcile_group; controller_module.reconcile_group = lambda group, inputs: store.published
    try: result = controller_module.run_group_controller(binding, BackendJobId("job"), facts, store, FakeAdmission(), executor)
    finally: controller_module.reconcile_group = original
    assert len(executor.calls) == 1 and result.exit_code == 2

def test_admission_exception_parks(tmp_path):
    facts, binding, store = _setup(tmp_path)
    result = controller_module.run_group_controller(binding, BackendJobId("job"), facts, store, FakeAdmission(error=RuntimeError("port")), FakeExecutor())
    assert result.reconciliation.aggregate == "parked"

def test_publication_conflict_parks(tmp_path):
    facts, binding, store = _setup(tmp_path)
    original = controller_module.reconcile_group; controller_module.reconcile_group = lambda group, inputs: GroupReconciliationV2(group.group_id, group.identity, (MemberReconciliationV2("task-0", "preserve_terminal", "different", None),))
    try: result = controller_module.run_group_controller(binding, BackendJobId("job"), facts, store, FakeAdmission(), FakeExecutor())
    finally: controller_module.reconcile_group = original
    assert result.reconciliation.aggregate == "parked"

def test_terminal_and_park_exit_two(tmp_path):
    result, _, _ = _run(tmp_path, ("preserve_terminal",)); assert result.exit_code == 2


def _real_fresh(tmp_path):
    group = _group(("real-task",))
    allocation = ConfiguredAllocation(AllocationId("real-allocation"), BackendJobId("real-job"), {})
    store = DirectoryStore(tmp_path / "real-store")
    facts = controller_module.GroupControllerFactsV1(group, allocation, "a" * 64, "real-nonce", "utc", (MemberRecoveryInputV2("real-task", "lineage-real-task"),))
    binding = BalsamGroupRegistrationReceipt(BalsamGroupRegistrationIntent(group.group_id, group.identity, digest({"real": True})), BackendJobId("real-job"))
    return group, allocation, facts, binding, store


def test_real_store_fresh_start_writes_intent_and_launches_once(tmp_path):
    group, allocation, facts, binding, store = _real_fresh(tmp_path)
    executor = FakeExecutor()
    result = controller_module.run_group_controller(binding, BackendJobId("real-job"), facts, store, FakeAdmission(), executor)
    assert result.exit_code == 2 and executor.calls == ["real-task"]
    assert list(Path(store.root).rglob("launch-intent.json"))


def test_real_store_recovery_after_executor_exception_parks_without_relaunch(tmp_path):
    group, allocation, facts, binding, store = _real_fresh(tmp_path)
    first = FakeExecutor(error=RuntimeError("lost acknowledgement"))
    controller_module.run_group_controller(binding, BackendJobId("real-job"), facts, store, FakeAdmission(), first)
    start_path = next(Path(store.root).rglob("start.json"))
    from tpen_executor.results import StartReceipt
    start = StartReceipt.decode(start_path.read_bytes())
    recovered = controller_module.GroupControllerFactsV1(group, allocation, facts.registration_receipt_digest, facts.claimant_nonce, facts.observed_at_utc, (MemberRecoveryInputV2("real-task", "lineage-real-task", start.attempt_id, start, None, None),))
    second = FakeExecutor()
    result = controller_module.run_group_controller(binding, BackendJobId("real-job"), recovered, store, FakeAdmission(), second)
    assert result.reconciliation.aggregate == "parked" and not second.calls


def test_real_store_stale_controller_parks_before_publish_or_launch(tmp_path):
    group, allocation, facts, binding, store = _real_fresh(tmp_path)
    first = controller_module.run_group_controller(binding, BackendJobId("real-job"), facts, store, FakeAdmission(), FakeExecutor())
    current = store.require_current_controller(group, 1, facts.claimant_nonce)
    newer = ControllerGenerationV1(group.group_id, 2, allocation.allocation_id, allocation.backend_job_id, facts.registration_receipt_digest, "newer-nonce", current.record_digest)
    store.acquire_controller_generation(group, newer)
    executor = FakeExecutor()
    result = controller_module.run_group_controller(binding, BackendJobId("real-job"), facts, store, FakeAdmission(), executor)
    assert result.reconciliation.aggregate == "parked" and not executor.calls

def test_static_audit_and_mutant_anchors():
    source = MODULE.read_text(encoding="utf-8")
    assert "from .balsam import" not in source.split("def _pre_start", 1)[0]
    assert all(token not in source for token in ("scheduler", "retry", "scancel"))
    names = {name for name, value in globals().items() if name.startswith("test_") and callable(value)}
    anchors = [row[2] for row in MUTANTS]; assert len(anchors) == len(set(anchors))
    for _, filename, before, after, killer in MUTANTS:
        assert source.count(before) == 1 and killer in names; compile(source.replace(before, after, 1), filename, "exec")
