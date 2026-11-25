"""Real-store B1e-2b controller tests and product-only mutant controls."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation, ControllerOutcomeReceiptV1
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, InvocationId
from tpen_executor.dispatch.completion import complete_leaf_result
from tpen_executor.dispatch.transport import LeafRequest
from tpen_executor.lifecycle.balsam import BalsamGroupRegistrationIntent, BalsamGroupRegistrationReceipt
from tpen_executor.lifecycle.controller import (
    GroupControllerFactsV1, MemberLaunchResult, _collected_attempt_record, run_group_controller,
)
from tpen_executor.lifecycle.local_executor import CollectionWaitTimeout, LeafCompletionContext
from tpen_executor.recovery import CoreAdmissionVerdictV2, MemberRecoveryInputV2
from tpen_executor.results import OutcomeReceipt
from tpen_executor.storage import DirectoryStore

from test_b1a_b_publication import _fixture as _successor_fixture
from test_b1a_reconciliation import _core, _group
from test_b1e1_local_executor import _environment as _local_environment
from test_b1e1_local_executor import _materializer as _local_materializer
from test_core_evidence_v3 import _typed_evidence_payload
from test_review_b1b2_controller import _real_fresh
from test_s3_transport_wire import _request, _synthetic_result


CONTROLLER = "executor/src/tpen_executor/lifecycle/controller.py"
CONTRACT = "executor/src/tpen_executor/dispatch/contract.py"
MUTANTS = (
    {
        "id": "submit-then-collect",
        "target": CONTROLLER,
        "old": "                current_owned = entry\n                try:\n                    store.record_controller_outcome(fence, controller, invocation, \"launched\", launched.detail or \"acknowledged\")\n                except Exception as error:\n                    _park_member(member.task_id, \"outcome-receipt\", error)\n                active.append(entry)\n                current_owned = None\n",
        "new": "                current_owned = entry\n                try:\n                    store.record_controller_outcome(fence, controller, invocation, \"launched\", launched.detail or \"acknowledged\")\n                except Exception as error:\n                    _park_member(member.task_id, \"outcome-receipt\", error)\n                active.append(entry)\n                executor.collect(handle, facts.poll_timeout)\n                current_owned = None\n",
        "killer": "test_two_submits_before_either_resolves",
    },
    {
        "id": "return-before-quiescence",
        "target": CONTROLLER,
        "old": "    try:\n        while pending or active:\n",
        "new": "    try:\n        while False:\n",
        "killer": "test_all_handles_are_collected",
    },
    {
        "id": "stop-after-first-fault",
        "target": CONTROLLER,
        "old": "                _park_member(member.task_id, \"collection\", error)\n                current_owned = None\n                if not isinstance(error, Exception):",
        "new": "                _park_member(member.task_id, \"collection\", error)\n                current_owned = None\n                return _park(group, \"collection-failed\")\n                if not isinstance(error, Exception):",
        "killer": "test_failed_collect_does_not_skip_other",
    },
    {
        "id": "skip-final-reconcile",
        "target": CONTROLLER,
        "old": "        final = _merge_final_reconciliation(\n            published, reconcile_group(group, tuple(final_inputs)),\n            parked_this_run, collected_members,\n        )\n    except Exception as error:",
        "new": "        final = published\n    except Exception as error:",
        "killer": "test_final_reconciliation_feeds_exit",
    },
    {
        "id": "drop-controller-park-merge",
        "target": CONTROLLER,
        "old": "        final = _merge_final_reconciliation(\n            published, reconcile_group(group, tuple(final_inputs)),\n            parked_this_run, collected_members,\n        )",
        "new": "        final = reconcile_group(group, tuple(final_inputs))",
        "killer": "test_final_reconciliation_preserves_controller_park",
    },
    {
        "id": "reject-invalid-context",
        "target": CONTROLLER,
        "old": "                validated = validate_leaf_result(result.request, result.result)",
        "new": "                validated = result.result",
        "killer": "test_crossed_result_parks",
    },
    {
        "id": "resubmit-after-poll-timeout",
        "target": CONTROLLER,
        "old": "                active.append(entry)\n                current_owned = None\n                continue",
        "new": "                active.append(entry)\n                executor.submit(fence, invocation)\n                current_owned = None\n                continue",
        "killer": "test_timeout_reuses_same_handle",
    },
    {
        "id": "successor-accept-segment-guard-flip",
        "target": CONTRACT,
        "old": "            accept_segment\n            and\n",
        "new": "            not accept_segment\n            and\n",
        "killer": "test_complete_leaf_result_rejects_segment_when_disabled",
    },
    {
        "id": "capacity-ignores-queue",
        "target": CONTROLLER,
        "old": "        while pending and len(active) < capacity:\n",
        "new": "        while pending:\n",
        "killer": "test_capacity_one_refills_after_collect",
    },
    {
        "id": "timeout-leaks-terminal-handle",
        "target": "executor/src/tpen_executor/lifecycle/local_executor.py",
        "old": "                    leaf_error = handle._future.exception(timeout=0)\n                    if leaf_error is not None:\n                        terminal = True",
        "new": "                    leaf_error = handle._future.exception(timeout=0)\n                    if leaf_error is not None:",
        "killer": "test_leaf_timeout_during_bounded_wait_consumes_handle",
    },
    {
        "id": "ack-write-does-not-park",
        "target": CONTROLLER,
        "old": "                except Exception as error:\n                    _park_member(member.task_id, \"outcome-receipt\", error)\n                active.append(entry)",
        "new": "                except Exception as error:\n                    reasons[member.task_id] = _reason(\"outcome-receipt\", error)\n                active.append(entry)",
        "killer": "test_failed_launched_ack_write_parks_but_collects",
    },
    {
        "id": "manifest-uses-last-attempt",
        "target": CONTROLLER,
        "old": "    return next((record for record in attempts\n                 if record.task_id == task_id and record.attempt_id == attempt_id), None)",
        "new": "    return next((record for record in attempts if record.task_id == task_id), None)",
        "killer": "test_collected_attempt_record_selects_exact_identity",
    },
    {
        "id": "interrupt-settles-failed",
        "target": CONTROLLER,
        "old": "        _cleanup_unconfirmed(current_owned)\n        raise",
        "new": "        _cleanup_unconfirmed()\n        raise",
        "killer": "test_submit_base_exception_records_unconfirmed_for_outstanding",
    },
    {
        "id": "submit-reason-renamed",
        "target": CONTROLLER,
        "old": '"executor-failed"',
        "new": '"submit-failed"',
        "killer": "test_controller_outcome_reason_codes_persist",
    },
    {
        "id": "ack-reason-renamed",
        "target": CONTROLLER,
        "old": '"invalid-launch-ack"',
        "new": '"invalid-submit-ack"',
        "killer": "test_controller_outcome_reason_codes_persist",
    },
    {
        "id": "shared-attempt-dedup",
        "target": CONTROLLER,
        "old": "            key = (entry[1].task_id, entry[1].attempt_id)\n",
        "new": "            key = entry[1].attempt_id\n",
        "killer": "test_shared_successor_attempt_ids_cleanup",
    },
)


class DurableAdmission:
    """Return admission facts only after the controller has durable receipts."""

    def admit(self, member: MemberRecoveryInputV2) -> CoreAdmissionVerdictV2 | None:
        if member.outcome_receipt is None:
            return member.core_verdict if member.start_receipt is not None else None
        if member.outcome_receipt.status == "completed":
            return _core("completed")
        return member.core_verdict


class EventExecutor:
    """Finite, controllable submit/collect fake returning real leaf contexts."""

    capacity = 2

    def __init__(
        self,
        group,
        allocation,
        *,
        capacity: int = 2,
        fail_task: str | None = None,
        invalid_task: str | None = None,
        timeout_once: bool = False,
        base_exception_task: str | None = None,
        submit_error_task: str | None = None,
        acknowledge: bool = True,
    ) -> None:
        self.group = group
        self.allocation = allocation
        self.capacity = capacity
        self.fail_task = fail_task
        self.invalid_task = invalid_task
        self.timeout_once = timeout_once
        self.base_exception_task = base_exception_task
        self.submit_error_task = submit_error_task
        self.acknowledge = acknowledge
        self.calls: list[tuple[str, str]] = []
        self.submits: list[tuple[object, object, object]] = []
        self.collects: list[object] = []
        self.fences: dict[str, object] = {}
        self.contexts: dict[str, LeafCompletionContext] = {}
        self.ready = {member.task_id: Event() for member in group.members}
        self.all_submitted = Event()

    def submit(self, fence, invocation):
        task = next(member for member in self.group.members if member.task_id == invocation.task_id)
        self.fences[task.task_id] = fence
        self.calls.append(("submit", task.task_id))
        if task.task_id == self.submit_error_task:
            raise RuntimeError("controlled submit failure")
        configured = _local_materializer(task.program)
        environment = _local_environment(task, configured, None)
        request = LeafRequest.for_attempt(fence, self.allocation, configured, environment)
        core_dir = Path(fence.output_dir) / "core"
        core_dir.mkdir(parents=True, exist_ok=True)
        evidence = _typed_evidence_payload(core_dir, "completed")
        result = replace(_synthetic_result(request), evidence=evidence)
        self.contexts[task.task_id] = LeafCompletionContext(result, request, configured, environment)
        handle = (task.task_id, len(self.submits))
        self.submits.append((fence, invocation, handle))
        if self.capacity == 1:
            self.ready[task.task_id].set()
            self.all_submitted.set()
        elif len(self.submits) == len(self.group.members):
            self.all_submitted.set()
            for event in self.ready.values():
                event.set()
        return MemberLaunchResult(self.acknowledge), handle

    def collect(self, handle, timeout):
        assert 0 < timeout <= 6
        task_id = handle[0]
        self.calls.append(("collect", task_id))
        self.collects.append(handle)
        assert self.all_submitted.wait(timeout)
        assert self.ready[task_id].wait(timeout)
        if self.timeout_once:
            self.timeout_once = False
            raise CollectionWaitTimeout("controlled poll")
        if task_id == self.fail_task:
            raise RuntimeError("controlled leaf failure")
        if task_id == self.base_exception_task:
            class StopNow(BaseException):
                pass
            raise StopNow("controlled interruption")
        context = self.contexts[task_id]
        if task_id == self.invalid_task:
            context = replace(context, result=replace(context.result, task_id="crossed-task"))
        return context


def _fresh_group(tmp_path: Path, task_ids: tuple[str, ...]):
    if task_ids == ("real-task",):
        return _real_fresh(tmp_path)
    group = _group(task_ids)
    allocation = ConfiguredAllocation(AllocationId("real-allocation"), BackendJobId("real-job"), {})
    facts = GroupControllerFactsV1(
        group,
        allocation,
        "a" * 64,
        "real-nonce",
        "2026-11-22T00:00:00Z",
        tuple(MemberRecoveryInputV2(member.task_id, member.lineage_id) for member in group.members),
    )
    binding = BalsamGroupRegistrationReceipt(
        BalsamGroupRegistrationIntent(group.group_id, group.identity, digest({"real": True})),
        allocation.backend_job_id,
    )
    return group, allocation, facts, binding, DirectoryStore(tmp_path / "real-store")


def _run_fresh(tmp_path: Path, task_ids=("real-task",), **executor_options):
    group, allocation, facts, binding, store = _fresh_group(tmp_path, tuple(task_ids))
    executor = EventExecutor(group, allocation, **executor_options)
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, DurableAdmission(), executor)
    return result, executor, store


def _outcomes(root: Path) -> list[OutcomeReceipt]:
    return [OutcomeReceipt.decode(path.read_bytes()) for path in sorted(root.rglob("outcome.json"))]


def test_two_submits_before_either_resolves(tmp_path: Path) -> None:
    result, executor, store = _run_fresh(tmp_path, ("one", "two"))
    assert result.exit_code == 0
    assert executor.calls[:2] == [("submit", "one"), ("submit", "two")]
    assert all(kind == "collect" for kind, _ in executor.calls[2:])
    assert len(_outcomes(Path(store.root))) == 2


def test_capacity_one_refills_after_collect(tmp_path: Path) -> None:
    result, executor, store = _run_fresh(tmp_path, ("one", "two"), capacity=1)
    assert result.exit_code == 0
    assert executor.calls == [
        ("submit", "one"), ("collect", "one"),
        ("submit", "two"), ("collect", "two"),
    ]
    assert len(_outcomes(Path(store.root))) == 2


def test_failed_collect_does_not_skip_other(tmp_path: Path) -> None:
    result, executor, store = _run_fresh(tmp_path, ("one", "two"), fail_task="one")
    outcomes = {receipt.task_id: receipt for receipt in _outcomes(Path(store.root))}
    assert result.exit_code == 2
    assert [task for kind, task in executor.calls if kind == "collect"] == ["one", "two"]
    assert outcomes["one"].status == "remote-quiescence-unconfirmed"
    assert outcomes["two"].status == "completed"


def test_timeout_reuses_same_handle(tmp_path: Path) -> None:
    result, executor, store = _run_fresh(tmp_path, timeout_once=True)
    assert result.exit_code == 0
    assert len(executor.submits) == 1
    assert len(executor.collects) == 2
    assert executor.collects[0] == executor.collects[1]
    receipts = _outcomes(Path(store.root))
    assert len(receipts) == 1 and receipts[0].status == "completed"


def test_all_handles_are_collected(tmp_path: Path) -> None:
    result, executor, store = _run_fresh(tmp_path, ("one", "two"))
    assert result.exit_code == 0
    assert {handle[0] for handle in executor.collects} == {"one", "two"}
    assert all(len(list(Path(fence.output_dir).glob("outcome.json"))) == 1 for fence in executor.fences.values())
    assert len(_outcomes(Path(store.root))) == 2


def test_unacknowledged_submit_gets_no_outcome_receipt(tmp_path: Path) -> None:
    result, executor, store = _run_fresh(tmp_path, acknowledge=False)
    fence = executor.fences["real-task"]
    assert result.exit_code == 2
    assert not list(Path(fence.output_dir).glob("outcome.json"))
    assert not executor.collects
    assert not _outcomes(Path(store.root))


def test_final_reconciliation_feeds_exit(tmp_path: Path) -> None:
    result, _executor, store = _run_fresh(tmp_path)
    assert result.exit_code == 0
    assert result.reconciliation.members[0].action == "preserve_completed"
    assert _outcomes(Path(store.root))[0].status == "completed"


def test_nonregular_completion_artifact_does_not_park_collected_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    group, allocation, facts, _binding, store = _fresh_group(tmp_path, ("one", "two"))
    group = type(group)(
        group.group_id,
        tuple(replace(member, target_horizon_units="iterations") for member in group.members),
        group.execution_plan,
    )
    facts = replace(facts, group=group)
    binding = BalsamGroupRegistrationReceipt(
        BalsamGroupRegistrationIntent(group.group_id, group.identity, digest({"invalid-completion": True})),
        allocation.backend_job_id,
    )
    executor = EventExecutor(group, allocation)
    original_manifest = DirectoryStore.manifest

    def manifest_with_invalid_optional_completion(self, current_group):
        task_dir = Path(self.root) / "groups" / current_group.identity / "one"
        (task_dir / "full-horizon-completion.json").mkdir()
        return original_manifest(self, current_group)

    monkeypatch.setattr(DirectoryStore, "manifest", manifest_with_invalid_optional_completion)
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, DurableAdmission(), executor)

    assert executor.calls == [("submit", "one"), ("submit", "two"), ("collect", "one"), ("collect", "two")]
    assert result.member_outcomes == (("one", "collected"), ("two", "collected"))
    assert all(member.action == "preserve_completed" for member in result.reconciliation.members)
    assert result.exit_code == 0
    invalid_destination = Path(store.root) / "groups" / executor.group.identity / "one" / "full-horizon-completion.json"
    assert invalid_destination.is_dir()
    valid_destination = Path(store.root) / "groups" / executor.group.identity / "two" / "full-horizon-completion.json"
    assert valid_destination.is_file()


def test_failed_launched_ack_write_parks_but_collects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = DirectoryStore.record_controller_outcome

    def fail_launched(self, fence, controller, invocation, status, reason=""):
        if status == "launched":
            raise OSError("acknowledgement ledger unavailable")
        return original(self, fence, controller, invocation, status, reason)

    monkeypatch.setattr(DirectoryStore, "record_controller_outcome", fail_launched)
    result, executor, store = _run_fresh(tmp_path)
    fence = executor.fences["real-task"]
    assert result.exit_code == 2
    assert len(executor.collects) == 1
    assert result.reconciliation.members[0].action == "park"
    receipts = _outcomes(Path(store.root))
    assert len(receipts) == 1 and receipts[0].status == "completed"
    assert not (Path(fence.output_dir) / "controller-outcome.json").exists()


def test_collected_attempt_record_selects_exact_identity() -> None:
    previous = SimpleNamespace(task_id="member", attempt_id="previous")
    collected = SimpleNamespace(task_id="member", attempt_id="collected")
    unrelated = SimpleNamespace(task_id="other", attempt_id="collected")
    for attempts in (
        (collected, previous, unrelated),
        (previous, unrelated, collected),
    ):
        assert _collected_attempt_record(attempts, "member", "collected") is collected


@pytest.mark.parametrize("attempt_ids", [("f" * 32, "0" * 32), ("0" * 32, "f" * 32)])
def test_final_reconciliation_matches_collected_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attempt_ids: tuple[str, str],
) -> None:
    import test_b1a_b_publication as publication_module
    import tpen_executor.storage as storage_module

    real_attempt_id = publication_module.AttemptId
    predecessor_id, successor_id = attempt_ids
    monkeypatch.setattr(storage_module, "uuid4", lambda: SimpleNamespace(hex=predecessor_id))
    monkeypatch.setattr(
        publication_module,
        "AttemptId",
        lambda value: real_attempt_id(successor_id if value == "successor-1" else value),
    )
    store, group, _lease, inputs, _bindings, successors, _fences = _successor_fixture(
        tmp_path, ("successor",), remote=False,
    )
    successor = successors["successor"][2]
    allocation = ConfiguredAllocation(
        successor.allocation_attempt.allocation_id,
        successor.allocation_attempt.backend_job_id,
        {},
    )
    facts = GroupControllerFactsV1(
        group, allocation, "a" * 64, "successor-controller", "2026-11-22T00:00:00Z", inputs, successors,
    )
    binding = BalsamGroupRegistrationReceipt(
        BalsamGroupRegistrationIntent(group.group_id, group.identity, digest({"successor": True})),
        allocation.backend_job_id,
    )
    executor = EventExecutor(group, allocation)
    original = DirectoryStore.manifest
    def reversed_manifest(self, current_group):
        # Insert a distinct later attempt only after the collected successor
        # has been completed, so it cannot supersede the fence during collect.
        if not hasattr(executor, "stale"):
            monkeypatch.setattr(storage_module, "uuid4", lambda: SimpleNamespace(hex="e" * 32))
            invocation = executor.submits[0][1]
            stale = self.begin_attempt(
                group, "successor", allocation, "stale", "local",
                allocation_attempt=invocation.allocation_attempt,
                invocation_id=InvocationId("stale-successor"),
            )
            self.finish_attempt(stale, "remote-quiescence-unconfirmed", "stale-finished")
            executor.stale = stale
        manifest = original(self, current_group)
        collected = executor.fences["successor"].attempt_id
        selected = next(item for item in manifest.attempts if item.attempt_id == collected)
        predecessor = next(item for item in manifest.attempts if item.attempt_id == real_attempt_id(predecessor_id))
        later = next(item for item in manifest.attempts if item.attempt_id == real_attempt_id("e" * 32))
        return replace(manifest, attempts=(selected, later, predecessor))
    monkeypatch.setattr(DirectoryStore, "manifest", reversed_manifest)
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, DurableAdmission(), executor)
    assert result.exit_code == 0
    assert result.reconciliation.members[0].action == "preserve_completed"
    assert any(receipt.status == "completed" for receipt in _outcomes(Path(store.root)))


@pytest.mark.parametrize("mode", ["submit", "begin", "ack"])
def test_submit_base_exception_records_unconfirmed_for_outstanding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    class StopNow(BaseException):
        pass

    group, allocation, facts, binding, store = _fresh_group(tmp_path, ("one", "two"))
    executor = EventExecutor(
        group, allocation, capacity=2,
        submit_error_task="two" if mode == "submit" else None,
    )
    if mode == "submit":
        def submit(fence, invocation):
            if invocation.task_id == "two":
                raise StopNow("submit interruption")
            return EventExecutor.submit(executor, fence, invocation)
        monkeypatch.setattr(executor, "submit", submit)
    elif mode == "begin":
        original = DirectoryStore.begin_attempt_current_controller
        def begin(self, group, task_id, *args, **kwargs):
            if task_id == "two":
                raise StopNow("begin interruption")
            return original(self, group, task_id, *args, **kwargs)
        monkeypatch.setattr(DirectoryStore, "begin_attempt_current_controller", begin)
    else:
        original = DirectoryStore.record_controller_outcome
        def record(self, fence, controller, invocation, status, reason=""):
            if invocation.task_id == "two" and status == "launched":
                raise StopNow("ack interruption")
            return original(self, fence, controller, invocation, status, reason)
        monkeypatch.setattr(DirectoryStore, "record_controller_outcome", record)
    with pytest.raises(StopNow):
        run_group_controller(binding, allocation.backend_job_id, facts, store, DurableAdmission(), executor)
    receipts = _outcomes(Path(store.root))
    assert all(receipt.status != "failed" for receipt in receipts)
    receipt_keys = {(receipt.task_id, receipt.attempt_id) for receipt in receipts}
    one_key = ("one", executor.fences["one"].attempt_id)
    if mode == "begin":
        expected_keys = {one_key}
        assert "two" not in executor.fences
    elif mode == "submit":
        # submit raised before returning an acknowledgement, so only the
        # already acknowledged member has a terminal operational receipt.
        expected_keys = {one_key}
        assert "two" not in executor.fences
    else:
        expected_keys = {one_key, ("two", executor.fences["two"].attempt_id)}
    assert receipt_keys == expected_keys
    assert all(receipt.status == "remote-quiescence-unconfirmed" for receipt in receipts)


def test_controller_outcome_reason_codes_persist(tmp_path: Path) -> None:
    (tmp_path / "failed").mkdir()
    failed, failed_executor, failed_store = _run_fresh(tmp_path / "failed", submit_error_task="real-task")
    assert "real-task" in failed_executor.fences, failed.member_reasons
    failed_receipt = ControllerOutcomeReceiptV1.decode(
        (Path(failed_executor.fences["real-task"].output_dir) / "controller-outcome.json").read_bytes()
    )
    assert failed.exit_code == 2 and failed_receipt.reason == "executor-failed"
    (tmp_path / "invalid").mkdir()
    invalid, invalid_executor, invalid_store = _run_fresh(tmp_path / "invalid", acknowledge=False)
    assert "real-task" in invalid_executor.fences, invalid.member_reasons
    invalid_receipt = ControllerOutcomeReceiptV1.decode(
        (Path(invalid_executor.fences["real-task"].output_dir) / "controller-outcome.json").read_bytes()
    )
    assert invalid.exit_code == 2 and invalid_receipt.reason == "invalid-launch-ack"


def test_two_member_collect_base_exception_unconfirms_each_acknowledged_fence(tmp_path: Path) -> None:
    with pytest.raises(BaseException):
        _run_fresh(tmp_path, ("one", "two"), base_exception_task="one")
    receipts = _outcomes(tmp_path / "real-store")
    assert len(receipts) == 2
    assert all(receipt.status == "remote-quiescence-unconfirmed" for receipt in receipts)


def test_final_manifest_failure_parks_after_completed_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tpen_executor.dispatch.completion as completion_module

    original_completion = completion_module.complete_leaf_result
    completion_succeeded = False

    def wrapped_completion(*args, **kwargs):
        nonlocal completion_succeeded
        result = original_completion(*args, **kwargs)
        completion_succeeded = True
        return result

    monkeypatch.setattr(completion_module, "complete_leaf_result", wrapped_completion)
    original_manifest = DirectoryStore.manifest

    def fail_final(self, group):
        if completion_succeeded:
            raise OSError("manifest unavailable")
        return original_manifest(self, group)

    monkeypatch.setattr(DirectoryStore, "manifest", fail_final)
    initial_publication: dict[str, bytes] = {}
    original_collect = EventExecutor.collect

    # The real store root is deterministic for this helper; capture its
    # publication bytes at the first collection, before completion/final read.
    def capture_publication(self, handle, timeout):
        if not initial_publication:
            root = tmp_path / "real-store"
            for path in (root / "groups" / ".reconciliation-publications").glob("*.json"):
                initial_publication[str(path.relative_to(root))] = path.read_bytes()
        return original_collect(self, handle, timeout)

    monkeypatch.setattr(EventExecutor, "collect", capture_publication)
    result, executor, store = _run_fresh(tmp_path)
    fence = executor.fences["real-task"]
    output = Path(fence.output_dir)
    assert len(executor.submits) == 1 and len(executor.collects) == 1
    assert (output / "outcome.json").is_file()
    assert (output / "core-outcome.json").is_file()
    assert (output / "segment.json").is_file()
    assert result.member_outcomes == (("real-task", "collected"),)
    assert result.reconciliation.members[0].action == "park"
    assert result.reconciliation.members[0].reason_code.startswith("final-reconciliation:")
    assert result.exit_code == 2
    final_publication = {
        str(path.relative_to(Path(store.root))): path.read_bytes()
        for path in (Path(store.root) / "groups" / ".reconciliation-publications").glob("*.json")
    }
    assert final_publication == initial_publication
    assert len(_outcomes(Path(store.root))) == 1


@pytest.mark.parametrize(
    "successor_ids",
    [("successor-shared", "successor-shared"), ("successor-a-next", "successor-b-next")],
)
def test_shared_successor_attempt_ids_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, successor_ids: tuple[str, str],
) -> None:
    names = ("successor-a", "successor-b")
    store, group, _lease, inputs, _bindings, successors, fences = _successor_fixture(
        tmp_path, names, remote=False,
    )
    rewritten_successors = {}
    for name, successor_id in zip(names, successor_ids):
        request, acceptance, proposed = successors[name]
        successor_attempt = AttemptId(successor_id)
        request = store.build_successor_admission_request(
            fences[name], successor_attempt, request["science_input_digest"], request["target_iterations"],
        )
        rewritten_successors[name] = (
            request,
            acceptance,
            replace(proposed, attempt_id=successor_attempt, invocation_id=InvocationId(name + "-next")),
        )
    successors = rewritten_successors
    allocation = ConfiguredAllocation(
        successors[names[0]][2].allocation_attempt.allocation_id,
        successors[names[0]][2].allocation_attempt.backend_job_id,
        {},
    )
    facts = GroupControllerFactsV1(
        group, allocation, "a" * 64, "shared-successor-controller", "2026-11-22T00:00:00Z",
        inputs, successors,
    )
    binding = BalsamGroupRegistrationReceipt(
        BalsamGroupRegistrationIntent(group.group_id, group.identity, digest({"shared-successors": True})),
        allocation.backend_job_id,
    )
    executor = EventExecutor(group, allocation, capacity=2)

    class StopNow(BaseException):
        pass

    original_record = DirectoryStore.record_controller_outcome

    def interrupt_second_ack(self, fence, controller, invocation, status, reason=""):
        if invocation.task_id == names[1] and status == "launched":
            raise StopNow("second acknowledgement interruption")
        return original_record(self, fence, controller, invocation, status, reason)

    monkeypatch.setattr(DirectoryStore, "record_controller_outcome", interrupt_second_ack)
    with pytest.raises(StopNow):
        run_group_controller(binding, allocation.backend_job_id, facts, store, DurableAdmission(), executor)

    receipts = _outcomes(Path(store.root))
    expected = {(name, AttemptId(successor_id)) for name, successor_id in zip(names, successor_ids)}
    matching = [receipt for receipt in receipts if (receipt.task_id, receipt.attempt_id) in expected]
    assert {(receipt.task_id, receipt.attempt_id) for receipt in matching} == expected
    assert len(matching) == len(expected)
    assert all(receipt.status == "remote-quiescence-unconfirmed" for receipt in matching)
    predecessor_keys = {(name, fence.attempt_id) for name, fence in fences.items()}
    assert all(
        receipt.status != "failed"
        for receipt in receipts
        if (receipt.task_id, receipt.attempt_id) not in predecessor_keys
    )
    assert [task_id for kind, task_id in executor.calls if kind == "submit"] == list(names)
    assert len(executor.submits) == 2


def test_crossed_result_parks(tmp_path: Path) -> None:
    result, executor, store = _run_fresh(tmp_path, invalid_task="real-task")
    fence = executor.fences["real-task"]
    output = Path(fence.output_dir)
    receipts = _outcomes(Path(store.root))
    assert result.exit_code == 2
    assert len(receipts) == 1 and receipts[0].status == "remote-quiescence-unconfirmed"
    assert not any((output / name).exists() for name in ("placement.json", "core-outcome.json", "segment.json"))


def test_admitted_successor_has_core_evidence_but_no_segment(tmp_path: Path) -> None:
    store, group, _lease, inputs, _bindings, successors, _fences = _successor_fixture(tmp_path, ("successor",), remote=False)
    successor = successors["successor"][2]
    allocation = ConfiguredAllocation(successor.allocation_attempt.allocation_id, successor.allocation_attempt.backend_job_id, {})
    facts = GroupControllerFactsV1(
        group, allocation, "a" * 64, "successor-controller", "2026-11-22T00:00:00Z", inputs, successors,
    )
    binding = BalsamGroupRegistrationReceipt(
        BalsamGroupRegistrationIntent(group.group_id, group.identity, digest({"successor": True})),
        allocation.backend_job_id,
    )
    executor = EventExecutor(group, allocation)
    result = run_group_controller(binding, allocation.backend_job_id, facts, store, DurableAdmission(), executor)
    output = Path(executor.fences["successor"].output_dir)
    assert result.exit_code == 0
    assert (output / "outcome.json").is_file()
    assert (output / "core-outcome.json").is_file()
    assert not (output / "segment.json").exists()


def test_base_exception_from_collect_propagates_after_terminal_record(tmp_path: Path) -> None:
    with pytest.raises(BaseException, match="controlled interruption"):
        _run_fresh(tmp_path, base_exception_task="real-task")
    receipts = _outcomes(tmp_path / "real-store")
    assert len(receipts) == 1 and receipts[0].status == "remote-quiescence-unconfirmed"


def test_complete_leaf_result_rejects_segment_when_disabled(tmp_path: Path) -> None:
    request, parts = _request(tmp_path)
    group, _allocation, lifecycle, configured, fence = parts
    (Path(fence.output_dir) / "core").mkdir()
    error = complete_leaf_result(
        lifecycle,
        fence,
        request,
        _synthetic_result(request),
        group=group,
        member=group.members[0],
        configured=configured,
        accept_segment=False,
    )
    assert error is None
    assert (Path(fence.output_dir) / "outcome.json").is_file()
    assert not (Path(fence.output_dir) / "segment.json").exists()


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True, False])
def test_poll_timeout_rejects_nonpositive_nonfinite_and_bool(tmp_path: Path, value) -> None:
    _, _, facts, _, _ = _real_fresh(tmp_path)
    with pytest.raises(ValueError):
        replace(facts, poll_timeout=value)


def test_mutant_rows_are_mappings_with_unique_anchors_and_killers() -> None:
    assert all(set(row) == {"id", "target", "old", "new", "killer"} for row in MUTANTS)
    assert len({row["id"] for row in MUTANTS}) == len(MUTANTS)
    assert len({(row["target"], row["old"]) for row in MUTANTS}) == len(MUTANTS)
    assert all(row["old"] != row["new"] and row["killer"].startswith("test_") for row in MUTANTS)
