"""Real-store B1e-2b controller tests and product-only mutant controls."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from threading import Event

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId
from tpen_executor.dispatch.completion import complete_leaf_result
from tpen_executor.dispatch.transport import LeafRequest
from tpen_executor.lifecycle.balsam import BalsamGroupRegistrationIntent, BalsamGroupRegistrationReceipt
from tpen_executor.lifecycle.controller import GroupControllerFactsV1, MemberLaunchResult, run_group_controller
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
        "old": "            active.append((member, fence, invocation, handle, controller, member.action == \"admit_successor\"))\n",
        "new": "            active.append((member, fence, invocation, handle, controller, member.action == \"admit_successor\"))\n            executor.collect(handle, facts.poll_timeout)\n",
        "killer": "test_two_submits_before_either_resolves",
    },
    {
        "id": "return-before-quiescence",
        "target": CONTROLLER,
        "old": "    while active:\n",
        "new": "    while False:\n",
        "killer": "test_all_handles_are_collected",
    },
    {
        "id": "stop-after-first-fault",
        "target": CONTROLLER,
        "old": "            _park_member(member.task_id, \"collection\", error)\n            if not isinstance(error, Exception):",
        "new": "            _park_member(member.task_id, \"collection\", error)\n            return _park(group, \"collection-failed\")\n            if not isinstance(error, Exception):",
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
        "old": "            active.append((member, fence, invocation, handle, owner, successor))\n            continue",
        "new": "            active.append((member, fence, invocation, handle, owner, successor))\n            executor.submit(fence, invocation)\n            continue",
        "killer": "test_timeout_reuses_same_handle",
    },
    {
        "id": "successor-accept-segment-guard-flip",
        "target": CONTRACT,
        "old": "            accept_segment\n            and\n",
        "new": "            not accept_segment\n            and\n",
        "killer": "test_complete_leaf_result_rejects_segment_when_disabled",
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
        fail_task: str | None = None,
        invalid_task: str | None = None,
        timeout_once: bool = False,
        base_exception_task: str | None = None,
        acknowledge: bool = True,
    ) -> None:
        self.group = group
        self.allocation = allocation
        self.fail_task = fail_task
        self.invalid_task = invalid_task
        self.timeout_once = timeout_once
        self.base_exception_task = base_exception_task
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
        self.calls.append(("submit", task.task_id))
        self.fences[task.task_id] = fence
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
        if len(self.submits) == len(self.group.members):
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
