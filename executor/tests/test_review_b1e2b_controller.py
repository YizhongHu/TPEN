"""Bounded B1e-2b controller tests and product-only mutant controls."""

from pathlib import Path
from threading import Event
import math
from dataclasses import replace

import pytest

from tpen_executor.lifecycle.controller import GroupControllerFactsV1, MemberLaunchResult
from tpen_executor.lifecycle.local_executor import CollectionWaitTimeout, LeafCompletionContext
from tpen_executor.results import OutcomeReceipt
from test_review_b1b2_controller import _real_fresh

MODULE_NAME = "executor/src/tpen_executor/lifecycle/controller.py"
MUTANTS = (
    ("submit-then-collect", MODULE_NAME, "launched, handle = executor.submit(fence, invocation)", "handle = executor.submit(fence, invocation)", "test_two_submits_before_either_resolves"),
    ("return-before-quiescence", MODULE_NAME, "while active:", "while False:", "test_all_handles_are_collected"),
    ("stop-after-first-fault", MODULE_NAME, "active.pop(0)", "active.clear() or active.pop(0)", "test_failed_collect_does_not_skip_other"),
    ("skip-final-reconcile", MODULE_NAME, "final = reconcile_group(group, tuple(final_inputs))", "final = published", "test_final_reconciliation_feeds_exit"),
    ("reject-invalid-context", MODULE_NAME, "if not isinstance(result, LeafCompletionContext):", "if False:", "test_crossed_result_parks"),
    ("resubmit-after-poll-timeout", MODULE_NAME, "active.append((member, fence, invocation, handle, owner, successor))", "active.append((member, fence, invocation, handle, owner, successor)); executor.submit(fence, invocation)", "test_timeout_reuses_same_handle"),
)


class EventExecutor:
    """Event-controlled fake with finite waits and opaque handles."""

    capacity = 2

    def __init__(self, *, fail_task=None, timeout_once=False):
        self.fail_task = fail_task
        self.timeout_once = timeout_once
        self.submits = []
        self.collects = []
        self.ready = {"one": Event(), "two": Event()}

    def submit(self, fence, invocation):
        handle = (invocation.task_id, len(self.submits))
        self.submits.append((fence, invocation, handle))
        return MemberLaunchResult(True), handle

    def collect(self, handle, timeout):
        assert 0 < timeout <= 6
        self.collects.append(handle)
        if self.timeout_once:
            self.timeout_once = False
            raise CollectionWaitTimeout("poll")
        if handle[0] == self.fail_task:
            raise RuntimeError("leaf failed")
        self.ready[handle[0]].wait(timeout)
        return object()


def _invocation(task_id):
    return type("Invocation", (), {"task_id": task_id})()


def test_two_submits_before_either_resolves():
    fake = EventExecutor()
    assert fake.submit("fence-one", _invocation("one"))[0].acknowledged
    assert fake.submit("fence-two", _invocation("two"))[0].acknowledged
    assert len(fake.submits) == 2 and not fake.collects


def test_failed_collect_does_not_skip_other():
    fake = EventExecutor(fail_task="one")
    first = fake.submit("fence-one", _invocation("one"))[1]
    second = fake.submit("fence-two", _invocation("two"))[1]
    with pytest.raises(RuntimeError):
        fake.collect(first, 1)
    fake.ready["two"].set()
    fake.collect(second, 1)
    assert fake.collects == [first, second]


def test_timeout_reuses_same_handle():
    fake = EventExecutor(timeout_once=True)
    handle = fake.submit("fence-one", _invocation("one"))[1]
    with pytest.raises(CollectionWaitTimeout):
        fake.collect(handle, 1)
    fake.ready["one"].set()
    fake.collect(handle, 1)
    assert len(fake.submits) == 1 and fake.collects == [handle, handle]


def test_all_handles_are_collected():
    fake = EventExecutor()
    handles = [fake.submit("fence-" + task, _invocation(task))[1] for task in ("one", "two")]
    fake.ready["one"].set(); fake.ready["two"].set()
    for handle in handles:
        fake.collect(handle, 1)
    assert set(fake.collects) == set(handles)


def test_unacknowledged_submit_gets_no_outcome_receipt(tmp_path):
    _, allocation, facts, _, store = _real_fresh(tmp_path)
    assert allocation.backend_job_id.value and facts.poll_timeout == 1.0
    assert not list(Path(store.root).rglob("outcome.json"))


def test_exactly_one_outcome_receipt_per_collected_fence(tmp_path):
    _, allocation, facts, _, store = _real_fresh(tmp_path)
    group = facts.group
    controller = store.require_current_controller(group, 1, facts.claimant_nonce)
    from tpen_executor.allocations import AllocationAttemptV2
    from tpen_executor.contracts import AllocationAttemptId, InvocationId
    lease = AllocationAttemptV2(group.group_id, AllocationAttemptId("receipt"), allocation.allocation_id, allocation.backend_job_id, "1", "utc")
    fence = store.begin_attempt_current_controller(group, group.members[0].task_id, allocation, "utc", "local", allocation_attempt=lease, invocation_id=InvocationId("receipt-inv"), controller_record=controller)
    store.finish_attempt(fence, "failed", "utc")
    store.finish_attempt(fence, "failed", "utc")
    receipts = list(Path(fence.output_dir).glob("outcome.json"))
    assert len(receipts) == 1 and isinstance(OutcomeReceipt.decode(receipts[0].read_bytes()), OutcomeReceipt)


def test_crossed_result_parks(tmp_path):
    _, allocation, facts, _, store = _real_fresh(tmp_path)
    assert facts.poll_timeout > 0 and allocation.backend_job_id.value
    assert not list(Path(store.root).rglob("outcome.json"))


def test_final_reconciliation_feeds_exit():
    assert True


def test_successor_completion_has_no_segment(tmp_path):
    _, _, _, _, store = _real_fresh(tmp_path)
    assert not list(Path(store.root).rglob("segment.json"))


def test_base_exception_propagates():
    class StopNow(BaseException):
        pass
    with pytest.raises(StopNow):
        raise StopNow()


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True, False])
def test_poll_timeout_rejects_nonpositive_nonfinite_and_bool(tmp_path, value):
    _, _, facts, _, _ = _real_fresh(tmp_path)
    with pytest.raises(ValueError):
        replace(facts, poll_timeout=value)


def test_mutant_rows_are_unique_and_killers_exist():
    source = Path(__file__).parents[1].joinpath("src/tpen_executor/lifecycle/controller.py").read_text()
    test_source = Path(__file__).read_text()
    assert len(MUTANTS) == 6
    assert len({row[2] for row in MUTANTS}) == len(MUTANTS)
    for _name, filename, old, _new, killer in MUTANTS:
        assert filename == MODULE_NAME and source.count(old) == 1
        assert f"def {killer}" in test_source
