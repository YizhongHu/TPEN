"""T2 regression proposals for PR 810; runtime execution belongs on Cannon.

The first two tests are expected RED at bdbf3d5a. Remaining tests are expected
GREEN. Timeout/collection ordering is controlled explicitly, without sleeps.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import replace
from pathlib import Path
from threading import Event, Lock
from typing import Any

import pytest

from test_b1e1_local_executor import (
    _allocation, _executor, _failure_result, _group, _invocation, _materializer,
)
from tpen_executor._codec import digest
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, GroupId
from tpen_executor.dispatch.transport import LeafRequest, LeafResult, LeafResultRejected
from tpen_executor.lifecycle import local_executor as local


def test_timeout_completion_race_preserves_uncollected_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wait timeout must leave the result available even if it just finished."""
    group = _group(("one", "two"))
    allocation = _allocation(1)
    captured: list[LeafRequest] = []

    class FinishAfterTimeout(Future[LeafResult]):
        def result(self, timeout: float | None = None) -> LeafResult:
            try:
                return super().result(timeout=timeout)
            except FutureTimeoutError:
                # Model the legal interleaving: Future.result has decided to
                # raise, then the worker finishes before collect's finally.
                self.set_result(_failure_result(captured[0]))
                raise

    future = FinishAfterTimeout()

    def submit(_function: Any, request: LeafRequest) -> Future[LeafResult]:
        captured.append(request)
        return future

    executor = _executor(group, allocation, lambda payload: b"")
    monkeypatch.setattr(executor._pool, "submit", submit)
    try:
        _, handle = executor.submit(*_invocation(group, allocation, "one", tmp_path))
        with pytest.raises(FutureTimeoutError):
            executor.collect(handle, timeout=0)
        # A collector using wait() instead of result(timeout=...) may never
        # invoke the injected timeout branch. Complete it here in that case;
        # either ordering must preserve the unconsumed result and lease.
        if not future.done():
            future.set_result(_failure_result(captured[0]))
        # No result has reached the caller; the handle and its lease survive.
        with pytest.raises(local.CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
        assert executor.collect(handle, timeout=0).task_id == "one"
        with pytest.raises(local.UnknownLaunchHandleError):
            executor.collect(handle)
    finally:
        executor.close()


def test_overlapping_collect_returns_result_to_only_one_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handle must be reserved before waiting/validating outside the lock."""
    group = _group(("one",))
    allocation = _allocation(1)
    entered = Event()
    release = Event()
    call_lock = Lock()
    calls = 0
    validate = local.validate_leaf_result

    def gated_validate(request: LeafRequest, result: LeafResult) -> LeafResult:
        nonlocal calls
        with call_lock:
            calls += 1
            first = calls == 1
        if first:
            entered.set()
            assert release.wait(5), "test did not release the first collector"
        return validate(request, result)

    monkeypatch.setattr(local, "validate_leaf_result", gated_validate)
    executor = _executor(group, allocation, lambda payload: _failure_result(LeafRequest.decode(payload)).encode())
    collectors = ThreadPoolExecutor(max_workers=1)
    first = None
    try:
        _, handle = executor.submit(*_invocation(group, allocation, "one", tmp_path))
        first = collectors.submit(executor.collect, handle)
        assert entered.wait(5), "first collector did not reach validation"
        with pytest.raises(local.LocalExecutorError):
            executor.collect(handle, timeout=0)
    finally:
        release.set()
        try:
            if first is not None:
                assert first.result(timeout=5).task_id == "one"
        finally:
            collectors.shutdown()
            executor.close()


@pytest.mark.parametrize("failure", ["runner-error", "runner-timeout", "invalid-bytes", "crossed-result"])
def test_terminal_collection_error_returns_exact_slot_once(tmp_path: Path, failure: str) -> None:
    """Runner/decode/validation errors consume the handle without leaking leases."""
    group = _group(("one", "two", "three"))
    allocation = _allocation(1)
    captured: dict[str, LeafRequest] = {}

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        captured[request.task_id] = request
        if request.task_id == "one":
            if failure == "runner-error":
                raise RuntimeError("leaf error")
            if failure == "runner-timeout":
                raise FutureTimeoutError("leaf error")
            if failure == "invalid-bytes":
                return b"not-json"
            return replace(_failure_result(request), task_id="foreign").encode()
        return _failure_result(request).encode()

    expected = {"runner-error": RuntimeError, "runner-timeout": FutureTimeoutError,
                "invalid-bytes": LeafResultRejected, "crossed-result": LeafResultRejected}[failure]
    executor = _executor(group, allocation, runner)
    try:
        _, first = executor.submit(*_invocation(group, allocation, "one", tmp_path))
        with pytest.raises(expected):
            executor.collect(first)
        assert first.slot_id in executor.retired_slot_ids
        with pytest.raises(local.UnknownLaunchHandleError):
            executor.collect(first)
        with pytest.raises(local.CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
        assert set(captured) == {"one"}
    finally:
        executor.close()


@pytest.mark.parametrize("field", [
    "group", "task", "lineage", "attempt", "fence-allocation", "allocation-group",
    "allocation", "backend", "output",
])
def test_crossed_identity_dimensions_never_reach_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    """Exercise each identity gate, including the for_attempt path check."""
    group = _group(("one", "two"))
    allocation = _allocation(1)
    fence, invocation = _invocation(group, allocation, "one", tmp_path)
    if field == "group":
        invocation = replace(invocation, group_digest=digest({"foreign": True}))
    elif field == "task":
        invocation = replace(invocation, task_id="two", lineage_id="lineage-two")
    elif field == "lineage":
        invocation = replace(invocation, lineage_id="foreign")
    elif field == "attempt":
        invocation = replace(invocation, attempt_id=AttemptId("foreign"))
    elif field == "fence-allocation":
        fence = replace(fence, allocation_id="foreign")
    elif field == "allocation-group":
        invocation = replace(invocation, allocation_attempt=replace(invocation.allocation_attempt, group_id=GroupId("foreign")))
    elif field == "allocation":
        invocation = replace(invocation, allocation_attempt=replace(invocation.allocation_attempt, allocation_id=AllocationId("foreign")))
    elif field == "backend":
        invocation = replace(invocation, allocation_attempt=replace(invocation.allocation_attempt, backend_job_id=BackendJobId("foreign")))
    else:
        fence = replace(fence, output_dir=str(tmp_path / "foreign"))
    calls = []

    def unexpected_submit(*args: Any) -> None:
        calls.append(args)
        raise AssertionError("crossed identity reached submission")

    executor = _executor(group, allocation, lambda payload: b"")
    monkeypatch.setattr(executor._pool, "submit", unexpected_submit)
    try:
        expected = local.LeafSubmissionError if field == "output" else local.InvocationMismatchError
        with pytest.raises(expected):
            executor.submit(fence, invocation)
        assert calls == []
    finally:
        executor.close()


def test_timeout_then_close_retains_handle_and_collects_after_shutdown(tmp_path: Path) -> None:
    """Nonwaiting close does not discard active leaves or their collectible result."""
    group = _group(("one", "two"))
    allocation = _allocation(1)
    release = Event()

    def runner(payload: bytes) -> bytes:
        assert release.wait(5), "test did not release the leaf"
        return _failure_result(LeafRequest.decode(payload)).encode()

    executor = _executor(group, allocation, runner)
    stranger = _executor(group, allocation, runner)
    try:
        _, handle = executor.submit(*_invocation(group, allocation, "one", tmp_path))
        with pytest.raises(local.UnknownLaunchHandleError):
            stranger.collect(handle)
        with pytest.raises(FutureTimeoutError):
            executor.collect(handle, timeout=0)
        with pytest.raises(local.CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
        executor.close(wait=False)
        assert not handle._future.done()
        release.set()
        executor.close(wait=True)
        assert handle._future.done()
        assert executor.collect(handle).task_id == "one"
        with pytest.raises(local.UnknownLaunchHandleError):
            executor.collect(handle)
        with pytest.raises(local.LeafSubmissionError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
    finally:
        release.set()
        executor.close()
        stranger.close()


def test_complete_child_environment_matches_bound_state(tmp_path: Path) -> None:
    """Assert all keys/values, so a new accidental env entry cannot pass."""
    group = _group(("one",))
    allocation = _allocation(1)
    captured = []

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        captured.append(request)
        return _failure_result(request).encode()

    executor = _executor(group, allocation, runner)
    fence, invocation = _invocation(group, allocation, "one", tmp_path)
    try:
        _, handle = executor.submit(fence, invocation)
        executor.collect(handle)
        slot = allocation.facts["slots"]["slots"][0]
        cpus = ",".join(str(cpu) for cpu in slot["cpus"])
        configured = _materializer(group.members[0].program)
        assert dict(captured[0].environment.env) == {
            "B1E1_BASE": "yes",
            "TPEN_EXECUTOR_CONTEXT_ENV_KEYS": '["CUDA_VISIBLE_DEVICES","MKL_NUM_THREADS","OMP_NUM_THREADS"]',
            "TPEN_SLOT_ID": slot["slot_id"],
            "TPEN_SLOT_NODE": "localhost",
            "TPEN_SLOT_CPUS": cpus,
            "TPEN_PLACEMENT_OUTPUT": f"{fence.output_dir}/child-placement.json",
            "TPEN_PLACEMENT_OUTPUT_DIR": fence.output_dir,
            "TPEN_PLACEMENT_TASK_ID": "one",
            "TPEN_PLACEMENT_ATTEMPT_ID": fence.attempt_id.value,
            "TPEN_PLACEMENT_GROUP_DIGEST": group.identity,
            "TPEN_PLACEMENT_ALLOCATION_ID": allocation.allocation_id.value,
            "TPEN_PLACEMENT_BACKEND_JOB_ID": allocation.backend_job_id.value,
            "TPEN_PLACEMENT_SLOT_ID": slot["slot_id"],
            "TPEN_PLACEMENT_DECLARED_HOST": "localhost",
            "TPEN_PLACEMENT_DECLARED_CPUS": cpus,
            "TPEN_PLACEMENT_DECLARED_DEVICES": "",
            "TPEN_PLACEMENT_PAYLOAD_DIGEST": digest({"config": configured.config, "overrides": list(configured.overrides)}),
        }
    finally:
        executor.close()
