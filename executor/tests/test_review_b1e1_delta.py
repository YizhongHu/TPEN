"""Delta-review proposals at ae2b104a; run only on Cannon.

Interrupted-wait regression is expected RED; bounded leaf-timeout retry is
expected GREEN. These expectations have not been measured by the reviewer.
"""

from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from test_b1e1_local_executor import _allocation, _executor, _failure_result, _group, _invocation
from tpen_executor.dispatch.transport import LeafRequest, LeafResult
from tpen_executor.lifecycle import local_executor as local


def test_interrupted_wait_keeps_live_leaf_handle_and_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupting the caller's wait does not terminate its running leaf."""
    group = _group(("one", "two"))
    allocation = _allocation(1)
    started = Event()
    release = Event()

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        if request.task_id == "one":
            started.set()
            assert release.wait(5), "test did not release the live leaf"
        return _failure_result(request).encode()

    executor = _executor(group, allocation, runner)
    try:
        _, handle = executor.submit(*_invocation(group, allocation, "one", tmp_path))
        assert started.wait(2), "leaf did not start"
        real_result = handle._future.result
        interrupted = False

        def interrupt_once(timeout: float | None = None) -> LeafResult:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                # Model SIGINT interrupting Condition.wait in Future.result.
                # The leaf remains blocked on release, so this is demonstrably
                # a caller-side interruption, not a stored leaf exception.
                raise KeyboardInterrupt("collector wait interrupted")
            return real_result(timeout=timeout)

        monkeypatch.setattr(handle._future, "result", interrupt_once)
        with pytest.raises(KeyboardInterrupt, match="collector wait interrupted"):
            executor.collect(handle, timeout=1)
        assert not handle._future.done(), "interruption must leave the leaf alive"
        with pytest.raises(local.CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
        release.set()
        assert executor.collect(handle, timeout=2).task_id == "one"
        with pytest.raises(local.UnknownLaunchHandleError):
            executor.collect(handle)
        _, next_handle = executor.submit(*_invocation(group, allocation, "two", tmp_path))
        assert next_handle.slot_id == handle.slot_id
        assert executor.collect(next_handle, timeout=2).task_id == "two"
    finally:
        release.set()
        executor.close()


def test_leaf_timeout_during_bounded_wait_consumes_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delivered leaf TimeoutError consumes its handle and slot immediately."""
    group = _group(("one", "two"))
    allocation = _allocation(1)
    leaf_error = FutureTimeoutError("terminal leaf timeout")

    class FailsDuringWait(Future[LeafResult]):
        def result(self, timeout: float | None = None) -> LeafResult:
            if not self.done():
                # Future was pending at collect entry, but completes with a
                # leaf exception before the bounded wait returns.
                self.set_exception(leaf_error)
            return super().result(timeout=timeout)

    future = FailsDuringWait()

    def submit(_function: Any, _request: LeafRequest) -> Future[LeafResult]:
        return future

    executor = _executor(group, allocation, lambda payload: b"")
    monkeypatch.setattr(executor._pool, "submit", submit)
    try:
        _, handle = executor.submit(*_invocation(group, allocation, "one", tmp_path))
        with pytest.raises(FutureTimeoutError) as first:
            executor.collect(handle, timeout=1)
        assert first.value is leaf_error
        assert future.done()
        assert handle.slot_id in executor.retired_slot_ids
        with pytest.raises(local.UnknownLaunchHandleError):
            executor.collect(handle)
        with pytest.raises(local.CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
    finally:
        executor.close()
