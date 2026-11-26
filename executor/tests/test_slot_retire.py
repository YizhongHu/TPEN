"""Slot retirement tests for terminal local-leaf collection outcomes."""

from __future__ import annotations

import ast
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from test_b1e1_local_executor import (
    _allocation, _environment, _failure_result, _group, _invocation, _materializer,
)
from tpen_executor.dispatch.transport import LeafRequest, LeafResultRejected
from tpen_executor.lifecycle.local_executor import (
    CapacityExceededError, CollectionWaitTimeout, LocalLeafExecutor,
)


MODULE = Path(__file__).parents[1] / "src" / "tpen_executor" / "lifecycle" / "local_executor.py"
MODULE_NAME = "executor/src/tpen_executor/lifecycle/local_executor.py"
MUTANTS = [
    {
        "id": "retire-unvalidated-result",
        "target": MODULE_NAME,
        "old": "            release_slot = True\n            return LeafCompletionContext(result, handle._request, handle._configured, handle._environment)",
        "new": "            release_slot = False\n            return LeafCompletionContext(result, handle._request, handle._configured, handle._environment)",
        "killer": "test_confirmed_reap_recycles_slot",
    },
    {
        "id": "refuse-retired-capacity",
        "target": MODULE_NAME,
        "old": "            if self._active >= self.capacity or (self._slots and not self._free_slots):",
        "new": "            if self._active >= self.capacity:",
        "killer": "test_unconfirmed_reap_retires_slot",
    },
    {
        "id": "requeue-terminal-error",
        "target": MODULE_NAME,
        "old": "                        if release_slot:\n                            self._free_slots.append(handle._slot)",
        "new": "                        if True:\n                            self._free_slots.append(handle._slot)",
        "killer": "test_leaf_exception_retires_slot",
    },
    {
        "id": "record-retired-slot",
        "target": MODULE_NAME,
        "old": "                            self._retired_slot_ids.add(handle._slot.slot_id)",
        "new": "                            self._retired_slot_ids.clear()",
        "killer": "test_retired_slot_ids_is_read_only_snapshot",
    },
]


def _executor(group: Any, allocation: Any, runner: Any) -> LocalLeafExecutor:
    return LocalLeafExecutor(group, allocation, _materializer, _environment, leaf_runner=runner)


def test_mutants_have_unique_compiling_anchors_and_killers() -> None:
    source = MODULE.read_text()
    module = Path(__file__).read_text()
    assert all(set(row) == {"id", "target", "old", "new", "killer"} for row in MUTANTS)
    assert len({row["id"] for row in MUTANTS}) == len(MUTANTS)
    assert len({(row["target"], row["old"]) for row in MUTANTS}) == len(MUTANTS)
    assert len({row["killer"] for row in MUTANTS}) == len(MUTANTS)
    for row in MUTANTS:
        assert source.count(row["old"]) == 1, row["id"]
        ast.parse(source.replace(row["old"], row["new"], 1))
        assert f"def {row['killer']}(" in module


def test_unconfirmed_reap_retires_slot(tmp_path: Path) -> None:
    group, allocation = _group(("one", "two")), _allocation(1)

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        result = _failure_result(request)
        return replace(result, teardown={**result.teardown, "child_group_reaped": False}).encode()

    executor = _executor(group, allocation, runner)
    try:
        with pytest.raises(LeafResultRejected, match="teardown"):
            executor.collect(executor.submit(*_invocation(group, allocation, "one", tmp_path))[1], timeout=2)
        assert executor.retired_slot_ids == ("slot-0",)
        with pytest.raises(CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
    finally:
        executor.close()


def test_leaf_exception_retires_slot(tmp_path: Path) -> None:
    group, allocation = _group(("one", "two")), _allocation(1)

    def runner(_payload: bytes) -> bytes:
        raise RuntimeError("leaf failed")

    executor = _executor(group, allocation, runner)
    try:
        handle = executor.submit(*_invocation(group, allocation, "one", tmp_path))[1]
        with pytest.raises(RuntimeError, match="leaf failed"):
            executor.collect(handle, timeout=2)
        assert executor.retired_slot_ids == ("slot-0",)
        with pytest.raises(CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
    finally:
        executor.close()


def test_confirmed_reap_recycles_slot(tmp_path: Path) -> None:
    group, allocation = _group(("one", "two")), _allocation(1)
    executor = _executor(group, allocation, lambda payload: _failure_result(LeafRequest.decode(payload)).encode())
    try:
        first = executor.submit(*_invocation(group, allocation, "one", tmp_path))[1]
        executor.collect(first, timeout=2)
        second = executor.submit(*_invocation(group, allocation, "two", tmp_path))[1]
        assert second.slot_id == first.slot_id == "slot-0"
        executor.collect(second, timeout=2)
        assert executor.retired_slot_ids == ()
    finally:
        executor.close()


def test_one_retired_slot_does_not_block_other_slot(tmp_path: Path) -> None:
    group, allocation = _group(("one", "two", "three")), _allocation(2)

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        if request.task_id == "one":
            result = _failure_result(request)
            return replace(result, teardown={**result.teardown, "child_group_reaped": False}).encode()
        return _failure_result(request).encode()

    executor = _executor(group, allocation, runner)
    try:
        first = executor.submit(*_invocation(group, allocation, "one", tmp_path))[1]
        with pytest.raises(LeafResultRejected):
            executor.collect(first, timeout=2)
        second = executor.submit(*_invocation(group, allocation, "two", tmp_path))[1]
        executor.collect(second, timeout=2)
        assert second.slot_id != first.slot_id
        with pytest.raises(CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "three", tmp_path))
    finally:
        executor.close()


def test_interrupted_wait_keeps_lease(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    group, allocation = _group(("one", "two")), _allocation(1)
    captured: list[LeafRequest] = []

    class InterruptOnce(Future[Any]):
        interrupted = False

        def result(self, timeout: float | None = None) -> Any:
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt
            return super().result(timeout=timeout)

    future = InterruptOnce()
    executor = _executor(group, allocation, lambda payload: b"")

    def submit(_function: Any, request: LeafRequest) -> Future[Any]:
        captured.append(request)
        future.set_result(_failure_result(request))
        return future

    monkeypatch.setattr(executor._pool, "submit", submit)
    try:
        handle = executor.submit(*_invocation(group, allocation, "one", tmp_path))[1]
        with pytest.raises(KeyboardInterrupt):
            executor.collect(handle, timeout=2)
        with pytest.raises(CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
        executor.collect(handle, timeout=2)
    finally:
        executor.close()


def test_collection_wait_timeout_keeps_lease(tmp_path: Path) -> None:
    group, allocation = _group(("one", "two")), _allocation(1)
    release = Event()

    def runner(payload: bytes) -> bytes:
        release.wait(2)
        return _failure_result(LeafRequest.decode(payload)).encode()

    executor = _executor(group, allocation, runner)
    try:
        handle = executor.submit(*_invocation(group, allocation, "one", tmp_path))[1]
        with pytest.raises(CollectionWaitTimeout):
            executor.collect(handle, timeout=0.01)
        with pytest.raises(CapacityExceededError):
            executor.submit(*_invocation(group, allocation, "two", tmp_path))
        release.set()
        executor.collect(handle, timeout=2)
    finally:
        release.set()
        executor.close()


def test_retired_slot_ids_is_read_only_snapshot(tmp_path: Path) -> None:
    group, allocation = _group(("one",)), _allocation(1)

    def runner(payload: bytes) -> bytes:
        request = LeafRequest.decode(payload)
        result = _failure_result(request)
        return replace(result, teardown={**result.teardown, "child_group_reaped": False}).encode()

    executor = _executor(group, allocation, runner)
    try:
        handle = executor.submit(*_invocation(group, allocation, "one", tmp_path))[1]
        with pytest.raises(LeafResultRejected):
            executor.collect(handle, timeout=2)
        snapshot = executor.retired_slot_ids
        assert isinstance(snapshot, tuple)
        with pytest.raises(AttributeError):
            snapshot.append("slot-0")  # type: ignore[attr-defined]
        assert executor.retired_slot_ids == ("slot-0",)
    finally:
        executor.close()
