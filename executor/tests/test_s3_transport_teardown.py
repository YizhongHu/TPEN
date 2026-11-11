"""S3 Layer C bounded remote teardown and quiescence contract tests."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

import pytest

import time

from tpen_executor.dispatch.parsl import _RemoteLeaf
from tpen_executor.dispatch.transport import LeafResultRejected, LeafTeardownReceipt, validate_leaf_result

from test_s3_transport_wire import _request, _synthetic_result


def test_leaf_teardown_receipt_is_versioned_and_bound_to_all_identities(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    receipt = LeafTeardownReceipt.for_request(request, inspection="not-run", child_group_reaped=True)
    assert receipt.to_dict()["schema"] == "tpen-executor/leaf-teardown-v1"
    assert receipt.to_dict()["child_group_reaped"] is True

    result = _synthetic_result(request, teardown=receipt.to_dict())
    assert validate_leaf_result(request, result) is result
    crossed = dict(receipt.to_dict(), attempt_id="crossed-attempt")
    with pytest.raises(ValueError, match="invalid leaf teardown facts"):
        LeafTeardownReceipt.from_mapping(crossed)


def test_remote_leaf_wait_accepts_only_leaf_ack(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    handle = _RemoteLeaf(request)
    future: Future[bytes] = Future()
    handle.bind_future(future)
    future.set_result(_synthetic_result(request).encode())
    assert handle.wait_for_teardown(0.1) is True


@pytest.mark.parametrize("field", ["task_id", "attempt_id", "allocation_id"], ids=["task", "attempt", "allocation"])
def test_remote_leaf_rejects_well_formed_crossed_reply_as_no_quiescence_ack(tmp_path: Path, field: str) -> None:
    request, _ = _request(tmp_path)
    handle = _RemoteLeaf(request)
    future: Future[bytes] = Future()
    handle.bind_future(future)
    crossed = _synthetic_result(request, **{field: "crossed-" + field})
    with pytest.raises(LeafResultRejected, match=field):
        validate_leaf_result(request, crossed)
    future.set_result(crossed.encode())
    assert handle.wait_for_teardown(0.1) is False


def test_remote_leaf_request_cancel_creates_identity_bound_marker(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    handle = _RemoteLeaf(request)
    handle.request_cancel()
    marker = (Path(request.output_dir) / "cancel.json").read_text()
    assert request.task_id in marker and request.attempt_id in marker


def test_remote_leaf_wait_uses_one_shared_deadline(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    handles = []
    for _ in range(2):
        handle = _RemoteLeaf(request)
        future: Future[bytes] = Future()
        handle.bind_future(future)
        handles.append(handle)
    deadline = time.monotonic() + 0.05
    for handle in handles:
        handle.wait_for_teardown(max(deadline - time.monotonic(), 0))
    assert time.monotonic() - (deadline - 0.05) < 0.09


def test_remote_leaf_never_signals_a_remote_process_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, _ = _request(tmp_path)
    handle = _RemoteLeaf(request)
    calls: list[tuple] = []
    monkeypatch.setattr("tpen_executor.dispatch.parsl.os.killpg", lambda *args: calls.append(args))
    handle.request_cancel()
    assert calls == []

