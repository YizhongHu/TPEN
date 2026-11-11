"""S3 Layer C bounded remote teardown and quiescence contract tests."""

from __future__ import annotations

from concurrent.futures import Future
import os
from pathlib import Path
import sys
import threading

import pytest

import time

from tpen_executor.dispatch.parsl import _RemoteLeaf
from tpen_executor.dispatch.transport import LeafResultRejected, LeafTeardownReceipt, validate_leaf_result
from tpen_executor.workers import SubprocessWorker

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


def test_delayed_ack_inside_bound_is_accepted(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    handle = _RemoteLeaf(request)
    future: Future[bytes] = Future()
    handle.bind_future(future)
    future.set_result(_synthetic_result(request).encode())
    assert handle.wait_for_teardown(0.1) is True


def test_delayed_ack_outside_bound_is_remote_quiescence_unconfirmed(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    handle = _RemoteLeaf(request)
    future: Future[bytes] = Future()
    handle.bind_future(future)
    assert handle.wait_for_teardown(0.001) is False


def test_worker_loss_is_remote_quiescence_unconfirmed(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    handle = _RemoteLeaf(request)
    future: Future[bytes] = Future()
    handle.bind_future(future)
    future.set_exception(RuntimeError("worker lost"))
    assert handle.wait_for_teardown(0.1) is False


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


def test_cancel_during_science_reaps_local_child_group(tmp_path: Path) -> None:
    request, _ = _request(tmp_path, timeout=5)
    output, _core, stdout, stderr, _ev_stdout, _ev_stderr = request.paths()
    result: list = []

    def run_child() -> None:
        result.append(SubprocessWorker().run(
            request.environment, ("-c", "import time; time.sleep(60)"),
            cwd=output, stdout_path=stdout, stderr_path=stderr, timeout=5,
            cancel_path=Path(output) / "cancel.json",
            cancel_identity={"schema": "tpen-executor/leaf-cancel-v1", "group_digest": request.group_digest,
                             "task_id": request.task_id, "attempt_id": request.attempt_id,
                             "allocation_id": request.allocation_id, "backend_job_id": request.backend_job_id},
        ))

    thread = threading.Thread(target=run_child)
    thread.start()
    time.sleep(0.05)
    _RemoteLeaf(request).request_cancel()
    thread.join(2)
    assert not thread.is_alive() and result and result[0].pid is not None
    if sys.platform.startswith("linux"):
        assert not Path(f"/proc/{result[0].pid}").exists()


def test_foreign_identity_cancel_marker_does_not_cancel_leaf(tmp_path: Path) -> None:
    request, _ = _request(tmp_path, timeout=2)
    marker = Path(request.output_dir) / "cancel.json"
    marker.write_text('{"schema":"tpen-executor/leaf-cancel-v1","task_id":"foreign"}')
    output, _core, stdout, stderr, _ev_stdout, _ev_stderr = request.paths()
    outcome = SubprocessWorker().run(
        request.environment, ("-c", "print('alive')"), cwd=output,
        stdout_path=stdout, stderr_path=stderr, timeout=2,
        cancel_path=marker, cancel_identity={"schema": "tpen-executor/leaf-cancel-v1", "task_id": request.task_id},
    )
    assert outcome.returncode == 0


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

