"""S3 Layer C bounded remote teardown and quiescence contract tests.

``S3C_MUTANT_TABLE`` lists the contract mutants M1-M8 for bounded remote
teardown and quiescence.  M1-M4 and M8 live inside
``ParslDispatcher._cancel_slot_pool``/``_run_slot_pool`` itself: their
killers must drive real cancellation/loss through ``ParslDispatcher`` with
a transport double, not call ``_RemoteLeaf`` methods directly (a prior
round's killers did the latter and could never observe these mutants).
M8 (review B1) is the global-admission-closure mutant.
"""

from __future__ import annotations

import ast
from concurrent.futures import Future
import dataclasses
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

import time

from tpen_executor.contracts import AllocationId, BackendJobId
from tpen_executor.dispatch import parsl as parsl_dispatch
from tpen_executor.dispatch.parsl import _RemoteLeaf
from tpen_executor.dispatch.transport import (
    LeafRequest,
    LeafResult,
    LeafResultRejected,
    LeafTeardownReceipt,
    run_leaf,
    validate_leaf_result,
)
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.placement import AllocationSlot
from tpen_executor.storage import DirectoryStore
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.workers import SubprocessWorker
from tpen_executor import workers as workers_module

from test_s3_transport_parsl import BINDING, _Parsl, _group, _lifecycle, _runs, _transport, _worker, fake_parsl
from test_s3_transport_wire import _request, _synthetic_result


PARSL_MODULE = "executor/src/tpen_executor/dispatch/parsl.py"
WORKERS_MODULE = "executor/src/tpen_executor/workers.py"


S3C_MUTANT_TABLE = (
    {
        "id": "M1", "target": PARSL_MODULE,
        "old": "                record[\"handle\"].request_cancel()\n",
        "new": "                os.killpg(os.getpid(), 0)\n",
        "killer": "test_dispatcher_cancel_never_signals_a_remote_process_group",
        "suite": "executor", "assertion": "the controller cancels a remote leaf only by requesting its marker; it never killpgs a remote process itself",
    },
    {
        "id": "M2", "target": PARSL_MODULE,
        "old": "                    acknowledged = handle.wait_for_teardown(max(remote_deadline - time.monotonic(), 0))\n",
        "new": "                    acknowledged = True\n",
        "killer": "test_dispatcher_cancel_with_no_ack_is_remote_quiescence_unconfirmed",
        "suite": "executor", "assertion": "status is decided only after actually waiting for the ack, never published as settled without any wait at all",
    },
    {
        "id": "M3", "target": PARSL_MODULE,
        "old": "            self._cancel_slot_pool(active, lifecycle, cancellation)\n            raise\n",
        "new": (
            "            if self._active_dfk is not None:\n"
            "                self._active_dfk.cleanup()\n"
            "            self._cancel_slot_pool(active, lifecycle, cancellation)\n"
            "            raise\n"
        ),
        "killer": "test_dispatcher_cancel_settles_every_leaf_before_dfk_cleanup",
        "suite": "executor", "assertion": "DFK cleanup never starts before every live remote leaf (ack inside the bound or bounded-unconfirmed) is settled; proven by event-controlled finish/cleanup ordering, not a wall-clock bound that an early-but-still-waiting cleanup could still pass",
    },
    {
        "id": "M4", "target": PARSL_MODULE,
        "old": (
            "                        remote_unconfirmed = True\n"
            "                        admission_closed = True\n"
            "                        try:\n"
            "                            lifecycle.finish(record[\"fence\"], \"remote-quiescence-unconfirmed\", self._utc_now(), None)\n"
        ),
        "new": (
            "                        remote_unconfirmed = True\n"
            "                        admission_closed = True\n"
            "                        try:\n"
            "                            lifecycle.finish(record[\"fence\"], \"completed\", self._utc_now(), None)\n"
        ),
        "killer": "test_dispatcher_terminal_remote_failure_never_refills_slot_or_publishes_completed",
        "suite": "executor", "assertion": "a terminal worker-loss/crossed-reply future during ordinary polling (no owner cancellation) is never published completed",
    },
    {
        "id": "M5", "target": PARSL_MODULE,
        "old": "            result = validate_leaf_result(self.request, LeafResult.decode(payload), cancellation_expected=True)\n",
        "new": "            result = LeafResult.decode(payload)\n",
        "killer": "test_remote_leaf_rejects_well_formed_crossed_reply_as_no_quiescence_ack",
        "suite": "executor", "assertion": "a stale task/attempt/allocation reply is rejected on identity; it is never accepted as a quiescence ack merely because it decodes",
    },
    {
        "id": "M6", "target": WORKERS_MODULE,
        "old": "                    watcher.start()\n                    watcher_started = True\n",
        "new": "                    watcher_started = False\n",
        "killer": "test_cancel_during_science_reaps_local_child_group",
        "suite": "executor", "assertion": "the leaf actually watches for its cancel marker; a leaf that never does so ignores cancellation and keeps running past the bound",
    },
    {
        "id": "M7", "target": PARSL_MODULE,
        "old": "                    acknowledged = handle.wait_for_teardown(max(remote_deadline - time.monotonic(), 0))\n",
        "new": "                    acknowledged = handle.wait_for_teardown(float(getattr(self.transport, \"teardown_timeout\", 30)))\n",
        "killer": "test_dispatcher_cancel_waits_one_shared_deadline_across_remote_leaves",
        "suite": "executor", "assertion": "every live remote leaf waits against one shared deadline, never its own fresh bound (timing control, kept per review alongside the M3 ordering control)",
    },
    {
        "id": "M8", "target": PARSL_MODULE,
        "old": "                        self._cancel_slot_pool(active, lifecycle, error)\n                        active.clear()\n",
        "new": "                        pass\n",
        "killer": "test_dispatcher_remote_loss_closes_admission_globally",
        "suite": "executor", "assertion": "a remote loss or crossed reply bounded-settles every other still-live record immediately, before anything else runs; without the sweep a sibling leaf would resolve only through the ordinary path later and be published completed instead of failed (review B1)",
    },
)


def test_s3c_mutant_table_targets_unique_executable_anchors() -> None:
    """Every Layer C mutant names a unique executable anchor, parses, and has a killer."""

    root = Path(__file__).parents[2]
    module = Path(__file__).read_text(encoding="utf-8")
    assert [row["id"] for row in S3C_MUTANT_TABLE] == ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    for row in S3C_MUTANT_TABLE:
        source = (root / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1, row
        assert row["old"] != row["new"] and row["suite"] == "executor" and row["assertion"]
        assert not row["old"].strip().startswith(("#", '"""', "'"))
        ast.parse(source.replace(row["old"], row["new"]))
        assert f"def {row['killer'].split('[')[0]}(" in module, row


def test_leaf_teardown_receipt_is_versioned_and_bound_to_all_identities(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    receipt = LeafTeardownReceipt.for_request(request, inspection="not-run", child_group_reaped=True)
    assert receipt.to_dict()["schema"] == "tpen-executor/leaf-teardown-v1"
    assert receipt.to_dict()["child_group_reaped"] is True

    result = _synthetic_result(request, teardown=receipt.to_dict())
    assert validate_leaf_result(request, result) is result
    # A well-formed but cross-attempt teardown receipt is rejected by the
    # controller's identity check, not merely by LeafTeardownReceipt's own
    # (purely syntactic) construction gate.
    crossed = dataclasses.replace(result, teardown={**receipt.to_dict(), "attempt_id": "crossed-attempt"})
    with pytest.raises(LeafResultRejected, match="teardown"):
        validate_leaf_result(request, crossed)


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
    # validate_leaf_result's rejection message names the short check (e.g.
    # "task"), not the LeafResult field name (e.g. "task_id").
    with pytest.raises(LeafResultRejected, match=field.removesuffix("_id")):
        validate_leaf_result(request, crossed)
    future.set_result(crossed.encode())
    assert handle.wait_for_teardown(0.1) is False


def test_remote_leaf_request_cancel_creates_identity_bound_marker(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    handle = _RemoteLeaf(request)
    handle.request_cancel()
    marker = (Path(request.output_dir) / "cancel.json").read_text()
    assert request.task_id in marker and request.attempt_id in marker


def test_cancel_during_science_reaps_local_child_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, _ = _request(tmp_path, timeout=5)
    output, _core, stdout, stderr, _ev_stdout, _ev_stderr = request.paths()
    result: list = []
    # Event-driven, not a fixed sleep (review P2/M6: a fixed sleep before
    # spawn can miss the anchor under load): wait for the child's own
    # evidence that it has actually started before requesting cancellation.
    started = Path(output) / "started.marker"
    owner_ident: dict[str, int] = {}
    events: list[tuple[int, object]] = []
    real_waitid = os.waitid
    real_killpg = os.killpg
    real_wait = subprocess.Popen.wait
    real_poll = subprocess.Popen.poll

    def waitid(id_type, pid, options):
        events.append((threading.get_ident(), "waitid"))
        return real_waitid(id_type, pid, options)

    def killpg(pid, sig):
        events.append((threading.get_ident(), ("killpg", sig)))
        return real_killpg(pid, sig)

    def wait(process, *args, **kwargs):
        result = real_wait(process, *args, **kwargs)
        events.append((threading.get_ident(), "reaped"))
        return result

    def poll(process):
        events.append((threading.get_ident(), "poll"))
        return real_poll(process)

    monkeypatch.setattr(os, "waitid", waitid)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    monkeypatch.setattr(subprocess.Popen, "poll", poll)

    def run_child() -> None:
        owner_ident["value"] = threading.get_ident()
        result.append(SubprocessWorker().run(
            request.environment,
            ("-c", f"import pathlib, time; pathlib.Path({str(started)!r}).write_text('1'); time.sleep(60)"),
            cwd=output, stdout_path=stdout, stderr_path=stderr, timeout=5,
            cancel_path=Path(output) / "cancel.json",
            cancel_identity={"schema": "tpen-executor/leaf-cancel-v1", "group_digest": request.group_digest,
                             "task_id": request.task_id, "attempt_id": request.attempt_id,
                             "allocation_id": request.allocation_id, "backend_job_id": request.backend_job_id},
        ))

    thread = threading.Thread(target=run_child)
    thread.start()
    deadline = time.monotonic() + 5
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists()
    _RemoteLeaf(request).request_cancel()
    thread.join(2)
    assert not thread.is_alive() and result and result[0].pid is not None
    assert all(kind != "poll" for _ident, kind in events)
    assert all(ident == owner_ident["value"] for ident, _kind in events)
    reap_index = next(index for index, (_ident, kind) in enumerate(events) if kind == "reaped")
    assert any(kind == ("killpg", signal.SIGKILL) for _ident, kind in events[:reap_index])
    assert all(kind not in (("killpg", signal.SIGKILL), ("killpg", 0)) for _ident, kind in events[reap_index + 1:])
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


def test_wait_for_teardown_accepts_a_real_cancelled_leafs_uninspected_ack(tmp_path: Path) -> None:
    """P2-4 killer: a real cancellation's uninspected teardown ack is accepted
    by the controller's own ack check, without weakening ordinary completion
    acceptance (real producer-to-controller bytes, not a synthetic result).
    """

    request, _parts = _request(tmp_path, timeout=5)
    handle = _RemoteLeaf(request)
    handle.request_cancel()
    # A real pre-spawn cancellation: run_leaf itself sees the marker already
    # in place (the leaf never distinguishes pre-spawn from during-science
    # cancellation at the result-shape level: both produce phases=("science",)
    # with inspection skipped).
    encoded = run_leaf(request.encode())
    result = LeafResult.decode(encoded)
    assert result.phases == ("science",) and result.inspection == "not-run"

    # Negative control: the ordinary completion path (no
    # cancellation_expected) still refuses to publish this as a
    # segment-worthy outcome -- it still requires inspection.
    with pytest.raises(LeafResultRejected, match="inspection"):
        validate_leaf_result(request, result)

    # The controller's own teardown-ack check accepts it: this exact
    # controller requested this exact cancellation.
    future: Future[bytes] = Future()
    handle.bind_future(future)
    future.set_result(encoded)
    assert handle.wait_for_teardown(1.0) is True



# ------------------------------------------------- dispatcher-level M1-M4


def _never_acknowledging_leaf(_index, _function, _args, _kwargs):
    """A remote leaf hook that never returns: the controller sees it as still live."""

    threading.Event().wait()


def _two_slot_lifecycle(tmp_path: Path) -> AttachedAllocation:
    slots = [AllocationSlot(f"slot-{index}", "node-a", (), (index,), (), ()).to_dict() for index in range(2)]
    facts = {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": slots}}
    root = tmp_path / "s3c-store"
    root.mkdir()
    return AttachedAllocation(
        ConfiguredAllocation(AllocationId("s3c-allocation"), BackendJobId("job-s3c"), facts), DirectoryStore(root),
    )


def _interrupt_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise KeyboardInterrupt on the controller's first poll, as a real owner cancellation would."""

    first = {"value": True}

    def interrupting_sleep(seconds: float) -> None:
        if first["value"]:
            first["value"] = False
            raise KeyboardInterrupt("S3C cancellation")
        time.sleep(seconds)

    monkeypatch.setattr("tpen_executor.dispatch.parsl._poll_sleep", interrupting_sleep)


def test_dispatcher_cancel_never_signals_a_remote_process_group(
    tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M1 killer: real cancellation through ParslDispatcher never killpgs a remote process.

    A prior round's killer (``test_remote_leaf_never_signals_a_remote_process_group``)
    calls ``_RemoteLeaf.request_cancel`` directly, which never had a killpg
    call to remove; it cannot observe a mutation inside
    ``ParslDispatcher._cancel_slot_pool``.  This one drives the real
    cancellation path.
    """

    fake_parsl.hooks["run_leaf"] = _never_acknowledging_leaf
    calls: list[tuple] = []
    monkeypatch.setattr("tpen_executor.dispatch.parsl.os.killpg", lambda *args: calls.append(args))
    _interrupt_once(monkeypatch)
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path), binding=BINDING, transport=_transport(teardown_timeout=0.1),
    )
    group = _group("s3c-cancel-m1", "member")
    lifecycle = _lifecycle(tmp_path)
    with pytest.raises(KeyboardInterrupt, match="S3C cancellation"):
        dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)
    assert calls == []
    attempt = lifecycle.store.manifest(group).attempts[0]
    assert attempt.outcome_receipt.status == "remote-quiescence-unconfirmed"


def test_dispatcher_cancel_with_no_ack_is_remote_quiescence_unconfirmed(
    tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M2/M4 killer: a live remote leaf that never acks is never published completed."""

    fake_parsl.hooks["run_leaf"] = _never_acknowledging_leaf
    _interrupt_once(monkeypatch)
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path), binding=BINDING, transport=_transport(teardown_timeout=0.1),
    )
    group = _group("s3c-cancel-m2m4", "member")
    lifecycle = _lifecycle(tmp_path)
    with pytest.raises(KeyboardInterrupt, match="S3C cancellation"):
        dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)
    manifest = lifecycle.store.manifest(group)
    attempt = manifest.attempts[0]
    assert attempt.outcome_receipt.status == "remote-quiescence-unconfirmed"
    assert manifest.segments == ()


def test_dispatcher_cancel_waits_one_shared_deadline_across_remote_leaves(
    tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M3 killer: two live remote leaves wait against one shared bound, not one each.

    This also bounds the DFK cleanup that follows ``_cancel_slot_pool`` in
    ``ParslDispatcher.run``'s ``finally``: cleanup cannot start until every
    live leaf's wait has resolved, so the one-shared-deadline property is
    exactly what keeps that cleanup from being delayed per live leaf.
    """

    fake_parsl.hooks["run_leaf"] = _never_acknowledging_leaf
    _interrupt_once(monkeypatch)
    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), transport=_transport(teardown_timeout=0.3))
    group = _group("s3c-cancel-m3", "first", "second")
    lifecycle = _two_slot_lifecycle(tmp_path)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt, match="S3C cancellation"):
        dispatcher.run(group, _runs(tmp_path, "first", "second"), lifecycle, timeout=60)
    elapsed = time.monotonic() - started
    # One shared 0.3s deadline for both live leaves costs ~0.3s total; a
    # per-record bound (the mutant) costs ~0.3s per remote record, ~0.6s here.
    assert elapsed < 0.5
    manifest = lifecycle.store.manifest(group)
    assert {attempt.outcome_receipt.status for attempt in manifest.attempts} == {"remote-quiescence-unconfirmed"}
    assert manifest.segments == ()


def test_dispatcher_cancel_settles_every_leaf_before_dfk_cleanup(
    tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M3 killer (restored, event-controlled ordering, not only a wall-clock
    bound): DFK cleanup must never start before every live remote leaf --
    whether it acks inside the shared bound or is never acknowledged -- is
    settled. An early ``dfk.cleanup()`` call followed by the same waits
    would still pass a wall-clock-only check; this one catches it by order.
    """

    ack_gate = threading.Event()

    def dispatching_hook(_index, _function, args, _kwargs):
        request = LeafRequest.decode(args[0])
        if request.task_id == "acking":
            # Review P2 (M3 determinism): check the wait's own return value
            # instead of ignoring it -- the gate is opened synchronously
            # from inside the interrupt below, so it is always set well
            # before this 5s bound; a False return here would mean the
            # test's own setup is broken, not a real scheduling race.
            assert ack_gate.wait(5) is True
            # A synthetic, already-valid result: real run_leaf would race
            # the marker request_cancel() writes against this hook's own
            # release timing (whether it sees the marker before/after
            # spawning science is nondeterministic and not what this test
            # is about -- only finish/cleanup ORDER is).
            return _synthetic_result(request).encode()
        threading.Event().wait()  # "stuck": never acks

    fake_parsl.hooks["run_leaf"] = dispatching_hook
    # A generous bound (not the tight 0.3s used elsewhere): this test's
    # oracle is event ORDER (finish before cleanup), never a wall-clock
    # measurement, so the bound only needs to comfortably outlast the
    # synthetic hook's own scheduling, not stay tight (review P2/M3).
    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), transport=_transport(teardown_timeout=1.0))
    group = _group("s3c-dfk-order", "acking", "stuck")
    lifecycle = _two_slot_lifecycle(tmp_path)

    # AttachedAllocation is a frozen record: patch the class method, not the
    # instance attribute (monkeypatch.setattr on a frozen instance raises
    # FrozenInstanceError both on apply and on its own teardown-time undo,
    # which leaves other patches in this fixture -- notably _poll_sleep --
    # stuck active for later tests).
    real_finish = AttachedAllocation.finish

    def spying_finish(self, fence, status, *args, **kwargs):
        fake_parsl.events.append(("finish", fence.task_id, status))
        return real_finish(self, fence, status, *args, **kwargs)

    monkeypatch.setattr(AttachedAllocation, "finish", spying_finish)

    # Open the ack gate from inside the interrupt itself -- synchronously,
    # the same call that raises -- rather than racing a timed background
    # thread against the interrupt. The main loop's current iteration is
    # already abandoned by the time this raises (the exception unwinds
    # straight to _cancel_slot_pool), so "acking" can never be settled by
    # the ordinary polling loop no matter how the background hook thread
    # is scheduled relative to this.
    first = {"value": True}

    def interrupting_sleep(seconds: float) -> None:
        if first["value"]:
            first["value"] = False
            ack_gate.set()
            raise KeyboardInterrupt("S3C cancellation")
        time.sleep(seconds)

    monkeypatch.setattr("tpen_executor.dispatch.parsl._poll_sleep", interrupting_sleep)

    with pytest.raises(KeyboardInterrupt, match="S3C cancellation"):
        dispatcher.run(group, _runs(tmp_path, "acking", "stuck"), lifecycle, timeout=60)

    finish_events = [event for event in fake_parsl.events if event[0] == "finish"]
    cleanup_events = [event for event in fake_parsl.events if event[0] == "cleanup"]
    assert len(finish_events) == 2 and len(cleanup_events) == 1
    # Both settlements -- the in-bound ack and the bounded-unconfirmed leaf
    # -- appear in event order strictly before DFK cleanup.
    assert max(fake_parsl.events.index(event) for event in finish_events) < fake_parsl.events.index(cleanup_events[0])
    assert {task_id: status for (_tag, task_id, status) in finish_events} == {
        "acking": "failed", "stuck": "remote-quiescence-unconfirmed",
    }


@pytest.mark.parametrize("case", ["worker-lost", "crossed-reply"])
def test_dispatcher_terminal_remote_failure_never_refills_slot_or_publishes_completed(
    tmp_path: Path, fake_parsl: _Parsl, case: str,
) -> None:
    """P1-2/M4 killer: a terminal remote future during ORDINARY polling (no
    owner cancellation at all) settles unconfirmed and never recycles its
    slot binding; a queued second member is never admitted to it.
    """

    def terminal_first(_index, function, args, kwargs):
        request = LeafRequest.decode(args[0])
        if request.task_id != "first":
            return function(*args, **kwargs)
        if case == "worker-lost":
            raise RuntimeError("worker lost")
        # A well-formed but crossed reply: real bytes that fail identity.
        result = LeafResult.decode(function(*args, **kwargs))
        return dataclasses.replace(result, attempt_id="another-attempt").encode()

    fake_parsl.hooks["run_leaf"] = terminal_first
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path), binding=BINDING, transport=_transport(teardown_timeout=0.1),
    )
    group = _group(f"s3c-worker-loss-{case}", "first", "second")
    lifecycle = _lifecycle(tmp_path)
    manifest = dispatcher.run(group, _runs(tmp_path, "first", "second"), lifecycle, timeout=60)
    assert [attempt.task_id for attempt in manifest.attempts] == ["first"]
    assert manifest.attempts[0].outcome_receipt.status == "remote-quiescence-unconfirmed"
    assert manifest.segments == ()
    assert fake_parsl.calls.get("run_leaf", 0) == 1


# ------------------------------------------------------- P1-1 residual


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="requires Linux /proc process-group visibility")
def test_group_quiescent_terminates_a_surviving_descendant(tmp_path: Path) -> None:
    """P1-1 residual killer: a same-group descendant that outlives its
    leader is TERMINATED by the worker's own group probe, not merely
    observed (the leaf owns the group it started with start_new_session).
    """

    request, _parts = _request(tmp_path, timeout=5)
    output, _core, stdout, stderr, _ev_stdout, _ev_stderr = request.paths()
    descendant_pid_file = Path(output) / "descendant.pid"
    script = (
        "import subprocess, sys\n"
        "child = subprocess.Popen(['sleep', '30'])\n"
        f"open({str(descendant_pid_file)!r}, 'w').write(str(child.pid))\n"
        "sys.exit(0)\n"
    )
    outcome = SubprocessWorker().run(
        request.environment, ("-c", script), cwd=output,
        stdout_path=stdout, stderr_path=stderr, timeout=5,
    )
    assert outcome.returncode == 0 and outcome.group_reaped is True
    descendant_pid = int(descendant_pid_file.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(descendant_pid, 0)


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="requires Linux /proc process-group visibility")
def test_real_normal_exit_group_quiescence_is_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request, _parts = _request(tmp_path, timeout=5)
    output, _core, stdout, stderr, _ev_stdout, _ev_stderr = request.paths()
    durations = []
    real_group_quiescent = workers_module._group_quiescent

    def timed_group_quiescent(*args, **kwargs):
        started = time.monotonic()
        result = real_group_quiescent(*args, **kwargs)
        durations.append(time.monotonic() - started)
        return result

    monkeypatch.setattr(workers_module, "_group_quiescent", timed_group_quiescent)
    outcome = SubprocessWorker().run(
        request.environment, ("-c", "pass"), cwd=output,
        stdout_path=stdout, stderr_path=stderr, timeout=5,
    )
    assert outcome.group_reaped is True
    assert durations and durations[0] < 0.5


@pytest.mark.parametrize(
    ("stat_text", "expected"),
    [
        ("123 (worker with spaces) S 1 77 0 0", 77),
        ("123 (worker ) with spaces) S 1 88 0 0", 88),
        ("123 S 1", None),
        ("123 (worker) S 1", None),
        ("123 (worker) S 1 nope", None),
    ],
    ids=["spaces", "right-paren", "missing-comm", "short-suffix", "non-int-pgrp"],
)
def test_proc_group_parser_handles_comm_text(stat_text: str, expected: int) -> None:
    assert workers_module._proc_group_from_stat(stat_text) == expected


def test_group_quiescence_kills_and_probes_before_leader_reap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The PGID kill is ordered after WNOWAIT and before any reap."""
    events = []
    pid = 1234

    def waitid(id_type, observed_pid, options):
        assert id_type == os.P_PID and observed_pid == pid
        events.append("waitid")
        return SimpleNamespace(si_pid=pid)

    def killpg(observed_pid, sig):
        assert observed_pid == pid
        events.append(("killpg", sig))

    monkeypatch.setattr(os, "waitid", waitid)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(workers_module, "_proc_group_has_members", lambda pgid, leader_pid: False)
    assert workers_module._leader_exited_without_reap(pid) is True
    assert workers_module._group_quiescent(pid) is True
    assert events == ["waitid", ("killpg", signal.SIGKILL)]


def test_group_quiescence_keeps_quarantine_when_proc_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(workers_module, "_proc_group_has_members", lambda pgid, leader_pid: None)
    assert workers_module._group_quiescent(1234) is False
    assert signals == [(1234, signal.SIGKILL)]


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="requires Linux /proc process-group visibility")
def test_normal_exit_keeps_quiescence_before_reap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Normal completion must not signal a PGID after its leader is reaped."""
    events = []
    request, _parts = _request(tmp_path, timeout=5)
    output, _core, stdout, stderr, _ev_stdout, _ev_stderr = request.paths()
    real_waitid = os.waitid
    real_killpg = os.killpg
    real_wait = subprocess.Popen.wait

    def waitid(id_type, pid, options):
        events.append("waitid")
        return real_waitid(id_type, pid, options)

    def killpg(pid, sig):
        events.append(("killpg", sig))
        return real_killpg(pid, sig)

    def wait(process, *args, **kwargs):
        result = real_wait(process, *args, **kwargs)
        events.append("reaped")
        return result

    monkeypatch.setattr(os, "waitid", waitid)
    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    outcome = SubprocessWorker().run(
        request.environment, ("-c", "pass"), cwd=output,
        stdout_path=stdout, stderr_path=stderr, timeout=5,
    )
    reap_index = events.index("reaped")
    assert outcome.group_reaped is True
    assert any(event == ("killpg", signal.SIGKILL) for event in events[:reap_index])
    assert all(event not in (("killpg", signal.SIGKILL), ("killpg", 0)) for event in events[reap_index + 1:])


def test_normal_observation_childprocesserror_is_uncertain_without_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An already-reaped leader returns an uncertain outcome, not a crash."""
    class AlreadyReaped:
        pid = 1234
        returncode = -signal.SIGKILL

        def wait(self, *args, **kwargs):
            raise AssertionError("already-reaped child must not be waited again")

    child = AlreadyReaped()
    signals = []

    def already_reaped(*args, **kwargs):
        raise ChildProcessError()

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(workers_module, "_leader_exited_without_reap", already_reaped)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    outcome = SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable), ("-c", "pass"), cwd=tmp_path,
        stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", timeout=5,
    )
    assert outcome.returncode == child.returncode
    assert outcome.group_reaped is False
    assert signals == []


def test_group_quiescence_skips_signal_after_unconfirmed_reap(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unverified post-reap PID is conservatively not group-quiescent."""
    signals = []
    monkeypatch.setattr(workers_module, "_proc_start_ticks", lambda pid: None)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    assert workers_module._group_quiescent(1234, leader_reaped=True, spawned_ticks=77) is False
    assert signals == []


def test_watcher_start_failure_reaps_the_child_as_an_uncertain_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1-1 residual killer: a watcher-thread start failure still reaps the
    spawned child (an uncertain exit, never a silently-true ack) -- the
    exception propagates and the child is confirmed dead.
    """

    request, _parts = _request(tmp_path, timeout=5)
    output, _core, stdout, stderr, _ev_stdout, _ev_stderr = request.paths()
    captured: dict[str, int] = {}
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        captured["pid"] = process.pid
        return process

    monkeypatch.setattr("tpen_executor.workers.subprocess.Popen", recording_popen)

    def failing_start(self):
        raise RuntimeError("thread start refused")

    monkeypatch.setattr(threading.Thread, "start", failing_start)

    with pytest.raises(RuntimeError, match="thread start refused"):
        SubprocessWorker().run(
            request.environment, ("-c", "import time; time.sleep(30)"), cwd=output,
            stdout_path=stdout, stderr_path=stderr, timeout=5,
            cancel_path=Path(output) / "cancel.json",
            cancel_identity={"schema": "tpen-executor/leaf-cancel-v1", "group_digest": request.group_digest,
                             "task_id": request.task_id, "attempt_id": request.attempt_id,
                             "allocation_id": request.allocation_id, "backend_job_id": request.backend_job_id},
        )
    assert "pid" in captured
    with pytest.raises(ProcessLookupError):
        os.kill(captured["pid"], 0)


# ------------------------------------------------------------ P2-4 gap


def test_wait_for_teardown_accepts_a_real_during_science_cancellation_ack(tmp_path: Path) -> None:
    """P2-4 gap killer: a real DURING-SCIENCE cancellation (not only a
    pre-spawn one) also produces an uninspected teardown ack that the
    controller's own check accepts, using real run_leaf producer bytes.
    """

    request, _parts = _request(tmp_path, timeout=5, mode="sleep")
    handle = _RemoteLeaf(request)
    encoded: list[bytes] = []

    def run_in_thread() -> None:
        encoded.append(run_leaf(request.encode()))

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    # The fake core's "sleep" mode blocks for 4s with no earlier file
    # write to poll; 0.3s is ample margin for the science child to exist
    # and be mid-sleep before cancelling.
    time.sleep(0.3)
    handle.request_cancel()
    thread.join(10)
    assert not thread.is_alive()
    [payload] = encoded
    result = LeafResult.decode(payload)
    assert result.phases == ("science",) and result.inspection == "not-run"

    with pytest.raises(LeafResultRejected, match="inspection"):
        validate_leaf_result(request, result)

    future: Future[bytes] = Future()
    handle.bind_future(future)
    future.set_result(payload)
    assert handle.wait_for_teardown(1.0) is True


# ------------------------------------------------------------------ B1


def test_dispatcher_remote_loss_closes_admission_globally(
    tmp_path: Path, fake_parsl: _Parsl,
) -> None:
    """B1 killer: a worker loss on one of two active slots closes
    admission for the WHOLE run, not just its own slot: a healthy
    completion on the other slot still settles (through the cancellation
    sweep, never published completed), but a third queued member is
    never admitted.
    """

    calls: list[str] = []

    def hook(_index, function, args, kwargs):
        request = LeafRequest.decode(args[0])
        calls.append(request.task_id)
        if request.task_id == "A":
            raise RuntimeError("worker lost")
        # "B": a deliberate delay so A's near-instant failure is reliably
        # processed first (the main loop polls every 0.01s).
        time.sleep(0.2)
        return _synthetic_result(request).encode()

    fake_parsl.hooks["run_leaf"] = hook
    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), transport=_transport(teardown_timeout=1.0))
    group = _group("s3c-b1-admission", "A", "B", "C")
    lifecycle = _two_slot_lifecycle(tmp_path)
    manifest = dispatcher.run(group, _runs(tmp_path, "A", "B", "C"), lifecycle, timeout=60)

    assert "C" not in calls
    assert "C" not in [attempt.task_id for attempt in manifest.attempts]
    statuses = {attempt.task_id: attempt.outcome_receipt.status for attempt in manifest.attempts}
    assert statuses["A"] == "remote-quiescence-unconfirmed"
    assert statuses["B"] == "failed"
    assert manifest.segments == ()


def test_cancel_slot_pool_isolates_one_records_request_cancel_failure(
    tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P1-3 killer: one record's injected request_cancel() failure must
    not stop cancellation and settlement of every other live record under
    the same shared deadline.
    """

    real_request_cancel = parsl_dispatch._RemoteLeaf.request_cancel

    def failing_request_cancel(self):
        if self.request.task_id == "A":
            raise OSError("injected marker write failure")
        return real_request_cancel(self)

    monkeypatch.setattr(parsl_dispatch._RemoteLeaf, "request_cancel", failing_request_cancel)

    b_gate = threading.Event()

    def dispatching_hook(_index, _function, args, _kwargs):
        request = LeafRequest.decode(args[0])
        if request.task_id == "B":
            b_gate.wait(5)
            return _synthetic_result(request).encode()
        threading.Event().wait()  # "A": never acks; its own cancel request also fails

    fake_parsl.hooks["run_leaf"] = dispatching_hook

    first = {"value": True}

    def interrupting_sleep(seconds: float) -> None:
        if first["value"]:
            first["value"] = False
            b_gate.set()
            raise KeyboardInterrupt("S3C cancellation")
        time.sleep(seconds)

    monkeypatch.setattr("tpen_executor.dispatch.parsl._poll_sleep", interrupting_sleep)

    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), transport=_transport(teardown_timeout=1.0))
    group = _group("s3c-p1-3-isolation", "A", "B")
    lifecycle = _two_slot_lifecycle(tmp_path)
    with pytest.raises(KeyboardInterrupt, match="S3C cancellation"):
        dispatcher.run(group, _runs(tmp_path, "A", "B"), lifecycle, timeout=60)
    manifest = lifecycle.store.manifest(group)
    statuses = {attempt.task_id: attempt.outcome_receipt.status for attempt in manifest.attempts}
    assert statuses["A"] == "remote-quiescence-unconfirmed"
    assert statuses["B"] == "failed"
    assert manifest.segments == ()
