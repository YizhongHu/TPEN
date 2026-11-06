"""P7 r3 behavior probes through real DFK/python_app and child processes."""
import json
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from tpen_executor.dispatch.contract import DispatcherContract, _ACTIVE_DISPATCH
from tpen_executor.dispatch.parsl import ParslDispatcher
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.storage import DirectoryStore
from test_p7_parsl_dispatcher import _allocation, _group, _runs, _make_stub_package, _worker_spec


PARSL_REQUIRED = pytest.mark.skipif(
    importlib.util.find_spec("parsl") is None,
    reason="requires the approved executor parsl extra",
)


def dispatcher_for(tmp_path):
    package = _make_stub_package(tmp_path)
    worker = _worker_spec(PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)}))
    return ParslDispatcher(worker, binding={"environment": {"CUDA_VISIBLE_DEVICES": "2"}})


def lifecycle_for(tmp_path, name="store"):
    return AttachedAllocation(_allocation(), DirectoryStore(tmp_path / name))


@pytest.mark.parametrize("repeat_interrupt", [False, True], ids=["one-interrupt", "repeated-interrupt"])
@PARSL_REQUIRED
def test_r3_t1_cancel_before_spawn_refuses_child_and_waits_for_app(tmp_path, monkeypatch, repeat_interrupt):
    """Cancellation requested before worker launch must prevent a new child.

    Both cases use actual DFK, python_app, AppFuture, and SubprocessWorker.
    The repeated case also requires quiescence after a second owner interrupt.
    """
    import tpen_executor.dispatch.parsl as module
    from parsl.dataflow.futures import AppFuture
    dispatcher = dispatcher_for(tmp_path)
    ready, release, terminal = threading.Event(), threading.Event(), threading.Event()
    futures, processes, finishes = [], [], []
    original_child = module._run_parsl_child
    original_attach = module._ChildProcessHandle.attach
    original_result = AppFuture.result
    original_factory = dispatcher._python_app_factory
    original_kill = module._ChildProcessHandle.request_kill
    kill_seen = threading.Event()
    owner = threading.get_ident()
    original_interrupt = KeyboardInterrupt("review first owner interruption")
    second_interrupt = SystemExit("review second owner interruption")
    interruptions = 0

    def child(*args, **kwargs):
        ready.set()
        assert release.wait(5), "review launch gate did not open"
        return original_child(*args, **kwargs)

    def attach(handle, process):
        processes.append((process, kill_seen.is_set(), terminal.is_set()))
        return original_attach(handle, process)

    def request_kill(handle):
        kill_seen.set()
        return original_kill(handle)

    def factory(**kwargs):
        decorate = original_factory(**kwargs)
        def decorator(func):
            app = decorate(func)
            def launch(*args, **kw):
                future = app(*args, **kw)
                assert isinstance(future, AppFuture)
                futures.append(future)
                return future
            return launch
        return decorator

    def interrupted_result(future, timeout=None):
        nonlocal interruptions
        if threading.get_ident() == owner and futures and future is futures[0]:
            if interruptions == 0:
                assert ready.wait(5), "worker did not reach pre-spawn gate"
                interruptions += 1
                timer.start()
                raise original_interrupt
            if repeat_interrupt and interruptions == 1:
                interruptions += 1
                raise second_interrupt
        return original_result(future, timeout)

    class Lifecycle(AttachedAllocation):
        def finish(self, fence, status, finished_at_utc, result_digest=None):
            finishes.append((fence.task_id, status, futures[0].done()))
            published = super().finish(fence, status, finished_at_utc, result_digest)
            terminal.set()
            release.set()
            return published

    monkeypatch.setattr(module, "_run_parsl_child", child)
    monkeypatch.setattr(module._ChildProcessHandle, "attach", attach)
    monkeypatch.setattr(module._ChildProcessHandle, "request_kill", request_kill)
    monkeypatch.setattr(AppFuture, "result", interrupted_result)
    object.__setattr__(dispatcher, "_python_app_factory", factory)
    timer = threading.Timer(1.0, release.set)
    lifecycle = Lifecycle(_allocation(), DirectoryStore(tmp_path / "store"))
    try:
        with pytest.raises(KeyboardInterrupt) as observed:
            dispatcher.run(_group("a", "b"), _runs(tmp_path, "a", "b", modes={"a": "sleep"}), lifecycle, timeout=2)
        assert observed.value is original_interrupt
    finally:
        release.set()
        timer.cancel()
        if timer.ident is not None:
            timer.join()
    observation = {"repeat": repeat_interrupt, "interruptions": interruptions,
        "finishes": finishes, "processes": [(p.pid, p.returncode, after_kill, after_terminal)
            for p, after_kill, after_terminal in processes]}
    print(json.dumps(observation))
    (tmp_path / "review-observation.json").write_text(json.dumps(observation))
    assert len(futures) == 1 and kill_seen.is_set()
    assert all(p.poll() is not None for p, _, _ in processes), "owned leader not reaped"
    assert len(finishes) == 1 and finishes[0][:2] == ("a", "failed")
    assert finishes[0][2], "attempt finished before its real app quiesced after repeated interruption"
    assert not processes, "child was spawned despite cancellation already requested before launch"


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt], ids=["ordinary", "base-exception"])
@PARSL_REQUIRED
def test_r3_t2_cleanup_error_releases_active_state_and_dispatch_guard(tmp_path, monkeypatch, error_type):
    """A real DFK cleanup failure must not poison later dispatch admission."""
    dispatcher = dispatcher_for(tmp_path)
    real_factory = dispatcher._dfk_type
    created = []
    failure = error_type("review DFK cleanup error")
    saved = _ACTIVE_DISPATCH.set(False)
    observations = {}

    def make_dfk(**kwargs):
        dfk = real_factory(**kwargs)
        cleanup = dfk.cleanup
        created.append(dfk)
        def failing_cleanup():
            cleanup()
            if dfk is created[0]:
                raise failure
        dfk.cleanup = failing_cleanup
        return dfk

    # Isolate backend lifecycle from science, retaining actual DFK ownership.
    monkeypatch.setattr(DispatcherContract, "_run_admitted", lambda *a, **k: "review-manifest")
    object.__setattr__(dispatcher, "_dfk_type", make_dfk)
    try:
        with pytest.raises(error_type) as observed:
            dispatcher.run(_group("a"), _runs(tmp_path, "a"), lifecycle_for(tmp_path))
        assert observed.value is failure
        observations.update(active_dfk=dispatcher._active_dfk is not None,
            active_binding=dispatcher._active_binding is not None, guard=_ACTIVE_DISPATCH.get())
        second_error = None
        try:
            second = dispatcher.run(_group("b"), _runs(tmp_path, "b"), lifecycle_for(tmp_path, "second"))
        except BaseException as error:
            second_error = str(error)
            second = None
        observations.update(second_error=second_error, second=second)
        print(json.dumps(observations))
        assert observations == {"active_dfk": False, "active_binding": False, "guard": False,
            "second_error": None, "second": "review-manifest"}
    finally:
        _ACTIVE_DISPATCH.reset(saved)


@PARSL_REQUIRED
def test_r3_t3_public_dispatch_on_supported_python(tmp_path):
    """Valid public dispatch works under every declared supported interpreter."""
    manifest = dispatcher_for(tmp_path).run(_group("a"), _runs(tmp_path, "a"), lifecycle_for(tmp_path))
    assert manifest.attempts[0].outcome_receipt.status == "completed"


@PARSL_REQUIRED
def test_r3_t4_guard_covers_backend_initialization_without_nested_artifacts(tmp_path):
    """GREEN: reentry during real backend initialization is refused early."""
    dispatcher = dispatcher_for(tmp_path)
    real_factory = dispatcher._dfk_type
    nested_root = tmp_path / "nested-store"
    def make_dfk(**kwargs):
        with pytest.raises(RuntimeError, match="not reentrant"):
            dispatcher.run(_group("nested"), _runs(tmp_path, "nested"),
                AttachedAllocation(_allocation(), DirectoryStore(nested_root)))
        assert not nested_root.exists()
        return real_factory(**kwargs)
    object.__setattr__(dispatcher, "_dfk_type", make_dfk)
    manifest = dispatcher.run(_group("a"), _runs(tmp_path, "a"), lifecycle_for(tmp_path))
    assert manifest.attempts[0].outcome_receipt.status == "completed"
    assert dispatcher._active_dfk is None and not _ACTIVE_DISPATCH.get()
