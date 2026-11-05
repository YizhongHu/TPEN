"""P7 r2 real-DFK probes, with no alternate production backend or source edits."""
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from parsl.dataflow.dflow import DataFlowKernel
from tpen_executor.dispatch.parsl import ParslDispatcher, UnsupportedParslExecutorError
from tpen_executor.dispatch.contract import DispatcherContract
from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.runners import EvidenceUnavailable
from tpen_executor.storage import DirectoryStore
from tpen_executor.workers import SubprocessWorker
from test_p7_parsl_dispatcher import _allocation, _group, _runs, _make_stub_package, _worker_spec


def _dispatcher(worker=None):
    return ParslDispatcher(worker or SimpleNamespace(materialize=lambda env: {}),
        binding={"environment": {"CUDA_VISIBLE_DEVICES": "2"}})


def _real_dfk(dispatcher, tmp_path):
    config = dispatcher._config_type(executors=[dispatcher._thread_pool_type(max_threads=1)],
        run_dir=str(tmp_path / "parsl"), usage_tracking=False)
    dfk = dispatcher._dfk_type(config=config)
    assert isinstance(dfk, DataFlowKernel)
    object.__setattr__(dispatcher, "_active_dfk", dfk)
    return dfk


def test_r2_t1_real_app_preserves_interpreter_setup_env_argv(tmp_path):
    """GREEN: the real python_app path carries all prepared environment fields."""
    dispatcher = _dispatcher()
    dfk = _real_dfk(dispatcher, tmp_path)
    setup = tmp_path / "setup.sh"
    setup.write_text("export REVIEW_SETUP='setup value'\n")
    output = tmp_path / "observed.json"
    unusual = "literal $HOME; spaces and 'quotes'"
    code = "import json,os,sys; open(sys.argv[1],'w').write(json.dumps([os.environ['REVIEW_ENV'],os.environ['REVIEW_SETUP'],sys.argv[2],os.getcwd(),sys.executable]))"
    environment = PreparedPythonEnvironment(sys.executable, SourceScript(str(setup)), {"REVIEW_ENV": "env value"})
    try:
        outcome = dispatcher._execute_process({}, environment, ("-c", code, str(output), unusual),
            cwd=str(tmp_path), stdout_path=str(tmp_path / "out"), stderr_path=str(tmp_path / "err"), timeout=5)
        assert outcome.returncode == 0 and not outcome.timed_out
        assert json.loads(output.read_text()) == ["env value", "setup value", unusual, str(tmp_path), sys.executable]
    finally:
        dfk.cleanup()
        object.__setattr__(dispatcher, "_active_dfk", None)


@pytest.mark.parametrize("exit_kind", ["success", "failure", "cancellation"])
def test_r2_t2_cleanup_real_dfk_through_normal_contract_lookup(tmp_path, monkeypatch, exit_kind):
    """GREEN: replace the inherited method directly; no global super hook."""
    dispatcher = _dispatcher()
    created, cleaned = [], []
    real_factory = dispatcher._dfk_type

    def make_dfk(**kwargs):
        dfk = real_factory(**kwargs)
        original_cleanup = dfk.cleanup

        def cleanup():
            cleaned.append(dfk)
            return original_cleanup()

        dfk.cleanup = cleanup
        created.append(dfk)
        return dfk

    def contract(self, *args, **kwargs):
        if exit_kind == "failure":
            raise RuntimeError("review contract failure")
        if exit_kind == "cancellation":
            raise KeyboardInterrupt("review contract cancellation")
        return "review-manifest"

    object.__setattr__(dispatcher, "_dfk_type", make_dfk)
    monkeypatch.setattr(DispatcherContract, "run", contract)
    lifecycle = AttachedAllocation(_allocation(), DirectoryStore(tmp_path / "store"))
    if exit_kind == "success":
        assert dispatcher.run(_group("a"), _runs(tmp_path, "a"), lifecycle) == "review-manifest"
    else:
        expected = RuntimeError if exit_kind == "failure" else KeyboardInterrupt
        with pytest.raises(expected, match="review contract"):
            dispatcher.run(_group("a"), _runs(tmp_path, "a"), lifecycle)
    assert len(created) == 1 and cleaned == created
    assert dispatcher._active_dfk is None and dispatcher._active_binding is None


@pytest.mark.parametrize("phase", ["before_spawn", "running"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit], ids=["keyboard", "system-exit"])
def test_r2_t3_real_owner_cancellation_quiesces_app_before_finish(tmp_path, monkeypatch, phase, error_type):
    """RED before spawn, GREEN running: finish must follow child/app quiescence.

    AppFuture and DFK are real. Only result interruption and scheduling gates
    are controlled; every process uses the unmodified SubprocessWorker.
    """
    import tpen_executor.dispatch.parsl as module
    from parsl.dataflow.futures import AppFuture
    package = _make_stub_package(tmp_path)
    dispatcher = _dispatcher(_worker_spec(PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})))
    ready, release, terminal = threading.Event(), threading.Event(), threading.Event()
    futures, processes, spawned_after_terminal, finishes = [], [], [], []
    original_child = module._run_parsl_child
    original_attach = module._ChildProcessHandle.attach
    original_result = AppFuture.result
    original_factory = dispatcher._python_app_factory
    owner = threading.get_ident()
    interruption = error_type("review owner interruption")
    fired = False

    def child(*args, **kwargs):
        if phase == "before_spawn":
            ready.set()
            assert release.wait(5), "review scheduling gate did not open"
        return original_child(*args, **kwargs)

    def attach(handle, process):
        processes.append(process)
        spawned_after_terminal.append(terminal.is_set())
        original_attach(handle, process)
        if phase == "running":
            ready.set()

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
        nonlocal fired
        if threading.get_ident() == owner and futures and future is futures[0] and not fired:
            assert ready.wait(5), "real worker did not reach interruption phase"
            fired = True
            timer.start()
            raise interruption
        return original_result(future, timeout)

    class Lifecycle(AttachedAllocation):
        def finish(self, fence, status, finished_at_utc, result_digest=None):
            finishes.append((fence.task_id, status, futures[0].done()))
            published = super().finish(fence, status, finished_at_utc, result_digest)
            terminal.set()
            # Open the gate only after durable finish publication, even when
            # the product finishes early. Correct code gets the timer first.
            release.set()
            return published

    monkeypatch.setattr(module, "_run_parsl_child", child)
    monkeypatch.setattr(module._ChildProcessHandle, "attach", attach)
    monkeypatch.setattr(AppFuture, "result", interrupted_result)
    object.__setattr__(dispatcher, "_python_app_factory", factory)
    timer = threading.Timer(1.0, release.set)
    lifecycle = Lifecycle(_allocation(), DirectoryStore(tmp_path / "store"))
    try:
        with pytest.raises(error_type) as observed:
            dispatcher.run(_group("a", "b"), _runs(tmp_path, "a", "b", modes={"a": "sleep"}), lifecycle, timeout=2)
        assert observed.value is interruption
    finally:
        release.set()
        timer.cancel()
        if timer.ident is not None:
            timer.join()
    print(json.dumps({"phase": phase, "error": error_type.__name__, "finishes": finishes, "spawned_after_terminal": spawned_after_terminal, "child_pids": [p.pid for p in processes], "child_returncodes": [p.returncode for p in processes]}))
    expected_processes = 0 if phase == "before_spawn" else 1
    assert fired and len(futures) == 1 and len(processes) == expected_processes
    assert all(process.poll() is not None for process in processes), "test-owned leader was not reaped"
    assert len(finishes) == 1 and finishes[0][:2] == ("a", "failed")
    assert finishes[0][2], "attempt finished while its real app was still active"
    assert not any(spawned_after_terminal), "child spawned after the attempt was finished"


def test_r2_t4_real_app_deadline_never_calls_future_cancel(tmp_path, monkeypatch):
    """GREEN replacement for old T5: child timeout owns the actual deadline."""
    from parsl.dataflow.futures import AppFuture
    dispatcher = _dispatcher()
    dfk = _real_dfk(dispatcher, tmp_path)

    def forbidden_cancel(self):
        pytest.fail("real AppFuture.cancel must never own a child deadline")

    monkeypatch.setattr(AppFuture, "cancel", forbidden_cancel)
    try:
        outcome = dispatcher._execute_process({}, PreparedPythonEnvironment(sys.executable),
            ("-c", "import time; time.sleep(30)"), cwd=str(tmp_path),
            stdout_path=str(tmp_path / "out"), stderr_path=str(tmp_path / "err"), timeout=0.05)
        assert outcome.timed_out and outcome.returncode is not None and outcome.returncode != 0
    finally:
        dfk.cleanup()
        object.__setattr__(dispatcher, "_active_dfk", None)


def test_r2_t5_unsupported_config_refused_before_any_artifact(tmp_path):
    """GREEN: unsupported geometry is a typed pre-admission refusal."""
    dispatcher = _dispatcher()
    object.__setattr__(dispatcher, "executor_config", SimpleNamespace(executors=[object()]))
    root = tmp_path / "store"
    with pytest.raises(UnsupportedParslExecutorError, match="in-process ThreadPool"):
        dispatcher.run(_group("a"), _runs(tmp_path, "a"), AttachedAllocation(_allocation(), DirectoryStore(root)))
    assert not root.exists()


def test_r2_t6_common_admission_refuses_before_dfk_artifacts(tmp_path):
    """RED: an invalid run mapping must be refused before DFK initialization."""
    dispatcher = _dispatcher()
    root = tmp_path / "store"
    with pytest.raises(ValueError, match="runs keys"):
        dispatcher.run(_group("a"), {}, AttachedAllocation(_allocation(), DirectoryStore(root)))
    assert not root.exists(), "invalid admission created Parsl-owned artifacts"


def test_r2_t7_real_app_success_does_not_hide_missing_evidence(tmp_path):
    """GREEN: actual app/process success leaves an absent core receipt visible."""
    package = _make_stub_package(tmp_path)

    class Evidence:
        def read(self, *args, **kwargs):
            raise EvidenceUnavailable("review missing public receipt")

    environment = PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})
    worker = SimpleNamespace(materialize=lambda env: {"environment": environment,
        "process": SubprocessWorker(), "runner": object(), "evidence": Evidence()})
    manifest = _dispatcher(worker).run(_group("a"), _runs(tmp_path, "a"),
        AttachedAllocation(_allocation(), DirectoryStore(tmp_path / "store")))
    receipt = manifest.attempts[0].outcome_receipt
    assert receipt.status == "completed" and receipt.result_digest is None
    assert not manifest.segments
