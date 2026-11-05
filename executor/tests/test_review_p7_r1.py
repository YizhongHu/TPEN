"""Independent P7 backend probes; no implementation or mutant edits."""
from concurrent.futures import TimeoutError
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tpen_executor.dispatch.parsl import ParslDispatcher, _run_parsl_child
from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.placement import BindingError, WorkerBinding
from tpen_executor.workers import ProcessOutcome
from test_p7_parsl_dispatcher import _allocation, _group, _runs
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.storage import DirectoryStore


def _dispatcher():
    return ParslDispatcher(SimpleNamespace(materialize=lambda env: {}),
                           binding={"environment": {"CUDA_VISIBLE_DEVICES": "0"}})


def _execute(dispatcher, tmp_path, timeout=None):
    return dispatcher._execute_process({}, PreparedPythonEnvironment(sys.executable),
        ("-c", "raise SystemExit(0)"), cwd=str(tmp_path),
        stdout_path=str(tmp_path / "stdout"), stderr_path=str(tmp_path / "stderr"),
        timeout=timeout)


def test_t1_native_dfk_submit_accepts_prepared_payload(tmp_path):
    """RED: exercise the actual pinned DFK API without the run/super blocker."""
    dispatcher = _dispatcher()
    config = dispatcher._config_type(
        executors=[dispatcher._thread_pool_type(max_threads=1)],
        run_dir=str(tmp_path / "parsl"), usage_tracking=False)
    dfk = dispatcher._dfk_type(config=config)
    object.__setattr__(dispatcher, "_active_dfk", dfk)
    try:
        assert _execute(dispatcher, tmp_path).returncode == 0
    finally:
        dfk.cleanup()
        object.__setattr__(dispatcher, "_active_dfk", None)


@pytest.mark.parametrize("exit_kind", ["success", "failure", "cancellation"])
def test_t2_dfk_cleanup_on_each_contract_exit(tmp_path, monkeypatch, exit_kind):
    """Isolate DFK ownership from both broken super and process submission.

    The contract-delegation seam returns or raises without starting a member.
    This is a cleanup unit probe, not proof of successful end-to-end dispatch.
    """
    import tpen_executor.dispatch.parsl as module
    dispatcher = _dispatcher()
    cleaned = []
    dfks = []

    def make_dfk(**kwargs):
        dfk = SimpleNamespace(cleanup=lambda: cleaned.append("cleanup"))
        dfks.append(dfk)
        return dfk

    def delegate(*args, **kwargs):
        if exit_kind == "failure":
            raise RuntimeError("review contract failure")
        if exit_kind == "cancellation":
            raise KeyboardInterrupt("review contract cancellation")
        return "review-manifest"

    object.__setattr__(dispatcher, "_dfk_type", make_dfk)
    monkeypatch.setattr(module, "super", lambda: SimpleNamespace(run=delegate), raising=False)
    lifecycle = AttachedAllocation(_allocation(), DirectoryStore(tmp_path / "store"))
    if exit_kind == "success":
        assert dispatcher.run(_group("a"), _runs(tmp_path, "a"), lifecycle) == "review-manifest"
    else:
        expected = RuntimeError if exit_kind == "failure" else KeyboardInterrupt
        with pytest.raises(expected, match="review contract"):
            dispatcher.run(_group("a"), _runs(tmp_path, "a"), lifecycle)
    assert len(dfks) == 1
    assert dispatcher._active_dfk is None and dispatcher._active_binding is None
    assert cleaned == ["cleanup"], "run returned without cleaning its DFK"


def test_t3_two_allocations_create_distinct_dfks(tmp_path, monkeypatch):
    """GREEN control: construction is per run; cleanup is tested separately."""
    import tpen_executor.dispatch.parsl as module
    dispatcher = _dispatcher()
    created = []

    def make_dfk(**kwargs):
        created.append(SimpleNamespace(cleanup=lambda: None))
        return created[-1]

    object.__setattr__(dispatcher, "_dfk_type", make_dfk)
    monkeypatch.setattr(module, "super", lambda: SimpleNamespace(run=lambda *a, **k: None), raising=False)
    for name in ("first", "second"):
        lifecycle = AttachedAllocation(_allocation(allocation_id=name), DirectoryStore(tmp_path / name))
        dispatcher.run(_group(name), _runs(tmp_path, name), lifecycle)
    assert len(created) == 2 and created[0] is not created[1]


def test_t4_owner_cancellation_reaches_active_future(tmp_path):
    """RED: owner cancellation must reach the backend before it escapes."""
    dispatcher = _dispatcher()
    cancelled = []

    def interrupted_result(timeout):
        raise KeyboardInterrupt("owner interrupted during result")

    future = SimpleNamespace(result=interrupted_result, cancel=lambda: cancelled.append(True))
    object.__setattr__(dispatcher, "_active_dfk", SimpleNamespace(submit=lambda *a, **k: future))
    with pytest.raises(KeyboardInterrupt, match="owner interrupted"):
        _execute(dispatcher, tmp_path)
    assert cancelled == [True], "active backend work was never asked to stop"


def test_t5_parsl_timeout_is_normalized_even_when_cancel_is_unsupported(tmp_path):
    """RED: use pinned AppFuture.cancel, which raises NotImplementedError."""
    from parsl.dataflow.futures import AppFuture
    dispatcher = _dispatcher()

    class Future:
        cancel = AppFuture.cancel

        def result(self, timeout):
            raise TimeoutError("backend result deadline")

    object.__setattr__(dispatcher, "_active_dfk", SimpleNamespace(submit=lambda *a, **k: Future()))
    assert _execute(dispatcher, tmp_path, timeout=0.01).timed_out


def test_t6_worker_rebuilds_setup_environment_and_exact_argv(tmp_path):
    """GREEN: direct worker payload consumes every environment field."""
    setup = tmp_path / "setup.sh"
    setup.write_text("export REVIEW_SETUP='setup value'\n")
    output = tmp_path / "observed.json"
    unusual = "literal $HOME; spaces and 'quotes'"
    code = "import json,os,sys; open(sys.argv[1],'w').write(json.dumps([os.environ['REVIEW_ENV'],os.environ['REVIEW_SETUP'],sys.argv[2],os.getcwd(),sys.executable]))"
    outcome = _run_parsl_child(sys.executable, str(setup), {"REVIEW_ENV": "env value"},
        ("-c", code, str(output), unusual), cwd=str(tmp_path),
        stdout_path=str(tmp_path / "out"), stderr_path=str(tmp_path / "err"), timeout=5)
    assert outcome.returncode == 0 and not outcome.timed_out
    assert json.loads(output.read_text()) == ["env value", "setup value", unusual, str(tmp_path), sys.executable]


def test_t7_direct_worker_child_timeout_is_reaped(tmp_path):
    """GREEN: child-side timeout works independently of owner Future handling."""
    outcome = _run_parsl_child(sys.executable, None, {}, ("-c", "import time; time.sleep(30)"),
        cwd=str(tmp_path), stdout_path=str(tmp_path / "out"),
        stderr_path=str(tmp_path / "err"), timeout=0.05)
    assert outcome.timed_out and outcome.returncode is not None and outcome.returncode != 0


@pytest.mark.parametrize("nodes", [["node-a", "node-b"], []])
def test_t8_explicit_binding_refuses_ambiguous_nodes(nodes):
    """RED: explicit binding must not silently discard an invalid node shape."""
    with pytest.raises(BindingError):
        WorkerBinding.from_mapping({"environment": {"CUDA_VISIBLE_DEVICES": "0"}, "nodes": nodes})


def test_t9_binding_overlay_preserves_prepared_environment(tmp_path):
    """GREEN: binding overrides visibility while retaining setup and other env."""
    dispatcher = _dispatcher()
    setup = SourceScript(str(tmp_path / "operator.sh"))
    original = PreparedPythonEnvironment(sys.executable, setup,
        {"CUDA_VISIBLE_DEVICES": "old", "REVIEW_ENV": "kept"})
    object.__setattr__(dispatcher, "_active_binding", WorkerBinding({"CUDA_VISIBLE_DEVICES": "2"}))
    prepared = dispatcher._worker_environment({"environment": original})
    assert prepared.python == original.python and prepared.setup == original.setup
    assert dict(prepared.env) == {"CUDA_VISIBLE_DEVICES": "2", "REVIEW_ENV": "kept"}
    assert dict(original.env) == {"CUDA_VISIBLE_DEVICES": "old", "REVIEW_ENV": "kept"}
