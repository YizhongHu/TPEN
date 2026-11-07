"""Independent P7 backend probes; no implementation or mutant edits."""
from concurrent.futures import TimeoutError
import json
import importlib.util
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


PARSL_REQUIRED = pytest.mark.skipif(
    importlib.util.find_spec("parsl") is None,
    reason="requires the approved executor parsl extra",
)


def _dispatcher():
    return ParslDispatcher(SimpleNamespace(materialize=lambda env: {}),
                           binding={"environment": {"CUDA_VISIBLE_DEVICES": "0"}})


def _execute(dispatcher, tmp_path, timeout=None):
    return dispatcher._execute_process({}, PreparedPythonEnvironment(sys.executable),
        ("-c", "raise SystemExit(0)"), cwd=str(tmp_path),
        stdout_path=str(tmp_path / "stdout"), stderr_path=str(tmp_path / "stderr"),
        timeout=timeout)


@PARSL_REQUIRED
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


@PARSL_REQUIRED
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
