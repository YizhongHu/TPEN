"""P7 r4 real backend terminal-failure probe; execute only in Slurm."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


_CHILD = r'''
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[2])
from parsl.dataflow.futures import AppFuture
from test_review_p7_r3 import dispatcher_for, lifecycle_for
from test_p7_parsl_dispatcher import _group, _runs
root = Path(sys.argv[1])
root.mkdir()
dispatcher = dispatcher_for(root)
real_factory = dispatcher._dfk_type
real_result = AppFuture.result
record = root / "terminal-future.json"

def make_dfk(**kwargs):
    dfk = real_factory(**kwargs)
    # An actual supported executor failure: public shutdown makes subsequent
    # submissions fail in its real concurrent.futures ThreadPoolExecutor.
    dfk.executors["threads"].shutdown()
    return dfk

def result(self, timeout=None):
    try:
        return real_result(self, timeout)
    except BaseException as error:
        if not record.exists():
            staged = record.with_suffix(".tmp")
            staged.write_text(json.dumps({"future_type": type(self).__name__,
                "done": self.done(), "error_type": type(error).__name__, "error": str(error)}))
            staged.replace(record)
        raise

object.__setattr__(dispatcher, "_dfk_type", make_dfk)
AppFuture.result = result
manifest = dispatcher.run(_group("a"), _runs(root, "a"), lifecycle_for(root))
assert manifest.attempts[0].outcome_receipt.status == "launch_failed"
(root / "returned.json").write_text(json.dumps({"status": manifest.attempts[0].outcome_receipt.status,
    "active_dfk": dispatcher._active_dfk is not None}))
'''


def test_r4_t1_terminal_real_future_failure_is_not_an_endless_reap_wait(tmp_path):
    """RED: a done Future exception must propagate, not be retried forever.

    A separate owner process bounds the defective loop without substituting
    a Future or backend. The spy only records the first actual result error.
    """
    root = tmp_path / "owner"
    tests_dir = str(Path(__file__).parent)
    out, err = tmp_path / "owner.out", tmp_path / "owner.err"
    hung = False
    with out.open("wb") as stdout, err.open("wb") as stderr:
        process = subprocess.Popen([sys.executable, "-c", _CHILD, str(root), tests_dir],
            stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            deadline = time.monotonic() + 10
            record = root / "terminal-future.json"
            while not record.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            if not record.exists():
                pytest.fail("real terminal Future failure was not reached: " + err.read_text()[-2000:])
            observation = json.loads(record.read_text())
            assert observation["future_type"] == "AppFuture" and observation["done"]
            assert observation["error_type"] == "RuntimeError"
            assert observation["error"] == "cannot schedule new futures after shutdown"
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                hung = True
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    observation.update(hung_after_terminal_failure=hung, owner_pid=process.pid,
        owner_returncode=process.returncode)
    print(json.dumps(observation))
    (tmp_path / "review-observation.json").write_text(json.dumps(observation))
    assert not hung, "dispatcher kept retrying an already-terminal real AppFuture failure"
    assert process.returncode == 0, err.read_text()[-2000:]
    assert json.loads((root / "returned.json").read_text()) == {"status": "launch_failed", "active_dfk": False}
