"""R2 delta probes. Execute only in the authorized Cannon allocation."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from tpen_executor.dispatch import LocalDispatcher
from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.storage import DirectoryStore
from tpen_executor.workers import ProcessOutcome, SubprocessWorker
from test_review_p5d_r1 import RecordingLifecycle, group, normal_worker, runs


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit], ids=["keyboard", "system-exit"])
def test_r2_t1_evidence_interruption_preserves_already_reaped_timeout(tmp_path, error_type):
    """RED: the accepted F1 decision preserves a more specific known status."""
    cancellation = error_type("inspection cancelled after child timed out")
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))

    class Process:
        def run(self, env, args, **kw):
            return ProcessOutcome(tuple(args), -9, kw["stdout_path"], kw["stderr_path"], True)

    class Evidence:
        def read(self, *args, **kw):
            raise cancellation

    worker = SimpleNamespace(materialize=lambda env: {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()})
    with pytest.raises(error_type) as observed:
        LocalDispatcher(worker).run(group("a", "b"), runs(tmp_path, "a", "b"), lifecycle)
    assert observed.value is cancellation
    assert [f.task_id for f in lifecycle.begun] == ["a"]
    assert len(lifecycle.finished) == 1
    assert lifecycle.finished[0][1] == "timed_out", lifecycle.finished


@pytest.mark.parametrize("phase", ["materialize", "process", "evidence"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit], ids=["keyboard", "system-exit"])
def test_r2_t2_cancellation_propagates_after_one_finish_without_next_member(tmp_path, phase, error_type):
    """GREEN: extend R1's finish-only control to the accepted cancellation policy."""
    cancellation = error_type("one cancellation")
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))

    class Process:
        def run(self, env, args, **kw):
            if phase == "process":
                raise cancellation
            return ProcessOutcome(tuple(args), 0, kw["stdout_path"], kw["stderr_path"], False)

    class Evidence:
        def read(self, *args, **kw):
            raise cancellation

    def materialize(env):
        if phase == "materialize":
            raise cancellation
        return {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()}

    with pytest.raises(error_type) as observed:
        LocalDispatcher(SimpleNamespace(materialize=materialize)).run(group("a", "b"), runs(tmp_path, "a", "b"), lifecycle)
    assert observed.value is cancellation
    assert [f.task_id for f in lifecycle.begun] == ["a"]
    assert lifecycle.finished == [("a", "failed", None)]
    assert lifecycle.store.manifest(group("a", "b")).attempts[0].outcome_receipt.status == "failed"


@pytest.mark.parametrize("during_setup", [False, True], ids=["science-child", "setup-child"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit], ids=["keyboard", "system-exit"])
def test_r2_t3_real_worker_kills_owned_group_and_reaps_before_propagation(tmp_path, monkeypatch, during_setup, error_type):
    """GREEN: no scheduler cancellation; interrupt one owned bounded child call."""
    ready = tmp_path / "ready"
    late = tmp_path / "late"
    script = tmp_path / "child.py"
    script.write_text(
        "import pathlib,subprocess,sys,time\n"
        "subprocess.Popen([sys.executable,'-c','import pathlib,sys,time;time.sleep(1.5);pathlib.Path(sys.argv[1]).write_text(\"late\")',sys.argv[2]])\n"
        "pathlib.Path(sys.argv[1]).write_text('ready')\n"
        "time.sleep(4)\n",
        encoding="utf-8",
    )
    args = (str(script), str(ready), str(late))
    environment = PreparedPythonEnvironment(sys.executable)
    if during_setup:
        setup = tmp_path / "setup.sh"
        setup.write_text(shlex.join((sys.executable, *args)) + "\n", encoding="utf-8")
        environment = PreparedPythonEnvironment(sys.executable, SourceScript(str(setup)))
        args = ("-c", "pass")
    original_wait = subprocess.Popen.wait
    original_killpg = os.killpg
    calls = []
    processes = []
    cancellation = error_type("review wait interruption")

    def wait(process, *a, **kw):
        calls.append("wait")
        if not processes:
            processes.append(process)
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), "bounded child must reach readiness before injected cancellation"
            raise cancellation
        result = original_wait(process, *a, **kw)
        calls.append("reaped")
        return result

    def killpg(pid, sig):
        assert processes and pid == processes[0].pid
        # review P1-1 residual: after reaping, the worker also terminates
        # and bounded-probes its own group (killpg(SIGKILL) then
        # killpg(0)), so a kill signal and a signal-0 probe are both
        # legitimate here now.
        assert sig in (signal.SIGKILL, 0)
        calls.append("killpg" if sig == signal.SIGKILL else "probe")
        return original_killpg(pid, sig)

    with monkeypatch.context() as patch:
        patch.setattr(subprocess.Popen, "wait", wait)
        patch.setattr(os, "killpg", killpg)
        with pytest.raises(error_type) as observed:
            SubprocessWorker().run(environment, args, cwd=tmp_path, stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", timeout=8)
    assert observed.value is cancellation
    # "killpg" (the owned-group kill), "reaped", then the post-reap
    # terminate+probe (killpg(SIGKILL) again, now a no-op since the group
    # is already gone, then the signal-0 probe) that confirms quiescence
    # before this worker call propagates.
    assert calls == ["wait", "killpg", "wait", "reaped", "killpg", "probe"], calls
    assert processes[0].returncode == -signal.SIGKILL
    time.sleep(1.8)
    assert not late.exists(), "owned descendant must not survive to write after cancellation"


@pytest.mark.parametrize("cleanup_error", [KeyboardInterrupt, SystemExit], ids=["second-keyboard", "cleanup-system-exit"])
def test_r2_t4_interrupted_cleanup_wait_cannot_return_without_reaping(tmp_path, monkeypatch, cleanup_error):
    """RED: inject a second catchable exit after group kill, before the reap."""
    cancellation = KeyboardInterrupt("original cancellation")
    waits = []
    kills = []

    class Child:
        pid = 123456789
        returncode = None

        def wait(self, *args, **kw):
            waits.append(len(waits) + 1)
            if len(waits) == 1:
                raise cancellation
            if len(waits) == 2:
                raise cleanup_error("cleanup wait interrupted")
            self.returncode = -9
            return -9

    child = Child()

    def killpg(pid, sig):
        # The first SIGKILL is the real, owned-group kill; every call
        # after that (review P1-1 residual's own terminate-then-probe)
        # finds nothing left, exactly as a real reaped group would.
        kills.append((pid, sig))
        if len(kills) > 1:
            raise ProcessLookupError()

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", lambda *args, **kw: child)
        patch.setattr(os, "killpg", killpg)
        with pytest.raises(KeyboardInterrupt) as observed:
            SubprocessWorker().run(PreparedPythonEnvironment(sys.executable), ("-c", "pass"), cwd=tmp_path, stdout_path=tmp_path / "out", stderr_path=tmp_path / "err")
    assert observed.value is cancellation
    assert kills == [(child.pid, signal.SIGKILL), (child.pid, signal.SIGKILL), (child.pid, 0)]
    assert child.returncode is not None, ("cleanup returned without confirmed reap", waits, kills)


def test_r2_t5_later_finish_error_does_not_hide_promised_first_begin_error(tmp_path):
    """RED: accepted F3 promises the first begin error after all members."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    begin_error = ValueError("first member begin refused")
    finish_error = OSError("later finish failed")
    checked = []

    def begin(g, task_id, started_at_utc):
        checked.append(task_id)
        if task_id == "a":
            raise begin_error
        return lifecycle.begin(g, task_id, started_at_utc)

    def finish(*args):
        result = lifecycle.finish(*args)
        if args[0].task_id == "b":
            raise finish_error
        return result

    observed = None
    try:
        LocalDispatcher(normal_worker()).run(group("a", "b", "c"), runs(tmp_path, "a", "b", "c"), SimpleNamespace(begin=begin, finish=finish, store=lifecycle.store))
    except Exception as exc:
        observed = exc
    assert checked == ["a", "b", "c"]
    assert [f[0] for f in lifecycle.finished] == ["b", "c"]
    assert observed is begin_error, ("first begin refusal was hidden", type(observed).__name__, str(observed))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "neg-inf"])
@pytest.mark.parametrize("target", ["dispatcher", "worker"])
def test_r2_t6_nonfinite_timeout_rejected_before_any_artifact(tmp_path, value, target):
    """GREEN: independently cover the newly chosen finite-value policy."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    with pytest.raises(ValueError, match="finite"):
        if target == "dispatcher":
            LocalDispatcher(normal_worker()).run(group("a"), runs(tmp_path, "a"), lifecycle, timeout=value)
        else:
            SubprocessWorker().run(PreparedPythonEnvironment(sys.executable), ("-c", "pass"), cwd=tmp_path, stdout_path=tmp_path / "out", stderr_path=tmp_path / "err", timeout=value)
    assert lifecycle.begun == []
    assert not (tmp_path / "store").exists()
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "err").exists()


def test_r2_t7_multiple_begin_refusals_preserve_first_original_exception(tmp_path):
    """GREEN: F3 ordinary path preserves first refusal and finishes survivors."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    first = OSError("first begin refusal")
    checked = []

    def begin(g, task_id, started_at_utc):
        checked.append(task_id)
        if task_id == "b":
            raise first
        if task_id == "c":
            raise ValueError("second begin refusal")
        return lifecycle.begin(g, task_id, started_at_utc)

    with pytest.raises(OSError) as observed:
        LocalDispatcher(normal_worker()).run(group("a", "b", "c", "d"), runs(tmp_path, "a", "b", "c", "d"), SimpleNamespace(begin=begin, finish=lifecycle.finish, store=lifecycle.store))
    assert observed.value is first
    assert checked == ["a", "b", "c", "d"]
    assert [f[0] for f in lifecycle.finished] == ["a", "d"]
