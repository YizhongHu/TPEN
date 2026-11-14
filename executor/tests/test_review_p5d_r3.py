"""R3 controls for the accepted delta; run only in the Cannon allocation."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tpen_executor.dispatch import LocalDispatcher
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.storage import DirectoryStore
from tpen_executor.workers import SubprocessWorker
from test_review_p5d_r1 import RecordingLifecycle, group, normal_worker, runs


@pytest.mark.parametrize("original_type", [KeyboardInterrupt, SystemExit], ids=["keyboard", "system-exit"])
def test_r3_t1_repeated_cleanup_interruptions_are_visible_on_original_exception(tmp_path, monkeypatch, original_type):
    """GREEN: reap after several interruptions and preserve each diagnostic."""
    original = original_type("original interruption")
    later = [KeyboardInterrupt("cleanup keyboard"), SystemExit("cleanup exit"), RuntimeError("cleanup transient error")]
    waits = []
    kills = []

    class Child:
        pid = 123456789
        returncode = None

        def wait(self, *args, **kwargs):
            index = len(waits)
            waits.append(index)
            if index == 0:
                raise original
            if index <= len(later):
                raise later[index - 1]
            self.returncode = -9
            return -9

    child = Child()

    def killpg(pid, sig):
        kills.append((pid, sig))
        if len(kills) > 1:
            raise ProcessLookupError()

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
        patch.setattr(os, "killpg", killpg)
        with pytest.raises(original_type) as observed:
            SubprocessWorker().run(PreparedPythonEnvironment(sys.executable), ("-c", "pass"), cwd=tmp_path, stdout_path=tmp_path / "out", stderr_path=tmp_path / "err")
    assert observed.value is original
    assert child.returncode == -9
    assert waits == [0, 1, 2, 3, 4]
    # Both calls are owner-side SIGKILLs: the initial interruption cleanup
    # and the cleanup quiescence step. Membership observation is signal-free.
    assert kills == [(child.pid, signal.SIGKILL), (child.pid, signal.SIGKILL)]
    notes = "\n".join(getattr(original, "__notes__", ()))
    for error in later:
        assert type(error).__name__ in notes and str(error) in notes, (error, notes)


@pytest.mark.parametrize("mode", ["both", "begin-only", "finish-only"])
def test_r3_t2_deferred_primary_error_and_secondary_finish_note_are_visible(tmp_path, mode):
    """GREEN: validate the accepted note policy as well as exception identity."""
    lifecycle = RecordingLifecycle(DirectoryStore(tmp_path / "store"))
    first_begin = ValueError("first begin refusal")
    later_finish = OSError("later member finish failure")
    checked = []

    def begin(g, task_id, started_at_utc):
        checked.append(task_id)
        if task_id == "a" and mode != "finish-only":
            raise first_begin
        return lifecycle.begin(g, task_id, started_at_utc)

    def finish(*args):
        result = lifecycle.finish(*args)
        if args[0].task_id == "b" and mode != "begin-only":
            raise later_finish
        return result

    observed = None
    try:
        LocalDispatcher(normal_worker()).run(group("a", "b", "c"), runs(tmp_path, "a", "b", "c"), SimpleNamespace(begin=begin, finish=finish, store=lifecycle.store))
    except Exception as error:
        observed = error
    assert checked == ["a", "b", "c"]
    expected_finishes = ["a", "b", "c"] if mode == "finish-only" else ["b", "c"]
    assert [f[0] for f in lifecycle.finished] == expected_finishes
    assert observed is (later_finish if mode == "finish-only" else first_begin)
    if mode == "both":
        notes = "\n".join(getattr(observed, "__notes__", ()))
        assert "OSError" in notes and str(later_finish) in notes, notes


def test_r3_t3_already_reaped_leader_does_not_require_another_wait(tmp_path, monkeypatch):
    """GREEN: a wait that reaped before interruption still propagates safely."""
    original = KeyboardInterrupt("interruption after reap")
    waits = []
    kills = []

    class Child:
        pid = 123456789
        returncode = None

        def wait(self, *args, **kwargs):
            waits.append(True)
            self.returncode = -9
            raise original

    def killpg(pid, sig):
        kills.append((pid, sig))
        raise ProcessLookupError("owned group is already gone")

    child = Child()
    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "Popen", lambda *args, **kwargs: child)
        patch.setattr(os, "killpg", killpg)
        with pytest.raises(KeyboardInterrupt) as observed:
            SubprocessWorker().run(PreparedPythonEnvironment(sys.executable), ("-c", "pass"), cwd=tmp_path, stdout_path=tmp_path / "out", stderr_path=tmp_path / "err")
    assert observed.value is original
    assert child.returncode == -9
    assert len(waits) == 1
    # The owned-group kill and the cleanup quiescence step are both SIGKILL;
    # membership observation is signal-free.
    assert kills == [(child.pid, signal.SIGKILL), (child.pid, signal.SIGKILL)]
