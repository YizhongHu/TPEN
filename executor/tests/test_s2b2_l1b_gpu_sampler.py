"""S2b-2 L1b: worker-owned lifetime GPU sampler, v3 receipt, and contract mutants.

The suite is executor-only.  Process-lifecycle tests drive the real
:mod:`tpen_executor.gpu_sampler` code and the real
:class:`~tpen_executor.workers.SubprocessWorker` against deterministic fake
``nvidia-smi`` scripts; receipt tests go through the real builder and
decoder.  The end-to-end producer outcomes crossing into the P8 reader live in
``experiments/toolkit/test_review_p8_toolkit_r3.py``.

Reaping determinism (review D4): a test whose command leaves an orphaned
descendant runs under the :func:`reaper` fixture, which makes this process a
Linux child subreaper and reaps only orphaned non-leader zombies of foreign
process groups, so ``killpg(pgid, 0)`` reaches ESRCH without depending on the
facility's init.  It is skipped by name off Linux.  Every other hung-command
fake ``exec``\\ s ``sleep`` so the command leader itself is the process the
sampler kills and reaps.  Every fixture kills what it recorded, even after a
mutant assertion fails.

Mutant ids: the r2 ids K1-K5 targeted the removed in-child sampler and are
retired, not renamed.  G1-G5 below are the r3 runtime-path contract mutants.
Historical N6/N8/N11 remain unclosed by any mutant id; N6/N8 objectives are
covered by named behaviour tests (``test_sampler_keeps_samples_taken_before_a_late_failure``
and the ``qualified-over-failed-sampler`` decoder case); N11 is L1a scope.
"""

from __future__ import annotations

import ast
import copy
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

from tpen_executor import gpu_sampler as gs
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.cli import _operational_exit
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.dispatch.contract import _placement_receipt_v3
from tpen_executor.dispatch.local import LocalDispatcher
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.gpu_telemetry import derive_device_decision, not_sampled_sampler, validate_sampler_wire
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.placement import AllocationSlot, ChildPlacementObservation
from tpen_executor.results import (
    PLACEMENT_NOT_SAMPLED_MESSAGE,
    PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE,
    PLACEMENT_UNQUALIFIED_MESSAGE,
    ManifestAttempt,
    OutcomeReceipt,
    RunManifest,
    SciencePlacementReceipt,
    SciencePlacementReceiptV2,
    SciencePlacementReceiptV3,
    StartReceipt,
    decode_placement_receipt,
)
from tpen_executor.runners import CoreRunEvidence, RUN_SHIM
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.workers import ProcessOutcome, SubprocessWorker

GPU_A = "GPU-12345678-1234-1234-1234-123456789ABC"
GPU_B = "GPU-12345678-1234-1234-1234-123456789ABD"
GPU_C = "GPU-12345678-1234-1234-1234-123456789ABE"
FIXTURES = Path(__file__).parent / "fixtures"
POSIX_GROUPS = pytest.mark.skipif(not hasattr(os, "killpg"), reason="requires POSIX process groups")
LINUX_CHILD = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not hasattr(os, "sched_setaffinity"),
    reason="requires the Linux child affinity contract",
)

L1B_MUTANT_TABLE = (
    {
        "id": "G1", "target": "executor/src/tpen_executor/workers.py",
        "old": "        interruptions = finish_gpu_sampler(sampler, sampler_output)",
        "new": "        interruptions = [] if timed_out else finish_gpu_sampler(sampler, sampler_output)",
        "killer": "test_worker_timeout_drains_sampler_groups_before_return",
        "suite": "executor", "assertion": "a worker timeout of the science child still drains every sampler group before release (D1)",
    },
    {
        "id": "G2", "target": "executor/src/tpen_executor/gpu_sampler.py",
        "old": "        kill_group(pgid)\n        deadline = time.monotonic() + self.cleanup_timeout",
        "new": "        pass\n        deadline = time.monotonic() + self.cleanup_timeout",
        "killer": "test_sampler_success_path_kills_lingering_descendant",
        "suite": "executor", "assertion": "the success path kills the retained group before claiming quiescence (D4 reaper)",
    },
    {
        "id": "G3", "target": "executor/src/tpen_executor/gpu_sampler.py",
        "old": "                thread.join(max(0.0, deadline - time.monotonic()))",
        "new": "                thread.join()",
        "killer": "test_sampler_finish_is_bounded_while_a_launch_blocks",
        "suite": "executor", "assertion": "finish returns non-quiescent evidence at the shared deadline even while Popen blocks (D2)",
    },
    {
        "id": "G4", "target": "executor/src/tpen_executor/gpu_sampler.py",
        "old": '            aggregate["outside_use_seen"] = aggregate["outside_use_seen"] or uuid not in self.declared',
        "new": '            aggregate["outside_use_seen"] = uuid not in self.declared',
        "killer": "test_sampler_aggregate_remembers_evicted_outside_use",
        "suite": "executor", "assertion": "outside use seen in a sample that is not retained stays in the aggregate",
    },
    {
        "id": "G5", "target": "executor/src/tpen_executor/gpu_telemetry.py",
        "old": '    if aggregate["outside_use_seen"]:',
        "new": '    if any(row["pid"] == child_pid and row["gpu_uuid"] not in declared for sample in sampler["samples"] for row in sample["pid_rows"]):',
        "killer": "test_outside_use_in_an_evicted_sample_dominates",
        "suite": "executor", "assertion": "the device decision uses the lifetime aggregate, not only retained samples",
    },
)


def test_l1b_mutant_table_targets_unique_executable_anchors() -> None:
    """Every mutant names a unique executable anchor, parses, and has a killer."""

    root = Path(__file__).parents[2]
    module = Path(__file__).read_text(encoding="utf-8")
    assert [row["id"] for row in L1B_MUTANT_TABLE] == ["G1", "G2", "G3", "G4", "G5"]
    for row in L1B_MUTANT_TABLE:
        source = (root / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1, row
        assert row["old"] != row["new"] and row["suite"] == "executor" and row["assertion"]
        # Anchors are code, not docstrings or comments, and the mutant parses.
        assert not row["old"].strip().startswith(("#", '"""', "'"))
        ast.parse(source.replace(row["old"], row["new"]))
        assert f"def {row['killer'].split('[')[0]}(" in module, row


# ------------------------------------------------------- process fixtures


PR_SET_CHILD_SUBREAPER = 36


class _Reaper:
    """Linux child subreaper that reaps orphaned group members for a test.

    Orphaned descendants of a command (a non-leader of a foreign process
    group) are reparented here and reaped as soon as they become zombies.
    Leaders (``pid == pgid``) and members of this process's own group are
    never touched, so ``subprocess`` keeps ownership of its own children.
    """

    def __init__(self) -> None:
        self._libc = ctypes.CDLL(None, use_errno=True)
        if self._libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")
        self._pid_files: list[Path] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="test-reaper", daemon=True)
        self._thread.start()

    def track(self, pid_file: Path) -> Path:
        """Kill every PID listed in ``pid_file`` at teardown."""

        self._pid_files.append(pid_file)
        return pid_file

    def reap_orphans(self) -> None:
        me, own_group = os.getpid(), os.getpgrp()
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                text = Path(f"/proc/{entry}/stat").read_text(encoding="utf-8")
            except OSError:
                continue
            fields = text[text.rfind(")") + 2 :].split()
            pid, state, ppid, pgrp = int(entry), fields[0], int(fields[1]), int(fields[2])
            if state == "Z" and ppid == me and pid != pgrp and pgrp != own_group:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass

    def _loop(self) -> None:
        while not self._stop.wait(0.005):
            self.reap_orphans()

    def close(self) -> None:
        _kill_listed(self._pid_files)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self.reap_orphans()
            time.sleep(0.02)
        self._stop.set()
        self._thread.join(1.0)
        self._libc.prctl(PR_SET_CHILD_SUBREAPER, 0, 0, 0, 0)


def _kill_listed(pid_files: list[Path]) -> None:
    """SIGKILL every PID recorded in the given files; ignore absent ones."""

    for pid_file in pid_files:
        if not pid_file.exists():
            continue
        for line in pid_file.read_text(encoding="utf-8").split():
            try:
                os.kill(int(line), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, ValueError):
                pass


@pytest.fixture
def reaper():
    """Deterministic orphan reaping for lifecycle tests (Linux subreaper)."""

    if not sys.platform.startswith("linux"):
        pytest.skip("named skip: deterministic orphan reaping needs a Linux subreaper (PR_SET_CHILD_SUBREAPER)")
    instance = _Reaper()
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def pid_files():
    """Collect command PID files and kill their PIDs even if the test fails."""

    files: list[Path] = []
    try:
        yield files
    finally:
        _kill_listed(files)


def _fake(directory: Path, name: str, body: str) -> str:
    """Write one executable POSIX shell fake and return its path."""

    path = directory / name
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


# The sampler under test runs in this process, so $PPID of a fake is the
# target PID of a direct GpuSampler test.
_OK_BODY = f"""\
case "$1" in
  --version) echo "NVIDIA-SMI fake" ;;
  --query-gpu=*) echo "{GPU_A}, 0000:00:01.0"; echo "{GPU_B}, 0000:00:02.0" ;;
  *) echo "$PPID, {GPU_A}" ;;
esac
"""


def _hung(directory: Path, pid_file: Path) -> str:
    """Fake whose leader records its PID and becomes ``sleep`` (no orphan)."""

    return _fake(directory, "nvidia-smi", f"echo $$ >> {pid_file}\nexec sleep 30\n")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pids(pid_file: Path) -> list[int]:
    return [int(line) for line in pid_file.read_text(encoding="utf-8").split()] if pid_file.exists() else []


def _sampler_for(executable: str, **kwargs) -> gs.GpuSampler:
    """GpuSampler targeting this process with entitlement A."""

    kwargs.setdefault("cadence", 0.01)
    return gs.GpuSampler(os.getpid(), (GPU_A,), executable=executable, **kwargs)


def _finished(sampler: gs.GpuSampler) -> dict[str, object]:
    sampler.start()
    return sampler.finish()


def _wait_for(sampler: gs.GpuSampler, predicate, bound: float = 10.0) -> dict[str, object]:
    deadline = time.monotonic() + bound
    record = sampler.record()
    while not predicate(record) and time.monotonic() < deadline:
        time.sleep(0.01)
        record = sampler.record()
    return record


# ---------------------------------------------------------------- sampler


@POSIX_GROUPS
def test_sampler_launch_failure_is_attempted_telemetry_failed(tmp_path: Path) -> None:
    """A missing executable is attempted telemetry failure with provenance."""

    record = _finished(_sampler_for(str(tmp_path / "missing-nvidia-smi")))
    assert record["attempted"] is True
    assert record["reason"] == "telemetry-failed"
    assert record["failure"]["code"] == "launch-failed"
    assert "FileNotFoundError" in record["failure"]["detail"]
    assert record["command_pids"] == [] and record["quiescent"] is True
    assert record["target_pid"] == os.getpid()
    assert validate_sampler_wire(record)["reason"] == "telemetry-failed"


@POSIX_GROUPS
def test_sampler_one_session_mechanism_launches_commands(tmp_path: Path) -> None:
    """start_new_session alone yields observed, quiescent, target-attributed rows."""

    record = _finished(_sampler_for(_fake(tmp_path, "nvidia-smi", _OK_BODY)))
    assert record["reason"] == "observed", record["failure"]
    assert record["quiescent"] is True and record["survivors"] == []
    # One version query, then inventory + process per complete sample.
    assert len(record["command_pids"]) == 1 + 2 * record["aggregate"]["samples_taken"]
    assert record["samples"][0]["pid_rows"] == [{"pid": os.getpid(), "gpu_uuid": GPU_A}]
    assert record["aggregate"]["exact_pid_gpus"] == [{"uuid": GPU_A, "pci.bus_id": "0000:00:01.0"}]
    assert validate_sampler_wire(record) == record


def test_claimed_evidence_is_attempted_and_never_overwritten(tmp_path: Path) -> None:
    """The pre-spawn claim is blocking evidence; absent evidence is never not-sampled."""

    path = tmp_path / "gpu-sampler.json"
    missing = gs.read_sampler_evidence(path)
    assert missing["attempted"] is True and missing["reason"] == "telemetry-failed"
    assert missing["failure"]["code"] == "sampler-evidence-invalid"
    assert gs.claim_sampler_evidence(str(path)) is True
    claimed = path.read_bytes()
    record = gs.read_sampler_evidence(path)
    assert record["attempted"] is True and record["reason"] == "telemetry-failed"
    assert record["failure"]["code"] == "sampler-unfinalized" and record["quiescent"] is False
    # A second claimant (for example an inspection process) cannot replace it.
    assert gs.claim_sampler_evidence(str(path)) is False
    assert path.read_bytes() == claimed
    assert gs.claim_sampler_evidence(None) is False


@POSIX_GROUPS
def test_sampler_success_path_kills_lingering_descendant(tmp_path: Path, reaper) -> None:
    """D4/G2: a successful command's lingering descendant is killed and confirmed gone."""

    pids = reaper.track(tmp_path / "descendants")
    executable = _fake(tmp_path, "nvidia-smi", f"""\
        sleep 30 </dev/null >/dev/null 2>&1 &
        echo $! >> {pids}
        """ + _OK_BODY)
    record = _finished(_sampler_for(executable))
    descendants = _pids(pids)
    assert record["reason"] == "observed", record["failure"]
    assert record["quiescent"] is True and record["survivors"] == []
    assert len(descendants) == len(record["command_pids"]) >= 3
    # Confirmed at return: the sampler saw ESRCH for every group.
    assert [_alive(pid) for pid in descendants] == [False] * len(descendants)


@POSIX_GROUPS
def test_sampler_timeout_kills_and_confirms_retained_group(tmp_path: Path, pid_files: list[Path]) -> None:
    """A timed-out command group is killed by its retained id and reaped."""

    pid_file = tmp_path / "hung.pids"
    pid_files.append(pid_file)
    started = time.monotonic()
    record = _finished(_sampler_for(_hung(tmp_path, pid_file), command_timeout=0.3, cleanup_timeout=0.5, join_timeout=3.0))
    assert time.monotonic() - started < 3.0
    assert record["failure"]["code"] == "command-timeout"
    assert record["quiescent"] is True and record["survivors"] == []
    assert len(record["command_pids"]) == 1
    assert [_alive(pid) for pid in record["command_pids"]] == [False]


@POSIX_GROUPS
def test_sampler_read_error_path_cleans_retained_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """An OSError while reading still kills, reaps, and confirms the group."""

    pid_file = tmp_path / "hung.pids"
    pid_files.append(pid_file)
    original = subprocess.Popen.communicate
    calls = []

    def failing_once(self, *args, **kwargs):
        calls.append(kwargs.get("timeout"))
        if len(calls) == 1:
            raise OSError("injected read failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(gs.subprocess.Popen, "communicate", failing_once)
    record = _finished(_sampler_for(_hung(tmp_path, pid_file), command_timeout=5.0, cleanup_timeout=1.0))
    assert record["failure"]["code"] == "read-failed"
    assert record["quiescent"] is True and record["survivors"] == []
    assert len(record["command_pids"]) == 1
    assert [_alive(pid) for pid in record["command_pids"]] == [False]


@POSIX_GROUPS
def test_sampler_finish_cancels_and_drains_inflight_command(tmp_path: Path, pid_files: list[Path]) -> None:
    """At the shared deadline finish kills the in-flight group and drains it."""

    pid_file = tmp_path / "hung.pids"
    pid_files.append(pid_file)
    sampler = _sampler_for(_hung(tmp_path, pid_file), command_timeout=10.0, cleanup_timeout=0.3, join_timeout=1.0)
    started = time.monotonic()
    record = _finished(sampler)
    assert time.monotonic() - started < 1.5
    assert record["failure"]["code"] == "sampler-cancelled"
    assert record["quiescent"] is True and record["survivors"] == []
    assert len(record["command_pids"]) == 1
    assert [_alive(pid) for pid in record["command_pids"]] == [False]


@POSIX_GROUPS
def test_sampler_unconfirmed_group_is_survivor_and_not_quiescent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Quiescence requires ESRCH; an unconfirmed group is preserved as a survivor."""

    monkeypatch.setattr(gs, "group_empty", lambda _pgid: False)
    record = _finished(_sampler_for(_fake(tmp_path, "nvidia-smi", _OK_BODY), cleanup_timeout=0.1))
    assert record["failure"]["code"] == "cleanup-unconfirmed"
    assert record["quiescent"] is False
    assert record["survivors"] == record["command_pids"][:1]
    assert validate_sampler_wire(record)["quiescent"] is False


@POSIX_GROUPS
def test_sampler_finish_is_bounded_while_a_launch_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """D2/G3: a blocked Popen cannot hold cancel, record, or finish past the deadline.

    The late launch that completes after cancellation is owned by the
    sampling thread, which kills and reaps it itself.
    """

    pid_file = tmp_path / "hung.pids"
    pid_files.append(pid_file)
    executable = _hung(tmp_path, pid_file)
    release = threading.Event()
    entered = threading.Event()
    real_popen = subprocess.Popen

    class BlockingPopen(real_popen):
        def __init__(self, *args, **kwargs):
            entered.set()
            release.wait(10.0)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(gs.subprocess, "Popen", BlockingPopen)
    timer = threading.Timer(3.0, release.set)
    timer.start()
    try:
        sampler = _sampler_for(executable, command_timeout=10.0, cleanup_timeout=0.2, join_timeout=0.6)
        sampler.start()
        assert entered.wait(5.0)
        started = time.monotonic()
        snapshot = sampler.record()
        assert time.monotonic() - started < 0.2 and snapshot["failure"]["code"] == "sampler-unfinalized"
        record = sampler.finish()
        elapsed = time.monotonic() - started
        assert elapsed < 1.5, elapsed
        assert record["failure"]["code"] == "sampler-join-expired"
        assert "launch was still in flight" in record["failure"]["detail"]
        assert record["quiescent"] is False
        assert validate_sampler_wire(record)["quiescent"] is False
    finally:
        release.set()
        timer.cancel()
    # The late launch is killed, drained, and reaped by its owning thread.
    sampler._thread.join(5.0)
    assert not sampler._thread.is_alive()
    late = sampler.record()
    assert late["survivors"] == [] and len(late["command_pids"]) == 1
    # The late leader may be killed before its shell records itself.
    assert [_alive(pid) for pid in late["command_pids"]] == [False]


@POSIX_GROUPS
def test_sampler_keeps_samples_taken_before_a_late_failure(tmp_path: Path) -> None:
    """Historical N6 objective: a late failure preserves every earlier sample."""

    counter = tmp_path / "count"
    executable = _fake(tmp_path, "nvidia-smi", f"""\
        n=$(cat {counter} 2>/dev/null || echo 0); n=$((n + 1)); echo $n > {counter}
        if [ "$n" -gt 3 ]; then exit 1; fi
        """ + _OK_BODY)
    sampler = _sampler_for(executable)
    sampler.start()
    record = _wait_for(sampler, lambda value: value["failure"]["code"] != "sampler-unfinalized")
    record = sampler.finish()
    assert record["failure"]["code"] == "command-failed"
    assert record["count"] == 1 and record["aggregate"]["samples_taken"] == 1
    assert validate_sampler_wire(record)["samples"] == record["samples"]


@POSIX_GROUPS
def test_sampler_retains_a_bounded_window_and_a_final_sample(tmp_path: Path) -> None:
    """The lifetime sampler retains at most ``retain`` samples and samples once after stop."""

    sampler = _sampler_for(_fake(tmp_path, "nvidia-smi", _OK_BODY), retain=3)
    sampler.start()
    first = _wait_for(sampler, lambda value: value["aggregate"]["samples_taken"] >= 1)["samples"][0]
    _wait_for(sampler, lambda value: value["aggregate"]["samples_taken"] >= 6)
    stopped = time.monotonic()
    record = sampler.finish()
    assert record["reason"] == "observed", record["failure"]
    assert record["count"] == 3 and record["aggregate"]["samples_taken"] > 6
    # The first sample is pinned; the newest retained sample is the final one.
    assert record["samples"][0] == first
    assert record["samples"][-1]["monotonic_start"] >= stopped
    assert validate_sampler_wire(record) == record


@POSIX_GROUPS
def test_sampler_aggregate_remembers_evicted_outside_use(tmp_path: Path) -> None:
    """G4: outside use in a sample that is not retained stays in the aggregate."""

    counter = tmp_path / "process-queries"
    executable = _fake(tmp_path, "nvidia-smi", f"""\
        case "$1" in
          --version) echo "NVIDIA-SMI fake" ;;
          --query-gpu=*) echo "{GPU_A}, 0000:00:01.0"; echo "{GPU_B}, 0000:00:02.0" ;;
          *) n=$(cat {counter} 2>/dev/null || echo 0); n=$((n + 1)); echo $n > {counter}
             if [ "$n" -eq 1 ]; then :; elif [ "$n" -eq 2 ]; then echo "$PPID, {GPU_B}"; else echo "$PPID, {GPU_A}"; fi ;;
        esac
        """)
    sampler = _sampler_for(executable, retain=2)
    sampler.start()
    _wait_for(sampler, lambda value: value["aggregate"]["samples_taken"] >= 5)
    record = sampler.finish()
    assert record["reason"] == "observed", record["failure"]
    retained_rows = {row["gpu_uuid"] for sample in record["samples"] for row in sample["pid_rows"]}
    assert GPU_B not in retained_rows and record["count"] == 2
    assert record["aggregate"]["outside_use_seen"] is True
    assert [row["uuid"] for row in record["aggregate"]["exact_pid_gpus"]] == [GPU_B, GPU_A]
    decision = derive_device_decision(validate_sampler_wire(record), os.getpid(), (GPU_A,))
    assert decision.code == "out-of-entitlement" and decision.active_uuids == (GPU_B, GPU_A)


# ------------------------------------------------------ worker ownership


def _worker_env(tmp_path: Path, bin_dir: Path, **extra: str) -> PreparedPythonEnvironment:
    values = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        gs.GPU_SAMPLER_OUTPUT_ENV: str(tmp_path / "gpu-sampler.json"),
        "TPEN_PLACEMENT_DECLARED_DEVICES": GPU_A,
        gs.GPU_SAMPLER_CADENCE_ENV: "0.02",
    }
    values.update(extra)
    return PreparedPythonEnvironment(sys.executable, None, values)


def _evidence(tmp_path: Path) -> dict[str, object]:
    payload = json.loads((tmp_path / "gpu-sampler.json").read_text(encoding="utf-8"))
    assert payload["schema"] == gs.EVIDENCE_SCHEMA
    return validate_sampler_wire(payload["gpu_sampler"])


_PID_FILE_BODY = f"""\
case "$1" in
  --version) echo "NVIDIA-SMI fake" ;;
  --query-gpu=*) echo "{GPU_A}, 0000:00:01.0" ;;
  *) pid=$(cat "$L1B_CHILD_PID_FILE" 2>/dev/null)
     if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "$pid, {GPU_A}"; fi ;;
esac
"""


@POSIX_GROUPS
def test_worker_samples_the_exact_child_pid_for_its_lifetime(tmp_path: Path) -> None:
    """H-v2 guarantee: a context created late in the child's life is still observed.

    The child shows a GPU row only from 0.3 s after start (a late CUDA
    context).  The worker keeps sampling the exact PID until exit and takes a
    final sample after the exit, so the verdict is ``matched``.
    """

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", _PID_FILE_BODY)
    pid_file = tmp_path / "child.pid"
    child = f"import os, time; time.sleep(0.3); open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(0.6)"
    outcome = SubprocessWorker().run(
        _worker_env(tmp_path, bin_dir, L1B_CHILD_PID_FILE=str(pid_file)), ("-c", child),
        cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
    )
    record = _evidence(tmp_path)
    assert outcome.returncode == 0
    assert record["reason"] == "observed", record["failure"]
    assert record["target_pid"] == outcome.pid
    # Startup samples have no row; the final sample, after exit, has none either.
    assert record["samples"][0]["pid_rows"] == [] and record["samples"][-1]["pid_rows"] == []
    assert record["aggregate"]["exact_pid_gpus"] == [{"uuid": GPU_A, "pci.bus_id": "0000:00:01.0"}]
    assert derive_device_decision(record, outcome.pid, (GPU_A,)).code == "matched"
    assert [_alive(pid) for pid in record["command_pids"]] == [False] * len(record["command_pids"])


@POSIX_GROUPS
def test_worker_child_that_never_uses_a_gpu_is_blocking_telemetry_failure(tmp_path: Path) -> None:
    """A declared-GPU child that never creates a context has no exact-PID row: blocking."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", _PID_FILE_BODY)
    outcome = SubprocessWorker().run(
        _worker_env(tmp_path, bin_dir, L1B_CHILD_PID_FILE=str(tmp_path / "never")), ("-c", "import time; time.sleep(0.3)"),
        cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
    )
    record = _evidence(tmp_path)
    assert outcome.returncode == 0 and record["reason"] == "observed"
    assert record["aggregate"]["exact_pid_gpus"] == []
    assert derive_device_decision(record, outcome.pid, (GPU_A,)).code == "no-exact-pid-row"


@POSIX_GROUPS
def test_worker_timeout_drains_sampler_groups_before_return(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """D1/G1: after a worker timeout kills the child, no sampler group survives return."""

    monkeypatch.setattr(gs, "COMMAND_TIMEOUT", 10.0)
    monkeypatch.setattr(gs, "CLEANUP_TIMEOUT", 0.3)
    monkeypatch.setattr(gs, "JOIN_TIMEOUT", 1.0)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pid_file = tmp_path / "hung.pids"
    pid_files.append(pid_file)
    _hung(bin_dir, pid_file)
    started = time.monotonic()
    outcome = SubprocessWorker().run(
        _worker_env(tmp_path, bin_dir), ("-c", "import time; time.sleep(30)"),
        cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log", timeout=0.5,
    )
    # Checked immediately at return, before any grace period: the slot is free.
    alive = [_alive(pid) for pid in _pids(pid_file)]
    assert outcome.timed_out is True and time.monotonic() - started < 5.0
    assert alive == [False]
    record = _evidence(tmp_path)
    assert [_alive(pid) for pid in record["command_pids"]] == [False]
    assert record["failure"]["code"] == "sampler-cancelled"
    assert record["quiescent"] is True and record["survivors"] == []
    assert record["target_pid"] == outcome.pid


class _Cancel(BaseException):
    """Owner cancellation delivered into the worker thread."""


@POSIX_GROUPS
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX interval timers")
def test_worker_cancellation_drains_sampler_groups_before_propagating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """D1: an owner cancellation kills the child and drains the sampler before it propagates."""

    monkeypatch.setattr(gs, "COMMAND_TIMEOUT", 10.0)
    monkeypatch.setattr(gs, "CLEANUP_TIMEOUT", 0.3)
    monkeypatch.setattr(gs, "JOIN_TIMEOUT", 1.0)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pid_file = tmp_path / "hung.pids"
    pid_files.append(pid_file)
    _hung(bin_dir, pid_file)

    def cancel(_signum, _frame):
        raise _Cancel()

    previous = signal.signal(signal.SIGALRM, cancel)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        with pytest.raises(_Cancel):
            SubprocessWorker().run(
                _worker_env(tmp_path, bin_dir), ("-c", "import time; time.sleep(30)"),
                cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
            )
        alive = [_alive(pid) for pid in _pids(pid_file)]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert alive == [False]
    record = _evidence(tmp_path)
    assert [_alive(pid) for pid in record["command_pids"]] == [False]
    assert record["failure"]["code"] == "sampler-cancelled"
    assert record["quiescent"] is True and record["survivors"] == []


@POSIX_GROUPS
def test_worker_keeps_the_evidence_path_from_the_child_and_never_overwrites(tmp_path: Path) -> None:
    """The child never sees the evidence path; an existing evidence file is not replaced."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", _OK_BODY)
    probe = f"import json, os; print(json.dumps({gs.GPU_SAMPLER_OUTPUT_ENV!r} in os.environ))"
    SubprocessWorker().run(
        _worker_env(tmp_path, bin_dir), ("-c", probe),
        cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
    )
    assert json.loads((tmp_path / "out.log").read_text(encoding="utf-8")) is False
    first = (tmp_path / "gpu-sampler.json").read_bytes()
    assert json.loads(first)["gpu_sampler"]["attempted"] is True
    SubprocessWorker().run(
        _worker_env(tmp_path, bin_dir), ("-c", "pass"),
        cwd=tmp_path, stdout_path=tmp_path / "out2.log", stderr_path=tmp_path / "err2.log",
    )
    assert (tmp_path / "gpu-sampler.json").read_bytes() == first


@POSIX_GROUPS
def test_worker_without_an_evidence_path_does_not_sample(tmp_path: Path) -> None:
    """A slot the controller did not assign an evidence path is never sampled."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "invoked"
    _fake(bin_dir, "nvidia-smi", f"touch {marker}\n")
    environment = _worker_env(tmp_path, bin_dir)
    values = {key: value for key, value in environment.env.items() if key != gs.GPU_SAMPLER_OUTPUT_ENV}
    SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable, None, values), ("-c", "pass"),
        cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
    )
    assert not marker.exists() and not (tmp_path / "gpu-sampler.json").exists()


# ------------------------------------------------------- real RUN_SHIM child


_SIDECAR_BODY = f"""\
case "$1" in
  --version) echo "NVIDIA-SMI fake" ;;
  --query-gpu=*) echo "{GPU_A}, 0000:00:01.0" ;;
  *) pid=$(sed -n 's/.*"pid": \\([0-9][0-9]*\\).*/\\1/p' "$TPEN_PLACEMENT_OUTPUT" 2>/dev/null)
     if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "$pid, {GPU_A}"; fi ;;
esac
"""


def _shim_environment(tmp_path: Path, path_dir: Path, *, devices: str) -> PreparedPythonEnvironment:
    package = tmp_path / "pkg" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text("import time\ndef main(argv):\n    time.sleep(0.5)\n    return 0\n", encoding="utf-8")
    values = {
        "PYTHONPATH": str(package.parent), "PATH": f"{path_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "TPEN_PLACEMENT_OUTPUT": str(tmp_path / "child-placement.json"), "TPEN_PLACEMENT_OUTPUT_DIR": str(tmp_path),
        "TPEN_PLACEMENT_DECLARED_CPUS": str(sorted(os.sched_getaffinity(0))[0]),
        "TPEN_PLACEMENT_DECLARED_HOST": socket.gethostname(), "TPEN_PLACEMENT_DECLARED_DEVICES": devices,
        "TPEN_PLACEMENT_RECEIPT_VERSION": "2",
        "TPEN_PLACEMENT_TASK_ID": "task-a", "TPEN_PLACEMENT_ATTEMPT_ID": "attempt-a",
        "TPEN_PLACEMENT_GROUP_DIGEST": digest("s2b2"), "TPEN_PLACEMENT_ALLOCATION_ID": "allocation-a",
        "TPEN_PLACEMENT_BACKEND_JOB_ID": "job-a", "TPEN_PLACEMENT_SLOT_ID": "logical-slot",
        "TPEN_PLACEMENT_PAYLOAD_DIGEST": digest("payload"), gs.GPU_SAMPLER_CADENCE_ENV: "0.02",
    }
    if devices:
        # The controller assigns the worker evidence path only to GPU slots.
        values[gs.GPU_SAMPLER_OUTPUT_ENV] = str(tmp_path / "gpu-sampler.json")
    return PreparedPythonEnvironment(sys.executable, None, values)


def _run_shim(tmp_path: Path, path_dir: Path, *, devices: str = GPU_A) -> tuple[ProcessOutcome, dict[str, object]]:
    environment = _shim_environment(tmp_path, path_dir, devices=devices)
    outcome = SubprocessWorker().run(
        environment, ("-c", RUN_SHIM, f"run.root={tmp_path}"), cwd=tmp_path,
        stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
    )
    return outcome, json.loads((tmp_path / "child-placement.json").read_text(encoding="utf-8"))


@LINUX_CHILD
def test_real_shim_child_is_sampled_by_its_worker(tmp_path: Path) -> None:
    """The unchanged L1a child is sampled by the worker for its exact PID."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", _SIDECAR_BODY)
    outcome, payload = _run_shim(tmp_path, bin_dir)
    record = _evidence(tmp_path)
    assert outcome.returncode == 0
    assert payload["schema"] == "tpen-executor/child-placement-v1" and "gpu_sampler" not in payload
    assert record["reason"] == "observed", record["failure"]
    assert record["target_pid"] == outcome.pid == payload["pid"]
    assert record["quiescent"] is True and record["survivors"] == []
    assert {row["pid"] for sample in record["samples"] for row in sample["pid_rows"]} == {outcome.pid}
    assert derive_device_decision(record, payload["pid"], (GPU_A,)).code == "matched"


@LINUX_CHILD
def test_real_shim_missing_nvidia_smi_is_telemetry_failed(tmp_path: Path) -> None:
    """A declared-GPU child without nvidia-smi records attempted telemetry failure."""

    empty = tmp_path / "empty-bin"
    empty.mkdir()
    environment = _shim_environment(tmp_path, empty, devices=GPU_A)
    values = dict(environment.env, PATH=str(empty))
    outcome = SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable, None, values), ("-c", RUN_SHIM, f"run.root={tmp_path}"),
        cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
    )
    record = _evidence(tmp_path)
    assert outcome.returncode == 0
    assert record["attempted"] is True and record["reason"] == "telemetry-failed"
    assert record["failure"]["code"] == "launch-failed"


@LINUX_CHILD
def test_real_shim_cpu_only_slot_is_not_sampled(tmp_path: Path) -> None:
    """A CPU-only slot intentionally never samples and has no worker evidence."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "invoked"
    _fake(bin_dir, "nvidia-smi", f"touch {marker}\nexit 1\n")
    outcome, payload = _run_shim(tmp_path, bin_dir, devices="")
    assert outcome.returncode == 0
    assert payload["schema"] == "tpen-executor/child-placement-v1"
    assert not marker.exists() and not (tmp_path / "gpu-sampler.json").exists()


# ------------------------------------------------------------ v3 receipts


def _group() -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    return ReadyGroup("s2b2", (TaskSpec("task-a", "task-a", program, digest("task-a"), 1, "steps"),), digest("s2b2"))


def _allocation(devices: tuple[str, ...]) -> ConfiguredAllocation:
    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))
    slot = AllocationSlot("logical-slot", "logical-label", devices, (allowed[0],), (), ()).to_dict()
    return ConfiguredAllocation(
        AllocationId("allocation-a"), BackendJobId("job-a"),
        {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot]}, "scheduler": {"PBS_NODEFILE": "physical-01\n"}},
    )


def _write_evidence(path: Path, record: object) -> None:
    path.write_text(json.dumps({"schema": gs.EVIDENCE_SCHEMA, "gpu_sampler": record}) + "\n", encoding="utf-8")


def _receipt(
    tmp_path: Path, sampler: object | None, devices: tuple[str, ...] = (GPU_A,), *, sampled: bool | None = None,
) -> SciencePlacementReceiptV3:
    """Build a v3 receipt through the real builder from worker evidence.

    ``sampled`` (default: ``sampler is not None``) says whether the controller
    assigned an evidence path; ``sampler`` is written there when given, so
    ``sampler=None, sampled=True`` reads whatever file already exists.
    """

    sampled = sampler is not None if sampled is None else sampled

    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"), ("payload.seed=1",))
    allowed = sorted(getattr(os, "sched_getaffinity", lambda _pid: (0,))(0))
    fence = SimpleNamespace(attempt_id=AttemptId("attempt-a"), output_dir=str(tmp_path / "out"))
    values = {
        "TPEN_SLOT_ID": "logical-slot", "TPEN_SLOT_NODE": "physical-01", "TPEN_SLOT_CPUS": str(allowed[0]),
        "TPEN_PLACEMENT_DECLARED_DEVICES": ",".join(devices), "TPEN_PLACEMENT_RECEIPT_VERSION": "2",
        "TPEN_PLACEMENT_HOST_STATUS": "mapped",
    }
    if sampled:
        values[gs.GPU_SAMPLER_OUTPUT_ENV] = str(tmp_path / "gpu-sampler.json")
    if sampled and sampler is not None:
        _write_evidence(tmp_path / "gpu-sampler.json", sampler)
    base = ChildPlacementObservation(
        "task-a", "attempt-a", group.identity, "allocation-a", "job-a", "logical-slot",
        digest({"config": configured.config, "overrides": list(configured.overrides)}), fence.output_dir,
        "physical-01", 1234, 55, (allowed[0],), {}, None, "2026-11-10T00:00:00Z", None,
    )
    return _placement_receipt_v3(
        base, _allocation(devices), fence, group, group.members[0],
        configured, PreparedPythonEnvironment(sys.executable, None, values), ProcessOutcome((), 0, "", "", False, 1234, 55),
    )


def _sampler(
    samples: list[list[tuple[int, str]]],
    *,
    inventory: tuple[str, ...] = (GPU_A, GPU_B),
    failure: str | None = None,
    declared: tuple[str, ...] = (GPU_A,),
    dropped: list[list[tuple[int, str]]] = (),
    target: int = 1234,
) -> dict[str, object]:
    """Build a valid worker sampler record with one sample per row list.

    ``dropped`` lists earlier samples that were taken but not retained; they
    contribute only to the aggregate.
    """

    pci = {uuid: f"0000:00:0{index}.0" for index, uuid in enumerate(inventory, 1)}
    rows_inventory = [{"uuid": uuid, "pci.bus_id": pci[uuid]} for uuid in inventory]
    records = [
        {"monotonic_start": 1.0 + 0.2 * index, "monotonic_end": 1.1 + 0.2 * index, "inventory": rows_inventory,
         "pid_rows": [{"pid": pid, "gpu_uuid": uuid} for pid, uuid in rows]}
        for index, rows in enumerate(samples)
    ]
    exact: list[dict[str, str]] = []
    for rows in (*dropped, *samples):
        for pid, uuid in rows:
            if pid == target and uuid not in [item["uuid"] for item in exact]:
                exact.append({"uuid": uuid, "pci.bus_id": pci[uuid]})
    end = records[-1]["monotonic_end"] + 0.05 if records else 1.05
    return {
        "status": "unavailable" if failure else "observed", "reason": "telemetry-failed" if failure else "observed",
        "method": "nvidia-smi-pid-uuid-v1", "version": "NVIDIA-SMI fake" if records else None, "attempted": True,
        "failure": {"code": failure, "detail": "fake failure"} if failure else None, "cadence_seconds": 0.1,
        "count": len(records), "monotonic_start": 1.0, "monotonic_end": end, "samples": records,
        "command_pids": [101, 102, 103], "survivors": [], "elapsed_seconds": end - 1.0,
        "join_timeout_seconds": 4.0, "quiescent": True, "target_pid": target,
        "aggregate": {
            "samples_taken": len(records) + len(dropped), "exact_pid_gpus": exact,
            "outside_use_seen": any(item["uuid"] not in declared for item in exact),
            "entitlement_unresolved_seen": bool(records or dropped) and not set(declared) <= set(inventory),
        },
    }


def test_four_outcomes_are_typed_and_distinct(tmp_path: Path) -> None:
    """matched, mismatch, telemetry-failed, and not-sampled stay distinct and typed."""

    cases = {
        "qualified": (_sampler([[(1234, GPU_A)]]), (GPU_A,), "matched", "accepted"),
        "mismatch": (_sampler([[(1234, GPU_B)]]), (GPU_A,), "out-of-entitlement", "not-accepted"),
        "telemetry-failed": (_sampler([], failure="launch-failed"), (GPU_A,), "sampler-failed", "not-accepted"),
        "not-sampled": (None, (), "cpu-only", "accepted"),
    }
    for reason_class, (sampler, devices, code, acceptance) in cases.items():
        case_dir = tmp_path / reason_class
        case_dir.mkdir()
        receipt = _receipt(case_dir, sampler, devices)
        assert receipt.schema == "tpen-executor/science-placement-receipt-v3"
        assert receipt.verdict["reason_class"] == reason_class
        assert receipt.gpu["reason"] == code
        assert receipt.verdict["science_acceptance"] == acceptance
        assert receipt.verdict["qualification"] == ("qualified" if reason_class == "qualified" else "unqualified")
        assert SciencePlacementReceiptV3.from_mapping(json.loads(json.dumps(receipt.to_dict()))).to_dict() == receipt.to_dict()
    assert _receipt(tmp_path / "not-sampled", None, ()).to_dict()["gpu_sampler"] == not_sampled_sampler()


def test_out_of_slot_use_in_any_sample_dominates(tmp_path: Path) -> None:
    """Outside use by the exact PID in any retained sample wins over later evidence."""

    cases = {
        "earlier-B-later-A": (_sampler([[(1234, GPU_B)], [(1234, GPU_A)]]), (GPU_A,)),
        "B-then-late-failure": (_sampler([[(1234, GPU_B)]], failure="command-failed"), (GPU_A,)),
        "B-with-unresolved-C": (_sampler([[(1234, GPU_B)]], declared=(GPU_A, GPU_C)), (GPU_A, GPU_C)),
    }
    for name, (sampler, devices) in cases.items():
        case_dir = tmp_path / name
        case_dir.mkdir()
        receipt = _receipt(case_dir, sampler, devices)
        assert receipt.verdict["reason_class"] == "mismatch", name
        assert receipt.verdict["device"] == "mismatch", name
        assert GPU_B in receipt.gpu["active_uuids"], name
        assert len(receipt.gpu_sampler["samples"]) == len(sampler["samples"]), name
        raw = receipt.to_dict()
        assert SciencePlacementReceiptV3.from_mapping(raw).verdict["reason_class"] == "mismatch", name
        # Relabelling the out-of-slot evidence as qualified is rejected.
        raw["verdict"]["reason_class"] = "qualified"
        with pytest.raises((TypeError, ValueError)):
            SciencePlacementReceiptV3.from_mapping(raw)


def test_outside_use_in_an_evicted_sample_dominates(tmp_path: Path) -> None:
    """G5: outside use seen only in a sample that was not retained is still mismatch."""

    sampler = _sampler([[(1234, GPU_A)], [(1234, GPU_A)]], dropped=[[(1234, GPU_B)]])
    assert sampler["aggregate"]["outside_use_seen"] is True
    receipt = _receipt(tmp_path, sampler)
    assert receipt.verdict["reason_class"] == "mismatch"
    assert receipt.gpu["reason"] == "out-of-entitlement"
    assert list(receipt.gpu["active_uuids"]) == [GPU_B, GPU_A]
    raw = receipt.to_dict()
    assert SciencePlacementReceiptV3.from_mapping(copy.deepcopy(raw)).verdict["reason_class"] == "mismatch"
    # Hiding the sticky flag while keeping the GPU is contradictory.
    raw["gpu_sampler"]["aggregate"]["outside_use_seen"] = False
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV3.from_mapping(raw)


def _set_all_rows(raw: dict, key: str, value: object) -> None:
    for sample in raw["gpu_sampler"]["samples"]:
        for row in sample["pid_rows"]:
            row[key] = value


def _blank_active_pci(raw: dict) -> None:
    """D3: the active UUID's PCI is whitespace everywhere it appears."""

    for sample in raw["gpu_sampler"]["samples"]:
        for row in sample["inventory"]:
            if row["uuid"] == GPU_A:
                row["pci.bus_id"] = " "
    raw["gpu_sampler"]["aggregate"]["exact_pid_gpus"][0]["pci.bus_id"] = " "
    raw["gpu"]["active_pci"] = [" "]


CONTRADICTIONS = {
    "quiescent-false": lambda raw: raw["gpu_sampler"].__setitem__("quiescent", False),
    "survivors": lambda raw: raw["gpu_sampler"].update(survivors=[101], quiescent=False),
    "foreign-pid-rows": lambda raw: _set_all_rows(raw, "pid", 9999),
    "outside-uuid": lambda raw: _set_all_rows(raw, "gpu_uuid", GPU_B),
    "row-absent-from-inventory": lambda raw: raw["gpu_sampler"]["samples"][0]["pid_rows"].append({"pid": 1234, "gpu_uuid": GPU_C}),
    "unresolved-entitlement": lambda raw: raw["declared"].__setitem__("devices", [GPU_A, GPU_C]),
    "sample-outside-outer": lambda raw: raw["gpu_sampler"]["samples"][0].update(monotonic_start=100.0, monotonic_end=101.0),
    "reversed-sample": lambda raw: raw["gpu_sampler"]["samples"][0].update(monotonic_start=1.1, monotonic_end=1.0),
    "null-outer-interval": lambda raw: raw["gpu_sampler"].update(monotonic_start=None, monotonic_end=None),
    "null-version": lambda raw: raw["gpu_sampler"].__setitem__("version", None),
    "elapsed-disagrees": lambda raw: raw["gpu_sampler"].__setitem__("elapsed_seconds", 9.0),
    "qualified-over-failed-sampler": lambda raw: raw["gpu_sampler"].update(status="unavailable", reason="telemetry-failed", failure={"code": "command-failed", "detail": "x"}),
    "attempted-not-sampled": lambda raw: raw["gpu_sampler"].update(status="unavailable", reason="not-sampled"),
    "unknown-failure-code": lambda raw: raw["gpu_sampler"].update(status="unavailable", reason="telemetry-failed", failure={"code": "invented", "detail": "x"}),
    "extra-sampler-key": lambda raw: raw["gpu_sampler"].__setitem__("extra", 1),
    "relabelled-mismatch": lambda raw: raw["verdict"].__setitem__("reason_class", "mismatch"),
    "active-uuid-tamper": lambda raw: raw["gpu"].__setitem__("active_uuids", [GPU_B]),
    "warning-tamper": lambda raw: raw["verdict"].__setitem__("warning", "quiet"),
    "inventory-pci-whitespace": _blank_active_pci,
    "target-pid-tamper": lambda raw: raw["gpu_sampler"].__setitem__("target_pid", 4321),
    "aggregate-outside-flag-tamper": lambda raw: raw["gpu_sampler"]["aggregate"].__setitem__("outside_use_seen", True),
    "aggregate-hides-dropped-outside": lambda raw: raw["gpu_sampler"]["aggregate"].update(
        samples_taken=raw["gpu_sampler"]["count"] + 1,
        exact_pid_gpus=raw["gpu_sampler"]["aggregate"]["exact_pid_gpus"] + [{"uuid": GPU_B, "pci.bus_id": "0000:00:02.0"}],
    ),
    "aggregate-omits-retained-gpu": lambda raw: raw["gpu_sampler"]["aggregate"].__setitem__("exact_pid_gpus", []),
    "aggregate-below-count": lambda raw: raw["gpu_sampler"]["aggregate"].__setitem__("samples_taken", 0),
    "aggregate-unresolved-tamper": lambda raw: raw["gpu_sampler"]["aggregate"].__setitem__("entitlement_unresolved_seen", True),
}


@pytest.mark.parametrize("name", sorted(CONTRADICTIONS))
def test_decoder_rejects_contradictory_sampler_evidence(tmp_path: Path, name: str) -> None:
    """Every tampered qualified receipt is rejected by the shared derivation."""

    raw = _receipt(tmp_path, _sampler([[(1234, GPU_A)]])).to_dict()
    assert SciencePlacementReceiptV3.from_mapping(copy.deepcopy(raw)).verdict["reason_class"] == "qualified"
    CONTRADICTIONS[name](raw)
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV3.from_mapping(raw)


def test_decoder_rejects_whitespace_failure_detail(tmp_path: Path) -> None:
    """D3: a whitespace-only failure detail is rejected, matching P8."""

    raw = _receipt(tmp_path, _sampler([], failure="command-failed")).to_dict()
    assert SciencePlacementReceiptV3.from_mapping(copy.deepcopy(raw)).verdict["reason_class"] == "telemetry-failed"
    raw["gpu_sampler"]["failure"]["detail"] = " "
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV3.from_mapping(raw)


def test_observed_but_unusable_telemetry_is_typed_failure(tmp_path: Path) -> None:
    """Usable-telemetry failures are typed receipts that keep the retained samples."""

    cases = {
        "no-exact-pid-row": (_sampler([[]]), (GPU_A,)),
        "foreign-only": (_sampler([[(9999, GPU_A)]]), (GPU_A,)),
        "entitlement-unresolved": (_sampler([[(1234, GPU_A)]], declared=(GPU_A, GPU_C)), (GPU_A, GPU_C)),
    }
    for name, (sampler, devices) in cases.items():
        case_dir = tmp_path / name
        case_dir.mkdir()
        receipt = _receipt(case_dir, sampler, devices)
        code = "no-exact-pid-row" if name == "foreign-only" else name
        assert receipt.verdict["reason_class"] == "telemetry-failed", name
        assert receipt.verdict["science_acceptance"] == "not-accepted", name
        assert receipt.gpu["reason"] == code, name
        assert receipt.refusal is None, name
        assert receipt.to_dict()["gpu_sampler"] == sampler, name


def test_missing_or_invalid_worker_evidence_becomes_typed_failure(tmp_path: Path) -> None:
    """A sampled slot without valid evidence is typed telemetry failure, not not-sampled."""

    for name in ("invalid-record", "wrong-envelope", "missing"):
        case_dir = tmp_path / name
        case_dir.mkdir()
        if name == "invalid-record":
            receipt = _receipt(case_dir, {"status": "observed"})
        else:
            if name == "wrong-envelope":
                (case_dir / "gpu-sampler.json").write_text(json.dumps({"gpu_sampler": {}}), encoding="utf-8")
            # "missing": the worker never wrote evidence (it died before claiming).
            receipt = _receipt(case_dir, None, sampled=True)
        assert receipt.verdict["reason_class"] == "telemetry-failed", name
        assert receipt.gpu_sampler["failure"]["code"] == "sampler-evidence-invalid", name
        assert receipt.gpu_sampler["attempted"] is True, name


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="requires Linux controller affinity contract")
def test_dispatch_records_typed_telemetry_failure_not_refusal(tmp_path: Path) -> None:
    """A no-row worker record crosses dispatch as telemetry-failed, rc 0, accepted segment.

    The controller assigns the evidence path to the science launch only; the
    inspection launch never receives it.
    """

    group = _group()
    configured = ConfiguredRun("task-a", str(tmp_path / "config.yaml"))
    (tmp_path / "config.yaml").write_text("ok", encoding="utf-8")
    sampler = _sampler([[(9999, GPU_A)]], target=123)
    seen: dict[str, object] = {}

    class Process:
        def run(self, environment, *args, **kwargs):
            values = environment.env
            seen["science"] = values.get(gs.GPU_SAMPLER_OUTPUT_ENV)
            base = ChildPlacementObservation(
                values["TPEN_PLACEMENT_TASK_ID"], values["TPEN_PLACEMENT_ATTEMPT_ID"], values["TPEN_PLACEMENT_GROUP_DIGEST"],
                values["TPEN_PLACEMENT_ALLOCATION_ID"], values["TPEN_PLACEMENT_BACKEND_JOB_ID"], values["TPEN_PLACEMENT_SLOT_ID"],
                digest({"config": configured.config, "overrides": []}), values["TPEN_PLACEMENT_OUTPUT_DIR"],
                "physical-01", 123, 55, (int(values["TPEN_PLACEMENT_DECLARED_CPUS"]),), {}, None, "2026-11-10T00:00:00Z", None,
            )
            Path(values["TPEN_PLACEMENT_OUTPUT"]).write_text(json.dumps(base.to_dict()) + "\n", encoding="utf-8")
            _write_evidence(Path(values[gs.GPU_SAMPLER_OUTPUT_ENV]), sampler)
            Path(values["TPEN_PLACEMENT_OUTPUT_DIR"], "core").mkdir(parents=True, exist_ok=True)
            Path(tmp_path / "checkpoint").mkdir(parents=True, exist_ok=True)
            return ProcessOutcome((sys.executable,), 0, "", "", False, 123, 55)

    class Evidence:
        def read(self, _process, environment, run_dir, **_kwargs):
            seen["inspection"] = environment.env.get(gs.GPU_SAMPLER_OUTPUT_ENV)
            return CoreRunEvidence(
                "tpen.run-evidence/v1", str(run_dir), "completed", "completed",
                {"checkpoint_dir": str(tmp_path / "checkpoint"), "schema_version": 1, "next_iteration": 1, "completed_updates": 1}, None,
            )

    worker = SimpleNamespace(materialize=lambda _env=None: {
        "environment": PreparedPythonEnvironment(sys.executable, None, {gs.GPU_SAMPLER_OUTPUT_ENV: str(tmp_path / "inherited.json")}),
        "process": Process(), "runner": object(), "evidence": Evidence(),
    })
    store = DirectoryStore(tmp_path / "store")
    manifest = LocalDispatcher(worker).run(group, {"task-a": configured}, AttachedAllocation(_allocation((GPU_A,)), store))
    attempt = manifest.attempts[0]
    # An inherited path is discarded; the controller's path is in the fenced output.
    assert seen["science"] == os.path.join(attempt.output_dir, "gpu-sampler.json")
    assert seen["inspection"] is None
    assert attempt.outcome_receipt.status == "completed"
    assert _operational_exit(manifest) == 0
    assert manifest.segments
    assert isinstance(attempt.placement_receipt, SciencePlacementReceiptV3)
    assert attempt.placement_receipt.verdict["reason_class"] == "telemetry-failed"
    assert attempt.placement_receipt.refusal is None
    assert manifest.placement_verdict["message"] == PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE
    assert store.manifest(group).attempts[0].placement_receipt.to_dict() == attempt.placement_receipt.to_dict()


def test_untouched_l1a_v2_bytes_decode_and_v2_stays_strict() -> None:
    """L1a v2 bytes (produced at f384f44d) still decode; v2 and v3 stay closed."""

    fixtures = json.loads((FIXTURES / "science-placement-receipts-l1a-v2.json").read_text(encoding="utf-8"))
    assert set(fixtures) == {"not-sampled", "mismatch"}
    for name, raw in fixtures.items():
        receipt = decode_placement_receipt(copy.deepcopy(raw))
        assert isinstance(receipt, SciencePlacementReceiptV2), name
        assert receipt.to_dict() == raw, name
        assert receipt.verdict["reason_class"] == name
        assert receipt.base.schema == SciencePlacementReceipt.schema
        with_sampler = dict(raw, gpu_sampler=not_sampled_sampler())
        with pytest.raises((TypeError, ValueError)):
            SciencePlacementReceiptV2.from_mapping(with_sampler)
        with pytest.raises((TypeError, ValueError)):
            SciencePlacementReceiptV3.from_mapping(dict(raw, schema=SciencePlacementReceiptV3.schema))
        with pytest.raises((TypeError, ValueError)):
            decode_placement_receipt(dict(raw, schema="tpen-executor/science-placement-receipt-v9"))


def _manifest_with(receipt: SciencePlacementReceiptV3) -> RunManifest:
    start = StartReceipt("task-a", receipt.attempt_id, AllocationId("allocation-a"), BackendJobId("job-a"), receipt.group_digest, "2026-11-10T00:00:00Z", receipt.output_dir)
    outcome = OutcomeReceipt("task-a", receipt.attempt_id, receipt.group_digest, "2026-11-10T00:00:01Z", "completed", receipt.output_dir)
    return RunManifest("s2b2", receipt.group_digest, (ManifestAttempt("task-a", receipt.attempt_id, start, outcome, receipt.output_dir, receipt),))


def test_manifest_message_is_derived_from_one_source(tmp_path: Path) -> None:
    """Every blocking reason serializes the shared message constant."""

    # The L1a mismatch/refusal message is pinned: L1a manifests must stay readable.
    assert PLACEMENT_UNQUALIFIED_MESSAGE == "GPU PLACEMENT UNQUALIFIED: placement mismatch/refusal requires science review."
    assert PLACEMENT_NOT_SAMPLED_MESSAGE.startswith("GPU PLACEMENT NOT-SAMPLED")
    cases = {
        "matched": (_sampler([[(1234, GPU_A)]]), (GPU_A,), None),
        "mismatch": (_sampler([[(1234, GPU_B)]]), (GPU_A,), PLACEMENT_UNQUALIFIED_MESSAGE),
        "evicted-mismatch": (_sampler([[(1234, GPU_A)]], dropped=[[(1234, GPU_B)]]), (GPU_A,), PLACEMENT_UNQUALIFIED_MESSAGE),
        "sampler-failed": (_sampler([], failure="command-failed"), (GPU_A,), PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
        "no-exact-pid-row": (_sampler([[]]), (GPU_A,), PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
        "entitlement-unresolved": (_sampler([[(1234, GPU_A)]], declared=(GPU_A, GPU_C)), (GPU_A, GPU_C), PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE),
    }
    for name, (sampler, devices, message) in cases.items():
        case_dir = tmp_path / name
        case_dir.mkdir()
        manifest = _manifest_with(_receipt(case_dir, sampler, devices))
        decoded = RunManifest.decode(manifest.encode())
        if message is None:
            assert decoded.placement_verdict == {}, name
            assert json.loads(manifest.encode())["schema"] == "tpen-executor/run-manifest-v2"
        else:
            assert decoded.placement_verdict["message"] == message, name
            assert decoded.placement_verdict["status"] == "unqualified", name


V3_FIXTURE_CASES = {
    "matched", "out-of-entitlement", "evicted-out-of-entitlement", "sampler-failed",
    "no-exact-pid-row", "entitlement-unresolved", "not-sampled",
}


def test_checked_in_v3_reader_fixture_is_executor_valid() -> None:
    """The P8 v3 fixture is executor producer output and still decodes byte-for-byte."""

    path = Path(__file__).parents[2] / "experiments" / "toolkit" / "fixtures" / "run-manifests-v3-sampler.json"
    if not path.is_file():
        pytest.skip("named skip: repository experiments fixtures are not present")
    manifests = json.loads(path.read_text(encoding="utf-8"))
    assert set(manifests) == V3_FIXTURE_CASES
    for name, manifest in manifests.items():
        decoded = RunManifest.decode(json.dumps(manifest))
        assert decoded.to_dict() == manifest, name
        assert isinstance(decoded.attempts[0].placement_receipt, SciencePlacementReceiptV3), name
