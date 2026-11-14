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
retired, not renamed.  Of the r3 ids, G1 (worker timeout drain), G3 (bounded
join) and G5 (decoder aggregate) are retired in r4 and remain covered by
their named behaviour tests; G2 is kept unchanged.  H1 (r4, worker ownership
at startup, E1) is kept.  In r5, H2 (the r4 post-finish fence, whose anchor
no longer exists), G4 (aggregate stickiness) and H3 (total gap budget) are
retired and remain covered by
``test_worker_fences_the_slot_while_a_held_launch_is_outstanding``,
``test_sampler_aggregate_remembers_evicted_outside_use`` and
``test_sampler_isolated_gaps_past_the_total_budget_fail``.  J1-J3 are the r5
runtime-path mutants for the pre-spawn slot lease (delta-3 F2), the durable
quarantine marker (F1) and the decoder gap timeline (F3).  Historical
N6/N8/N11 remain unclosed by any mutant id; N6/N8 objectives are covered by
named behaviour tests (``test_sampler_keeps_samples_taken_before_a_late_failure``
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
from tpen_executor.dispatch.contract import _admit_slot, _placement_receipt_v3, _with_child_placement, _without_gpu_sampler
from tpen_executor.dispatch.local import LocalDispatcher
from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.gpu_telemetry import (
    MAX_JOIN_TIMEOUT,
    derive_device_decision,
    empty_gaps,
    not_sampled_sampler,
    validate_sampler_wire,
)
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
from tpen_executor.runners import CoreRunEvidence, EvidenceUnavailable, RUN_SHIM, TPENEvidence
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor import workers as workers_module
from tpen_executor.workers import LaunchPurpose, ProcessOutcome, SubprocessWorker

GPU_A = "GPU-12345678-1234-1234-1234-123456789ABC"
GPU_B = "GPU-12345678-1234-1234-1234-123456789ABD"
GPU_C = "GPU-12345678-1234-1234-1234-123456789ABE"
FIXTURES = Path(__file__).parent / "fixtures"
POSIX_GROUPS = pytest.mark.skipif(not hasattr(os, "killpg"), reason="requires POSIX process groups")
LINUX_PROC_GROUPS = pytest.mark.skipif(
    not Path("/proc/self/stat").is_file(), reason="requires Linux /proc process-group visibility",
)
LINUX_CHILD = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not hasattr(os, "sched_setaffinity"),
    reason="requires the Linux child affinity contract",
)

L1B_MUTANT_TABLE = (
    {
        "id": "J1", "target": "executor/src/tpen_executor/workers.py",
        "old": "            if sampler_output is not None:\n                lease = acquire_slot_lease(",
        "new": "            if False:\n                lease = acquire_slot_lease(",
        "killer": "test_interrupted_evidence_publication_keeps_the_slot_quarantined",
        "suite": "executor", "assertion": "the slot is quarantined before spawn, so an interrupted evidence publication cannot leave it reusable (delta-3 F2)",
    },
    {
        "id": "J2", "target": "executor/src/tpen_executor/gpu_sampler.py",
        "old": "    foreign = [marker for marker in _markers(quarantine_dir) if marker not in held]",
        "new": "    foreign = []",
        "killer": "test_durable_quarantine_refuses_a_fresh_controller_on_the_same_store",
        "suite": "executor", "assertion": "a durable marker in the attempt store makes a fresh controller refuse the slot (delta-3 F1)",
    },
    {
        "id": "K1", "target": "executor/src/tpen_executor/workers.py",
        "old": "        inspector = purpose is LaunchPurpose.INSPECTION\n",
        "new": "        inspector = purpose is LaunchPurpose.INSPECTION or GPU_SAMPLER_OUTPUT_ENV not in environment.env\n",
        "killer": "test_legacy_science_fallback_is_refused_on_a_held_slot",
        "suite": "executor", "assertion": "an unsampled launch, such as the legacy science fallback, is never inferred to be an inspection (delta-4 D4-1)",
    },
    {
        "id": "K2", "target": "executor/src/tpen_executor/gpu_telemetry.py",
        "old": "                if newest_start is None or event_end > newest_start or (first_end is not None and previous_end < first_end):\n                    raise ValueError(\"gpu_sampler gap run resets where no dropped successful sample can lie\")",
        "new": "                if False:\n                    raise ValueError(\"gpu_sampler gap run resets where no dropped successful sample can lie\")",
        "killer": "test_decoder_checks_a_coherent_gap_timeline",
        "suite": "executor", "assertion": "a dropped success witnesses a reset only inside its eviction window (delta-4 D4-2)",
    },
    {
        "id": "K3", "target": "executor/src/tpen_executor/gpu_telemetry.py",
        "old": "    if exceeded and any(success_start >= previous_end for success_start, _success_end in successes):\n        raise ValueError(\"gpu_sampler gap budget excess is followed by a successful sample\")",
        "new": "    if False:\n        raise ValueError(\"gpu_sampler gap budget excess is followed by a successful sample\")",
        "killer": "test_decoder_checks_a_coherent_gap_timeline",
        "suite": "executor", "assertion": "no successful sample follows the attempt that exceeded a gap budget (delta-4 D4-2)",
    },
    {
        "id": "L1", "target": "executor/src/tpen_executor/workers.py",
        "old": "        group_reaped = _group_quiescent(spawned_pid)\n        interruptions = finish_gpu_sampler(sampler, sampler_output, lease, group_reaped=group_reaped)",
        "new": "        group_reaped = True\n        interruptions = finish_gpu_sampler(sampler, sampler_output, lease, group_reaped=group_reaped)",
        "killer": "test_group_quiescence_unconfirmed_retains_quarantine_and_refuses_fresh_admission",
        "suite": "executor", "assertion": "the quarantine marker is never released on a sampler's own clean command cleanup alone; science-group quiescence must be independently observed before settling the lease (review B2)",
    },
)


def test_l1b_mutant_table_targets_unique_executable_anchors() -> None:
    """Every mutant names a unique executable anchor, parses, and has a killer."""

    root = Path(__file__).parents[2]
    module = Path(__file__).read_text(encoding="utf-8")
    assert [row["id"] for row in L1B_MUTANT_TABLE] == ["J1", "J2", "K1", "K2", "K3", "L1"]
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
    assert "was not found on the prepared PATH" in record["failure"]["detail"]
    assert record["executable"] is None
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
    # The budget in force is recorded even when no gap occurred.
    assert record["gaps"] == empty_gaps(gs.MAX_CONSECUTIVE_GAPS, gs.MAX_TOTAL_GAPS)
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
    # A zero gap budget makes the first transient failure terminal.
    record = _finished(_sampler_for(
        _hung(tmp_path, pid_file), command_timeout=0.3, cleanup_timeout=0.5, join_timeout=3.0,
        max_consecutive_gaps=0, max_total_gaps=0,
    ))
    assert time.monotonic() - started < 3.0
    assert record["failure"]["code"] == "gap-budget-exceeded"
    assert [event["code"] for event in record["gaps"]["events"]] == ["command-timeout"]
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
    record = _finished(_sampler_for(
        _hung(tmp_path, pid_file), command_timeout=5.0, cleanup_timeout=1.0, max_consecutive_gaps=0, max_total_gaps=0,
    ))
    assert record["failure"]["code"] == "gap-budget-exceeded"
    assert [event["code"] for event in record["gaps"]["events"]] == ["read-failed"]
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
    # Unconfirmed cleanup is never forgiven by the gap budget.
    assert record["gaps"]["count"] == 0
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
    # Persistent failure: the second consecutive gap exceeds the budget.
    assert record["failure"]["code"] == "gap-budget-exceeded"
    assert [event["code"] for event in record["gaps"]["events"]] == ["command-failed", "command-failed"]
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


# ------------------------------------------------ gap budget, MIG, deadline


def _counted(directory: Path, counter: Path, fail_when: str, body: str = _OK_BODY) -> str:
    """Fake that counts its invocations in ``counter`` and exits 1 when ``fail_when`` holds."""

    return _fake(directory, "nvidia-smi", f"""\
        n=$(cat {counter} 2>/dev/null || echo 0); n=$((n + 1)); echo $n > {counter}
        if {fail_when}; then exit 1; fi
        """ + body)


@POSIX_GROUPS
def test_sampler_forgives_an_isolated_transient_gap_and_records_it(tmp_path: Path) -> None:
    """Q1: one failed attempt inside the budget is a recorded gap, not a failure.

    Invocations: 1 version, 2-3 first sample, 4 inventory, 5 process (fails).
    """

    executable = _counted(tmp_path, tmp_path / "count", '[ "$n" -eq 5 ]')
    sampler = _sampler_for(executable)
    sampler.start()
    _wait_for(sampler, lambda value: value["aggregate"]["samples_taken"] >= 3)
    record = sampler.finish()
    assert record["reason"] == "observed", record["failure"]
    gaps = record["gaps"]
    assert (gaps["max_consecutive_allowed"], gaps["max_total_allowed"]) == (gs.MAX_CONSECUTIVE_GAPS, gs.MAX_TOTAL_GAPS)
    assert (gaps["count"], gaps["max_consecutive"]) == (1, 1)
    assert [(event["code"], event["consecutive"]) for event in gaps["events"]] == [("command-failed", 1)]
    assert gaps["total_seconds"] == gaps["events"][0]["monotonic_end"] - gaps["events"][0]["monotonic_start"] > 0
    assert validate_sampler_wire(record) == record


@POSIX_GROUPS
def test_sampler_isolated_gaps_past_the_total_budget_fail(tmp_path: Path) -> None:
    """Q1/H3: isolated gaps never exceed the consecutive budget but do exceed the total.

    Every fourth invocation fails, so each gap is followed by a complete
    sample; the fourth gap passes ``MAX_TOTAL_GAPS`` and is terminal.
    """

    assert gs.MAX_TOTAL_GAPS == 3 and gs.MAX_CONSECUTIVE_GAPS == 1
    executable = _counted(tmp_path, tmp_path / "count", '[ $((n % 4)) -eq 0 ]')
    sampler = _sampler_for(executable)
    sampler.start()
    _wait_for(sampler, lambda value: value["gaps"]["count"] > gs.MAX_TOTAL_GAPS or value["failure"]["code"] != "sampler-unfinalized")
    record = sampler.finish()
    assert record["failure"]["code"] == "gap-budget-exceeded", record["failure"]
    assert record["gaps"]["count"] == gs.MAX_TOTAL_GAPS + 1 and record["gaps"]["max_consecutive"] == 1
    # Telemetry failure is still attempted, never relabelled not-sampled.
    assert record["attempted"] is True and record["reason"] == "telemetry-failed"
    assert record["aggregate"]["samples_taken"] >= gs.MAX_TOTAL_GAPS
    assert validate_sampler_wire(record) == record


@POSIX_GROUPS
def test_sampler_never_forgives_malformed_output(tmp_path: Path) -> None:
    """Q1: malformed output is terminal even with budget to spare."""

    counter = tmp_path / "count"
    executable = _fake(tmp_path, "nvidia-smi", f"""\
        n=$(cat {counter} 2>/dev/null || echo 0); n=$((n + 1)); echo $n > {counter}
        if [ "$n" -eq 5 ]; then echo "not, a, row"; exit 0; fi
        """ + _OK_BODY)
    sampler = _sampler_for(executable)
    sampler.start()
    record = _wait_for(sampler, lambda value: value["failure"]["code"] != "sampler-unfinalized")
    record = sampler.finish()
    assert record["failure"]["code"] == "malformed-output"
    assert record["gaps"]["count"] == 0
    assert validate_sampler_wire(record)["failure"]["code"] == "malformed-output"


@POSIX_GROUPS
@pytest.mark.parametrize("where", ["inventory", "process"])
def test_sampler_mig_identifiers_are_a_distinct_unsupported_failure(tmp_path: Path, where: str) -> None:
    """Q5: a MIG identifier is the typed ``mig-unsupported`` failure, not malformed output."""

    mig = "MIG-12345678-1234-1234-1234-123456789ABC"
    inventory = f'echo "{mig}, 0000:00:01.0"' if where == "inventory" else f'echo "{GPU_A}, 0000:00:01.0"'
    executable = _fake(tmp_path, "nvidia-smi", f"""\
        case "$1" in
          --version) echo "NVIDIA-SMI fake" ;;
          --query-gpu=*) {inventory} ;;
          *) echo "$PPID, {mig}" ;;
        esac
        """)
    record = _finished(_sampler_for(executable))
    assert record["failure"]["code"] == "mig-unsupported", record["failure"]
    assert mig in record["failure"]["detail"]
    assert record["gaps"]["count"] == 0
    receipt = _receipt(tmp_path, _sampler([], failure="mig-unsupported"))
    assert receipt.verdict["reason_class"] == "telemetry-failed"
    assert receipt.verdict["reason"] == "telemetry-failed: attempted GPU sampler failed (mig-unsupported)"


def test_gap_budget_excess_is_a_typed_receipt_failure(tmp_path: Path) -> None:
    """Q1: an exceeded budget derives ``telemetry-failed`` with its own reason; relabelling is rejected."""

    gaps = empty_gaps(1, 3)
    raw = {"gpu_sampler": {"gaps": gaps}}
    _add_gaps(raw, ("command-failed", 1.0, 1.01, 1), ("command-failed", 1.02, 1.03, 2))
    receipt = _receipt(tmp_path, _sampler([], failure="gap-budget-exceeded", gaps=gaps))
    assert receipt.verdict["reason"] == "telemetry-failed: attempted GPU sampler failed (gap-budget-exceeded)"
    tampered = receipt.to_dict()
    tampered["gpu_sampler"]["failure"]["code"] = "command-failed"
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV3.from_mapping(tampered)


def test_finish_deadline_fits_two_worst_case_attempts() -> None:
    """Orchestrator decision (a): the finish bound is derived from the command bounds.

    An attempt in progress plus a final attempt, each of ``ATTEMPT_COMMANDS``
    commands bounded by read plus cleanup, plus the cancel reserve.
    """

    worst_attempt = gs.ATTEMPT_COMMANDS * (gs.COMMAND_TIMEOUT + gs.CLEANUP_TIMEOUT)
    reserve = gs.CLEANUP_TIMEOUT + gs.SCHEDULING_MARGIN
    assert gs.JOIN_TIMEOUT == gs.finish_deadline(gs.COMMAND_TIMEOUT, gs.CLEANUP_TIMEOUT) == 2 * worst_attempt + reserve
    assert (gs.COMMAND_TIMEOUT, gs.CLEANUP_TIMEOUT, gs.ATTEMPT_COMMANDS, gs.JOIN_TIMEOUT) == (5.0, 2.0, 3, 45.0)
    assert gs.JOIN_TIMEOUT <= MAX_JOIN_TIMEOUT
    sampler = gs.GpuSampler.from_environment(1, {})
    assert sampler.join_timeout == gs.JOIN_TIMEOUT and sampler.command_timeout == gs.COMMAND_TIMEOUT


@POSIX_GROUPS
def test_sampler_final_sample_survives_one_transient_failure(tmp_path: Path) -> None:
    """Orchestrator decision (a) with Q1: a transient failure at stop is retried within the bound.

    The fake fails exactly once after the stop flag exists; either the
    in-progress or the final attempt absorbs it, and a complete sample that
    starts after stop is still taken.
    """

    flag, once = tmp_path / "stopped", tmp_path / "failed-once"
    executable = _fake(tmp_path, "nvidia-smi", f"""\
        if [ -e {flag} ] && [ ! -e {once} ]; then touch {once}; exit 1; fi
        """ + _OK_BODY)
    sampler = _sampler_for(executable, command_timeout=0.5, cleanup_timeout=0.1, join_timeout=gs.finish_deadline(0.5, 0.1))
    sampler.start()
    _wait_for(sampler, lambda value: value["aggregate"]["samples_taken"] >= 1)
    flag.touch()
    stopped = time.monotonic()
    record = sampler.finish()
    assert once.exists()
    assert record["reason"] == "observed", record["failure"]
    assert record["gaps"]["count"] == 1
    assert record["samples"][-1]["monotonic_start"] >= stopped
    assert validate_sampler_wire(record) == record


@POSIX_GROUPS
def test_sampler_final_retry_must_fit_the_remaining_deadline(tmp_path: Path) -> None:
    """Orchestrator decision (a): no final retry starts unless a worst-case attempt fits."""

    flag = tmp_path / "stopped"
    executable = _fake(tmp_path, "nvidia-smi", f"if [ -e {flag} ]; then exit 1; fi\n" + _OK_BODY)
    # join_timeout is far below two worst-case attempts (2 * 3 * 0.6 s); a
    # generous gap budget isolates the remaining-deadline rule.
    sampler = _sampler_for(
        executable, command_timeout=0.5, cleanup_timeout=0.1, join_timeout=1.0, max_consecutive_gaps=5, max_total_gaps=5,
    )
    sampler.start()
    _wait_for(sampler, lambda value: value["aggregate"]["samples_taken"] >= 1)
    flag.touch()
    started = time.monotonic()
    record = sampler.finish()
    assert time.monotonic() - started < 1.0
    assert record["failure"]["code"] == "command-failed", record["failure"]
    assert "no retry fits the finish deadline" in record["failure"]["detail"]
    assert record["gaps"]["count"] <= 2 and record["quiescent"] is True
    assert validate_sampler_wire(record) == record


@POSIX_GROUPS
def test_worker_resolves_nvidia_smi_through_the_prepared_path_and_records_it(tmp_path: Path) -> None:
    """Orchestrator decision (b): the prepared PATH resolves nvidia-smi, not the setup script.

    The setup script prepends a directory holding a different fake; the
    sampler must use the descriptor's PATH and record that path and version.
    """

    bin_dir, shadow = tmp_path / "bin", tmp_path / "setup-bin"
    bin_dir.mkdir()
    shadow.mkdir()
    _fake(bin_dir, "nvidia-smi", _PID_FILE_BODY)
    _fake(shadow, "nvidia-smi", "echo shadowed; exit 1\n")
    setup = tmp_path / "setup.sh"
    setup.write_text(f'export PATH="{shadow}:$PATH"\n', encoding="utf-8")
    environment = _worker_env(tmp_path, bin_dir, L1B_CHILD_PID_FILE=str(tmp_path / "never"))
    outcome = SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable, SourceScript(str(setup)), dict(environment.env)),
        ("-c", "import time; time.sleep(0.2)"),
        cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
    )
    record = _evidence(tmp_path)
    assert outcome.returncode == 0 and record["reason"] == "observed", record["failure"]
    assert record["executable"] == str(bin_dir / "nvidia-smi")
    assert record["version"] == "NVIDIA-SMI fake"


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


# ------------------------------------- worker startup, slot fence, ambient


class _InterruptingThread(threading.Thread):
    """Sampler thread whose ``start`` is interrupted before or after it really starts."""

    mode = "before"
    launched: Path | None = None

    def start(self) -> None:
        if self.name != "tpen-gpu-sampler":
            return super().start()
        if self.mode == "before":
            raise _Cancel()
        super().start()
        # Interrupt only once a real command session exists.
        deadline = time.monotonic() + 5.0
        while not _pids(self.launched) and time.monotonic() < deadline:
            time.sleep(0.01)
        raise _Cancel()


def _hung_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> tuple[PreparedPythonEnvironment, Path]:
    monkeypatch.setattr(gs, "COMMAND_TIMEOUT", 10.0)
    monkeypatch.setattr(gs, "CLEANUP_TIMEOUT", 0.3)
    monkeypatch.setattr(gs, "JOIN_TIMEOUT", 1.0)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pid_file = tmp_path / "hung.pids"
    pid_files.append(pid_file)
    _hung(bin_dir, pid_file)
    return _worker_env(tmp_path, bin_dir), pid_file


def _run_interrupted(tmp_path: Path, environment: PreparedPythonEnvironment, pid_file: Path) -> list[bool]:
    """Run a long child whose startup is interrupted; return hung-command liveness at propagation."""

    started = time.monotonic()
    with pytest.raises(_Cancel):
        SubprocessWorker().run(
            environment, ("-c", "import time; time.sleep(30)"),
            cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
        )
    alive = [_alive(pid) for pid in _pids(pid_file)]
    assert time.monotonic() - started < 8.0
    return alive


@POSIX_GROUPS
def test_worker_interrupted_before_the_sampler_thread_starts_is_settled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """E1: an interruption before the thread starts leaves finalized, quiescent, blocking evidence."""

    environment, pid_file = _hung_worker(tmp_path, monkeypatch, pid_files)
    monkeypatch.setattr(_InterruptingThread, "mode", "before")
    monkeypatch.setattr(gs.threading, "Thread", _InterruptingThread)
    assert _run_interrupted(tmp_path, environment, pid_file) == []
    record = _evidence(tmp_path)
    assert record["failure"]["code"] == "sampler-cancelled" and record["reason"] == "telemetry-failed"
    assert record["quiescent"] is True and record["command_pids"] == []


@POSIX_GROUPS
def test_worker_interrupted_during_sampler_start_keeps_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """E1: an interruption inside ``Thread.start`` after the thread runs cannot orphan its command.

    The thread handle is published before ``start``, so finish still cancels,
    kills, and reaps the hung command before the interruption propagates.
    """

    environment, pid_file = _hung_worker(tmp_path, monkeypatch, pid_files)
    monkeypatch.setattr(_InterruptingThread, "mode", "during")
    monkeypatch.setattr(_InterruptingThread, "launched", pid_file)
    monkeypatch.setattr(gs.threading, "Thread", _InterruptingThread)
    assert _run_interrupted(tmp_path, environment, pid_file) == [False]
    record = _evidence(tmp_path)
    assert record["failure"]["code"] == "sampler-cancelled"
    assert record["quiescent"] is True and record["survivors"] == []
    assert [_alive(pid) for pid in record["command_pids"]] == [False]


@POSIX_GROUPS
def test_worker_interrupted_right_after_sampler_start_keeps_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """E1/H1: an interruption right after ``start`` returns, before the worker continues, is cleaned.

    The worker holds the sampler before calling ``start``, so its outer
    cleanup never depends on a startup helper returning a handle.
    """

    environment, pid_file = _hung_worker(tmp_path, monkeypatch, pid_files)
    original = gs.GpuSampler.start

    def start_then_interrupt(self) -> None:
        original(self)
        deadline = time.monotonic() + 5.0
        while not _pids(pid_file) and time.monotonic() < deadline:
            time.sleep(0.01)
        raise _Cancel()

    monkeypatch.setattr(gs.GpuSampler, "start", start_then_interrupt)
    assert _run_interrupted(tmp_path, environment, pid_file) == [False]
    record = _evidence(tmp_path)
    assert record["failure"]["code"] == "sampler-cancelled"
    assert record["quiescent"] is True and record["survivors"] == []
    assert [_alive(pid) for pid in record["command_pids"]] == [False]


@POSIX_GROUPS
def test_worker_fences_the_slot_while_a_held_launch_is_outstanding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """E2/H2: a command created before Popen publishes it keeps the slot fenced until it is gone.

    The first command process really exists but its ``Popen`` is held, so it
    is in no published handle.  The worker returns within its bound with
    non-quiescent evidence, a second science launch on the same slot is
    refused before spawning, and the slot is reusable only once the sampler
    thread has killed and reaped the held process.
    """

    environment, pid_file = _hung_worker(tmp_path, monkeypatch, pid_files)
    monkeypatch.setattr(gs, "CLEANUP_TIMEOUT", 0.2)
    monkeypatch.setattr(gs, "JOIN_TIMEOUT", 0.6)
    slot = "slot-e2-held-launch"
    release, held = threading.Event(), threading.Event()
    real_popen = subprocess.Popen

    class HeldPopen(real_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Hold only the first sampler command, never the science child.
            if not held.is_set() and str(args[0][0]).endswith("nvidia-smi"):
                # The process exists; its handle is not yet published.
                held.set()
                release.wait(10.0)

    monkeypatch.setattr(gs.subprocess, "Popen", HeldPopen)
    values = dict(environment.env, TPEN_SLOT_ID=slot)
    marker = tmp_path / "second-child-ran"
    try:
        started = time.monotonic()
        outcome = SubprocessWorker().run(
            PreparedPythonEnvironment(sys.executable, None, values), ("-c", "import time; time.sleep(0.2)"),
            cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
        )
        assert outcome.returncode == 0 and time.monotonic() - started < 5.0
        assert held.is_set() and [_alive(pid) for pid in _pids(pid_file)] == [True]
        record = _evidence(tmp_path)
        assert record["failure"]["code"] == "sampler-join-expired"
        assert "launch was still in flight" in record["failure"]["detail"]
        assert record["quiescent"] is False
        # The slot is fenced: no child is spawned on it.
        second = dict(values, **{gs.GPU_SAMPLER_OUTPUT_ENV: str(tmp_path / "second.json")})
        with pytest.raises(gs.SlotQuarantined, match="launch is in flight"):
            SubprocessWorker().run(
                PreparedPythonEnvironment(sys.executable, None, second), ("-c", f"open({str(marker)!r}, 'w')"),
                cwd=tmp_path, stdout_path=tmp_path / "out2.log", stderr_path=tmp_path / "err2.log",
            )
        assert not marker.exists() and not (tmp_path / "second.json").exists()
    finally:
        release.set()
    # The owning thread publishes, kills, and reaps the late process.
    deadline = time.monotonic() + 10.0
    while any(_alive(pid) for pid in _pids(pid_file)) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert [_alive(pid) for pid in _pids(pid_file)] == [False]
    deadline = time.monotonic() + 5.0
    while gs._FENCED.get(slot) and time.monotonic() < deadline:
        try:
            gs.ensure_slot_clean(slot)
        except gs.SlotQuarantined:
            time.sleep(0.02)
    third = dict(values, **{gs.GPU_SAMPLER_OUTPUT_ENV: str(tmp_path / "third.json")})
    outcome = SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable, None, third), ("-c", f"open({str(marker)!r}, 'w')"),
        cwd=tmp_path, stdout_path=tmp_path / "out3.log", stderr_path=tmp_path / "err3.log",
    )
    assert outcome.returncode == 0 and marker.exists() and slot not in gs._FENCED


# ------------------------------------- durable slot quarantine (delta-3 F1/F2)


def _slot_values(environment: PreparedPythonEnvironment, store: Path, slot: str) -> dict[str, str]:
    """Placed slot identity plus the controller-assigned quarantine directory."""

    values = dict(
        environment.env,
        TPEN_SLOT_ID=slot,
        TPEN_PLACEMENT_ALLOCATION_ID="allocation-a",
        TPEN_PLACEMENT_BACKEND_JOB_ID="job-a",
        TPEN_PLACEMENT_DECLARED_HOST=socket.gethostname(),
        TPEN_PLACEMENT_SLOT_ID=slot,
    )
    values[gs.SLOT_QUARANTINE_ENV] = gs.slot_quarantine_dir(store, values)
    return values


class _HeldLaunch:
    """Hold the first ``nvidia-smi`` Popen after its process exists, until released.

    The held command is in no published handle, so the worker's bounded
    finish cannot confirm its cleanup; the sampler thread kills and reaps it
    once the Popen is released.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.release, self.held = threading.Event(), threading.Event()
        release, held = self.release, self.held
        real_popen = subprocess.Popen

        class HeldPopen(real_popen):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if not held.is_set() and str(args[0][0]).endswith("nvidia-smi"):
                    held.set()
                    release.wait(10.0)

        monkeypatch.setattr(gs.subprocess, "Popen", HeldPopen)


def _held_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> tuple[PreparedPythonEnvironment, Path]:
    environment, pid_file = _hung_worker(tmp_path, monkeypatch, pid_files)
    monkeypatch.setattr(gs, "CLEANUP_TIMEOUT", 0.2)
    monkeypatch.setattr(gs, "JOIN_TIMEOUT", 0.6)
    return environment, pid_file


def _run_held_first_launch(tmp_path: Path, values: dict[str, str], pid_file: Path, held: _HeldLaunch) -> ProcessOutcome:
    """Run the first sampled launch; it returns in bound with its command held alive."""

    started = time.monotonic()
    outcome = SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable, None, values), ("-c", "import time; time.sleep(0.2)"),
        cwd=tmp_path, stdout_path=tmp_path / "first.out", stderr_path=tmp_path / "first.err",
    )
    assert outcome.returncode == 0 and time.monotonic() - started < 5.0
    assert held.held.is_set() and [_alive(pid) for pid in _pids(pid_file)] == [True]
    return outcome


def _await_release(pid_file: Path, slot: str, quarantine: str | None) -> None:
    """Wait until the owning sampler reaped the held command and the slot is released."""

    deadline = time.monotonic() + 10.0
    while any(_alive(pid) for pid in _pids(pid_file)) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert [_alive(pid) for pid in _pids(pid_file)] == [False]
    deadline = time.monotonic() + 5.0
    while True:
        try:
            gs.ensure_slot_clean(slot, quarantine)
            return
        except gs.SlotQuarantined:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def _markers(quarantine: str) -> list[Path]:
    directory = Path(quarantine)
    return sorted(directory.iterdir()) if directory.exists() else []


_FRESH_CONTROLLER = """\
import json, sys
from types import SimpleNamespace
from tpen_executor import gpu_sampler as gs
from tpen_executor.dispatch.contract import _admit_slot
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.workers import SubprocessWorker
spec = json.loads(sys.argv[1])
verdicts = {}
try:
    SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable, None, spec["sampled"]), ("-c", spec["child"]),
        cwd=spec["cwd"], stdout_path=spec["cwd"] + "/fresh.out", stderr_path=spec["cwd"] + "/fresh.err",
    )
    verdicts["sampled"] = "ran"
except gs.SlotQuarantined:
    verdicts["sampled"] = "SlotQuarantined"
try:
    _admit_slot(PreparedPythonEnvironment(sys.executable, None, spec["unsampled"]), SimpleNamespace(root=spec["store"]))
    verdicts["admission"] = "admitted"
except gs.SlotQuarantined:
    verdicts["admission"] = "SlotQuarantined"
print(json.dumps(verdicts))
"""


def _fresh_controller(tmp_path: Path, store: Path, values: dict[str, str], fast_bin: Path, ran: Path) -> dict[str, str]:
    """Launch on the same slot and store from a fresh Python process (a restarted controller).

    The fresh process imports this test's own ``tpen_executor`` source tree.
    Its sampled launch uses a responsive fake, and its unsampled science
    launch goes through dispatch admission without any quarantine key.
    """

    sampled = dict(values, PATH=f"{fast_bin}{os.pathsep}/usr/bin{os.pathsep}/bin")
    sampled[gs.GPU_SAMPLER_OUTPUT_ENV] = str(tmp_path / "fresh.json")
    unsampled = {key: value for key, value in values.items() if key not in (gs.GPU_SAMPLER_OUTPUT_ENV, gs.SLOT_QUARANTINE_ENV)}
    spec = {"sampled": sampled, "unsampled": unsampled, "store": str(store), "cwd": str(tmp_path),
            "child": f"open({str(ran)!r}, 'w')"}
    # The source tree under test comes first (a mutant arm's copy included).
    search = os.pathsep.join(filter(None, (str(Path(gs.__file__).resolve().parents[1]), os.environ.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-B", "-c", _FRESH_CONTROLLER, json.dumps(spec)], cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=search), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@POSIX_GROUPS
def test_durable_quarantine_refuses_a_fresh_controller_on_the_same_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """Delta-3 F1/J2: a restarted controller using the same attempt store refuses the slot.

    While the first launch's held command is alive, the durable marker
    written before spawn makes a fresh Python process refuse both a sampled
    launch (worker) and an unsampled science launch (dispatch admission) on
    the same slot, before any claim or spawn.  Once the owning sampler has
    reaped the command, the marker is removed and the same fresh controller
    launches (control).
    """

    environment, pid_file = _held_environment(tmp_path, monkeypatch, pid_files)
    fast_bin = tmp_path / "fast-bin"
    fast_bin.mkdir()
    _fake(fast_bin, "nvidia-smi", _OK_BODY)
    store, slot = tmp_path / "store", "slot-f1-fresh-controller"
    values = _slot_values(environment, store, slot)
    quarantine = values[gs.SLOT_QUARANTINE_ENV]
    ran = tmp_path / "fresh-child-ran"
    held = _HeldLaunch(monkeypatch)
    try:
        _run_held_first_launch(tmp_path, values, pid_file, held)
        markers = _markers(quarantine)
        assert len(markers) == 1
        identity = json.loads(markers[0].read_text(encoding="utf-8"))
        assert identity["schema"] == gs.QUARANTINE_SCHEMA and identity["TPEN_PLACEMENT_SLOT_ID"] == slot
        assert _fresh_controller(tmp_path, store, values, fast_bin, ran) == {
            "sampled": "SlotQuarantined", "admission": "SlotQuarantined",
        }
        assert not ran.exists() and not (tmp_path / "fresh.json").exists()
    finally:
        held.release.set()
    _await_release(pid_file, slot, quarantine)
    assert _markers(quarantine) == []
    assert _fresh_controller(tmp_path, store, values, fast_bin, ran) == {"sampled": "ran", "admission": "admitted"}
    assert ran.exists() and _markers(quarantine) == []


@POSIX_GROUPS
def test_quarantined_slot_refuses_unsampled_science_but_admits_an_inspection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """Delta-3 F1 / delta-4 D4-1: every science launch on a quarantined slot is refused, sampled or not.

    An unsampled launch carrying the same ``TPEN_SLOT_ID`` is refused before
    spawning, also when the owner environment or the descriptor carries the
    retired ``inspector`` role key; only a launch with the typed
    ``purpose=LaunchPurpose.INSPECTION`` argument runs.
    """

    environment, pid_file = _held_environment(tmp_path, monkeypatch, pid_files)
    store, slot = tmp_path / "store", "slot-f1-unsampled"
    values = _slot_values(environment, store, slot)
    unsampled = {key: value for key, value in values.items() if key not in (gs.GPU_SAMPLER_OUTPUT_ENV, gs.SLOT_QUARANTINE_ENV)}
    science, inspected = tmp_path / "science-ran", tmp_path / "inspector-ran"
    held = _HeldLaunch(monkeypatch)
    try:
        _run_held_first_launch(tmp_path, values, pid_file, held)
        with pytest.raises(gs.SlotQuarantined, match="launch is in flight"):
            SubprocessWorker().run(
                PreparedPythonEnvironment(sys.executable, None, unsampled), ("-c", f"open({str(science)!r}, 'w')"),
                cwd=tmp_path, stdout_path=tmp_path / "science.out", stderr_path=tmp_path / "science.err",
            )
        # Neither an ambient nor a descriptor role value is an exemption: the
        # retired environment key is never honored (delta-4 D4-1).
        monkeypatch.setenv(gs.LAUNCH_ROLE_ENV, "inspector")
        for overlay in (unsampled, {**unsampled, gs.LAUNCH_ROLE_ENV: "inspector"}):
            with pytest.raises(gs.SlotQuarantined):
                SubprocessWorker().run(
                    PreparedPythonEnvironment(sys.executable, None, overlay), ("-c", f"open({str(science)!r}, 'w')"),
                    cwd=tmp_path, stdout_path=tmp_path / "science.out", stderr_path=tmp_path / "science.err",
                )
        # A bare string is not the typed purpose and grants nothing.
        with pytest.raises(TypeError, match="LaunchPurpose"):
            SubprocessWorker().run(
                PreparedPythonEnvironment(sys.executable, None, unsampled), ("-c", f"open({str(science)!r}, 'w')"),
                cwd=tmp_path, stdout_path=tmp_path / "science.out", stderr_path=tmp_path / "science.err",
                purpose="inspection",
            )
        assert not science.exists()
        # The stripped environment handed to the inspection carries no role.
        inspection = _without_gpu_sampler(PreparedPythonEnvironment(sys.executable, None, {**values, gs.LAUNCH_ROLE_ENV: "inspector"}))
        assert not {gs.LAUNCH_ROLE_ENV, gs.GPU_SAMPLER_OUTPUT_ENV, gs.SLOT_QUARANTINE_ENV} & set(inspection.env)
        probe = f"import os; open({str(inspected)!r}, 'w').write(str({gs.LAUNCH_ROLE_ENV!r} in os.environ))"
        outcome = SubprocessWorker().run(
            inspection, ("-c", probe),
            cwd=tmp_path, stdout_path=tmp_path / "inspect.out", stderr_path=tmp_path / "inspect.err",
            purpose=LaunchPurpose.INSPECTION,
        )
        # The ambient role key is scrubbed: the inspection child never sees it.
        assert outcome.returncode == 0 and inspected.read_text(encoding="utf-8") == "False"
        monkeypatch.delenv(gs.LAUNCH_ROLE_ENV)
    finally:
        held.release.set()
    _await_release(pid_file, slot, values[gs.SLOT_QUARANTINE_ENV])
    outcome = SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable, None, unsampled), ("-c", f"open({str(science)!r}, 'w')"),
        cwd=tmp_path, stdout_path=tmp_path / "science.out", stderr_path=tmp_path / "science.err",
    )
    assert outcome.returncode == 0 and science.exists()


@POSIX_GROUPS
@pytest.mark.parametrize(
    "descriptor",
    [
        pytest.param({}, id="legacy-slot-only"),
        pytest.param({gs.LAUNCH_ROLE_ENV: "inspector"}, id="legacy-with-retired-role"),
        pytest.param({"TPEN_SLOT_NODE": "node-a"}, id="incompletely-placed"),
    ],
)
def test_legacy_science_fallback_is_refused_on_a_held_slot(tmp_path: Path, descriptor: dict[str, str]) -> None:
    """Delta-4 D4-1: the legacy/no-unique-slot SCIENCE fallback never becomes an inspection.

    A real unsettled :class:`SlotLease` holds the slot.  A legacy or
    incompletely placed descriptor carrying that ``TPEN_SLOT_ID`` goes
    through the real :func:`_with_child_placement` fallback (no unique
    physical slot), dispatch admission, and the real worker: it must be
    refused with ``SlotQuarantined`` and no child spawned.  Controls: the
    explicit inspection of the same environment runs while the lease is
    held, and the science fallback runs once the lease is released.
    """

    # Unique per case: in-process leases are keyed by slot id.
    slot = "slot-d4-1-" + ("-".join(sorted(descriptor)) or "bare")
    store = tmp_path / "store"
    legacy = PreparedPythonEnvironment(sys.executable, None, {"TPEN_SLOT_ID": slot, **descriptor})
    allocation = ConfiguredAllocation(AllocationId("allocation-d4"), BackendJobId("job-d4"), {})
    science, inspected = tmp_path / "science-ran", tmp_path / "inspection-ran"
    lease = gs.acquire_slot_lease(slot, None, {"test": "no child spawned; unsettled lease"})
    try:
        fallback = _with_child_placement(legacy, allocation, None, None, None, None)
        # Stripping worker-only keys never grants a role.
        assert fallback.env == {"TPEN_SLOT_ID": slot, **{k: v for k, v in descriptor.items() if k != gs.LAUNCH_ROLE_ENV}}
        # Admission has no placed identity to check; the worker is the backstop.
        _admit_slot(fallback, SimpleNamespace(root=str(store)))
        with pytest.raises(gs.SlotQuarantined, match="still owned by its worker"):
            SubprocessWorker().run(
                fallback, ("-c", f"open({str(science)!r}, 'w')"),
                cwd=tmp_path, stdout_path=tmp_path / "science.out", stderr_path=tmp_path / "science.err", timeout=10,
            )
        assert not science.exists()
        # Explicit-inspection control: the typed purpose is the only exemption.
        outcome = SubprocessWorker().run(
            fallback, ("-c", f"open({str(inspected)!r}, 'w')"),
            cwd=tmp_path, stdout_path=tmp_path / "inspect.out", stderr_path=tmp_path / "inspect.err", timeout=10,
            purpose=LaunchPurpose.INSPECTION,
        )
        assert outcome.returncode == 0 and inspected.exists() and not lease.settled
    finally:
        # This lease is a standalone test fixture with no real science
        # child; group_reaped=True is the trivially-correct value (review
        # B2 made settle_slot_lease's default conservative/unreleased).
        gs.settle_slot_lease(lease, None, group_reaped=True)
    # Later-clean-reuse control: once the lease is released the same fallback runs.
    assert slot not in gs._FENCED
    outcome = SubprocessWorker().run(
        fallback, ("-c", f"open({str(science)!r}, 'w')"),
        cwd=tmp_path, stdout_path=tmp_path / "science.out", stderr_path=tmp_path / "science.err", timeout=10,
    )
    assert outcome.returncode == 0 and science.exists()


def test_inspection_reader_is_the_only_inspection_launch(tmp_path: Path) -> None:
    """Delta-4 D4-1: ``TPENEvidence.read`` launches with the typed inspection purpose.

    The reader is the one trusted inspection call site; the worker default
    is a science launch.
    """

    calls: list[dict[str, object]] = []

    class Recording:
        def run(self, environment, args, **kwargs):
            calls.append(kwargs)
            raise OSError("recorded; no child")

    with pytest.raises(EvidenceUnavailable):
        TPENEvidence().read(
            Recording(), PreparedPythonEnvironment(sys.executable, None, {}), tmp_path,
            stdout_path=tmp_path / "out", stderr_path=tmp_path / "err",
        )
    assert [call["purpose"] for call in calls] == [LaunchPurpose.INSPECTION]
    assert SubprocessWorker.run.__kwdefaults__["purpose"] is LaunchPurpose.SCIENCE


@POSIX_GROUPS
def test_interrupted_evidence_publication_keeps_the_slot_quarantined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path]) -> None:
    """Delta-3 F2/J1: an interruption of the final evidence write cannot unfence the slot.

    The slot is quarantined before spawn, so when the publication of the
    non-quiescent record is interrupted the interruption propagates, the
    claimed record stays unfinalized, the slot is refused in process and by
    its durable marker, and it is released only once the sampler has
    reaped the held command.
    """

    environment, pid_file = _held_environment(tmp_path, monkeypatch, pid_files)
    store, slot = tmp_path / "store", "slot-f2-publication"
    values = _slot_values(environment, store, slot)
    quarantine = values[gs.SLOT_QUARANTINE_ENV]

    def interrupted_publication(_path, _record):
        raise _Cancel()

    monkeypatch.setattr(gs, "_write_evidence", interrupted_publication)
    second = tmp_path / "second-ran"
    held = _HeldLaunch(monkeypatch)
    try:
        started = time.monotonic()
        with pytest.raises(_Cancel):
            SubprocessWorker().run(
                PreparedPythonEnvironment(sys.executable, None, values), ("-c", "import time; time.sleep(0.2)"),
                cwd=tmp_path, stdout_path=tmp_path / "first.out", stderr_path=tmp_path / "first.err",
            )
        assert time.monotonic() - started < 5.0
        assert held.held.is_set() and [_alive(pid) for pid in _pids(pid_file)] == [True]
        assert _evidence(tmp_path)["failure"]["code"] == "sampler-unfinalized"
        with pytest.raises(gs.SlotQuarantined, match="launch is in flight"):
            gs.ensure_slot_clean(slot, quarantine)
        assert len(_markers(quarantine)) == 1
        other = dict(values, **{gs.GPU_SAMPLER_OUTPUT_ENV: str(tmp_path / "second.json")})
        with pytest.raises(gs.SlotQuarantined):
            SubprocessWorker().run(
                PreparedPythonEnvironment(sys.executable, None, other), ("-c", f"open({str(second)!r}, 'w')"),
                cwd=tmp_path, stdout_path=tmp_path / "second.out", stderr_path=tmp_path / "second.err",
            )
        assert not second.exists() and not (tmp_path / "second.json").exists()
    finally:
        held.release.set()
    _await_release(pid_file, slot, quarantine)
    assert _markers(quarantine) == [] and slot not in gs._FENCED


@POSIX_GROUPS
def test_interrupted_publication_with_confirmed_cleanup_releases_the_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Delta-3 F2 control: a quiescent sampler whose publication fails does not quarantine.

    The interruption still propagates and the claimed record stays
    unfinalized (science-blocking), but every command is confirmed gone, so
    the lease and its marker are released and the next launch runs.
    """

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", _OK_BODY)
    store, slot = tmp_path / "store", "slot-f2-quiescent"
    values = _slot_values(_worker_env(tmp_path, bin_dir), store, slot)
    quarantine = values[gs.SLOT_QUARANTINE_ENV]
    real_write = gs._write_evidence

    def interrupted_publication(_path, _record):
        raise _Cancel()

    monkeypatch.setattr(gs, "_write_evidence", interrupted_publication)
    with pytest.raises(_Cancel):
        SubprocessWorker().run(
            PreparedPythonEnvironment(sys.executable, None, values), ("-c", "import time; time.sleep(0.2)"),
            cwd=tmp_path, stdout_path=tmp_path / "first.out", stderr_path=tmp_path / "first.err",
        )
    assert _evidence(tmp_path)["failure"]["code"] == "sampler-unfinalized"
    assert slot not in gs._FENCED and _markers(quarantine) == []
    monkeypatch.setattr(gs, "_write_evidence", real_write)
    marker = tmp_path / "next-ran"
    following = dict(values, **{gs.GPU_SAMPLER_OUTPUT_ENV: str(tmp_path / "next.json")})
    outcome = SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable, None, following), ("-c", f"open({str(marker)!r}, 'w')"),
        cwd=tmp_path, stdout_path=tmp_path / "next.out", stderr_path=tmp_path / "next.err",
    )
    assert outcome.returncode == 0 and marker.exists() and slot not in gs._FENCED


@POSIX_GROUPS
def test_worker_ignores_an_ambient_evidence_path(tmp_path: Path) -> None:
    """E3: only the prepared descriptor opts into sampling; an inherited key never does.

    Negative controls: an H-v2-shaped descriptor with ``env={}`` and the
    stripped environment from ``_without_gpu_sampler``, both run while the
    owner environment carries the key and a callable fake ``nvidia-smi``.
    """

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "invoked"
    _fake(bin_dir, "nvidia-smi", f"touch {marker}\n" + _OK_BODY)
    ambient = tmp_path / "ambient.json"
    probe = f"import json, os; print(json.dumps({gs.GPU_SAMPLER_OUTPUT_ENV!r} in os.environ))"
    saved = {key: os.environ.get(key) for key in (gs.GPU_SAMPLER_OUTPUT_ENV, "PATH")}
    os.environ[gs.GPU_SAMPLER_OUTPUT_ENV] = str(ambient)
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{saved['PATH'] or os.defpath}"
    try:
        for name, environment in {
            "env-empty": PreparedPythonEnvironment(sys.executable, None, {}),
            "inspection": _without_gpu_sampler(_worker_env(tmp_path, bin_dir)),
        }.items():
            assert gs.GPU_SAMPLER_OUTPUT_ENV not in environment.env, name
            SubprocessWorker().run(
                environment, ("-c", probe),
                cwd=tmp_path, stdout_path=tmp_path / f"{name}.out", stderr_path=tmp_path / f"{name}.err",
            )
            assert json.loads((tmp_path / f"{name}.out").read_text(encoding="utf-8")) is False, name
            assert not ambient.exists() and not marker.exists(), name
            assert not (tmp_path / "gpu-sampler.json").exists(), name
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@POSIX_GROUPS
def test_worker_timeout_with_fast_commands_keeps_usable_telemetry(tmp_path: Path) -> None:
    """Q4: a worker timeout of the child does not by itself fail usable, quiescent telemetry."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", _PID_FILE_BODY)
    pid_file = tmp_path / "child.pid"
    child = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    outcome = SubprocessWorker().run(
        _worker_env(tmp_path, bin_dir, L1B_CHILD_PID_FILE=str(pid_file)), ("-c", child),
        cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log", timeout=0.8,
    )
    record = _evidence(tmp_path)
    assert outcome.timed_out is True
    assert record["reason"] == "observed" and record["failure"] is None, record["failure"]
    assert record["quiescent"] is True
    assert derive_device_decision(record, outcome.pid, (GPU_A,)).code == "matched"


@POSIX_GROUPS
@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="requires POSIX interval timers")
def test_worker_cancellation_with_fast_commands_keeps_usable_telemetry(tmp_path: Path) -> None:
    """Q4: an owner cancellation of the child keeps usable, quiescent telemetry observed."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", _PID_FILE_BODY)
    pid_file = tmp_path / "child.pid"
    child = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"

    def cancel(_signum, _frame):
        raise _Cancel()

    previous = signal.signal(signal.SIGALRM, cancel)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.8)
        with pytest.raises(_Cancel):
            SubprocessWorker().run(
                _worker_env(tmp_path, bin_dir, L1B_CHILD_PID_FILE=str(pid_file)), ("-c", child),
                cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    record = _evidence(tmp_path)
    assert record["reason"] == "observed" and record["failure"] is None, record["failure"]
    assert record["quiescent"] is True
    assert derive_device_decision(record, record["target_pid"], (GPU_A,)).code == "matched"


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
    gaps: dict[str, object] | None = None,
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
        "executable": "/opt/fake/bin/nvidia-smi", "gaps": empty_gaps(1, 3) if gaps is None else gaps,
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


def _add_gaps(raw: dict, *events: tuple[str, float, float, int]) -> None:
    """Append gap events ``(code, start, end, consecutive)`` with consistent counters."""

    gaps = raw["gpu_sampler"]["gaps"]
    for code, start, end, run in events:
        gaps["events"].append({"code": code, "monotonic_start": start, "monotonic_end": end, "consecutive": run})
    gaps["count"] = len(gaps["events"])
    gaps["max_consecutive"] = max(event["consecutive"] for event in gaps["events"])
    gaps["total_seconds"] = sum(event["monotonic_end"] - event["monotonic_start"] for event in gaps["events"])


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
    # r4: resolved executable and the recorded gap budget (manager ruling Q1).
    "null-executable": lambda raw: raw["gpu_sampler"].__setitem__("executable", None),
    "gap-count-disagrees": lambda raw: raw["gpu_sampler"]["gaps"].__setitem__("count", 1),
    "gap-unforgivable-code": lambda raw: _add_gaps(raw, ("malformed-output", 1.11, 1.15, 1)),
    "gap-outside-outer": lambda raw: _add_gaps(raw, ("command-failed", 2.0, 2.1, 1)),
    "gap-budget-exceeded-but-observed": lambda raw: _add_gaps(raw, ("command-failed", 1.11, 1.12, 1), ("command-failed", 1.13, 1.14, 2)),
    "gap-total-disagrees": lambda raw: (_add_gaps(raw, ("command-timeout", 1.11, 1.12, 1)), raw["gpu_sampler"]["gaps"].__setitem__("total_seconds", 5.0)),
    "gap-run-numbering": lambda raw: _add_gaps(raw, ("command-failed", 1.11, 1.12, 2)),
    "gap-budget-above-cap": lambda raw: raw["gpu_sampler"]["gaps"].__setitem__("max_total_allowed", 17),
    # r5 (delta-3 F3): a coherent attempt timeline.  The sample is [1.0, 1.1].
    "gap-overlaps-success": lambda raw: _add_gaps(raw, ("command-timeout", 1.0, 1.1, 1)),
    "gap-reset-without-success": lambda raw: _add_gaps(raw, ("command-failed", 1.11, 1.12, 1), ("command-failed", 1.13, 1.14, 1)),
    # r6 (delta-4 D4-2): a dropped success cannot lie after the newest retained sample.
    "gap-reset-after-the-newest-success": lambda raw: (
        raw["gpu_sampler"]["aggregate"].__setitem__("samples_taken", raw["gpu_sampler"]["count"] + 1),
        _add_gaps(raw, ("command-failed", 1.11, 1.12, 1), ("command-failed", 1.13, 1.14, 1)),
    ),
}


@pytest.mark.parametrize("name", sorted(CONTRADICTIONS))
def test_decoder_rejects_contradictory_sampler_evidence(tmp_path: Path, name: str) -> None:
    """Every tampered qualified receipt is rejected by the shared derivation."""

    raw = _receipt(tmp_path, _sampler([[(1234, GPU_A)]])).to_dict()
    assert SciencePlacementReceiptV3.from_mapping(copy.deepcopy(raw)).verdict["reason_class"] == "qualified"
    CONTRADICTIONS[name](raw)
    with pytest.raises((TypeError, ValueError)):
        SciencePlacementReceiptV3.from_mapping(raw)


# (accepted, retained samples, dropped samples, failure, budget, gap events)
# for one record whose samples are [1.0, 1.1] and, with two, [1.2, 1.3].
GAP_TIMELINES = {
    "reset-after-a-retained-success": (True, 2, 0, None, (1, 3), [(1.11, 1.15, 1), (1.31, 1.32, 1)]),
    "run-across-a-retained-success": (False, 2, 0, None, (2, 3), [(1.11, 1.15, 1), (1.31, 1.32, 2)]),
    # r6 (delta-4 D4-2): the evicted success lies between the two retained
    # samples; in r5 the gaps followed the only retained sample, where no
    # dropped success can lie.
    "reset-after-an-evicted-success": (True, 2, 1, None, (1, 3), [(1.11, 1.12, 1), (1.13, 1.14, 1)]),
    "resets-beyond-the-evicted-samples": (False, 2, 1, None, (1, 3), [(1.11, 1.115, 1), (1.12, 1.125, 1), (1.13, 1.135, 1)]),
    "excess-as-the-last-event": (True, 1, 0, "gap-budget-exceeded", (1, 3), [(1.11, 1.115, 1), (1.12, 1.125, 2)]),
    "events-after-the-first-excess": (
        False, 1, 1, "gap-budget-exceeded", (1, 3), [(1.11, 1.115, 1), (1.12, 1.125, 2), (1.13, 1.135, 1)],
    ),
    # r6 (delta-4 D4-2): eviction window and the excess as the last attempt.
    "reset-after-the-newest-success": (False, 2, 1, None, (1, 3), [(1.31, 1.32, 1), (1.33, 1.34, 1)]),
    "success-after-the-excess": (False, 2, 0, "gap-budget-exceeded", (1, 3), [(1.11, 1.115, 1), (1.12, 1.125, 2)]),
}


@pytest.mark.parametrize("name", sorted(GAP_TIMELINES))
def test_decoder_checks_a_coherent_gap_timeline(name: str) -> None:
    """Delta-3 F3: gaps, resets, and the first excess agree with the successful samples.

    With complete history every reset needs a retained success in between;
    with dropped history only what is provable is checked (one dropped
    sample per unproven reset, inside its eviction window: delta-4 D4-2),
    and evicted successes legitimately justify resets (controls).  No event
    or success may follow the first budget excess.
    """

    accepted, retained, dropped, failure, budget, events = GAP_TIMELINES[name]
    gaps = empty_gaps(*budget)
    _add_gaps({"gpu_sampler": {"gaps": gaps}}, *[("command-failed", *event) for event in events])
    record = _sampler([[(1234, GPU_A)]] * retained, dropped=[[(1234, GPU_A)]] * dropped, failure=failure, gaps=gaps)
    if accepted:
        assert validate_sampler_wire(record)["gaps"] == gaps
    else:
        with pytest.raises(ValueError, match="gap"):
            validate_sampler_wire(record)


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
                "tpen.run-evidence/v2", str(run_dir), "completed", "completed",
                {"checkpoint_dir": str(tmp_path / "checkpoint"), "schema_version": 1, "next_iteration": 1, "completed_updates": 1}, None,
                {"schema": "tpen-config-identity-v1", "sha256": "a" * 64},
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
    "gaps-within-budget", "gap-budget-exceeded", "mig-unsupported",
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


# -------------------------------------------- B2: science quiescence gate


@LINUX_PROC_GROUPS
def test_group_quiescence_confirmed_releases_quarantine_before_next_admission(
    tmp_path: Path, pid_files: list[Path],
) -> None:
    """B2 clean control: with science-group quiescence OBSERVED (the real,
    unmocked ``_group_quiescent`` probe), the sampler's own clean command
    cleanup releases the marker, and a fresh same-allocation controller
    admits the slot again.
    """

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", _OK_BODY)
    store = tmp_path / "store"
    slot = "slot-b2-control"
    values = _slot_values(_worker_env(tmp_path, bin_dir), store, slot)
    quarantine = values[gs.SLOT_QUARANTINE_ENV]

    outcome = SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable, None, values), ("-c", "pass"),
        cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
    )
    assert outcome.returncode == 0 and outcome.group_reaped is True
    assert _markers(quarantine) == []
    gs.ensure_slot_clean(slot, quarantine)  # does not raise: a fresh controller is admitted


@POSIX_GROUPS
def test_group_quiescence_unconfirmed_retains_quarantine_and_refuses_fresh_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid_files: list[Path],
) -> None:
    """B2 killer: workers.py:433 (pre-fix) settled the sampler/lease BEFORE
    the group probe at :439, so a clean sampler could release the marker
    while the science group's quiescence was never checked at all.  With
    an observed-NEGATIVE science-group fact, the marker must be retained
    even though the sampler's own command cleanup is clean, and a fresh
    same-allocation/store controller must refuse the slot.

    ``_group_quiescent`` is monkeypatched to return the negative fact
    directly: a same-group test descendant cannot be made to survive
    SIGKILL (the real worker always kills then probes its own group, see
    the P1-1 residual fix), so this represents what the probe would
    observe for a genuinely cross-session-detached descendant -- the one
    case ``_group_quiescent`` cannot reach by design (it only ever
    discovers or signals a process inside the group it itself started).
    """

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake(bin_dir, "nvidia-smi", _OK_BODY)
    store = tmp_path / "store"
    slot = "slot-b2-retained"
    values = _slot_values(_worker_env(tmp_path, bin_dir), store, slot)
    quarantine = values[gs.SLOT_QUARANTINE_ENV]

    monkeypatch.setattr(workers_module, "_group_quiescent", lambda pid, **kwargs: False)
    outcome = SubprocessWorker().run(
        PreparedPythonEnvironment(sys.executable, None, values), ("-c", "pass"),
        cwd=tmp_path, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
    )
    assert outcome.returncode == 0 and outcome.group_reaped is False
    assert len(_markers(quarantine)) == 1
    with pytest.raises(gs.SlotQuarantined, match="science process group"):
        gs.ensure_slot_clean(slot, quarantine)
