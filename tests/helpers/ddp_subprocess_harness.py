"""Reusable safe CPU/Gloo subprocess harness (DF1, Stage T1).

The pytest process is never a distributed worker: this module launches a
fresh child process group per invocation via genuine OS subprocesses (never
``torch.multiprocessing.spawn``), captures per-rank receipts, and terminates
the entire child group if the outer watchdog expires. Three nested bounds
govern every invocation: process-group timeout < harness watchdog timeout
< scheduler wall time (the third bound is external to this module and is
the caller's Slurm wall-time budget).

A communicator that has timed out or mismatched is poisoned and is never
reused: every call to :func:`run_gloo_subprocess_group` derives a fresh,
invocation-unique rendezvous file and launches brand-new subprocesses.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tests.helpers.ddp_fault_injection import FaultKind, FaultPlan, write_fault_plan

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAP_WAIT_SECONDS = 5.0


@dataclass(frozen=True)
class HarnessBounds:
    """The first two of the harness's three nested bounds.

    Parameters
    ----------
    process_group_timeout : float
        Seconds passed as ``timeout=`` to ``dist.init_process_group`` in
        every child; bounds each individual collective call.
    watchdog_timeout : float
        Seconds the parent harness waits for each child before force-killing
        the whole group. Must exceed ``process_group_timeout`` so the inner
        bound has a chance to resolve a collective-level fault before the
        outer bound resorts to killing the group.
    """

    process_group_timeout: float
    watchdog_timeout: float

    def __post_init__(self) -> None:
        if not (self.watchdog_timeout > self.process_group_timeout):
            raise ValueError(
                f"watchdog_timeout ({self.watchdog_timeout}) must exceed "
                f"process_group_timeout ({self.process_group_timeout})"
            )


@dataclass(frozen=True)
class RankReceipt:
    """One rank's self-reported evidence, written only on that rank's own success path."""

    rank: int
    world_size: int
    hostname: str
    pid: int
    phase_sequence: list[str]
    collective_result: float | None
    fault_kind: str


@dataclass(frozen=True)
class HarnessResult:
    """Deterministic result of one :func:`run_gloo_subprocess_group` invocation.

    Parameters
    ----------
    receipts : tuple of RankReceipt or None
        One slot per rank, in rank order. ``None`` means that rank never
        wrote its receipt (crashed, killed, or raised before reaching it) --
        never silently omitted from the tuple.
    watchdog_fired : bool
        Whether the outer watchdog had to force-kill any child (as opposed
        to every child exiting on its own, including via its own
        process-group timeout, within the watchdog window).
    all_reaped : bool
        Whether every child's process GROUP (not just its direct PID) is
        confirmed gone after this call returns.
    culprit_rank : int or None
        The fault plan's target rank, or ``None`` if no fault was injected.
        Read directly from the plan rather than inferred from exit codes,
        since some faults (e.g. a skipped collective) leave their target
        rank exiting cleanly while an innocent peer pays the cost.
    publication_observed : bool
        Whether the group-wide COMPLETE marker was written.
    rendezvous_path : str
        The fresh, invocation-unique file used for this call's rendezvous.
        Never reused across calls, and never a fixed name.
    """

    receipts: tuple[RankReceipt | None, ...]
    watchdog_fired: bool
    all_reaped: bool
    culprit_rank: int | None
    publication_observed: bool
    rendezvous_path: str


def run_gloo_subprocess_group(
    world_size: int,
    fault_plan: FaultPlan | None,
    bounds: HarnessBounds,
    tmp_path: Path,
    *,
    decoy_grandchild_rank: int | None = None,
) -> HarnessResult:
    """Launch ``world_size`` fresh CPU/Gloo worker subprocesses and collect results.

    Every call is independent: a fresh rendezvous file, fresh subprocesses,
    fresh receipt/state paths. Nothing from a prior call is reused.
    """

    rendezvous_fd, rendezvous_path_str = tempfile.mkstemp(dir=tmp_path, prefix="rdzv-")
    os.close(rendezvous_fd)
    os.unlink(rendezvous_path_str)  # torch's FileStore requires a nonexistent path

    complete_marker_path = tmp_path / "COMPLETE"

    fault_plan_path: Path | None = None
    if fault_plan is not None and fault_plan.kind != FaultKind.NONE:
        fault_plan_path = tmp_path / "fault_plan.json"
        write_fault_plan(fault_plan, fault_plan_path)

    procs: list[subprocess.Popen] = []
    receipt_paths: list[Path] = []
    for rank in range(world_size):
        receipt_path = tmp_path / f"receipt_{rank}.json"
        state_path = tmp_path / f"state_{rank}.json"
        receipt_paths.append(receipt_path)
        argv = [
            sys.executable,
            "-m",
            "tests.helpers.ddp_worker_entrypoint",
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--rendezvous-file",
            rendezvous_path_str,
            "--receipt-path",
            str(receipt_path),
            "--state-path",
            str(state_path),
            "--complete-marker-path",
            str(complete_marker_path),
            "--pg-timeout",
            str(bounds.process_group_timeout),
        ]
        if fault_plan_path is not None:
            argv += ["--fault-plan-path", str(fault_plan_path)]
        if decoy_grandchild_rank == rank:
            grandchild_pid_path = tmp_path / f"grandchild_{rank}.pid"
            argv += ["--spawn-decoy-grandchild", "--grandchild-pid-path", str(grandchild_pid_path)]
        procs.append(subprocess.Popen(argv, cwd=str(_REPO_ROOT), start_new_session=True))

    # A shared deadline, not a per-process timeout: waiting on N processes
    # sequentially with independent per-process timeouts would let the
    # total wait stack up to N * watchdog_timeout, silently blowing the
    # outer bound the watchdog exists to enforce.
    exit_codes: list[int | None] = [None] * world_size
    deadline = time.monotonic() + bounds.watchdog_timeout
    while time.monotonic() < deadline and any(code is None for code in exit_codes):
        for i, proc in enumerate(procs):
            if exit_codes[i] is None:
                exit_codes[i] = proc.poll()
        if any(code is None for code in exit_codes):
            time.sleep(0.05)
    watchdog_fired = any(code is None for code in exit_codes)

    # Unconditional, not gated on watchdog_fired: a rank whose OWN
    # process-group timeout resolved it (so it exited on its own, within the
    # watchdog window) can still leave a decoy/worker-spawned grandchild
    # behind in its process group, since that grandchild does not die just
    # because its parent exited. "Verifies no workers survived" must hold on
    # every path, not only the one where the outer watchdog had to act.
    # PermissionError is treated the same as ProcessLookupError throughout
    # this function: these PIDs are our own children's process-group
    # leaders, so we always have permission to signal them WHILE they are
    # ours. A PermissionError here means the OS has already recycled that
    # exact PID for a process owned by someone else -- i.e. our own group is
    # already gone -- not a genuine access restriction on our own child.
    _GONE = (ProcessLookupError, PermissionError)

    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except _GONE:
            pass

    # The all_reaped verification happens IMMEDIATELY after the kill sweep,
    # before any further reaping of direct children below -- every extra
    # moment (and every extra os.waitpid this process performs elsewhere)
    # is a moment in which the OS could recycle one of these exact PIDs for
    # an unrelated process, which would make a bare killpg(pid, 0) probe a
    # false positive for "still alive". A short grace period lets SIGKILL
    # actually land before the first probe; this narrows the race, it does
    # not eliminate it in principle.
    time.sleep(0.2)
    all_reaped = True
    for proc in procs:
        try:
            os.killpg(proc.pid, 0)
            all_reaped = False
        except _GONE:
            pass

    reap_deadline = time.monotonic() + _REAP_WAIT_SECONDS
    while time.monotonic() < reap_deadline and any(code is None for code in exit_codes):
        for i, proc in enumerate(procs):
            if exit_codes[i] is None:
                exit_codes[i] = proc.poll()
        if any(code is None for code in exit_codes):
            time.sleep(0.05)

    receipts: list[RankReceipt | None] = []
    for receipt_path in receipt_paths:
        if receipt_path.exists():
            receipts.append(RankReceipt(**json.loads(receipt_path.read_text())))
        else:
            receipts.append(None)

    culprit_rank = (
        fault_plan.target_rank if fault_plan is not None and fault_plan.kind != FaultKind.NONE else None
    )

    return HarnessResult(
        receipts=tuple(receipts),
        watchdog_fired=watchdog_fired,
        all_reaped=all_reaped,
        culprit_rank=culprit_rank,
        publication_observed=complete_marker_path.exists(),
        rendezvous_path=rendezvous_path_str,
    )


__all__ = [
    "HarnessBounds",
    "HarnessResult",
    "RankReceipt",
    "run_gloo_subprocess_group",
]
