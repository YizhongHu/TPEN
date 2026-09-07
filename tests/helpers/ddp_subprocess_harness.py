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

SCOPE NOTE, load-bearing: ``HarnessResult.publication_observed`` refers to a
``COMPLETE`` marker file written by rank 0 in ``tmp_path``, behind a
``dist.barrier()``, inside the synthetic worker run by
:mod:`tests.helpers.ddp_worker_entrypoint`. It is not, and does not
exercise, TPEN's real checkpoint publication path under
``tpen/checkpoint/`` (owned by a different, concurrent lane) -- this module
touches no file under ``tpen/``. The fault names shared with that
machinery (``CRASH_AFTER_PUBLISH``, ``CRASH_DURING_CHECKPOINT``, and the
``*_STATE_WRITE``/``*_PUBLICATION`` phases in
:mod:`tests.helpers.ddp_fault_injection`) describe this synthetic sequence
only; see that module's own SCOPE NOTE.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tests.helpers.ddp_fault_injection import FaultKind, FaultPlan, validate_fault_plan, write_fault_plan

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
class MalformedReceipt:
    """A receipt file existed but could not be parsed into a :class:`RankReceipt`.

    Distinguished from ``None`` (never written) because the two have
    different diagnostic meaning: ``None`` means the rank died before
    reaching the write at all, while this means a write was in progress (or
    the file is otherwise corrupt) when collection ran.
    """

    error: str


@dataclass(frozen=True)
class HarnessResult:
    """Deterministic result of one :func:`run_gloo_subprocess_group` invocation.

    Parameters
    ----------
    receipts : tuple of RankReceipt, MalformedReceipt, or None
        One slot per rank, in rank order. ``None`` means that rank never
        wrote its receipt (crashed, killed, or raised before reaching it).
        ``MalformedReceipt`` means a receipt file existed but could not be
        parsed (e.g. truncated by a kill mid-write) -- collection never
        raises past a bad receipt, never silently omitted from the tuple.
    watchdog_fired : bool
        Whether the outer watchdog had to force-kill any child (as opposed
        to every child exiting on its own, including via its own
        process-group timeout, within the watchdog window).
    all_reaped : bool
        Whether every child's process GROUP (not just its direct PID) is
        confirmed gone after this call returns.
    culprit_rank : int or None
        The rank whose own child process self-reported applying the fault,
        or ``None`` if no rank ever did. DERIVED from each rank's own
        captured log (see ``invocation_dir``), never read directly from the
        input fault plan: a plan whose ``target_rank`` never actually
        matches a running rank (or whose match never fires, e.g. a phase
        that is never reached) now correctly yields ``None`` instead of
        echoing a target that was never applied. A rank's self-report is
        written the instant it applies its fault, before the fault's
        effect (raise, exit, or stall) can take hold, so it survives even
        when that rank never reaches its own receipt write -- including the
        case of a skipped collective, where the target rank exits cleanly
        while an innocent peer pays the cost, and only the target's own log
        carries the self-report.
    publication_observed : bool
        Whether the group-wide COMPLETE marker was written.
    rendezvous_path : str
        The fresh, invocation-unique file used for this call's rendezvous.
        Never reused across calls, and never a fixed name.
    exit_codes : tuple of int or None
        One slot per rank, in rank order. ``None`` means the rank was still
        outstanding when the watchdog window closed; the subsequent reap
        loop then back-fills ``-9`` for a killed rank once its exit is
        observed, so ``None`` survives only if the reap window also expires.
    invocation_dir : str
        The fresh, invocation-unique directory holding every artifact this
        call produced: the rendezvous file, ``fault_plan.json``, each
        rank's ``receipt_{rank}.json``, ``state_{rank}.json``, and
        ``rank_{rank}.log``, and the ``COMPLETE`` marker. Never reused
        across calls unless the caller explicitly passes the same
        ``invocation_dir`` to two calls itself.
    """

    receipts: tuple[RankReceipt | MalformedReceipt | None, ...]
    watchdog_fired: bool
    all_reaped: bool
    culprit_rank: int | None
    publication_observed: bool
    rendezvous_path: str
    exit_codes: tuple[int | None, ...]
    invocation_dir: str


def run_gloo_subprocess_group(
    world_size: int,
    fault_plan: FaultPlan | None,
    bounds: HarnessBounds,
    tmp_path: Path,
    *,
    worker_module: str = "tests.helpers.ddp_worker_entrypoint",
    worker_extra_args: Sequence[str] = (),
    decoy_grandchild_rank: int | None = None,
    invocation_dir: Path | None = None,
) -> HarnessResult:
    """Launch ``world_size`` fresh CPU/Gloo worker subprocesses and collect results.

    Every call is independent: unless ``invocation_dir`` is supplied, a
    fresh per-invocation subdirectory (``tempfile.mkdtemp``, so its name is
    guaranteed unique) holds the rendezvous file, the fault plan, every
    rank's receipt/state/log, and the completion marker. Nothing from a
    prior call sharing the same ``tmp_path`` is reachable by this one.
    ``invocation_dir`` is an escape hatch for a caller that needs a
    deterministic, pre-known path -- e.g. to pre-seed a malformed receipt
    before the call -- at which point isolation across calls sharing that
    directory is the caller's own choice, not an accident.
    """

    validate_fault_plan(fault_plan)

    if invocation_dir is None:
        invocation_dir = Path(tempfile.mkdtemp(dir=tmp_path, prefix="invocation-"))

    rendezvous_fd, rendezvous_path_str = tempfile.mkstemp(dir=invocation_dir, prefix="rdzv-")
    os.close(rendezvous_fd)
    os.unlink(rendezvous_path_str)  # torch's FileStore requires a nonexistent path

    complete_marker_path = invocation_dir / "COMPLETE"

    fault_plan_path: Path | None = None
    if fault_plan is not None and fault_plan.kind != FaultKind.NONE:
        fault_plan_path = invocation_dir / "fault_plan.json"
        write_fault_plan(fault_plan, fault_plan_path)

    procs: list[subprocess.Popen] = []
    receipt_paths: list[Path] = []
    log_paths: list[Path] = []
    for rank in range(world_size):
        receipt_path = invocation_dir / f"receipt_{rank}.json"
        state_path = invocation_dir / f"state_{rank}.json"
        receipt_paths.append(receipt_path)
        log_path = invocation_dir / f"rank_{rank}.log"
        log_paths.append(log_path)
        argv = [
            sys.executable,
            "-m",
            worker_module,
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
        argv.extend(worker_extra_args)
        if fault_plan_path is not None:
            argv += ["--fault-plan-path", str(fault_plan_path)]
        if decoy_grandchild_rank == rank:
            grandchild_pid_path = invocation_dir / f"grandchild_{rank}.pid"
            argv += ["--spawn-decoy-grandchild", "--grandchild-pid-path", str(grandchild_pid_path)]
        # Captured per rank so an expected-failure test still has an
        # attributable diagnostic on disk: Popen would otherwise inherit
        # pytest's own stdout/stderr, whose fd-level capture discards child
        # tracebacks for a PASSING negative test. Closing the parent's
        # handle right after Popen is safe -- the child already holds its
        # own dup'd descriptor from the fork/exec inside the constructor.
        with open(log_path, "wb") as log_file:
            procs.append(
                subprocess.Popen(
                    argv,
                    cwd=str(_REPO_ROOT),
                    start_new_session=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            )

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

    # Reap the DIRECT children before probing liveness below. A killed but
    # not-yet-waited-on child is a ZOMBIE: on Linux, killpg(pid, 0) against
    # a zombie's process group succeeds (the process-table entry still
    # exists until its parent reaps it), which reads as "still alive" even
    # though SIGKILL has already landed. Measured on Cannon: a sole,
    # already-killed rank (exit_codes=(-9,)) still failed the liveness
    # check until this reordering. Reaping our own children first removes
    # that false positive for them; it does nothing for a grandchild, which
    # is the actual thing the liveness check below still needs to catch.
    reap_deadline = time.monotonic() + _REAP_WAIT_SECONDS
    while time.monotonic() < reap_deadline and any(code is None for code in exit_codes):
        for i, proc in enumerate(procs):
            if exit_codes[i] is None:
                exit_codes[i] = proc.poll()
        if any(code is None for code in exit_codes):
            time.sleep(0.05)

    # A brief grace period before the liveness check below: an orphaned
    # grandchild (reparented once its own parent, one of our direct
    # children, has been reaped above) is itself a zombie until whatever
    # subreaper it lands on reaps it, which is not instantaneous. This is
    # the same zombie hazard as above, for a process we cannot reap
    # ourselves. PID reuse is the residual risk this whole ordering trades
    # for: once a child is reaped, the OS is free to recycle its PID (and,
    # more rarely, a new unrelated session could even reuse its exact
    # PGID), which would make a bare killpg(pid, 0) probe a false positive
    # for "still alive". Neither race is eliminated in principle, only
    # narrowed.
    time.sleep(0.2)
    all_reaped = True
    for proc in procs:
        try:
            os.killpg(proc.pid, 0)
            all_reaped = False
        except _GONE:
            pass

    receipts: list[RankReceipt | MalformedReceipt | None] = []
    for receipt_path in receipt_paths:
        if not receipt_path.exists():
            receipts.append(None)
            continue
        try:
            receipts.append(RankReceipt(**json.loads(receipt_path.read_text())))
        except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
            # Narrow catch only: a rank killed mid-write (now impossible for
            # this worker's own atomic write, but still possible for any
            # future writer of this file, or a filesystem without rename
            # atomicity) leaves truncated/invalid JSON on disk. Collection
            # must report that as data, not raise -- but must never widen to
            # bare Exception, which would hide defects this harness exists
            # to surface.
            receipts.append(MalformedReceipt(error=f"{type(exc).__name__}: {exc}"))

    # Derived from each rank's own captured log, never from the input plan:
    # only the rank whose code path actually matched target_rank/phase ever
    # writes the controlled self-report line (see
    # tests.helpers.ddp_worker_entrypoint._report_fault_applied), and it
    # writes it before the fault's effect (raise, exit, or stall) can take
    # hold -- so the report survives even on a path that crashes before
    # reaching its own receipt write. A plan naming a rank or phase that
    # never actually fires (e.g. target_rank outside range(world_size))
    # correctly yields None here instead of echoing an unapplied target.
    culprit_rank: int | None = None
    for rank, log_path in enumerate(log_paths):
        if not log_path.exists():
            continue
        log_text = log_path.read_bytes().decode("utf-8", errors="replace")
        if "ddp harness injected fault" in log_text:
            culprit_rank = rank
            break

    return HarnessResult(
        receipts=tuple(receipts),
        watchdog_fired=watchdog_fired,
        all_reaped=all_reaped,
        culprit_rank=culprit_rank,
        publication_observed=complete_marker_path.exists(),
        rendezvous_path=rendezvous_path_str,
        exit_codes=tuple(exit_codes),
        invocation_dir=str(invocation_dir),
    )


__all__ = [
    "HarnessBounds",
    "HarnessResult",
    "MalformedReceipt",
    "RankReceipt",
    "run_gloo_subprocess_group",
]
