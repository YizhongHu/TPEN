"""Worker-owned GPU PID-to-UUID sampler for one science child lifetime.

:class:`tpen_executor.workers.SubprocessWorker` launches the science child and
owns this sampler.  Because the worker outlives the child, it can watch the
child's whole lifetime and own cleanup of every ``nvidia-smi`` command group
on every path, including a worker timeout or cancellation that kills the
science process group.

Lifecycle
---------
1. Before the child is spawned, :func:`claim_sampler_evidence` exclusively
   creates the evidence file holding an attempted, unfinalized,
   non-quiescent record.  A worker that dies at any later point leaves that
   blocking ``telemetry-failed`` record, never ``not-sampled``.
2. After spawn, the worker first holds a :class:`GpuSampler` targeting the
   exact child PID and only then calls :meth:`GpuSampler.start`, which
   publishes its thread handle before the thread starts.  An interruption at
   any point of startup therefore leaves the worker holding a sampler it can
   settle (review E1).  The sampler takes a first sample immediately, then
   one every ``cadence`` seconds for as long as the child lives.
3. When the worker has reaped the child, on every path,
   :func:`finish_gpu_sampler` stops sampling, lets the sampler take its final
   sample, cancels whatever is still in flight at the shared deadline, kills,
   drains, reaps, and confirms every command group empty, and atomically
   rewrites the evidence file.
4. If the shared deadline passes while a command launch or a group cleanup
   is still outstanding, the evidence is non-quiescent and the worker fences
   the slot (:func:`ensure_slot_clean`): no later sampled (science) launch
   on that slot by this process proceeds until the outstanding work is
   confirmed gone (review E2).  The bound is kept; the slot is never
   silently reused.  The fence is process-local: every dispatcher runs its
   workers in the controller process, and a process that exits leaves only
   the truthful non-quiescent evidence behind.

Telemetry quality is independent of the run's operational outcome (manager
ruling Q4): a worker timeout or cancellation of the science child still calls
:func:`finish_gpu_sampler`, so usable, quiescent observations survive it.
Only a sampler that is itself cancelled (for example a command still in
flight at the deadline) or never finalized is ``telemetry-failed``.

Timeout budget
--------------
One sample attempt runs at most :data:`ATTEMPT_COMMANDS` commands, each
bounded by :data:`COMMAND_TIMEOUT` plus :data:`CLEANUP_TIMEOUT`.  When finish
starts, at worst one attempt is in progress and one final attempt follows, so
:func:`finish_deadline` is two worst-case attempts plus a cancel-and-cleanup
reserve: ``2 * 3 * (5 + 2) + (2 + 1) = 45`` seconds with the defaults.  A
failed final attempt is retried only while a full worst-case attempt still
fits the remaining deadline.  The values are unmeasured judgment defaults, to
be calibrated on the real-GPU smoke.

Transient command failures are bounded by the gap budget
(:data:`MAX_CONSECUTIVE_GAPS`, :data:`MAX_TOTAL_GAPS`, manager ruling Q1),
recorded in the evidence together with every gap.

``nvidia-smi`` is resolved once through the ``PATH`` of the worker's prepared
child environment (the owner environment overlaid by the descriptor); the
descriptor's setup script is never applied to it.  The resolved path and the
reported version are recorded.  MIG device identifiers are unsupported and
fail with the distinct ``mig-unsupported`` code.

At most :data:`~tpen_executor.gpu_telemetry.MAX_SAMPLES` samples are retained.
The closed aggregate covers every complete sample, retained or not, so
out-of-entitlement use seen once is never forgotten.  The record is validated
by :func:`tpen_executor.gpu_telemetry.validate_sampler_wire`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
from typing import Any

from ._codec import decode_json
from .gpu_telemetry import (
    GAP_CODES,
    MAX_CADENCE,
    MAX_SAMPLES,
    MIN_CADENCE,
    SAMPLER_METHOD,
    empty_aggregate,
    empty_gaps,
    invalid_sampler,
    normalize_sampler,
)

#: Worker-only environment key naming the evidence file; never seen by the child.
GPU_SAMPLER_OUTPUT_ENV = "TPEN_PLACEMENT_GPU_SAMPLER_OUTPUT"
#: Optional bounded knobs read from the prepared child environment.
GPU_SAMPLER_CADENCE_ENV = "TPEN_PLACEMENT_GPU_SAMPLER_CADENCE"
GPU_SAMPLER_RETAIN_ENV = "TPEN_PLACEMENT_GPU_SAMPLER_RETAIN"
DECLARED_DEVICES_ENV = "TPEN_PLACEMENT_DECLARED_DEVICES"
#: Child-environment key naming the slot that an unconfirmed cleanup fences.
SLOT_ID_ENV = "TPEN_SLOT_ID"
EVIDENCE_SCHEMA = "tpen-executor/gpu-sampler-evidence-v1"
SAMPLER_EXECUTABLE = "nvidia-smi"
#: Default seconds between samples.  A CUDA context the exact child holds for
#: longer than one cadence plus one sample duration is always observed.
DEFAULT_CADENCE = 2.0
#: Per-command read bound and per-group cleanup bound, in seconds.
COMMAND_TIMEOUT = 5.0
CLEANUP_TIMEOUT = 2.0
#: Scheduling slack added to the cancel-and-cleanup reserve, in seconds.
SCHEDULING_MARGIN = 1.0
#: Commands in one sample attempt: the version query (until it has
#: succeeded once), the inventory query, and the process query.
ATTEMPT_COMMANDS = 3
#: Gap budget (manager ruling Q1): failed sample attempts with a transient
#: code (:data:`~tpen_executor.gpu_telemetry.GAP_CODES`) that are forgiven.
#: Conservative judgment defaults: an isolated failure is skipped, two in a
#: row or a fourth over the lifetime are telemetry failure.  Recorded in
#: every record; calibrated on the real-GPU smoke.
MAX_CONSECUTIVE_GAPS = 1
MAX_TOTAL_GAPS = 3
#: Maximum finish attempts across repeated owner interruptions.
FINISH_ATTEMPTS = 16


def finish_deadline(command_timeout: float, cleanup_timeout: float) -> float:
    """Return the worst-case shared finish bound for the given command bounds.

    Parameters
    ----------
    command_timeout, cleanup_timeout : float
        Per-command read bound and per-group cleanup bound, in seconds.

    Returns
    -------
    float
        ``2 * ATTEMPT_COMMANDS * (command_timeout + cleanup_timeout)`` (the
        attempt in progress when finish starts plus the final attempt) plus
        the cancel-and-cleanup reserve ``cleanup_timeout + SCHEDULING_MARGIN``.
    """

    attempt = ATTEMPT_COMMANDS * (command_timeout + cleanup_timeout)
    return 2 * attempt + cleanup_timeout + SCHEDULING_MARGIN


#: Shared bound of :meth:`GpuSampler.finish`, in seconds (45 s by default).
JOIN_TIMEOUT = finish_deadline(COMMAND_TIMEOUT, CLEANUP_TIMEOUT)
GPU_UUID_PATTERN = re.compile(r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")
#: MIG device identifiers (``MIG-<uuid>`` or the legacy ``MIG-GPU-.../i/c``).
MIG_PREFIX = "MIG-"
VERSION_ARGS = ("--version",)
INVENTORY_ARGS = ("--query-gpu=uuid,pci.bus_id", "--format=csv,noheader,nounits")
PROCESS_ARGS = ("--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits")


class GpuSamplerFailure(Exception):
    """Typed sampler failure carried into ``failure.code``/``failure.detail``.

    Parameters
    ----------
    code : str
        Closed code from :data:`tpen_executor.gpu_telemetry.SAMPLER_FAILURE_CODES`.
    detail : str
        Human-readable provenance.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def csv_rows(raw: str, width: int) -> list[list[str]]:
    """Split ``csv,noheader`` output into rows of exactly ``width`` cells.

    Parameters
    ----------
    raw : str
        Command standard output.
    width : int
        Required number of non-empty cells per row.

    Returns
    -------
    list of list of str
        Stripped cells; blank lines are ignored.

    Raises
    ------
    GpuSamplerFailure
        ``malformed-output`` if any non-blank row has the wrong shape.
    """

    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != width or not all(cells):
            raise GpuSamplerFailure("malformed-output", f"nvidia-smi row has malformed columns: {line!r}")
        rows.append(cells)
    return rows


def kill_group(pgid: int) -> None:
    """Send SIGKILL to a retained process group, ignoring an already-empty group."""

    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # ESRCH means the group is already gone; EPERM is resolved by the
        # emptiness confirmation, which then reports the group as a survivor.
        pass


def group_empty(pgid: int) -> bool:
    """Return whether ``killpg(pgid, 0)`` reports that no group member exists.

    Parameters
    ----------
    pgid : int
        Process-group id retained from launch.

    Returns
    -------
    bool
        True only on ESRCH.  A zombie member still holds the group, so an
        unreaped zombie is never reported empty.
    """

    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def declared_devices(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Return the canonical declared GPU entitlement from a child environment."""

    return tuple(item.strip().upper() for item in environ.get(DECLARED_DEVICES_ENV, "").split(",") if item.strip())


def _device_uuid(value: str, what: str) -> str:
    """Return a canonical physical GPU UUID from ``nvidia-smi`` output.

    Raises
    ------
    GpuSamplerFailure
        ``mig-unsupported`` for a MIG device identifier, otherwise
        ``malformed-output`` for anything that is not a physical GPU UUID.
    """

    if value.upper().startswith(MIG_PREFIX):
        raise GpuSamplerFailure("mig-unsupported", f"{what} names a MIG device, which the sampler does not support: {value!r}")
    if not GPU_UUID_PATTERN.fullmatch(value):
        raise GpuSamplerFailure("malformed-output", f"{what} is not a physical GPU UUID: {value!r}")
    return value.upper()


class GpuSampler:
    """Lifetime sampler of live PID-to-GPU-UUID rows for one target PID.

    Each ``nvidia-smi`` invocation runs in its own session, so its process
    group id equals the leader PID and is retained at launch.  Every command
    exit path (success, read error, timeout, cancellation, late launch)
    SIGKILLs that retained group, reaps the leader, and polls
    ``killpg(pgid, 0)`` until it reports ESRCH within ``cleanup_timeout``; an
    unconfirmed group is recorded as a survivor and the record is never
    quiescent.

    The lock is never held across a launch, a read, or a wait, so
    :meth:`cancel`, :meth:`record`, and :meth:`finish` cannot be blocked by a
    slow ``Popen``.  The sampling thread is the sole owner of every process it
    launches: a launch that completes after cancellation is killed and reaped
    by that thread as soon as its handle is published.  A launch begins only
    under the lock and never after :meth:`cancel`, so once cancelled the set
    of processes that can ever exist is fixed; quiescence is a statement
    about that set, not about the Python thread.

    Parameters
    ----------
    target_pid : int
        PID of the science child whose rows decide the device verdict.
    declared : sequence of str
        Canonical declared GPU entitlement of the slot.
    cadence : float, default=DEFAULT_CADENCE
        Seconds between samples, clamped to the wire bounds.
    retain : int, default=16
        Maximum number of retained samples, clamped to ``[1, 16]``.
    executable : str, default="nvidia-smi"
        Command resolved once through the ``PATH`` of ``env``.
    env : mapping, optional
        Command environment (the prepared child environment).
    command_timeout, cleanup_timeout, join_timeout : float
        Per-command read bound, per-group cleanup bound, and the shared bound
        of :meth:`finish`.
    max_consecutive_gaps, max_total_gaps : int
        Gap budget for transient failed sample attempts.
    """

    def __init__(
        self,
        target_pid: int,
        declared: Sequence[str],
        *,
        cadence: float = DEFAULT_CADENCE,
        retain: int = MAX_SAMPLES,
        executable: str = SAMPLER_EXECUTABLE,
        env: Mapping[str, str] | None = None,
        command_timeout: float = COMMAND_TIMEOUT,
        cleanup_timeout: float = CLEANUP_TIMEOUT,
        join_timeout: float = JOIN_TIMEOUT,
        max_consecutive_gaps: int = MAX_CONSECUTIVE_GAPS,
        max_total_gaps: int = MAX_TOTAL_GAPS,
    ) -> None:
        cadence = float(cadence)
        self.target_pid = int(target_pid)
        self.declared = tuple(declared)
        self.cadence = min(max(cadence, MIN_CADENCE), MAX_CADENCE) if math.isfinite(cadence) else DEFAULT_CADENCE
        self.retain = min(max(int(retain), 1), MAX_SAMPLES)
        self.executable = executable
        self.env = None if env is None else dict(env)
        self.command_timeout = command_timeout
        self.cleanup_timeout = cleanup_timeout
        self.join_timeout = join_timeout
        self.max_consecutive_gaps = int(max_consecutive_gaps)
        self.max_total_gaps = int(max_total_gaps)
        # Worst case of one sample attempt, and the part of the shared finish
        # bound reserved for cancel-and-cleanup.
        self._attempt_worst = ATTEMPT_COMMANDS * (command_timeout + cleanup_timeout)
        self._cancel_reserve = min(cleanup_timeout + SCHEDULING_MARGIN, join_timeout / 2)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None
        self._deadline: float | None = None
        # Retained pgid of the in-flight command; the cancellation handle.
        self._current: int | None = None
        self._launching = False
        self._attempted = False
        self._cancelled = False
        self._done = False
        self._join_expired = False
        self._failure: tuple[str, str] | None = None
        # Detail of the gap-budget excess, set under the lock with the gap
        # that caused it so no snapshot can show an excess without its code.
        self._exceeded: str | None = None
        self._resolved: str | None = None
        self._version: str | None = None
        self._started: float | None = None
        self._ended: float | None = None
        # Retained samples as [sample, pinned] in chronological order.
        self._retained: list[list[Any]] = []
        self._aggregate = empty_aggregate()
        self._gaps = empty_gaps(self.max_consecutive_gaps, self.max_total_gaps)
        self._consecutive = 0
        self._command_pids: list[int] = []
        self._survivors: list[int] = []

    @classmethod
    def from_environment(cls, target_pid: int, environ: Mapping[str, str]) -> "GpuSampler":
        """Build a sampler from the prepared child environment.

        Parameters
        ----------
        target_pid : int
            Spawned science child PID.
        environ : mapping
            Child environment: declared devices, optional bounded cadence and
            retention knobs, and the ``PATH`` used to resolve ``nvidia-smi``.

        Returns
        -------
        GpuSampler
            Sampler using the module-level timeout bounds and gap budget.
        """

        try:
            cadence = float(environ.get(GPU_SAMPLER_CADENCE_ENV, DEFAULT_CADENCE))
            retain = int(environ.get(GPU_SAMPLER_RETAIN_ENV, MAX_SAMPLES))
        except (TypeError, ValueError):
            cadence, retain = DEFAULT_CADENCE, MAX_SAMPLES
        return cls(
            target_pid, declared_devices(environ), cadence=cadence, retain=retain, env=environ,
            command_timeout=COMMAND_TIMEOUT, cleanup_timeout=CLEANUP_TIMEOUT, join_timeout=JOIN_TIMEOUT,
            max_consecutive_gaps=MAX_CONSECUTIVE_GAPS, max_total_gaps=MAX_TOTAL_GAPS,
        )

    def begin(self) -> None:
        """Record that sampling is attempted before any command can launch."""

        with self._lock:
            if self._attempted:
                return
            self._attempted = True
            self._started = time.monotonic()
            self._failure = ("sampler-unfinalized", "sampler evidence was not finalized before it was read")

    def start(self) -> None:
        """Begin (if needed) and start the daemon sampling thread.

        The thread handle is published before ``Thread.start`` so an owner
        that holds this sampler can always settle it.  If starting is
        interrupted, whether the thread began is unknown, so every further
        launch is forbidden and the interruption propagates; :meth:`finish`
        then settles on process state alone.
        """

        self.begin()
        thread = threading.Thread(target=self._run, name="tpen-gpu-sampler", daemon=True)
        with self._lock:
            self._thread = thread
        try:
            thread.start()
        except RuntimeError as error:
            # The interpreter refused to create the thread: it never runs.
            with self._lock:
                self._failure = ("sampler-error", f"sampler thread did not start: {error}")
                self._ended = time.monotonic()
                self._done = True
            self._finished.set()
        except BaseException:
            with self._lock:
                self._failure = ("sampler-cancelled", "sampler start was interrupted")
            self.cancel()
            raise

    def cancel(self) -> None:
        """Stop further launches and SIGKILL the in-flight command group.

        The kill is a non-blocking syscall made under the lock, so it cannot
        race with the owning thread clearing the handle.  That thread then
        drains, reaps, and confirms the retained group.
        """

        with self._lock:
            self._cancelled = True
            self._stop.set()
            if self._current is not None:
                kill_group(self._current)

    def _quiescent_locked(self) -> bool:
        """Return whether no command process can exist now or later (lock held).

        True when no further launch can begin (the thread finished, or the
        sampler is cancelled), no launch is in flight, no published command
        is still being reaped, and no group was left unconfirmed.
        """

        return (
            (self._done or self._cancelled)
            and not self._launching
            and self._current is None
            and not self._survivors
        )

    def _settled(self) -> bool:
        """Return whether finish may stop waiting: done, or cancelled and quiescent."""

        with self._lock:
            return self._done or (self._cancelled and self._quiescent_locked())

    def _wait_settled(self, until: float) -> bool:
        """Wait until :meth:`_settled` or the monotonic time ``until``."""

        while not self._settled():
            remaining = until - time.monotonic()
            if remaining <= 0:
                return False
            self._finished.wait(min(remaining, 0.01))
        return True

    def finish(self) -> dict[str, Any]:
        """Stop sampling within the shared deadline and return the final record.

        The first call fixes a deadline ``join_timeout`` seconds ahead; later
        calls (after an interruption) reuse it.  The thread is first given
        the deadline minus the cancel reserve to take its final sample; then
        the in-flight command is cancelled and the rest of the deadline is
        spent waiting for every launched group to be confirmed gone.  If a
        launch or a cleanup is still outstanding at the deadline, the record
        is ``sampler-join-expired`` and not quiescent.

        Returns
        -------
        dict
            The final wire record.
        """

        with self._lock:
            if self._deadline is None:
                self._deadline = time.monotonic() + self.join_timeout
            deadline = self._deadline
        self._stop.set()
        if not self._wait_settled(deadline - self._cancel_reserve):
            self.cancel()
            if not self._wait_settled(deadline):
                with self._lock:
                    self._join_expired = True
        return self.record()

    def cleanup_confirmed(self) -> bool:
        """Return whether every command process of this sampler is confirmed gone.

        Unlike the frozen record, a group recorded as a survivor is checked
        again, so cleanup that completed after the deadline is recognised.

        Returns
        -------
        bool
            True when no launch can begin or is in flight, no published
            command is being reaped, and every survivor group reports ESRCH.
        """

        with self._lock:
            if not (self._done or self._cancelled) or self._launching or self._current is not None:
                return False
            survivors = list(self._survivors)
        return all(group_empty(pgid) for pgid in survivors)

    def outstanding(self) -> str:
        """Describe the outstanding work that keeps cleanup unconfirmed."""

        with self._lock:
            parts = []
            if self._launching:
                parts.append("a command launch is in flight")
            if self._current is not None:
                parts.append(f"command group {self._current} is still being reaped")
            if self._survivors:
                parts.append(f"command groups {self._survivors} were not confirmed empty")
            if not (self._done or self._cancelled):
                parts.append("the sampling thread may still launch commands")
            return "; ".join(parts) or "none"

    def record(self) -> dict[str, Any]:
        """Return a JSON-shaped snapshot of the sampler evidence.

        Returns
        -------
        dict
            ``observed`` only when the thread finished without failure, with
            at least one sample, no survivors, and no join expiry; otherwise
            attempted ``telemetry-failed`` with a typed failure.
        """

        with self._lock:
            ended = self._ended if self._done else time.monotonic()
            started = self._started if self._started is not None else ended
            survivors = list(self._survivors)
            failure = self._failure
            if self._exceeded is not None:
                failure = ("gap-budget-exceeded", self._exceeded)
            elif self._join_expired:
                detail = "sampler work outlived the shared finish deadline"
                if self._launching:
                    detail += " while a command launch was still in flight"
                failure = ("sampler-join-expired", detail)
            elif not self._done and self._cancelled and (failure is None or failure[0] == "sampler-unfinalized"):
                failure = ("sampler-cancelled", "sampler was cancelled before its thread finished")
            elif failure is None and survivors:
                failure = ("cleanup-unconfirmed", f"command process groups were not confirmed empty: {survivors}")
            elif failure is None and not self._retained:
                failure = ("no-sample", "sampler finished without a complete sample")
            # Quiescence is claimed only when no command process can exist.
            quiescent = self._quiescent_locked()
            observed = failure is None and quiescent
            aggregate = self._aggregate
            gaps = self._gaps
            return {
                "status": "observed" if observed else "unavailable",
                "reason": "observed" if observed else "telemetry-failed",
                "method": SAMPLER_METHOD,
                "version": self._version,
                "attempted": True,
                "failure": None if observed else {"code": failure[0], "detail": failure[1]},
                "cadence_seconds": self.cadence,
                "count": len(self._retained),
                "monotonic_start": started,
                "monotonic_end": ended,
                "samples": [
                    {
                        "monotonic_start": sample["monotonic_start"],
                        "monotonic_end": sample["monotonic_end"],
                        "inventory": [dict(row) for row in sample["inventory"]],
                        "pid_rows": [dict(row) for row in sample["pid_rows"]],
                    }
                    for sample, _pinned in self._retained
                ],
                "command_pids": list(self._command_pids),
                "survivors": survivors,
                "elapsed_seconds": ended - started,
                "join_timeout_seconds": self.join_timeout,
                "quiescent": quiescent,
                "target_pid": self.target_pid,
                "aggregate": {
                    "samples_taken": aggregate["samples_taken"],
                    "exact_pid_gpus": [dict(row) for row in aggregate["exact_pid_gpus"]],
                    "outside_use_seen": aggregate["outside_use_seen"],
                    "entitlement_unresolved_seen": aggregate["entitlement_unresolved_seen"],
                },
                "executable": self._resolved,
                "gaps": dict(gaps, events=[dict(event) for event in gaps["events"]]),
            }

    def _resolve(self) -> str:
        """Resolve the executable once through the prepared ``PATH``.

        Raises
        ------
        GpuSamplerFailure
            ``launch-failed`` when no executable is found.
        """

        source = os.environ if self.env is None else self.env
        search = source.get("PATH", os.defpath)
        resolved = shutil.which(self.executable, path=search)
        if resolved is None:
            raise GpuSamplerFailure("launch-failed", f"{self.executable} was not found on the prepared PATH {search!r}")
        with self._lock:
            self._resolved = resolved
        return resolved

    def _run(self) -> None:
        """Sample immediately, then every cadence until stop, then once more.

        A failed attempt with a transient code is a gap: it is recorded and
        sampling continues while the gap budget holds.  A failed final
        attempt is retried only while a worst-case attempt still fits the
        remaining finish deadline.
        """

        failure: tuple[str, str] | None = ("sampler-error", "sampler thread exited abnormally")
        try:
            self._resolve()
            while True:
                # An attempt that starts after stop is the final one, taken
                # once the worker has observed the child's exit.
                final = self._stop.is_set()
                attempt_start = time.monotonic()
                try:
                    self._attempt()
                except GpuSamplerFailure as error:
                    if error.code not in GAP_CODES:
                        raise
                    self._record_gap(error, attempt_start)
                    if final:
                        if not self._final_retry_fits():
                            raise GpuSamplerFailure(
                                error.code, f"final sample failed and no retry fits the finish deadline: {error.detail}",
                            ) from None
                        continue
                    self._stop.wait(self.cadence)
                    continue
                with self._lock:
                    self._consecutive = 0
                if final:
                    break
                self._stop.wait(self.cadence)
            failure = None
        except GpuSamplerFailure as error:
            failure = (error.code, error.detail)
        except Exception as error:
            # The sampler must never take down the worker.
            failure = ("sampler-error", f"{type(error).__name__}: {error}")
        finally:
            with self._lock:
                self._failure = failure
                self._ended = time.monotonic()
                self._done = True
            self._finished.set()

    def _final_retry_fits(self) -> bool:
        """Return whether a worst-case attempt still fits the finish deadline."""

        with self._lock:
            deadline = self._deadline
        if deadline is None:
            return True
        return deadline - time.monotonic() >= self._attempt_worst + self._cancel_reserve

    def _record_gap(self, error: GpuSamplerFailure, attempt_start: float) -> None:
        """Record one failed attempt as a gap and enforce the gap budget.

        Raises
        ------
        GpuSamplerFailure
            ``gap-budget-exceeded`` when this gap passes either budget.
        """

        with self._lock:
            self._consecutive += 1
            event = {
                "code": error.code, "monotonic_start": attempt_start, "monotonic_end": time.monotonic(),
                "consecutive": self._consecutive,
            }
            gaps = self._gaps
            gaps["events"].append(event)
            gaps["count"] += 1
            gaps["max_consecutive"] = max(gaps["max_consecutive"], self._consecutive)
            gaps["total_seconds"] += event["monotonic_end"] - event["monotonic_start"]
            count, consecutive = gaps["count"], self._consecutive
            if count > self.max_total_gaps or consecutive > self.max_consecutive_gaps:
                self._exceeded = (
                    f"{count} gaps ({consecutive} consecutive) exceed the budget of {self.max_total_gaps} total "
                    f"and {self.max_consecutive_gaps} consecutive; last: {error.code}: {error.detail}"
                )
            exceeded = self._exceeded
        if exceeded is not None:
            raise GpuSamplerFailure("gap-budget-exceeded", exceeded)

    def _attempt(self) -> None:
        """Run one sample attempt: the version query until known, then one sample."""

        with self._lock:
            known = self._version is not None
        if not known:
            version = self._command(VERSION_ARGS).strip()
            if not version:
                raise GpuSamplerFailure("malformed-output", "nvidia-smi --version printed nothing")
            with self._lock:
                self._version = version
        self._take_sample()

    def _take_sample(self) -> None:
        """Take one inventory/process sample and fold it into the evidence."""

        sample_start = time.monotonic()
        inventory = []
        pci_by_uuid: dict[str, str] = {}
        for uuid, pci in csv_rows(self._command(INVENTORY_ARGS), 2):
            uuid = _device_uuid(uuid, "inventory UUID")
            if uuid in pci_by_uuid:
                raise GpuSamplerFailure("malformed-output", f"inventory repeats GPU UUID {uuid}")
            pci_by_uuid[uuid] = pci
            inventory.append({"uuid": uuid, "pci.bus_id": pci})
        if not inventory:
            raise GpuSamplerFailure("malformed-output", "nvidia-smi inventory is empty")
        rows = []
        for pid, uuid in csv_rows(self._command(PROCESS_ARGS), 2):
            uuid = _device_uuid(uuid, "process row UUID")
            if not (pid.isascii() and pid.isdigit()) or int(pid) <= 0:
                raise GpuSamplerFailure("malformed-output", f"process row is malformed: {pid!r}, {uuid!r}")
            # A PID row must name a GPU from the same sample's inventory.
            if uuid not in pci_by_uuid:
                raise GpuSamplerFailure("malformed-output", f"process row names GPU {uuid} absent from the inventory")
            rows.append({"pid": int(pid), "gpu_uuid": uuid})
        sample = {
            "monotonic_start": sample_start, "monotonic_end": time.monotonic(),
            "inventory": inventory, "pid_rows": rows,
        }
        with self._lock:
            self._fold(sample, pci_by_uuid)

    def _fold(self, sample: dict[str, Any], pci_by_uuid: Mapping[str, str]) -> None:
        """Fold one complete sample into the aggregate and the retained window.

        Called with the lock held.  The aggregate covers every sample; the
        retained window keeps the newest sample and pins the first sample and
        the first witness of each new exact-target GPU (at most ``retain - 1``
        pins), evicting the oldest unpinned sample when full.
        """

        aggregate = self._aggregate
        known = {row["uuid"] for row in aggregate["exact_pid_gpus"]}
        novel = aggregate["samples_taken"] == 0
        for row in sample["pid_rows"]:
            if row["pid"] != self.target_pid:
                continue
            uuid = row["gpu_uuid"]
            if uuid not in known:
                known.add(uuid)
                novel = True
                aggregate["exact_pid_gpus"].append({"uuid": uuid, "pci.bus_id": pci_by_uuid[uuid]})
            # Outside use is sticky: once seen it is never cleared.
            aggregate["outside_use_seen"] = aggregate["outside_use_seen"] or uuid not in self.declared
        if not set(self.declared) <= set(pci_by_uuid):
            aggregate["entitlement_unresolved_seen"] = True
        aggregate["samples_taken"] += 1
        pins = sum(1 for _sample, pinned in self._retained if pinned)
        self._retained.append([sample, novel and pins < self.retain - 1])
        while len(self._retained) > self.retain:
            oldest = next(index for index, (_sample, pinned) in enumerate(self._retained) if not pinned)
            del self._retained[oldest]

    def _command(self, args: tuple[str, ...]) -> str:
        """Run one command in its own session and clean its group on every path.

        Raises
        ------
        GpuSamplerFailure
            ``launch-failed``, ``command-timeout``, ``read-failed``,
            ``cleanup-unconfirmed``, ``sampler-cancelled``, or
            ``command-failed``.
        """

        with self._lock:
            if self._cancelled:
                raise GpuSamplerFailure("sampler-cancelled", "sampler was cancelled before the command launched")
            self._launching = True
        try:
            # One session-creation mechanism: start_new_session makes the
            # leader its own group, so the group id is the leader PID.  The
            # launch happens outside the lock so it never blocks cancel/finish.
            process = subprocess.Popen(
                (self._resolved or self.executable,) + tuple(args),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True, env=self.env,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            with self._lock:
                self._launching = False
            raise GpuSamplerFailure("launch-failed", f"{type(error).__name__}: {error}") from None
        # Retain the group id now: getpgid(pid) fails once the leader is reaped.
        pgid = process.pid
        with self._lock:
            self._launching = False
            self._command_pids.append(pgid)
            self._current = pgid
            late = self._cancelled
        failure = None
        stdout = ""
        try:
            if late:
                # Cancellation raced with the launch: this thread owns the
                # late process and kills it before reading anything.
                kill_group(pgid)
                failure = GpuSamplerFailure("sampler-cancelled", "sampler was cancelled while the command was launching")
            else:
                stdout, _ = process.communicate(timeout=self.command_timeout)
        except subprocess.TimeoutExpired:
            failure = GpuSamplerFailure("command-timeout", f"{args[0]} exceeded {self.command_timeout} s")
        except (OSError, ValueError) as error:
            failure = GpuSamplerFailure("read-failed", f"{type(error).__name__}: {error}")
        finally:
            confirmed = self._reap_group(process, pgid)
            with self._lock:
                self._current = None
                cancelled = self._cancelled
        if not confirmed:
            raise GpuSamplerFailure("cleanup-unconfirmed", f"process group {pgid} was not confirmed empty")
        if failure is not None:
            raise failure
        if cancelled:
            raise GpuSamplerFailure("sampler-cancelled", "sampler was cancelled while a command was in flight")
        if process.returncode != 0:
            raise GpuSamplerFailure("command-failed", f"{args[0]} exited with status {process.returncode}")
        return stdout

    def _reap_group(self, process: subprocess.Popen, pgid: int) -> bool:
        """Kill the retained group, reap its leader, and confirm it is empty.

        Returns
        -------
        bool
            True only when ``killpg(pgid, 0)`` reports ESRCH within
            ``cleanup_timeout``; otherwise ``pgid`` is kept as a survivor.
        """

        kill_group(pgid)
        deadline = time.monotonic() + self.cleanup_timeout
        if process.returncode is None:
            try:
                # Drain the pipes and reap the leader after the group kill.
                process.communicate(timeout=self.cleanup_timeout)
            except (subprocess.TimeoutExpired, OSError, ValueError):
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except (subprocess.TimeoutExpired, OSError):
                    pass
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        while not group_empty(pgid):
            if time.monotonic() >= deadline:
                with self._lock:
                    self._survivors.append(pgid)
                return False
            time.sleep(0.01)
        return True


def _write_evidence(path: str, record: Mapping[str, Any]) -> None:
    """Atomically replace the evidence file with one sampler record."""

    temporary = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": EVIDENCE_SCHEMA, "gpu_sampler": record}, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _unfinalized_record() -> dict[str, Any]:
    """Return the attempted record written before the child is spawned."""

    record = invalid_sampler("worker sampler evidence was not finalized")
    record["failure"] = {"code": "sampler-unfinalized", "detail": "worker sampler evidence was not finalized"}
    return record


def claim_sampler_evidence(path: str | None) -> bool:
    """Exclusively create the evidence file before the child is spawned.

    Parameters
    ----------
    path : str or None
        Controller-assigned evidence path, or ``None`` when the slot is not
        sampled.

    Returns
    -------
    bool
        True when this worker now owns the file and must sample.  An existing
        file is never overwritten, so a second process launched with the same
        environment (for example an inspection) cannot replace the science
        child's evidence.
    """

    if path is None:
        return False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except OSError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": EVIDENCE_SCHEMA, "gpu_sampler": _unfinalized_record()}, sort_keys=True) + "\n")
    return True


class SlotQuarantined(RuntimeError):
    """Refusal to launch on a slot whose earlier sampler cleanup is unconfirmed.

    Raised by :func:`ensure_slot_clean` before any child is spawned, so the
    dispatcher records an ordinary launch failure for the member and the
    fenced slot is never reused while a command may still be alive.
    """


# Process-wide slot fence: slot id -> samplers whose cleanup is unconfirmed.
# Every executor dispatcher runs its workers in this (controller) process.
_FENCE_LOCK = threading.Lock()
_FENCED: dict[str, list[GpuSampler]] = {}


def slot_fence_key(environ: Mapping[str, str]) -> str:
    """Return the slot id that a worker launch with ``environ`` occupies."""

    return environ.get(SLOT_ID_ENV, "")


def fence_slot(slot: str, sampler: GpuSampler) -> None:
    """Fence ``slot`` until ``sampler``'s outstanding command work is confirmed gone.

    Parameters
    ----------
    slot : str
        Slot id from :func:`slot_fence_key`.
    sampler : GpuSampler
        Finished sampler whose cleanup was not confirmed within its bound.
    """

    with _FENCE_LOCK:
        _FENCED.setdefault(slot, []).append(sampler)


def ensure_slot_clean(slot: str) -> None:
    """Refuse to reuse a slot while an earlier sampler's cleanup is unconfirmed.

    Each fenced sampler is cancelled again (re-killing any published group)
    and re-checked with :meth:`GpuSampler.cleanup_confirmed`; confirmed ones
    are released from the fence.

    Parameters
    ----------
    slot : str
        Slot id of the launch about to happen.

    Raises
    ------
    SlotQuarantined
        If any earlier sampler on ``slot`` still has outstanding work.
    """

    with _FENCE_LOCK:
        pending = _FENCED.get(slot, [])
        for sampler in pending:
            sampler.cancel()
        remaining = [sampler for sampler in pending if not sampler.cleanup_confirmed()]
        if remaining:
            _FENCED[slot] = remaining
        else:
            _FENCED.pop(slot, None)
    if remaining:
        detail = "; ".join(sampler.outstanding() for sampler in remaining)
        raise SlotQuarantined(f"slot {slot!r} is fenced: earlier GPU sampler cleanup is unconfirmed ({detail})")


def finish_gpu_sampler(sampler: GpuSampler | None, path: str | None, slot: str = "") -> list[BaseException]:
    """Finish the sampler, confirm its groups, publish the record, and fence if dirty.

    The worker calls this after reaping the science child on every path.  An
    interruption raised while finishing is collected and finishing resumes
    under the same shared deadline, so a repeated owner interruption cannot
    abandon command-group cleanup.  If cleanup is still unconfirmed when the
    bound is spent, ``slot`` is fenced (:func:`fence_slot`) before return.

    Parameters
    ----------
    sampler : GpuSampler or None
        Sampler held by the worker since before it was started.
    path : str or None
        Evidence path owned by this worker.
    slot : str, default=""
        Slot id of the launch, from :func:`slot_fence_key`.

    Returns
    -------
    list of BaseException
        Interruptions observed while finishing, for the caller to propagate.
    """

    if sampler is None or path is None:
        return []
    interruptions: list[BaseException] = []
    record = None
    # finish() is bounded by its shared deadline, so retries end promptly;
    # the attempt cap only guards against a deterministic error.
    for _attempt in range(FINISH_ATTEMPTS):
        try:
            record = sampler.finish()
            break
        except BaseException as error:
            interruptions.append(error)
    if record is not None:
        try:
            _write_evidence(path, record)
        except OSError:
            # The claimed unfinalized record stays in place: still attempted,
            # telemetry-failed, and science-blocking.
            pass
    # Bounded is not clean: outstanding launch or cleanup work fences the slot.
    if not sampler.cleanup_confirmed():
        sampler.cancel()
        fence_slot(slot, sampler)
    return interruptions


def read_sampler_evidence(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read worker evidence for a sampled slot as a validated sampler record.

    Parameters
    ----------
    path : str or os.PathLike
        Controller-assigned evidence path.

    Returns
    -------
    dict
        The validated record, or typed ``sampler-evidence-invalid`` failed
        evidence when the file is missing, not a regular file, or malformed.
        A sampled slot without valid evidence is never ``not-sampled``.
    """

    candidate = Path(path)
    try:
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            return invalid_sampler("worker sampler evidence is not a regular file")
        payload = decode_json(candidate.read_bytes())
    except FileNotFoundError:
        return invalid_sampler("worker sampler evidence is missing")
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        return invalid_sampler(f"worker sampler evidence is unreadable: {type(error).__name__}")
    if not isinstance(payload, dict) or set(payload) != {"schema", "gpu_sampler"} or payload["schema"] != EVIDENCE_SCHEMA:
        return invalid_sampler("worker sampler evidence keys or schema differ")
    return normalize_sampler(payload["gpu_sampler"])


__all__ = [
    "ATTEMPT_COMMANDS", "COMMAND_TIMEOUT", "CLEANUP_TIMEOUT", "DEFAULT_CADENCE", "EVIDENCE_SCHEMA",
    "GPU_SAMPLER_CADENCE_ENV", "GPU_SAMPLER_OUTPUT_ENV", "GPU_SAMPLER_RETAIN_ENV", "GpuSampler", "GpuSamplerFailure",
    "JOIN_TIMEOUT", "MAX_CONSECUTIVE_GAPS", "MAX_TOTAL_GAPS", "SCHEDULING_MARGIN", "SLOT_ID_ENV", "SlotQuarantined",
    "claim_sampler_evidence", "csv_rows", "declared_devices", "ensure_slot_clean", "fence_slot", "finish_deadline",
    "finish_gpu_sampler", "group_empty", "kill_group", "read_sampler_evidence", "slot_fence_key",
]
