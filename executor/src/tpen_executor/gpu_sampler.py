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
2. After spawn, :func:`start_gpu_sampler` starts a :class:`GpuSampler`
   targeting the exact child PID.  It takes a first sample immediately, then
   one every ``cadence`` seconds for as long as the child lives.
3. When the worker has reaped the child, on every path,
   :func:`finish_gpu_sampler` stops sampling, lets the sampler take its final
   sample, cancels whatever is still in flight at the shared deadline, kills,
   drains, reaps, and confirms every command group empty, and atomically
   rewrites the evidence file.  The worker releases the slot only afterwards.

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
import signal
import stat
import subprocess
import threading
import time
from typing import Any

from ._codec import decode_json
from .gpu_telemetry import (
    DEFAULT_JOIN_TIMEOUT,
    MAX_CADENCE,
    MAX_SAMPLES,
    MIN_CADENCE,
    SAMPLER_METHOD,
    empty_aggregate,
    invalid_sampler,
    normalize_sampler,
)

#: Worker-only environment key naming the evidence file; never seen by the child.
GPU_SAMPLER_OUTPUT_ENV = "TPEN_PLACEMENT_GPU_SAMPLER_OUTPUT"
#: Optional bounded knobs read from the prepared child environment.
GPU_SAMPLER_CADENCE_ENV = "TPEN_PLACEMENT_GPU_SAMPLER_CADENCE"
GPU_SAMPLER_RETAIN_ENV = "TPEN_PLACEMENT_GPU_SAMPLER_RETAIN"
DECLARED_DEVICES_ENV = "TPEN_PLACEMENT_DECLARED_DEVICES"
EVIDENCE_SCHEMA = "tpen-executor/gpu-sampler-evidence-v1"
SAMPLER_EXECUTABLE = "nvidia-smi"
#: Default seconds between samples.  A CUDA context the exact child holds for
#: longer than one cadence plus one sample duration is always observed.
DEFAULT_CADENCE = 2.0
#: Per-command read bound and per-group cleanup bound, in seconds.
COMMAND_TIMEOUT = 10.0
CLEANUP_TIMEOUT = 2.0
#: Shared bound of :meth:`GpuSampler.finish`, in seconds.
JOIN_TIMEOUT = DEFAULT_JOIN_TIMEOUT
#: Maximum finish attempts across repeated owner interruptions.
FINISH_ATTEMPTS = 16
GPU_UUID_PATTERN = re.compile(r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")
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
    by that thread as soon as its handle is published.

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
        Command resolved through the ``PATH`` of ``env``.
    env : mapping, optional
        Command environment (the prepared child environment).
    command_timeout, cleanup_timeout, join_timeout : float
        Per-command read bound, per-group cleanup bound, and the shared bound
        of :meth:`finish`.
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
        # Part of the shared finish bound reserved for cancel-and-cleanup.
        self._cleanup_reserve = min(2 * cleanup_timeout, join_timeout / 2)
        self._lock = threading.Lock()
        self._stop = threading.Event()
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
        self._version: str | None = None
        self._started: float | None = None
        self._ended: float | None = None
        # Retained samples as [sample, pinned] in chronological order.
        self._retained: list[list[Any]] = []
        self._aggregate = empty_aggregate()
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
            Sampler using the module-level timeout bounds.
        """

        try:
            cadence = float(environ.get(GPU_SAMPLER_CADENCE_ENV, DEFAULT_CADENCE))
            retain = int(environ.get(GPU_SAMPLER_RETAIN_ENV, MAX_SAMPLES))
        except (TypeError, ValueError):
            cadence, retain = DEFAULT_CADENCE, MAX_SAMPLES
        return cls(
            target_pid, declared_devices(environ), cadence=cadence, retain=retain, env=environ,
            command_timeout=COMMAND_TIMEOUT, cleanup_timeout=CLEANUP_TIMEOUT, join_timeout=JOIN_TIMEOUT,
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
        """Begin (if needed) and start the daemon sampling thread."""

        self.begin()
        thread = threading.Thread(target=self._run, name="tpen-gpu-sampler", daemon=True)
        try:
            thread.start()
        except RuntimeError as error:
            with self._lock:
                self._failure = ("sampler-error", f"sampler thread did not start: {error}")
                self._ended = time.monotonic()
                self._done = True
            return
        self._thread = thread

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

    def finish(self) -> dict[str, Any]:
        """Stop sampling within the shared deadline and return the final record.

        The first call fixes a deadline ``join_timeout`` seconds ahead; later
        calls (after an interruption) reuse it.  The thread is first given
        the deadline minus a cleanup reserve to take its final sample; then
        the in-flight command is cancelled and the thread gets the rest of
        the deadline.  A thread still alive at the deadline, for example one
        blocked inside a launch, yields ``sampler-join-expired`` evidence that
        is not quiescent.

        Returns
        -------
        dict
            The final wire record.
        """

        with self._lock:
            if self._deadline is None:
                self._deadline = time.monotonic() + self.join_timeout
            deadline = self._deadline
        thread = self._thread
        if thread is not None:
            self._stop.set()
            thread.join(max(0.0, deadline - self._cleanup_reserve - time.monotonic()))
            if thread.is_alive():
                self.cancel()
                thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                with self._lock:
                    self._join_expired = True
        return self.record()

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
            if self._join_expired:
                detail = "sampler thread outlived the shared finish deadline"
                if self._launching:
                    detail += " while a command launch was still in flight"
                failure = ("sampler-join-expired", detail)
            elif failure is None and survivors:
                failure = ("cleanup-unconfirmed", f"command process groups were not confirmed empty: {survivors}")
            elif failure is None and not self._retained:
                failure = ("no-sample", "sampler finished without a complete sample")
            # Quiescence is claimed only for a joined thread whose every
            # command group was confirmed empty.
            quiescent = self._done and not self._join_expired and not survivors
            observed = failure is None and quiescent
            aggregate = self._aggregate
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
            }

    def _run(self) -> None:
        """Sample immediately, then every cadence until stop, then once more."""

        failure: tuple[str, str] | None = ("sampler-error", "sampler thread exited abnormally")
        try:
            version = self._command(VERSION_ARGS).strip()
            if not version:
                raise GpuSamplerFailure("malformed-output", "nvidia-smi --version printed nothing")
            with self._lock:
                self._version = version
            while True:
                # A sample that starts after stop is the final sample, taken
                # once the worker has observed the child's exit.
                final = self._stop.is_set()
                self._take_sample()
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

    def _take_sample(self) -> None:
        """Take one inventory/process sample and fold it into the evidence."""

        sample_start = time.monotonic()
        inventory = []
        pci_by_uuid: dict[str, str] = {}
        for uuid, pci in csv_rows(self._command(INVENTORY_ARGS), 2):
            if not GPU_UUID_PATTERN.fullmatch(uuid):
                raise GpuSamplerFailure("malformed-output", f"inventory UUID is not a physical GPU UUID: {uuid!r}")
            uuid = uuid.upper()
            if uuid in pci_by_uuid:
                raise GpuSamplerFailure("malformed-output", f"inventory repeats GPU UUID {uuid}")
            pci_by_uuid[uuid] = pci
            inventory.append({"uuid": uuid, "pci.bus_id": pci})
        if not inventory:
            raise GpuSamplerFailure("malformed-output", "nvidia-smi inventory is empty")
        rows = []
        for pid, uuid in csv_rows(self._command(PROCESS_ARGS), 2):
            if not (pid.isascii() and pid.isdigit()) or int(pid) <= 0 or not GPU_UUID_PATTERN.fullmatch(uuid):
                raise GpuSamplerFailure("malformed-output", f"process row is malformed: {pid!r}, {uuid!r}")
            uuid = uuid.upper()
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
                (self.executable,) + tuple(args),
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


def start_gpu_sampler(path: str | None, target_pid: int, environ: Mapping[str, str]) -> GpuSampler | None:
    """Start the lifetime sampler for a spawned child that owns evidence.

    Parameters
    ----------
    path : str or None
        Evidence path claimed by :func:`claim_sampler_evidence`, or ``None``.
    target_pid : int
        Spawned science child PID.
    environ : mapping
        Prepared child environment.

    Returns
    -------
    GpuSampler or None
        The running sampler, or ``None`` when this child is not sampled.
    """

    if path is None:
        return None
    sampler = GpuSampler.from_environment(target_pid, environ)
    sampler.start()
    return sampler


def finish_gpu_sampler(sampler: GpuSampler | None, path: str | None) -> list[BaseException]:
    """Finish the sampler, confirm its groups, and publish the final record.

    The worker calls this after reaping the science child on every path and
    before it releases the slot.  An interruption raised while finishing is
    collected and finishing resumes under the same shared deadline, so a
    repeated owner interruption cannot abandon command-group cleanup.

    Parameters
    ----------
    sampler : GpuSampler or None
        Sampler returned by :func:`start_gpu_sampler`.
    path : str or None
        Evidence path owned by this worker.

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
    "COMMAND_TIMEOUT", "CLEANUP_TIMEOUT", "DEFAULT_CADENCE", "EVIDENCE_SCHEMA", "GPU_SAMPLER_CADENCE_ENV",
    "GPU_SAMPLER_OUTPUT_ENV", "GPU_SAMPLER_RETAIN_ENV", "GpuSampler", "GpuSamplerFailure", "JOIN_TIMEOUT",
    "claim_sampler_evidence", "csv_rows", "declared_devices", "finish_gpu_sampler", "group_empty", "kill_group",
    "read_sampler_evidence", "start_gpu_sampler",
]
