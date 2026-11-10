"""Bounded in-child GPU PID-to-UUID sampler for the science RUN_SHIM.

The executor embeds this file's source text into
:data:`tpen_executor.runners.RUN_SHIM`, so the code runs inside the science
child's own interpreter, which is not required to have the executor
installed.  The module must therefore stay stdlib-only and valid as a fragment
of one ``python -c`` program: no ``from __future__`` imports, no relative
imports, and module-scope names prefixed ``gpu``/``Gpu``/``GPU`` so it cannot
shadow the shim's own globals.  Executor tests import it directly to drive the
same code against deterministic fake commands.

The record returned by :meth:`GpuSampler.record` is the child half of the
closed ``gpu_sampler`` wire validated by :mod:`tpen_executor.gpu_telemetry`.
Its binding vocabulary is:

``not-sampled``
    Sampling was intentionally never attempted (``attempted`` is false).
``telemetry-failed``
    Sampling was attempted and did not finish with confirmed-quiescent,
    complete evidence; ``failure`` carries a typed code.
``observed``
    Every sample completed, every command process group was confirmed empty,
    and the sampler thread was joined.
"""

import atexit
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time

GPU_SAMPLER_METHOD = "nvidia-smi-pid-uuid-v1"
GPU_SAMPLER_EXECUTABLE = "nvidia-smi"
GPU_UUID_PATTERN = re.compile(r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")
GPU_COMMAND_TIMEOUT = 1.0
GPU_CLEANUP_TIMEOUT = 1.0
GPU_JOIN_TIMEOUT = 4.0
GPU_VERSION_ARGS = ("--version",)
GPU_INVENTORY_ARGS = ("--query-gpu=uuid,pci.bus_id", "--format=csv,noheader,nounits")
GPU_PROCESS_ARGS = ("--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits")


def gpu_not_sampled_record():
    """Return the exact wire record for intentionally unattempted sampling.

    Returns
    -------
    dict
        A fresh ``not-sampled`` record; ``attempted`` is false.
    """

    return {
        "status": "unavailable", "reason": "not-sampled", "method": GPU_SAMPLER_METHOD,
        "version": None, "attempted": False, "failure": None, "cadence_seconds": 0.0,
        "count": 0, "monotonic_start": None, "monotonic_end": None, "samples": [],
        "command_pids": [], "survivors": [], "elapsed_seconds": 0.0,
        "join_timeout_seconds": GPU_JOIN_TIMEOUT, "quiescent": True,
    }


class GpuSamplerFailure(Exception):
    """Typed sampler failure carried into ``failure.code``/``failure.detail``.

    Parameters
    ----------
    code : str
        Closed failure code from :data:`tpen_executor.gpu_telemetry.CHILD_FAILURE_CODES`.
    detail : str
        Human-readable provenance.
    """

    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code
        self.detail = detail


def gpu_csv_rows(raw, width):
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


def gpu_kill_group(pgid):
    """Send SIGKILL to a retained process group, ignoring an already-empty group."""

    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # ESRCH means the group is already gone; EPERM is resolved by the
        # emptiness confirmation, which then reports the group as a survivor.
        pass


def gpu_group_empty(pgid):
    """Return whether ``killpg(pgid, 0)`` reports that no group member exists.

    Parameters
    ----------
    pgid : int
        Process-group id retained from launch.

    Returns
    -------
    bool
        True only on ESRCH.  A permission error means some process still
        holds the id, so the group is not confirmed empty.
    """

    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


class GpuSampler:
    """Bounded background sampler of live PID-to-GPU-UUID rows.

    Each ``nvidia-smi`` invocation runs in its own session, so its process
    group id equals the leader PID and is retained at launch.  Every command
    exit path (success, read error, timeout, cancellation) SIGKILLs that
    retained group, reaps the leader, and polls ``killpg(pgid, 0)`` until it
    reports ESRCH within ``cleanup_timeout``; an unconfirmed group is recorded
    as a survivor and the record is never quiescent.

    Parameters
    ----------
    cadence : float, default=0.1
        Seconds between samples, clamped to ``[0.01, 1.0]``.
    limit : int, default=3
        Maximum number of samples, clamped to ``[1, 16]``.
    executable : str, default="nvidia-smi"
        Command resolved through ``PATH``.
    command_timeout, cleanup_timeout, join_timeout : float
        Per-command read bound, per-group cleanup bound, and final join bound.
    """

    def __init__(
        self,
        cadence=0.1,
        limit=3,
        executable=GPU_SAMPLER_EXECUTABLE,
        command_timeout=GPU_COMMAND_TIMEOUT,
        cleanup_timeout=GPU_CLEANUP_TIMEOUT,
        join_timeout=GPU_JOIN_TIMEOUT,
    ):
        cadence = float(cadence)
        self.cadence = min(max(cadence, 0.01), 1.0) if math.isfinite(cadence) else 0.1
        self.limit = min(max(int(limit), 1), 16)
        self.executable = executable
        self.command_timeout = command_timeout
        self.cleanup_timeout = cleanup_timeout
        self.join_timeout = join_timeout
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        # (process, pgid) of the in-flight command; the cancellation handle.
        self._current = None
        self._attempted = False
        self._cancelled = False
        self._done = False
        self._join_expired = False
        self._failure = None
        self._version = None
        self._started = None
        self._ended = None
        self._samples = []
        self._command_pids = []
        self._survivors = []

    @classmethod
    def from_environment(cls, environ):
        """Build a sampler from the bounded RUN_SHIM environment knobs."""

        try:
            limit = int(environ.get("TPEN_PLACEMENT_GPU_SAMPLER_COUNT", "3"))
            cadence = float(environ.get("TPEN_PLACEMENT_GPU_SAMPLER_CADENCE", "0.1"))
        except (TypeError, ValueError):
            limit, cadence = 3, 0.1
        return cls(cadence, limit)

    def begin(self):
        """Record that sampling is attempted before any command can launch.

        Until the thread finishes, :meth:`record` reports ``telemetry-failed``
        with ``sampler-unfinalized``, so a child killed mid-sample can never
        leave a ``not-sampled`` sidecar behind.
        """

        with self._lock:
            if self._attempted:
                return
            self._attempted = True
            self._started = time.monotonic()
            self._failure = ("sampler-unfinalized", "sampler evidence was not finalized before the sidecar was read")

    def start(self):
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

    def cancel(self):
        """Stop further launches and SIGKILL the in-flight command group.

        The sampling thread's own cleanup then drains, reaps, and confirms the
        retained group, so cancellation needs no second owner of the process.
        """

        with self._lock:
            self._cancelled = True
            current = self._current
        self._stop.set()
        if current is not None:
            gpu_kill_group(current[1])

    def finish(self):
        """Stop sampling within the join bound and return the final record.

        Returns
        -------
        dict
            The final wire record.  When the thread outlives the join bound,
            the in-flight command is cancelled and the thread receives one
            cleanup bound; if it still lives, the record is
            ``sampler-join-expired`` and not quiescent.
        """

        thread = self._thread
        if thread is not None:
            self._stop.set()
            thread.join(self.join_timeout)
            if thread.is_alive():
                self.cancel()
                thread.join(2 * self.cleanup_timeout)
                if thread.is_alive():
                    with self._lock:
                        self._join_expired = True
        return self.record()

    def record(self):
        """Return a JSON-shaped snapshot of the sampler evidence.

        Returns
        -------
        dict
            ``not-sampled`` before :meth:`begin`; otherwise ``observed`` only
            when the thread finished without failure, with at least one
            sample, no survivors, and no join expiry.
        """

        with self._lock:
            if not self._attempted:
                return gpu_not_sampled_record()
            ended = self._ended if self._done else time.monotonic()
            survivors = list(self._survivors)
            failure = self._failure
            if self._join_expired:
                failure = ("sampler-join-expired", "sampler thread outlived its bounded join and cancellation")
            elif failure is None and survivors:
                failure = ("cleanup-unconfirmed", f"command process groups were not confirmed empty: {survivors}")
            elif failure is None and not self._samples:
                failure = ("no-sample", "sampler finished without a complete sample")
            # Quiescence is claimed only for a joined thread whose every
            # command group was confirmed empty.
            quiescent = self._done and not self._join_expired and not survivors
            observed = failure is None and quiescent
            return {
                "status": "observed" if observed else "unavailable",
                "reason": "observed" if observed else "telemetry-failed",
                "method": GPU_SAMPLER_METHOD,
                "version": self._version,
                "attempted": True,
                "failure": None if observed else {"code": failure[0], "detail": failure[1]},
                "cadence_seconds": self.cadence,
                "count": len(self._samples),
                "monotonic_start": self._started,
                "monotonic_end": ended,
                "samples": [
                    {
                        "monotonic_start": sample["monotonic_start"],
                        "monotonic_end": sample["monotonic_end"],
                        "inventory": [dict(row) for row in sample["inventory"]],
                        "pid_rows": [dict(row) for row in sample["pid_rows"]],
                    }
                    for sample in self._samples
                ],
                "command_pids": list(self._command_pids),
                "survivors": survivors,
                "elapsed_seconds": ended - self._started,
                "join_timeout_seconds": self.join_timeout,
                "quiescent": quiescent,
            }

    def _run(self):
        """Take the mandatory first sample, then sample until stop or limit."""

        failure = ("sampler-error", "sampler thread exited abnormally")
        try:
            while True:
                sample_start = time.monotonic()
                version, inventory, rows = self._sample()
                sample_end = time.monotonic()
                with self._lock:
                    self._version = self._version or version
                    self._samples.append({
                        "monotonic_start": sample_start, "monotonic_end": sample_end,
                        "inventory": inventory, "pid_rows": rows,
                    })
                    count = len(self._samples)
                if count >= self.limit or self._stop.wait(self.cadence):
                    break
            failure = None
        except GpuSamplerFailure as error:
            failure = (error.code, error.detail)
        except Exception as error:
            # The sampler must never take down the science child.
            failure = ("sampler-error", f"{type(error).__name__}: {error}")
        finally:
            with self._lock:
                self._failure = failure
                self._ended = time.monotonic()
                self._done = True

    def _sample(self):
        """Run one version/inventory/process query triple and validate it."""

        version = self._command(GPU_VERSION_ARGS).strip()
        if not version:
            raise GpuSamplerFailure("malformed-output", "nvidia-smi --version printed nothing")
        inventory = []
        seen = set()
        for uuid, pci in gpu_csv_rows(self._command(GPU_INVENTORY_ARGS), 2):
            if not GPU_UUID_PATTERN.fullmatch(uuid):
                raise GpuSamplerFailure("malformed-output", f"inventory UUID is not a physical GPU UUID: {uuid!r}")
            uuid = uuid.upper()
            if uuid in seen:
                raise GpuSamplerFailure("malformed-output", f"inventory repeats GPU UUID {uuid}")
            seen.add(uuid)
            inventory.append({"uuid": uuid, "pci.bus_id": pci})
        if not inventory:
            raise GpuSamplerFailure("malformed-output", "nvidia-smi inventory is empty")
        rows = []
        for pid, uuid in gpu_csv_rows(self._command(GPU_PROCESS_ARGS), 2):
            if not (pid.isascii() and pid.isdigit()) or int(pid) <= 0 or not GPU_UUID_PATTERN.fullmatch(uuid):
                raise GpuSamplerFailure("malformed-output", f"process row is malformed: {pid!r}, {uuid!r}")
            uuid = uuid.upper()
            # A PID row must name a GPU from the same sample's inventory.
            if uuid not in seen:
                raise GpuSamplerFailure("malformed-output", f"process row names GPU {uuid} absent from the inventory")
            rows.append({"pid": int(pid), "gpu_uuid": uuid})
        return version, inventory, rows

    def _command(self, args):
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
            try:
                # One session-creation mechanism: start_new_session makes the
                # leader its own group, so the group id is the leader PID.
                process = subprocess.Popen(
                    (self.executable,) + tuple(args),
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, start_new_session=True,
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                raise GpuSamplerFailure("launch-failed", f"{type(error).__name__}: {error}") from None
            # Retain the group id now: getpgid(pid) fails once the leader is reaped.
            pgid = process.pid
            self._command_pids.append(process.pid)
            self._current = (process, pgid)
        failure = None
        stdout = ""
        try:
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
        if cancelled:
            raise GpuSamplerFailure("sampler-cancelled", "sampler was cancelled while a command was in flight")
        if failure is not None:
            raise failure
        if process.returncode != 0:
            raise GpuSamplerFailure("command-failed", f"{args[0]} exited with status {process.returncode}")
        return stdout

    def _reap_group(self, process, pgid):
        """Kill the retained group, reap its leader, and confirm it is empty.

        Returns
        -------
        bool
            True only when ``killpg(pgid, 0)`` reports ESRCH within
            ``cleanup_timeout``; otherwise ``pgid`` is kept as a survivor.
        """

        gpu_kill_group(pgid)
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
        while not gpu_group_empty(pgid):
            if time.monotonic() >= deadline:
                with self._lock:
                    self._survivors.append(pgid)
                return False
            time.sleep(0.01)
        return True


def gpu_finalize_sidecar(sampler, observation, path):
    """Finish the sampler and atomically rewrite the child placement sidecar.

    Parameters
    ----------
    sampler : GpuSampler
        Started sampler owned by this child.
    observation : dict
        Child placement observation; its ``gpu_sampler`` is replaced.
    path : str
        Sidecar path read by the controller after process quiescence.
    """

    observation["gpu_sampler"] = sampler.finish()
    temporary = f"{path}.gpu-{os.getpid()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(observation, sort_keys=True) + "\n")
        os.replace(temporary, path)
    except OSError as error:
        print(f"child-placement-refused: could not finalize sidecar: {error}", file=sys.stderr)


def gpu_register_finalizer(sampler, observation, path):
    """Register :func:`gpu_finalize_sidecar` to run at normal interpreter exit."""

    atexit.register(gpu_finalize_sidecar, sampler, observation, path)
