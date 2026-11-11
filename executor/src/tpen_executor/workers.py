"""Owner-process subprocess workers for prepared Python environments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import enum
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
from typing import Any

from ._records import frozen_record
from .environments import PreparedPythonEnvironment
from .gpu_sampler import (
    GPU_SAMPLER_OUTPUT_ENV,
    LAUNCH_ROLE_ENV,
    QUARANTINE_IDENTITY_KEYS,
    SLOT_QUARANTINE_ENV,
    GpuSampler,
    acquire_slot_lease,
    claim_sampler_evidence,
    ensure_slot_clean,
    finish_gpu_sampler,
    settle_slot_lease,
    slot_fence_key,
)


def _proc_start_ticks(pid: int) -> int | None:
    """Read a live child's Linux process-generation start token."""

    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    close = stat_text.rfind(")")
    if close < 0:
        return None
    fields_after_comm = stat_text[close + 1 :].strip().split()
    if len(fields_after_comm) <= 19:
        return None
    try:
        ticks = int(fields_after_comm[19])
    except ValueError:
        return None
    return ticks if ticks >= 0 else None


class LaunchPurpose(enum.Enum):
    """Typed purpose of one worker launch (review delta-4 D4-1).

    ``SCIENCE`` is the default for every launch.  ``INSPECTION`` is passed
    only by the trusted inspection call site
    (:meth:`tpen_executor.runners.TPENEvidence.read`), which reads a
    member's own output and is therefore exempt from the slot quarantine
    and never sampled.

    Notes
    -----
    This is a correctness guard against accidental slot reuse, not a
    security boundary: any caller that controls the worker invocation can
    pass ``INSPECTION``.  It is a typed argument rather than an environment
    value so that no descriptor, configuration overlay, or ambient owner
    environment can grant the exemption, and no environment-rewriting
    helper (such as the legacy unsampled science fallback) can infer it.
    """

    SCIENCE = "science"
    INSPECTION = "inspection"


@frozen_record
class ProcessOutcome:
    """Immutable outcome of one subprocess invocation.

    Parameters
    ----------
    argv : tuple of str
        Interpreter and arguments passed to the child.
    returncode : int or None
        Reaped child return code.
    stdout_path, stderr_path : str
        The two caller-named log paths.
    timed_out : bool
        Whether the process-group timeout path was taken.
    pid : int or None
        Spawned child PID, or ``None`` when no child was spawned.
    proc_start_ticks : int or None
        Linux ``/proc/<pid>/stat`` start token captured while the child was
        live, or ``None`` when it could not be observed.
    """

    argv: tuple[str, ...]
    returncode: int | None
    stdout_path: str
    stderr_path: str
    timed_out: bool
    pid: int | None = None
    proc_start_ticks: int | None = None


@frozen_record
class SubprocessWorker:
    """Launch one prepared Python child without shell-interpolated arguments.

    Notes
    -----
    The owner environment is copied first, the descriptor overlay is applied
    second, and the setup script is sourced last. Each invocation starts a
    new POSIX session so timeout cleanup owns the complete process group.
    The caller-named stdout and stderr log files are truncated for each run.
    A timeout must be finite and positive. Cancellation sends ``SIGKILL`` to
    the owned process group, waits until the leader is reaped even if cleanup
    waits are interrupted again, annotates later interruptions on the original
    exception, and then propagates that original exception.

    When the controller assigns a GPU sampler evidence path
    (:data:`~tpen_executor.gpu_sampler.GPU_SAMPLER_OUTPUT_ENV`) in the
    descriptor overlay, this worker owns the lifetime GPU sampler of the
    child: it claims the evidence before spawn, holds the sampler before
    starting it, samples the exact child PID until exit, and on every path
    (normal exit, timeout, cancellation) finishes it within its bound and
    publishes the final record before returning.  Only the descriptor opts a
    launch into sampling: an inherited owner-environment value is ignored,
    and the key is removed from the child environment either way.  A sampled
    launch quarantines its slot (``TPEN_SLOT_ID``) before spawning, in
    process and, when the descriptor names a quarantine directory, durably
    in the attempt store; the quarantine is released only once the
    sampler's command cleanup is confirmed.  Every launch carrying a slot id
    except an inspection (``purpose=LaunchPurpose.INSPECTION``) first
    checks the quarantine and raises
    :class:`~tpen_executor.gpu_sampler.SlotQuarantined` before spawning if
    the slot is still held.  The quarantine directory key is worker-only
    and, like the evidence path, is read only from the descriptor and never
    reaches the child.  The retired environment role key
    (:data:`~tpen_executor.gpu_sampler.LAUNCH_ROLE_ENV`) is never honored
    and is scrubbed from the child.  A timeout or cancellation of the
    child does not by itself make the telemetry fail; only a sampler that is
    itself cancelled or unfinalized does.
    """

    def run(
        self,
        environment: PreparedPythonEnvironment,
        args: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None,
        stdout_path: str | os.PathLike[str],
        stderr_path: str | os.PathLike[str],
        timeout: float | None = None,
        process_handle: Any = None,
        pre_spawn_guard: Callable[[], bool] | None = None,
        purpose: LaunchPurpose = LaunchPurpose.SCIENCE,
    ) -> ProcessOutcome:
        """Run the prepared interpreter with exact argument boundaries.

        Parameters
        ----------
        environment : PreparedPythonEnvironment
            Prepared interpreter, optional setup, and environment overlay.
        args : sequence of str
            Child arguments; strings are never interpolated into shell text.
        cwd : str or os.PathLike or None
            Child working directory.
        stdout_path, stderr_path : str or os.PathLike
            Required caller-named binary logs, truncated for this run.
        timeout : float or None, optional
            Positive timeout in seconds for the whole process group.
        process_handle : object, optional
            Thread-safe owner handle notified of the child process group. The
            handle must provide ``attach(process)`` and ``clear(process)``;
            it may request an owner-side cancellation before attachment.
        pre_spawn_guard : callable returning bool, optional
            Optional thread-safe cancellation check made immediately before
            ``Popen``. When it returns true, no child is spawned and a
            never-spawned ``ProcessOutcome`` is returned.
        purpose : LaunchPurpose, optional
            Launch purpose; ``SCIENCE`` by default.  Only the trusted
            inspection call site passes ``INSPECTION``, which is exempt from
            the slot quarantine and never sampled.  This is a correctness
            guard, not a security boundary (see :class:`LaunchPurpose`).

        Returns
        -------
        ProcessOutcome
            Reaped return code, log paths, and timeout status.

        Raises
        ------
        TypeError, ValueError
            If inputs have invalid types (including a ``purpose`` that is not
            a :class:`LaunchPurpose`) or timeout is not finite and positive.
        OSError
            If process launch or log creation fails.
        SlotQuarantined
            If this is a science launch on a slot whose earlier sampled
            launch still has unconfirmed command cleanup; no child is spawned.
        """

        if not isinstance(environment, PreparedPythonEnvironment):
            raise TypeError("environment must be PreparedPythonEnvironment")
        if isinstance(args, (str, bytes, bytearray)) or any(type(arg) is not str for arg in args):
            raise TypeError("args must be a sequence of strings")
        if pre_spawn_guard is not None and not callable(pre_spawn_guard):
            raise TypeError("pre_spawn_guard must be callable or None")
        if type(purpose) is not LaunchPurpose:
            # A bare string such as "inspection" never grants the exemption.
            raise TypeError("purpose must be a LaunchPurpose")
        if timeout is not None and (
            type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive or None")
        logical_argv = (environment.python, *tuple(args))
        command = list(logical_argv)
        if environment.setup is not None:
            command = [
                "bash",
                "-c",
                'source "$1" && shift && exec "$@"',
                "bash",
                environment.setup.path,
                *logical_argv,
            ]
        child_env = os.environ.copy()
        child_env.update(environment.env)
        # Only the explicit prepared descriptor opts this launch into
        # sampling or names its quarantine directory; an ambient
        # owner-environment value never does (review E3).
        sampler_output = environment.env.get(GPU_SAMPLER_OUTPUT_ENV)
        quarantine_dir = environment.env.get(SLOT_QUARANTINE_ENV)
        # These keys belong to this worker, never the child, whichever
        # environment they came from.  The retired role key is scrubbed and
        # never read: only the typed ``purpose`` argument exempts a launch
        # (review delta-4 D4-1).
        for key in (GPU_SAMPLER_OUTPUT_ENV, SLOT_QUARANTINE_ENV, LAUNCH_ROLE_ENV):
            child_env.pop(key, None)
        inspector = purpose is LaunchPurpose.INSPECTION
        if inspector:
            # An inspection reads a member's output; it is never sampled.
            sampler_output = None
        slot = slot_fence_key(child_env)
        stdout_name = os.fspath(stdout_path)
        stderr_name = os.fspath(stderr_path)
        timed_out = False
        with Path(stdout_name).open("wb") as stdout, Path(stderr_name).open("wb") as stderr:
            if pre_spawn_guard is not None and pre_spawn_guard():
                return ProcessOutcome(logical_argv, None, stdout_name, stderr_name, timed_out)
            # Every science launch on a slot whose earlier sampled launch is
            # still unconfirmed is refused before any evidence claim or spawn,
            # sampled or not.  Only an explicit inspection is exempt; an
            # unsampled launch is never assumed to be one (review delta-3 F1,
            # delta-4 D4-1).
            if not inspector and (slot or sampler_output is not None):
                ensure_slot_clean(slot, quarantine_dir)
            # Claim before spawn: a worker that dies later leaves attempted,
            # unfinalized evidence, never an absent or not-sampled record.
            if not claim_sampler_evidence(sampler_output):
                sampler_output = None
            # Quarantine the slot before spawn, so every later fallible step
            # (including evidence publication) happens under it (delta-3 F2).
            lease = None
            if sampler_output is not None:
                lease = acquire_slot_lease(slot, quarantine_dir, {
                    **{key: child_env.get(key) for key in QUARANTINE_IDENTITY_KEYS},
                    "evidence_path": sampler_output, "owner_pid": os.getpid(), "owner_host": socket.gethostname(),
                })
            try:
                process = subprocess.Popen(
                    command,
                    cwd=None if cwd is None else os.fspath(cwd),
                    env=child_env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except OSError:
                # Popen raised from the launch itself: no child exists.
                settle_slot_lease(lease, None)
                raise
            except BaseException:
                # An interrupted spawn may have left a child no one owns.
                settle_slot_lease(lease, None, child_known=False)
                raise
            spawned_pid = process.pid
            spawned_ticks = _proc_start_ticks(spawned_pid)
            sampler = None
            try:
                if process_handle is not None:
                    process_handle.attach(process)
                if sampler_output is not None:
                    # Hold the sampler before any asynchronous work starts, so
                    # an interruption during or right after start() still
                    # leaves this worker owning its cleanup (review E1).
                    sampler = GpuSampler.from_environment(spawned_pid, child_env)
                    sampler.start()
                try:
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    # Kill the isolated session so setup/science descendants stop too.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = process.wait()
            except BaseException as original:
                # An interrupt while waiting must not orphan the isolated
                # session. Reap the leader before propagating cancellation.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                cleanup_errors: list[BaseException] = []
                while process.returncode is None:
                    try:
                        process.wait()
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
                # The cancelled child is reaped; drain the sampler's command
                # groups and settle the lease before the cancellation leaves
                # this worker.  A further interruption of that finalizer is
                # recorded on the original, which still propagates.
                try:
                    cleanup_errors.extend(finish_gpu_sampler(sampler, sampler_output, lease))
                except BaseException as finish_error:
                    cleanup_errors.append(finish_error)
                for cleanup_error in cleanup_errors:
                    original.add_note(f"cleanup wait interruption: {cleanup_error!r}")
                raise
            finally:
                if process_handle is not None:
                    process_handle.clear(process)
        # Normal exit and worker timeout: the child is reaped; finish the
        # sampler and settle the lease, which stays quarantined if cleanup is
        # still unconfirmed.
        interruptions = finish_gpu_sampler(sampler, sampler_output, lease)
        if interruptions:
            raise interruptions[0]
        return ProcessOutcome(logical_argv, returncode, stdout_name, stderr_name, timed_out, spawned_pid, spawned_ticks)


__all__ = ["LaunchPurpose", "ProcessOutcome", "SubprocessWorker"]
