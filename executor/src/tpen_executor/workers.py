"""Owner-process subprocess workers for prepared Python environments."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import subprocess
from typing import Any

from .environments import PreparedPythonEnvironment


@dataclass(frozen=True, slots=True)
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
    """

    argv: tuple[str, ...]
    returncode: int | None
    stdout_path: str
    stderr_path: str
    timed_out: bool


@dataclass(frozen=True, slots=True)
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

        Returns
        -------
        ProcessOutcome
            Reaped return code, log paths, and timeout status.

        Raises
        ------
        TypeError, ValueError
            If inputs have invalid types or timeout is not finite and positive.
        OSError
            If process launch or log creation fails.
        """

        if not isinstance(environment, PreparedPythonEnvironment):
            raise TypeError("environment must be PreparedPythonEnvironment")
        if isinstance(args, (str, bytes, bytearray)) or any(type(arg) is not str for arg in args):
            raise TypeError("args must be a sequence of strings")
        if pre_spawn_guard is not None and not callable(pre_spawn_guard):
            raise TypeError("pre_spawn_guard must be callable or None")
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
        stdout_name = os.fspath(stdout_path)
        stderr_name = os.fspath(stderr_path)
        timed_out = False
        with Path(stdout_name).open("wb") as stdout, Path(stderr_name).open("wb") as stderr:
            if pre_spawn_guard is not None and pre_spawn_guard():
                return ProcessOutcome(logical_argv, None, stdout_name, stderr_name, timed_out)
            process = subprocess.Popen(
                command,
                cwd=None if cwd is None else os.fspath(cwd),
                env=child_env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                if process_handle is not None:
                    process_handle.attach(process)
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
                for cleanup_error in cleanup_errors:
                    original.add_note(f"cleanup wait interruption: {cleanup_error!r}")
                raise
            finally:
                if process_handle is not None:
                    process_handle.clear(process)
        return ProcessOutcome(logical_argv, returncode, stdout_name, stderr_name, timed_out)


__all__ = ["ProcessOutcome", "SubprocessWorker"]
