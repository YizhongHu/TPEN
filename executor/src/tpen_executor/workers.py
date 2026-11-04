"""Owner-process subprocess workers for prepared Python environments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

from .environments import PreparedPythonEnvironment


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    """Immutable outcome of one subprocess invocation."""

    argv: tuple[str, ...]
    returncode: int | None
    stdout_path: str
    stderr_path: str
    timed_out: bool


@dataclass(frozen=True, slots=True)
class SubprocessWorker:
    """Launch one prepared Python child without shell-interpolated arguments."""

    def run(
        self,
        environment: PreparedPythonEnvironment,
        args: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None,
        stdout_path: str | os.PathLike[str],
        stderr_path: str | os.PathLike[str],
        timeout: float | None = None,
    ) -> ProcessOutcome:
        """Run the prepared interpreter with exact argument boundaries."""

        if not isinstance(environment, PreparedPythonEnvironment):
            raise TypeError("environment must be PreparedPythonEnvironment")
        if isinstance(args, (str, bytes, bytearray)) or any(type(arg) is not str for arg in args):
            raise TypeError("args must be a sequence of strings")
        if timeout is not None and (type(timeout) not in (int, float) or timeout <= 0):
            raise ValueError("timeout must be positive or None")
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
            process = subprocess.Popen(
                command,
                cwd=None if cwd is None else os.fspath(cwd),
                env=child_env,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                returncode = process.wait()
        return ProcessOutcome(logical_argv, returncode, stdout_name, stderr_name, timed_out)


__all__ = ["ProcessOutcome", "SubprocessWorker"]
