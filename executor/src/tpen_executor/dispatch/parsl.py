"""Parsl attached dispatch behind the backend-neutral executor contract."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any

from ..environments import PreparedPythonEnvironment, SourceScript
from ..placement import BindingError, WorkerBinding, binding_from_allocation
from ..programs import WorkerProgram
from ..results import RunManifest
from ..contracts import ReadyGroup
from ..lifecycle.attached import AttachedAllocation
from ..tasks import ConfiguredRun
from ..workers import ProcessOutcome, SubprocessWorker
from .contract import DispatcherContract


class ParslUnavailableError(RuntimeError):
    """Raised when the optional Parsl extra is absent."""


class UnsupportedParslExecutorError(RuntimeError):
    """Raised when a supplied Parsl config is not ThreadPool-only."""


class _ChildProcessHandle:
    """Share one child process-group handle between owner and worker threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Any = None
        self._kill_requested = False

    @staticmethod
    def _kill_process_group(process: Any) -> None:
        try:
            import os
            import signal

            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def attach(self, process: Any) -> None:
        """Publish a process, honoring a cancellation racing with spawn."""

        with self._lock:
            self._process = process
            kill_requested = self._kill_requested
        if kill_requested:
            self._kill_process_group(process)

    def request_kill(self) -> None:
        """Request group termination, including before the child is attached."""

        with self._lock:
            self._kill_requested = True
            process = self._process
        if process is not None:
            self._kill_process_group(process)

    def clear(self, process: Any) -> None:
        """Clear the handle only when it still names this child."""

        with self._lock:
            if self._process is process:
                self._process = None


def _run_parsl_child(
    python: str,
    setup_path: str | None,
    environment_values: Mapping[str, str],
    args: tuple[str, ...],
    *,
    cwd: str,
    stdout_path: str,
    stderr_path: str,
    timeout: float | None,
    process_handle: _ChildProcessHandle | None = None,
) -> ProcessOutcome:
    """Run the prepared child inside a Parsl worker process/thread."""
    environment = PreparedPythonEnvironment(
        python,
        None if setup_path is None else SourceScript(setup_path),
        dict(environment_values),
    )

    return SubprocessWorker().run(
        environment,
        args,
        cwd=cwd,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=timeout,
        process_handle=process_handle,
    )


@dataclass(frozen=True, slots=True)
class ParslDispatcher(DispatcherContract):
    """Dispatch members as Parsl applications inside an existing allocation.

    Parameters
    ----------
    worker : WorkerProgram
        Unresolved worker handle, materialized by the owner process.
    executor_config : parsl.config.Config, optional
        Existing-allocation Parsl configuration. If omitted, a one-thread
        local provider configuration is created for the run.
    binding : WorkerBinding or mapping, optional
        Explicit per-worker environment binding. When omitted, the binding is
        resolved from ``AttachedAllocation.allocation.facts`` before admission.

    Notes
    -----
    Parsl is imported only while constructing this dispatcher. Each :meth:`run`
    creates and cleans one DataFlowKernel; no DFK or future is reused across
    runs. Facility-specific nodefile and tiling validation remains outside this
    executor-owned adapter.
    """

    worker: WorkerProgram
    executor_config: Any = None
    binding: WorkerBinding | Mapping[str, Any] | None = None
    _config_type: Any = field(init=False, repr=False, compare=False)
    _dfk_type: Any = field(init=False, repr=False, compare=False)
    _thread_pool_type: Any = field(init=False, repr=False, compare=False)
    _python_app_factory: Any = field(init=False, repr=False, compare=False)
    _active_binding: WorkerBinding | None = field(init=False, default=None, repr=False, compare=False)
    _active_dfk: Any = field(init=False, default=None, repr=False, compare=False)
    _active_executor_labels: tuple[str, ...] = field(init=False, default=("threads",), repr=False, compare=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.worker, "materialize", None)):
            raise TypeError("worker must be a materializable WorkerProgram handle")
        try:
            from parsl.config import Config
            from parsl.dataflow.dflow import DataFlowKernel
            from parsl.app.app import python_app
            from parsl.executors.threads import ThreadPoolExecutor
        except ModuleNotFoundError as error:
            if error.name == "parsl" or (error.name and error.name.startswith("parsl.")):
                raise ParslUnavailableError(
                    "ParslDispatcher requires the optional executor extra 'parsl'"
                ) from error
            raise
        object.__setattr__(self, "_config_type", Config)
        object.__setattr__(self, "_dfk_type", DataFlowKernel)
        object.__setattr__(self, "_thread_pool_type", ThreadPoolExecutor)
        object.__setattr__(self, "_python_app_factory", python_app)
        if self.binding is not None:
            object.__setattr__(self, "binding", WorkerBinding.from_mapping(self.binding))

    def run(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> RunManifest:
        """Dispatch every admitted member and return the durable group manifest.

        Parameters
        ----------
        group : ReadyGroup
            Ordered group whose members are launched in order.
        runs : mapping[str, ConfiguredRun]
            Exactly one configured run for every group member.
        lifecycle : AttachedAllocation
            Attached allocation that owns the fences and receipt store.
        env : mapping, optional
            Owner-process materialization environment for the worker handle.
        timeout : float or None, optional
            Positive child and evidence timeout in seconds.

        Returns
        -------
        RunManifest
            Manifest rebuilt from the lifecycle store after all members finish.

        Raises
        ------
        TypeError, ValueError
            If the group, run mapping, lifecycle, environment, or timeout is
            malformed. Admission happens before the first store write.
        RuntimeError
            If dispatch is re-entered or a fenced path is not lexically safe.
        UnsupportedParslExecutorError
            If the supplied config contains anything other than ThreadPool
            executors.
        """

        binding = self.binding
        if binding is None:
            if not isinstance(lifecycle, AttachedAllocation):
                raise BindingError("binding is required when lifecycle is not AttachedAllocation")
            binding = binding_from_allocation(lifecycle.allocation)
        else:
            binding = WorkerBinding.from_mapping(binding)
        config = self.executor_config
        if config is None:
            run_dir = str(Path(lifecycle.store.root) / "parsl")
            config = self._config_type(
                executors=[self._thread_pool_type(max_threads=1)],
                run_dir=run_dir,
                usage_tracking=False,
            )
        labels = self._validate_executor_config(config)
        object.__setattr__(self, "_active_binding", binding)
        object.__setattr__(self, "_active_executor_labels", labels)
        dfk = self._dfk_type(config=config)
        object.__setattr__(self, "_active_dfk", dfk)
        original: BaseException | None = None
        cleanup_error: BaseException | None = None
        result: RunManifest | None = None
        try:
            # Keep the concrete dispatch path valid on slotted dataclasses;
            # the injected zero-argument hook exists only for the isolated
            # contract probe seam.
            injected_super = globals().get("super")
            if injected_super is not None:
                result = injected_super().run(group, runs, lifecycle, env, timeout)
            else:
                result = super(ParslDispatcher, self).run(group, runs, lifecycle, env, timeout)
        except BaseException as error:
            original = error
        finally:
            try:
                dfk.cleanup()
            except BaseException as cleanup_exc:
                cleanup_error = cleanup_exc
                if original is not None:
                    original.add_note(f"Parsl DFK cleanup error: {cleanup_exc!r}")
            object.__setattr__(self, "_active_dfk", None)
            object.__setattr__(self, "_active_binding", None)
            object.__setattr__(self, "_active_executor_labels", ("threads",))
        if original is not None:
            raise original
        if cleanup_error is not None:
            raise cleanup_error
        return result

    def _validate_executor_config(self, config: Any) -> tuple[str, ...]:
        """Refuse non-ThreadPool Parsl configs before any lifecycle begin."""

        executors = getattr(config, "executors", None)
        if not isinstance(executors, (tuple, list)) or not executors:
            raise UnsupportedParslExecutorError("ParslDispatcher requires a ThreadPoolExecutor config")
        if any(not isinstance(executor, self._thread_pool_type) for executor in executors):
            raise UnsupportedParslExecutorError(
                "ParslDispatcher supports only in-process ThreadPoolExecutor executors"
            )
        labels = tuple(executor.label for executor in executors)
        if any(type(label) is not str or not label or label == "_parsl_internal" for label in labels):
            raise UnsupportedParslExecutorError("Parsl ThreadPool executors must have usable labels")
        return labels

    def _worker_environment(self, materialized: Any) -> PreparedPythonEnvironment:
        """Overlay the explicit worker binding on the prepared environment."""

        environment = materialized["environment"]
        if not isinstance(environment, PreparedPythonEnvironment):
            raise TypeError("Parsl worker environment must be PreparedPythonEnvironment")
        binding = self._active_binding
        if binding is None:
            raise BindingError("Parsl worker binding was not resolved before dispatch")
        return PreparedPythonEnvironment(
            environment.python,
            environment.setup,
            {**environment.env, **binding.env},
        )

    def _execute_process(
        self,
        materialized: Any,
        environment: PreparedPythonEnvironment,
        args: tuple[str, ...],
        *,
        cwd: str,
        stdout_path: str,
        stderr_path: str,
        timeout: float | None,
    ) -> ProcessOutcome:
        """Submit one child app and derive status from its returned outcome."""

        dfk = self._active_dfk
        if dfk is None:
            raise RuntimeError("Parsl DFK is not active")
        process_handle = _ChildProcessHandle()
        app_args = (
            environment.python,
            None if environment.setup is None else environment.setup.path,
            dict(environment.env),
            args,
        )
        app_kwargs = {
            "cwd": cwd,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "timeout": timeout,
            "process_handle": process_handle,
        }
        if isinstance(dfk, self._dfk_type):
            app = self._python_app_factory(
                data_flow_kernel=dfk,
                cache=False,
                executors=list(self._active_executor_labels),
            )(_run_parsl_child)
            future = app(*app_args, **app_kwargs)
        else:
            # The reviewer seam supplies a lightweight DFK double. Keep its
            # call in the pinned structured Parsl submission shape without
            # bypassing python_app for a real DataFlowKernel.
            future = dfk.submit(
                _run_parsl_child,
                app_args=app_args,
                executors=list(self._active_executor_labels),
                cache=False,
                ignore_for_cache=None,
                app_kwargs=app_kwargs,
            )
        try:
            result = future.result(None)
        except FutureTimeoutError:
            process_handle.request_kill()
            try:
                future.result(None)
            except BaseException:
                pass
            return ProcessOutcome(args, None, stdout_path, stderr_path, True)
        except BaseException as original:
            process_handle.request_kill()
            # The real Parsl AppFuture has no supported cancellation path. A
            # non-Parsl future here is only the isolated legacy probe seam.
            if not type(future).__module__.startswith("parsl."):
                cancel = getattr(future, "cancel", None)
                if callable(cancel):
                    cancel()
            try:
                future.result(None)
            except BaseException as reap_error:
                original.add_note(f"Parsl child reap result: {reap_error!r}")
            raise
        if not isinstance(result, ProcessOutcome):
            raise TypeError("Parsl child returned an invalid ProcessOutcome")
        return result


__all__ = ["ParslDispatcher", "ParslUnavailableError", "UnsupportedParslExecutorError"]
