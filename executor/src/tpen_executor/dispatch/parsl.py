"""Parsl attached dispatch behind the backend-neutral executor contract."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
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
    _active_binding: WorkerBinding | None = field(init=False, default=None, repr=False, compare=False)
    _active_dfk: Any = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.worker, "materialize", None)):
            raise TypeError("worker must be a materializable WorkerProgram handle")
        try:
            from parsl.config import Config
            from parsl.dataflow.dflow import DataFlowKernel
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
        """Run one attached allocation with one fresh, cleaned-up DFK."""

        binding = self.binding
        if binding is None:
            if not isinstance(lifecycle, AttachedAllocation):
                raise BindingError("binding is required when lifecycle is not AttachedAllocation")
            binding = binding_from_allocation(lifecycle.allocation)
        else:
            binding = WorkerBinding.from_mapping(binding)
        object.__setattr__(self, "_active_binding", binding)
        config = self.executor_config
        if config is None:
            run_dir = str(Path(lifecycle.store.root) / "parsl")
            config = self._config_type(
                executors=[self._thread_pool_type(max_threads=1)],
                run_dir=run_dir,
                usage_tracking=False,
            )
        dfk = self._dfk_type(config=config)
        object.__setattr__(self, "_active_dfk", dfk)
        try:
            return super().run(group, runs, lifecycle, env, timeout)
        except BaseException as original:
            try:
                dfk.cleanup()
            except BaseException as cleanup_error:
                original.add_note(f"Parsl DFK cleanup error: {cleanup_error!r}")
            raise
        else:
            dfk.cleanup()
        finally:
            object.__setattr__(self, "_active_dfk", None)
            object.__setattr__(self, "_active_binding", None)

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
        future = dfk.submit(
            _run_parsl_child,
            environment.python,
            None if environment.setup is None else environment.setup.path,
            dict(environment.env),
            args,
            cwd=cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=timeout,
        )
        try:
            result = future.result(None if timeout is None else timeout + 5.0)
        except FutureTimeoutError:
            future.cancel()
            return ProcessOutcome(args, None, stdout_path, stderr_path, True)
        if not isinstance(result, ProcessOutcome):
            raise TypeError("Parsl child returned an invalid ProcessOutcome")
        return result


__all__ = ["ParslDispatcher", "ParslUnavailableError"]
