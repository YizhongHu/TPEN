"""Parsl attached dispatch behind the backend-neutral executor contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from pathlib import Path
import threading
import time
from typing import Any

from ..environments import PreparedPythonEnvironment, SourceScript
from ..placement import BindingError, WorkerBinding, bindings_from_allocation, physical_slots_from_allocation, slots_from_allocation
from ..programs import WorkerProgram
from .._records import frozen_record
from ..results import RunManifest
from ..contracts import ReadyGroup
from ..lifecycle.attached import AttachedAllocation
from ..tasks import ConfiguredRun
from ..workers import ProcessOutcome, SubprocessWorker
from .contract import (
    DispatcherContract,
    _member_paths,
    _placement_refusal_from_environment,
    _with_child_context_keys,
    _with_child_placement,
)


class ParslUnavailableError(RuntimeError):
    """Raised when the optional Parsl extra is absent."""


class UnsupportedParslExecutorError(RuntimeError):
    """Raised when a supplied Parsl config is not ThreadPool-only."""


def _poll_sleep(seconds: float) -> None:
    """Yield between controller future polls."""

    time.sleep(seconds)


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

    def pre_spawn_guard(self) -> bool:
        """Return whether owner cancellation was requested before spawn."""

        with self._lock:
            return self._kill_requested

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


@frozen_record
class _ChildFailure:
    """Carry a worker-thread BaseException through Parsl's future."""

    error: BaseException


def _run_parsl_child_captured(*args: Any, **kwargs: Any) -> ProcessOutcome | _ChildFailure:
    """Run the child app while keeping BaseException observable to the owner."""

    try:
        return _run_parsl_child(*args, **kwargs)
    except BaseException as error:
        return _ChildFailure(error)


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
        pre_spawn_guard=None if process_handle is None else process_handle.pre_spawn_guard,
    )


@frozen_record
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

        token = self._begin_dispatch(group, runs, lifecycle, env, timeout)
        dfk: Any = None
        original: BaseException | None = None
        try:
            bindings = self._resolve_bindings(lifecycle)
            config = self.executor_config
            if config is None:
                run_dir = str(Path(lifecycle.store.root) / "parsl")
                config = self._config_type(
                    executors=[self._thread_pool_type(max_threads=len(bindings))],
                    run_dir=run_dir,
                    usage_tracking=False,
                )
            labels = self._validate_executor_config(config, len(bindings))
            if len(bindings) == 1:
                object.__setattr__(self, "_active_binding", bindings[0])
            object.__setattr__(self, "_active_executor_labels", labels)
            dfk = self._dfk_type(config=config)
            object.__setattr__(self, "_active_dfk", dfk)
            try:
                if len(bindings) == 1:
                    return super(ParslDispatcher, self).run(group, runs, lifecycle, env, timeout, _admitted=True)
                return self._run_slot_pool(group, runs, lifecycle, env, timeout, bindings)
            except BaseException as error:
                original = error
                raise
        finally:
            cleanup_error: BaseException | None = None
            try:
                if dfk is not None:
                    dfk.cleanup()
            except BaseException as error:
                cleanup_error = error
            finally:
                object.__setattr__(self, "_active_dfk", None)
                object.__setattr__(self, "_active_binding", None)
                object.__setattr__(self, "_active_executor_labels", ("threads",))
                self._end_dispatch(token)
            if cleanup_error is not None:
                if original is not None:
                    original.add_note(f"Parsl DFK cleanup error: {cleanup_error!r}")
                else:
                    raise cleanup_error

    def validate(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        """Perform common admission plus strict attached-slot admission."""

        super(ParslDispatcher, self).validate(group, runs, lifecycle, env, timeout)
        self._resolve_bindings(lifecycle)

    def _resolve_bindings(self, lifecycle: AttachedAllocation) -> tuple[WorkerBinding, ...]:
        """Resolve immutable per-slot bindings before any lifecycle write."""

        if not isinstance(lifecycle, AttachedAllocation):
            if self.binding is None:
                raise BindingError("binding is required when lifecycle is not AttachedAllocation")
            return (WorkerBinding.from_mapping(self.binding),)
        slots = physical_slots_from_allocation(lifecycle.allocation)
        if slots:
            if self.binding is not None:
                raise BindingError("explicit binding is ambiguous when allocation slots are declared")
            return tuple(slot.binding() for slot in slots)
        if self.binding is not None:
            return (WorkerBinding.from_mapping(self.binding),)
        return bindings_from_allocation(lifecycle.allocation)

    def _run_slot_pool(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None,
        timeout: float | None,
        bindings: tuple[WorkerBinding, ...],
    ) -> RunManifest:
        """Run members through a bounded controller-owned slot pool."""

        queued = iter(group.members)
        available = list(bindings)
        active: dict[Any, dict[str, Any]] = {}
        begin_error: Exception | None = None
        finish_error: Exception | None = None

        def launch(member: Any, binding: WorkerBinding) -> None:
            nonlocal begin_error, finish_error
            configured = runs[member.task_id]
            try:
                fence = lifecycle.begin(group, member.task_id, self._utc_now())
            except Exception as error:
                if begin_error is None:
                    begin_error = error
                available.append(binding)
                return
            try:
                materialized = self.worker.materialize(env)
                environment = self._worker_environment(materialized, binding)
                environment = _with_child_context_keys(environment, lifecycle.allocation)
                environment = _with_child_placement(
                    environment,
                    lifecycle.allocation,
                    fence,
                    group,
                    member,
                    configured,
                )
                placement_refusal = _placement_refusal_from_environment(environment)
                if placement_refusal is not None:
                    completion_error = self._complete_member(
                        group,
                        member,
                        configured,
                        lifecycle,
                        fence,
                        materialized,
                        environment,
                        None,
                        timeout,
                        error=placement_refusal,
                    )
                    if completion_error is not None and finish_error is None:
                        finish_error = completion_error
                    available.append(binding)
                    return
                process = materialized["process"]
                materialized["runner"]
                evidence = materialized["evidence"]
                if not callable(getattr(process, "run", None)):
                    raise TypeError("materialized worker process lacks run()")
                if not callable(getattr(evidence, "read", None)):
                    raise TypeError("materialized worker evidence reader lacks read()")
                output_dir, _core_dir, stdout_path, stderr_path, _evidence_stdout, _evidence_stderr = _member_paths(fence)
                args = (*configured.command().args, f"run.root={output_dir}", "run.layout=flat", "run.run_id=core")
                future, handle = self._submit_process(
                    materialized,
                    environment,
                    args,
                    cwd=output_dir,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    timeout=timeout,
                )
            except Exception:
                try:
                    lifecycle.finish(fence, "launch_failed", self._utc_now(), None)
                except BaseException as error:
                    if finish_error is None and isinstance(error, Exception):
                        finish_error = error
                available.append(binding)
                return
            except BaseException:
                try:
                    lifecycle.finish(fence, "failed", self._utc_now(), None)
                except BaseException as error:
                    if finish_error is None and isinstance(error, Exception):
                        finish_error = error
                available.append(binding)
                raise
            active[future] = {"member": member, "configured": configured, "fence": fence, "materialized": materialized, "environment": environment, "binding": binding, "handle": handle}

        def fill() -> None:
            while available:
                try:
                    member = next(queued)
                except StopIteration:
                    return
                binding = available.pop(0)
                launch(member, binding)

        try:
            fill()
            while active:
                completed = next((future for future in active if future.done()), None)
                if completed is None:
                    _poll_sleep(0.01)
                    continue
                record = active.pop(completed)
                binding = record["binding"]
                completion_started = False
                try:
                    result = completed.result(None)
                    if isinstance(result, _ChildFailure):
                        raise result.error
                    if not isinstance(result, ProcessOutcome):
                        raise TypeError("Parsl child returned an invalid ProcessOutcome")
                    completion_started = True
                    completion_error = self._complete_member(
                        group,
                        record["member"],
                        record["configured"],
                        lifecycle,
                        record["fence"],
                        record["materialized"],
                        record["environment"],
                        result,
                        timeout,
                    )
                    if completion_error is not None and finish_error is None:
                        finish_error = completion_error
                except Exception as error:
                    if completion_started:
                        if finish_error is None:
                            finish_error = error
                    else:
                        completion_error = self._complete_member(
                            group,
                            record["member"],
                            record["configured"],
                            lifecycle,
                            record["fence"],
                            record["materialized"],
                            record["environment"],
                            None,
                            timeout,
                            error=error,
                        )
                        if completion_error is not None and finish_error is None:
                            finish_error = completion_error
                except BaseException as cancellation:
                    if not completion_started:
                        completion_error = self._complete_member(
                            group,
                            record["member"],
                            record["configured"],
                            lifecycle,
                            record["fence"],
                            record["materialized"],
                            record["environment"],
                            None,
                            timeout,
                            error=cancellation,
                        )
                        if completion_error is not None:
                            try:
                                cancellation.add_note(f"slot {record['member'].task_id} finish unresolved: {completion_error!r}")
                            except BaseException:
                                pass
                    raise
                available.append(binding)
                fill()
        except BaseException as cancellation:
            self._cancel_slot_pool(active, lifecycle, cancellation)
            raise
        if begin_error is not None:
            if finish_error is not None:
                begin_error.add_note(f"finish error after deferred begin failure: {finish_error!r}")
            raise begin_error
        if finish_error is not None:
            raise finish_error
        return lifecycle.store.manifest(group)

    @staticmethod
    def _utc_now() -> str:
        """Return a UTC timestamp without coupling slot control to storage."""

        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def _cancel_slot_pool(self, active: Mapping[Any, Mapping[str, Any]], lifecycle: AttachedAllocation, cancellation: BaseException) -> None:
        """Kill and reap every started child before failed cancellation finishes."""

        for record in active.values():
            record["handle"].request_kill()
        for future, record in active.items():
            reap_errors: list[BaseException] = []
            while True:
                try:
                    if future.done():
                        break
                    _poll_sleep(0.01)
                except BaseException as error:
                    reap_errors.append(error)
            try:
                try:
                    future.result(None)
                except BaseException as error:
                    reap_errors.append(error)
            except BaseException as error:
                reap_errors.append(error)
            for error in reap_errors:
                try:
                    cancellation.add_note(f"slot {record['member'].task_id} reap result: {error!r}")
                except BaseException:
                    pass
            try:
                lifecycle.finish(record["fence"], "failed", self._utc_now(), None)
            except BaseException as error:
                try:
                    cancellation.add_note(f"slot {record['member'].task_id} finish unresolved: {error!r}")
                except BaseException:
                    pass

    def _validate_executor_config(self, config: Any, required_threads: int | None = None) -> tuple[str, ...]:
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
        if required_threads is not None:
            capacity = sum(
                threads
                for executor in executors
                if type(threads := getattr(executor, "max_threads", None)) is int and threads > 0
            )
            if capacity < required_threads:
                raise UnsupportedParslExecutorError(
                    "Parsl ThreadPool executor capacity is smaller than the admitted slot pool"
                )
        return labels

    def _worker_environment(self, materialized: Any, binding: WorkerBinding | None = None) -> PreparedPythonEnvironment:
        """Overlay the explicit worker binding on the prepared environment."""

        environment = materialized["environment"]
        if not isinstance(environment, PreparedPythonEnvironment):
            raise TypeError("Parsl worker environment must be PreparedPythonEnvironment")
        binding = self._active_binding if binding is None else binding
        if binding is None:
            raise BindingError("Parsl worker binding was not resolved before dispatch")
        return PreparedPythonEnvironment(
            environment.python,
            environment.setup,
            {**environment.env, **binding.env},
        )

    def _submit_process(
        self,
        materialized: Any,
        environment: PreparedPythonEnvironment,
        args: tuple[str, ...],
        *,
        cwd: str,
        stdout_path: str,
        stderr_path: str,
        timeout: float | None,
    ) -> tuple[Any, _ChildProcessHandle]:
        """Submit one child app without waiting for its future."""

        dfk = self._active_dfk
        if dfk is None:
            raise RuntimeError("Parsl DFK is not active")
        process_handle = _ChildProcessHandle()
        app = self._python_app_factory(
            data_flow_kernel=dfk,
            cache=False,
            executors=list(self._active_executor_labels),
        )(_run_parsl_child_captured)
        try:
            future = app(
                environment.python,
                None if environment.setup is None else environment.setup.path,
                dict(environment.env),
                args,
                cwd=cwd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout=timeout,
                process_handle=process_handle,
            )
        except BaseException:
            # The app may be interrupted before it returns its future.  Mark
            # the handle first so a racing worker spawn is killed on attach.
            process_handle.request_kill()
            raise
        return future, process_handle

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
        app = self._python_app_factory(
            data_flow_kernel=dfk,
            cache=False,
            executors=list(self._active_executor_labels),
        )(_run_parsl_child_captured)
        try:
            future = app(*app_args, **app_kwargs)
        except BaseException:
            # The app may be interrupted before it returns its future.  Mark
            # the handle first so a racing worker spawn is killed on attach.
            process_handle.request_kill()
            raise
        try:
            result = future.result(None)
        except BaseException as original:
            process_handle.request_kill()
            reap_errors: list[BaseException] = []
            while not future.done():
                try:
                    future.result(None)
                except BaseException as reap_error:
                    reap_errors.append(reap_error)
            for reap_error in reap_errors:
                original.add_note(f"Parsl child reap result: {reap_error!r}")
            raise
        if not isinstance(result, ProcessOutcome):
            if isinstance(result, _ChildFailure):
                raise result.error
            raise TypeError("Parsl child returned an invalid ProcessOutcome")
        return result


__all__ = ["ParslDispatcher", "ParslUnavailableError", "UnsupportedParslExecutorError"]
