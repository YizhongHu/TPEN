"""Parsl executor that attaches work to an existing allocation.

Parsl is deliberately imported only while dispatching.  The module therefore
remains importable in planning and unit-test environments that do not install
the optional ``parsl`` extra.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .dispatch import AllocationContext, DispatchRecord, DispatchSpec
from .task_state import _deadline_guard_reached


_HOSTNAME_PATTERN = re.compile(
    r"^(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)


def validate_accelerator_tiling(visibility_values: Sequence[str]) -> None:
    """Reject row bindings that do not partition a node's accelerators.

    Bindings must be disjoint and cover ``0..N-1``: overlap would double-book an
    accelerator, gaps would not describe a node. HOW MANY accelerators a node has
    is a planning setting owned by the caller
    (:data:`experiments.baselines.pipeline.ACCELERATORS_PER_NODE`), not something
    discovered here -- this only checks that what arrived is well formed.

    The replaced check required exactly four bindings, which conflated "four
    accelerators per node" with "four rows per node" and so rejected every row
    width above one.
    """

    if not visibility_values:
        raise ValueError("at least one accelerator binding is required; got none")
    flat: list[int] = []
    for value in visibility_values:
        tokens = [token.strip() for token in str(value).split(",") if token.strip()]
        if not tokens:
            raise ValueError(f"accelerator binding is empty: {value!r}")
        try:
            flat.extend(int(token) for token in tokens)
        except ValueError as exc:
            raise ValueError(f"accelerator binding must be integers: {value!r}") from exc
    if sorted(flat) != list(range(len(flat))):
        raise ValueError(
            "accelerator bindings must partition a node's local indices 0..N-1 "
            f"exactly once; got {sorted(flat)} from {tuple(visibility_values)}"
        )


def validate_pbs_nodefile(
    nodefile: str | os.PathLike[str] | None,
    *,
    requested_node_count: int,
) -> tuple[str, ...]:
    """Validate and canonicalize ``PBS_NODEFILE`` without importing Parsl.

    This is intentionally an independently callable preflight boundary: the
    executor invokes it before its first Parsl import, and tests exercise it
    directly in environments where the optional Parsl dependency is absent.

    Parameters
    ----------
    nodefile : path-like or None
        Path supplied by the scheduler.
    requested_node_count : int
        Number of distinct hosts requested by the allocation.

    Returns
    -------
    tuple of str
        Unique, lower-case hostnames in first-seen order.

    Raises
    ------
    RuntimeError
        If the nodefile is missing, unreadable, empty, malformed, or contains
        a host count different from the requested count.
    """

    if requested_node_count < 1:
        raise ValueError("requested node count must be positive")
    if nodefile is None:
        raise RuntimeError(
            "PBS_NODEFILE missing: actual host count 0, "
            f"expected {requested_node_count}"
        )
    path = Path(nodefile)
    if not path.exists():
        raise RuntimeError(
            f"PBS_NODEFILE missing at {path}: actual host count 0, "
            f"expected {requested_node_count}"
        )
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"PBS_NODEFILE unreadable at {path}: actual host count unavailable, "
            f"expected {requested_node_count} ({exc})"
        ) from exc

    hosts: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(contents.splitlines(), start=1):
        host = line.strip().lower()
        if not host:
            continue
        if not _HOSTNAME_PATTERN.fullmatch(host):
            raise RuntimeError(
                f"PBS_NODEFILE unknown hostname format on line {line_number}: {line.strip()!r}; "
                f"actual host count {len(hosts)}, expected {requested_node_count}"
            )
        if host not in seen:
            seen.add(host)
            hosts.append(host)

    if not hosts:
        raise RuntimeError(
            f"PBS_NODEFILE empty at {path}: actual host count 0, "
            f"expected {requested_node_count}"
        )
    if len(hosts) != requested_node_count:
        raise RuntimeError(
            f"PBS_NODEFILE host count mismatch at {path}: actual host count {len(hosts)}, "
            f"expected {requested_node_count}"
        )
    return tuple(hosts)


def _run_dispatch_payload(
    argv: tuple[str, ...],
    cwd: str,
    environment: Mapping[str, str],
    output_directory: str,
    attempt_id: str,
    visibility_variable: str | None = None,
    visibility_value: str | None = None,
) -> dict[str, Any]:
    """Run one immutable argv and write only stdlib worker evidence.

    This function is intentionally self-contained: Parsl serializes it by
    value, and a worker must never import this package to execute a row.
    """

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    stdout_path = output_path / "stdout.log"
    stderr_path = output_path / "stderr.log"
    status_path = output_path / "attempt_status.json"
    worker_visibility_value = (
        os.environ.get(visibility_variable) if visibility_variable is not None else None
    )
    worker_environment = os.environ | {str(key): str(value) for key, value in environment.items()}
    if visibility_variable is not None:
        if visibility_value is not None:
            worker_environment[visibility_variable] = visibility_value
        elif worker_visibility_value is None:
            worker_environment.pop(visibility_variable, None)
        else:
            # A tiled worker owns this value. Do not let the serialized
            # dispatcher environment replace it during payload execution.
            worker_environment[visibility_variable] = worker_visibility_value
    started_at = time.time()
    # Keep this worker payload stdlib-only. Parsl serializes it by value, and
    # importing the toolkit here would make worker deserialization depend on
    # the submission checkout being present on every worker's sys.path.
    import socket

    hostname = socket.gethostname().lower()
    placement: dict[str, Any] = {
        "attempt_id": attempt_id,
        "hostname": hostname,
        "fqdn": socket.getfqdn(),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "identity": os.environ.get("PARSL_WORKER_ID") or os.environ.get("PARSL_MANAGER_ID"),
        "registration_at_unix": started_at,
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cuda_visible_devices": worker_environment.get("CUDA_VISIBLE_DEVICES"),
        "cwd": str(Path(cwd).resolve()),
        "result_dir": str(output_path.resolve()),
        "started_at_unix": started_at,
        "gpus": [],
    }
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        query = None
    if query is not None and query.returncode == 0:
        for line in query.stdout.splitlines():
            name, uuid, bus_id = (part.strip() for part in line.split(",", 2))
            placement["gpus"].append({"name": name, "uuid": uuid, "pci_bus_id": bus_id})
    launch_error: str | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=worker_environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
            returncode: int | None = int(completed.returncode)
        except OSError as exc:
            launch_error = repr(exc)
            stderr.write(launch_error + "\n")
            returncode = None
    ended_at = time.time()
    status: dict[str, Any] = {
        "status": "success" if returncode == 0 else "failed",
        "attempt_id": attempt_id,
        "argv": list(argv),
        "cwd": cwd,
        "returncode": returncode,
        "visibility_variable": visibility_variable,
        "visibility_value": visibility_value,
        "inherited_visibility_value": (
            worker_environment.get(visibility_variable) if visibility_variable is not None else None
        ),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_at_unix": started_at,
        "ended_at_unix": ended_at,
        "elapsed_sec": ended_at - started_at,
        "placement": {**placement, "ended_at_unix": ended_at, "returncode": returncode},
    }
    if launch_error is not None:
        status["error"] = launch_error
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return {**status, "attempt_status_path": str(status_path)}


AppRunner = Callable[..., Any]


@dataclass(frozen=True)
class ParslAttachExecutor:
    """Dispatch a ready batch concurrently through Parsl HTEX.

    ``app_runner`` is an intentional unit-test seam.  It receives the same
    keyword arguments as the Parsl application and may return either a payload
    mapping or an object with a blocking ``result()`` method.
    """

    app_runner: AppRunner | None = None
    _runner: AppRunner | None = field(default=None, init=False, repr=False, compare=False)
    _context_key: str | None = field(default=None, init=False, repr=False, compare=False)
    _runner_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    def dispatch(
        self,
        dispatches: Sequence[DispatchSpec],
        *,
        context: AllocationContext,
    ) -> tuple[DispatchRecord, ...]:
        """Run the ready batch and return its acceptance records.

        The caller owns ordering and retries.  This method only starts the
        supplied batch, waits for it, and verifies its declared completion.
        """

        context.validate()
        ready_dispatches = tuple(dispatch.validate() for dispatch in dispatches)
        if not ready_dispatches:
            return ()
        deadline_unix = context.deadline_unix()
        if _deadline_guard_reached(deadline_unix, context.deadline_guard_min):
            raise RuntimeError("allocation deadline guard reached; refusing new Parsl dispatches")
        if context.run_root is None:
            raise ValueError("ParslAttachExecutor requires context.run_root as the launch attempt directory")

        launch_attempt_dir = Path(context.run_root)
        tiled = bool(context.visibility_values) or context.nodes_per_block is not None
        if tiled:
            visibility_variable = context.visibility_variable
            if visibility_variable in context.environment:
                raise ValueError(
                    "TILED visibility override from context.environment is not allowed for "
                    f"key {visibility_variable!r}"
                )
            for dispatch in ready_dispatches:
                if visibility_variable in dispatch.environment:
                    raise ValueError(
                        "TILED visibility override from dispatch.environment is not allowed for "
                        f"key {visibility_variable!r}"
                    )
        runner = self.app_runner or self._real_runner(context, launch_attempt_dir)
        submitted: list[tuple[DispatchSpec, Any]] = []
        for dispatch in ready_dispatches:
            output_directory = launch_attempt_dir / "dispatch" / dispatch.attempt_id
            environment = os.environ | {**context.environment, **dispatch.environment}
            if tiled:
                # available_accelerators binds each HTEX worker. The task must
                # inherit that worker-owned environment; dispatcher visibility
                # is not a resource identity.
                environment.pop(context.visibility_variable, None)
            runner_kwargs: dict[str, Any] = {
                "argv": dispatch.argv,
                "cwd": dispatch.cwd,
                "environment": environment,
                "output_directory": str(output_directory),
                "attempt_id": dispatch.attempt_id,
            }
            if tiled:
                runner_kwargs["visibility_variable"] = context.visibility_variable
            submitted.append(
                (
                    dispatch,
                    runner(**runner_kwargs),
                )
            )

        records: list[DispatchRecord] = []
        for dispatch, future in submitted:
            payload = _result_payload(future)
            _check_completed_dispatch(dispatch, payload)
            records.append(
                DispatchRecord.accepted(
                    dispatch,
                    backend="parsl_attach",
                    launcher_job_id=context.allocation_id,
                    submitted_command=dispatch.argv,
                    metadata={"attempt_status_path": payload.get("attempt_status_path")},
                )
            )
        return tuple(records)

    def _real_runner(self, context: AllocationContext, launch_attempt_dir: Path) -> AppRunner:
        """Load the process-global Parsl DFK once and reuse its application."""

        context_key = json.dumps(context.to_dict(), sort_keys=True, separators=(",", ":"))
        with self._runner_lock:
            if self._runner is None:
                runner = _parsl_app_runner(context, launch_attempt_dir)
                object.__setattr__(self, "_runner", runner)
                object.__setattr__(self, "_context_key", context_key)
                atexit.register(self.close)
                return runner
            if self._context_key != context_key:
                raise RuntimeError(
                    "ParslAttachExecutor cannot reuse its loaded DFK with a different "
                    "AllocationContext"
                )
            return self._runner

    def close(self) -> None:
        """Clean up the DFK loaded by this executor and clear Parsl global state."""

        with self._runner_lock:
            if self._runner is None:
                return
            import parsl

            parsl.dfk().cleanup()
            parsl.clear()
            object.__setattr__(self, "_runner", None)
            object.__setattr__(self, "_context_key", None)


def _parsl_app_runner(context: AllocationContext, launch_attempt_dir: Path) -> AppRunner:
    """Build and load the allocation-local Parsl application lazily."""

    if context.nodes_per_block is not None:
        validate_accelerator_tiling(context.visibility_values)
        validate_pbs_nodefile(
            os.environ.get("PBS_NODEFILE"), requested_node_count=context.nodes_per_block
        )

    import parsl
    from parsl.app.app import python_app
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    provider_options: dict[str, Any] = {
        "init_blocks": 1,
        "min_blocks": 1,
        "max_blocks": 1,
    }
    if context.nodes_per_block is not None:
        from parsl.launchers import MpiExecLauncher

        provider_options.update(
            nodes_per_block=context.nodes_per_block,
            launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"),
            worker_init="export TMPDIR=/tmp",
        )

    executor_options: dict[str, Any] = {
        "label": "parsl-attach",
        "provider": LocalProvider(**provider_options),
        "max_workers_per_node": len(context.visibility_values) or 1,
    }
    if context.visibility_values:
        executor_options["available_accelerators"] = context.visibility_values
    config = Config(
        executors=[HighThroughputExecutor(**executor_options)],
        retries=0,
        run_dir=str(launch_attempt_dir / "parsl"),
        usage_tracking=False,
    )
    parsl.load(config)
    # ``dill`` serializes module-level functions by reference.  Construct a
    # by-value function with only stdlib globals so workers never need the
    # submission checkout to import ``experiments`` while deserializing it.
    import types

    worker_payload = types.FunctionType(
        _run_dispatch_payload.__code__,
        {
            "__builtins__": __builtins__,
            "__name__": "__parsl_worker_payload__",
            "Path": Path,
            "json": json,
            "os": os,
            "subprocess": subprocess,
            "time": time,
        },
        _run_dispatch_payload.__name__,
        _run_dispatch_payload.__defaults__,
    )
    return python_app(worker_payload)


def _result_payload(value: Any) -> Mapping[str, Any]:
    """Resolve a Parsl future or injectable test result into a mapping."""

    payload = value.result() if hasattr(value, "result") else value
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Parsl app returned non-mapping payload: {payload!r}")
    return payload


def _check_completed_dispatch(dispatch: DispatchSpec, payload: Mapping[str, Any]) -> None:
    """Mirror the legacy post-subprocess completion predicate check."""

    returncode = payload.get("returncode")
    if returncode != 0:
        raise RuntimeError(f"dispatch {dispatch.attempt_id!r} exited with returncode {returncode!r}")
    if dispatch.completion.policy != "none" and not dispatch.completion.is_complete():
        message = (
            f"command exited 0 but completion predicate {dispatch.completion.policy!r} "
            f"was not satisfied for dispatch {dispatch.attempt_id!r}: "
            f"{dispatch.completion.to_dict()!r}"
        )
        status_path = payload.get("attempt_status_path")
        if status_path:
            path = Path(str(status_path))
            status = json.loads(path.read_text(encoding="utf-8"))
            status["status"] = "failed"
            status["completion_error"] = message
            path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(message)
