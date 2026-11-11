"""Parsl attached dispatch behind the backend-neutral executor contract.

Two routes exist.  The default route runs each member's child in an
in-process Parsl ThreadPool, exactly as before S3.  The attached remote route
(S3 Layer B) is selected only by an explicit :class:`AttachedTransportSpec`:
each run builds one fresh HighThroughputExecutor with a LocalProvider and an
MpiExecLauncher inside the already-held allocation, preflights the shared
attempt store on every worker, and submits the S3 Layer A leaf
(:func:`tpen_executor.dispatch.transport.run_leaf`) as canonical request
bytes.  The controller remains the only lifecycle, store and manifest writer.

Contract mutant controls M1-M5 are listed in
``executor/tests/test_s3_transport_parsl.py`` (``S3B_MUTANT_TABLE``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field, fields
import fcntl
import math
import os
from pathlib import Path
import re
import shlex
import socket
import sys
import threading
import time
from typing import Any, ClassVar
import uuid

from ..environments import PreparedPythonEnvironment, SourceScript
from ..placement import BindingError, PlacementRefusal, WorkerBinding, bindings_from_allocation, physical_slots_from_allocation, slots_from_allocation
from ..programs import WorkerProgram
from .._records import frozen_record
from ..results import RunManifest
from ..contracts import ReadyGroup
from ..lifecycle.attached import AttachedAllocation
from ..runners import CoreRunEvidence, EvidenceUnavailable, TPENCommand, TPENEvidence
from ..tasks import ConfiguredRun
from ..workers import ProcessOutcome, SubprocessWorker
from .contract import (
    DispatcherContract,
    _member_paths,
    _admit_slot,
    _placement_refusal_from_environment,
    _with_child_context_keys,
    _with_child_placement,
    _with_placement_refusal,
)
from .transport import (
    WORKER_AFFINITY_REFUSAL_CODES,
    LeafPlacementRefused,
    LeafRequest,
    LeafResult,
    run_leaf,
    validate_leaf_result,
)


class ParslUnavailableError(RuntimeError):
    """Raised when the optional Parsl extra is absent."""


class UnsupportedParslExecutorError(RuntimeError):
    """Raised when a supplied Parsl config is not ThreadPool-only."""


class UnsupportedTransportError(RuntimeError):
    """Raised when an attached remote transport cannot be admitted.

    Every refusal of this type happens before the first lifecycle write, so
    no attempt exists for the refused run.
    """


class SharedStorePreflightError(UnsupportedTransportError):
    """Raised when the attempt store is not one shared, lockable path."""


class RemoteLeafLaunchError(RuntimeError):
    """A remote leaf returned a launch error instead of a science outcome."""


class RemoteInspectionError(RuntimeError):
    """A remote leaf reported failed or missing inspection for its science."""


ATTACHED_TRANSPORT_SCHEMA = "tpen-executor/attached-transport-v1"
# The single reviewed attached form.  Anything else is refused at admission.
_ATTACHED_FORM = {"executor": "htex", "provider": "local", "launcher": "mpiexec"}
_LABEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
_PREFLIGHT_PREFIX = "store-preflight-"
# HTEX starts its interchange and worker pool as console scripts found on
# PATH by default.  The attached route runs both modules with the current
# interpreter instead, so neither depends on PATH.
_INTERCHANGE_MODULE = "parsl.executors.high_throughput.interchange"
_WORKER_POOL_MODULE = "parsl.executors.high_throughput.process_worker_pool"
_WORKER_POOL_SCRIPT = "process_worker_pool.py "


@frozen_record
class AttachedTransportSpec:
    """Typed selection of the attached HTEX/LocalProvider/MpiExec transport.

    The spec carries only immutable, generic values.  Facility values (worker
    environment setup, interchange address, launcher overrides) are injected
    by an operator-owned profile; the executor supplies no facility default.
    No field acquires, replaces or retries an allocation: the LocalProvider
    starts exactly one block of workers inside the allocation that is already
    held by the controller.

    Parameters
    ----------
    executor, provider, launcher : str
        The attached form; only ``"htex"``, ``"local"`` and ``"mpiexec"``
        are admitted.
    label : str
        Parsl executor label that every submitted app names.
    nodes_per_block : int
        Number of allocation nodes the single block spans.
    max_workers_per_node : int
        HighThroughputExecutor workers per node.  The product with
        ``nodes_per_block`` is the exact worker count the store preflight
        waits for.
    worker_init : str
        Operator-owned shell prelude for the worker block (environment setup).
    address : str or None
        Interchange address reachable from the workers, or ``None`` for the
        Parsl default.
    mpiexec_bind_cmd, mpiexec_overrides : str
        Operator-owned MpiExecLauncher arguments.
    preflight_timeout : int or float
        Positive bound in seconds for every worker to register and pass the
        shared-store preflight.

    Raises
    ------
    TypeError
        If a value has the wrong exact type.
    ValueError
        If the form is not the reviewed attached form or a value is invalid.
    """

    schema: ClassVar[str] = ATTACHED_TRANSPORT_SCHEMA
    executor: str = "htex"
    provider: str = "local"
    launcher: str = "mpiexec"
    label: str = "attached-htex"
    nodes_per_block: int = 1
    max_workers_per_node: int = 1
    worker_init: str = ""
    address: str | None = None
    mpiexec_bind_cmd: str = "--bind-to"
    mpiexec_overrides: str = ""
    preflight_timeout: int | float = 120
    teardown_timeout: int | float = 30

    def __post_init__(self) -> None:
        for name in ("executor", "provider", "launcher", "label", "worker_init", "mpiexec_bind_cmd", "mpiexec_overrides"):
            if type(getattr(self, name)) is not str:
                raise TypeError(f"{name} must be a string")
        # Only the reviewed attached form is admitted; another executor,
        # provider or launcher is refused, never mapped onto this one.
        form = {name: getattr(self, name) for name in _ATTACHED_FORM}
        if form != _ATTACHED_FORM:
            raise ValueError(f"only the attached htex/local/mpiexec form is admitted, got {form!r}")
        if re.fullmatch(_LABEL_PATTERN, self.label) is None or self.label == "_parsl_internal":
            raise ValueError("label must be a short executor label")
        for name in ("nodes_per_block", "max_workers_per_node"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.address is not None and (type(self.address) is not str or not self.address):
            raise TypeError("address must be a non-empty string or None")
        for name in ("preflight_timeout", "teardown_timeout"):
            timeout = getattr(self, name)
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def workers(self) -> int:
        """Return the exact number of workers the block provides."""

        return self.nodes_per_block * self.max_workers_per_node

    def to_dict(self) -> dict[str, Any]:
        """Return the strict mapping form, including the schema."""

        return {
            "schema": self.schema,
            **{item.name: getattr(self, item.name) for item in fields(self)},
        }

    @classmethod
    def from_mapping(cls, data: Any) -> "AttachedTransportSpec":
        """Decode one strict operator-profile mapping.

        Parameters
        ----------
        data : mapping
            Mapping with ``schema`` equal to :data:`ATTACHED_TRANSPORT_SCHEMA`
            and any subset of the spec fields.

        Raises
        ------
        TypeError, ValueError
            If ``data`` is not a mapping, names an unknown key or schema, or
            fails spec validation.
        """

        if isinstance(data, (str, bytes, bytearray)) or not isinstance(data, Mapping):
            raise TypeError("attached transport must be a mapping")
        values = dict(data)
        if values.pop("schema", None) != cls.schema:
            raise ValueError(f"attached transport schema must be {cls.schema!r}")
        unknown = sorted(set(values) - {item.name for item in fields(cls)})
        if unknown:
            raise ValueError(f"attached transport has unknown keys: {unknown!r}")
        return cls(**values)


def _require_standard_worker(materialized: Any) -> None:
    """Refuse a worker program that the remote leaf would not run faithfully.

    The leaf launches science with :class:`SubprocessWorker`, the exact
    ``RUN_SHIM`` command and :class:`TPENEvidence`.  A program with any other
    process, runner or evidence reader would be silently replaced on the
    worker, so it is refused instead.

    Raises
    ------
    UnsupportedTransportError
        If a slot is missing or is not the exact standard type.
    """

    expected = {
        "environment": PreparedPythonEnvironment,
        "process": SubprocessWorker,
        "runner": TPENCommand,
        "evidence": TPENEvidence,
    }
    for name, kind in expected.items():
        try:
            value = materialized[name]
        except (KeyError, TypeError) as error:
            raise UnsupportedTransportError(f"remote worker program lacks its {name} slot") from error
        if type(value) is not kind:
            raise UnsupportedTransportError(
                f"remote worker program {name} must be exactly {kind.__name__}, got {type(value).__name__}"
            )
    if materialized["environment"].setup is not None and type(materialized["environment"].setup) is not SourceScript:
        raise UnsupportedTransportError("remote worker program setup must be exactly SourceScript")


def _interpreter_launch_commands(default_launch_cmd: Any) -> tuple[list[str], str]:
    """Return HTEX interchange and worker-pool commands for this interpreter.

    Parameters
    ----------
    default_launch_cmd : str
        Parsl's own ``DEFAULT_LAUNCH_CMD`` template.  Only its leading
        ``process_worker_pool.py`` script name is replaced, so every
        placeholder Parsl fills stays exactly Parsl's.

    Returns
    -------
    tuple
        ``interchange_launch_cmd`` (an argv list) and ``launch_cmd`` (a
        format template), both running :data:`sys.executable` ``-m``.

    Raises
    ------
    UnsupportedTransportError
        If there is no current interpreter path or the Parsl template is not
        the reviewed ``process_worker_pool.py`` form.
    """

    if type(default_launch_cmd) is not str or not default_launch_cmd.startswith(_WORKER_POOL_SCRIPT):
        raise UnsupportedTransportError("Parsl's worker launch command is not the reviewed process_worker_pool.py form")
    if not sys.executable or not os.path.isabs(sys.executable):
        raise UnsupportedTransportError("the attached transport needs an absolute current interpreter path")
    # The worker template is a shell string that Parsl formats: quote the
    # interpreter and escape any brace in its path.
    python = shlex.quote(sys.executable).replace("{", "{{").replace("}", "}}")
    worker = f"{python} -m {_WORKER_POOL_MODULE} {default_launch_cmd[len(_WORKER_POOL_SCRIPT):]}"
    return [sys.executable, "-m", _INTERCHANGE_MODULE], worker


def _lockf_probe(path: str) -> None:
    """Take and release one non-blocking POSIX ``lockf`` on ``path``."""

    fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.lockf(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _shared_store_probe(root: str, real_root: str, nonce: str, expected: int, timeout: float) -> dict[str, Any]:
    """Worker side of the shared-store preflight; returns plain facts.

    Each probe confirms that ``root`` resolves to the controller's canonical
    path and that the controller-written token is visible with its nonce.
    It proves POSIX ``lockf`` on a lock-capability file private to this
    probe, never on the shared token, so a peer probe holding its own lock
    cannot make a healthy store look unlockable.  It then waits at a file
    barrier until ``expected`` probes have arrived.  A worker runs one task
    at a time, so the barrier completes only when every worker of the block
    holds a probe at once; a missing, slow or store-blind worker makes it
    time out.

    Parameters
    ----------
    root : str
        Lexical attempt-store root.
    real_root : str
        Controller ``realpath`` of ``root``.
    nonce : str
        Token content written by the controller for this run.
    expected : int
        Exact worker count of the block.
    timeout : float
        Bound in seconds for the whole probe.

    Returns
    -------
    dict
        ``{"host", "pid", "root"}`` of this worker.

    Raises
    ------
    RuntimeError, OSError, TimeoutError
        If any check fails or the barrier is not reached in time.
    """

    deadline = time.monotonic() + timeout
    if os.path.realpath(root) != real_root:
        raise RuntimeError("attempt store resolves to a different canonical path on this worker")
    probe_dir = os.path.join(root, "parsl", _PREFLIGHT_PREFIX + nonce)
    token = os.path.join(probe_dir, "token")
    with open(token, encoding="utf-8") as handle:
        if handle.read() != nonce:
            raise RuntimeError("attempt store token differs on this worker")
    # Lock capability is proven on a file only this probe ever opens: probes
    # on other workers never contend for it, so ordinary contention between
    # probes cannot become a refusal.  The shared token above still proves
    # store identity.
    probe_id = uuid.uuid4().hex
    private_lock = os.path.join(probe_dir, f"lock-{probe_id}")
    os.close(os.open(private_lock, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600))
    _lockf_probe(private_lock)
    host, pid = socket.gethostname(), os.getpid()
    with open(os.path.join(probe_dir, f"arrival-{probe_id}"), "x", encoding="utf-8") as handle:
        handle.write(f"{host} {pid}\n")
    while sum(name.startswith("arrival-") for name in os.listdir(probe_dir)) < expected:
        if time.monotonic() > deadline:
            raise TimeoutError("not every worker reached the shared-store preflight barrier")
        time.sleep(0.05)
    return {"host": host, "pid": pid, "root": real_root}


class _LeafEvidence:
    """Replay a remote leaf's inspection facts on the controller.

    The shared completion path calls ``read`` exactly where a local
    completion inspects; this reader returns the evidence the leaf already
    decoded on its worker and never launches anything on the controller.
    """

    def __init__(self, result: LeafResult) -> None:
        self._result = result

    def read(self, *_args: Any, **_kwargs: Any) -> CoreRunEvidence:
        """Return leaf evidence or raise the leaf's inspection failure."""

        result = self._result
        if result.inspection == "evidence":
            evidence = result.core_evidence()
            assert evidence is not None
            return evidence
        message = "" if result.inspection_error is None else result.inspection_error["message"]
        if result.inspection == "unavailable":
            raise EvidenceUnavailable(message)
        # "error", or "not-run" where the controller expected inspection: the
        # leaf facts cannot support a published evidence record.
        raise RemoteInspectionError(f"remote inspection {result.inspection}: {message}")


class _RemoteLeaf:
    """Controller-side record of one submitted remote leaf.

    The controller never signals a remote process: :meth:`request_kill` is a
    no-op, and bounded remote teardown acknowledgement is S3 Layer C.
    """

    def __init__(self, request: LeafRequest) -> None:
        self.request = request
        self.future: Any = None

    def request_kill(self) -> None:
        """Request leaf-owned cancellation without signalling a remote PID.

        Parsl does not expose a portable remote-PID signal operation.  The
        attached transport therefore treats this as a transport cancellation
        request; only the leaf's returned teardown receipt can settle it.
        """

        self.request_cancel()

    def request_cancel(self) -> None:
        """Send the transport cancellation request (never ``killpg``)."""

        return None

    def bind_future(self, future: Any) -> None:
        self.future = future

    def wait_for_teardown(self, bound: float) -> bool:
        """Wait for a leaf result and accept only its validated teardown ack."""

        deadline = time.monotonic() + bound
        while self.future is not None and not self.future.done() and time.monotonic() < deadline:
            _poll_sleep(min(0.01, max(deadline - time.monotonic(), 0)))
        if self.future is None or not self.future.done():
            return False
        try:
            payload = self.future.result(None)
            result = validate_leaf_result(self.request, LeafResult.decode(payload))
        except BaseException:
            return False
        return result.teardown.get("child_group_reaped") is True

    def resolve(self, payload: Any) -> tuple[ProcessOutcome, dict[str, Any]]:
        """Validate the returned bytes and adapt them to the completion path.

        Returns
        -------
        tuple
            The reaped science :class:`ProcessOutcome` and a materialized
            mapping whose evidence reader replays the leaf inspection.

        Raises
        ------
        TypeError, ValueError, LeafResultRejected
            If the bytes are not a result answering this request.
        PlacementRefusal
            If the leaf refused its slot's CPUs on its worker before science.
        RemoteLeafLaunchError
            If the leaf returned any other launch error instead of science.
        """

        if type(payload) is not bytes:
            raise TypeError("remote leaf returned non-bytes")
        result = validate_leaf_result(self.request, LeafResult.decode(payload))
        if result.science is None:
            error = result.launch_error or {"type": "unknown", "message": ""}
            if error["type"] == LeafPlacementRefused.__name__:
                # Only the closed worker-affinity codes become typed
                # placement refusals; any other text stays a launch error.
                code, _, reason = error["message"].partition(": ")
                if code in WORKER_AFFINITY_REFUSAL_CODES and reason:
                    raise PlacementRefusal(code, reason)
            raise RemoteLeafLaunchError(f"{error['type']}: {error['message']}")
        science = result.science
        outcome = ProcessOutcome(
            tuple(science["argv"]),
            science["returncode"],
            science["stdout_path"],
            science["stderr_path"],
            science["timed_out"],
            science["pid"],
            science["proc_start_ticks"],
        )
        return outcome, {"process": None, "evidence": _LeafEvidence(result)}


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
    transport : AttachedTransportSpec or mapping, optional
        Explicit selection of the attached remote HTEX/LocalProvider/MpiExec
        route.  When omitted, dispatch is the unchanged in-process ThreadPool
        route.  It cannot be combined with ``executor_config``.

    Notes
    -----
    Parsl is imported only while constructing this dispatcher. Each :meth:`run`
    creates and cleans one DataFlowKernel; no DFK or future is reused across
    runs. Facility-specific nodefile and tiling validation remains outside this
    executor-owned adapter.

    With ``transport`` set, admission additionally requires an
    :class:`AttachedAllocation`, the standard worker program, enough workers
    for every slot and an existing attempt-store root, all before the first
    lifecycle write.  Each run then builds one fresh HTEX config and DFK,
    preflights the shared store on every worker before the first ``begin``,
    and submits each member as one Layer A leaf request with Parsl retries
    disabled.  Leaf facts are validated against their request and published
    by the controller through the shared completion path.
    """

    worker: WorkerProgram
    executor_config: Any = None
    binding: WorkerBinding | Mapping[str, Any] | None = None
    transport: AttachedTransportSpec | Mapping[str, Any] | None = None
    _config_type: Any = field(init=False, repr=False, compare=False)
    _dfk_type: Any = field(init=False, repr=False, compare=False)
    _thread_pool_type: Any = field(init=False, repr=False, compare=False)
    _htex_type: Any = field(init=False, default=None, repr=False, compare=False)
    _htex_launch_commands: tuple[list[str], str] | None = field(init=False, default=None, repr=False, compare=False)
    _local_provider_type: Any = field(init=False, default=None, repr=False, compare=False)
    _mpiexec_launcher_type: Any = field(init=False, default=None, repr=False, compare=False)
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
        if self.transport is not None:
            # Typed admission happens at construction: an unknown form never
            # reaches run(), so it can never begin an attempt.
            if self.executor_config is not None:
                raise UnsupportedTransportError("transport and executor_config are mutually exclusive")
            if type(self.transport) is not AttachedTransportSpec:
                if isinstance(self.transport, AttachedTransportSpec):
                    raise TypeError("transport must be exactly AttachedTransportSpec")
                object.__setattr__(self, "transport", AttachedTransportSpec.from_mapping(self.transport))
            # The remote route's Parsl types are imported only when selected,
            # so the default ThreadPool route imports exactly what it did.
            from parsl.executors import HighThroughputExecutor
            from parsl.executors.high_throughput.executor import DEFAULT_LAUNCH_CMD
            from parsl.launchers import MpiExecLauncher
            from parsl.providers import LocalProvider

            object.__setattr__(self, "_htex_launch_commands", _interpreter_launch_commands(DEFAULT_LAUNCH_CMD))
            object.__setattr__(self, "_htex_type", HighThroughputExecutor)
            object.__setattr__(self, "_local_provider_type", LocalProvider)
            object.__setattr__(self, "_mpiexec_launcher_type", MpiExecLauncher)

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
        UnsupportedTransportError
            If a selected attached transport cannot be admitted; its
            :class:`SharedStorePreflightError` subclass is raised when the
            attempt store is not one shared, lockable path on every worker.
            Both are raised before the first lifecycle write.
        """

        token = self._begin_dispatch(group, runs, lifecycle, env, timeout)
        dfk: Any = None
        original: BaseException | None = None
        try:
            bindings = self._resolve_bindings(lifecycle)
            config = self.executor_config
            if self.transport is not None:
                # A fresh executor/provider/launcher set for every run: none
                # of them, nor the DFK below, outlives this allocation's run.
                run_dir = str(Path(lifecycle.store.root) / "parsl")
                config = self._attached_config(self.transport, run_dir)
                labels = self._validate_attached_config(config, self.transport)
            else:
                if config is None:
                    run_dir = str(Path(lifecycle.store.root) / "parsl")
                    config = self._config_type(
                        executors=[self._thread_pool_type(max_threads=len(bindings))],
                        run_dir=run_dir,
                        usage_tracking=False,
                    )
                labels = self._validate_executor_config(config, len(bindings))
            if len(bindings) == 1 and self.transport is None:
                object.__setattr__(self, "_active_binding", bindings[0])
            object.__setattr__(self, "_active_executor_labels", labels)
            dfk = self._dfk_type(config=config)
            object.__setattr__(self, "_active_dfk", dfk)
            try:
                if self.transport is not None:
                    # Every worker must see the same lockable store before
                    # the first begin; a refusal leaves no attempt.
                    self._preflight_shared_store(lifecycle, self.transport)
                    return self._run_slot_pool(group, runs, lifecycle, env, timeout, bindings)
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
        """Perform common admission plus strict attached-slot admission.

        With an attached transport this also refuses, before any lifecycle
        write, a lifecycle that is not :class:`AttachedAllocation`, a
        non-standard worker program, more slots than workers, and a store
        root that is not an existing absolute POSIX directory.
        """

        super(ParslDispatcher, self).validate(group, runs, lifecycle, env, timeout)
        bindings = self._resolve_bindings(lifecycle)
        if self.transport is not None:
            self._admit_transport(lifecycle, env, len(bindings))

    def _admit_transport(self, lifecycle: AttachedAllocation, env: Mapping[str, Any] | None, slots: int) -> None:
        """Refuse an attached remote run that cannot be dispatched faithfully."""

        transport = self.transport
        if type(transport) is not AttachedTransportSpec:
            raise UnsupportedTransportError("transport must be an admitted AttachedTransportSpec")
        if not isinstance(lifecycle, AttachedAllocation):
            raise UnsupportedTransportError("the attached transport requires an AttachedAllocation lifecycle")
        if slots > transport.workers:
            raise UnsupportedTransportError("attached transport has fewer workers than admitted slots")
        # The leaf always runs the standard worker; refuse any other program.
        _require_standard_worker(self.worker.materialize(env))
        root = getattr(lifecycle.store, "root", None)
        if os.name != "posix" or type(root) is not str or not os.path.isabs(root):
            raise SharedStorePreflightError("attempt store root must be an absolute POSIX path")
        if not os.path.isdir(root):
            raise SharedStorePreflightError("attempt store root must already exist before an attached run")

    def _resolve_bindings(self, lifecycle: AttachedAllocation) -> tuple[WorkerBinding, ...]:
        """Resolve immutable per-slot bindings before any lifecycle write."""

        if not isinstance(lifecycle, AttachedAllocation):
            if self.binding is None:
                raise BindingError("binding is required when lifecycle is not AttachedAllocation")
            return (WorkerBinding.from_mapping(self.binding),)
        try:
            slots = physical_slots_from_allocation(lifecycle.allocation)
        except PlacementRefusal:
            # Keep the raw slot binding long enough for the shared completion
            # path to publish the typed prelaunch v2 refusal.
            slots = ()
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
                # A fence is begun before preparation. Keep both values
                # defined so preparation errors reach typed completion.
                materialized = None
                environment = None
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
                    # A remote slot's CPUs are not the controller's: the
                    # attached leaf checks them on its worker before science.
                    check_controller_affinity=self.transport is None,
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
                _admit_slot(environment, fence)
                process = materialized["process"]
                materialized["runner"]
                evidence = materialized["evidence"]
                if not callable(getattr(process, "run", None)):
                    raise TypeError("materialized worker process lacks run()")
                if not callable(getattr(evidence, "read", None)):
                    raise TypeError("materialized worker evidence reader lacks read()")
                output_dir, _core_dir, stdout_path, stderr_path, _evidence_stdout, _evidence_stderr = _member_paths(fence)
                args = (*configured.command().args, f"run.root={output_dir}", "run.layout=flat", "run.run_id=core")
                if self.transport is not None:
                    future, handle = self._submit_leaf(materialized, environment, lifecycle, fence, configured, timeout)
                else:
                    future, handle = self._submit_process(
                        materialized,
                        environment,
                        args,
                        cwd=output_dir,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        timeout=timeout,
                    )
            except Exception as error:
                completion_error = self._complete_member(
                    group, member, configured, lifecycle, fence, materialized, environment,
                    None, timeout, error=error,
                )
                if completion_error is not None and finish_error is None:
                    finish_error = completion_error
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
                    if isinstance(record["handle"], _RemoteLeaf):
                        # Validated leaf facts replace controller inspection;
                        # the controller still publishes every record.
                        try:
                            result, record["materialized"] = record["handle"].resolve(result)
                        except PlacementRefusal as refusal:
                            # The worker refused the slot before science:
                            # publish its typed refusal, as a controller
                            # placement refusal is published.
                            record["environment"] = _with_placement_refusal(record["environment"], refusal)
                            raise
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
            if isinstance(record["handle"], _RemoteLeaf):
                # A remote leaf is neither signalled nor awaited here; its
                # attempt is finished failed and never accepted.  Bounded
                # remote teardown acknowledgement is S3 Layer C.
                handle = record["handle"]
                handle.request_cancel()
                transport = self.transport
                bound = float(getattr(transport, "teardown_timeout", 30)) if transport is not None else 30.0
                acknowledged = handle.wait_for_teardown(bound)
                try:
                    status = "failed" if acknowledged else "remote-quiescence-unconfirmed"
                    lifecycle.finish(record["fence"], status, self._utc_now(), None)
                except BaseException as error:
                    try:
                        cancellation.add_note(f"slot {record['member'].task_id} finish unresolved: {error!r}")
                    except BaseException:
                        pass
                continue
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

    def _attached_config(self, transport: AttachedTransportSpec, run_dir: str) -> Any:
        """Build one fresh attached HTEX config for a single run.

        Parameters
        ----------
        transport : AttachedTransportSpec
            Admitted transport selection.
        run_dir : str
            Parsl run directory below the attempt-store root.

        Returns
        -------
        parsl.config.Config
            One HighThroughputExecutor with a one-block LocalProvider and an
            MpiExecLauncher.  Retries are disabled and the strategy only
            starts the initial block, so nothing is acquired, replaced or
            retried after submission.  The interchange and the worker pool
            run as modules of the current interpreter, independent of PATH.
        """

        launcher = self._mpiexec_launcher_type(
            bind_cmd=transport.mpiexec_bind_cmd,
            overrides=transport.mpiexec_overrides,
        )
        provider = self._local_provider_type(
            nodes_per_block=transport.nodes_per_block,
            launcher=launcher,
            init_blocks=1,
            min_blocks=1,
            max_blocks=1,
            worker_init=transport.worker_init,
        )
        assert self._htex_launch_commands is not None
        interchange_launch_cmd, launch_cmd = self._htex_launch_commands
        executor = self._htex_type(
            label=transport.label,
            provider=provider,
            address=transport.address,
            max_workers_per_node=transport.max_workers_per_node,
            # Parsl must not rewrite the slot binding: no affinity and no
            # accelerator assignment by the worker pool.
            cpu_affinity="none",
            available_accelerators=(),
            prefetch_capacity=0,
            # Never resolve interchange.py or process_worker_pool.py on PATH.
            interchange_launch_cmd=list(interchange_launch_cmd),
            launch_cmd=launch_cmd,
        )
        return self._config_type(
            executors=[executor],
            run_dir=run_dir,
            retries=0,
            strategy="none",
            usage_tracking=0,
        )

    def _validate_attached_config(self, config: Any, transport: AttachedTransportSpec) -> tuple[str, ...]:
        """Refuse a run config that is not exactly the admitted attached form."""

        executors = getattr(config, "executors", None)
        if not isinstance(executors, (tuple, list)) or len(executors) != 1:
            raise UnsupportedTransportError("attached transport requires exactly one executor")
        executor = executors[0]
        provider = getattr(executor, "provider", None)
        if (
            type(executor) is not self._htex_type
            or type(provider) is not self._local_provider_type
            or type(getattr(provider, "launcher", None)) is not self._mpiexec_launcher_type
        ):
            raise UnsupportedTransportError("attached transport admits only HTEX with LocalProvider and MpiExecLauncher")
        if executor.label != transport.label:
            raise UnsupportedTransportError("attached executor label differs from the transport label")
        if getattr(config, "retries", None) != 0 or getattr(config, "retry_handler", None) is not None:
            raise UnsupportedTransportError("attached transport must not retry submissions")
        return (transport.label,)

    def _preflight_shared_store(self, lifecycle: AttachedAllocation, transport: AttachedTransportSpec) -> None:
        """Fail closed unless every worker sees the same lockable store.

        The controller writes a nonce token below the store's Parsl run
        directory and checks POSIX ``lockf`` on it, then submits one
        :func:`_shared_store_probe` per worker.  All probes must return the
        controller's canonical root within ``preflight_timeout``.  Probes
        prove ``lockf`` on private files, so they never contend with each
        other, and no Parsl app is ever retried.

        Raises
        ------
        SharedStorePreflightError
            If any controller or worker check fails or times out.  No
            lifecycle record has been written.
        """

        root = lifecycle.store.root
        real_root = os.path.realpath(root)
        nonce = uuid.uuid4().hex
        probe_dir = os.path.join(root, "parsl", _PREFLIGHT_PREFIX + nonce)
        try:
            os.makedirs(probe_dir)
            token = os.path.join(probe_dir, "token")
            with open(token, "x", encoding="utf-8") as handle:
                handle.write(nonce)
            _lockf_probe(token)
        except OSError as error:
            raise SharedStorePreflightError(f"attempt store is not a lockable POSIX directory: {error}") from error
        app = self._python_app_factory(
            data_flow_kernel=self._active_dfk,
            cache=False,
            executors=[transport.label],
        )(_shared_store_probe)
        deadline = time.monotonic() + transport.preflight_timeout
        futures = [
            app(root, real_root, nonce, transport.workers, transport.preflight_timeout)
            for _ in range(transport.workers)
        ]
        for future in futures:
            try:
                facts = future.result(timeout=max(deadline - time.monotonic(), 0) + 5)
            except Exception as error:
                raise SharedStorePreflightError(f"a worker failed the shared-store preflight: {error!r}") from error
            if not isinstance(facts, dict) or facts.get("root") != real_root:
                raise SharedStorePreflightError("a worker reported a different canonical attempt-store path")

    def _submit_leaf(
        self,
        materialized: Any,
        environment: PreparedPythonEnvironment,
        lifecycle: AttachedAllocation,
        fence: Any,
        configured: ConfiguredRun,
        timeout: float | None,
    ) -> tuple[Any, _RemoteLeaf]:
        """Submit one begun member as a Layer A leaf on the attached HTEX.

        Only the canonical request bytes cross the Parsl boundary.  The
        environment inside them is the controller's prepared environment
        after the slot binding, context-key and placement overlays; the leaf
        applies it over the worker's own environment, so neither controller
        nor worker ambient values can replace a bound key.
        """

        # The materialized program is re-checked per member: the leaf never
        # runs a program it would silently replace.
        _require_standard_worker(materialized)
        request = LeafRequest.for_attempt(fence, lifecycle.allocation, configured, environment, timeout)
        dfk = self._active_dfk
        if dfk is None:
            raise RuntimeError("Parsl DFK is not active")
        app = self._python_app_factory(
            data_flow_kernel=dfk,
            cache=False,
            executors=list(self._active_executor_labels),
        )(run_leaf)
        handle = _RemoteLeaf(request)
        future = app(request.encode())
        handle.bind_future(future)
        return future, handle

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


__all__ = [
    "ATTACHED_TRANSPORT_SCHEMA",
    "AttachedTransportSpec",
    "ParslDispatcher",
    "ParslUnavailableError",
    "RemoteInspectionError",
    "RemoteLeafLaunchError",
    "SharedStorePreflightError",
    "UnsupportedParslExecutorError",
    "UnsupportedTransportError",
]
