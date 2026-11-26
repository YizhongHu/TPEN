"""Non-blocking local leaf submission for one already-fenced member.

``LocalLeafExecutor`` owns only the submit/collect seam.  The caller owns the
durable fence, invocation, launch intent, and outcome receipts.  A submit
creates one :class:`~tpen_executor.dispatch.transport.LeafRequest` through
``LeafRequest.for_attempt`` and schedules exactly one ``run_leaf`` call in a
bounded owner-process thread pool.  Collection decodes and validates the
returned :class:`~tpen_executor.dispatch.transport.LeafResult` before it is
returned to the controller.

The module adds no child environment variables.  The base
``PreparedPythonEnvironment`` is supplied by the existing worker materialiser;
the existing placement helpers may add ``TPEN_EXECUTOR_CONTEXT_ENV_KEYS``,
``CUDA_VISIBLE_DEVICES``, ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, declared
``TPEN_CTX_*`` observation keys, ``TPEN_SLOT_ID``, ``TPEN_SLOT_NODE``,
``TPEN_SLOT_CPUS``, ``TPEN_PLACEMENT_OUTPUT``,
``TPEN_PLACEMENT_OUTPUT_DIR``, ``TPEN_PLACEMENT_TASK_ID``,
``TPEN_PLACEMENT_ATTEMPT_ID``, ``TPEN_PLACEMENT_GROUP_DIGEST``,
``TPEN_PLACEMENT_ALLOCATION_ID``, ``TPEN_PLACEMENT_BACKEND_JOB_ID``,
``TPEN_PLACEMENT_SLOT_ID``, ``TPEN_PLACEMENT_DECLARED_HOST``,
``TPEN_PLACEMENT_DECLARED_CPUS``, ``TPEN_PLACEMENT_DECLARED_DEVICES``,
``TPEN_PLACEMENT_PAYLOAD_DIGEST``, and, for an existing v2 placement binding,
``TPEN_PLACEMENT_RECEIPT_VERSION``, ``TPEN_PLACEMENT_HOST_STATUS``,
``TPEN_PLACEMENT_GPU_SAMPLER_OUTPUT``, and ``TPEN_SLOT_QUARANTINE_DIR``.
Their source remains the allocation/slot binding, the owner worker
environment, and the fenced placement overlay.  ``OPENBLAS_NUM_THREADS`` and
``NUMEXPR_NUM_THREADS`` may be carried by an existing slot environment.  The
local path does not add ``TPEN_ROUTED_STRICT_HOST``; that is an existing D2
routed-leaf policy.  No ``TPEN_EXECUTOR_LAUNCH_ROLE`` or other executor-local
key is introduced here.

The child receives only the arguments already defined by
``LeafRequest.science_args``: ``TPENCommand(config, overrides)`` produces
``-c RUN_SHIM --config <absolute-config> <immutable-overrides...>`` and the
existing code-bound overrides append ``run.root=<fenced-output>``,
``run.layout=flat``, and ``run.run_id=core``.  This module adds no CLI
argument.  The module itself never imports ``tpen``; ``RUN_SHIM`` is consumed
only by the science child.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
import threading
from typing import Any

from ..allocations import ConfiguredAllocation, MemberInvocationV2
from ..configured_program import ConfiguredLoad
from ..contracts import ProgramSpecRef, ReadyGroupV2, TaskSpec
from ..dispatch.contract import _with_child_context_keys, _with_child_placement
from ..dispatch.contract import ConfiguredProgramRouteRefused
from ..dispatch.transport import LeafRequest, LeafResult, LeafResultRejected, run_leaf, validate_leaf_result
from ..environments import PreparedPythonEnvironment
from ..placement import AllocationSlot, physical_slots_from_allocation
from ..storage import AttemptFence
from ..tasks import ConfiguredRun
from .controller import MemberLaunchResult


class LocalExecutorError(RuntimeError):
    """Base class for typed local-executor admission and lifecycle errors."""


class InvocationMismatchError(LocalExecutorError):
    """Raised when an invocation or fence crosses the immutable bindings."""


class CapacityExceededError(LocalExecutorError):
    """Raised when all admitted local slots already have outstanding work."""


class UnknownLaunchHandleError(LocalExecutorError):
    """Raised when collecting an unknown or already-collected handle."""


class HandleCollectionInProgressError(LocalExecutorError):
    """Raised when another collector already owns a handle's collection."""


class LeafSubmissionError(LocalExecutorError):
    """Raised when the local worker cannot accept a leaf submission."""


class CollectionWaitTimeout(FutureTimeoutError, LocalExecutorError):
    """Raised when a nonterminal leaf has not completed within one poll."""


@dataclass(frozen=True)
class LeafCompletionContext:
    """Opaque executor-owned completion facts returned by ``collect``."""

    result: LeafResult
    request: LeafRequest
    configured: ConfiguredRun
    environment: PreparedPythonEnvironment

    def __getattr__(self, name: str) -> Any:
        """Preserve result-like access for callers migrating from ``collect``."""

        return getattr(self.result, name)


class MemberLaunchHandle:
    """Opaque handle for one accepted local leaf.

    The identity properties are intentionally public so a controller can carry
    them to its completion path, while the future and request remain private
    executor state.  A handle is valid only with the executor that issued it.
    """

    __slots__ = (
        "_future", "_request", "_group_digest", "_task_id", "_lineage_id",
        "_attempt_id", "_allocation_id", "_backend_job_id", "_invocation_id", "_slot",
        "_configured", "_environment",
    )

    def __init__(self, request: LeafRequest, invocation: MemberInvocationV2, future: Future[Any], slot: AllocationSlot | None,
                 configured: ConfiguredRun, environment: PreparedPythonEnvironment) -> None:
        self._future = future
        self._request = request
        self._slot = slot
        self._group_digest = request.group_digest
        self._task_id = request.task_id
        self._lineage_id = invocation.lineage_id
        self._attempt_id = request.attempt_id
        self._allocation_id = request.allocation_id
        self._backend_job_id = request.backend_job_id
        self._invocation_id = invocation.invocation_id.value
        self._configured = configured
        self._environment = environment

    @property
    def request(self) -> LeafRequest:
        """Return the immutable request submitted for this handle."""

        return self._request

    @property
    def configured(self) -> ConfiguredRun:
        """Return the configured run bound to this handle."""

        return self._configured

    @property
    def environment(self) -> PreparedPythonEnvironment:
        """Return the prepared environment bound to this handle."""

        return self._environment

    @property
    def group_digest(self) -> str:
        """Return the group identity bound to this handle."""

        return self._group_digest

    @property
    def task_id(self) -> str:
        """Return the task identity bound to this handle."""

        return self._task_id

    @property
    def lineage_id(self) -> str:
        """Return the task lineage identity bound to this handle."""

        return self._lineage_id

    @property
    def attempt_id(self) -> str:
        """Return the attempt identity bound to this handle."""

        return self._attempt_id

    @property
    def allocation_id(self) -> str:
        """Return the allocation identity bound to this handle."""

        return self._allocation_id

    @property
    def backend_job_id(self) -> str:
        """Return the backend job identity bound to this handle."""

        return self._backend_job_id

    @property
    def invocation_id(self) -> str:
        """Return the invocation identity bound to this handle."""

        return self._invocation_id

    @property
    def slot_id(self) -> str | None:
        """Return the exact allocation slot leased to this handle."""

        return None if self._slot is None else self._slot.slot_id


EnvironmentFactory = Callable[[TaskSpec, ConfiguredRun, AllocationSlot | None], PreparedPythonEnvironment]
ProgramMaterializer = Callable[[ProgramSpecRef], ConfiguredRun]


class LocalLeafExecutor:
    """Submit and collect bounded, already-fenced local leaves.

    Parameters
    ----------
    group : ReadyGroupV2
        Immutable group whose task and lineage bindings are admitted.
    allocation : ConfiguredAllocation
        Immutable allocation whose identity and slots bound every request.
    materialize_program : callable or mapping
        Per-task materialiser from a task's :class:`ProgramSpecRef` to its
        strict :class:`ConfiguredRun`.  A mapping may be supplied for tests or
        callers that have already materialised the task table.
    environment : PreparedPythonEnvironment or callable
        Base worker environment, or a factory receiving ``(task, configured,
        slot)``.  The existing context and placement helpers add their bound
        fields; this class adds none.
    capacity : int, optional
        Maximum outstanding handles.  When omitted, the number of declared
        allocation slots is used, or one for the legacy no-slot path.
    timeout : int or float, optional
        Leaf science timeout passed to ``LeafRequest.for_attempt``.
    leaf_runner : callable, optional
        Injectable ``run_leaf``-compatible callable returning canonical bytes.
        The production default is the existing transport function.
    """

    def __init__(
        self,
        group: ReadyGroupV2,
        allocation: ConfiguredAllocation,
        materialize_program: ProgramMaterializer | Mapping[str, ConfiguredRun],
        environment: PreparedPythonEnvironment | EnvironmentFactory,
        *,
        capacity: int | None = None,
        timeout: int | float | None = None,
        leaf_runner: Callable[[bytes], bytes | LeafResult] | None = None,
    ) -> None:
        if not isinstance(group, ReadyGroupV2):
            raise TypeError("group must be ReadyGroupV2")
        if not isinstance(allocation, ConfiguredAllocation):
            raise TypeError("allocation must be ConfiguredAllocation")
        if not callable(materialize_program) and not isinstance(materialize_program, Mapping):
            raise TypeError("materialize_program must be callable or a task mapping")
        if not isinstance(environment, PreparedPythonEnvironment) and not callable(environment):
            raise TypeError("environment must be PreparedPythonEnvironment or a factory")
        if timeout is not None and (type(timeout) not in (int, float) or timeout <= 0):
            raise ValueError("timeout must be positive or None")
        if leaf_runner is not None and not callable(leaf_runner):
            raise TypeError("leaf_runner must be callable")
        try:
            slots = tuple(physical_slots_from_allocation(allocation))
        except Exception:
            # Slot decoding is an immutable construction boundary.  Preserve
            # the original typed refusal rather than silently falling back to
            # an unbounded or scheduler-discovered capacity.
            raise
        declared_capacity = len(slots)
        bound_capacity = declared_capacity or (capacity if capacity is not None else 1)
        if capacity is None:
            capacity = bound_capacity
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if declared_capacity and capacity > declared_capacity:
            raise CapacityExceededError("capacity exceeds the allocation slot bound")

        self._group = group
        self._allocation = allocation
        self._materialize_program = materialize_program
        self._environment = environment
        self._slots = slots[:capacity]
        self.capacity = capacity
        self.timeout = timeout
        self._leaf_runner = run_leaf if leaf_runner is None else leaf_runner
        self._pool = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix="tpen-local-leaf")
        self._lock = threading.Lock()
        self._active = 0
        self._free_slots = list(self._slots)
        self._retired_slot_ids: set[str] = set()
        self._handles: dict[int, MemberLaunchHandle] = {}
        self._collected: set[int] = set()
        self._collecting: set[int] = set()

    @property
    def retired_slot_ids(self) -> tuple[str, ...]:
        """Return the immutable snapshot of slots retired by terminal errors."""

        with self._lock:
            return tuple(sorted(self._retired_slot_ids))

    def _task(self, task_id: str) -> TaskSpec:
        """Return the one immutable task bound to ``task_id``."""

        for task in self._group.members:
            if task.task_id == task_id:
                return task
        raise InvocationMismatchError(f"task {task_id!r} is not a member of the configured group")

    def _validate_invocation(self, fence: AttemptFence, invocation: MemberInvocationV2) -> TaskSpec:
        """Validate all group, task, lineage, allocation and fence identities."""

        if not isinstance(fence, AttemptFence) or not isinstance(invocation, MemberInvocationV2):
            raise InvocationMismatchError("submit requires AttemptFence and MemberInvocationV2")
        task = self._task(invocation.task_id)
        if (
            invocation.group_digest != self._group.identity
            or invocation.group_digest != fence.group_digest
            or invocation.task_id != fence.task_id
            or invocation.lineage_id != task.lineage_id
            or invocation.attempt_id != fence.attempt_id
            or fence.allocation_id != self._allocation.allocation_id.value
            or invocation.allocation_attempt.group_id != self._group.group_id
            or invocation.allocation_attempt.allocation_id != self._allocation.allocation_id
            or invocation.allocation_attempt.backend_job_id != self._allocation.backend_job_id
        ):
            raise InvocationMismatchError("invocation or fence crosses the immutable executor bindings")
        return task

    def _configured(self, task: TaskSpec) -> ConfiguredRun:
        """Materialise and validate the configured run for ``task``."""

        materializer = self._materialize_program
        configured = materializer[task.task_id] if isinstance(materializer, Mapping) else materializer(task.program)
        if not isinstance(configured, ConfiguredRun) or configured.id != task.task_id:
            raise LeafSubmissionError("program materialiser returned a run for the wrong task")
        return configured

    def _base_environment(self, task: TaskSpec, configured: ConfiguredRun, slot: AllocationSlot | None) -> PreparedPythonEnvironment:
        """Obtain one prepared environment from the existing worker seam."""

        value = self._environment(task, configured, slot) if callable(self._environment) else self._environment
        if not isinstance(value, PreparedPythonEnvironment):
            raise LeafSubmissionError("environment factory did not return PreparedPythonEnvironment")
        if slot is not None:
            value = PreparedPythonEnvironment(value.python, value.setup, {**value.env, **slot.env})
        return value

    def _submit(
        self,
        fence: AttemptFence,
        invocation: MemberInvocationV2,
        *,
        admitted_checkpoint_path: str | None = None,
    ) -> tuple[MemberLaunchResult, MemberLaunchHandle]:
        """Accept one leaf, optionally binding its configured resume path."""

        task = self._validate_invocation(fence, invocation)
        with self._lock:
            if self._active >= self.capacity or (self._slots and not self._free_slots):
                raise CapacityExceededError("local leaf capacity is already admitted")
            slot = self._free_slots.pop() if self._free_slots else None
            self._active += 1
        try:
            configured = self._configured(task)
            if admitted_checkpoint_path is not None:
                if configured.program is None:
                    raise LeafSubmissionError("admitted successor requires a configured program")
                configured = replace(
                    configured,
                    program=replace(
                        configured.program,
                        load=ConfiguredLoad("train_resume", admitted_checkpoint_path),
                    ),
                )
            environment = self._base_environment(task, configured, slot)
            environment = _with_child_context_keys(environment, self._allocation)
            environment = _with_child_placement(
                environment,
                self._allocation,
                fence,
                self._group,
                task,
                configured,
                selected_slot=slot,
            )
            request = LeafRequest.for_attempt(
                fence,
                self._allocation,
                configured,
                environment,
                self.timeout,
                selected_slot=slot,
            )
            future = self._pool.submit(self._run_leaf, request)
        except Exception as error:
            with self._lock:
                self._active -= 1
                if slot is not None:
                    self._free_slots.append(slot)
            if isinstance(error, (InvocationMismatchError, CapacityExceededError, LeafSubmissionError, ConfiguredProgramRouteRefused)):
                raise
            raise LeafSubmissionError(f"local leaf submission failed: {error}") from error
        handle = MemberLaunchHandle(request, invocation, future, slot, configured, environment)
        with self._lock:
            self._handles[id(handle)] = handle
        return MemberLaunchResult(True, status="submitted", detail="local leaf accepted"), handle

    def submit(self, fence: AttemptFence, invocation: MemberInvocationV2) -> tuple[MemberLaunchResult, MemberLaunchHandle]:
        """Accept exactly one legacy/fresh leaf without waiting for completion."""

        return self._submit(fence, invocation)

    def submit_admitted_successor(
        self,
        fence: AttemptFence,
        invocation: MemberInvocationV2,
        checkpoint_path: str,
    ) -> tuple[MemberLaunchResult, MemberLaunchHandle]:
        """Accept an R2 successor only with its admitted train-resume path."""

        return self._submit(fence, invocation, admitted_checkpoint_path=checkpoint_path)

    def _run_leaf(self, request: LeafRequest) -> LeafResult:
        """Run one canonical request and decode its canonical result."""

        raw = self._leaf_runner(request.encode())
        if isinstance(raw, LeafResult):
            return raw
        try:
            return LeafResult.decode(raw)
        except (TypeError, ValueError) as error:
            raise LeafResultRejected(f"leaf runner returned an invalid LeafResult: {error}") from error

    def collect(self, handle: MemberLaunchHandle, timeout: float | None = None) -> LeafCompletionContext:
        """Wait for and validate one issued handle exactly once."""

        if not isinstance(handle, MemberLaunchHandle):
            raise UnknownLaunchHandleError("collect requires a MemberLaunchHandle")
        key = id(handle)
        with self._lock:
            if self._handles.get(key) is not handle or key in self._collected:
                raise UnknownLaunchHandleError("launch handle is unknown or already collected")
            if key in self._collecting:
                raise HandleCollectionInProgressError("launch handle is already being collected")
            self._collecting.add(key)
        terminal = False
        release_slot = False
        was_done = handle._future.done()
        try:
            try:
                result = handle._future.result(timeout=timeout)
            except FutureTimeoutError:
                # With a finite timeout, a future that was not done when this
                # attempt started produced a wait timeout.  It may complete
                # before this handler runs, but that later state is not the
                # outcome delivered by this collection attempt.  ``timeout``
                # of None cannot be a wait timeout, so this is a leaf-raised
                # TimeoutError and is terminal.
                terminal = timeout is None or was_done
                if not terminal and handle._future.done():
                    # A leaf TimeoutError can complete during the bounded
                    # wait.  Re-raise it now and consume its handle and lease.
                    leaf_error = handle._future.exception(timeout=0)
                    if leaf_error is not None:
                        terminal = True
                        raise leaf_error
                if not terminal:
                    # Preserve the handle and its slot lease for a later
                    # collection attempt.  This is the only retryable wait
                    # boundary; a leaf-raised TimeoutError remains terminal.
                    raise CollectionWaitTimeout("leaf collection poll timed out") from None
                # A leaf-raised TimeoutError is terminal, but still belongs
                # to the caller as the exact leaf exception.
                raise
            except BaseException as error:
                # Only the future's own exception consumes the handle; caller interrupts must retain its lease.
                terminal = handle._future.done() and handle._future.exception(timeout=0) is error
                raise
            terminal = True
            if not isinstance(result, LeafResult):
                raise LeafResultRejected("leaf runner returned a non-LeafResult")
            result = validate_leaf_result(handle._request, result)
            release_slot = True
            return LeafCompletionContext(result, handle._request, handle._configured, handle._environment)
        finally:
            with self._lock:
                self._collecting.discard(key)
                if terminal and key not in self._collected:
                    self._collected.add(key)
                    self._active -= 1
                    if handle._slot is not None:
                        if release_slot:
                            self._free_slots.append(handle._slot)
                        else:
                            self._retired_slot_ids.add(handle._slot.slot_id)

    def close(self, *, wait: bool = True) -> None:
        """Shut down the local submission pool after outstanding work settles."""

        self._pool.shutdown(wait=wait)


__all__ = [
    "CapacityExceededError",
    "CollectionWaitTimeout",
    "LeafCompletionContext",
    "HandleCollectionInProgressError",
    "InvocationMismatchError",
    "LeafSubmissionError",
    "LocalExecutorError",
    "LocalLeafExecutor",
    "MemberLaunchHandle",
    "UnknownLaunchHandleError",
]
