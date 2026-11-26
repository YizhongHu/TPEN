"""Typed composition of the group-store, core-port, and local leaf seams.

This module is deliberately a small adapter.  It owns the outer binding for a
group and turns the science-process port into the legacy, member-shaped port
accepted by :func:`run_group_controller`; controller fencing and receipt
publication remain in their existing owners.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from .._records import frozen_record
from ..allocations import ConfiguredAllocation, ControllerGenerationV1
from ..contracts import ReadyGroupV2
from ..environments import PreparedPythonEnvironment
from ..storage import DirectoryStore
from ..tasks import ConfiguredRun
from ..recovery import CoreAdmissionPort, CoreAdmissionUnavailable, MemberRecoveryInputV2
from ..programs import MaterializedProgram
from .controller import GroupControllerFactsV1, GroupControllerOutcome, run_group_controller
from .core_port import CoreAdmissionProcessRequest, ScienceProcessCoreAdmissionPort
from .group_store import (
    GroupStoreAuthorityPublished,
    GroupStoreRefused,
    InitialGroupStoreRequest,
    ReentryGroupStoreRequest,
    initial,
    reentry,
)
from .local_executor import LocalLeafExecutor


@frozen_record
class PreparedGroupInput:
    """Immutable owner-process inputs for one admitted group.

    ``env`` is intentionally explicit: it is passed unchanged to both program
    materializers and is never read from process-global state.
    """

    group: ReadyGroupV2
    allocation: ConfiguredAllocation
    controller: ControllerGenerationV1
    member_inputs: tuple[MemberRecoveryInputV2, ...]
    allocation_program: Any
    worker_program: Any
    env: Mapping[str, Any]
    store_root: str
    execution_bundle_digest: str
    deployment_lineage: str
    registration_receipt_digest: str
    claimant_nonce: str
    observed_at_utc: str
    runs: Mapping[str, ConfiguredRun] | None = None
    predecessor_record_digest: str | None = None
    timeout: float = 30.0
    core_timeout: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.group, ReadyGroupV2) or not isinstance(self.allocation, ConfiguredAllocation):
            raise TypeError("group and allocation must be typed records")
        if not isinstance(self.controller, ControllerGenerationV1):
            raise TypeError("controller must be ControllerGenerationV1")
        members = tuple(self.member_inputs)
        if len(members) != len(self.group.members):
            raise ValueError("member_inputs must cover every group member")
        if {item.task_id for item in members} != {item.task_id for item in self.group.members}:
            raise ValueError("member_inputs do not match group members")
        if not isinstance(self.env, Mapping):
            raise TypeError("env must be a mapping")
        if self.runs is not None:
            if not isinstance(self.runs, Mapping) or set(self.runs) != {item.task_id for item in self.group.members}:
                raise ValueError("runs must cover every group member")
            if any(not isinstance(value, ConfiguredRun) for value in self.runs.values()):
                raise TypeError("runs must contain ConfiguredRun records")
        if type(self.timeout) not in (int, float) or self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if type(self.core_timeout) not in (int, float) or self.core_timeout <= 0:
            raise ValueError("core_timeout must be positive")

    def materialize(self) -> tuple[MaterializedProgram, MaterializedProgram]:
        """Materialize allocation and worker programs with the same explicit env."""
        if not callable(getattr(self.allocation_program, "materialize", None)):
            raise GroupStoreRefused("allocation program is not materializable")
        if not callable(getattr(self.worker_program, "materialize", None)):
            raise GroupStoreRefused("worker program is not materializable")
        allocation = self.allocation_program.materialize(env=self.env)
        worker = self.worker_program.materialize(env=self.env)
        return allocation, worker


@frozen_record
class PackedGroupRequest:
    """The closed request consumed by :class:`GroupRuntimeDispatcher`."""

    prepared: PreparedGroupInput
    reentry: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedGroupInput):
            raise TypeError("prepared must be PreparedGroupInput")


class _BoundCorePort(CoreAdmissionPort):
    """Bind immutable task/group facts before entering the science process."""

    def __init__(self, prepared: PreparedGroupInput, port: ScienceProcessCoreAdmissionPort, output_root: Path,
                 worker_environment: PreparedPythonEnvironment) -> None:
        self._prepared, self._port, self._output_root = prepared, port, output_root
        self._worker_environment = worker_environment

    def admit(self, member: MemberRecoveryInputV2) -> Any:
        task = next((item for item in self._prepared.group.members if item.task_id == member.task_id), None)
        if task is None or task.lineage_id != member.lineage_id:
            raise CoreAdmissionUnavailable("core request crosses immutable group member")
        predecessor = None if member.attempt_id is None else {
            "task_id": task.task_id,
            "lineage_id": task.lineage_id,
            "attempt_id": member.attempt_id.value,
        }
        request = CoreAdmissionProcessRequest(
            task.task_id, task.lineage_id,
            None if member.start_receipt is None else member.start_receipt.output_dir,
            task.science_digest,
            None if predecessor is None else {
                "predecessor": predecessor,
                "successor": predecessor,
                "science_input_digest": task.science_digest,
            },
            {"execution_plan": self._prepared.group.execution_plan.to_dict()},
            self._prepared.group.execution_plan.profile_name,
            # The science interpreter uses the worker-owned prepared env.
            self._worker_environment,
        )
        base = self._output_root / member.task_id
        base.mkdir(parents=True, exist_ok=True)
        return self._port.admit(
            request, member,
            stdout_path=base / "stdout.log",
            stderr_path=base / "stderr.log",
            cwd=base,
            timeout=self._prepared.core_timeout,
        )


class GroupRuntimeDispatcher:
    """Compose one immutable group through store, controller, and local leaves."""

    def __init__(self, *, core_port: ScienceProcessCoreAdmissionPort | None = None,
                 controller_runner: Callable[..., Any] = run_group_controller,
                 local_executor_factory: Callable[..., Any] = LocalLeafExecutor,
                 timeout: float = 30.0, core_timeout: float = 30.0) -> None:
        if type(timeout) not in (int, float) or timeout <= 0 or type(core_timeout) not in (int, float) or core_timeout <= 0:
            raise ValueError("timeouts must be positive")
        self.core_port = ScienceProcessCoreAdmissionPort() if core_port is None else core_port
        self.controller_runner = controller_runner
        self.local_executor_factory = local_executor_factory
        self.timeout = timeout
        self.core_timeout = core_timeout

    def run(self, request: PackedGroupRequest) -> GroupControllerOutcome | Any:
        prepared = request.prepared
        allocation_program, worker_program = prepared.materialize()
        if "dispatcher" not in allocation_program:
            raise GroupStoreRefused("allocation materialization lacks dispatcher")
        if not callable(getattr(allocation_program.dispatcher, "run", None)):
            raise GroupStoreRefused("allocation materialization dispatcher is not runnable")
        store_result = reentry(
            ReentryGroupStoreRequest(prepared.group, prepared.execution_bundle_digest,
                                     prepared.deployment_lineage, prepared.controller),
            prepared.store_root,
        ) if request.reentry else initial(
            InitialGroupStoreRequest(prepared.group, prepared.execution_bundle_digest,
                                     prepared.deployment_lineage, prepared.controller),
            prepared.store_root,
        )
        output_root = Path(store_result.invocation_result_path or prepared.store_root) / "core-admission"
        facts = GroupControllerFactsV1(
            prepared.group, prepared.allocation, prepared.registration_receipt_digest,
            prepared.claimant_nonce, prepared.observed_at_utc, prepared.member_inputs,
            controller_generation=prepared.controller.generation,
            predecessor_record_digest=prepared.predecessor_record_digest,
            poll_timeout=prepared.timeout,
        )
        runs = prepared.runs
        if runs is None:
            raise GroupStoreRefused("group runtime requires owner-materialized configured runs")
        executor = self.local_executor_factory(
            prepared.group, prepared.allocation, runs, worker_program.environment,
            capacity=len(prepared.group.members), timeout=prepared.timeout,
        )
        try:
            return self.controller_runner(
                store_result.binding, prepared.allocation.backend_job_id, facts,
                DirectoryStore(prepared.store_root),
                _BoundCorePort(prepared, self.core_port, output_root, worker_program.environment), executor,
            )
        finally:
            close = getattr(executor, "close", None)
            if callable(close):
                close()


__all__ = ["GroupRuntimeDispatcher", "PackedGroupRequest", "PreparedGroupInput"]
