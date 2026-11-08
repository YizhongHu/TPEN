"""Shared lifecycle contract for attached executor dispatchers.

Placement mutant controls include N1 (preserve declared devices for preset
device slots), H1 (refuse controller CPU-outside-affinity placements), H2
(turn invalid receipt construction into typed refusal evidence), and F12
(refused receipts use a positive sentinel rather than emitting PID zero).
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import math
import os
import re
from typing import Any

from .._codec import digest
from ..contracts import ReadyGroup
from ..environments import PreparedPythonEnvironment
from ..lifecycle.attached import AttachedAllocation
from ..results import RunManifest, SciencePlacementReceipt
from ..placement import (
    PLACEMENT_OUTPUT_ENV,
    PLACEMENT_REFUSAL_ENV,
    PlacementRefusal,
    canonical_host,
    parse_cpu_set,
    placement_environment,
    read_child_placement,
    slots_from_allocation,
)
from ..runners import CONTEXT_ENV_KEY_PATTERN, CHILD_CONTEXT_KEYS_ENV, CoreRunEvidence, EvidenceUnavailable
from ..tasks import ConfiguredRun


_ACTIVE_DISPATCH: ContextVar[bool] = ContextVar("tpen_executor_dispatch", default=False)


def _lexical(path: str | os.PathLike[str]) -> str:
    """Return an absolute normalized path without resolving symlinks."""

    return os.path.abspath(os.path.normpath(os.fspath(path)))


def _inside(root: str, path: str) -> bool:
    """Return whether ``path`` is lexically below ``root``."""

    return path == root or path.startswith(root + os.sep)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for lifecycle receipts."""

    return datetime.now(timezone.utc).isoformat()


def _evidence_payload(evidence: CoreRunEvidence) -> dict[str, Any]:
    """Return the canonical JSON-shaped form of one decoded evidence record."""

    if not isinstance(evidence, CoreRunEvidence):
        raise TypeError("evidence reader must return CoreRunEvidence")
    return {
        "schema": evidence.schema,
        "run_dir": evidence.run_dir,
        "verdict": evidence.verdict,
        "run_status": evidence.run_status,
        "checkpoint": evidence.checkpoint,
        "refusal": evidence.refusal,
    }


def _member_paths(fence: Any) -> tuple[str, str, str, str, str, str]:
    """Validate and derive all dispatcher-owned paths for one attempt."""

    root = _lexical(fence.root)
    output_dir = _lexical(fence.output_dir)
    if not _inside(root, output_dir) or output_dir == root:
        raise RuntimeError("attempt output directory escapes the allocation store root")
    core_dir = _lexical(os.path.join(output_dir, "core"))
    if not _inside(output_dir, core_dir):
        raise RuntimeError("core run directory escapes the attempt fence")
    stdout_path = _lexical(os.path.join(output_dir, "stdout.log"))
    stderr_path = _lexical(os.path.join(output_dir, "stderr.log"))
    evidence_stdout = _lexical(os.path.join(output_dir, "evidence.stdout.log"))
    evidence_stderr = _lexical(os.path.join(output_dir, "evidence.stderr.log"))
    for path in (stdout_path, stderr_path, evidence_stdout, evidence_stderr):
        if not _inside(output_dir, path):
            raise RuntimeError("dispatcher log path escapes the attempt fence")
    return output_dir, core_dir, stdout_path, stderr_path, evidence_stdout, evidence_stderr


def _status(process_outcome: Any) -> str:
    """Map one reaped child outcome to the closed operational vocabulary."""

    if type(getattr(process_outcome, "timed_out", None)) is not bool:
        raise TypeError("process outcome has an invalid timed_out flag")
    return "timed_out" if process_outcome.timed_out else (
        "completed" if process_outcome.returncode == 0 else "failed"
    )


def _child_context_keys(allocation: Any) -> tuple[str, ...]:
    """Return only validated-context environment names for child observation."""

    keys = {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}
    facts = getattr(allocation, "facts", {})
    attached = facts.get("attached_context") if isinstance(facts, Mapping) else None
    context = attached.get("context") if isinstance(attached, Mapping) else None
    declared = context.get("environment") if isinstance(context, Mapping) else None
    if isinstance(declared, Mapping):
        keys.update(
            key
            for key in declared
            if type(key) is str and re.fullmatch(CONTEXT_ENV_KEY_PATTERN, key) is not None
        )
    visibility = context.get("visibility_variable") if isinstance(context, Mapping) else None
    if type(visibility) is str and re.fullmatch(CONTEXT_ENV_KEY_PATTERN, visibility) is not None:
        keys.add(visibility)
    return tuple(sorted(keys))


def _with_child_context_keys(environment: Any, allocation: Any) -> Any:
    """Pass the observation allowlist to a prepared science environment."""

    if not isinstance(environment, PreparedPythonEnvironment):
        return environment
    overlay = dict(environment.env)
    overlay[CHILD_CONTEXT_KEYS_ENV] = json.dumps(_child_context_keys(allocation), separators=(",", ":"))
    return PreparedPythonEnvironment(environment.python, environment.setup, overlay)


def _with_child_placement(
    environment: Any,
    allocation: Any,
    fence: Any,
    group: ReadyGroup,
    member: Any,
    configured: ConfiguredRun,
) -> Any:
    """Overlay the one concrete slot and fenced identities for the child.

    Legacy workers without a declared slot retain their existing behavior.
    Once a slot is present, all placement fields are explicit and immutable
    from the controller's point of view; only the science child applies CPU
    affinity in :data:`RUN_SHIM` before importing core.

    The slot environment is merged into the prepared child environment so
    legacy S1 logical labels remain visible to the child; placement-specific
    enforcement is enabled only by the explicit sidecar output path.

    CPU-only slots intentionally retain an empty declared-device field.
    """

    if not isinstance(environment, PreparedPythonEnvironment):
        return environment
    values = dict(environment.env)
    slot_id = values.get("TPEN_SLOT_ID")
    host = values.get("TPEN_SLOT_NODE")
    cpus = values.get("TPEN_SLOT_CPUS")
    devices = values.get("TPEN_PLACEMENT_DECLARED_DEVICES", "")
    if not devices:
        # Parsl preset bindings carry the entitlement as CUDA visibility;
        # recover it without turning CPU-only slots into device slots.
        devices = values.get("CUDA_VISIBLE_DEVICES", "")
    if not (slot_id and host and cpus):
        slots = slots_from_allocation(allocation)
        if len(slots) != 1:
            # A local worker with no slot is a legacy path; a multi-slot
            # allocation must be resolved by the backend worker (Parsl).
            return environment
        slot = slots[0]
        slot_id, host, cpus = slot.slot_id, slot.node, ",".join(str(cpu) for cpu in slot.cpus)
        devices = ",".join(slot.devices)
        values.update(slot.env)
    declared_cpus = parse_cpu_set(cpus)
    controller_refusal: dict[str, str] | None = None
    try:
        permitted_cpus = set(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        controller_refusal = {
            "code": "placement-controller-affinity-unavailable",
            "reason": f"controller CPU affinity is unavailable: {error}",
        }
    else:
        if not set(declared_cpus).issubset(permitted_cpus):
            controller_refusal = {
                "code": "placement-cpu-outside-controller-affinity",
                "reason": "declared slot CPUs are not all permitted to this controller",
            }
    payload_digest = digest({"config": configured.config, "overrides": list(configured.overrides)})
    placement_values = {
        PLACEMENT_OUTPUT_ENV: os.path.join(fence.output_dir, "child-placement.json"),
        "TPEN_PLACEMENT_OUTPUT_DIR": fence.output_dir,
        "TPEN_PLACEMENT_TASK_ID": member.task_id,
        "TPEN_PLACEMENT_ATTEMPT_ID": fence.attempt_id.value,
        "TPEN_PLACEMENT_GROUP_DIGEST": group.identity,
        "TPEN_PLACEMENT_ALLOCATION_ID": allocation.allocation_id.value,
        "TPEN_PLACEMENT_BACKEND_JOB_ID": allocation.backend_job_id.value,
        "TPEN_PLACEMENT_SLOT_ID": slot_id,
        "TPEN_PLACEMENT_DECLARED_HOST": host,
        "TPEN_PLACEMENT_DECLARED_CPUS": ",".join(str(cpu) for cpu in declared_cpus),
        "TPEN_PLACEMENT_DECLARED_DEVICES": devices,
        "TPEN_PLACEMENT_PAYLOAD_DIGEST": payload_digest,
    }
    if controller_refusal is not None:
        placement_values[PLACEMENT_REFUSAL_ENV] = json.dumps(controller_refusal, sort_keys=True)
    values = placement_environment(values, **placement_values)
    return PreparedPythonEnvironment(environment.python, environment.setup, values)


def _placement_refusal_from_environment(environment: Any) -> PlacementRefusal | None:
    """Decode a controller-side placement refusal carried to completion."""

    if not isinstance(environment, PreparedPythonEnvironment):
        return None
    encoded = environment.env.get(PLACEMENT_REFUSAL_ENV)
    if encoded is None:
        return None
    try:
        refusal = json.loads(encoded)
        if type(refusal) is not dict or set(refusal) != {"code", "reason"}:
            raise ValueError("placement refusal has invalid keys")
        return PlacementRefusal(refusal["code"], refusal["reason"])
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return PlacementRefusal("placement-refusal-malformed", str(error))


def _placement_receipt(
    observation: Any,
    allocation: Any,
    fence: Any,
    group: ReadyGroup,
    member: Any,
    configured: ConfiguredRun,
    environment: Any,
    process_outcome: Any = None,
) -> SciencePlacementReceipt:
    """Validate a quiescent child sidecar and convert it to a receipt."""

    if not isinstance(environment, PreparedPythonEnvironment):
        raise PlacementRefusal("placement-environment", "child environment is not a prepared Python environment")
    values = environment.env
    expected_payload = digest({"config": configured.config, "overrides": list(configured.overrides)})
    checks = {
        "task": observation.task_id == member.task_id,
        "attempt": observation.attempt_id == fence.attempt_id.value,
        "group": observation.group_digest == group.identity,
        "allocation": observation.allocation_id == allocation.allocation_id.value,
        "backend": observation.backend_job_id == allocation.backend_job_id.value,
        "slot": observation.slot_id == values.get("TPEN_SLOT_ID"),
        "payload": observation.payload_digest == expected_payload,
        "output": observation.output_dir == fence.output_dir,
    }
    refusal = observation.refusal
    if not all(checks.values()):
        failed = ",".join(name for name, passed in checks.items() if not passed)
        refusal = {"code": "placement-identity-mismatch", "reason": f"child sidecar identity mismatch: {failed}"}
    if observation.proc_start_ticks <= 0:
        refusal = refusal or {"code": "placement-start-token-missing", "reason": "child start ticks are missing"}
    expected_pid = getattr(process_outcome, "pid", None)
    expected_ticks = getattr(process_outcome, "proc_start_ticks", None)
    if observation.pid != expected_pid or observation.proc_start_ticks != expected_ticks:
        refusal = refusal or {
            "code": "placement-generation-mismatch",
            "reason": "child sidecar PID/start token does not match the spawned live process",
        }
    expected_host = values.get("TPEN_SLOT_NODE")
    expected_cpus = parse_cpu_set(values.get("TPEN_SLOT_CPUS", ""))
    host_match = expected_host is not None and canonical_host(observation.host) == canonical_host(expected_host)
    affinity_match = expected_cpus == observation.affinity
    if not host_match or not affinity_match:
        refusal = refusal or {
            "code": "placement-affinity-mismatch",
            "reason": f"child host/affinity differs from declared slot ({observation.host!r}, {observation.affinity!r})",
        }
    return SciencePlacementReceipt(
        member.task_id,
        fence.attempt_id,
        group.identity,
        allocation.allocation_id.value,
        allocation.backend_job_id.value,
        values.get("TPEN_SLOT_ID", observation.slot_id),
        expected_payload,
        fence.output_dir,
        {
            "host": observation.host,
            "pid": observation.pid,
            "proc_start_ticks": observation.proc_start_ticks,
            "affinity": list(observation.affinity),
            "thread_settings": observation.thread_settings,
            "cuda_visible_devices": observation.cuda_visible_devices,
            "observed_at_utc": observation.observed_at_utc,
        },
        {
            "host": expected_host,
            "cpus": list(expected_cpus),
            "devices": [item for item in values.get("TPEN_PLACEMENT_DECLARED_DEVICES", "").split(",") if item],
        },
        {
            "status": "unavailable",
            "reason": "active GPU sampler is owned by S2b-2",
            "method": "s2b-1-child-generation-no-sampler",
            "started_at_utc": observation.observed_at_utc,
            "finished_at_utc": _utc_now(),
            "samples": [],
            "active_uuids": [],
            "active_pci": [],
        },
        {
            "host": "matched" if host_match else "mismatch",
            "affinity": "matched" if affinity_match else "mismatch",
            "device": "unavailable",
        },
        refusal,
    )


def _placement_refusal_receipt(
    observation_error: PlacementRefusal,
    allocation: Any,
    fence: Any,
    group: ReadyGroup,
    member: Any,
    configured: ConfiguredRun,
    environment: Any,
    process_outcome: Any,
) -> SciencePlacementReceipt:
    """Encode a missing or malformed child sidecar as explicit refusal evidence."""

    if not isinstance(environment, PreparedPythonEnvironment):
        raise PlacementRefusal("placement-environment", "child environment is not a prepared Python environment")
    values = environment.env
    try:
        declared_cpus = [int(item) for item in values.get("TPEN_SLOT_CPUS", "").split(",") if item]
    except ValueError:
        declared_cpus = []
    if not declared_cpus:
        declared_cpus = [0]
    declared_devices = [item for item in values.get("TPEN_PLACEMENT_DECLARED_DEVICES", "").split(",") if item]
    pid = getattr(process_outcome, "pid", None)
    ticks = getattr(process_outcome, "proc_start_ticks", None)
    return SciencePlacementReceipt(
        member.task_id,
        fence.attempt_id,
        group.identity,
        allocation.allocation_id.value,
        allocation.backend_job_id.value,
        values.get("TPEN_SLOT_ID", "unavailable"),
        digest({"config": configured.config, "overrides": list(configured.overrides)}),
        fence.output_dir,
        {
            "host": values.get("TPEN_SLOT_NODE", "unknown"),
            "pid": pid if type(pid) is int and pid > 0 else 1,
            "proc_start_ticks": ticks if type(ticks) is int and ticks >= 0 else 0,
            "affinity": [],
            "thread_settings": {name: values.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "cuda_visible_devices": values.get("CUDA_VISIBLE_DEVICES"),
            "observed_at_utc": _utc_now(),
        },
        {
            "host": values.get("TPEN_SLOT_NODE", "unknown"),
            "cpus": declared_cpus,
            "devices": declared_devices,
        },
        {
            "status": "unavailable",
            "reason": "active GPU sampler is owned by S2b-2",
            "method": "s2b-1-child-generation-no-sampler",
            "started_at_utc": _utc_now(),
            "finished_at_utc": _utc_now(),
            "samples": [],
            "active_uuids": [],
            "active_pci": [],
        },
        {"host": "refused", "affinity": "refused", "device": "unavailable"},
        observation_error.to_dict(),
    )


def _ensure_placement_refusal_environment(
    environment: Any,
    allocation: Any,
    fence: Any,
    group: ReadyGroup,
    member: Any,
    configured: ConfiguredRun,
    refusal: PlacementRefusal,
) -> Any:
    """Add a publication target when placement setup failed before overlay."""

    if not isinstance(environment, PreparedPythonEnvironment):
        return environment
    if PLACEMENT_OUTPUT_ENV in environment.env:
        return environment
    values = dict(environment.env)
    slot_id = values.get("TPEN_SLOT_ID", "unavailable")
    host = values.get("TPEN_SLOT_NODE", "unknown")
    cpus = values.get("TPEN_SLOT_CPUS", "0")
    try:
        slots = slots_from_allocation(allocation)
    except (TypeError, ValueError):
        slots = ()
    if len(slots) == 1:
        slot = slots[0]
        slot_id = slot.slot_id
        host = slot.node
        cpus = ",".join(str(cpu) for cpu in slot.cpus)
        values.update(slot.env)
    try:
        declared_cpus = parse_cpu_set(cpus)
    except PlacementRefusal:
        declared_cpus = (0,)
    placement_values = {
        PLACEMENT_OUTPUT_ENV: os.path.join(fence.output_dir, "child-placement.json"),
        "TPEN_PLACEMENT_OUTPUT_DIR": fence.output_dir,
        "TPEN_PLACEMENT_TASK_ID": member.task_id,
        "TPEN_PLACEMENT_ATTEMPT_ID": fence.attempt_id.value,
        "TPEN_PLACEMENT_GROUP_DIGEST": group.identity,
        "TPEN_PLACEMENT_ALLOCATION_ID": allocation.allocation_id.value,
        "TPEN_PLACEMENT_BACKEND_JOB_ID": allocation.backend_job_id.value,
        "TPEN_PLACEMENT_SLOT_ID": slot_id,
        "TPEN_PLACEMENT_DECLARED_HOST": host,
        "TPEN_PLACEMENT_DECLARED_CPUS": ",".join(str(cpu) for cpu in declared_cpus),
        "TPEN_PLACEMENT_DECLARED_DEVICES": values.get("TPEN_PLACEMENT_DECLARED_DEVICES", ""),
        "TPEN_PLACEMENT_PAYLOAD_DIGEST": digest({"config": configured.config, "overrides": list(configured.overrides)}),
        PLACEMENT_REFUSAL_ENV: json.dumps(refusal.to_dict(), sort_keys=True),
    }
    values = placement_environment(values, **placement_values)
    return PreparedPythonEnvironment(environment.python, environment.setup, values)


def _prepare_child_placement(
    environment: Any,
    allocation: Any,
    fence: Any,
    group: ReadyGroup,
    member: Any,
    configured: ConfiguredRun,
) -> Any:
    """Prepare placement and preserve a refusal before backend launch.

    A controller-side refusal can be raised while constructing the overlay,
    before the overlay containing its publication path is returned.  Normalize
    that exception at the refusal point so Local and Parsl share the same
    durable receipt path.
    """

    try:
        return _with_child_placement(environment, allocation, fence, group, member, configured)
    except PlacementRefusal as refusal:
        return _ensure_placement_refusal_environment(
            environment, allocation, fence, group, member, configured, refusal
        )


class DispatcherContract:
    """Implement admission, fencing, outcome, evidence, and segment semantics.

    Concrete dispatchers only provide :meth:`_execute_process`.  This keeps
    backend framework state out of the durable executor contract and ensures
    that Local and Parsl dispatchers publish identical records.
    """

    __slots__ = ()

    worker: Any

    def _worker_environment(self, materialized: Any) -> Any:
        """Return the prepared environment used by the child and inspector."""

        return materialized["environment"]

    def _execute_process(
        self,
        materialized: Any,
        environment: Any,
        args: tuple[str, ...],
        *,
        cwd: str,
        stdout_path: str,
        stderr_path: str,
        timeout: float | None,
    ) -> Any:
        """Run one child in the backend represented by ``self``."""

        process = materialized["process"]
        return process.run(
            environment,
            args,
            cwd=cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=timeout,
        )

    def validate(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        """Perform common admission without touching backend or store state."""

        self._validate_inputs(group, runs, lifecycle, env, timeout)

    def _begin_dispatch(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None,
        timeout: float | None,
    ) -> Any:
        """Validate inputs and reserve the process-local dispatch slot."""

        if _ACTIVE_DISPATCH.get():
            raise RuntimeError("dispatcher.run is not reentrant")
        self.validate(group, runs, lifecycle, env, timeout)
        return _ACTIVE_DISPATCH.set(True)

    @staticmethod
    def _end_dispatch(token: Any) -> None:
        """Release a dispatch slot acquired by :meth:`_begin_dispatch`."""

        _ACTIVE_DISPATCH.reset(token)

    def run(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        *,
        _admitted: bool = False,
    ) -> RunManifest:
        """Dispatch every admitted member and return its durable group manifest.

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

        Notes
        -----
        Every begun fence receives one best-effort finish on cancellation.
        Child return code and timeout determine operational status; backend
        future success is never used as a completion signal.

        Admission is completed before the first lifecycle write. Ordinary
        member failures are recorded and processing continues to later members.
        """

        if _admitted:
            if not _ACTIVE_DISPATCH.get():
                raise RuntimeError("admitted dispatch requires an active dispatch slot")
            return self._run_admitted(group, runs, lifecycle, env, timeout)
        token = self._begin_dispatch(group, runs, lifecycle, env, timeout)
        try:
            return self._run_admitted(group, runs, lifecycle, env, timeout)
        finally:
            self._end_dispatch(token)

    def _run_admitted(
        self,
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None,
        timeout: float | None,
    ) -> RunManifest:
        """Run the shared lifecycle after admission and guard setup."""

        begin_error: Exception | None = None
        finish_error: Exception | None = None
        for member in group.members:
            configured = runs[member.task_id]
            try:
                fence = lifecycle.begin(group, member.task_id, _utc_now())
            except Exception as error:
                if begin_error is None:
                    begin_error = error
                continue

            materialized: Any = None
            environment: Any = None
            try:
                materialized = self.worker.materialize(env)
                environment = self._worker_environment(materialized)
                environment = _with_child_context_keys(environment, lifecycle.allocation)
                environment = _prepare_child_placement(
                    environment,
                    lifecycle.allocation,
                    fence,
                    group,
                    member,
                    configured,
                )
                placement_refusal = _placement_refusal_from_environment(environment)
                if placement_refusal is not None:
                    raise placement_refusal
                process = materialized["process"]
                materialized["runner"]
                evidence = materialized["evidence"]
                if not callable(getattr(process, "run", None)):
                    raise TypeError("materialized worker process lacks run()")
                if not callable(getattr(evidence, "read", None)):
                    raise TypeError("materialized worker evidence reader lacks read()")
                output_dir, _core_dir, stdout_path, stderr_path, _evidence_stdout, _evidence_stderr = _member_paths(fence)
                args = (*configured.command().args, f"run.root={output_dir}", "run.layout=flat", "run.run_id=core")
                outcome = self._execute_process(
                    materialized,
                    environment,
                    args,
                    cwd=output_dir,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    timeout=timeout,
                )
            except BaseException as error:
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
                    error=error,
                )
                if completion_error is not None:
                    if isinstance(error, Exception) and finish_error is None:
                        finish_error = completion_error
                    else:
                        try:
                            error.add_note(f"finish error during dispatch: {completion_error!r}")
                        except BaseException:
                            pass
                if not isinstance(error, Exception):
                    raise
                continue
            completion_error = self._complete_member(
                group,
                member,
                configured,
                lifecycle,
                fence,
                materialized,
                environment,
                outcome,
                timeout,
            )
            if completion_error is not None and finish_error is None:
                finish_error = completion_error

        if begin_error is not None:
            if finish_error is not None:
                begin_error.add_note(f"finish error after deferred begin failure: {finish_error!r}")
            raise begin_error
        if finish_error is not None:
            raise finish_error
        return lifecycle.store.manifest(group)

    def _complete_member(
        self,
        group: ReadyGroup,
        member: Any,
        configured: ConfiguredRun,
        lifecycle: AttachedAllocation,
        fence: Any,
        materialized: Any,
        environment: Any,
        outcome: Any,
        timeout: float | None,
        *,
        error: BaseException | None = None,
    ) -> Exception | None:
        """Inspect and publish one quiescent child from the controller.

        Backend workers return only a plain process outcome.  This method is
        deliberately called by the owner/controller thread after the backend
        future and child process have both quiesced; it is the only shared
        completion path that writes finish and accepted-segment receipts.
        """

        invalid_outcome = False
        if error is None:
            try:
                status = _status(outcome)
            except Exception:
                status = "launch_failed"
                invalid_outcome = True
        else:
            # Ordinary launch/worker errors are durable launch failures; a
            # BaseException is a cancellation and remains an operational
            # failure while its original exception propagates.
            status = "launch_failed" if isinstance(error, Exception) else "failed"
        result_digest: str | None = None
        segment_evidence: CoreRunEvidence | None = None
        inspection_error: BaseException | None = None
        if error is None and not invalid_outcome:
            try:
                evidence = materialized["evidence"]
                process = materialized["process"]
                output_dir, core_dir, _stdout, _stderr, evidence_stdout, evidence_stderr = _member_paths(fence)
                if not callable(getattr(evidence, "read", None)):
                    raise TypeError("materialized worker evidence reader lacks read()")
                decoded = evidence.read(
                    process,
                    environment,
                    core_dir,
                    cwd=output_dir,
                    stdout_path=evidence_stdout,
                    stderr_path=evidence_stderr,
                    timeout=timeout,
                )
                result_digest = digest(_evidence_payload(decoded))
                segment_evidence = decoded
            except EvidenceUnavailable:
                pass
            except Exception:
                status = "launch_failed"
            except BaseException as caught:
                # Preserve a known process timeout while still terminalizing
                # the begun fence before propagating owner cancellation.
                if status != "timed_out":
                    status = "failed"
                inspection_error = caught
        try:
            lifecycle.finish(fence, status, _utc_now(), result_digest)
        except Exception as finish_error:
            if inspection_error is None:
                return finish_error
            try:
                inspection_error.add_note(f"finish error during inspection cancellation: {finish_error!r}")
            except BaseException:
                pass
        placement_path = None
        if isinstance(environment, PreparedPythonEnvironment):
            placement_path = environment.env.get("TPEN_PLACEMENT_OUTPUT")
        if placement_path is not None:
            try:
                requested_refusal = _placement_refusal_from_environment(environment)
                if requested_refusal is not None:
                    raise requested_refusal
                observation = read_child_placement(placement_path)
                receipt = _placement_receipt(
                    observation,
                    lifecycle.allocation,
                    fence,
                    group,
                    member,
                    configured,
                    environment,
                    outcome,
                )
            except PlacementRefusal as placement_error:
                receipt = _placement_refusal_receipt(
                    placement_error,
                    lifecycle.allocation,
                    fence,
                    group,
                    member,
                    configured,
                    environment,
                    outcome,
                )
            except (TypeError, ValueError) as placement_error:
                receipt = _placement_refusal_receipt(
                    PlacementRefusal("placement-receipt-invalid", str(placement_error)),
                    lifecycle.allocation,
                    fence,
                    group,
                    member,
                    configured,
                    environment,
                    outcome,
                )
            try:
                lifecycle.store.publish_placement(fence, receipt)
            except Exception as placement_error:
                return placement_error
        if inspection_error is not None:
            raise inspection_error
        if (
            status == "completed"
            and segment_evidence is not None
            and segment_evidence.verdict in {"completed", "incomplete"}
            and segment_evidence.checkpoint is not None
        ):
            existing_segments = lifecycle.store.manifest(group).segments
            if not any(segment.task_id == fence.task_id for segment in existing_segments):
                try:
                    lifecycle.store.accept_segment(fence, segment_evidence, None)
                except Exception as error:
                    return error
        return None

    @staticmethod
    def _validate_inputs(
        group: ReadyGroup,
        runs: Mapping[str, ConfiguredRun],
        lifecycle: AttachedAllocation,
        env: Mapping[str, Any] | None,
        timeout: float | None,
    ) -> None:
        """Validate all admission inputs before touching the lifecycle store."""

        if not isinstance(group, ReadyGroup):
            raise TypeError("group must be ReadyGroup")
        if isinstance(runs, (str, bytes, bytearray)) or not isinstance(runs, Mapping):
            raise ValueError("runs must be a mapping of task IDs to ConfiguredRun")
        if not callable(getattr(lifecycle, "begin", None)) or not callable(getattr(lifecycle, "finish", None)):
            raise TypeError("lifecycle must provide begin() and finish()")
        if not hasattr(lifecycle, "store") or not callable(getattr(lifecycle.store, "manifest", None)):
            raise TypeError("lifecycle must expose a manifest-capable store")
        if env is not None and (isinstance(env, (str, bytes, bytearray)) or not isinstance(env, Mapping)):
            raise TypeError("env must be a mapping or None")
        if timeout is not None and (
            type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive or None")
        store_root = getattr(lifecycle.store, "root", None)
        if isinstance(store_root, (str, os.PathLike)) and "${" in os.fspath(store_root):
            raise ValueError("store root must not contain OmegaConf interpolation syntax '${'")
        expected = {member.task_id for member in group.members}
        actual = set(runs)
        if actual != expected:
            raise ValueError("runs keys must match group member task IDs exactly")
        for task_id, configured in runs.items():
            if not isinstance(configured, ConfiguredRun):
                raise TypeError("runs values must be ConfiguredRun records")
            if configured.id != task_id:
                raise ValueError("ConfiguredRun.id must match its mapping task ID")


__all__ = ["DispatcherContract"]
