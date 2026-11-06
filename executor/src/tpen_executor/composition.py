"""Admission and owner-process composition for the public executor CLI."""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from pathlib import Path
import stat
from typing import Any

from ._codec import decode_json
from .allocations import ConfiguredAllocation
from .configuration import StagedHydra
from .contracts import ReadyGroup
from .dispatch import LocalDispatcher, ParslDispatcher, ParslUnavailableError
from .environments import PreparedPythonEnvironment
from .lifecycle import AttachedAllocation
from .placement import BindingError, binding_from_allocation
from .programs import WorkerProgram
from .results import RunManifest
from .storage import DirectoryStore, _write_new
from .tasks import ConfiguredRunTable
from ._records import frozen_record


class CLIRefusal(ValueError):
    """A named input or admission refusal that maps to CLI exit 2."""

    def __init__(self, code: str, reason: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("CLI refusal code must be a non-empty string")
        if type(reason) is not str or not reason:
            raise TypeError("CLI refusal reason must be a non-empty string")
        self.code = code
        self.reason = reason
        super().__init__(reason)

    def to_dict(self) -> dict[str, str]:
        """Return the public CLI error envelope."""

        return {"schema": "tpen-executor/cli-error-v1", "code": self.code, "reason": self.reason}


def _path(path: str | os.PathLike[str], name: str) -> Path:
    """Validate a caller-named path without resolving symlinks."""

    value = os.fspath(path)
    if type(value) is not str or not value:
        raise CLIRefusal("invalid-path", f"{name} must be a non-empty path")
    return Path(value)


def _read_regular(path: Path, name: str) -> bytes:
    """Read one regular, non-symlink file exactly once."""

    try:
        info = path.lstat()
    except OSError as error:
        raise CLIRefusal("input-missing", f"{name} could not be read: {error}") from error
    if stat.S_ISLNK(info.st_mode):
        raise CLIRefusal("input-symlink", f"{name} must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise CLIRefusal("input-not-file", f"{name} must be a regular file")
    try:
        return path.read_bytes()
    except (OSError, UnicodeError) as error:
        raise CLIRefusal("input-io-error", f"{name} could not be read: {error}") from error


def _decode(path: Path, name: str) -> Mapping[str, Any]:
    """Read and strictly decode one JSON object."""

    try:
        return decode_json(_read_regular(path, name))
    except CLIRefusal:
        raise
    except (RecursionError, TypeError, ValueError, UnicodeError) as error:
        raise CLIRefusal("invalid-json", f"{name} is not one strict JSON object: {error}") from error


def _validate_worker(worker: WorkerProgram) -> None:
    """Materialize a worker once to validate its executor-owned shape."""

    if not isinstance(worker, WorkerProgram):
        raise CLIRefusal("worker-shape", "worker-config must contain a WorkerProgram marker")
    try:
        materialized = worker.materialize()
    except Exception as error:
        raise CLIRefusal("worker-shape", f"worker-config could not be materialized: {error}") from error
    required = {"environment", "process", "runner", "evidence"}
    if set(materialized.slots) != required:
        raise CLIRefusal("worker-shape", f"worker slots must be exactly {sorted(required)!r}")
    if not isinstance(materialized.environment, PreparedPythonEnvironment):
        raise CLIRefusal("worker-shape", "worker environment must be PreparedPythonEnvironment")
    if not callable(getattr(materialized.process, "run", None)):
        raise CLIRefusal("worker-shape", "worker process must provide run()")
    if not callable(getattr(materialized.evidence, "read", None)):
        raise CLIRefusal("worker-shape", "worker evidence must provide read()")


def load_worker(path: str | os.PathLike[str]) -> WorkerProgram:
    """Load an unresolved worker marker from a YAML config file.

    Parameters
    ----------
    path : str or os.PathLike
        YAML file containing the executor worker marker.

    Returns
    -------
    WorkerProgram
        Unresolved worker handle whose concrete slots are admitted once.

    Raises
    ------
    CLIRefusal
        If the file is unreadable, malformed, or has the wrong worker shape.
    """

    config_path = _path(path, "worker-config")
    try:
        source = _read_regular(config_path, "worker-config").decode("utf-8")
        staged = StagedHydra.stage_source(source)
    except CLIRefusal:
        raise
    except Exception as error:
        raise CLIRefusal("worker-config", f"worker-config is invalid: {error}") from error
    worker = staged.programs.get("worker")
    if not isinstance(worker, WorkerProgram):
        raise CLIRefusal("worker-shape", "worker-config must contain a top-level worker marker")
    _validate_worker(worker)
    return worker


def _validate_group_and_runs(group: ReadyGroup, table: ConfiguredRunTable) -> None:
    """Validate sidecar identity, order, and every configured command."""

    if table.group_digest != group.identity:
        raise CLIRefusal("group-digest-mismatch", "run-table group_digest does not match ReadyGroup.identity")
    member_ids = tuple(member.task_id for member in group.members)
    run_ids = tuple(run.id for run in table.runs)
    if run_ids != member_ids:
        raise CLIRefusal("run-members-mismatch", "run-table ids must exactly match ordered ReadyGroup members")
    for task_id in member_ids:
        try:
            DirectoryStore._validate_task_id(task_id)
        except (TypeError, ValueError) as error:
            raise CLIRefusal("unsafe-task-id", f"task id {task_id!r} is not a safe store namespace: {error}") from error


def _reserve_store(path: str | os.PathLike[str]) -> DirectoryStore:
    """Atomically create the previously absent store root."""

    root = Path(os.path.abspath(os.path.normpath(os.fspath(path))))
    try:
        root.mkdir()
    except FileExistsError as error:
        raise CLIRefusal("store-exists", f"store root already exists: {root}") from error
    except OSError as error:
        raise CLIRefusal("store-create", f"store root could not be created: {error}") from error
    try:
        return DirectoryStore(root)
    except Exception as error:
        raise CLIRefusal("store-invalid", f"store root is not usable: {error}") from error


@frozen_record
class DispatchPlan:
    """Fully admitted executor inputs, with all writes still pending.

    Parameters
    ----------
    group : ReadyGroup
        Ordered group admitted for dispatch.
    table : ConfiguredRunTable
        Sidecar commands bound to the group's ordered members.
    allocation : ConfiguredAllocation
        Backend allocation identity owning the store.
    worker : WorkerProgram
        Unresolved worker handle with validated concrete slots.
    dispatcher : object
        Local or optional-Parsl dispatcher selected by admission.
    lifecycle : AttachedAllocation
        Store-owning lifecycle used for fenced receipts.
    timeout : float or None
        Positive child/evidence timeout, if supplied.

    Returns
    -------
    DispatchPlan
        Frozen record carrying the admitted dispatch inputs.

    Raises
    ------
    TypeError, ValueError
        If a field does not satisfy the frozen dispatch-plan contract.
    """

    group: ReadyGroup
    table: ConfiguredRunTable
    allocation: ConfiguredAllocation
    worker: WorkerProgram
    dispatcher: Any
    lifecycle: AttachedAllocation
    timeout: float | None


def admit(
    input_path: str | os.PathLike[str],
    store_path: str | os.PathLike[str],
    allocation_path: str | os.PathLike[str],
    worker_config_path: str | os.PathLike[str],
    runs_path: str | os.PathLike[str],
    *,
    dispatcher_name: str = "local",
    timeout: float | None = None,
) -> DispatchPlan:
    """Decode and preflight a dispatch without beginning an attempt.

    Parameters
    ----------
    input_path, store_path, allocation_path, worker_config_path, runs_path : str or os.PathLike
        Input records, fresh store root, allocation record, worker YAML, and
        configured-runs sidecar.
    dispatcher_name : {"local", "parsl"}, default="local"
        Attached dispatcher backend.
    timeout : float or None, optional
        Positive timeout passed to child and evidence operations.

    Returns
    -------
    DispatchPlan
        Fully admitted inputs with a reserved, empty store.

    Raises
    ------
    CLIRefusal
        If any input fails decoding, validation, worker-shape, or reservation
        admission before a dispatch attempt begins.
    """

    if dispatcher_name not in {"local", "parsl"}:
        raise CLIRefusal("invalid-dispatcher", f"unknown dispatcher {dispatcher_name!r}")
    if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
        raise CLIRefusal("invalid-timeout", "timeout must be finite and strictly positive")

    input_file = _path(input_path, "input")
    if input_file.name == "executor_input_diagnostics.json":
        raise CLIRefusal("diagnostic-input", "executor_input_diagnostics.json is not dispatchable input")
    diagnostic = input_file.with_name("executor_input_diagnostics.json")
    if os.path.lexists(diagnostic):
        raise CLIRefusal("diagnostic-present", "current executor_input_diagnostics.json blocks dispatch")
    raw_group = _decode(input_file, "input")
    try:
        group = ReadyGroup.from_mapping(raw_group)
    except (RecursionError, TypeError, ValueError) as error:
        raise CLIRefusal("invalid-ready-group", f"input is not a valid ReadyGroup: {error}") from error
    if not group.members:
        raise CLIRefusal("missing-ready-group", "input must contain a non-empty ReadyGroup")

    raw_allocation = _decode(_path(allocation_path, "allocation"), "allocation")
    try:
        allocation = ConfiguredAllocation.from_mapping(raw_allocation)
    except (RecursionError, TypeError, ValueError) as error:
        raise CLIRefusal("invalid-allocation", f"allocation is invalid: {error}") from error
    raw_table = _decode(_path(runs_path, "runs"), "runs")
    try:
        table = ConfiguredRunTable.from_mapping(raw_table)
    except (RecursionError, TypeError, ValueError) as error:
        raise CLIRefusal("invalid-runs", f"run-table is invalid: {error}") from error
    try:
        _validate_group_and_runs(group, table)
    except RecursionError as error:
        raise CLIRefusal("admission-depth", "record validation exceeded the supported nesting depth") from error
    worker = load_worker(worker_config_path)

    try:
        if dispatcher_name == "local":
            dispatcher = LocalDispatcher(worker)
        else:
            binding = binding_from_allocation(allocation)
            dispatcher = ParslDispatcher(worker, binding=binding)
    except ParslUnavailableError as error:
        raise CLIRefusal("parsl-unavailable", str(error)) from error
    except (BindingError, TypeError, ValueError) as error:
        raise CLIRefusal("dispatcher-admission", str(error)) from error

    store_file = _path(store_path, "store")
    if "${" in os.fspath(store_file):
        raise CLIRefusal("store-path", "store path must not contain OmegaConf interpolation syntax '${'")
    store = _reserve_store(store_file)
    lifecycle = AttachedAllocation(allocation, store)
    return DispatchPlan(group, table, allocation, worker, dispatcher, lifecycle, timeout)


_publish_new = _write_new


def dispatch(plan: DispatchPlan, manifest_name: str = "run-manifest.json") -> RunManifest:
    """Run an admitted plan and publish its manifest exactly once.

    Parameters
    ----------
    plan : DispatchPlan
        Inputs returned by :func:`admit`.
    manifest_name : str, default="run-manifest.json"
        Fresh manifest name below the reserved store root.

    Returns
    -------
    RunManifest
        Manifest returned by the dispatcher and durably published.

    Raises
    ------
    TypeError
        If ``plan`` is not a :class:`DispatchPlan`.
    BaseException
        The primary dispatch or publication failure, with secondary failures
        attached as notes when both operations fail.
    """

    if not isinstance(plan, DispatchPlan):
        raise TypeError("plan must be DispatchPlan")
    manifest_path = Path(plan.lifecycle.store.root) / manifest_name
    manifest: RunManifest | None = None
    primary: BaseException | None = None
    try:
        manifest = plan.dispatcher.run(
            plan.group,
            {run.id: run for run in plan.table.runs},
            plan.lifecycle,
            timeout=plan.timeout,
        )
    except BaseException as error:
        primary = error
        try:
            manifest = plan.lifecycle.store.manifest(plan.group)
        except BaseException as recovery_error:
            error.add_note(f"partial manifest recovery failed: {recovery_error!r}")
    if manifest is not None:
        try:
            _publish_new(manifest_path, manifest.encode() + b"\n")
        except BaseException as publication_error:
            if primary is not None:
                primary.add_note(f"manifest publication failed: {publication_error!r}")
            else:
                primary = publication_error
    if primary is not None:
        raise primary
    if manifest is None:
        raise RuntimeError("dispatcher returned no manifest")
    return manifest


__all__ = ["CLIRefusal", "DispatchPlan", "admit", "dispatch", "load_worker"]
