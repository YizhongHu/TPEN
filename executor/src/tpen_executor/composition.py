"""Admission and owner-process composition for the public executor CLI."""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
from pathlib import Path
import stat
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ._codec import decode_json
from .allocations import ConfiguredAllocation
from .configuration import StagedHydra
from .contracts import ReadyGroup
from .dispatch import LocalDispatcher, ParslDispatcher, ParslUnavailableError
from .lifecycle import AttachedAllocation
from .placement import BindingError, binding_from_allocation
from .programs import WorkerProgram
from .results import RunManifest
from .storage import DirectoryStore
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
    except (TypeError, ValueError, UnicodeError) as error:
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
    if not callable(getattr(materialized.process, "run", None)):
        raise CLIRefusal("worker-shape", "worker process must provide run()")
    if not callable(getattr(materialized.evidence, "read", None)):
        raise CLIRefusal("worker-shape", "worker evidence must provide read()")


def load_worker(path: str | os.PathLike[str]) -> WorkerProgram:
    """Load an unresolved worker marker from a YAML config file."""

    config_path = _path(path, "worker-config")
    try:
        source = _read_regular(config_path, "worker-config").decode("utf-8")
        loaded = OmegaConf.create(source)
        if not isinstance(loaded, DictConfig):
            raise TypeError("worker-config must be a YAML mapping")
        raw = OmegaConf.to_container(loaded, resolve=False, enum_to_str=True)
        if isinstance(raw, Mapping) and isinstance(raw.get("_target_"), str):
            loaded = OmegaConf.create({"worker": raw})
        staged = StagedHydra.stage(loaded)
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
    """Fully admitted executor inputs, with all writes still pending."""

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
    """Decode and preflight a dispatch without beginning an attempt."""

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
    except (TypeError, ValueError) as error:
        raise CLIRefusal("invalid-ready-group", f"input is not a valid ReadyGroup: {error}") from error
    if not group.members:
        raise CLIRefusal("missing-ready-group", "input must contain a non-empty ReadyGroup")

    raw_allocation = _decode(_path(allocation_path, "allocation"), "allocation")
    try:
        allocation = ConfiguredAllocation.from_mapping(raw_allocation)
    except (TypeError, ValueError) as error:
        raise CLIRefusal("invalid-allocation", f"allocation is invalid: {error}") from error
    raw_table = _decode(_path(runs_path, "runs"), "runs")
    try:
        table = ConfiguredRunTable.from_mapping(raw_table)
    except (TypeError, ValueError) as error:
        raise CLIRefusal("invalid-runs", f"run-table is invalid: {error}") from error
    _validate_group_and_runs(group, table)
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

    store = _reserve_store(store_path)
    lifecycle = AttachedAllocation(allocation, store)
    return DispatchPlan(group, table, allocation, worker, dispatcher, lifecycle, timeout)


def _publish_new(path: Path, payload: bytes) -> None:
    """Publish a complete manifest at a previously absent path."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            fd = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd != -1:
            os.close(fd)


def dispatch(plan: DispatchPlan, manifest_name: str = "run-manifest.json") -> RunManifest:
    """Run an admitted plan and publish its manifest exactly once."""

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
