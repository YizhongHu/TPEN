"""Launch one resolved HI training row through TPEN's production runner.

The source materializer has no process facts, so this adapter accepts an
operator-supplied :class:`~tpen.distributed.ExecutionTopology`, records its
facts under the manifest's designated ``topology`` subtree, resolves the row
through L1, and delegates execution to ``tpen.run.run_from_config``.  It does
not choose inventory rows, submit scheduler jobs, iterate cells, or launch
distributed workers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
import re
from threading import RLock
from typing import Any

from omegaconf import DictConfig

from tpen.accelerator import AcceleratorIdentity, AcceleratorKind
from tpen.artifacts import RunResult
from tpen.distributed import ExecutionTopology
from tpen.run import run_from_config


_STAGE_API = import_module("experiments.atomistic.he-importance.stage_coordinate")
_TRAIN_CONFIG = import_module("experiments.atomistic.he-importance.train_config")
_TOPOLOGY_FACT_KEYS = frozenset(
    {
        "global_rank",
        "global_size",
        "local_rank",
        "local_size",
        "node_rank",
        "node_size",
        "host",
        "pid",
        "device",
        "job_id",
        "device_identity",
        # Common launcher spellings are refused outside the boundary too.
        "rank",
        "world_size",
        "launcher",
        "launcher_job_id",
        # Common environment and launcher spellings which carry the same
        # process facts under a different name.  This boundary is lexical and
        # deliberately independent of the scientific manifest vocabulary.
        "hostname",
        "host_name",
        "process_id",
        "processid",
        "device_id",
        "deviceid",
        "cuda_visible_devices",
        "visible_devices",
        "num_nodes",
        "node_count",
        "num_gpus",
        "gpu_id",
        "master_addr",
        "master_port",
        "slurm_ntasks",
        "slurm_nprocs",
        "slurm_procid",
        "slurm_localid",
        "slurm_job_num_nodes",
        "pmi_size",
        "pmi_rank",
        "pmix_rank",
        "ompi_comm_world_size",
        "ompi_comm_world_rank",
        "mpi_localrankid",
    }
)
_NORMALIZED_TOPOLOGY_FACT_KEYS = frozenset(
    re.sub(r"[^a-z0-9]", "", key.lower()) for key in _TOPOLOGY_FACT_KEYS
)
_LAUNCHER_FACT_PREFIXES = ("slurm", "pmi", "pmix", "ompi", "mpi")
_LAUNCHER_FACT_SUFFIXES = (
    "rank",
    "worldsize",
    "jobid",
    "processid",
    "hostname",
    "visibledevices",
)
_TOPOLOGY_HANDOFF_LOCK = RLock()


class LaunchValidationError(ValueError):
    """A materialized HI row cannot enter the launch seam."""


@dataclass(frozen=True)
class LaunchPlan:
    """The topology-bound row and L1-resolved config handed to the runner."""

    cell: Any
    config: DictConfig
    topology: Mapping[str, Any]


def _source_cell(source: Any) -> Any:
    """Extract the materialized cell from a cell or a training packet."""

    cell = getattr(source, "cell", source)
    if not all(hasattr(cell, name) for name in ("manifest", "content_hash", "output_path")):
        raise LaunchValidationError("source must be a MaterializedCell or TrainingPacket")
    return cell


def execution_topology_facts(topology: ExecutionTopology | Mapping[str, Any]) -> dict[str, Any]:
    """Serialize typed launcher facts into the manifest topology boundary."""

    if isinstance(topology, ExecutionTopology):
        identity = topology.device_identity
        return {
            "global_rank": topology.global_rank,
            "global_size": topology.global_size,
            "local_rank": topology.local_rank,
            "local_size": topology.local_size,
            "node_rank": topology.node_rank,
            "node_size": topology.node_size,
            "host": topology.host,
            "pid": topology.pid,
            "device": topology.device,
            "job_id": topology.job_id,
            "device_identity": (
                None
                if identity is None
                else {
                    "kind": identity.kind.value,
                    "index": identity.index,
                    "uuid": identity.uuid,
                }
            ),
        }
    if isinstance(topology, Mapping):
        return dict(topology)
    raise LaunchValidationError("topology must be an ExecutionTopology or mapping")


def _reject_execution_facts_outside_topology(manifest: Mapping[str, Any]) -> None:
    """Refuse launcher-shaped facts in scientific or payload namespaces.

    The detector traverses mappings and non-text sequences to arbitrary depth.
    Root ``topology`` is the sole excluded subtree; every other namespace is
    inspected.  Keys are compared after lower-casing and removing separators,
    so common launcher/environment aliases such as ``WORLD_SIZE``,
    ``hostname``, and ``process_id`` cannot bypass the boundary.
    """

    def is_execution_fact_key(key: object) -> bool:
        if not isinstance(key, str):
            return False
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        return normalized in _NORMALIZED_TOPOLOGY_FACT_KEYS or normalized.startswith(
            _LAUNCHER_FACT_PREFIXES
        ) or normalized.endswith(_LAUNCHER_FACT_SUFFIXES)

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if is_execution_fact_key(key):
                    raise LaunchValidationError(
                        f"execution fact {key!r} must be under manifest.topology, found at {path}.{key}"
                    )
                visit(nested, f"{path}.{key}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    for key, value in manifest.items():
        if key != _STAGE_API.TOPOLOGY_KEY:
            visit(value, key)


def _typed_runner_topology(topology: ExecutionTopology | Mapping[str, Any]) -> ExecutionTopology:
    """Return the typed topology that the TPEN consumer stores in its context."""

    if isinstance(topology, ExecutionTopology):
        return topology
    facts = execution_topology_facts(topology)
    required = (
        "global_rank",
        "global_size",
        "local_rank",
        "local_size",
        "node_rank",
        "node_size",
        "host",
        "pid",
        "device",
    )
    missing = tuple(name for name in required if name not in facts)
    if missing:
        raise LaunchValidationError(
            "production runner topology is missing required facts: " + ", ".join(missing)
        )
    identity_value = facts.get("device_identity")
    if identity_value is None or isinstance(identity_value, AcceleratorIdentity):
        identity = identity_value
    elif isinstance(identity_value, Mapping):
        try:
            identity = AcceleratorIdentity(
                kind=AcceleratorKind(str(identity_value["kind"])),
                index=identity_value.get("index"),
                uuid=identity_value.get("uuid"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LaunchValidationError("topology.device_identity is malformed") from error
    else:
        raise LaunchValidationError("topology.device_identity must be a mapping or null")
    try:
        return ExecutionTopology(
            global_rank=facts["global_rank"],
            global_size=facts["global_size"],
            local_rank=facts["local_rank"],
            local_size=facts["local_size"],
            node_rank=facts["node_rank"],
            node_size=facts["node_size"],
            host=facts["host"],
            pid=facts["pid"],
            device=facts["device"],
            job_id=facts.get("job_id"),
            device_identity=identity,
        )
    except (TypeError, ValueError) as error:
        raise LaunchValidationError(f"invalid production runner topology: {error}") from error


@contextmanager
def _topology_at_production_consumer(topology: ExecutionTopology):
    """Bind ``topology`` to TPEN's existing context construction path.

    ``run_from_config`` is the production callable and currently exposes no
    topology keyword, while its consumer ``prepare_run_context`` does.  The
    narrow adapter below binds that existing consumer seam for the duration of
    one launch; it does not reimplement runner setup or execution.
    """

    production_globals = run_from_config.__globals__
    original_prepare = production_globals["prepare_run_context"]

    def prepare_with_topology(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("topology") is not None:
            raise LaunchValidationError("production runner supplied a conflicting execution topology")
        kwargs["topology"] = topology
        return original_prepare(*args, **kwargs)

    with _TOPOLOGY_HANDOFF_LOCK:
        production_globals["prepare_run_context"] = prepare_with_topology
        try:
            yield
        finally:
            production_globals["prepare_run_context"] = original_prepare


def populate_execution_topology(source: Any, topology: Mapping[str, Any]) -> Any:
    """Bind non-empty launch facts to a source row's designated subtree."""

    cell = _source_cell(source)
    try:
        _STAGE_API.validate_materialized_manifest(cell.manifest)
    except Exception as error:
        raise LaunchValidationError("source manifest is not a valid HI train row") from error
    _reject_execution_facts_outside_topology(cell.manifest)
    try:
        return _STAGE_API.with_execution_topology(cell, topology)
    except Exception as error:
        raise LaunchValidationError(str(error)) from error


def prepare_train_launch(
    source: Any,
    topology: ExecutionTopology | Mapping[str, Any] | None = None,
) -> LaunchPlan:
    """Populate topology and resolve one source row through L1."""

    facts = execution_topology_facts(topology) if topology is not None else {}
    cell = populate_execution_topology(source, facts)
    config = _TRAIN_CONFIG.resolve_train_config(cell)
    return LaunchPlan(cell=cell, config=config, topology=cell.manifest[_STAGE_API.TOPOLOGY_KEY])


def _exit_code(result: int | RunResult) -> int:
    """Normalize production and injected runner results to a process code."""

    if isinstance(result, RunResult):
        return 1 if result.status == "failed" else 0
    if type(result) is not int:
        raise LaunchValidationError("runner must return an int or RunResult")
    return result


def launch_train(
    source: Any,
    topology: ExecutionTopology | Mapping[str, Any] | None = None,
    *,
    runner: Callable[[DictConfig], int | RunResult] = run_from_config,
) -> int:
    """Resolve one topology-bound row and execute TPEN's production runner."""

    plan = prepare_train_launch(source, topology)
    if runner is run_from_config:
        if topology is None:
            raise LaunchValidationError("launch topology is required")
        runtime_topology = _typed_runner_topology(topology)
        with _topology_at_production_consumer(runtime_topology):
            return _exit_code(runner(plan.config))
    return _exit_code(runner(plan.config))


__all__ = [
    "LaunchPlan",
    "LaunchValidationError",
    "execution_topology_facts",
    "launch_train",
    "populate_execution_topology",
    "prepare_train_launch",
]
