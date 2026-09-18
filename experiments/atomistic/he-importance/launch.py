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
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from omegaconf import DictConfig

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
    }
)


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
    """Refuse launcher-shaped facts in scientific or payload namespaces."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key in _TOPOLOGY_FACT_KEYS:
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
    return _exit_code(runner(plan.config))


__all__ = [
    "LaunchPlan",
    "LaunchValidationError",
    "execution_topology_facts",
    "launch_train",
    "populate_execution_topology",
    "prepare_train_launch",
]
