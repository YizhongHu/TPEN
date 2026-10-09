"""Launch one resolved HI training row through the production runner.

The source materializer has no process facts, so this adapter accepts
operator-supplied execution facts, records them under the manifest's
designated ``topology`` subtree, resolves the row through L1, and delegates
execution to the sanctioned runner entry point. It does not choose inventory
rows, submit scheduler jobs, iterate cells, or launch distributed workers.
"""

from __future__ import annotations

from collections.abc import Buffer, Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
import inspect
import re
from typing import Any

from omegaconf import DictConfig

from tpen.run import run_from_config


_STAGE_API = import_module("experiments.atomistic.he-importance.stage_coordinate")
_TRAIN_CONFIG = import_module("experiments.atomistic.he-importance.train_config")


@dataclass(frozen=True)
class _OwnedLaunchCell:
    """Launch-owned view of a bound row handed to downstream consumers."""

    manifest: Mapping[str, Any]
    content_hash: str
    output_path: Any
    seed_streams: Mapping[str, Any]
# This vocabulary is deliberately closed. Names outside this declaration belong
# to the scientific manifest unless a future owner adds them and gives the
# production consumer a representation for them.
_DECLARED_EXECUTION_FACT_KEYS = frozenset(
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
        "cuda_device_id",
        "rank_id",
        "process_count",
        "pbs_jobid",
        "pbs_nodenum",
        "pbs_tasknum",
        "nvidia_visible_devices",
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
_NORMALIZED_DECLARED_EXECUTION_FACT_KEYS = frozenset(
    re.sub(r"[^a-z0-9]", "", key.lower()) for key in _DECLARED_EXECUTION_FACT_KEYS
)
_RUNNER_TOPOLOGY_FACT_KEYS = frozenset(
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


def _has_buffer_capability(value: object) -> bool:
    """Detect buffer exporters without invoking caller-defined attribute hooks."""

    if isinstance(value, Buffer):
        return True
    value_type = type(value)
    real_mro = type.__dict__["__mro__"].__get__(value_type)
    for base in real_mro:
        namespace = type.__dict__["__dict__"].__get__(base)
        if "__buffer__" in namespace:
            return True
    return False


def _source_cell(source: Any) -> Any:
    """Extract the materialized cell from a cell or a training packet."""

    cell = getattr(source, "cell", source)
    if not all(hasattr(cell, name) for name in ("manifest", "content_hash", "output_path")):
        raise LaunchValidationError("source must be a MaterializedCell or TrainingPacket")
    if type(cell.content_hash) is not str:
        raise LaunchValidationError("source content_hash must be an exact str")
    return cell


def execution_topology_facts(topology: object) -> dict[str, Any]:
    """Serialize launcher facts into the manifest topology boundary."""

    if _has_buffer_capability(topology):
        raise LaunchValidationError(
            f"binary buffer refused: {type(topology).__name__}"
        )
    if isinstance(topology, Mapping):
        fields = dict(topology)
    else:
        try:
            identity = topology.device_identity  # type: ignore[attr-defined]
            if _has_buffer_capability(identity):
                raise LaunchValidationError(
                    f"binary buffer refused: {type(identity).__name__}"
                )
            kind = None if identity is None else identity.kind
            if _has_buffer_capability(kind):
                raise LaunchValidationError(
                    f"binary buffer refused: {type(kind).__name__}"
                )
            fields = {
                "global_rank": topology.global_rank,  # type: ignore[attr-defined]
                "global_size": topology.global_size,  # type: ignore[attr-defined]
                "local_rank": topology.local_rank,  # type: ignore[attr-defined]
                "local_size": topology.local_size,  # type: ignore[attr-defined]
                "node_rank": topology.node_rank,  # type: ignore[attr-defined]
                "node_size": topology.node_size,  # type: ignore[attr-defined]
                "host": topology.host,  # type: ignore[attr-defined]
                "pid": topology.pid,  # type: ignore[attr-defined]
                "device": topology.device,  # type: ignore[attr-defined]
                "job_id": topology.job_id,  # type: ignore[attr-defined]
                "device_identity": (
                    None
                    if identity is None
                    else {
                        "kind": getattr(kind, "value", kind),
                        "index": identity.index,
                        "uuid": identity.uuid,
                    }
                ),
            }
        except AttributeError as error:
            raise LaunchValidationError(
                "topology must be an execution topology or mapping"
            ) from error
    return _detach_topology_facts(fields)


def _reject_execution_facts_outside_topology(manifest: Mapping[str, Any]) -> None:
    """Refuse launcher-shaped facts in scientific or payload namespaces.

    The declared vocabulary is the normalized form of
    ``_DECLARED_EXECUTION_FACT_KEYS``. The detector traverses mappings and
    non-text sequences recursively; the root ``topology`` is the sole excluded
    subtree. Names outside that declaration are scientific data, even when they
    resemble process metadata (for example ``matrix_rank`` or ``low_rank``).
    This closed lexical vocabulary cannot
    classify every spelling: the guard closes named mechanisms, not the class.
    Whoever adds an execution fact to the production consumer owns adding its
    spellings here. The declaration contains the canonical
    rank/size, host/pid/device, job, identity, and explicitly listed launcher
    and environment aliases. It carries five ``slurm_*``, three
    ``pmi*``/``pmix``, and two ``ompi_*`` spellings. TPEN's production
    facility, ALCF Polaris, is PBS; that probe has now run: the facility sets
    27 ``PBS_*`` names, 26 of which are accepted outside ``manifest.topology``.
    ``pbs_jobid``, ``pbs_nodenum``, and ``pbs_tasknum`` are declared as the
    per-process execution facts among them. The remaining 24 are deliberately
    excluded as submission context and provenance (queue, account, paths,
    ``PBS_O_*`` origin names) rather than per-process execution facts.
    """

    def is_execution_fact_key(key: object) -> bool:
        if not isinstance(key, str):
            # Backstop: L1's materializer refuses non-string identity keys first.
            return False
        normalized = re.sub(r"[^a-z0-9]", "", str.lower(key))
        return normalized in _NORMALIZED_DECLARED_EXECUTION_FACT_KEYS

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
        if str.__eq__(key, _STAGE_API.TOPOLOGY_KEY) is not True:
            visit(value, key)


def _validate_runner_topology_facts(
    facts: Mapping[str, Any], *, reject_unsupported: bool = True
) -> None:
    """Validate the facts accepted by the production runner boundary.

    This mirrors the value invariants of TPEN's topology type without naming
    that type across the experiment boundary. The production runner performs
    the authoritative typed normalization; this seam must still reject bad
    values before any injected runner is called.
    """

    if reject_unsupported:
        unsupported = tuple(key for key in facts if key not in _RUNNER_TOPOLOGY_FACT_KEYS)
        if unsupported:
            names = ", ".join(repr(key) for key in unsupported)
            raise LaunchValidationError(
                "unsupported production runner topology facts: " + names
            )
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

    for name in ("global_size", "local_size", "node_size"):
        size = facts[name]
        if type(size) is not int:
            raise LaunchValidationError(
                f"{name} must be an int, got {type(size).__name__}"
            )
        if size < 1:
            raise LaunchValidationError(f"{name} must be positive, got {size}")
    for rank_name, size_name in (
        ("global_rank", "global_size"),
        ("local_rank", "local_size"),
        ("node_rank", "node_size"),
    ):
        rank = facts[rank_name]
        size = facts[size_name]
        if rank is not None and type(rank) is not int:
            raise LaunchValidationError(
                f"{rank_name} must be an int or None, got {type(rank).__name__}"
            )
        if rank is not None and not 0 <= rank < size:
            raise LaunchValidationError(f"{rank_name} must be in [0, {size})")
    if not facts["host"]:
        raise LaunchValidationError("host must be nonempty")
    try:
        if facts["pid"] < 1:
            raise LaunchValidationError("pid must be positive")
    except TypeError as error:
        raise LaunchValidationError(
            f"pid must be an int, got {type(facts['pid']).__name__}"
        ) from error
    if not facts["device"]:
        raise LaunchValidationError("device must be nonempty")

    identity_value = facts.get("device_identity")
    if identity_value is None:
        return
    if isinstance(identity_value, Mapping):
        unsupported_identity = set(identity_value) - {"kind", "index", "uuid"}
        if reject_unsupported and unsupported_identity:
            names = ", ".join(repr(key) for key in sorted(unsupported_identity, key=str))
            raise LaunchValidationError(
                "unsupported topology.device_identity keys: " + names
            )
        try:
            kind = identity_value["kind"]
            if kind not in {"cpu", "cuda", "rocm", "other"}:
                raise ValueError(kind)
            return
        except (KeyError, TypeError, ValueError) as error:
            raise LaunchValidationError("topology.device_identity is malformed") from error
    raise LaunchValidationError("topology.device_identity must be a mapping or null")


def _detach_topology_facts(value: Any) -> Any:
    """Recursively materialize the closed topology-facts value schema."""

    if value is None or any(
        type(value) is scalar for scalar in (str, int, float, bool)
    ):
        return value
    if _has_buffer_capability(value):
        raise LaunchValidationError(f"binary buffer refused: {type(value).__name__}")
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise LaunchValidationError("topology fact mapping keys must be exact str")
            copied[key] = _detach_topology_facts(item)
        return copied
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_detach_topology_facts(item) for item in value]
    raise LaunchValidationError(
        f"unsupported topology fact value type: {type(value).__name__}"
    )


def populate_execution_topology(source: Any, topology: Mapping[str, Any]) -> Any:
    """Bind non-empty launch facts to a source row's designated subtree."""

    topology = _detach_topology_facts(topology)
    cell = _source_cell(source)
    try:
        _STAGE_API.validate_materialized_manifest(cell.manifest)
    except Exception as error:
        raise LaunchValidationError("source manifest is not a valid HI train row") from error
    try:
        bound = _STAGE_API.with_execution_topology(cell, topology)
        # The stage API preserves the caller's concrete cell type.  Capture its
        # frozen result once, then hand only this module-owned carrier onward so
        # later consumers cannot obtain a fresh caller-controlled manifest read.
        bound_hash = bound.content_hash
        if type(bound_hash) is not str:
            raise LaunchValidationError("bound content_hash must be an exact str")
        owned = _OwnedLaunchCell(
            manifest=_STAGE_API._freeze(_detach_topology_facts(bound.manifest)),
            content_hash=bound_hash,
            output_path=bound.output_path,
            seed_streams=_detach_topology_facts(bound.seed_streams),
        )
        _reject_execution_facts_outside_topology(owned.manifest)
        return owned
    except Exception as error:
        raise LaunchValidationError(str(error)) from error


def prepare_train_launch(
    source: Any,
    topology: object | None = None,
) -> LaunchPlan:
    """Populate topology and resolve one source row through L1."""

    facts = execution_topology_facts(topology) if topology is not None else {}
    cell = populate_execution_topology(source, facts)
    config = _TRAIN_CONFIG.resolve_train_config(cell)
    return LaunchPlan(cell=cell, config=config, topology=cell.manifest[_STAGE_API.TOPOLOGY_KEY])


def _exit_code(result: object) -> int:
    """Require runners to return an integer process code."""

    if type(result) is not int:
        raise LaunchValidationError("runner must return an int")
    return result


def launch_train(
    source: Any,
    topology: object | None = None,
    *,
    runner: Callable[..., object] = run_from_config,
) -> int:
    """Resolve one topology-bound row and execute TPEN's production runner.

    Custom runners receive the canonical execution-facts mapping when their
    signature can accept a ``topology`` keyword or arbitrary keyword
    arguments; plain config-only doubles retain the existing config-only call.
    The mapping is the declared experiment-side contract because the typed
    TPEN topology is owned behind this boundary. This closes the named
    capability-blind identity-dispatch mechanism, not every way a callable
    could ignore or mishandle a topology it accepts.
    """

    facts = execution_topology_facts(topology) if topology is not None else {}
    plan = prepare_train_launch(source, facts)
    if topology is not None:
        # Topology is caller-open, so config-only runners retain extra facts.
        # Strict key rejection remains below when the mapping crosses into TPEN.
        _validate_runner_topology_facts(facts, reject_unsupported=False)
    if runner is run_from_config:
        if topology is None:
            # Backstop: L1's empty-mapping refusal is reached first.
            raise LaunchValidationError("launch topology is required")
        _validate_runner_topology_facts(facts)
        return _exit_code(runner(plan.config, topology=facts))
    if topology is not None:
        parameters = inspect.signature(runner).parameters.values()
        accepts_topology = any(
            str.__eq__(parameter.name, "topology") is True
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if accepts_topology:
            _validate_runner_topology_facts(facts)
            return _exit_code(runner(plan.config, topology=facts))
    return _exit_code(runner(plan.config))


__all__ = [
    "LaunchPlan",
    "LaunchValidationError",
    "execution_topology_facts",
    "launch_train",
    "populate_execution_topology",
    "prepare_train_launch",
]
