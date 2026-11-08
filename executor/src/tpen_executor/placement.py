"""Explicit allocation bindings and child-side placement observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, ClassVar

from .allocations import ConfiguredAllocation
from ._codec import _freeze, decode_json, fields, sha256_text, text, thaw
from ._records import frozen_record


class BindingError(ValueError):
    """Raised when a worker binding is absent or cannot be trusted."""


class PlacementRefusal(ValueError):
    """Raised when a child cannot prove its declared placement."""

    def __init__(self, code: str, reason: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("placement refusal code must be a non-empty string")
        if type(reason) is not str or not reason:
            raise TypeError("placement refusal reason must be a non-empty string")
        self.code = code
        self.reason = reason
        super().__init__(reason)

    def to_dict(self) -> dict[str, str]:
        """Return the typed refusal envelope."""

        return {"code": self.code, "reason": self.reason}


CHILD_PLACEMENT_SCHEMA = "tpen-executor/child-placement-v1"
PLACEMENT_OUTPUT_ENV = "TPEN_PLACEMENT_OUTPUT"
PLACEMENT_REFUSAL_ENV = "TPEN_PLACEMENT_REFUSAL"


def canonical_host(value: str) -> str:
    """Return the stable short-host key used for placement comparisons.

    Raw host values remain in the sidecar and receipt. The comparison key
    removes case, a trailing dot, and an optional DNS suffix so a scheduler's
    short node label can match the child's FQDN without hiding the originals.
    """

    if type(value) is not str or not value.strip():
        raise PlacementRefusal("invalid-host", "placement host must be non-empty text")
    return value.strip().lower().rstrip(".").split(".", 1)[0]


def parse_proc_start_ticks(pid: int, stat_text: str | None = None) -> int:
    """Parse Linux ``/proc/<pid>/stat`` start ticks around the final ``)``.

    Parameters
    ----------
    pid : int
        Positive process ID whose stat record is being parsed.
    stat_text : str, optional
        Test-provided stat text.  If omitted, the live proc file is read.

    Returns
    -------
    int
        Field 22, the process start time in clock ticks.

    Raises
    ------
    PlacementRefusal
        If proc is absent or the record is malformed.  Splitting the complete
        line on whitespace is deliberately avoided because ``comm`` may hold
        spaces and parentheses.
    """

    if type(pid) is not int or isinstance(pid, bool) or pid <= 0:
        raise PlacementRefusal("invalid-pid", "child pid must be a positive integer")
    if stat_text is None:
        try:
            stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise PlacementRefusal("proc-unavailable", "Linux process start ticks are unavailable") from error
    close = stat_text.rfind(")")
    if close < 0:
        raise PlacementRefusal("malformed-proc-stat", "process stat has no terminating comm delimiter")
    fields_after_comm = stat_text[close + 1 :].strip().split()
    # The first token after the final ')' is field 3 (state); starttime is 22.
    if len(fields_after_comm) <= 19:
        raise PlacementRefusal("malformed-proc-stat", "process stat does not contain start ticks")
    try:
        ticks = int(fields_after_comm[19])
    except (TypeError, ValueError) as error:
        raise PlacementRefusal("malformed-proc-stat", "process start ticks are not an integer") from error
    if ticks < 0:
        raise PlacementRefusal("malformed-proc-stat", "process start ticks are negative")
    return ticks


def parse_cpu_set(value: str) -> tuple[int, ...]:
    """Parse a comma-separated declared CPU set into canonical order."""

    if type(value) is not str or not value.strip():
        raise PlacementRefusal("invalid-cpu-set", "declared CPU set must be non-empty text")
    try:
        cpus = tuple(sorted({int(part) for part in value.split(",")}))
    except (TypeError, ValueError) as error:
        raise PlacementRefusal("invalid-cpu-set", "declared CPU set contains a non-integer") from error
    if not cpus or any(cpu < 0 for cpu in cpus):
        raise PlacementRefusal("invalid-cpu-set", "declared CPU set contains an invalid CPU")
    return cpus


def utc_timestamp() -> str:
    """Return an explicit UTC timestamp for placement evidence."""

    return datetime.now(timezone.utc).isoformat()


@frozen_record
class ChildPlacementObservation:
    """Strict sidecar emitted by the science child before core import."""

    schema: ClassVar[str] = CHILD_PLACEMENT_SCHEMA
    task_id: str
    attempt_id: str
    group_digest: str
    allocation_id: str
    backend_job_id: str
    slot_id: str
    payload_digest: str
    output_dir: str
    host: str
    pid: int
    proc_start_ticks: int
    affinity: tuple[int, ...]
    thread_settings: dict[str, str | None]
    cuda_visible_devices: str | None
    observed_at_utc: str
    refusal: dict[str, str] | None

    def __post_init__(self) -> None:
        for name in (
            "task_id", "attempt_id", "allocation_id", "backend_job_id", "slot_id", "output_dir", "host",
        ):
            object.__setattr__(self, name, text(getattr(self, name), name))
        object.__setattr__(self, "group_digest", sha256_text(self.group_digest, "group_digest"))
        object.__setattr__(self, "payload_digest", sha256_text(self.payload_digest, "payload_digest"))
        if type(self.pid) is not int or isinstance(self.pid, bool) or self.pid <= 0:
            raise TypeError("pid must be a positive integer")
        if type(self.proc_start_ticks) is not int or isinstance(self.proc_start_ticks, bool) or self.proc_start_ticks < 0:
            raise TypeError("proc_start_ticks must be a nonnegative integer")
        object.__setattr__(self, "affinity", _int_tuple(self.affinity, "child affinity"))
        if not self.affinity and self.refusal is None:
            raise TypeError("child affinity must not be empty")
        if not isinstance(self.thread_settings, Mapping):
            raise TypeError("thread_settings must be a mapping")
        if any(type(key) is not str or type(value) is not str and value is not None for key, value in self.thread_settings.items()):
            raise TypeError("thread_settings must map names to strings or null")
        object.__setattr__(self, "thread_settings", _freeze(self.thread_settings, "thread_settings"))
        if self.cuda_visible_devices is not None and type(self.cuda_visible_devices) is not str:
            raise TypeError("cuda_visible_devices must be a string or null")
        object.__setattr__(self, "observed_at_utc", text(self.observed_at_utc, "observed_at_utc"))
        if self.refusal is not None:
            if not isinstance(self.refusal, Mapping) or set(self.refusal) != {"code", "reason"}:
                raise TypeError("refusal must contain code and reason")
            object.__setattr__(self, "refusal", _freeze(self.refusal, "refusal"))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical child-side observation payload."""

        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "group_digest": self.group_digest,
            "allocation_id": self.allocation_id,
            "backend_job_id": self.backend_job_id,
            "slot_id": self.slot_id,
            "payload_digest": self.payload_digest,
            "output_dir": self.output_dir,
            "host": self.host,
            "pid": self.pid,
            "proc_start_ticks": self.proc_start_ticks,
            "affinity": list(self.affinity),
            "thread_settings": dict(self.thread_settings),
            "cuda_visible_devices": self.cuda_visible_devices,
            "observed_at_utc": self.observed_at_utc,
            "refusal": self.refusal,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> "ChildPlacementObservation":
        """Decode one strict child placement sidecar."""

        expected = {
            "schema", "task_id", "attempt_id", "group_digest", "allocation_id", "backend_job_id", "slot_id",
            "payload_digest", "output_dir", "host", "pid", "proc_start_ticks", "affinity", "thread_settings",
            "cuda_visible_devices", "observed_at_utc", "refusal",
        }
        fields(data, expected, cls.schema, "ChildPlacementObservation")
        if any(type(data[key]) is not str for key in expected - {"pid", "proc_start_ticks", "affinity", "thread_settings", "cuda_visible_devices", "refusal", "schema"}):
            raise TypeError("child placement identity fields must be strings")
        return cls(
            data["task_id"], data["attempt_id"], data["group_digest"], data["allocation_id"], data["backend_job_id"],
            data["slot_id"], data["payload_digest"], data["output_dir"], data["host"], data["pid"],
            data["proc_start_ticks"], tuple(data["affinity"]), data["thread_settings"],
            data["cuda_visible_devices"], data["observed_at_utc"], data["refusal"],
        )


def read_child_placement(path: str | os.PathLike[str]) -> ChildPlacementObservation:
    """Read one non-symlink child placement sidecar.

    Parameters
    ----------
    path : str or pathlib.Path
        Candidate sidecar path.

    Returns
    -------
    ChildPlacementObservation
        Strictly decoded child placement evidence.

    Raises
    ------
    PlacementRefusal
        If the path is missing, a symlink, or malformed.
    """

    candidate = Path(path)
    try:
        info = candidate.lstat()
    except OSError as error:
        raise PlacementRefusal("placement-missing", "child placement sidecar is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PlacementRefusal("placement-invalid-file", "child placement sidecar must be a regular file")
    try:
        return ChildPlacementObservation.from_mapping(decode_json(candidate.read_bytes()))
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise PlacementRefusal("placement-malformed", "child placement sidecar is not a strict record") from error


def placement_environment(base: Mapping[str, str], **values: str) -> dict[str, str]:
    """Overlay executor placement facts without modifying controller state.

    Parameters
    ----------
    base : mapping[str, str]
        Existing prepared child environment.
    **values : str
        Placement fields to overlay; only declared devices may be empty.

    Returns
    -------
    dict[str, str]
        A copied environment containing the placement fields.

    Raises
    ------
    PlacementRefusal
        If a placement field is not valid text.
    """

    result = dict(base)
    for key, value in values.items():
        if type(value) is not str or (not value and key != "TPEN_PLACEMENT_DECLARED_DEVICES"):
            raise PlacementRefusal("invalid-placement-fact", f"placement fact {key!r} must be non-empty text")
        result[key] = value
    return result


def _normalize_nodes(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one generic node shape or refuse ambiguous geometry."""

    if "nodes" not in value:
        return dict(value)
    nodes = value["nodes"]
    if isinstance(nodes, (str, bytes, bytearray)) or not isinstance(nodes, Sequence):
        raise BindingError("worker binding nodes must be a sequence")
    if len(nodes) != 1 or type(nodes[0]) is not str or not nodes[0]:
        raise BindingError("worker binding must identify exactly one worker node")
    normalized = dict(value)
    existing = normalized.get("node")
    if existing is not None and existing != nodes[0]:
        raise BindingError("worker binding node and nodes disagree")
    normalized["node"] = nodes[0]
    return normalized


@frozen_record
class WorkerBinding:
    """Immutable environment binding for one Parsl worker.

    Parameters
    ----------
    environment : mapping[str, str]
        Environment entries applied after the prepared worker environment.
    worker_id, node : str or None, optional
        Optional operator-visible binding identities. They are carried as
        metadata by callers and are not inferred from scheduler state.
    """

    environment: Mapping[str, str] | tuple[tuple[str, str], ...]
    worker_id: str | None = None
    node: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.environment, Mapping):
            raw_entries = tuple(self.environment.items())
        elif isinstance(self.environment, tuple):
            raw_entries = self.environment
        else:
            raise BindingError("worker binding environment must be a mapping or pairs")
        if not raw_entries:
            raise BindingError("worker binding environment must not be empty")
        entries: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in raw_entries:
            if not isinstance(item, tuple) or len(item) != 2:
                raise BindingError("worker binding environment entries must be pairs")
            key, value = item
            if type(key) is not str or not key or type(value) is not str:
                raise BindingError("worker binding environment keys and values must be strings")
            if not value and key != "TPEN_PLACEMENT_DECLARED_DEVICES":
                raise BindingError("worker binding environment values must be non-empty strings")
            if key in seen:
                raise BindingError(f"worker binding contains duplicate environment key {key!r}")
            seen.add(key)
            entries.append((key, value))
        object.__setattr__(self, "environment", tuple(entries))
        for name in ("worker_id", "node"):
            value = getattr(self, name)
            if value is not None and (type(value) is not str or not value):
                raise BindingError(f"{name} must be a non-empty string or None")

    @property
    def env(self) -> Mapping[str, str]:
        """Return the immutable environment mapping for this worker."""

        return MappingProxyType(dict(self.environment))

    @classmethod
    def from_mapping(cls, value: Any) -> "WorkerBinding":
        """Decode one explicit binding mapping."""

        if isinstance(value, WorkerBinding):
            return value
        if not isinstance(value, Mapping):
            raise BindingError("worker binding must be a mapping")
        value = _normalize_nodes(value)
        environment = value.get("environment", value.get("env"))
        if environment is None:
            environment = {
                key: item
                for key, item in value.items()
                if key not in {"worker_id", "node", "nodes", "visibility", "accelerator_visibility"}
            }
        if not isinstance(environment, Mapping):
            raise BindingError("worker binding environment must be a mapping")
        normalized: dict[str, str] = {}
        for key, item in environment.items():
            if type(key) is not str or not key:
                raise BindingError("worker binding environment keys must be strings")
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                item = ",".join(str(part) for part in item)
            if type(item) is not str or (not item and key != "TPEN_PLACEMENT_DECLARED_DEVICES"):
                raise BindingError("worker binding environment values must be non-empty strings")
            normalized[key] = item
        visibility = value.get("visibility", value.get("accelerator_visibility"))
        if visibility is not None and "CUDA_VISIBLE_DEVICES" not in normalized:
            if isinstance(visibility, Sequence) and not isinstance(visibility, (str, bytes, bytearray)):
                visibility = ",".join(str(part) for part in visibility)
            if type(visibility) is not str or not visibility:
                raise BindingError("accelerator visibility must be a non-empty string or sequence")
            normalized["CUDA_VISIBLE_DEVICES"] = visibility
        if not normalized:
            raise BindingError("worker binding must declare a worker environment")
        return cls(
            tuple(normalized.items()),
            value.get("worker_id"),
            value.get("node"),
        )


_SLOT_SCHEMA = "tpen-executor/allocation-slots-v1"
_THREAD_NAMES = frozenset({"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"})


@frozen_record
class AllocationSlot:
    """Immutable generic entitlement for one attached allocation slot.

    Parameters
    ----------
    slot_id, node : str
        Stable slot and worker-node identities.
    devices : sequence of str
        Device identifiers entitled to this slot.  Empty is valid for a
        CPU-only attached allocation.
    cpus : sequence of int
        Disjoint logical CPU identifiers declared for this slot.
    threads : mapping[str, int]
        Positive thread settings propagated to the child environment.
    environment : mapping[str, str]
        Additional child environment entries.  Reserved derived entries must
        agree with the slot entitlement when supplied.
    """

    schema: ClassVar[str] = _SLOT_SCHEMA
    slot_id: str
    node: str
    devices: tuple[str, ...]
    cpus: tuple[int, ...]
    threads: tuple[tuple[str, int], ...]
    environment: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name, value in (("slot_id", self.slot_id), ("node", self.node)):
            if type(value) is not str or not value.strip():
                raise BindingError(f"slot {name} must be a non-empty string")
        devices = _string_tuple(self.devices, "slot devices")
        if len(set(devices)) != len(devices):
            raise BindingError("slot devices must be unique")
        cpus = _int_tuple(self.cpus, "slot cpus")
        if not cpus or any(cpu < 0 for cpu in cpus) or len(set(cpus)) != len(cpus):
            raise BindingError("slot cpus must be a non-empty set of unique non-negative integers")
        if isinstance(self.threads, Mapping):
            raw_threads = tuple(self.threads.items())
        else:
            raw_threads = self.threads
        normalized_threads: list[tuple[str, int]] = []
        seen_threads: set[str] = set()
        for item in raw_threads:
            if not isinstance(item, tuple) or len(item) != 2:
                raise BindingError("slot thread settings must be pairs")
            key, value = item
            if type(key) is not str or key not in _THREAD_NAMES or key in seen_threads:
                raise BindingError("slot thread settings contain an invalid or duplicate name")
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise BindingError("slot thread settings must be positive integers")
            seen_threads.add(key)
            normalized_threads.append((key, value))
        if isinstance(self.environment, Mapping):
            raw_environment = tuple(self.environment.items())
        else:
            raw_environment = self.environment
        normalized_environment: list[tuple[str, str]] = []
        seen_environment: set[str] = set()
        for item in raw_environment:
            if not isinstance(item, tuple) or len(item) != 2:
                raise BindingError("slot environment entries must be pairs")
            key, value = item
            if type(key) is not str or not key or key in seen_environment:
                raise BindingError("slot environment contains an invalid or duplicate name")
            if type(value) is not str or not value:
                raise BindingError("slot environment values must be non-empty strings")
            seen_environment.add(key)
            normalized_environment.append((key, value))
        env = dict(normalized_environment)
        if devices:
            visible = ",".join(devices)
            if "CUDA_VISIBLE_DEVICES" in env and env["CUDA_VISIBLE_DEVICES"] != visible:
                raise BindingError("slot CUDA_VISIBLE_DEVICES disagrees with device entitlement")
        for key, value in normalized_threads:
            if key in env and env[key] != str(value):
                raise BindingError(f"slot environment {key!r} disagrees with thread setting")
        derived = {
            "TPEN_SLOT_ID": self.slot_id,
            "TPEN_SLOT_NODE": self.node,
            "TPEN_SLOT_CPUS": ",".join(str(cpu) for cpu in cpus),
        }
        if devices:
            derived["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        derived.update({key: str(value) for key, value in normalized_threads})
        for key, value in derived.items():
            if key in env and env[key] != value:
                raise BindingError(f"slot environment {key!r} disagrees with derived binding")
        object.__setattr__(self, "devices", devices)
        object.__setattr__(self, "cpus", cpus)
        object.__setattr__(self, "threads", tuple(normalized_threads))
        object.__setattr__(self, "environment", tuple(normalized_environment))

    @property
    def env(self) -> Mapping[str, str]:
        """Return the complete child overlay for this slot."""

        values = dict(self.environment)
        values.update({"TPEN_SLOT_ID": self.slot_id, "TPEN_SLOT_NODE": self.node, "TPEN_SLOT_CPUS": ",".join(str(cpu) for cpu in self.cpus)})
        values["TPEN_PLACEMENT_DECLARED_DEVICES"] = ",".join(self.devices)
        if self.devices:
            values["CUDA_VISIBLE_DEVICES"] = ",".join(self.devices)
        values.update({key: str(value) for key, value in self.threads})
        return MappingProxyType(values)

    def binding(self) -> WorkerBinding:
        """Return the worker binding derived from this immutable slot."""

        return WorkerBinding(self.env, self.slot_id, self.node)

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-shaped slot payload."""

        return {
            "slot_id": self.slot_id,
            "node": self.node,
            "devices": list(self.devices),
            "cpus": list(self.cpus),
            "threads": dict(self.threads),
            "environment": dict(self.environment),
        }

    @classmethod
    def from_mapping(cls, value: Any) -> "AllocationSlot":
        """Decode one strict slot payload."""

        if not isinstance(value, Mapping):
            raise BindingError("allocation slot must be a mapping")
        expected = {"slot_id", "node", "devices", "cpus", "threads", "environment"}
        if set(value) != expected:
            missing, extra = expected - set(value), set(value) - expected
            raise BindingError(f"allocation slot keys differ (missing={sorted(missing)!r}, extra={sorted(extra)!r})")
        devices = _string_tuple(value["devices"], "slot devices")
        cpus = _int_tuple(value["cpus"], "slot cpus")
        threads = value["threads"]
        environment = value["environment"]
        if not isinstance(threads, Mapping) or not isinstance(environment, Mapping):
            raise BindingError("slot threads and environment must be mappings")
        return cls(value["slot_id"], value["node"], devices, cpus, tuple(threads.items()), tuple(environment.items()))


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    """Validate one non-string sequence of non-empty strings."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BindingError(f"{name} must be a sequence")
    result = tuple(value)
    if any(type(item) is not str or not item for item in result):
        raise BindingError(f"{name} must contain non-empty strings")
    return result


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    """Validate one non-string sequence of integer entitlements."""

    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BindingError(f"{name} must be a sequence")
    result = tuple(value)
    if any(type(item) is not int or isinstance(item, bool) for item in result):
        raise BindingError(f"{name} must contain integers")
    return result


def slots_from_allocation(allocation: ConfiguredAllocation) -> tuple[AllocationSlot, ...]:
    """Decode and strictly validate the optional versioned slot record.

    Parameters
    ----------
    allocation : ConfiguredAllocation
        Allocation facts containing the optional versioned ``slots`` record.

    Returns
    -------
    tuple of AllocationSlot
        Immutable, disjoint slot entitlements in declared order.

    Raises
    ------
    BindingError
        If the record is absent, malformed, overlapping, or names unsupported
        multi-node geometry.
    """

    if not isinstance(allocation, ConfiguredAllocation):
        raise BindingError("slot resolution requires ConfiguredAllocation")
    raw = allocation.facts.get("slots")
    if raw is None:
        return ()
    if not isinstance(raw, Mapping) or set(raw) != {"schema", "slots"} or raw.get("schema") != _SLOT_SCHEMA:
        raise BindingError(f"allocation slots must use schema {_SLOT_SCHEMA!r}")
    entries = raw["slots"]
    if isinstance(entries, (str, bytes, bytearray)) or not isinstance(entries, Sequence) or not entries:
        raise BindingError("allocation slots must be a non-empty sequence")
    decoded = tuple(AllocationSlot.from_mapping(entry) for entry in entries)
    slot_ids = [slot.slot_id for slot in decoded]
    if len(set(slot_ids)) != len(slot_ids):
        raise BindingError("allocation slots must have unique slot_id values")
    devices = [device for slot in decoded for device in slot.devices]
    if len(set(devices)) != len(devices):
        raise BindingError("allocation slots must have disjoint device entitlements")
    cpus = [cpu for slot in decoded for cpu in slot.cpus]
    if len(set(cpus)) != len(cpus):
        raise BindingError("allocation slots must have disjoint CPU entitlements")
    nodes = [slot.node for slot in decoded]
    if any(type(node) is not str or not node.strip() for node in nodes):
        raise BindingError("allocation slots must identify unambiguous nodes")
    if len(set(nodes)) != 1:
        raise BindingError("allocation slots must identify exactly one attached worker node")
    return decoded


def bindings_from_allocation(allocation: ConfiguredAllocation) -> tuple[WorkerBinding, ...]:
    """Return per-slot bindings, or the legacy one-binding fallback.

    Parameters
    ----------
    allocation : ConfiguredAllocation
        Allocation facts to resolve.

    Returns
    -------
    tuple of WorkerBinding
        One immutable worker binding per declared slot, or one legacy binding.

    Raises
    ------
    BindingError
        If the slot record or legacy binding is not admissible.
    """

    slots = slots_from_allocation(allocation)
    if slots:
        return tuple(slot.binding() for slot in slots)
    return (binding_from_allocation(allocation),)


def binding_from_allocation(allocation: ConfiguredAllocation) -> WorkerBinding:
    """Resolve an explicit worker binding from configured allocation facts.

    Accepted fact shapes are ``binding``/``worker_binding`` mappings or a
    mapping under ``environment`` plus optional ``visibility`` fields. A
    missing shape is a typed refusal; scheduler discovery and facility-shaped
    nodefile parsing remain outside this executor module.
    """

    if not isinstance(allocation, ConfiguredAllocation):
        raise BindingError("binding resolution requires ConfiguredAllocation")
    facts = allocation.facts
    raw = facts.get("binding", facts.get("worker_binding"))
    if raw is None:
        raw = {
            key: facts[key]
            for key in ("environment", "env", "visibility", "accelerator_visibility", "worker_id", "node", "nodes")
            if key in facts
        }
    if not raw:
        raise BindingError("configured allocation has no worker binding")
    return WorkerBinding.from_mapping(raw)


__all__ = [
    "AllocationSlot",
    "BindingError",
    "CHILD_PLACEMENT_SCHEMA",
    "ChildPlacementObservation",
    "PLACEMENT_OUTPUT_ENV",
    "PlacementRefusal",
    "WorkerBinding",
    "binding_from_allocation",
    "bindings_from_allocation",
    "parse_cpu_set",
    "parse_proc_start_ticks",
    "placement_environment",
    "read_child_placement",
    "slots_from_allocation",
    "utc_timestamp",
]
