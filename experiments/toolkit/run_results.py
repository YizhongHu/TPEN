"""Strict, executor-independent readers for scientific run manifests.

The toolkit deliberately decodes the published JSON instead of importing the
executor package.  The optional inspection callback is the public core
inspection seam; the default callback is loaded lazily so the toolkit remains
usable in the root experiment environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import string
from types import MappingProxyType
from typing import Any, Callable, Mapping


RUN_MANIFEST_V1 = "tpen-executor/run-manifest-v1"
RUN_MANIFEST_V2 = "tpen-executor/run-manifest-v2"
RUN_MANIFEST_V3 = "tpen-executor/run-manifest-v3"
ACCEPTED_SEGMENT_SCHEMA = "tpen-executor/accepted-segment-v1"
SCIENCE_PLACEMENT_RECEIPT_SCHEMA = "tpen-executor/science-placement-receipt-v1"
SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2 = "tpen-executor/science-placement-receipt-v2"
OUTCOME_STATUSES = frozenset({"completed", "failed", "timed_out", "launch_failed"})
PLACEMENT_COMPARISON_STATES = frozenset({"matched", "mismatch", "unavailable", "refused"})
_HEX_DIGITS = frozenset(string.hexdigits.lower())
_GPU_UUID = re.compile(r"^GPU-[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$")


class ManifestRefusal(ValueError):
    """Typed refusal for an unreadable or scientifically inconsistent manifest.

    Parameters
    ----------
    code : str
        Stable machine-readable refusal category.
    reason : str
        Human-readable explanation of the rejected manifest or evidence.
    """

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


ManifestReadError = ManifestRefusal


@dataclass(frozen=True, slots=True)
class CommittedSegment:
    """A selected segment projected from one accepted executor record.

    Parameters
    ----------
    task_id, attempt_id : str
        Logical task and executor attempt identities.
    predecessor_checkpoint : str or None
        Prior checkpoint in the committed chain, or ``None`` for a fresh start.
    checkpoint_dir : str
        Core-published checkpoint directory.
    next_iteration, completed_updates : int
        Nonnegative progress coordinates copied from core evidence.
    placement_receipt : mapping or None
        Independently validated operational placement evidence for the attempt,
        when the executor published it.  This does not change scientific
        segment selection or constitute a placement qualification verdict.
    """

    task_id: str
    attempt_id: str
    predecessor_checkpoint: str | None
    checkpoint_dir: str
    next_iteration: int
    completed_updates: int
    placement_receipt: Mapping[str, Any] | None = None
    resolved_config_identity: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class CommittedChain:
    """The exactly-once accepted segment chain for one logical task.

    Parameters
    ----------
    task_id : str
        Logical task identity.
    segments : tuple of CommittedSegment
        Ordered, predecessor-linked accepted segments; no attempt is repeated.
    """

    task_id: str
    segments: tuple[CommittedSegment, ...]


@dataclass(frozen=True, slots=True)
class PreservedAttempt:
    """A manifest attempt retained as evidence but excluded from selection.

    Parameters
    ----------
    task_id, attempt_id, output_dir : str
        Original executor attempt identity and fenced output directory.
    outcome : str or None
        Published operational outcome status, if one exists.
    reason : str, default="not-selected"
        Why the attempt is preserved rather than counted in a committed chain.
    placement_receipt : mapping or None
        Independently validated operational placement evidence for the attempt.
    """

    task_id: str
    attempt_id: str
    output_dir: str
    outcome: str | None
    reason: str = "not-selected"
    placement_receipt: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ResultSet:
    """Selected chains plus original attempts that were not selected.

    Parameters
    ----------
    chains : tuple of CommittedChain
        Exactly-once committed coordinate chains grouped by task.
    preserved_attempts : tuple of PreservedAttempt
        Original non-selected attempts retained as evidence and never counted.
    """

    chains: tuple[CommittedChain, ...]
    preserved_attempts: tuple[PreservedAttempt, ...]
    placement_verdict: Mapping[str, Any] | None = None

    @property
    def chains_by_task(self) -> Mapping[str, CommittedChain]:
        """Return a task-indexed view without changing the immutable result."""

        return MappingProxyType({chain.task_id: chain for chain in self.chains})

    @property
    def non_selected_attempts(self) -> tuple[PreservedAttempt, ...]:
        """Compatibility name for the preserved-evidence collection."""

        return self.preserved_attempts

    @property
    def preserved_evidence(self) -> tuple[PreservedAttempt, ...]:
        """Return original attempts retained as preserved evidence."""

        return self.preserved_attempts

    @property
    def placement_unqualified(self) -> bool:
        """Whether the manifest carries a loud, non-science-qualified verdict."""

        return self.placement_verdict is not None and self.placement_verdict.get("status") == "unqualified"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestRefusal("duplicate-json-key", f"duplicate JSON member {key!r}")
        value[key] = item
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except ManifestRefusal:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise ManifestRefusal("malformed-json", f"cannot decode manifest JSON: {error}") from error
    if not isinstance(value, dict):
        raise ManifestRefusal("manifest-not-object", "manifest JSON must be an object")
    return value


def _fields(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestRefusal("malformed-record", f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise ManifestRefusal(
            "malformed-record",
            f"{name} keys differ (missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r})",
        )
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ManifestRefusal("malformed-record", f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ManifestRefusal("malformed-record", f"{name} must be a nonnegative integer")
    return value


def _sha256(value: Any, name: str) -> str:
    """Validate the lowercase SHA-256 wire representation used by executor."""

    _text(value, name)
    if len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ManifestRefusal("malformed-record", f"{name} must be a lowercase SHA-256 digest")
    if value != value.lower():
        raise ManifestRefusal("malformed-record", f"{name} must be a lowercase SHA-256 digest")
    return value


def _gpu_uuid(value: Any, name: str) -> str:
    """Validate the physical GPU UUID grammar used by v2 receipts."""

    _text(value, name)
    if _GPU_UUID.fullmatch(value) is None:
        raise ManifestRefusal("malformed-record", f"{name} must be a canonical physical GPU UUID")
    return value


def _receipt_host_key(value: Any, name: str) -> str:
    """Derive the executor's physical-host comparison key."""

    _text(value, name)
    complete = value.strip().lower().rstrip(".")
    labels = complete.split(".")
    if not complete or any(not label for label in labels):
        raise ManifestRefusal("malformed-record", f"{name} contains an empty hostname label")
    return labels[0]


def _finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float | int:
    if type(value) not in {int, float} or isinstance(value, bool) or not math.isfinite(value):
        raise ManifestRefusal("malformed-record", f"{name} must be a finite number")
    if nonnegative and value < 0:
        raise ManifestRefusal("malformed-record", f"{name} must be nonnegative")
    return value


def _validate_json(value: Any, path: str = "placement") -> None:
    """Reject non-JSON-shaped or non-finite placement evidence."""

    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ManifestRefusal("non-finite-placement-evidence", f"{path} contains a non-finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise ManifestRefusal("non-string-placement-key", f"{path} contains a non-string key")
            _validate_json(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item, f"{path}[]")
        return
    raise ManifestRefusal("malformed-placement-evidence", f"{path} is not JSON-shaped")


def _freeze_json(value: Any) -> Any:
    """Return immutable views of validated JSON-shaped placement evidence."""

    _validate_json(value)

    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _placement_v2_host_mapping(value: Any) -> dict[str, Any]:
    record = dict(_fields(value, {"status", "reason", "source", "slot_id", "host"}, "placement host mapping"))
    for key in ("reason", "slot_id", "host"):
        _text(record[key], f"placement.host_mapping.{key}")
    _text(record["status"], "placement.host_mapping.status")
    if record["status"] not in {"matched", "mismatch", "unavailable"}:
        raise ManifestRefusal("malformed-record", "placement.host_mapping.status is invalid")
    _text(record["source"], "placement.host_mapping.source")
    if record["source"] not in {"scheduler-node-map", "declared-slot"}:
        raise ManifestRefusal("malformed-record", "placement.host_mapping.source is invalid")
    if record["status"] == "unavailable" and record["host"] != "unavailable":
        raise ManifestRefusal("malformed-record", "unavailable host mapping must not name a logical host")
    if record["status"] != "unavailable" and record["source"] != "scheduler-node-map":
        raise ManifestRefusal("malformed-record", "mapped host evidence must come from the scheduler node map")
    return record


def _placement_v2_verdict(value: Any, declared: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(_fields(
        value,
        {"host", "device", "qualification", "reason_class", "reason", "science_acceptance", "warning"},
        "placement verdict",
    ))
    for key in ("host", "device"):
        _text(record[key], f"placement.verdict.{key}")
        if record[key] not in PLACEMENT_COMPARISON_STATES:
            raise ManifestRefusal("malformed-record", f"placement.verdict.{key} is invalid")
    _text(record["qualification"], "placement.verdict.qualification")
    if record["qualification"] not in {"qualified", "unqualified"}:
        raise ManifestRefusal("malformed-record", "placement qualification is invalid")
    _text(record["reason_class"], "placement.verdict.reason_class")
    if record["reason_class"] not in {"qualified", "mismatch", "telemetry-failed", "not-sampled", "refused"}:
        raise ManifestRefusal("malformed-record", "placement verdict reason class is invalid")
    _text(record["reason"], "placement.verdict.reason")
    _text(record["science_acceptance"], "placement.verdict.science_acceptance")
    if record["science_acceptance"] not in {"accepted", "not-accepted"}:
        raise ManifestRefusal("malformed-record", "placement.verdict.science_acceptance is invalid")
    if type(record["warning"]) is not str:
        raise ManifestRefusal("malformed-record", "placement.verdict.warning must be text")
    expected_acceptance = "accepted" if record["reason_class"] in {"qualified", "not-sampled"} else "not-accepted"
    if record["science_acceptance"] != expected_acceptance:
        raise ManifestRefusal("malformed-record", "placement verdict science acceptance contradicts reason class")
    if declared["devices"]:
        if not record["warning"]:
            raise ManifestRefusal("unqualified-placement-not-loud", "declared GPU placement requires a loud warning")
        if record["reason_class"] == "not-sampled" and "GPU PLACEMENT NOT-SAMPLED" not in record["warning"]:
            raise ManifestRefusal("unqualified-placement-not-loud", "not-sampled placement warning is not loud")
        if record["reason_class"] in {"mismatch", "telemetry-failed"} and "GPU PLACEMENT UNQUALIFIED" not in record["warning"]:
            raise ManifestRefusal("unqualified-placement-not-loud", "mismatched placement warning is not loud")
        if record["reason_class"] == "refused" and not any(
            phrase in record["warning"]
            for phrase in ("GPU PLACEMENT REFUSED", "GPU PLACEMENT UNQUALIFIED")
        ):
            raise ManifestRefusal("unqualified-placement-not-loud", "refused placement warning is not loud")
    return record


def _placement_v2_sampler(value: Any) -> dict[str, Any]:
    expected = {"status", "reason", "method", "version", "cadence_seconds", "count", "monotonic_start", "monotonic_end", "inventory", "pid_rows", "samples", "command_pids", "survivors", "elapsed_seconds", "join_timeout_seconds", "quiescent"}
    record = dict(_fields(value, expected, "placement gpu sampler"))
    if record["status"] not in {"observed", "unavailable"} or record["method"] != "nvidia-smi-pid-uuid-v1":
        raise ManifestRefusal("malformed-record", "placement gpu sampler status or method is invalid")
    _text(record["reason"], "placement.gpu_sampler.reason")
    if record["version"] is not None:
        _text(record["version"], "placement.gpu_sampler.version")
    for key in ("cadence_seconds", "elapsed_seconds", "join_timeout_seconds"):
        _finite_number(record[key], f"placement.gpu_sampler.{key}", nonnegative=True)
    if not 1.0 <= record["join_timeout_seconds"] <= 4.0:
        raise ManifestRefusal("malformed-record", "placement gpu sampler join bound is invalid")
    _nonnegative_int(record["count"], "placement.gpu_sampler.count")
    if record["count"] > 16 or type(record["quiescent"]) is not bool:
        raise ManifestRefusal("malformed-record", "placement gpu sampler count or quiescence is invalid")
    for key in ("inventory", "pid_rows", "samples", "command_pids", "survivors"):
        if isinstance(record[key], (str, bytes, bytearray)) or not isinstance(record[key], list):
            raise ManifestRefusal("malformed-record", f"placement.gpu_sampler.{key} must be a sequence")
    if record["quiescent"] and record["survivors"]:
        raise ManifestRefusal("malformed-record", "quiescent sampler cannot report survivors")
    for item in record["inventory"]:
        if set(item) != {"uuid", "pci.bus_id"}:
            raise ManifestRefusal("malformed-record", "placement gpu inventory row is invalid")
        _gpu_uuid(item["uuid"], "placement.gpu_sampler.inventory.uuid")
        _text(item["pci.bus_id"], "placement.gpu_sampler.inventory.pci.bus_id")
    for item in record["pid_rows"]:
        if set(item) != {"pid", "gpu_uuid"}:
            raise ManifestRefusal("malformed-record", "placement gpu PID row is invalid")
        _nonnegative_int(item["pid"], "placement.gpu_sampler.pid_rows.pid")
        if item["pid"] == 0:
            raise ManifestRefusal("malformed-record", "placement gpu PID must be positive")
        _gpu_uuid(item["gpu_uuid"], "placement.gpu_sampler.pid_rows.gpu_uuid")
    for pid in record["command_pids"]:
        if type(pid) is not int or pid <= 0:
            raise ManifestRefusal("malformed-record", "placement.gpu_sampler command PID must be positive")
    for sample in record["samples"]:
        _fields(sample, {"monotonic_start", "monotonic_end", "inventory", "pid_rows"}, "placement gpu sample")
        _finite_number(sample["monotonic_start"], "placement gpu sample start")
        _finite_number(sample["monotonic_end"], "placement gpu sample end")
        if not isinstance(sample["inventory"], list) or not isinstance(sample["pid_rows"], list):
            raise ManifestRefusal("malformed-record", "placement gpu sample rows must be sequences")
        for item in sample["inventory"]:
            _fields(item, {"uuid", "pci.bus_id"}, "placement gpu sample inventory")
            _gpu_uuid(item["uuid"], "placement gpu sample inventory uuid")
            _text(item["pci.bus_id"], "placement gpu sample inventory pci.bus_id")
        for item in sample["pid_rows"]:
            _fields(item, {"pid", "gpu_uuid"}, "placement gpu sample PID row")
            if type(item["pid"]) is not int or item["pid"] <= 0:
                raise ManifestRefusal("malformed-record", "placement gpu sample PID must be positive")
            _gpu_uuid(item["gpu_uuid"], "placement gpu sample PID uuid")
    if record["status"] == "observed" and (record["count"] <= 0 or record["count"] != len(record["samples"]) or not record["command_pids"]):
        raise ManifestRefusal("malformed-record", "observed sampler is incomplete")
    if record["count"] != len(record["samples"]):
        raise ManifestRefusal("malformed-record", "sampler count disagrees with samples")
    return record


def _placement_receipt(value: Any, attempt: Mapping[str, Any], start: Mapping[str, Any], group_digest: str) -> dict[str, Any]:
    """Decode an optional v1/v2 placement receipt with closed schemas."""

    _validate_json(value)
    schema = value.get("schema") if isinstance(value, dict) else None
    if schema == SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2:
        record = dict(_fields(
            value,
            {
                "schema", "task_id", "attempt_id", "group_digest", "allocation_id",
                "backend_job_id", "slot_id", "payload_digest", "output_dir", "child",
                "declared", "gpu", "comparison", "refusal", "host_mapping", "gpu_sampler", "verdict",
            },
            "placement receipt v2",
        ))
        base = dict(record)
        base.pop("host_mapping")
        base.pop("gpu_sampler")
        base.pop("verdict")
        base["schema"] = SCIENCE_PLACEMENT_RECEIPT_SCHEMA
        base = _placement_receipt(base, attempt, start, group_digest)
        record["child"] = base["child"]
        record["declared"] = base["declared"]
        record["gpu"] = base["gpu"]
        record["comparison"] = base["comparison"]
        record["refusal"] = base["refusal"]
        record["host_mapping"] = _placement_v2_host_mapping(record["host_mapping"])
        record["gpu_sampler"] = _placement_v2_sampler(record["gpu_sampler"])
        record["verdict"] = _placement_v2_verdict(record["verdict"], record["declared"])
        if record["verdict"]["reason_class"] == "not-sampled" and (record["gpu_sampler"]["reason"] != "not-sampled" or record["gpu_sampler"]["count"]):
            raise ManifestRefusal("malformed-record", "not-sampled verdict contains sampler evidence")
        if record["verdict"]["reason_class"] == "telemetry-failed" and record["gpu_sampler"]["reason"] != "telemetry-failed":
            raise ManifestRefusal("malformed-record", "telemetry-failed verdict lacks failed sampler evidence")
        if record["refusal"] is None:
            for device in record["declared"]["devices"]:
                _gpu_uuid(device, "placement.declared.devices")
        if record["host_mapping"]["slot_id"] != record["slot_id"]:
            raise ManifestRefusal("malformed-record", "host mapping slot_id disagrees with the base receipt")
        if record["host_mapping"]["status"] != record["verdict"]["host"]:
            raise ManifestRefusal("malformed-record", "host mapping disagrees with placement verdict")
        host_comparison = record["comparison"]["host"]
        host_status = record["verdict"]["host"]
        if host_comparison != host_status and not (host_status == "unavailable" and host_comparison == "refused"):
            raise ManifestRefusal("malformed-record", "placement comparison disagrees with placement verdict")
        if record["comparison"]["device"] != record["verdict"]["device"]:
            raise ManifestRefusal("malformed-record", "placement comparison disagrees with placement verdict")

        child_host = _receipt_host_key(record["child"]["host"], "placement.child.host")
        declared_host = _receipt_host_key(record["declared"]["host"], "placement.declared.host")
        affinity_matches = bool(record["child"]["affinity"]) and (
            tuple(sorted(record["child"]["affinity"])) == tuple(sorted(record["declared"]["cpus"]))
        )
        if host_status == "unavailable":
            derived_host_status = "unavailable"
            if host_comparison not in {"unavailable", "refused"}:
                raise ManifestRefusal("malformed-record", "unavailable host map contradicts base host comparison")
        else:
            mapped_host = _receipt_host_key(record["host_mapping"]["host"], "placement.host_mapping.host")
            if declared_host != mapped_host:
                raise ManifestRefusal("malformed-record", "declared host differs from physical scheduler host")
            derived_host_status = "matched" if child_host == mapped_host else "mismatch"
            if host_status != derived_host_status or host_comparison != derived_host_status:
                raise ManifestRefusal("malformed-record", "host mapping contradicts child host evidence")
        derived_affinity_status = "matched" if affinity_matches else (
            "mismatch" if record["child"]["affinity"] else "unavailable"
        )
        if record["refusal"] is None and record["comparison"]["affinity"] != derived_affinity_status:
            raise ManifestRefusal("malformed-record", "affinity comparison is not derived from child evidence")
        if record["refusal"] is not None and record["comparison"]["affinity"] not in {
            derived_affinity_status, "refused"
        }:
            raise ManifestRefusal("malformed-record", "refused affinity comparison contradicts child evidence")
        has_postlaunch_mismatch = derived_host_status == "mismatch" or derived_affinity_status == "mismatch"
        if record["verdict"]["qualification"] == "qualified" and record["gpu_sampler"]["status"] == "observed":
            expected_reason_class = "qualified"
        elif has_postlaunch_mismatch or record["verdict"]["device"] == "mismatch":
            expected_reason_class = "mismatch"
        elif record["refusal"] is not None and record["gpu_sampler"]["reason"] != "telemetry-failed":
            expected_reason_class = "refused"
        elif record["gpu_sampler"]["reason"] == "not-sampled":
            expected_reason_class = "not-sampled"
        elif record["gpu_sampler"]["reason"] == "telemetry-failed":
            expected_reason_class = "telemetry-failed"
        else:
            expected_reason_class = "refused"
        if record["verdict"]["reason_class"] != expected_reason_class:
            raise ManifestRefusal("malformed-record", "placement verdict does not preserve mismatch/refusal precedence")
        if expected_reason_class == "not-sampled":
            if (
                record["refusal"] is not None
                or host_status != "matched"
                or derived_host_status != "matched"
                or host_comparison != "matched"
                or derived_affinity_status != "matched"
            ):
                raise ManifestRefusal("malformed-record", "not-sampled verdict requires matching mapped host and affinity evidence")
        elif expected_reason_class == "mismatch":
            if not has_postlaunch_mismatch and record["verdict"]["device"] != "mismatch":
                raise ManifestRefusal("malformed-record", "mismatch verdict requires post-launch mismatch evidence")
        elif expected_reason_class == "telemetry-failed":
            if record["gpu_sampler"]["reason"] != "telemetry-failed":
                raise ManifestRefusal("malformed-record", "telemetry-failed verdict lacks failed sampler evidence")
        elif expected_reason_class == "refused":
            if record["refusal"] is None or has_postlaunch_mismatch or host_status != "unavailable":
                raise ManifestRefusal("malformed-record", "refused verdict requires a prelaunch refusal")
        return record

    record = dict(_fields(
        value,
        {
            "schema", "task_id", "attempt_id", "group_digest", "allocation_id",
            "backend_job_id", "slot_id", "payload_digest", "output_dir", "child",
            "declared", "gpu", "comparison", "refusal",
        },
        "placement receipt",
    ))
    if record["schema"] != SCIENCE_PLACEMENT_RECEIPT_SCHEMA:
        raise ManifestRefusal("unknown-schema", "placement receipt schema is unsupported")
    for key in ("task_id", "attempt_id", "allocation_id", "backend_job_id", "slot_id", "output_dir"):
        _text(record[key], f"placement.{key}")
    _sha256(record["group_digest"], "placement.group_digest")
    _sha256(record["payload_digest"], "placement.payload_digest")

    child = dict(_fields(
        record["child"],
        {"host", "pid", "proc_start_ticks", "affinity", "thread_settings", "cuda_visible_devices", "observed_at_utc"},
        "placement child",
    ))
    _text(child["host"], "placement.child.host")
    _nonnegative_int(child["pid"], "placement.child.pid")
    _nonnegative_int(child["proc_start_ticks"], "placement.child.proc_start_ticks")
    affinity = child["affinity"]
    if isinstance(affinity, (str, bytes, bytearray)) or not isinstance(affinity, list):
        raise ManifestRefusal("malformed-record", "placement.child.affinity must be a sequence")
    if any(type(cpu) is not int or cpu < 0 for cpu in affinity):
        raise ManifestRefusal("malformed-record", "placement.child.affinity must contain nonnegative integers")
    thread_settings = child["thread_settings"]
    if not isinstance(thread_settings, dict) or any(
        type(key) is not str or (type(item) is not str and item is not None)
        for key, item in thread_settings.items()
    ):
        raise ManifestRefusal("malformed-record", "placement.child.thread_settings is invalid")
    if child["cuda_visible_devices"] is not None and type(child["cuda_visible_devices"]) is not str:
        raise ManifestRefusal("malformed-record", "placement.child.cuda_visible_devices must be text or null")
    _text(child["observed_at_utc"], "placement.child.observed_at_utc")

    declared = dict(_fields(record["declared"], {"host", "cpus", "devices"}, "placement declared"))
    _text(declared["host"], "placement.declared.host")
    cpus = declared["cpus"]
    if isinstance(cpus, (str, bytes, bytearray)) or not isinstance(cpus, list) or not cpus:
        raise ManifestRefusal("malformed-record", "placement.declared.cpus must be a non-empty sequence")
    if any(type(cpu) is not int or cpu < 0 for cpu in cpus):
        raise ManifestRefusal("malformed-record", "placement.declared.cpus must contain nonnegative integers")
    devices = declared["devices"]
    if isinstance(devices, (str, bytes, bytearray)) or not isinstance(devices, list):
        raise ManifestRefusal("malformed-record", "placement.declared.devices must be a sequence")
    if any(type(device) is not str or not device for device in devices):
        raise ManifestRefusal("malformed-record", "placement.declared.devices must contain non-empty strings")

    gpu = dict(_fields(
        record["gpu"],
        {"status", "reason", "method", "started_at_utc", "finished_at_utc", "samples", "active_uuids", "active_pci"},
        "placement gpu",
    ))
    for key in ("status", "reason", "method", "started_at_utc", "finished_at_utc"):
        _text(gpu[key], f"placement.gpu.{key}")
    if gpu["status"] not in {"unavailable", "qualified"}:
        raise ManifestRefusal("malformed-record", "placement.gpu.status is invalid")
    if any(isinstance(gpu[key], (str, bytes, bytearray)) or not isinstance(gpu[key], list) for key in ("samples", "active_uuids", "active_pci")):
        raise ManifestRefusal("malformed-record", "placement.gpu sequences are invalid")
    if any(type(item) is not str or not item for key in ("active_uuids", "active_pci") for item in gpu[key]):
        raise ManifestRefusal("malformed-record", "placement.gpu active device values must be non-empty strings")

    comparison = dict(_fields(record["comparison"], {"host", "affinity", "device"}, "placement comparison"))
    for key in ("host", "affinity", "device"):
        _text(comparison[key], f"placement.comparison.{key}")
        if comparison[key] not in PLACEMENT_COMPARISON_STATES:
            raise ManifestRefusal("malformed-record", "placement comparison values use an unknown state")

    refusal = record["refusal"]
    if refusal is not None:
        refusal = dict(_fields(refusal, {"code", "reason"}, "placement refusal"))
        _text(refusal["code"], "placement.refusal.code")
        _text(refusal["reason"], "placement.refusal.reason")
    if child["pid"] == 0 and refusal is None:
        raise ManifestRefusal("malformed-record", "placement.child.pid must be positive unless placement was refused")
    if not affinity and refusal is None:
        raise ManifestRefusal("malformed-record", "placement.child.affinity cannot be empty without refusal")

    if (
        record["task_id"], record["attempt_id"], record["group_digest"], record["allocation_id"],
        record["backend_job_id"], record["output_dir"],
    ) != (
        attempt["task_id"], attempt["attempt_id"], group_digest, start["allocation_id"],
        start["backend_job_id"], attempt["output_dir"],
    ):
        raise ManifestRefusal("attempt-receipt-mismatch", "manifest attempt disagrees with its placement receipt")
    record["child"] = child
    record["declared"] = declared
    record["gpu"] = gpu
    record["comparison"] = comparison
    record["refusal"] = refusal
    return record


def _config_identity(value: Any) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"schema", "sha256"}:
        raise ManifestRefusal("malformed-record", "segment.resolved_config_identity has invalid keys")
    if value["schema"] != "tpen-config-identity-v1":
        raise ManifestRefusal("unknown-schema", "resolved-config identity schema is unsupported")
    digest = _sha256(value["sha256"], "segment.resolved_config_identity.sha256")
    return MappingProxyType({"schema": value["schema"], "sha256": digest})


def _attempt(value: Any, group_digest: str) -> dict[str, Any]:
    expected = {"schema", "task_id", "attempt_id", "start_receipt", "outcome_receipt", "output_dir"}
    if isinstance(value, dict) and "placement_receipt" in value:
        expected.add("placement_receipt")
    record = dict(_fields(value, expected, "manifest attempt"))
    if record["schema"] != "tpen-executor/manifest-attempt-v1":
        raise ManifestRefusal("unknown-schema", "manifest attempt schema is unsupported")
    for key in ("task_id", "attempt_id", "output_dir"):
        _text(record[key], f"attempt.{key}")
    start = dict(_fields(record["start_receipt"], {"schema", "task_id", "attempt_id", "allocation_id", "backend_job_id", "group_digest", "started_at_utc", "output_dir"}, "start receipt"))
    if start["schema"] != "tpen-executor/start-receipt-v1":
        raise ManifestRefusal("unknown-schema", "start receipt schema is unsupported")
    for key in ("task_id", "attempt_id", "allocation_id", "backend_job_id", "started_at_utc", "output_dir"):
        _text(start[key], f"start.{key}")
    _sha256(start["group_digest"], "start.group_digest")
    if (start["task_id"], start["attempt_id"], start["output_dir"]) != (record["task_id"], record["attempt_id"], record["output_dir"]):
        raise ManifestRefusal("attempt-receipt-mismatch", "manifest attempt disagrees with its start receipt")
    outcome = record["outcome_receipt"]
    if outcome is not None:
        outcome = dict(_fields(outcome, {"schema", "task_id", "attempt_id", "group_digest", "finished_at_utc", "status", "output_dir", "result_digest"}, "outcome receipt"))
        if outcome["schema"] != "tpen-executor/outcome-receipt-v1":
            raise ManifestRefusal("unknown-schema", "outcome receipt schema is unsupported")
        for key in ("task_id", "attempt_id", "finished_at_utc", "status", "output_dir"):
            _text(outcome[key], f"outcome.{key}")
        _sha256(outcome["group_digest"], "outcome.group_digest")
        if outcome["status"] not in OUTCOME_STATUSES:
            raise ManifestRefusal("malformed-record", "outcome.status is not a supported operational status")
        if outcome["result_digest"] is not None:
            _sha256(outcome["result_digest"], "outcome.result_digest")
        if (
            outcome["task_id"], outcome["attempt_id"], outcome["group_digest"], outcome["output_dir"]
        ) != (
            record["task_id"], record["attempt_id"], start["group_digest"], record["output_dir"]
        ):
            raise ManifestRefusal("attempt-receipt-mismatch", "manifest attempt disagrees with its outcome receipt")
    record["outcome_receipt"] = outcome
    placement = None if "placement_receipt" not in record or record["placement_receipt"] is None else _placement_receipt(record["placement_receipt"], record, start, group_digest)
    record["placement_receipt"] = placement
    return record


def _segment(value: Any) -> dict[str, Any]:
    base = {"schema", "task_id", "attempt_id", "predecessor_checkpoint", "checkpoint_dir", "next_iteration", "completed_updates"}
    actual_fields = set(value) if isinstance(value, dict) else None
    if actual_fields not in (base, base | {"resolved_config_identity"}):
        raise ManifestRefusal("malformed-record", "accepted segment keys differ")
    if "resolved_config_identity" in value and value["resolved_config_identity"] is None:
        raise ManifestRefusal("malformed-record", "segment.resolved_config_identity must not be null")
    record = dict(value)
    if record["schema"] != ACCEPTED_SEGMENT_SCHEMA:
        raise ManifestRefusal("unknown-schema", "accepted segment schema is unsupported")
    _text(record["task_id"], "segment.task_id")
    _text(record["attempt_id"], "segment.attempt_id")
    if record["predecessor_checkpoint"] is not None:
        _text(record["predecessor_checkpoint"], "segment.predecessor_checkpoint")
    _text(record["checkpoint_dir"], "segment.checkpoint_dir")
    _nonnegative_int(record["next_iteration"], "segment.next_iteration")
    _nonnegative_int(record["completed_updates"], "segment.completed_updates")
    record["resolved_config_identity"] = _config_identity(record.get("resolved_config_identity"))
    return record


def _manifest(path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any] | None]:
    value = _load_json(path)
    schema = value.get("schema")
    if schema == RUN_MANIFEST_V1:
        _fields(value, {"schema", "group_id", "group_digest", "attempts"}, "run manifest")
        raw_segments: tuple[dict[str, Any], ...] = ()
    elif schema == RUN_MANIFEST_V2:
        _fields(value, {"schema", "group_id", "group_digest", "attempts", "segments"}, "run manifest")
        if isinstance(value["segments"], (str, bytes, bytearray)) or not isinstance(value["segments"], list):
            raise ManifestRefusal("malformed-record", "manifest segments must be a sequence")
        raw_segments = tuple(_segment(item) for item in value["segments"])
        placement_verdict = None
    elif schema == RUN_MANIFEST_V3:
        _fields(value, {"schema", "group_id", "group_digest", "attempts", "segments", "placement_verdict"}, "run manifest")
        if isinstance(value["segments"], (str, bytes, bytearray)) or not isinstance(value["segments"], list):
            raise ManifestRefusal("malformed-record", "manifest segments must be a sequence")
        raw_segments = tuple(_segment(item) for item in value["segments"])
        _validate_json(value["placement_verdict"], "manifest.placement_verdict")
        if not isinstance(value["placement_verdict"], dict):
            raise ManifestRefusal("malformed-record", "manifest placement verdict must be an object")
        placement_verdict = dict(value["placement_verdict"])
    else:
        raise ManifestRefusal("unknown-schema", f"unsupported run manifest schema {schema!r}")
    if schema == RUN_MANIFEST_V1:
        placement_verdict = None
    _text(value["group_id"], "manifest.group_id")
    _sha256(value["group_digest"], "manifest.group_digest")
    if isinstance(value["attempts"], (str, bytes, bytearray)) or not isinstance(value["attempts"], list):
        raise ManifestRefusal("malformed-record", "manifest attempts must be a sequence")
    attempts = tuple(_attempt(item, value["group_digest"]) for item in value["attempts"])
    for attempt in attempts:
        if attempt["start_receipt"]["group_digest"] != value["group_digest"]:
            raise ManifestRefusal("group-digest-mismatch", "start receipt group_digest differs from manifest")
        if (
            attempt["outcome_receipt"] is not None
            and attempt["outcome_receipt"]["group_digest"] != value["group_digest"]
        ):
            raise ManifestRefusal("group-digest-mismatch", "outcome receipt group_digest differs from manifest")
    if len({item["attempt_id"] for item in attempts}) != len(attempts):
        raise ManifestRefusal("duplicate-attempt", "manifest contains duplicate attempt identities")
    actual_blocking = {
        (
            attempt["task_id"],
            attempt["attempt_id"],
            attempt["placement_receipt"]["verdict"]["reason_class"],
            attempt["placement_receipt"]["verdict"]["reason"],
        )
        for attempt in attempts
        if attempt["placement_receipt"] is not None
        and attempt["placement_receipt"]["schema"] == SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2
        and attempt["placement_receipt"]["declared"]["devices"]
        and attempt["placement_receipt"]["verdict"]["qualification"] == "unqualified"
        and attempt["placement_receipt"]["verdict"]["reason_class"] in {"mismatch", "refused", "telemetry-failed"}
    }
    actual_not_sampled = {
        (
            attempt["task_id"],
            attempt["attempt_id"],
            attempt["placement_receipt"]["verdict"]["reason_class"],
            attempt["placement_receipt"]["verdict"]["reason"],
        )
        for attempt in attempts
        if attempt["placement_receipt"] is not None
        and attempt["placement_receipt"]["schema"] == SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2
        and attempt["placement_receipt"]["declared"]["devices"]
        and attempt["placement_receipt"]["verdict"]["reason_class"] == "not-sampled"
    }
    if schema in {RUN_MANIFEST_V1, RUN_MANIFEST_V2} and (actual_blocking or actual_not_sampled):
        raise ManifestRefusal("placement-verdict-missing", "v2 placement evidence must not hide its manifest verdict")
    if schema == RUN_MANIFEST_V3:
        has_blocking = bool(actual_blocking)
        expected_keys = {"status", "science_acceptance", "message", "unqualified_gpu_segments"}
        expected_status = "unqualified"
        expected_acceptance = "not-accepted"
        expected_message = "GPU PLACEMENT UNQUALIFIED: telemetry or physical placement evidence requires science review."
        if not has_blocking:
            expected_keys = {"status", "science_acceptance", "message", "not_sampled_gpu_segments"}
            expected_status = "not-sampled"
            expected_acceptance = "accepted"
            expected_message = "GPU PLACEMENT NOT-SAMPLED: L1a intentionally does not sample devices; H-v2 performs its own delivered-GPU check."
        elif actual_not_sampled:
            expected_keys.add("not_sampled_gpu_segments")
        if set(placement_verdict) != expected_keys:
            raise ManifestRefusal("malformed-record", "manifest placement verdict keys differ")
        if placement_verdict["status"] != expected_status or placement_verdict["science_acceptance"] != expected_acceptance:
            raise ManifestRefusal("malformed-record", "manifest placement verdict status disagrees with v2 receipts")
        _text(placement_verdict["message"], "manifest.placement_verdict.message")
        if placement_verdict["message"] != expected_message:
            raise ManifestRefusal("unqualified-placement-not-loud", "manifest placement verdict must be loud")

        def _entries(value: Any, name: str, allowed: set[str]) -> set[tuple[str, str, str, str]]:
            if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, list):
                raise ManifestRefusal("malformed-record", f"{name} must be a sequence")
            result: set[tuple[str, str, str, str]] = set()
            for item in value:
                entry = dict(_fields(item, {"task_id", "attempt_id", "reason_class", "reason"}, name))
                for key in entry:
                    _text(entry[key], f"{name}.{key}")
                if entry["reason_class"] not in allowed:
                    raise ManifestRefusal("malformed-record", f"{name} reason class is invalid")
                result.add((entry["task_id"], entry["attempt_id"], entry["reason_class"], entry["reason"]))
            if len(result) != len(value):
                raise ManifestRefusal("malformed-record", f"{name} contains duplicate entries")
            return result

        if has_blocking:
            advertised_blocking = _entries(placement_verdict["unqualified_gpu_segments"], "manifest unqualified GPU segment", {"mismatch", "refused", "telemetry-failed"})
        else:
            advertised_blocking = set()
        advertised_not_sampled = (
            _entries(placement_verdict["not_sampled_gpu_segments"], "manifest not-sampled GPU segment", {"not-sampled"})
            if actual_not_sampled
            else set()
        )
        if advertised_blocking != actual_blocking or advertised_not_sampled != actual_not_sampled:
            raise ManifestRefusal("placement-verdict-mismatch", "manifest placement verdict disagrees with v2 receipts")
    return dict(value), attempts, raw_segments, placement_verdict


def _check_segments(attempts: tuple[dict[str, Any], ...], segments: tuple[dict[str, Any], ...]) -> None:
    by_id = {attempt["attempt_id"]: attempt for attempt in attempts}
    if len({segment["attempt_id"] for segment in segments}) != len(segments):
        raise ManifestRefusal("duplicate-segment", "an attempt appears in more than one accepted segment")
    previous: dict[str, dict[str, Any]] = {}
    for segment in segments:
        attempt = by_id.get(segment["attempt_id"])
        if attempt is None:
            raise ManifestRefusal("uncommitted-segment", "accepted segment references an unknown attempt")
        if attempt["task_id"] != segment["task_id"]:
            raise ManifestRefusal("segment-task-mismatch", "accepted segment task differs from its attempt")
        outcome = attempt["outcome_receipt"]
        if outcome is None or outcome["status"] != "completed":
            raise ManifestRefusal("segment-on-failed-attempt", "accepted segment does not reference a completed attempt")
        prior = previous.get(segment["task_id"])
        if prior is None:
            if segment["predecessor_checkpoint"] is not None:
                raise ManifestRefusal("wrong-predecessor", "first accepted segment must be a fresh start")
        else:
            if segment["predecessor_checkpoint"] != prior["checkpoint_dir"]:
                raise ManifestRefusal("wrong-predecessor", "accepted segment predecessor does not match prior checkpoint")
            prior_coord = (prior["next_iteration"], prior["completed_updates"])
            coordinate = (segment["next_iteration"], segment["completed_updates"])
            if coordinate[0] < prior_coord[0] or coordinate[1] < prior_coord[1] or coordinate == prior_coord:
                raise ManifestRefusal("non-increasing-coordinate", "accepted segment progress coordinate is not strictly increasing")
        previous[segment["task_id"]] = segment


def _public_evidence(value: Any) -> tuple[str, Mapping[str, Any] | None, dict[str, str] | None]:
    if isinstance(value, Mapping):
        verdict = value.get("verdict")
        checkpoint = value.get("checkpoint")
    else:
        verdict = getattr(value, "verdict", None)
        checkpoint = getattr(value, "checkpoint", None)
    if type(verdict) is not str or verdict not in {"completed", "incomplete"}:
        raise ManifestRefusal("core-evidence-mismatch", "core inspection did not report completed or incomplete evidence")
    if not isinstance(checkpoint, Mapping):
        raise ManifestRefusal("core-evidence-mismatch", "core inspection did not report a checkpoint")
    expected = {"checkpoint_dir", "schema_version", "next_iteration", "completed_updates"}
    if set(checkpoint) != expected:
        raise ManifestRefusal("core-evidence-mismatch", "core checkpoint evidence has an unexpected shape")
    _text(checkpoint["checkpoint_dir"], "core.checkpoint_dir")
    if type(checkpoint["schema_version"]) is not int:
        raise ManifestRefusal("core-evidence-mismatch", "core.schema_version must be an integer")
    _nonnegative_int(checkpoint["next_iteration"], "core.next_iteration")
    _nonnegative_int(checkpoint["completed_updates"], "core.completed_updates")
    raw_identity = value.get("resolved_config_identity") if isinstance(value, Mapping) else getattr(value, "resolved_config_identity", None)
    return verdict, checkpoint, _config_identity(raw_identity)


def _default_inspect_run(path: str | Path) -> Any:
    """Load the public core inspector only when a caller did not inject one."""

    try:
        from tpen.inspection import inspect_run
    except ImportError as error:
        raise ManifestRefusal("core-inspector-unavailable", "public core inspection is unavailable") from error
    return inspect_run(path)


def read(
    manifest_path: str | Path,
    inspect_run: Callable[[str | Path], Any] | None = None,
) -> ResultSet:
    """Read one manifest and return its exactly-once committed result set.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Published v1 or v2 run manifest JSON.
    inspect_run : callable, optional
        Public core inspection callback.  It receives the fenced core run
        directory for each selected segment.  When omitted, the callback is
        loaded lazily from ``tpen.inspection``.

    Returns
    -------
    ResultSet
        Per-task accepted chains and non-selected original attempts.

    Raises
    ------
    ManifestRefusal
        If the manifest, chain, or public core evidence is inconsistent.
    """

    path = Path(manifest_path)
    _, attempts, segments, placement_verdict = _manifest(path)
    _check_segments(attempts, segments)
    inspector = inspect_run or _default_inspect_run
    selected_ids = {segment["attempt_id"] for segment in segments}
    chains: list[CommittedChain] = []
    by_task: dict[str, list[CommittedSegment]] = {}
    by_attempt = {attempt["attempt_id"]: attempt for attempt in attempts}
    for segment in segments:
        attempt = by_attempt[segment["attempt_id"]]
        run_dir = Path(attempt["output_dir"]) / "core"
        try:
            evidence = inspector(run_dir)
        except ManifestRefusal:
            raise
        except Exception as error:
            raise ManifestRefusal("core-evidence-unavailable", f"core inspection failed for {run_dir}: {error}") from error
        _, checkpoint, evidence_identity = _public_evidence(evidence)
        if dict(checkpoint) != {
            "checkpoint_dir": segment["checkpoint_dir"],
            "schema_version": checkpoint["schema_version"],
            "next_iteration": segment["next_iteration"],
            "completed_updates": segment["completed_updates"],
        }:
            raise ManifestRefusal("core-evidence-mismatch", f"core evidence disagrees with accepted segment {segment['attempt_id']!r}")
        if segment.get("resolved_config_identity") is not None and segment["resolved_config_identity"] != evidence_identity:
            raise ManifestRefusal("core-evidence-mismatch", f"core identity disagrees with accepted segment {segment['attempt_id']!r}")
        projected = CommittedSegment(
            segment["task_id"],
            segment["attempt_id"],
            segment["predecessor_checkpoint"],
            segment["checkpoint_dir"],
            segment["next_iteration"],
            segment["completed_updates"],
            _freeze_json(attempt["placement_receipt"]),
            segment.get("resolved_config_identity"),
        )
        by_task.setdefault(projected.task_id, []).append(projected)
    for task_id, task_segments in by_task.items():
        chains.append(CommittedChain(task_id, tuple(task_segments)))
    preserved = tuple(
        PreservedAttempt(
            attempt["task_id"],
            attempt["attempt_id"],
            attempt["output_dir"],
            None if attempt["outcome_receipt"] is None else attempt["outcome_receipt"]["status"],
            "uncommitted" if attempt["outcome_receipt"] is None else (
                "failed-or-incomplete" if attempt["outcome_receipt"]["status"] != "completed" else "not-selected"
            ),
            _freeze_json(attempt["placement_receipt"]),
        )
        for attempt in attempts
        if attempt["attempt_id"] not in selected_ids
    )
    frozen_placement_verdict = None if placement_verdict is None else _freeze_json(placement_verdict)
    return ResultSet(tuple(chains), preserved, frozen_placement_verdict)


__all__ = [
    "ACCEPTED_SEGMENT_SCHEMA",
    "CommittedChain",
    "CommittedSegment",
    "ManifestReadError",
    "ManifestRefusal",
    "PreservedAttempt",
    "RUN_MANIFEST_V1",
    "RUN_MANIFEST_V2",
    "RUN_MANIFEST_V3",
    "SCIENCE_PLACEMENT_RECEIPT_SCHEMA",
    "SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2",
    "ResultSet",
    "read",
]
