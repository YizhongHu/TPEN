"""Closed semantics of in-child GPU sampler evidence.

This module owns the ``gpu_sampler`` wire carried by child placement v2
sidecars and science placement v3 receipts.  :func:`validate_sampler_wire` is
the single structural validator and :func:`derive_device_decision` the single
device decision; the receipt builder and the receipt decoder both reach them
through :class:`tpen_executor.results.SciencePlacementReceiptV3`, and the P8
toolkit reader mirrors them independently.

Binding vocabulary (manager decision split-20261110): ``not-sampled`` means
sampling was intentionally never attempted and is not science-blocking;
``telemetry-failed`` means sampling was attempted and did not yield usable,
confirmed-quiescent evidence and is science-blocking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any

from ._codec import thaw

SAMPLER_METHOD = "nvidia-smi-pid-uuid-v1"
SAMPLER_KEYS = frozenset({
    "status", "reason", "method", "version", "attempted", "failure", "cadence_seconds", "count",
    "monotonic_start", "monotonic_end", "samples", "command_pids", "survivors", "elapsed_seconds",
    "join_timeout_seconds", "quiescent",
})
SAMPLE_KEYS = frozenset({"monotonic_start", "monotonic_end", "inventory", "pid_rows"})
MAX_SAMPLES = 16
MAX_JOIN_TIMEOUT = 4.0
#: Failure codes the in-child sampler may emit.
CHILD_FAILURE_CODES = frozenset({
    "launch-failed", "command-failed", "command-timeout", "read-failed", "malformed-output",
    "cleanup-unconfirmed", "no-sample", "sampler-cancelled", "sampler-join-expired",
    "sampler-unfinalized", "sampler-error",
})
#: Failure code the controller substitutes for structurally invalid child evidence.
EVIDENCE_INVALID = "sampler-evidence-invalid"
FAILURE_CODES = CHILD_FAILURE_CODES | {EVIDENCE_INVALID}
#: Device decision code -> (device comparison state, human-readable reason).
DEVICE_DECISIONS: Mapping[str, tuple[str, str]] = {
    "matched": ("matched", "live science PID used only GPU UUIDs inside its physical slot entitlement"),
    "out-of-entitlement": ("mismatch", "live science PID used a GPU UUID outside its physical slot entitlement"),
    "sampler-failed": ("unavailable", "attempted GPU sampler failed"),
    "entitlement-unresolved": ("unavailable", "declared GPU entitlement is absent from a sampled inventory"),
    "no-exact-pid-row": ("unavailable", "no sampled row names the live science PID"),
    "sampler-not-run": ("unavailable", "declared GPU slot was not sampled"),
    "cpu-only": ("unavailable", "CPU-only slot has no GPU entitlement"),
}
_GPU_UUID = re.compile(r"^GPU-[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$")
_STATUS_REASONS = {("observed", "observed"), ("unavailable", "telemetry-failed"), ("unavailable", "not-sampled")}


def not_sampled_sampler() -> dict[str, Any]:
    """Return the exact ``not-sampled`` sampler record.

    Returns
    -------
    dict
        Equal to the in-child ``gpu_not_sampled_record()``.
    """

    return {
        "status": "unavailable", "reason": "not-sampled", "method": SAMPLER_METHOD,
        "version": None, "attempted": False, "failure": None, "cadence_seconds": 0.0,
        "count": 0, "monotonic_start": None, "monotonic_end": None, "samples": [],
        "command_pids": [], "survivors": [], "elapsed_seconds": 0.0,
        "join_timeout_seconds": MAX_JOIN_TIMEOUT, "quiescent": True,
    }


def invalid_sampler(detail: str) -> dict[str, Any]:
    """Return a typed ``telemetry-failed`` record replacing invalid child evidence.

    Parameters
    ----------
    detail : str
        Why the child record failed :func:`validate_sampler_wire`.

    Returns
    -------
    dict
        Attempted, failed, non-quiescent evidence with no samples.  Invalid
        evidence cannot prove sampling was intentionally skipped, so it is
        conservatively science-blocking rather than ``not-sampled``.
    """

    return {
        "status": "unavailable", "reason": "telemetry-failed", "method": SAMPLER_METHOD,
        "version": None, "attempted": True,
        "failure": {"code": EVIDENCE_INVALID, "detail": (detail or "invalid sampler evidence")[:500]},
        "cadence_seconds": 0.0, "count": 0, "monotonic_start": None, "monotonic_end": None,
        "samples": [], "command_pids": [], "survivors": [], "elapsed_seconds": 0.0,
        "join_timeout_seconds": MAX_JOIN_TIMEOUT, "quiescent": False,
    }


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"gpu_sampler {name} must be a finite number")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"gpu_sampler {name} must be a positive integer")
    return value


def _uuid(value: Any, name: str) -> str:
    if type(value) is not str or _GPU_UUID.fullmatch(value) is None:
        raise ValueError(f"gpu_sampler {name} must be a canonical physical GPU UUID")
    return value


def _rows(value: Any, name: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"gpu_sampler {name} must be a sequence")
    return list(value)


def validate_sampler_wire(value: Any) -> dict[str, Any]:
    """Validate one closed ``gpu_sampler`` record and return a JSON copy.

    Parameters
    ----------
    value : mapping
        Child-produced or persisted sampler evidence.

    Returns
    -------
    dict
        Thawed JSON-shaped copy of the validated record.

    Raises
    ------
    ValueError
        If keys, labels, provenance, quiescence, bounds, ordering, or
        same-sample inventory membership are inconsistent.
    """

    if not isinstance(value, Mapping):
        raise ValueError("gpu_sampler must be a mapping")
    sampler = thaw(value)
    if set(sampler) != SAMPLER_KEYS:
        raise ValueError("gpu_sampler keys differ")
    if sampler["method"] != SAMPLER_METHOD:
        raise ValueError("gpu_sampler method is invalid")
    status, reason = sampler["status"], sampler["reason"]
    if (status, reason) not in _STATUS_REASONS:
        raise ValueError("gpu_sampler status/reason pair is invalid")
    if type(sampler["attempted"]) is not bool or type(sampler["quiescent"]) is not bool:
        raise ValueError("gpu_sampler attempted/quiescent must be booleans")
    # not-sampled means intentionally unattempted; it has one exact shape.
    if (reason == "not-sampled") != (not sampler["attempted"]):
        raise ValueError("gpu_sampler not-sampled must be exactly the unattempted state")
    if reason == "not-sampled":
        if sampler != not_sampled_sampler():
            raise ValueError("not-sampled gpu_sampler must carry no sampler evidence")
        return sampler
    failure = sampler["failure"]
    if reason == "observed":
        if failure is not None:
            raise ValueError("observed gpu_sampler cannot carry a failure")
    elif (
        not isinstance(failure, Mapping)
        or set(failure) != {"code", "detail"}
        or failure["code"] not in FAILURE_CODES
        or type(failure["detail"]) is not str
        or not failure["detail"]
    ):
        raise ValueError("telemetry-failed gpu_sampler requires a typed failure")
    version = sampler["version"]
    if version is not None and (type(version) is not str or not version.strip()):
        raise ValueError("gpu_sampler version must be non-empty text or null")
    cadence = _number(sampler["cadence_seconds"], "cadence_seconds")
    if cadence != 0.0 and not 0.01 <= cadence <= 1.0:
        raise ValueError("gpu_sampler cadence is outside its bound")
    join_timeout = _number(sampler["join_timeout_seconds"], "join_timeout_seconds")
    if not 0.0 < join_timeout <= MAX_JOIN_TIMEOUT:
        raise ValueError("gpu_sampler join timeout is outside its bound")
    elapsed = _number(sampler["elapsed_seconds"], "elapsed_seconds")
    count = sampler["count"]
    samples = _rows(sampler["samples"], "samples")
    if type(count) is not int or not 0 <= count <= MAX_SAMPLES or count != len(samples):
        raise ValueError("gpu_sampler count disagrees with its samples")
    command_pids = [_positive_int(pid, "command PID") for pid in _rows(sampler["command_pids"], "command_pids")]
    survivors = [_positive_int(pid, "survivor") for pid in _rows(sampler["survivors"], "survivors")]
    if len(set(command_pids)) != len(command_pids) or len(set(survivors)) != len(survivors):
        raise ValueError("gpu_sampler PIDs repeat")
    if not set(survivors) <= set(command_pids):
        raise ValueError("gpu_sampler survivor was never launched")
    # Quiescence is the confirmed absence of survivors, never a free label.
    if sampler["quiescent"] and survivors:
        raise ValueError("quiescent gpu_sampler cannot report survivors")
    start, end = sampler["monotonic_start"], sampler["monotonic_end"]
    if start is None:
        if end is not None or count or elapsed != 0.0:
            raise ValueError("gpu_sampler without an outer interval cannot carry samples")
    else:
        start, end = _number(start, "monotonic_start"), _number(end, "monotonic_end")
        if end < start or abs(elapsed - (end - start)) > 1e-6:
            raise ValueError("gpu_sampler outer interval disagrees with elapsed time")
    if status == "observed":
        if not sampler["quiescent"] or version is None or start is None or count < 1 or not command_pids:
            raise ValueError("observed gpu_sampler requires quiescent, complete, versioned samples")
    previous_end = start
    pci_by_uuid: dict[str, str] = {}
    for sample in samples:
        if not isinstance(sample, Mapping) or set(sample) != SAMPLE_KEYS:
            raise ValueError("gpu_sampler sample keys differ")
        sample_start = _number(sample["monotonic_start"], "sample start")
        sample_end = _number(sample["monotonic_end"], "sample end")
        # Samples are ordered, non-overlapping, and inside the outer interval.
        if sample_end < sample_start or sample_start < previous_end or sample_end > end:
            raise ValueError("gpu_sampler sample interval is not ordered inside the outer interval")
        previous_end = sample_end
        inventory: set[str] = set()
        for item in _rows(sample["inventory"], "inventory"):
            if not isinstance(item, Mapping) or set(item) != {"uuid", "pci.bus_id"}:
                raise ValueError("gpu_sampler inventory row is invalid")
            uuid = _uuid(item["uuid"], "inventory UUID")
            pci = item["pci.bus_id"]
            if type(pci) is not str or not pci or uuid in inventory or pci_by_uuid.setdefault(uuid, pci) != pci:
                raise ValueError("gpu_sampler inventory row is inconsistent")
            inventory.add(uuid)
        if not inventory:
            raise ValueError("gpu_sampler sample inventory is empty")
        for item in _rows(sample["pid_rows"], "pid_rows"):
            if not isinstance(item, Mapping) or set(item) != {"pid", "gpu_uuid"}:
                raise ValueError("gpu_sampler PID row is invalid")
            _positive_int(item["pid"], "row PID")
            # A PID row must name a GPU from its own sample's inventory.
            if _uuid(item["gpu_uuid"], "row UUID") not in inventory:
                raise ValueError("gpu_sampler PID row is absent from its sample inventory")
    return sampler


def normalize_child_sampler(value: Any) -> dict[str, Any]:
    """Validate child evidence, or replace it with typed failed evidence.

    Parameters
    ----------
    value : mapping
        Raw ``gpu_sampler`` from a child placement v2 sidecar.

    Returns
    -------
    dict
        The validated record, preserved verbatim including all samples, or
        :func:`invalid_sampler` when the child record is structurally invalid.
    """

    try:
        return validate_sampler_wire(value)
    except (TypeError, ValueError) as error:
        return invalid_sampler(str(error))


@dataclass(frozen=True)
class DeviceDecision:
    """Device verdict derived from validated sampler evidence.

    Parameters
    ----------
    code : str
        Key of :data:`DEVICE_DECISIONS`.
    active_uuids, active_pci : tuple of str
        Ordered union of GPUs used by the exact science PID across all
        samples; populated only for ``matched`` and ``out-of-entitlement``.
    """

    code: str
    active_uuids: tuple[str, ...] = ()
    active_pci: tuple[str, ...] = ()

    @property
    def device(self) -> str:
        """Placement comparison state for the device."""

        return DEVICE_DECISIONS[self.code][0]

    @property
    def text(self) -> str:
        """Human-readable device reason."""

        return DEVICE_DECISIONS[self.code][1]


def derive_device_decision(sampler: Mapping[str, Any], child_pid: int, declared_devices: Sequence[str]) -> DeviceDecision:
    """Derive the device verdict from the union of exact-child-PID rows.

    Parameters
    ----------
    sampler : mapping
        Record already accepted by :func:`validate_sampler_wire`.
    child_pid : int
        PID of the live science child named by the placement receipt.
    declared_devices : sequence of str
        Canonical physical UUID entitlement of the slot.

    Returns
    -------
    DeviceDecision
        Out-of-entitlement use in any sample dominates every other outcome;
        then attempted failure, unresolved entitlement, and a missing
        exact-PID row are all telemetry failures; only then ``matched``.

    Raises
    ------
    ValueError
        If the entitlement is not canonical or a CPU-only slot carries
        attempted sampler evidence.
    """

    declared = [_uuid(device, "declared device") for device in declared_devices]
    if len(set(declared)) != len(declared):
        raise ValueError("declared GPU entitlement repeats a UUID")
    if not declared:
        if sampler["reason"] != "not-sampled":
            raise ValueError("CPU-only slot cannot carry attempted GPU sampler evidence")
        return DeviceDecision("cpu-only")
    if sampler["reason"] == "not-sampled":
        return DeviceDecision("sampler-not-run")
    active: list[str] = []
    active_pci: list[str] = []
    for sample in sampler["samples"]:
        pci_by_uuid = {item["uuid"]: item["pci.bus_id"] for item in sample["inventory"]}
        for row in sample["pid_rows"]:
            if row["pid"] == child_pid and row["gpu_uuid"] not in active:
                active.append(row["gpu_uuid"])
                active_pci.append(pci_by_uuid[row["gpu_uuid"]])
    if any(uuid not in declared for uuid in active):
        return DeviceDecision("out-of-entitlement", tuple(active), tuple(active_pci))
    if sampler["reason"] == "telemetry-failed":
        return DeviceDecision("sampler-failed")
    if any(not set(declared) <= {item["uuid"] for item in sample["inventory"]} for sample in sampler["samples"]):
        return DeviceDecision("entitlement-unresolved")
    if not active:
        return DeviceDecision("no-exact-pid-row")
    return DeviceDecision("matched", tuple(active), tuple(active_pci))


__all__ = [
    "CHILD_FAILURE_CODES", "DEVICE_DECISIONS", "DeviceDecision", "EVIDENCE_INVALID", "FAILURE_CODES",
    "SAMPLER_KEYS", "SAMPLER_METHOD", "derive_device_decision", "invalid_sampler", "normalize_child_sampler",
    "not_sampled_sampler", "validate_sampler_wire",
]
