"""Closed semantics of worker-owned GPU sampler evidence.

This module owns the ``gpu_sampler`` wire written by the worker-side sampler
(:mod:`tpen_executor.gpu_sampler`) and carried by science placement v3
receipts.  :func:`validate_sampler_wire` is the single structural validator
and :func:`derive_device_decision` the single device decision; the receipt
builder and the receipt decoder both reach them through
:class:`tpen_executor.results.SciencePlacementReceiptV3`, and the P8 toolkit
reader mirrors them independently.

The sampler watches the science child for its whole lifetime but retains at
most :data:`MAX_SAMPLES` samples.  The closed ``aggregate`` record summarises
every complete sample, retained or not: the ordered union of GPUs used by the
exact target PID and whether out-of-entitlement use or an unresolved
entitlement was ever seen.  The device decision is taken from the aggregate,
which must agree with the retained samples.

Transient command failures (fix round r4, manager ruling Q1) are bounded by a
gap budget that the record carries explicitly in its closed ``gaps`` block:
every failed sample attempt is recorded, and exceeding either the
consecutive or the total budget is the terminal ``gap-budget-exceeded``
failure.  Only :data:`GAP_CODES` are ever forgiven; outside use,
contradictory or malformed output, MIG identifiers, and unconfirmed cleanup
never are.  MIG device identifiers are unsupported and fail with the distinct
``mig-unsupported`` code (manager ruling Q5).

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
    "join_timeout_seconds", "quiescent", "target_pid", "aggregate", "executable", "gaps",
})
SAMPLE_KEYS = frozenset({"monotonic_start", "monotonic_end", "inventory", "pid_rows"})
AGGREGATE_KEYS = frozenset({"samples_taken", "exact_pid_gpus", "outside_use_seen", "entitlement_unresolved_seen"})
GAP_KEYS = frozenset({"max_consecutive_allowed", "max_total_allowed", "count", "max_consecutive", "total_seconds", "events"})
GAP_EVENT_KEYS = frozenset({"code", "monotonic_start", "monotonic_end", "consecutive"})
#: The only failure codes a gap budget may forgive: transient command/read
#: failures.  Every other failure code is terminal.
GAP_CODES = frozenset({"command-failed", "command-timeout", "read-failed"})
#: Hard wire cap on either recorded gap budget; it bounds the event list.
GAP_BUDGET_CAP = 16
#: Maximum number of retained samples; the aggregate covers every sample.
MAX_SAMPLES = 16
#: Bounds of the lifetime sampling cadence, in seconds between samples.
MIN_CADENCE = 0.01
MAX_CADENCE = 60.0
#: Join bound carried by records of a sampler that never ran (``not-sampled``
#: and controller-substituted invalid evidence).  The worker's own derived
#: finish bound is :data:`tpen_executor.gpu_sampler.JOIN_TIMEOUT`.
UNRUN_JOIN_TIMEOUT = 30.0
#: Wire bound of any recorded finish deadline.
MAX_JOIN_TIMEOUT = 60.0
#: Failure codes the worker-side sampler may emit.
SAMPLER_FAILURE_CODES = frozenset({
    "launch-failed", "command-failed", "command-timeout", "read-failed", "malformed-output",
    "cleanup-unconfirmed", "no-sample", "sampler-cancelled", "sampler-join-expired",
    "sampler-unfinalized", "sampler-error", "mig-unsupported", "gap-budget-exceeded",
})
#: Failure code the controller substitutes for missing or invalid sampler evidence.
EVIDENCE_INVALID = "sampler-evidence-invalid"
FAILURE_CODES = SAMPLER_FAILURE_CODES | {EVIDENCE_INVALID}
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


def empty_aggregate() -> dict[str, Any]:
    """Return the aggregate of a sampler that has taken no sample.

    Returns
    -------
    dict
        Zero samples, no exact-PID GPUs, and no outside or unresolved use.
    """

    return {"samples_taken": 0, "exact_pid_gpus": [], "outside_use_seen": False, "entitlement_unresolved_seen": False}


def empty_gaps(max_consecutive_allowed: int = 0, max_total_allowed: int = 0) -> dict[str, Any]:
    """Return the gap block of a sampler that has recorded no failed attempt.

    Parameters
    ----------
    max_consecutive_allowed, max_total_allowed : int, default=0
        Gap budget in force.  Records of a sampler that never ran carry the
        zero budget, so their exact shape does not depend on calibration.

    Returns
    -------
    dict
        The budget, zero counters, and an empty event list.
    """

    return {
        "max_consecutive_allowed": max_consecutive_allowed, "max_total_allowed": max_total_allowed,
        "count": 0, "max_consecutive": 0, "total_seconds": 0.0, "events": [],
    }


def not_sampled_sampler() -> dict[str, Any]:
    """Return the exact ``not-sampled`` sampler record.

    Returns
    -------
    dict
        The single legal record for intentionally unattempted sampling.
    """

    return {
        "status": "unavailable", "reason": "not-sampled", "method": SAMPLER_METHOD,
        "version": None, "attempted": False, "failure": None, "cadence_seconds": 0.0,
        "count": 0, "monotonic_start": None, "monotonic_end": None, "samples": [],
        "command_pids": [], "survivors": [], "elapsed_seconds": 0.0,
        "join_timeout_seconds": UNRUN_JOIN_TIMEOUT, "quiescent": True,
        "target_pid": None, "aggregate": empty_aggregate(), "executable": None, "gaps": empty_gaps(),
    }


def invalid_sampler(detail: str) -> dict[str, Any]:
    """Return a typed ``telemetry-failed`` record replacing invalid child evidence.

    Parameters
    ----------
    detail : str
        Why the worker evidence was missing or failed :func:`validate_sampler_wire`.

    Returns
    -------
    dict
        Attempted, failed, non-quiescent evidence with no samples.  Missing or
        invalid evidence for a sampled slot cannot prove sampling was
        intentionally skipped, so it is conservatively science-blocking
        rather than ``not-sampled``.
    """

    return {
        "status": "unavailable", "reason": "telemetry-failed", "method": SAMPLER_METHOD,
        "version": None, "attempted": True,
        "failure": {"code": EVIDENCE_INVALID, "detail": (detail.strip() or "invalid sampler evidence")[:500]},
        "cadence_seconds": 0.0, "count": 0, "monotonic_start": None, "monotonic_end": None,
        "samples": [], "command_pids": [], "survivors": [], "elapsed_seconds": 0.0,
        "join_timeout_seconds": UNRUN_JOIN_TIMEOUT, "quiescent": False,
        "target_pid": None, "aggregate": empty_aggregate(), "executable": None, "gaps": empty_gaps(),
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


def _text(value: Any) -> bool:
    """Return whether ``value`` is text with at least one non-whitespace character.

    This matches the P8 reader's non-empty text rule exactly.
    """

    return type(value) is str and bool(value.strip())


def _validate_aggregate(sampler: Mapping[str, Any], retained: list[tuple[str, str]], pci_by_uuid: Mapping[str, str]) -> None:
    """Check the lifetime aggregate against the retained samples.

    Parameters
    ----------
    sampler : mapping
        Sampler record whose top-level fields were already validated.
    retained : list of tuple of str
        Ordered ``(uuid, pci)`` union of exact-target rows in retained samples.
    pci_by_uuid : mapping
        PCI bus id of every UUID seen in a retained inventory.

    Raises
    ------
    ValueError
        If the aggregate keys or types are invalid, it covers fewer samples
        than are retained, it omits a retained exact-target GPU, or, when no
        sample was dropped, it differs from the retained union.
    """

    aggregate = sampler["aggregate"]
    if not isinstance(aggregate, Mapping) or set(aggregate) != AGGREGATE_KEYS:
        raise ValueError("gpu_sampler aggregate keys differ")
    taken = aggregate["samples_taken"]
    if type(taken) is not int or taken < sampler["count"] or (taken == 0) != (sampler["count"] == 0):
        raise ValueError("gpu_sampler aggregate sample total disagrees with its retained samples")
    if type(aggregate["outside_use_seen"]) is not bool or type(aggregate["entitlement_unresolved_seen"]) is not bool:
        raise ValueError("gpu_sampler aggregate flags must be booleans")
    if taken and sampler["target_pid"] is None:
        raise ValueError("gpu_sampler samples require a target PID")
    union: list[tuple[str, str]] = []
    for item in _rows(aggregate["exact_pid_gpus"], "aggregate exact_pid_gpus"):
        if not isinstance(item, Mapping) or set(item) != {"uuid", "pci.bus_id"}:
            raise ValueError("gpu_sampler aggregate GPU row is invalid")
        uuid, pci = _uuid(item["uuid"], "aggregate UUID"), item["pci.bus_id"]
        if not _text(pci) or uuid in {seen for seen, _ in union} or pci_by_uuid.get(uuid, pci) != pci:
            raise ValueError("gpu_sampler aggregate GPU row is inconsistent")
        union.append((uuid, pci))
    if not set(retained) <= set(union):
        raise ValueError("gpu_sampler aggregate omits a retained exact-target GPU")
    # With nothing dropped the aggregate is fully determined by the samples.
    if taken == sampler["count"] and union != retained:
        raise ValueError("gpu_sampler aggregate differs from its complete retained samples")


def _validate_gaps(sampler: Mapping[str, Any], start: float | None, end: float | None) -> None:
    """Check the closed gap block against its budget and the failure code.

    Parameters
    ----------
    sampler : mapping
        Sampler record whose top-level fields were already validated.
    start, end : float or None
        Validated outer interval of the record.

    Raises
    ------
    ValueError
        If the keys or types are invalid, a budget exceeds
        :data:`GAP_BUDGET_CAP`, an event is not a forgivable transient code,
        events are unordered or outside the outer interval, the consecutive
        run numbering or the counters disagree with the events, the attempt
        timeline is incoherent, the record continues past the first budget
        excess, or ``gap-budget-exceeded`` is claimed without an excess (or
        an excess is labelled otherwise).

    Notes
    -----
    Attempts are sequential and a successful attempt is exactly one sample,
    which resets the consecutive run (review delta-3 F3).  So no gap may
    overlap a retained sample, a run cannot continue across a retained
    sample, and every reset after the first gap needs a successful sample
    between it and the previous gap.  A reset with no retained sample in
    between is provable only when no sample was dropped; with dropped
    history such resets are accepted only while there are enough dropped
    samples to justify them, one distinct sample per reset.  The first
    budget excess ends sampling, so no event before the last may exceed
    either budget.
    """

    gaps = sampler["gaps"]
    if not isinstance(gaps, Mapping) or set(gaps) != GAP_KEYS:
        raise ValueError("gpu_sampler gaps keys differ")
    counters = ("max_consecutive_allowed", "max_total_allowed", "count", "max_consecutive")
    if any(type(gaps[key]) is not int or gaps[key] < 0 for key in counters):
        raise ValueError("gpu_sampler gap counters must be non-negative integers")
    allowed_consecutive, allowed_total = gaps["max_consecutive_allowed"], gaps["max_total_allowed"]
    if allowed_consecutive > GAP_BUDGET_CAP or allowed_total > GAP_BUDGET_CAP:
        raise ValueError("gpu_sampler gap budget exceeds its wire cap")
    events = _rows(gaps["events"], "gap events")
    if gaps["count"] != len(events):
        raise ValueError("gpu_sampler gap count disagrees with its events")
    # Retained successful attempts, already validated as ordered intervals.
    successes = [(float(sample["monotonic_start"]), float(sample["monotonic_end"])) for sample in sampler["samples"]]
    # Successful samples taken but not retained; zero means complete history.
    dropped = sampler["aggregate"]["samples_taken"] - sampler["count"]
    previous_end, run, longest, summed, unproven_resets = start, 0, 0, 0.0, 0
    for index, event in enumerate(events):
        if not isinstance(event, Mapping) or set(event) != GAP_EVENT_KEYS:
            raise ValueError("gpu_sampler gap event keys differ")
        if event["code"] not in GAP_CODES:
            raise ValueError("gpu_sampler gap event is not a forgivable transient failure")
        event_start = _number(event["monotonic_start"], "gap start")
        event_end = _number(event["monotonic_end"], "gap end")
        consecutive = _positive_int(event["consecutive"], "gap consecutive run")
        # Gaps are ordered, non-overlapping, and inside the outer interval.
        if start is None or event_end < event_start or event_start < previous_end or event_end > end:
            raise ValueError("gpu_sampler gap interval is not ordered inside the outer interval")
        # A run either starts afresh or extends the previous gap's run.
        if consecutive not in (1, run + 1):
            raise ValueError("gpu_sampler gap consecutive run numbering is inconsistent")
        # A failed attempt never overlaps a successful one.
        if any(event_start < success_end and success_start < event_end for success_start, success_end in successes):
            raise ValueError("gpu_sampler gap overlaps a retained successful sample")
        if index:
            # Whether a retained success lies between the previous gap and this one.
            succeeded_between = any(
                previous_end <= success_start and success_end <= event_start for success_start, success_end in successes
            )
            if succeeded_between and consecutive != 1:
                raise ValueError("gpu_sampler gap run continues across a retained successful sample")
            if consecutive == 1 and not succeeded_between:
                # Only a dropped sample can have reset this run.
                unproven_resets += 1
        # The first excess ends sampling: no earlier event may exceed a budget.
        if index < len(events) - 1 and (index + 1 > allowed_total or consecutive > allowed_consecutive):
            raise ValueError("gpu_sampler gaps continue past the first budget excess")
        previous_end, run = event_end, consecutive
        longest = max(longest, consecutive)
        summed += event_end - event_start
    # Each unproven reset needs its own dropped sample (none with complete history).
    if unproven_resets > dropped:
        raise ValueError("gpu_sampler gap run resets without an intervening successful sample")
    if gaps["max_consecutive"] != longest:
        raise ValueError("gpu_sampler gap max_consecutive disagrees with its events")
    if abs(_number(gaps["total_seconds"], "gap total_seconds") - summed) > 1e-6:
        raise ValueError("gpu_sampler gap total_seconds disagrees with its events")
    # Sampling stops at the first excess, so no counter passes its budget by more than one.
    if len(events) > allowed_total + 1 or longest > allowed_consecutive + 1:
        raise ValueError("gpu_sampler gaps continue past the first budget excess")
    exceeded = len(events) > allowed_total or longest > allowed_consecutive
    failure = sampler["failure"]
    if exceeded != (failure is not None and failure["code"] == "gap-budget-exceeded"):
        raise ValueError("gpu_sampler gap budget excess and gap-budget-exceeded failure disagree")


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
        or not _text(failure["detail"])
    ):
        raise ValueError("telemetry-failed gpu_sampler requires a typed failure")
    version = sampler["version"]
    if version is not None and (type(version) is not str or not version.strip()):
        raise ValueError("gpu_sampler version must be non-empty text or null")
    executable = sampler["executable"]
    if executable is not None and not _text(executable):
        raise ValueError("gpu_sampler executable must be non-empty text or null")
    cadence = _number(sampler["cadence_seconds"], "cadence_seconds")
    if cadence != 0.0 and not MIN_CADENCE <= cadence <= MAX_CADENCE:
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
    target = sampler["target_pid"]
    if target is not None:
        _positive_int(target, "target PID")
    start, end = sampler["monotonic_start"], sampler["monotonic_end"]
    if start is None:
        if end is not None or count or elapsed != 0.0:
            raise ValueError("gpu_sampler without an outer interval cannot carry samples")
    else:
        start, end = _number(start, "monotonic_start"), _number(end, "monotonic_end")
        if end < start or abs(elapsed - (end - start)) > 1e-6:
            raise ValueError("gpu_sampler outer interval disagrees with elapsed time")
    if status == "observed":
        if (
            not sampler["quiescent"] or version is None or executable is None or start is None
            or count < 1 or not command_pids
        ):
            raise ValueError("observed gpu_sampler requires quiescent, complete, versioned samples from a resolved executable")
    previous_end = start
    pci_by_uuid: dict[str, str] = {}
    retained: list[tuple[str, str]] = []
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
            if not _text(pci) or uuid in inventory or pci_by_uuid.setdefault(uuid, pci) != pci:
                raise ValueError("gpu_sampler inventory row is inconsistent")
            inventory.add(uuid)
        if not inventory:
            raise ValueError("gpu_sampler sample inventory is empty")
        for item in _rows(sample["pid_rows"], "pid_rows"):
            if not isinstance(item, Mapping) or set(item) != {"pid", "gpu_uuid"}:
                raise ValueError("gpu_sampler PID row is invalid")
            pid = _positive_int(item["pid"], "row PID")
            # A PID row must name a GPU from its own sample's inventory.
            uuid = _uuid(item["gpu_uuid"], "row UUID")
            if uuid not in inventory:
                raise ValueError("gpu_sampler PID row is absent from its sample inventory")
            if pid == target and uuid not in {seen for seen, _ in retained}:
                retained.append((uuid, pci_by_uuid[uuid]))
    _validate_aggregate(sampler, retained, pci_by_uuid)
    _validate_gaps(sampler, start, end)
    return sampler


def normalize_sampler(value: Any) -> dict[str, Any]:
    """Validate worker evidence, or replace it with typed failed evidence.

    Parameters
    ----------
    value : mapping
        Raw ``gpu_sampler`` record read from the worker evidence file.

    Returns
    -------
    dict
        The validated record, preserved verbatim including all retained
        samples, or :func:`invalid_sampler` when it is structurally invalid.
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
        Ordered union of GPUs used by the exact science PID across every
        sample, retained or not (the aggregate); populated only for
        ``matched`` and ``out-of-entitlement``.
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
    """Derive the device verdict from the lifetime exact-child-PID aggregate.

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
        Out-of-entitlement use in any sample, retained or not, dominates
        every other outcome; then attempted failure, unresolved entitlement,
        and a missing exact-PID row are all telemetry failures; only then
        ``matched``.

    Raises
    ------
    ValueError
        If the entitlement is not canonical, a CPU-only slot carries
        attempted sampler evidence, the sampler watched a different PID, or
        the aggregate flags contradict the aggregate GPUs or retained samples.
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
    if sampler["target_pid"] is not None and sampler["target_pid"] != child_pid:
        raise ValueError("gpu_sampler watched a PID other than the science child")
    aggregate = sampler["aggregate"]
    active = tuple(item["uuid"] for item in aggregate["exact_pid_gpus"])
    active_pci = tuple(item["pci.bus_id"] for item in aggregate["exact_pid_gpus"])
    # The flags are durable lifetime facts; they must agree with the GPUs
    # and, where retained samples show the condition, with those samples.
    if aggregate["outside_use_seen"] != any(uuid not in declared for uuid in active):
        raise ValueError("gpu_sampler aggregate outside-use flag contradicts its exact-PID GPUs")
    unresolved = any(not set(declared) <= {item["uuid"] for item in sample["inventory"]} for sample in sampler["samples"])
    if unresolved and not aggregate["entitlement_unresolved_seen"]:
        raise ValueError("gpu_sampler aggregate hides an unresolved entitlement in a retained sample")
    if aggregate["samples_taken"] == sampler["count"] and aggregate["entitlement_unresolved_seen"] != unresolved:
        raise ValueError("gpu_sampler aggregate unresolved flag differs from its complete retained samples")
    if aggregate["outside_use_seen"]:
        return DeviceDecision("out-of-entitlement", active, active_pci)
    if sampler["reason"] == "telemetry-failed":
        return DeviceDecision("sampler-failed")
    if aggregate["entitlement_unresolved_seen"]:
        return DeviceDecision("entitlement-unresolved")
    if not active:
        return DeviceDecision("no-exact-pid-row")
    return DeviceDecision("matched", active, active_pci)


__all__ = [
    "AGGREGATE_KEYS", "DEVICE_DECISIONS", "DeviceDecision", "EVIDENCE_INVALID", "FAILURE_CODES",
    "GAP_BUDGET_CAP", "GAP_CODES", "GAP_EVENT_KEYS", "GAP_KEYS", "MAX_CADENCE", "MAX_JOIN_TIMEOUT", "MAX_SAMPLES",
    "MIN_CADENCE", "SAMPLER_FAILURE_CODES", "SAMPLER_KEYS", "SAMPLER_METHOD", "UNRUN_JOIN_TIMEOUT",
    "derive_device_decision", "empty_aggregate", "empty_gaps", "invalid_sampler", "normalize_sampler",
    "not_sampled_sampler", "validate_sampler_wire",
]
