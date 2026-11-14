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
MANIFEST_ATTEMPT_V1 = "tpen-executor/manifest-attempt-v1"
MANIFEST_ATTEMPT_V2 = "tpen-executor/manifest-attempt-v2"
ACCEPTED_SEGMENT_SCHEMA = "tpen-executor/accepted-segment-v1"
SCIENCE_PLACEMENT_RECEIPT_SCHEMA = "tpen-executor/science-placement-receipt-v1"
SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2 = "tpen-executor/science-placement-receipt-v2"
SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V3 = "tpen-executor/science-placement-receipt-v3"
OUTCOME_STATUSES = frozenset({"completed", "failed", "timed_out", "launch_failed", "remote-quiescence-unconfirmed"})
PLACEMENT_COMPARISON_STATES = frozenset({"matched", "mismatch", "unavailable", "refused"})
_HEX_DIGITS = frozenset(string.hexdigits.lower())
_GPU_UUID = re.compile(r"^GPU-[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$")
# Independent mirror of the executor placement vocabulary.  These strings are
# deliberately duplicated rather than imported: the toolkit never imports the
# executor, and the real producer-to-reader tests prove the two sides agree.
PLACEMENT_UNQUALIFIED_MESSAGE = "GPU PLACEMENT UNQUALIFIED: placement mismatch/refusal requires science review."
PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE = (
    "GPU PLACEMENT UNQUALIFIED: attempted GPU telemetry failed or placement mismatch/refusal requires science review."
)
PLACEMENT_NOT_SAMPLED_MESSAGE = (
    "GPU PLACEMENT NOT-SAMPLED: L1a intentionally does not sample devices; H-v2 performs its own delivered-GPU check."
)
PLACEMENT_V3_WARNINGS = {
    "qualified": "GPU PLACEMENT QUALIFIED: live PID UUID telemetry matched the declared physical slot.",
    "mismatch": "GPU PLACEMENT UNQUALIFIED: physical placement evidence requires review.",
    "refused": "GPU PLACEMENT UNQUALIFIED: physical placement evidence requires review.",
    "telemetry-failed": "GPU PLACEMENT UNQUALIFIED: attempted GPU telemetry failed; science acceptance is blocked.",
}
_L1A_UNQUALIFIED_WARNING = "GPU PLACEMENT UNQUALIFIED: physical placement evidence requires review."
GPU_SAMPLER_METHOD = "nvidia-smi-pid-uuid-v1"
GPU_SAMPLER_KEYS = frozenset({
    "status", "reason", "method", "version", "attempted", "failure", "cadence_seconds", "count",
    "monotonic_start", "monotonic_end", "samples", "command_pids", "survivors", "elapsed_seconds",
    "join_timeout_seconds", "quiescent", "target_pid", "aggregate", "executable", "gaps",
})
GPU_SAMPLER_AGGREGATE_KEYS = frozenset({"samples_taken", "exact_pid_gpus", "outside_use_seen", "entitlement_unresolved_seen"})
GPU_SAMPLER_MAX_SAMPLES = 16
GPU_SAMPLER_CADENCE_BOUNDS = (0.01, 60.0)
GPU_SAMPLER_MAX_JOIN_TIMEOUT = 60.0
# Gap budget block (mirrors the executor's gpu_telemetry): only transient
# command/read failures may be forgiven, and either budget is capped on the wire.
GPU_SAMPLER_GAP_KEYS = frozenset({"max_consecutive_allowed", "max_total_allowed", "count", "max_consecutive", "total_seconds", "events"})
GPU_SAMPLER_GAP_EVENT_KEYS = frozenset({"code", "monotonic_start", "monotonic_end", "consecutive"})
GPU_SAMPLER_GAP_CODES = frozenset({"command-failed", "command-timeout", "read-failed"})
GPU_SAMPLER_GAP_BUDGET_CAP = 16
GPU_SAMPLER_FAILURE_CODES = frozenset({
    "launch-failed", "command-failed", "command-timeout", "read-failed", "malformed-output",
    "cleanup-unconfirmed", "no-sample", "sampler-cancelled", "sampler-join-expired",
    "sampler-unfinalized", "sampler-error", "sampler-evidence-invalid", "mig-unsupported", "gap-budget-exceeded",
})
GPU_DEVICE_DECISIONS = {
    "matched": ("matched", "live science PID used only GPU UUIDs inside its physical slot entitlement"),
    "out-of-entitlement": ("mismatch", "live science PID used a GPU UUID outside its physical slot entitlement"),
    "sampler-failed": ("unavailable", "attempted GPU sampler failed"),
    "entitlement-unresolved": ("unavailable", "declared GPU entitlement is absent from a sampled inventory"),
    "no-exact-pid-row": ("unavailable", "no sampled row names the live science PID"),
    "sampler-not-run": ("unavailable", "declared GPU slot was not sampled"),
    "cpu-only": ("unavailable", "CPU-only slot has no GPU entitlement"),
}
_GPU_NOT_SAMPLED = {
    "status": "unavailable", "reason": "not-sampled", "method": GPU_SAMPLER_METHOD,
    "version": None, "attempted": False, "failure": None, "cadence_seconds": 0.0,
    "count": 0, "monotonic_start": None, "monotonic_end": None, "samples": [],
    "command_pids": [], "survivors": [], "elapsed_seconds": 0.0,
    "join_timeout_seconds": 30.0, "quiescent": True, "target_pid": None,
    "aggregate": {"samples_taken": 0, "exact_pid_gpus": [], "outside_use_seen": False, "entitlement_unresolved_seen": False},
    "executable": None,
    "gaps": {"max_consecutive_allowed": 0, "max_total_allowed": 0, "count": 0, "max_consecutive": 0, "total_seconds": 0.0, "events": []},
}


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
    core_outcome: Mapping[str, Any] | None = None


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
    if record["qualification"] != "unqualified":
        raise ManifestRefusal("malformed-record", "L1a v2 placement cannot claim qualification")
    _text(record["reason_class"], "placement.verdict.reason_class")
    if record["reason_class"] not in {"mismatch", "not-sampled", "refused"}:
        raise ManifestRefusal("malformed-record", "L1a placement verdict reason class is invalid")
    _text(record["reason"], "placement.verdict.reason")
    _text(record["science_acceptance"], "placement.verdict.science_acceptance")
    if record["science_acceptance"] not in {"accepted", "not-accepted"}:
        raise ManifestRefusal("malformed-record", "placement.verdict.science_acceptance is invalid")
    if type(record["warning"]) is not str:
        raise ManifestRefusal("malformed-record", "placement.verdict.warning must be text")
    if record["device"] != "unavailable":
        raise ManifestRefusal("malformed-record", "L1a device verdict must be unavailable")
    expected_acceptance = "accepted" if record["reason_class"] == "not-sampled" else "not-accepted"
    if record["science_acceptance"] != expected_acceptance:
        raise ManifestRefusal("malformed-record", "placement verdict science acceptance contradicts reason class")
    if declared["devices"]:
        if not record["warning"]:
            raise ManifestRefusal("unqualified-placement-not-loud", "declared GPU placement requires a loud warning")
        if record["reason_class"] == "not-sampled" and "GPU PLACEMENT NOT-SAMPLED" not in record["warning"]:
            raise ManifestRefusal("unqualified-placement-not-loud", "not-sampled placement warning is not loud")
        if record["reason_class"] == "mismatch" and "GPU PLACEMENT UNQUALIFIED" not in record["warning"]:
            raise ManifestRefusal("unqualified-placement-not-loud", "mismatched placement warning is not loud")
        if record["reason_class"] == "refused" and not any(
            phrase in record["warning"]
            for phrase in ("GPU PLACEMENT REFUSED", "GPU PLACEMENT UNQUALIFIED")
        ):
            raise ManifestRefusal("unqualified-placement-not-loud", "refused placement warning is not loud")
    return record


def _sampler_refusal(reason: str) -> ManifestRefusal:
    return ManifestRefusal("malformed-record", f"placement gpu sampler {reason}")


def _sampler_number(value: Any, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise _sampler_refusal(f"{name} must be a finite number")
    return float(value)


def _sampler_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise _sampler_refusal(f"{name} must be a positive integer")
    return value


def _sampler_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise _sampler_refusal(f"{name} must be a sequence")
    return value


def _placement_v3_aggregate(record: Mapping[str, Any], retained: list[tuple[str, str]], pci_by_uuid: Mapping[str, str]) -> None:
    """Mirror the executor's lifetime-aggregate check against retained samples."""

    aggregate = _fields(record["aggregate"], set(GPU_SAMPLER_AGGREGATE_KEYS), "placement gpu sampler aggregate")
    taken = aggregate["samples_taken"]
    if type(taken) is not int or taken < record["count"] or (taken == 0) != (record["count"] == 0):
        raise _sampler_refusal("aggregate sample total disagrees with its retained samples")
    if type(aggregate["outside_use_seen"]) is not bool or type(aggregate["entitlement_unresolved_seen"]) is not bool:
        raise _sampler_refusal("aggregate flags must be booleans")
    if taken and record["target_pid"] is None:
        raise _sampler_refusal("samples require a target PID")
    union: list[tuple[str, str]] = []
    for item in _sampler_list(aggregate["exact_pid_gpus"], "aggregate exact_pid_gpus"):
        item = _fields(item, {"uuid", "pci.bus_id"}, "placement gpu sampler aggregate GPU")
        uuid = _gpu_uuid(item["uuid"], "placement gpu sampler aggregate uuid")
        pci = _text(item["pci.bus_id"], "placement gpu sampler aggregate pci.bus_id")
        if uuid in {seen for seen, _ in union} or pci_by_uuid.get(uuid, pci) != pci:
            raise _sampler_refusal("aggregate GPU row is inconsistent")
        union.append((uuid, pci))
    if not set(retained) <= set(union):
        raise _sampler_refusal("aggregate omits a retained exact-target GPU")
    if taken == record["count"] and union != retained:
        raise _sampler_refusal("aggregate differs from its complete retained samples")


def _placement_v3_gaps(record: Mapping[str, Any], start: float | None, end: float | None) -> None:
    """Mirror the executor's gap-budget check (manager ruling Q1).

    Every recorded gap must be a forgivable transient failure, ordered inside
    the outer interval, with consistent run numbering and counters; sampling
    must stop at the first budget excess, and ``gap-budget-exceeded`` must be
    the failure exactly when a budget was exceeded.

    The attempt timeline must be coherent (review delta-3 F3): no gap
    overlaps a retained successful sample, a run never continues across a
    retained sample, every reset after the first gap needs a successful
    sample in between, and resets with no retained sample in between are
    accepted only while enough samples were dropped to justify them (none
    with complete history).  No event before the last may exceed a budget.

    A dropped success witnesses a reset only inside its possible eviction
    window (review delta-4 D4-2): before the always-retained newest sample
    and, when two or more samples are retained (so the first is pinned),
    after the first retained sample; each reset needs its own dropped
    success.  No successful sample follows the attempt that exceeded a
    budget.
    """

    gaps = _fields(record["gaps"], set(GPU_SAMPLER_GAP_KEYS), "placement gpu sampler gaps")
    counters = ("max_consecutive_allowed", "max_total_allowed", "count", "max_consecutive")
    if any(type(gaps[key]) is not int or gaps[key] < 0 for key in counters):
        raise _sampler_refusal("gap counters must be non-negative integers")
    allowed_consecutive, allowed_total = gaps["max_consecutive_allowed"], gaps["max_total_allowed"]
    if allowed_consecutive > GPU_SAMPLER_GAP_BUDGET_CAP or allowed_total > GPU_SAMPLER_GAP_BUDGET_CAP:
        raise _sampler_refusal("gap budget exceeds its wire cap")
    events = _sampler_list(gaps["events"], "gap events")
    if gaps["count"] != len(events):
        raise _sampler_refusal("gap count disagrees with its events")
    # Retained successful attempts (validated) and the count of dropped ones.
    successes = [(float(sample["monotonic_start"]), float(sample["monotonic_end"])) for sample in record["samples"]]
    dropped = record["aggregate"]["samples_taken"] - record["count"]
    # Eviction window of a dropped success: before the newest retained sample
    # and, with two or more retained (first pinned), after the first.
    newest_start = successes[-1][0] if successes else None
    first_end = successes[0][1] if len(successes) > 1 else None
    previous_end, run, longest, summed, unproven_resets = start, 0, 0, 0.0, 0
    for index, event in enumerate(events):
        event = _fields(event, set(GPU_SAMPLER_GAP_EVENT_KEYS), "placement gpu sampler gap event")
        if event["code"] not in GPU_SAMPLER_GAP_CODES:
            raise _sampler_refusal("gap event is not a forgivable transient failure")
        event_start = _sampler_number(event["monotonic_start"], "gap start")
        event_end = _sampler_number(event["monotonic_end"], "gap end")
        consecutive = _sampler_positive_int(event["consecutive"], "gap consecutive run")
        if start is None or event_end < event_start or event_start < previous_end or event_end > end:
            raise _sampler_refusal("gap interval is not ordered inside the outer interval")
        if consecutive not in (1, run + 1):
            raise _sampler_refusal("gap consecutive run numbering is inconsistent")
        # A failed attempt never overlaps a successful one.
        if any(event_start < success_end and success_start < event_end for success_start, success_end in successes):
            raise _sampler_refusal("gap overlaps a retained successful sample")
        if index:
            succeeded_between = any(
                previous_end <= success_start and success_end <= event_start for success_start, success_end in successes
            )
            if succeeded_between and consecutive != 1:
                raise _sampler_refusal("gap run continues across a retained successful sample")
            if consecutive == 1 and not succeeded_between:
                # Only a dropped sample inside the eviction window can have reset this run.
                if newest_start is None or event_end > newest_start or (first_end is not None and previous_end < first_end):
                    raise _sampler_refusal("gap run resets where no dropped successful sample can lie")
                unproven_resets += 1
        # The first excess ends sampling: no earlier event may exceed a budget.
        if index < len(events) - 1 and (index + 1 > allowed_total or consecutive > allowed_consecutive):
            raise _sampler_refusal("gaps continue past the first budget excess")
        previous_end, run = event_end, consecutive
        longest = max(longest, consecutive)
        summed += event_end - event_start
    if unproven_resets > dropped:
        raise _sampler_refusal("gap run resets without an intervening successful sample")
    if gaps["max_consecutive"] != longest:
        raise _sampler_refusal("gap max_consecutive disagrees with its events")
    if abs(_sampler_number(gaps["total_seconds"], "gap total_seconds") - summed) > 1e-6:
        raise _sampler_refusal("gap total_seconds disagrees with its events")
    if len(events) > allowed_total + 1 or longest > allowed_consecutive + 1:
        raise _sampler_refusal("gaps continue past the first budget excess")
    exceeded = len(events) > allowed_total or longest > allowed_consecutive
    # The excess attempt is the last attempt: no success may follow it.
    if exceeded and any(success_start >= previous_end for success_start, _success_end in successes):
        raise _sampler_refusal("gap budget excess is followed by a successful sample")
    failure = record["failure"]
    if exceeded != (failure is not None and failure["code"] == "gap-budget-exceeded"):
        raise _sampler_refusal("gap budget excess and gap-budget-exceeded failure disagree")


def _placement_v3_sampler(value: Any) -> dict[str, Any]:
    """Mirror the executor's closed ``gpu_sampler`` validator.

    Rejects unknown keys, inconsistent status/reason/attempted labels,
    untyped failures, observation without confirmed quiescence or a
    resolved executable, survivors under a quiescent label, unordered or
    uncontained sample intervals, PID rows absent from their own sample's
    inventory, a lifetime aggregate that disagrees with the retained samples,
    and a gap block that disagrees with its budget or failure code.
    """

    record = dict(_fields(value, set(GPU_SAMPLER_KEYS), "placement gpu sampler"))
    if record["method"] != GPU_SAMPLER_METHOD:
        raise _sampler_refusal("method is invalid")
    status, reason = record["status"], record["reason"]
    if (status, reason) not in {("observed", "observed"), ("unavailable", "telemetry-failed"), ("unavailable", "not-sampled")}:
        raise _sampler_refusal("status/reason pair is invalid")
    if type(record["attempted"]) is not bool or type(record["quiescent"]) is not bool:
        raise _sampler_refusal("attempted/quiescent must be booleans")
    if (reason == "not-sampled") != (not record["attempted"]):
        raise _sampler_refusal("not-sampled must be exactly the unattempted state")
    if reason == "not-sampled":
        if record != _GPU_NOT_SAMPLED:
            raise _sampler_refusal("not-sampled record must carry no sampler evidence")
        return record
    failure = record["failure"]
    if reason == "observed":
        if failure is not None:
            raise _sampler_refusal("observed record cannot carry a failure")
    else:
        failure = dict(_fields(failure, {"code", "detail"}, "placement gpu sampler failure"))
        if failure["code"] not in GPU_SAMPLER_FAILURE_CODES:
            raise _sampler_refusal("failure code is unknown")
        _text(failure["detail"], "placement.gpu_sampler.failure.detail")
    version = record["version"]
    if version is not None:
        _text(version, "placement.gpu_sampler.version")
    executable = record["executable"]
    if executable is not None:
        _text(executable, "placement.gpu_sampler.executable")
    cadence = _sampler_number(record["cadence_seconds"], "cadence_seconds")
    if cadence != 0.0 and not GPU_SAMPLER_CADENCE_BOUNDS[0] <= cadence <= GPU_SAMPLER_CADENCE_BOUNDS[1]:
        raise _sampler_refusal("cadence is outside its bound")
    if not 0.0 < _sampler_number(record["join_timeout_seconds"], "join_timeout_seconds") <= GPU_SAMPLER_MAX_JOIN_TIMEOUT:
        raise _sampler_refusal("join timeout is outside its bound")
    elapsed = _sampler_number(record["elapsed_seconds"], "elapsed_seconds")
    samples = _sampler_list(record["samples"], "samples")
    count = record["count"]
    if type(count) is not int or not 0 <= count <= GPU_SAMPLER_MAX_SAMPLES or count != len(samples):
        raise _sampler_refusal("count disagrees with its samples")
    command_pids = [_sampler_positive_int(pid, "command PID") for pid in _sampler_list(record["command_pids"], "command_pids")]
    survivors = [_sampler_positive_int(pid, "survivor") for pid in _sampler_list(record["survivors"], "survivors")]
    if len(set(command_pids)) != len(command_pids) or len(set(survivors)) != len(survivors):
        raise _sampler_refusal("PIDs repeat")
    if not set(survivors) <= set(command_pids):
        raise _sampler_refusal("survivor was never launched")
    if record["quiescent"] and survivors:
        raise _sampler_refusal("quiescent record cannot report survivors")
    target = record["target_pid"]
    if target is not None:
        _sampler_positive_int(target, "target PID")
    start, end = record["monotonic_start"], record["monotonic_end"]
    if start is None:
        if end is not None or count or elapsed != 0.0:
            raise _sampler_refusal("record without an outer interval cannot carry samples")
    else:
        start, end = _sampler_number(start, "monotonic_start"), _sampler_number(end, "monotonic_end")
        if end < start or abs(elapsed - (end - start)) > 1e-6:
            raise _sampler_refusal("outer interval disagrees with elapsed time")
    if status == "observed" and (
        not record["quiescent"] or version is None or executable is None or start is None
        or count < 1 or not command_pids
    ):
        raise _sampler_refusal("observed record requires quiescent, complete, versioned samples from a resolved executable")
    previous_end = start
    pci_by_uuid: dict[str, str] = {}
    retained: list[tuple[str, str]] = []
    for sample in samples:
        sample = _fields(sample, {"monotonic_start", "monotonic_end", "inventory", "pid_rows"}, "placement gpu sample")
        sample_start = _sampler_number(sample["monotonic_start"], "sample start")
        sample_end = _sampler_number(sample["monotonic_end"], "sample end")
        if sample_end < sample_start or sample_start < previous_end or sample_end > end:
            raise _sampler_refusal("sample interval is not ordered inside the outer interval")
        previous_end = sample_end
        inventory: set[str] = set()
        for item in _sampler_list(sample["inventory"], "inventory"):
            item = _fields(item, {"uuid", "pci.bus_id"}, "placement gpu sample inventory")
            uuid = _gpu_uuid(item["uuid"], "placement gpu sample inventory uuid")
            pci = _text(item["pci.bus_id"], "placement gpu sample inventory pci.bus_id")
            if uuid in inventory or pci_by_uuid.setdefault(uuid, pci) != pci:
                raise _sampler_refusal("inventory row is inconsistent")
            inventory.add(uuid)
        if not inventory:
            raise _sampler_refusal("sample inventory is empty")
        for item in _sampler_list(sample["pid_rows"], "pid_rows"):
            item = _fields(item, {"pid", "gpu_uuid"}, "placement gpu sample PID row")
            pid = _sampler_positive_int(item["pid"], "row PID")
            uuid = _gpu_uuid(item["gpu_uuid"], "placement gpu sample PID uuid")
            if uuid not in inventory:
                raise _sampler_refusal("PID row is absent from its sample inventory")
            if pid == target and uuid not in {seen for seen, _ in retained}:
                retained.append((uuid, pci_by_uuid[uuid]))
    _placement_v3_aggregate(record, retained, pci_by_uuid)
    _placement_v3_gaps(record, start, end)
    return record


def _placement_v3_device(sampler: Mapping[str, Any], child_pid: int, declared_devices: list[str]) -> tuple[str, list[str], list[str]]:
    """Mirror the executor device decision over the lifetime aggregate."""

    declared = [_gpu_uuid(device, "placement.declared.devices") for device in declared_devices]
    if len(set(declared)) != len(declared):
        raise ManifestRefusal("malformed-record", "declared GPU entitlement repeats a UUID")
    if not declared:
        if sampler["reason"] != "not-sampled":
            raise ManifestRefusal("malformed-record", "CPU-only slot cannot carry attempted GPU sampler evidence")
        return "cpu-only", [], []
    if sampler["reason"] == "not-sampled":
        return "sampler-not-run", [], []
    if sampler["target_pid"] is not None and sampler["target_pid"] != child_pid:
        raise ManifestRefusal("malformed-record", "placement gpu sampler watched a PID other than the science child")
    aggregate = sampler["aggregate"]
    active = [item["uuid"] for item in aggregate["exact_pid_gpus"]]
    active_pci = [item["pci.bus_id"] for item in aggregate["exact_pid_gpus"]]
    if aggregate["outside_use_seen"] != any(uuid not in declared for uuid in active):
        raise ManifestRefusal("malformed-record", "placement gpu sampler outside-use flag contradicts its exact-PID GPUs")
    unresolved = any(not set(declared) <= {item["uuid"] for item in sample["inventory"]} for sample in sampler["samples"])
    if unresolved and not aggregate["entitlement_unresolved_seen"]:
        raise ManifestRefusal("malformed-record", "placement gpu sampler aggregate hides an unresolved entitlement")
    if aggregate["samples_taken"] == sampler["count"] and aggregate["entitlement_unresolved_seen"] != unresolved:
        raise ManifestRefusal("malformed-record", "placement gpu sampler unresolved flag differs from its complete samples")
    # Out-of-entitlement use in any sample, retained or not, dominates.
    if aggregate["outside_use_seen"]:
        return "out-of-entitlement", active, active_pci
    if sampler["reason"] == "telemetry-failed":
        return "sampler-failed", [], []
    if aggregate["entitlement_unresolved_seen"]:
        return "entitlement-unresolved", [], []
    if not active:
        return "no-exact-pid-row", [], []
    return "matched", active, active_pci


def _placement_receipt_v3(value: Any, attempt: Mapping[str, Any], start: Mapping[str, Any], group_digest: str) -> dict[str, Any]:
    """Decode a closed sampler-bearing v3 receipt and re-derive its verdict.

    Host, affinity, and refusal evidence is decoded through the unchanged L1a
    v2 reader as a shadow record; the device verdict, ``gpu`` block, device
    comparison, warning, and science acceptance must equal the derivation.
    """

    record = dict(_fields(
        value,
        {
            "schema", "task_id", "attempt_id", "group_digest", "allocation_id",
            "backend_job_id", "slot_id", "payload_digest", "output_dir", "child",
            "declared", "gpu", "comparison", "refusal", "host_mapping", "gpu_sampler", "verdict",
        },
        "placement receipt v3",
    ))
    verdict = dict(_fields(
        record["verdict"],
        {"host", "device", "qualification", "reason_class", "reason", "science_acceptance", "warning"},
        "placement verdict v3",
    ))
    if any(type(item) is not str for item in verdict.values()):
        raise ManifestRefusal("malformed-record", "placement verdict v3 values must be text")
    reason_class = verdict["reason_class"]
    if reason_class not in {"qualified", "mismatch", "telemetry-failed", "not-sampled", "refused"}:
        raise ManifestRefusal("malformed-record", "placement verdict v3 reason class is invalid")
    base = {key: item for key, item in record.items() if key not in {"host_mapping", "gpu_sampler", "verdict"}}
    base["schema"] = SCIENCE_PLACEMENT_RECEIPT_SCHEMA
    base = _placement_receipt(base, attempt, start, group_digest)
    # The L1a class is forced by evidence; only the matched-host class is
    # refined by device telemetry.
    if reason_class == "refused":
        l1a_class = "refused"
    elif reason_class == "mismatch" and base["refusal"] is not None:
        l1a_class = "mismatch"
    else:
        l1a_class = "not-sampled"
    declares_devices = bool(base["declared"]["devices"])
    shadow = {key: item for key, item in record.items() if key not in {"gpu_sampler"}}
    shadow["schema"] = SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2
    shadow["gpu"] = {
        "status": "unavailable", "reason": "not-sampled", "method": "not-sampled",
        "started_at_utc": base["gpu"]["started_at_utc"], "finished_at_utc": base["gpu"]["finished_at_utc"],
        "samples": [], "active_uuids": [], "active_pci": [],
    }
    shadow["comparison"] = dict(base["comparison"], device="unavailable")
    shadow["verdict"] = {
        "host": verdict["host"], "device": "unavailable", "qualification": "unqualified",
        "reason_class": l1a_class, "reason": verdict["reason"],
        "science_acceptance": "accepted" if l1a_class == "not-sampled" else "not-accepted",
        "warning": (PLACEMENT_NOT_SAMPLED_MESSAGE if l1a_class == "not-sampled" else _L1A_UNQUALIFIED_WARNING) if declares_devices else "",
    }
    shadow = _placement_receipt(shadow, attempt, start, group_digest)
    sampler = _placement_v3_sampler(record["gpu_sampler"])
    code, active, active_pci = _placement_v3_device(sampler, base["child"]["pid"], base["declared"]["devices"])
    device, text = GPU_DEVICE_DECISIONS[code]
    if l1a_class in {"refused", "mismatch"}:
        expected_class, expected_reason = l1a_class, verdict["reason"]
    elif device == "matched":
        expected_class, expected_reason = "qualified", f"qualified: {text}"
    elif device == "mismatch":
        expected_class, expected_reason = "mismatch", f"mismatch: {text}"
    elif code == "cpu-only":
        expected_class, expected_reason = "not-sampled", f"not-sampled: {text}"
    else:
        expected_class, expected_reason = "telemetry-failed", f"telemetry-failed: {text}"
        if code == "sampler-failed":
            expected_reason = f"{expected_reason} ({sampler['failure']['code']})"
    qualified = expected_class == "qualified"
    shows_devices = device in {"matched", "mismatch"}
    expected_gpu = {
        "status": "qualified" if qualified else "unavailable",
        "reason": code,
        "method": "not-sampled" if sampler["reason"] == "not-sampled" else GPU_SAMPLER_METHOD,
        "samples": [],
        "active_uuids": active if shows_devices else [],
        "active_pci": active_pci if shows_devices else [],
    }
    expected_verdict = {
        "host": shadow["verdict"]["host"],
        "device": device,
        "qualification": "qualified" if qualified else "unqualified",
        "reason_class": expected_class,
        "reason": expected_reason,
        "science_acceptance": "accepted" if expected_class in {"qualified", "not-sampled"} else "not-accepted",
        "warning": PLACEMENT_V3_WARNINGS.get(expected_class, "") if declares_devices else "",
    }
    if verdict != expected_verdict:
        raise ManifestRefusal("malformed-record", "placement verdict v3 is not derived from its host and sampler evidence")
    if any(base["gpu"][key] != item for key, item in expected_gpu.items()):
        raise ManifestRefusal("malformed-record", "placement gpu evidence v3 is not derived from its sampler evidence")
    if base["comparison"]["device"] != device:
        raise ManifestRefusal("malformed-record", "placement device comparison v3 is not derived from its sampler evidence")
    record.update({
        "child": base["child"], "declared": base["declared"], "gpu": base["gpu"],
        "comparison": base["comparison"], "refusal": base["refusal"],
        "host_mapping": shadow["host_mapping"], "gpu_sampler": sampler, "verdict": verdict,
    })
    return record


def _placement_receipt(value: Any, attempt: Mapping[str, Any], start: Mapping[str, Any], group_digest: str) -> dict[str, Any]:
    """Decode an optional v1/v2/v3 placement receipt with closed schemas."""

    _validate_json(value)
    schema = value.get("schema") if isinstance(value, dict) else None
    if schema == SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V3:
        return _placement_receipt_v3(value, attempt, start, group_digest)
    if schema == SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2:
        record = dict(_fields(
            value,
            {
                "schema", "task_id", "attempt_id", "group_digest", "allocation_id",
                "backend_job_id", "slot_id", "payload_digest", "output_dir", "child",
                "declared", "gpu", "comparison", "refusal", "host_mapping", "verdict",
            },
            "placement receipt v2",
        ))
        base = dict(record)
        base.pop("host_mapping")
        base.pop("verdict")
        base["schema"] = SCIENCE_PLACEMENT_RECEIPT_SCHEMA
        base = _placement_receipt(base, attempt, start, group_digest)
        record["child"] = base["child"]
        record["declared"] = base["declared"]
        record["gpu"] = base["gpu"]
        record["comparison"] = base["comparison"]
        record["refusal"] = base["refusal"]
        record["host_mapping"] = _placement_v2_host_mapping(record["host_mapping"])
        record["verdict"] = _placement_v2_verdict(record["verdict"], record["declared"])
        if (
            record["gpu"]["status"] != "unavailable"
            or record["gpu"]["reason"] != "not-sampled"
            or record["gpu"]["method"] != "not-sampled"
        ):
            raise ManifestRefusal("malformed-record", "L1a v2 GPU evidence must be unavailable/not-sampled")
        if record["gpu"]["samples"] or record["gpu"]["active_uuids"] or record["gpu"]["active_pci"]:
            raise ManifestRefusal("malformed-record", "L1a v2 GPU evidence must not contain samples")
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
        expected_reason_class = "mismatch" if has_postlaunch_mismatch else ("refused" if record["refusal"] is not None else "not-sampled")
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
            if not has_postlaunch_mismatch or record["refusal"] is None:
                raise ManifestRefusal("malformed-record", "mismatch verdict requires post-launch mismatch evidence")
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


def _evidence_mapping(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise ManifestRefusal("core-evidence-mismatch", f"{label} is not a mapping")
    mapped = to_dict()
    if not isinstance(mapped, Mapping):
        raise ManifestRefusal("core-evidence-mismatch", f"{label} is not a mapping")
    return dict(mapped)


def _v3_evidence(value: Any, label: str) -> tuple[dict[str, Any], bool]:
    evidence = dict(_fields(
        _evidence_mapping(value, label),
        {
            "schema", "run_dir", "verdict", "run_status", "checkpoint", "refusal",
            "resolved_config_identity", "outcome", "checkpoint_ref",
        },
        label,
    ))
    if evidence["schema"] != "tpen.run-evidence/v3":
        raise ManifestRefusal("unknown-schema", f"{label} schema is unsupported")
    _text(evidence["run_dir"], f"{label}.run_dir")
    _text(evidence["verdict"], f"{label}.verdict")
    if evidence["verdict"] not in {"completed", "incomplete", "failed", "refused"}:
        raise ManifestRefusal("malformed-record", f"{label}.verdict is invalid")
    if evidence["run_status"] is not None:
        _text(evidence["run_status"], f"{label}.run_status")
    _text(evidence["outcome"], f"{label}.outcome")
    if evidence["outcome"] not in {"completed", "planned_yield", "failed", "unknown", "refused"}:
        raise ManifestRefusal("malformed-record", f"{label}.outcome is invalid")
    if evidence["verdict"] != "refused" and evidence["run_status"] != evidence["outcome"]:
        raise ManifestRefusal("malformed-record", f"{label} status disagrees with outcome")

    refusal = evidence["refusal"]
    if evidence["verdict"] == "refused" and refusal is None:
        raise ManifestRefusal("malformed-record", f"{label} refused evidence requires refusal")
    if evidence["verdict"] != "refused" and refusal is not None:
        raise ManifestRefusal("malformed-record", f"{label} non-refused evidence cannot contain refusal")
    if refusal is not None:
        refusal = dict(_fields(refusal, {"code", "reason"}, f"{label}.refusal"))
        _text(refusal["code"], f"{label}.refusal.code")
        _text(refusal["reason"], f"{label}.refusal.reason")

    checkpoint = evidence["checkpoint"]
    if checkpoint is not None:
        checkpoint = dict(_fields(
            checkpoint,
            {"checkpoint_dir", "schema_version", "next_iteration", "completed_updates"},
            f"{label}.checkpoint",
        ))
        _text(checkpoint["checkpoint_dir"], f"{label}.checkpoint.checkpoint_dir")
        if type(checkpoint["schema_version"]) is not int:
            raise ManifestRefusal("malformed-record", f"{label}.checkpoint.schema_version must be an integer")
        _nonnegative_int(checkpoint["next_iteration"], f"{label}.checkpoint.next_iteration")
        _nonnegative_int(checkpoint["completed_updates"], f"{label}.checkpoint.completed_updates")

    if evidence["verdict"] == "completed" and (checkpoint is None or refusal is not None):
        raise ManifestRefusal("malformed-record", f"{label} completed evidence requires checkpoint and no refusal")
    if evidence["verdict"] == "refused" and (refusal is None or checkpoint is not None):
        raise ManifestRefusal("malformed-record", f"{label} refused evidence requires refusal and no checkpoint")

    identity = _config_identity(evidence["resolved_config_identity"])
    if identity is None:
        raise ManifestRefusal("malformed-record", f"{label} requires resolved config identity")

    checkpoint_ref = evidence["checkpoint_ref"]
    if checkpoint_ref is not None:
        checkpoint_ref = dict(_fields(
            checkpoint_ref,
            {
                "schema", "checkpoint_dir", "schema_version", "kind", "next_iteration", "completed_updates",
                "manifest_sha256", "model_sha256", "provenance", "content_id", "profile",
            },
            f"{label}.checkpoint_ref",
        ))
        if checkpoint_ref["schema"] != "tpen.checkpoint-ref/v1":
            raise ManifestRefusal("unknown-schema", f"{label}.checkpoint_ref schema is unsupported")
        _text(checkpoint_ref["checkpoint_dir"], f"{label}.checkpoint_ref.checkpoint_dir")
        _text(checkpoint_ref["kind"], f"{label}.checkpoint_ref.kind")
        for key in ("schema_version", "next_iteration", "completed_updates"):
            _nonnegative_int(checkpoint_ref[key], f"{label}.checkpoint_ref.{key}")
        for key in ("manifest_sha256", "model_sha256", "content_id"):
            _sha256(checkpoint_ref[key], f"{label}.checkpoint_ref.{key}")
        if not isinstance(checkpoint_ref["provenance"], dict):
            raise ManifestRefusal("malformed-record", f"{label}.checkpoint_ref.provenance must be an object")
        if checkpoint_ref["profile"] not in {"train_resume", "model_only"}:
            raise ManifestRefusal("malformed-record", f"{label}.checkpoint_ref.profile is invalid")

    if evidence["outcome"] == "planned_yield" and evidence["verdict"] != "refused":
        if checkpoint is None or checkpoint_ref is None or checkpoint_ref["profile"] != "train_resume":
            raise ManifestRefusal("malformed-record", f"{label} planned_yield requires a train_resume checkpoint_ref")
        if any(checkpoint_ref[key] != checkpoint[key] for key in ("checkpoint_dir", "next_iteration", "completed_updates")):
            raise ManifestRefusal("malformed-record", f"{label} planned_yield checkpoint_ref does not match checkpoint summary")
    eligible = (
        evidence["verdict"] != "refused"
        and refusal is None
        and evidence["outcome"] == "planned_yield"
        and checkpoint is not None
        and checkpoint_ref is not None
        and checkpoint_ref["profile"] == "train_resume"
        and checkpoint_ref["checkpoint_dir"] == checkpoint["checkpoint_dir"]
        and checkpoint_ref["next_iteration"] == checkpoint["next_iteration"]
        and checkpoint_ref["completed_updates"] == checkpoint["completed_updates"]
    )
    evidence["refusal"] = refusal
    evidence["checkpoint"] = checkpoint
    evidence["checkpoint_ref"] = checkpoint_ref
    evidence["resolved_config_identity"] = dict(identity)
    return evidence, eligible


def _core_outcome(value: Any, attempt: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"schema", "task_id", "attempt_id", "evidence", "continuation_eligible"}
    record = dict(_fields(value, expected, "core outcome record"))
    if record["schema"] != "tpen-executor/core-outcome-v1":
        raise ManifestRefusal("unknown-schema", "core outcome schema is unsupported")
    for key in ("task_id", "attempt_id"):
        _text(record[key], f"core_outcome.{key}")
    if (record["task_id"], record["attempt_id"]) != (attempt["task_id"], attempt["attempt_id"]):
        raise ManifestRefusal("attempt-receipt-mismatch", "manifest attempt disagrees with its core outcome record")
    if type(record["continuation_eligible"]) is not bool:
        raise ManifestRefusal("malformed-record", "core_outcome.continuation_eligible must be a boolean")
    evidence = dict(_fields(record["evidence"], {
        "schema", "run_dir", "verdict", "run_status", "checkpoint", "refusal",
        "resolved_config_identity", "outcome", "checkpoint_ref",
    }, "core outcome evidence"))
    if evidence["schema"] != "tpen.run-evidence/v3":
        raise ManifestRefusal("unknown-schema", "core outcome evidence schema is unsupported")
    _text(evidence["run_dir"], "core_outcome.evidence.run_dir")
    _text(evidence["verdict"], "core_outcome.evidence.verdict")
    if evidence["verdict"] not in {"completed", "incomplete", "failed", "refused"}:
        raise ManifestRefusal("malformed-record", "core outcome evidence verdict is invalid")
    if evidence["run_status"] is not None:
        _text(evidence["run_status"], "core_outcome.evidence.run_status")
    _text(evidence["outcome"], "core_outcome.evidence.outcome")
    if evidence["outcome"] not in {"completed", "planned_yield", "failed", "unknown", "refused"}:
        raise ManifestRefusal("malformed-record", "core outcome evidence outcome is invalid")
    if evidence["verdict"] != "refused" and evidence["run_status"] != evidence["outcome"]:
        raise ManifestRefusal("malformed-record", "core outcome status disagrees with outcome")
    refusal = evidence["refusal"]
    if evidence["verdict"] == "refused" and refusal is None:
        raise ManifestRefusal("malformed-record", "refused core outcome requires refusal")
    if evidence["verdict"] != "refused" and refusal is not None:
        raise ManifestRefusal("malformed-record", "non-refused core outcome cannot contain refusal")
    if refusal is not None:
        refusal = dict(_fields(refusal, {"code", "reason"}, "core outcome refusal"))
        _text(refusal["code"], "core_outcome.refusal.code")
        _text(refusal["reason"], "core_outcome.refusal.reason")
    identity = _config_identity(evidence["resolved_config_identity"])
    if identity is None:
        raise ManifestRefusal("malformed-record", "v3 core outcome requires resolved config identity")
    checkpoint = evidence["checkpoint"]
    if evidence["verdict"] == "completed" and (checkpoint is None or refusal is not None):
        raise ManifestRefusal("malformed-record", "completed core outcome requires checkpoint and no refusal")
    if evidence["verdict"] == "refused" and (refusal is None or checkpoint is not None):
        raise ManifestRefusal("malformed-record", "refused core outcome requires refusal and no checkpoint")
    if checkpoint is not None:
        checkpoint = dict(_fields(checkpoint, {"checkpoint_dir", "schema_version", "next_iteration", "completed_updates"}, "core outcome checkpoint"))
        _text(checkpoint["checkpoint_dir"], "core_outcome.checkpoint.checkpoint_dir")
        if type(checkpoint["schema_version"]) is not int:
            raise ManifestRefusal("malformed-record", "core outcome checkpoint schema_version must be an integer")
        _nonnegative_int(checkpoint["next_iteration"], "core_outcome.checkpoint.next_iteration")
        _nonnegative_int(checkpoint["completed_updates"], "core_outcome.checkpoint.completed_updates")
    checkpoint_ref = evidence["checkpoint_ref"]
    if checkpoint_ref is not None:
        checkpoint_ref = dict(_fields(checkpoint_ref, {
            "schema", "checkpoint_dir", "schema_version", "kind", "next_iteration", "completed_updates",
            "manifest_sha256", "model_sha256", "provenance", "content_id", "profile",
        }, "core outcome checkpoint_ref"))
        if checkpoint_ref["schema"] != "tpen.checkpoint-ref/v1":
            raise ManifestRefusal("unknown-schema", "core outcome checkpoint_ref schema is unsupported")
        _text(checkpoint_ref["checkpoint_dir"], "core_outcome.checkpoint_ref.checkpoint_dir")
        _text(checkpoint_ref["kind"], "core_outcome.checkpoint_ref.kind")
        for key in ("schema_version", "next_iteration", "completed_updates"):
            _nonnegative_int(checkpoint_ref[key], f"core_outcome.checkpoint_ref.{key}")
        for key in ("manifest_sha256", "model_sha256", "content_id"):
            _sha256(checkpoint_ref[key], f"core_outcome.checkpoint_ref.{key}")
        if not isinstance(checkpoint_ref["provenance"], dict):
            raise ManifestRefusal("malformed-record", "core outcome checkpoint_ref provenance must be an object")
        if checkpoint_ref["profile"] not in {"train_resume", "model_only"}:
            raise ManifestRefusal("malformed-record", "core outcome checkpoint_ref profile is invalid")
    if evidence["outcome"] == "planned_yield" and evidence["verdict"] != "refused":
        if checkpoint is None or checkpoint_ref is None or checkpoint_ref["profile"] != "train_resume":
            raise ManifestRefusal("malformed-record", "planned_yield core outcome requires a train_resume checkpoint_ref")
        if any(checkpoint_ref[key] != checkpoint[key] for key in ("checkpoint_dir", "next_iteration", "completed_updates")):
            raise ManifestRefusal("malformed-record", "planned_yield checkpoint_ref does not match checkpoint summary")
    eligible = (
        evidence["verdict"] != "refused"
        and evidence["refusal"] is None
        and evidence["outcome"] == "planned_yield"
        and checkpoint is not None
        and checkpoint_ref is not None
        and checkpoint_ref["profile"] == "train_resume"
        and checkpoint_ref["checkpoint_dir"] == checkpoint["checkpoint_dir"]
        and checkpoint_ref["next_iteration"] == checkpoint["next_iteration"]
        and checkpoint_ref["completed_updates"] == checkpoint["completed_updates"]
    )
    if record["continuation_eligible"] != eligible:
        raise ManifestRefusal("malformed-record", "core outcome eligibility disagrees with evidence")
    record["evidence"] = evidence
    record["resolved_config_identity"] = dict(identity)
    return record


def _successor_admission(value: Any, attempt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one executor successor-admission manifest projection."""

    record = dict(_fields(
        value,
        {"schema", "predecessor", "successor", "request_digest", "core_acceptance_digest"},
        "successor admission record",
    ))
    if record["schema"] != "tpen-executor/successor-admission-v1":
        raise ManifestRefusal("unknown-schema", "successor admission schema is unsupported")
    predecessor = dict(_fields(
        record["predecessor"],
        {"task_id", "attempt_id", "content_id"},
        "successor admission predecessor",
    ))
    successor = dict(_fields(
        record["successor"],
        {"task_id", "attempt_id"},
        "successor admission successor",
    ))
    for key in ("task_id", "attempt_id"):
        _text(predecessor[key], f"successor_admission.predecessor.{key}")
        _text(successor[key], f"successor_admission.successor.{key}")
    _sha256(predecessor["content_id"], "successor_admission.predecessor.content_id")
    _sha256(record["request_digest"], "successor_admission.request_digest")
    _sha256(record["core_acceptance_digest"], "successor_admission.core_acceptance_digest")
    if (predecessor["task_id"], predecessor["attempt_id"]) != (attempt["task_id"], attempt["attempt_id"]):
        raise ManifestRefusal(
            "attempt-receipt-mismatch",
            "manifest attempt disagrees with its successor admission predecessor",
        )
    record["predecessor"] = predecessor
    record["successor"] = successor
    return record


def _attempt(value: Any, group_digest: str) -> dict[str, Any]:
    expected = {"schema", "task_id", "attempt_id", "start_receipt", "outcome_receipt", "output_dir"}
    if isinstance(value, dict) and "placement_receipt" in value:
        expected.add("placement_receipt")
    schema = value.get("schema") if isinstance(value, Mapping) else None
    if schema == MANIFEST_ATTEMPT_V1:
        record = dict(_fields(value, expected, "manifest attempt"))
    elif schema == MANIFEST_ATTEMPT_V2:
        actual = set(value)
        allowed = (
            expected | {"core_outcome"},
            expected | {"core_outcome", "successor_admission"},
            expected | {"successor_admission"},
        )
        if actual not in allowed:
            raise ManifestRefusal("malformed-record", "manifest attempt keys differ")
        record = dict(_fields(value, actual, "manifest attempt"))
        if "successor_admission" in record and (
            "core_outcome" not in record or record["core_outcome"] is None
        ):
            raise ManifestRefusal("malformed-record", "successor_admission requires core_outcome")
        if "core_outcome" not in record or record["core_outcome"] is None:
            raise ManifestRefusal("malformed-record", "manifest-attempt-v2 requires core_outcome")
    else:
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
    core_outcome = None if "core_outcome" not in record or record["core_outcome"] is None else _core_outcome(record["core_outcome"], record)
    if core_outcome is not None and outcome is not None and core_outcome["evidence"]["outcome"] == "planned_yield" and outcome["status"] == "completed":
        raise ManifestRefusal("malformed-record", "planned_yield core outcome cannot have a completed operational receipt")
    record["core_outcome"] = core_outcome
    if "successor_admission" in record:
        record["successor_admission"] = _successor_admission(record["successor_admission"], record)
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
        and attempt["placement_receipt"]["schema"] in {SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2, SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V3}
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
        and attempt["placement_receipt"]["schema"] in {SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2, SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V3}
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
        expected_message = (
            PLACEMENT_TELEMETRY_UNQUALIFIED_MESSAGE
            if any(entry[2] == "telemetry-failed" for entry in actual_blocking)
            else PLACEMENT_UNQUALIFIED_MESSAGE
        )
        if not has_blocking:
            expected_keys = {"status", "science_acceptance", "message", "not_sampled_gpu_segments"}
            expected_status = "not-sampled"
            expected_acceptance = "accepted"
            expected_message = PLACEMENT_NOT_SAMPLED_MESSAGE
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
            advertised_blocking = _entries(
                placement_verdict["unqualified_gpu_segments"], "manifest unqualified GPU segment",
                {"mismatch", "refused", "telemetry-failed"},
            )
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
        if attempt.get("core_outcome") is not None and attempt["core_outcome"]["evidence"]["outcome"] != "completed":
            raise ManifestRefusal("segment-on-noncompleted-core-outcome", "accepted segment does not reference a completed core outcome")
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


def _public_evidence(value: Any) -> tuple[str, Mapping[str, Any] | None, dict[str, str] | None, dict[str, Any] | None]:
    mapped = _evidence_mapping(value, "core inspection result")
    if mapped.get("schema") == "tpen.run-evidence/v3":
        evidence, _eligible = _v3_evidence(mapped, "core inspection evidence")
        return evidence["verdict"], evidence["checkpoint"], evidence["resolved_config_identity"], evidence
    verdict = mapped.get("verdict")
    checkpoint = mapped.get("checkpoint")
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
    return verdict, checkpoint, _config_identity(mapped.get("resolved_config_identity")), None


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
        _, checkpoint, evidence_identity, inspected_v3 = _public_evidence(evidence)
        if inspected_v3 is not None and inspected_v3["outcome"] != "completed":
            raise ManifestRefusal("segment-on-noncompleted-core-outcome", "accepted segment inspection outcome is not completed")
        core_outcome = attempt.get("core_outcome")
        if core_outcome is not None and inspected_v3 is None:
            raise ManifestRefusal("core-evidence-mismatch", "core outcome requires a v3 inspection result")
        if core_outcome is not None and inspected_v3 is not None:
            if core_outcome["evidence"] != inspected_v3:
                raise ManifestRefusal("core-evidence-mismatch", f"core outcome disagrees with inspected evidence {segment['attempt_id']!r}")
            _inspected_eligible = _v3_evidence(inspected_v3, "core inspection evidence")[1]
            if core_outcome["continuation_eligible"] != _inspected_eligible:
                raise ManifestRefusal("core-evidence-mismatch", f"core outcome eligibility disagrees with inspected evidence {segment['attempt_id']!r}")
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
            _freeze_json(attempt["core_outcome"]),
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
    "MANIFEST_ATTEMPT_V1",
    "MANIFEST_ATTEMPT_V2",
    "PreservedAttempt",
    "RUN_MANIFEST_V1",
    "RUN_MANIFEST_V2",
    "RUN_MANIFEST_V3",
    "SCIENCE_PLACEMENT_RECEIPT_SCHEMA",
    "SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V2",
    "SCIENCE_PLACEMENT_RECEIPT_SCHEMA_V3",
    "ResultSet",
    "read",
]
