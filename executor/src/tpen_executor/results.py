"""Operational receipts and the executor run manifest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import field
from typing import Any, ClassVar

from ._codec import _freeze, canonical_bytes, decode_json, fields, sequence, sha256_text, text, thaw
from .contracts import AllocationId, AttemptId, BackendJobId
from ._records import frozen_record

OUTCOME_STATUSES = frozenset({"completed", "failed", "timed_out", "launch_failed"})
PLACEMENT_COMPARISON_STATES = frozenset({"matched", "mismatch", "unavailable", "refused"})


def _identity(value: Any, cls: type, name: str) -> Any:
    if not isinstance(value, cls):
        raise TypeError(f"{name} must be {cls.__name__}")
    return value


@frozen_record
class StartReceipt:
    """Durable observation that an attempt was started.

    Parameters
    ----------
    task_id : str
        Logical task identity.
    attempt_id, allocation_id, backend_job_id : identity records
        Operational identities.
    group_digest : str
        ``ReadyGroup.identity`` for the issuing group.
    started_at_utc, output_dir : str
        Start timestamp and fenced output namespace.
    """

    schema: ClassVar[str] = "tpen-executor/start-receipt-v1"
    task_id: str
    attempt_id: AttemptId
    allocation_id: AllocationId
    backend_job_id: BackendJobId
    group_digest: str
    started_at_utc: str
    output_dir: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", text(self.task_id, "task_id"))
        _identity(self.attempt_id, AttemptId, "attempt_id"); _identity(self.allocation_id, AllocationId, "allocation_id"); _identity(self.backend_job_id, BackendJobId, "backend_job_id")
        object.__setattr__(self, "group_digest", sha256_text(self.group_digest, "group_digest"))
        object.__setattr__(self, "started_at_utc", text(self.started_at_utc, "started_at_utc")); object.__setattr__(self, "output_dir", text(self.output_dir, "output_dir"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "task_id": self.task_id, "attempt_id": self.attempt_id.value, "allocation_id": self.allocation_id.value, "backend_job_id": self.backend_job_id.value, "group_digest": self.group_digest, "started_at_utc": self.started_at_utc, "output_dir": self.output_dir}

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "StartReceipt":
        fields(data, {"schema", "task_id", "attempt_id", "allocation_id", "backend_job_id", "group_digest", "started_at_utc", "output_dir"}, cls.schema, "StartReceipt")
        if any(type(data[key]) is not str for key in ("task_id", "attempt_id", "allocation_id", "backend_job_id", "group_digest", "started_at_utc", "output_dir")): raise TypeError("StartReceipt has an invalid field type")
        return cls(data["task_id"], AttemptId(data["attempt_id"]), AllocationId(data["allocation_id"]), BackendJobId(data["backend_job_id"]), data["group_digest"], data["started_at_utc"], data["output_dir"])

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "StartReceipt":
        return cls.from_mapping(decode_json(data))


@frozen_record
class OutcomeReceipt:
    """Durable observation of an attempt's operational outcome.

    ``status`` is a closed operational classification. Scientific validity is
    supplied by core evidence, not by this receipt.

    Parameters
    ----------
    task_id : str
        Logical task identity.
    attempt_id : AttemptId
        Attempt identity.
    group_digest : str
        ``ReadyGroup.identity`` for the issuing group.
    finished_at_utc, status, output_dir : str
        Outcome timestamp, operational status, and output namespace.
    result_digest : str or None
        Optional digest of the produced result.
    """

    schema: ClassVar[str] = "tpen-executor/outcome-receipt-v1"
    task_id: str
    attempt_id: AttemptId
    group_digest: str
    finished_at_utc: str
    status: str
    output_dir: str
    result_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", text(self.task_id, "task_id")); _identity(self.attempt_id, AttemptId, "attempt_id")
        object.__setattr__(self, "group_digest", sha256_text(self.group_digest, "group_digest"))
        object.__setattr__(self, "finished_at_utc", text(self.finished_at_utc, "finished_at_utc")); object.__setattr__(self, "status", text(self.status, "status")); object.__setattr__(self, "output_dir", text(self.output_dir, "output_dir"))
        if self.status not in OUTCOME_STATUSES:
            raise ValueError(f"status must be one of {sorted(OUTCOME_STATUSES)!r}")
        if self.result_digest is not None: object.__setattr__(self, "result_digest", sha256_text(self.result_digest, "result_digest"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "task_id": self.task_id, "attempt_id": self.attempt_id.value, "group_digest": self.group_digest, "finished_at_utc": self.finished_at_utc, "status": self.status, "output_dir": self.output_dir, "result_digest": self.result_digest}

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "OutcomeReceipt":
        fields(data, {"schema", "task_id", "attempt_id", "group_digest", "finished_at_utc", "status", "output_dir", "result_digest"}, cls.schema, "OutcomeReceipt")
        if any(type(data[key]) is not str for key in ("task_id", "attempt_id", "group_digest", "finished_at_utc", "status", "output_dir")) or (data["result_digest"] is not None and type(data["result_digest"]) is not str): raise TypeError("OutcomeReceipt has an invalid field type")
        return cls(data["task_id"], AttemptId(data["attempt_id"]), data["group_digest"], data["finished_at_utc"], data["status"], data["output_dir"], data["result_digest"])

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "OutcomeReceipt":
        return cls.from_mapping(decode_json(data))


@frozen_record
class AllocationObservation:
    """Backend-neutral observation of an allocation.

    Parameters
    ----------
    allocation_id : AllocationId
        Allocation being observed.
    observed_at_utc, state : str
        Observation timestamp and backend-neutral state text.
    metadata : mapping
        JSON-shaped backend metadata, deeply frozen on construction.
    """

    schema: ClassVar[str] = "tpen-executor/allocation-observation-v1"
    allocation_id: AllocationId
    observed_at_utc: str
    state: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        _identity(self.allocation_id, AllocationId, "allocation_id"); object.__setattr__(self, "observed_at_utc", text(self.observed_at_utc, "observed_at_utc")); object.__setattr__(self, "state", text(self.state, "state"))
        object.__setattr__(self, "metadata", _freeze(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "allocation_id": self.allocation_id.value, "observed_at_utc": self.observed_at_utc, "state": self.state, "metadata": thaw(self.metadata)}

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "AllocationObservation":
        fields(data, {"schema", "allocation_id", "observed_at_utc", "state", "metadata"}, cls.schema, "AllocationObservation")
        if any(type(data[key]) is not str for key in ("allocation_id", "observed_at_utc", "state")): raise TypeError("AllocationObservation scalar fields have invalid types")
        if not isinstance(data["metadata"], dict): raise TypeError("metadata must be a mapping")
        return cls(AllocationId(data["allocation_id"]), data["observed_at_utc"], data["state"], data["metadata"])

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "AllocationObservation":
        return cls.from_mapping(decode_json(data))


@frozen_record
class SciencePlacementReceipt:
    """Immutable child-generation and placement evidence for one attempt.

    This record intentionally keeps device evidence separate from operational
    and scientific verdicts.  L1a publishes ``gpu.status == "unavailable"``
    with reason ``not-sampled``; active device telemetry belongs to L1b.

    Parameters
    ----------
    task_id, group_digest, allocation_id, backend_job_id, slot_id : str
        Identities that bind the child to its fenced allocation slot.
    attempt_id : AttemptId
        Store-owned attempt identity.
    payload_digest, output_dir : str
        Digest of the configured child payload and its fenced output root.
    child, declared, gpu, comparison : mapping
        Strict JSON-shaped evidence sections.  They are deeply frozen.
    refusal : mapping or None
        Typed child/controller refusal, when placement could not be proven.
    """

    schema: ClassVar[str] = "tpen-executor/science-placement-receipt-v1"
    task_id: str
    attempt_id: AttemptId
    group_digest: str
    allocation_id: str
    backend_job_id: str
    slot_id: str
    payload_digest: str
    output_dir: str
    child: dict[str, Any]
    declared: dict[str, Any]
    gpu: dict[str, Any]
    comparison: dict[str, Any]
    refusal: dict[str, str] | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "allocation_id", "backend_job_id", "slot_id", "output_dir"):
            object.__setattr__(self, name, text(getattr(self, name), name))
        _identity(self.attempt_id, AttemptId, "attempt_id")
        object.__setattr__(self, "group_digest", sha256_text(self.group_digest, "group_digest"))
        object.__setattr__(self, "payload_digest", sha256_text(self.payload_digest, "payload_digest"))
        expected_sections = {
            "child": {"host", "pid", "proc_start_ticks", "affinity", "thread_settings", "cuda_visible_devices", "observed_at_utc"},
            "declared": {"host", "cpus", "devices"},
            "gpu": {"status", "reason", "method", "started_at_utc", "finished_at_utc", "samples", "active_uuids", "active_pci"},
            "comparison": {"host", "affinity", "device"},
        }
        for name in ("child", "declared", "gpu", "comparison"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            if set(value) != expected_sections[name]:
                raise ValueError(f"{name} has invalid keys")
        child = self.child
        if type(child["host"]) is not str or not child["host"] or type(child["pid"]) is not int or child["pid"] < 0:
            raise TypeError("child host and pid are invalid")
        if child["pid"] == 0 and self.refusal is None:
            raise TypeError("child pid must be positive unless placement was refused")
        if type(child["proc_start_ticks"]) is not int or child["proc_start_ticks"] < 0:
            raise TypeError("child proc_start_ticks must be a nonnegative integer")
        if isinstance(child["affinity"], (str, bytes, bytearray)) or not isinstance(child["affinity"], Sequence):
            raise TypeError("child affinity must be a sequence")
        if any(type(cpu) is not int or cpu < 0 for cpu in child["affinity"]):
            raise TypeError("child affinity must contain nonnegative integers")
        if not child["affinity"] and self.refusal is None:
            raise ValueError("child affinity cannot be empty without refusal")
        if not isinstance(child["thread_settings"], Mapping) or any(
            type(key) is not str or (type(value) is not str and value is not None)
            for key, value in child["thread_settings"].items()
        ):
            raise TypeError("child thread_settings is invalid")
        if child["cuda_visible_devices"] is not None and type(child["cuda_visible_devices"]) is not str:
            raise TypeError("child cuda_visible_devices must be text or null")
        for key in ("observed_at_utc",):
            text(child[key], f"child.{key}")
        declared = self.declared
        if type(declared["host"]) is not str or not declared["host"]:
            raise TypeError("declared host must be non-empty text")
        if isinstance(declared["cpus"], (str, bytes, bytearray)) or not isinstance(declared["cpus"], Sequence) or not declared["cpus"]:
            raise TypeError("declared cpus must be a non-empty sequence")
        if any(type(cpu) is not int or cpu < 0 for cpu in declared["cpus"]):
            raise TypeError("declared cpus must contain nonnegative integers")
        if isinstance(declared["devices"], (str, bytes, bytearray)) or not isinstance(declared["devices"], Sequence) or any(type(device) is not str or not device for device in declared["devices"]):
            raise TypeError("declared devices must be a sequence of strings")
        gpu = self.gpu
        for key in ("status", "reason", "method", "started_at_utc", "finished_at_utc"):
            text(gpu[key], f"gpu.{key}")
        if gpu["status"] not in {"unavailable", "qualified"}:
            raise ValueError("gpu.status is invalid")
        for key in ("samples", "active_uuids", "active_pci"):
            if isinstance(gpu[key], (str, bytes, bytearray)) or not isinstance(gpu[key], Sequence):
                raise TypeError(f"gpu.{key} must be a sequence")
        if any(type(value) is not str or not value for value in tuple(gpu["active_uuids"]) + tuple(gpu["active_pci"])):
            raise TypeError("gpu active device values must be non-empty strings")
        comparison = self.comparison
        if any(type(comparison[key]) is not str or not comparison[key] for key in ("host", "affinity", "device")):
            raise TypeError("placement comparison values must be non-empty strings")
        if any(comparison[key] not in PLACEMENT_COMPARISON_STATES for key in ("host", "affinity", "device")):
            raise ValueError("placement comparison values use an unknown state")
        object.__setattr__(self, "child", _freeze(child, "child"))
        object.__setattr__(self, "declared", _freeze(declared, "declared"))
        object.__setattr__(self, "gpu", _freeze(gpu, "gpu"))
        object.__setattr__(self, "comparison", _freeze(comparison, "comparison"))
        if self.refusal is not None:
            if not isinstance(self.refusal, Mapping) or set(self.refusal) != {"code", "reason"}:
                raise TypeError("refusal must contain code and reason")
            if any(type(self.refusal[key]) is not str or not self.refusal[key] for key in ("code", "reason")):
                raise TypeError("refusal code and reason must be non-empty strings")
            object.__setattr__(self, "refusal", _freeze(dict(self.refusal), "refusal"))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-shaped receipt."""

        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id.value,
            "group_digest": self.group_digest,
            "allocation_id": self.allocation_id,
            "backend_job_id": self.backend_job_id,
            "slot_id": self.slot_id,
            "payload_digest": self.payload_digest,
            "output_dir": self.output_dir,
            "child": thaw(self.child),
            "declared": thaw(self.declared),
            "gpu": thaw(self.gpu),
            "comparison": thaw(self.comparison),
            "refusal": None if self.refusal is None else thaw(self.refusal),
        }

    def encode(self) -> bytes:
        """Encode this receipt as canonical JSON bytes."""

        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "SciencePlacementReceipt":
        """Decode and strictly validate one placement receipt."""

        expected = {
            "schema", "task_id", "attempt_id", "group_digest", "allocation_id",
            "backend_job_id", "slot_id", "payload_digest", "output_dir", "child",
            "declared", "gpu", "comparison", "refusal",
        }
        fields(data, expected, cls.schema, "SciencePlacementReceipt")
        if any(type(data[key]) is not str for key in (
            "task_id", "attempt_id", "group_digest", "allocation_id", "backend_job_id",
            "slot_id", "payload_digest", "output_dir",
        )):
            raise TypeError("SciencePlacementReceipt identity fields must be strings")
        if any(not isinstance(data[key], Mapping) for key in ("child", "declared", "gpu", "comparison")):
            raise TypeError("SciencePlacementReceipt evidence sections must be mappings")
        if data["refusal"] is not None and not isinstance(data["refusal"], Mapping):
            raise TypeError("SciencePlacementReceipt refusal must be a mapping or null")
        return cls(
            data["task_id"], AttemptId(data["attempt_id"]), data["group_digest"],
            data["allocation_id"], data["backend_job_id"], data["slot_id"],
            data["payload_digest"], data["output_dir"], data["child"], data["declared"],
            data["gpu"], data["comparison"], data["refusal"],
        )

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "SciencePlacementReceipt":
        """Decode canonical JSON or an already-decoded mapping."""

        return cls.from_mapping(decode_json(data))


@frozen_record
class SciencePlacementReceiptV2:
    """Versioned physical-host placement receipt without device sampling.

    SciencePlacementReceipt remains the immutable v1 wire record.  L1a
    adds only the closed physical host map and verdict envelope.  Device
    sampling is intentionally absent here and belongs to the dependent L1b
    layer.
    """

    schema: ClassVar[str] = "tpen-executor/science-placement-receipt-v2"
    base: SciencePlacementReceipt
    host_mapping: Mapping[str, Any]
    verdict: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.base, SciencePlacementReceipt):
            raise TypeError("v2 base must be SciencePlacementReceipt")
        expected_host = {"status", "reason", "source", "slot_id", "host"}
        if not isinstance(self.host_mapping, Mapping) or set(self.host_mapping) != expected_host:
            raise ValueError("v2 host_mapping keys differ")
        host_status = self.host_mapping["status"]
        if host_status not in {"matched", "mismatch", "unavailable"}:
            raise ValueError("v2 host_mapping status is invalid")
        source = self.host_mapping["source"]
        if source not in {"scheduler-node-map", "declared-slot"}:
            raise ValueError("v2 host_mapping source is invalid")
        for key in ("reason", "slot_id", "host"):
            if type(self.host_mapping[key]) is not str or not self.host_mapping[key]:
                raise TypeError(f"v2 host_mapping {key} must be non-empty text")
        if host_status == "unavailable" and self.host_mapping["host"] != "unavailable":
            raise ValueError("unavailable v2 host mapping must not name a logical host")
        if host_status != "unavailable" and source != "scheduler-node-map":
            raise ValueError("mapped v2 host evidence must come from the scheduler node map")
        expected_verdict = {
            "host", "device", "qualification", "reason_class", "reason",
            "science_acceptance", "warning",
        }
        if not isinstance(self.verdict, Mapping) or set(self.verdict) != expected_verdict:
            raise ValueError("v2 verdict keys differ")
        for key in expected_verdict:
            if type(self.verdict[key]) is not str or (key != "warning" and not self.verdict[key]):
                raise TypeError(f"v2 verdict {key} must be non-empty text")
        if self.verdict["host"] not in PLACEMENT_COMPARISON_STATES:
            raise ValueError("v2 verdict host state is invalid")
        if self.verdict["device"] != "unavailable":
            raise ValueError("L1a device verdict must be unavailable")
        if self.verdict["qualification"] != "unqualified":
            raise ValueError("L1a placement qualification must be unqualified")
        if self.verdict["reason_class"] not in {"mismatch", "not-sampled", "refused"}:
            raise ValueError("v2 verdict reason_class is invalid for L1a")
        if self.verdict["science_acceptance"] not in {"accepted", "not-accepted"}:
            raise ValueError("v2 verdict science acceptance is invalid")
        if self.base.gpu["status"] != "unavailable" or self.base.gpu["reason"] != "not-sampled":
            raise ValueError("L1a v2 receipts must carry device reason not-sampled")
        child_host = self.base.child["host"].strip().lower().rstrip(".").split(".", 1)[0]
        declared_host = self.base.declared["host"].strip().lower().rstrip(".").split(".", 1)[0]
        mapped_host = self.host_mapping["host"].strip().lower().rstrip(".").split(".", 1)[0]
        affinity_matches = tuple(sorted(self.base.child["affinity"])) == tuple(sorted(self.base.declared["cpus"]))
        if host_status == "unavailable":
            if self.base.comparison["host"] not in {"unavailable", "refused"}:
                raise ValueError("unavailable v2 host map contradicts base host comparison")
        else:
            if declared_host != mapped_host:
                raise ValueError("v2 declared host differs from physical scheduler host")
            expected_host_status = "matched" if child_host == mapped_host else "mismatch"
            if host_status != expected_host_status or self.base.comparison["host"] != expected_host_status:
                raise ValueError("v2 host mapping contradicts child host evidence")
        if self.base.comparison["affinity"] == "matched" and not affinity_matches:
            raise ValueError("v2 matched affinity contradicts child affinity")
        if self.base.comparison["affinity"] == "mismatch" and affinity_matches:
            raise ValueError("v2 mismatched affinity contradicts child affinity")
        if self.verdict["host"] != host_status:
            raise ValueError("v2 verdict host contradicts host mapping")
        has_postlaunch_mismatch = (
            self.base.comparison["host"] == "mismatch"
            or self.base.comparison["affinity"] == "mismatch"
        )
        if self.verdict["reason_class"] == "not-sampled":
            if self.base.refusal is not None or has_postlaunch_mismatch:
                raise ValueError("not-sampled verdict cannot hide a placement mismatch or refusal")
            if self.verdict["science_acceptance"] != "accepted":
                raise ValueError("not-sampled device evidence is not science-blocking")
        elif self.verdict["reason_class"] == "mismatch":
            if not has_postlaunch_mismatch or self.base.refusal is None:
                raise ValueError("mismatch verdict requires post-launch mismatch evidence")
            if self.verdict["science_acceptance"] != "not-accepted":
                raise ValueError("mismatched placement cannot be science-accepted")
        elif self.verdict["reason_class"] == "refused":
            if self.base.refusal is None or has_postlaunch_mismatch:
                raise ValueError("refused verdict requires a prelaunch refusal")
            if self.verdict["science_acceptance"] != "not-accepted":
                raise ValueError("refused placement cannot be science-accepted")
        if self.base.declared["devices"] and not self.verdict["warning"]:
            raise ValueError("unqualified declared GPU placement requires a loud warning")
        for name, value in (("host_mapping", self.host_mapping), ("verdict", self.verdict)):
            object.__setattr__(self, name, _freeze(value, name))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def to_dict(self) -> dict[str, Any]:
        result = self.base.to_dict()
        result.update({
            "schema": self.schema,
            "host_mapping": thaw(self.host_mapping),
            "verdict": thaw(self.verdict),
        })
        return result

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "SciencePlacementReceiptV2":
        if not isinstance(data, Mapping):
            raise TypeError("SciencePlacementReceiptV2 must be a mapping")
        expected = {
            "schema", "task_id", "attempt_id", "group_digest", "allocation_id", "backend_job_id", "slot_id",
            "payload_digest", "output_dir", "child", "declared", "gpu", "comparison", "refusal",
            "host_mapping", "verdict",
        }
        if set(data) != expected or data.get("schema") != cls.schema:
            raise ValueError("SciencePlacementReceiptV2 keys or schema differ")
        base = dict(data)
        base.pop("host_mapping")
        base.pop("verdict")
        base["schema"] = SciencePlacementReceipt.schema
        return cls(
            SciencePlacementReceipt.from_mapping(base),
            data["host_mapping"],
            data["verdict"],
        )

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "SciencePlacementReceiptV2":
        return cls.from_mapping(decode_json(data))


@frozen_record
class ManifestAttempt:
    """One group member's attempt and receipt references.

    Parameters
    ----------
    task_id, output_dir : str
        Redundant indexed fields cross-checked against the start receipt.
    attempt_id : AttemptId
        Attempt identity.
    start_receipt : StartReceipt
        Required start observation.
    outcome_receipt : OutcomeReceipt or None
        Optional terminal observation.
    """

    schema: ClassVar[str] = "tpen-executor/manifest-attempt-v1"
    task_id: str
    attempt_id: AttemptId
    start_receipt: StartReceipt
    outcome_receipt: OutcomeReceipt | None
    output_dir: str
    placement_receipt: SciencePlacementReceipt | SciencePlacementReceiptV2 | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", text(self.task_id, "task_id")); _identity(self.attempt_id, AttemptId, "attempt_id"); _identity(self.start_receipt, StartReceipt, "start_receipt"); _identity(self.start_receipt.attempt_id, AttemptId, "start_receipt.attempt_id")
        if self.task_id != self.start_receipt.task_id or self.attempt_id != self.start_receipt.attempt_id or self.output_dir != self.start_receipt.output_dir:
            raise ValueError("manifest attempt disagrees with start receipt")
        if self.outcome_receipt is not None and not isinstance(self.outcome_receipt, OutcomeReceipt): raise TypeError("outcome_receipt must be OutcomeReceipt or None")
        if self.outcome_receipt is not None and (self.task_id != self.outcome_receipt.task_id or self.attempt_id != self.outcome_receipt.attempt_id or self.output_dir != self.outcome_receipt.output_dir or self.start_receipt.group_digest != self.outcome_receipt.group_digest):
            raise ValueError("manifest attempt disagrees with outcome receipt")
        if self.placement_receipt is not None:
            if not isinstance(self.placement_receipt, (SciencePlacementReceipt, SciencePlacementReceiptV2)):
                raise TypeError("placement_receipt must be SciencePlacementReceipt, SciencePlacementReceiptV2, or None")
            if (
                self.task_id != self.placement_receipt.task_id
                or self.attempt_id != self.placement_receipt.attempt_id
                or self.output_dir != self.placement_receipt.output_dir
                or self.start_receipt.group_digest != self.placement_receipt.group_digest
                or self.start_receipt.allocation_id.value != self.placement_receipt.allocation_id
                or self.start_receipt.backend_job_id.value != self.placement_receipt.backend_job_id
            ):
                raise ValueError("manifest attempt disagrees with placement receipt")
        object.__setattr__(self, "output_dir", text(self.output_dir, "output_dir"))

    def to_dict(self) -> dict[str, Any]:
        result = {"schema": self.schema, "task_id": self.task_id, "attempt_id": self.attempt_id.value, "start_receipt": self.start_receipt.to_dict(), "outcome_receipt": None if self.outcome_receipt is None else self.outcome_receipt.to_dict(), "output_dir": self.output_dir}
        if self.placement_receipt is not None:
            result["placement_receipt"] = self.placement_receipt.to_dict()
        return result

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "ManifestAttempt":
        legacy = {"schema", "task_id", "attempt_id", "start_receipt", "outcome_receipt", "output_dir"}
        current = legacy | {"placement_receipt"}
        if not isinstance(data, Mapping) or set(data) not in (legacy, current):
            raise ValueError("ManifestAttempt keys differ")
        fields(data, set(data), cls.schema, "ManifestAttempt")
        if any(type(data[key]) is not str for key in ("task_id", "attempt_id", "output_dir")): raise TypeError("ManifestAttempt has an invalid scalar field")
        outcome = None if data["outcome_receipt"] is None else OutcomeReceipt.from_mapping(data["outcome_receipt"])
        placement_data = data.get("placement_receipt")
        if placement_data is None:
            placement = None
        elif placement_data.get("schema") == SciencePlacementReceiptV2.schema:
            placement = SciencePlacementReceiptV2.from_mapping(placement_data)
        else:
            placement = SciencePlacementReceipt.from_mapping(placement_data)
        return cls(data["task_id"], AttemptId(data["attempt_id"]), StartReceipt.from_mapping(data["start_receipt"]), outcome, data["output_dir"], placement)

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "ManifestAttempt":
        return cls.from_mapping(decode_json(data))


@frozen_record
class AcceptedSegment:
    """One core-validated segment selected for scientific continuation.

    Parameters
    ----------
    task_id : str
        Logical task whose attempt published the checkpoint.
    attempt_id : AttemptId
        Attempt that produced the checkpoint.
    predecessor_checkpoint : str or None
        Checkpoint directory from which this segment continued, or ``None``
        for a fresh-start segment.
    checkpoint_dir : str
        Core-published checkpoint directory for this segment.
    next_iteration, completed_updates : int
        Independent progress coordinates copied from core public evidence.
    """

    schema: ClassVar[str] = "tpen-executor/accepted-segment-v1"
    task_id: str
    attempt_id: AttemptId
    predecessor_checkpoint: str | None
    checkpoint_dir: str
    next_iteration: int
    completed_updates: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", text(self.task_id, "task_id"))
        _identity(self.attempt_id, AttemptId, "attempt_id")
        if self.predecessor_checkpoint is not None:
            object.__setattr__(self, "predecessor_checkpoint", text(self.predecessor_checkpoint, "predecessor_checkpoint"))
        object.__setattr__(self, "checkpoint_dir", text(self.checkpoint_dir, "checkpoint_dir"))
        if type(self.next_iteration) is not int or self.next_iteration < 0:
            raise TypeError("next_iteration must be a nonnegative integer")
        if type(self.completed_updates) is not int or self.completed_updates < 0:
            raise TypeError("completed_updates must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id.value,
            "predecessor_checkpoint": self.predecessor_checkpoint,
            "checkpoint_dir": self.checkpoint_dir,
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
        }

    def encode(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "AcceptedSegment":
        fields(
            data,
            {
                "schema",
                "task_id",
                "attempt_id",
                "predecessor_checkpoint",
                "checkpoint_dir",
                "next_iteration",
                "completed_updates",
            },
            cls.schema,
            "AcceptedSegment",
        )
        if any(type(data[key]) is not str for key in ("task_id", "attempt_id", "checkpoint_dir")):
            raise TypeError("AcceptedSegment text fields have invalid types")
        if data["predecessor_checkpoint"] is not None and type(data["predecessor_checkpoint"]) is not str:
            raise TypeError("predecessor_checkpoint must be a string or null")
        return cls(
            data["task_id"],
            AttemptId(data["attempt_id"]),
            data["predecessor_checkpoint"],
            data["checkpoint_dir"],
            data["next_iteration"],
            data["completed_updates"],
        )

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "AcceptedSegment":
        return cls.from_mapping(decode_json(data))


@frozen_record
class RunManifest:
    """Versioned group manifest that points at operational receipts.

    ``group_digest`` is the ``ReadyGroup.identity`` value. Every receipt in
    ``attempts`` must carry the same value.

    Parameters
    ----------
    group_id : str
        Stable group identity.
    group_digest : str
        Order-sensitive ``ReadyGroup.identity``.
    attempts : sequence of ManifestAttempt
        Attempt records, with unique attempt identities.
    segments : sequence of AcceptedSegment, optional
        Accepted scientific segments, ordered independently for each task.
    """

    schema: ClassVar[str] = "tpen-executor/run-manifest-v2"
    V1_SCHEMA: ClassVar[str] = "tpen-executor/run-manifest-v1"
    V2_SCHEMA: ClassVar[str] = schema
    V3_SCHEMA: ClassVar[str] = "tpen-executor/run-manifest-v3"
    group_id: str
    group_digest: str
    attempts: tuple[ManifestAttempt, ...]
    segments: tuple[AcceptedSegment, ...] = ()
    placement_verdict: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", text(self.group_id, "group_id")); object.__setattr__(self, "group_digest", sha256_text(self.group_digest, "group_digest"))
        attempts = sequence(self.attempts, "attempts")
        if any(not isinstance(attempt, ManifestAttempt) for attempt in attempts): raise TypeError("attempts must contain ManifestAttempt records")
        if len({attempt.attempt_id for attempt in attempts}) != len(attempts): raise ValueError("duplicate attempt_id in manifest")
        if any(attempt.start_receipt.group_digest != self.group_digest or (attempt.outcome_receipt is not None and attempt.outcome_receipt.group_digest != self.group_digest) for attempt in attempts):
            raise ValueError("receipt group_digest does not match manifest group_digest")
        object.__setattr__(self, "attempts", attempts)
        segments = sequence(self.segments, "segments")
        if any(not isinstance(segment, AcceptedSegment) for segment in segments):
            raise TypeError("segments must contain AcceptedSegment records")
        attempts_by_id = {attempt.attempt_id: attempt for attempt in attempts}
        if len({segment.attempt_id for segment in segments}) != len(segments):
            raise ValueError("duplicate attempt_id in accepted segments")
        previous_by_task: dict[str, AcceptedSegment] = {}
        for segment in segments:
            attempt = attempts_by_id.get(segment.attempt_id)
            if attempt is None:
                raise ValueError("accepted segment references an unknown attempt")
            if attempt.task_id != segment.task_id:
                raise ValueError("accepted segment task_id disagrees with attempt")
            if attempt.outcome_receipt is None or attempt.outcome_receipt.status != "completed":
                raise ValueError("accepted segment requires a completed attempt")
            previous = previous_by_task.get(segment.task_id)
            if previous is None:
                if segment.predecessor_checkpoint is not None:
                    raise ValueError("first accepted segment must have no predecessor")
            else:
                if segment.predecessor_checkpoint != previous.checkpoint_dir:
                    raise ValueError("accepted segment predecessor disagrees with prior checkpoint")
                previous_coordinate = (previous.next_iteration, previous.completed_updates)
                coordinate = (segment.next_iteration, segment.completed_updates)
                if coordinate[0] < previous_coordinate[0] or coordinate[1] < previous_coordinate[1] or coordinate == previous_coordinate:
                    raise ValueError("accepted segment progress coordinate is not strictly increasing")
            previous_by_task[segment.task_id] = segment
        object.__setattr__(self, "segments", segments)
        placement_verdict = self.placement_verdict
        if not isinstance(placement_verdict, Mapping):
            raise TypeError("placement_verdict must be a mapping")
        not_sampled = []
        blocking = []
        for attempt in attempts:
            receipt = attempt.placement_receipt
            if not isinstance(receipt, SciencePlacementReceiptV2) or not receipt.declared["devices"]:
                continue
            entry = {
                "task_id": receipt.task_id,
                "attempt_id": receipt.attempt_id.value,
                "reason_class": receipt.verdict["reason_class"],
                "reason": receipt.verdict["reason"],
            }
            if receipt.verdict["reason_class"] == "not-sampled":
                not_sampled.append(entry)
            else:
                blocking.append(entry)
        derived_verdict: Mapping[str, Any] = {}
        if blocking:
            derived_verdict = {
                "status": "unqualified",
                "science_acceptance": "not-accepted",
                "message": "GPU PLACEMENT UNQUALIFIED: placement mismatch/refusal requires science review.",
                "unqualified_gpu_segments": blocking,
            }
            if not_sampled:
                derived_verdict = dict(derived_verdict)
                derived_verdict["not_sampled_gpu_segments"] = not_sampled
        elif not_sampled:
            derived_verdict = {
                "status": "not-sampled",
                "science_acceptance": "accepted",
                "message": "GPU PLACEMENT NOT-SAMPLED: L1a intentionally does not sample devices; H-v2 performs its own delivered-GPU check.",
                "not_sampled_gpu_segments": not_sampled,
            }
        if placement_verdict and dict(placement_verdict) != dict(derived_verdict):
            raise ValueError("manifest placement_verdict disagrees with placement receipts")
        placement_verdict = derived_verdict
        object.__setattr__(self, "placement_verdict", _freeze(placement_verdict, "placement_verdict"))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema": self.V3_SCHEMA if self.placement_verdict else self.schema,
            "group_id": self.group_id,
            "group_digest": self.group_digest,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "segments": [segment.to_dict() for segment in self.segments],
        }
        if self.placement_verdict:
            result["placement_verdict"] = thaw(self.placement_verdict)
        return result

    def encode(self) -> bytes: return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "RunManifest":
        if not isinstance(data, Mapping):
            raise TypeError("RunManifest must be a mapping")
        schema = data.get("schema")
        if schema == cls.V1_SCHEMA:
            fields(data, {"schema", "group_id", "group_digest", "attempts"}, cls.V1_SCHEMA, "RunManifest")
            segments: tuple[AcceptedSegment, ...] = ()
            placement_verdict: Mapping[str, Any] = {}
        elif schema in (cls.schema, cls.V3_SCHEMA):
            expected = {"schema", "group_id", "group_digest", "attempts", "segments"}
            if schema == cls.V3_SCHEMA:
                expected.add("placement_verdict")
            fields(data, expected, schema, "RunManifest")
            segments = tuple(AcceptedSegment.from_mapping(item) for item in sequence(data["segments"], "segments"))
            placement_verdict = data.get("placement_verdict", {})
            if not isinstance(placement_verdict, Mapping):
                raise TypeError("placement_verdict must be a mapping")
        else:
            raise ValueError("RunManifest schema is unknown")
        if type(data["group_id"]) is not str or type(data["group_digest"]) is not str: raise TypeError("RunManifest scalar fields have invalid types")
        return cls(data["group_id"], data["group_digest"], tuple(ManifestAttempt.from_mapping(item) for item in sequence(data["attempts"], "attempts")), segments, placement_verdict)

    @classmethod
    def decode(cls, data: bytes | str | dict[str, Any]) -> "RunManifest":
        return cls.from_mapping(decode_json(data))


__all__ = ["AcceptedSegment", "AllocationObservation", "ManifestAttempt", "OutcomeReceipt", "OUTCOME_STATUSES", "RunManifest", "SciencePlacementReceipt", "SciencePlacementReceiptV2", "StartReceipt"]


def encode_manifest(manifest: RunManifest) -> bytes:
    """Encode one validated run manifest."""

    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be RunManifest")
    return manifest.encode()


def decode_manifest(data: bytes | str | dict[str, Any]) -> RunManifest:
    """Decode one run manifest."""

    return RunManifest.decode(data)
