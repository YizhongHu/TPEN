"""Core-owned admission facts for the core/executor recovery boundary.

This module deliberately has no executor dependency.  It inspects one core
run, applies the declared native profile, and (for a typed planned yield)
delegates successor selection to :mod:`tpen.execution.recovery`.  The result
is a small canonical mapping which a separate composition root may hand to an
executor-side decoder.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tpen.checkpoint.profiles import ProfileCheck, check_profile
from tpen.inspection import RUN_EVIDENCE_SCHEMA_V3, RunEvidence, inspect_run

from .recovery import (
    ScienceIntent,
    SuccessorAdmissionDecodeRefusal,
    SuccessorAdmissionRequest,
    decode_successor_admission_request,
    select_successor_admission,
)


CORE_ADMISSION_SCHEMA = "tpen-core-admission-v1"
_ADMISSION_FIELDS = {
    "schema",
    "outcome",
    "admission",
    "checkpoint",
    "quiescence",
    "profile",
    "reason",
}
_DIGEST_CHARS = frozenset("0123456789abcdef")
_CHECKPOINT_ADMISSION_REFUSALS = frozenset(
    {
        "corrupt-publication-catalog",
        "incomplete-publication",
        "missing-checkpoint-ref",
        "model-only-checkpoint",
        "outcome-checkpoint-mismatch",
        "stale-checkpoint-ref",
        "torn-publication-catalog",
        "unpublished-checkpoint-ref",
    }
)


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _digest(value: Any, name: str) -> str:
    value = _text(value, name)
    if len(value) != 64 or value != value.lower() or any(char not in _DIGEST_CHARS for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class CoreAdmissionContextV1:
    """Validated core inputs for one member admission evaluation.

    Parameters
    ----------
    task_id, lineage_id : str
        Immutable allocation identity supplied by the composition root.
    core_run_dir : path-like or None
        Core run leaf to inspect.  ``None`` means that this member has not
        started and therefore has no core admission mapping.
    science_input_digest : str
        Lowercase SHA-256 digest binding the successor request to science
        intent.  The digest is opaque to this boundary.
    successor_request : SuccessorAdmissionRequest or mapping, optional
        Existing closed request for a typed planned-yield successor.
    unresolved_config : mapping
        Unresolved configuration syntax checked by the native profile gate.
    profile_name : str
        Declared native recovery profile name.

    Notes
    -----
    The composition root guarantees the task/lineage-to-leaf binding before
    constructing this context.  Core evidence does not carry lineage, so this
    evaluator binds the leaf, request evidence, and science digest but does
    not claim to verify that outer task/lineage guarantee.
    """

    task_id: str
    lineage_id: str
    core_run_dir: str | None
    science_input_digest: str
    successor_request: SuccessorAdmissionRequest | Mapping[str, Any] | None
    unresolved_config: Mapping[str, Any]
    profile_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, "lineage_id"))
        if self.core_run_dir is not None:
            if not isinstance(self.core_run_dir, (str, Path)):
                raise TypeError("core_run_dir must be path-like or None")
            object.__setattr__(self, "core_run_dir", str(Path(self.core_run_dir).resolve()))
        object.__setattr__(self, "science_input_digest", _digest(self.science_input_digest, "science_input_digest"))
        if self.successor_request is not None and not isinstance(
            self.successor_request, (SuccessorAdmissionRequest, Mapping)
        ):
            raise TypeError("successor_request must be a request, mapping, or None")
        if not isinstance(self.unresolved_config, Mapping):
            raise TypeError("unresolved_config must be a mapping")
        object.__setattr__(self, "unresolved_config", MappingProxyType(dict(self.unresolved_config)))
        object.__setattr__(self, "profile_name", _text(self.profile_name, "profile_name"))


def _mapping(
    outcome: str,
    admission: str,
    checkpoint: str,
    profile: Mapping[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    """Build the closed v1 mapping shared with the executor decoder."""

    return {
        "schema": CORE_ADMISSION_SCHEMA,
        "outcome": outcome,
        "admission": admission,
        "checkpoint": checkpoint,
        "quiescence": "unknown",
        "profile": None if profile is None else dict(profile),
        "reason": _text(reason, "reason"),
    }


def _unknown(reason: str) -> dict[str, Any]:
    return _mapping("unknown", "unknown", "unknown", None, reason)


def _profile_mapping(result: ProfileCheck) -> dict[str, Any]:
    return result.to_dict()


class _DigestBoundIntent(ScienceIntent):
    """Minimal ScienceIntent carrier when only the immutable digest is given."""

    __slots__ = ("_bound_digest",)

    @property
    def science_input_digest(self) -> str:  # pragma: no cover - exercised through selector
        return self._bound_digest


def _intent_for(context: CoreAdmissionContextV1, request: SuccessorAdmissionRequest) -> ScienceIntent:
    intent = _DigestBoundIntent({}, target_iterations=request.target_iterations)
    object.__setattr__(intent, "_bound_digest", context.science_input_digest)
    return intent


def _same_path(left: str, right: str) -> bool:
    """Compare two leaf references after resolving their filesystem spelling."""

    try:
        return Path(left).resolve() == Path(right).resolve()
    except (OSError, RuntimeError, TypeError):
        return False


def _request_binds_to_inspection(
    context: CoreAdmissionContextV1,
    evidence: Any,
    request: SuccessorAdmissionRequest,
) -> bool:
    """Require one request to name the exact freshly inspected core facts."""

    request_evidence = request.core_evidence
    if not _same_path(context.core_run_dir or "", evidence.run_dir):
        return False
    if not _same_path(request_evidence.run_dir, evidence.run_dir):
        return False
    if request_evidence.resolved_config_identity != evidence.resolved_config_identity:
        return False
    if request.predecessor.resolved_config_identity != evidence.resolved_config_identity:
        return False
    if request_evidence.checkpoint_ref != evidence.checkpoint_ref:
        return False
    if request_evidence.checkpoint != evidence.checkpoint:
        # A missing/stale publication is itself the typed core refusal.  The
        # inspection cannot retain a selected summary after that refusal, but
        # the bound ref and identity above still prevent cross-leaf reuse.
        refusal_code = None if evidence.refusal is None else evidence.refusal.get("code")
        if refusal_code not in _CHECKPOINT_ADMISSION_REFUSALS:
            return False
    return request.science_input_digest == context.science_input_digest


def evaluate_core_admission(context: CoreAdmissionContextV1) -> dict[str, Any] | None:
    """Evaluate core evidence into a canonical ``tpen-core-admission-v1`` map.

    The mapping has the six executor verdict fields ``outcome``, ``admission``,
    ``checkpoint``, ``quiescence``, ``profile``, and ``reason`` plus ``schema``.
    ``quiescence`` is always ``"unknown"`` because only the executor can
    settle operational teardown.  A missing run leaf returns ``None`` so the
    executor can apply its existing ``start_fresh`` rule.
    """

    if not isinstance(context, CoreAdmissionContextV1):
        raise TypeError("context must be CoreAdmissionContextV1")
    if context.core_run_dir is None:
        return None

    try:
        evidence = inspect_run(context.core_run_dir)
    except Exception as exc:  # core I/O/decoder failures are not typed refusals
        return _unknown(f"inspection-error:{type(exc).__name__}")

    request: SuccessorAdmissionRequest | None = None
    if context.successor_request is not None:
        try:
            decoded = (
                context.successor_request
                if isinstance(context.successor_request, SuccessorAdmissionRequest)
                else decode_successor_admission_request(context.successor_request)
            )
        except Exception as exc:
            return _unknown(f"malformed-successor-request:{type(exc).__name__}")
        if isinstance(decoded, SuccessorAdmissionDecodeRefusal):
            return _unknown(decoded.code.value)
        request = decoded
        if request.predecessor.task_id != context.task_id or request.successor.task_id != context.task_id:
            return _unknown("identity-mismatch")
        if not _request_binds_to_inspection(context, evidence, request):
            return _unknown("identity-mismatch")

    if evidence.inspection_unavailable:
        return _unknown("inspection-unavailable")
    if evidence.schema != RUN_EVIDENCE_SCHEMA_V3:
        return _unknown("untyped-core-evidence")
    if evidence.verdict == "refused" or evidence.refusal is not None:
        reason = None if evidence.refusal is None else evidence.refusal.get("code")
        if reason in _CHECKPOINT_ADMISSION_REFUSALS and request is not None:
            return _mapping("successor", "refused", "invalid", None, reason)
        return _unknown(reason or "invalid-core-evidence")
    expected_verdict = {
        "completed": "completed",
        "failed": "failed",
        "unknown": "incomplete",
        "refused": "incomplete",
        "planned_yield": "incomplete",
    }
    if evidence.outcome not in expected_verdict or evidence.verdict != expected_verdict[evidence.outcome]:
        return _unknown("invalid-core-evidence")

    if evidence.outcome == "completed":
        return _mapping("completed", "unknown", "unknown", None, "completed")
    if evidence.outcome in {"failed", "refused"}:
        reason = "failed" if evidence.outcome == "failed" else "refused"
        if evidence.refusal is not None:
            reason = evidence.refusal["code"]
        return _mapping("terminal", "refused", "absent", None, reason)
    if evidence.outcome == "unknown":
        return _unknown("unknown")
    if evidence.outcome != "planned_yield":
        return _unknown("unsupported-outcome")

    if not evidence.continuation_eligible:
        return _unknown("non-eligible-outcome")
    if request is None:
        return _unknown("missing-successor-request")

    try:
        profile_result = check_profile(context.unresolved_config, context.profile_name)
    except Exception as exc:
        return _unknown(f"profile-check-error:{type(exc).__name__}")
    profile = _profile_mapping(profile_result)
    if profile_result.verdict == "refused":
        assert profile_result.refusal is not None
        return _mapping(
            "successor",
            "refused",
            "invalid",
            profile,
            profile_result.refusal["code"],
        )

    try:
        result = select_successor_admission(_intent_for(context, request), request)
    except Exception as exc:
        return _unknown(f"admission-error:{type(exc).__name__}")
    if result.accepted:
        return _mapping("successor", "admitted", "valid", profile, "admitted")
    assert result.refusal is not None
    return _mapping("successor", "refused", "invalid", profile, result.refusal.code.value)


def encode_core_admission(mapping: Mapping[str, Any]) -> bytes:
    """Encode one v1 mapping as canonical UTF-8 JSON bytes."""

    if set(mapping) != _ADMISSION_FIELDS or mapping.get("schema") != CORE_ADMISSION_SCHEMA:
        raise ValueError("core admission mapping has an invalid closed shape")
    return json.dumps(dict(mapping), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def core_admission_digest(mapping: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of canonical admission bytes."""

    return hashlib.sha256(encode_core_admission(mapping)).hexdigest()


__all__ = [
    "CORE_ADMISSION_SCHEMA",
    "CoreAdmissionContextV1",
    "core_admission_digest",
    "encode_core_admission",
    "evaluate_core_admission",
]
