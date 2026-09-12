"""Pre-registered complete-outcome selection and interval reporting contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import importlib
import importlib.util
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from content_traversal import CRITERIA_DECLARATION, ContentRefusal, ContentTraversalError, freeze_content, project_content
except ModuleNotFoundError:  # Direct experiment-file execution has no package initializer.
    spec = importlib.util.spec_from_file_location("he_importance_content_traversal", Path(__file__).with_name("content_traversal.py"))
    if spec is None or spec.loader is None:
        raise
    content_traversal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(content_traversal)
    CRITERIA_DECLARATION = content_traversal.CRITERIA_DECLARATION
    ContentRefusal = content_traversal.ContentRefusal
    ContentTraversalError = content_traversal.ContentTraversalError
    freeze_content = content_traversal.freeze_content
    project_content = content_traversal.project_content


class OutcomeRefusal(str, Enum):
    """Stable identities for selection-boundary refusals."""
    CRITERIA_DIGEST_MISMATCH = "criteria_digest_mismatch"
    PREREGISTRATION_DIGEST_MISMATCH = "preregistration_digest_mismatch"
    CANONICAL_ENCODING_FAILED = "canonical_encoding_failed"
    CANDIDATE_ID_NOT_STRING = "candidate_id_not_string"
    CANDIDATE_ID_EMPTY = "candidate_id_empty"
    CANDIDATE_IDS_NOT_UNIQUE = "candidate_ids_not_unique"
    INTERVAL_NOT_DECLARED = "interval_not_declared"
    OUTCOME_PACKET_NOT_PACKET = "outcome_packet_not_packet"
    OUTCOME_CANDIDATE_ID_NOT_STRING = "outcome_candidate_id_not_string"
    OUTCOME_CANDIDATE_ID_EMPTY = "outcome_candidate_id_empty"
    OUTCOME_NOT_CHAIN_COMPLETE_RECORD = "outcome_not_chain_complete_record"
    OUTCOMES_NOT_COMPLETE_SET = "outcomes_not_complete_set"


class OutcomeSelectionError(ValueError):
    """A preregistration, attachment, or reporting boundary failed."""
    def __init__(self, message: str, *, refusal: OutcomeRefusal | ContentRefusal | None = None, path: tuple[Any, ...] = ()):
        super().__init__(message)
        self.refusal = refusal
        self.path = path


class Reportability(str, Enum):
    """Whether a cell may be represented as complete in a report."""
    REPORTABLE_AS_COMPLETE = "reportable-as-complete"
    UNREPORTABLE_AS_COMPLETE = "unreportable-as-complete"


def _packet_types() -> tuple[type[Any], type[Enum]]:
    module = importlib.import_module("experiments.atomistic.he-importance.inference_packet")
    return module.InferencePacket, module.ChainState


def _translate_content_error(error: ContentTraversalError) -> OutcomeSelectionError:
    return OutcomeSelectionError(str(error), refusal=error.refusal, path=error.path)


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)
        return encoded.encode("ascii")
    except (TypeError, ValueError) as error:
        raise OutcomeSelectionError("canonical JSON encoding failed", refusal=OutcomeRefusal.CANONICAL_ENCODING_FAILED) from error


def _canonical_digest(value: Any) -> str:
    """Return SHA-256 of canonical criteria bytes; criteria scope is unchanged."""
    return sha256(_canonical_bytes(value)).hexdigest()


def _validate_identifiers(candidate_ids: Sequence[str]) -> tuple[str, ...]:
    try:
        identifiers = tuple(candidate_ids)
    except TypeError as error:
        raise OutcomeSelectionError("committed candidate IDs must be a sequence", refusal=OutcomeRefusal.CANDIDATE_ID_NOT_STRING) from error
    for identifier in identifiers:
        if type(identifier) is not str:
            raise OutcomeSelectionError("committed candidate IDs must be exact strings", refusal=OutcomeRefusal.CANDIDATE_ID_NOT_STRING)
        if not identifier:
            raise OutcomeSelectionError("committed candidate IDs must be non-empty", refusal=OutcomeRefusal.CANDIDATE_ID_EMPTY)
    if not identifiers:
        raise OutcomeSelectionError("committed candidate IDs must be non-empty", refusal=OutcomeRefusal.CANDIDATE_ID_EMPTY)
    if len(set(identifiers)) != len(identifiers):
        raise OutcomeSelectionError("committed candidate IDs must be unique", refusal=OutcomeRefusal.CANDIDATE_IDS_NOT_UNIQUE)
    return identifiers


@dataclass(frozen=True)
class IntervalContract:
    """The estimator and interval property declared before outcomes attach."""
    estimator: str
    interval_form: str
    coverage: float
    multiplicity_method: str

    def __post_init__(self) -> None:
        for value in (self.estimator, self.interval_form, self.multiplicity_method):
            if type(value) is not str or not value.strip():
                raise OutcomeSelectionError("interval contract names must be non-empty", refusal=OutcomeRefusal.INTERVAL_NOT_DECLARED)
        if type(self.coverage) not in (int, float) or not isfinite(self.coverage) or not 0.0 < self.coverage < 1.0:
            raise OutcomeSelectionError("interval coverage must lie strictly between zero and one", refusal=OutcomeRefusal.INTERVAL_NOT_DECLARED)


def _envelope(candidate_ids: tuple[str, ...], criteria: Mapping[str, Any], interval: IntervalContract) -> dict[str, Any]:
    return {"schema": "he-importance/preregistration/v1", "candidate_ids": candidate_ids, "criteria": criteria, "interval": {"estimator": interval.estimator, "interval_form": interval.interval_form, "coverage": interval.coverage, "multiplicity_method": interval.multiplicity_method}}


def _preregistration_digest(candidate_ids: tuple[str, ...], criteria: Mapping[str, Any], interval: IntervalContract) -> str:
    try:
        projected = project_content(_envelope(candidate_ids, criteria, interval), CRITERIA_DECLARATION)
    except ContentTraversalError as error:
        raise _translate_content_error(error) from error
    return sha256(_canonical_bytes(projected)).hexdigest()


@dataclass(frozen=True)
class SelectionCommitment:
    """Tamper-evident selection and interval declaration made before outcomes."""
    candidate_ids: tuple[str, ...]
    criteria: Mapping[str, Any]
    interval: IntervalContract
    criteria_digest: str
    preregistration_digest: str

    def __post_init__(self) -> None:
        identifiers = _validate_identifiers(self.candidate_ids)
        if type(self.interval) is not IntervalContract:
            raise OutcomeSelectionError("interval must be a declared interval contract", refusal=OutcomeRefusal.INTERVAL_NOT_DECLARED)
        try:
            frozen = freeze_content(self.criteria, CRITERIA_DECLARATION)
        except ContentTraversalError as error:
            raise _translate_content_error(error) from error
        object.__setattr__(self, "candidate_ids", identifiers)
        object.__setattr__(self, "criteria", frozen)
        expected_criteria = _canonical_digest(project_content(frozen, CRITERIA_DECLARATION))
        expected_preregistration = _preregistration_digest(identifiers, frozen, self.interval)
        if self.criteria_digest != expected_criteria:
            raise OutcomeSelectionError("selection criteria digest does not match commitment", refusal=OutcomeRefusal.CRITERIA_DIGEST_MISMATCH)
        if self.preregistration_digest != expected_preregistration:
            raise OutcomeSelectionError("preregistration digest does not match commitment", refusal=OutcomeRefusal.PREREGISTRATION_DIGEST_MISMATCH)

    @classmethod
    def create(cls, candidate_ids: Sequence[str], criteria: Mapping[str, Any], interval: IntervalContract) -> "SelectionCommitment":
        """Create a detached, content-addressed declaration before attachment."""
        identifiers = _validate_identifiers(candidate_ids)
        if type(interval) is not IntervalContract:
            raise OutcomeSelectionError("interval must be a declared interval contract", refusal=OutcomeRefusal.INTERVAL_NOT_DECLARED)
        try:
            frozen = freeze_content(criteria, CRITERIA_DECLARATION)
            projected = project_content(frozen, CRITERIA_DECLARATION)
        except ContentTraversalError as error:
            raise _translate_content_error(error) from error
        return cls(identifiers, frozen, interval, _canonical_digest(projected), _preregistration_digest(identifiers, frozen, interval))

    def verify(self) -> None:
        """Check criteria first, then the complete current preregistration envelope."""
        try:
            frozen = freeze_content(self.criteria, CRITERIA_DECLARATION)
            current_criteria = project_content(frozen, CRITERIA_DECLARATION)
        except ContentTraversalError as error:
            raise _translate_content_error(error) from error
        if _canonical_digest(current_criteria) != self.criteria_digest:
            raise OutcomeSelectionError("selection criteria digest does not match commitment", refusal=OutcomeRefusal.CRITERIA_DIGEST_MISMATCH)
        identifiers = _validate_identifiers(self.candidate_ids)
        if type(self.interval) is not IntervalContract:
            raise OutcomeSelectionError("interval must be a declared interval contract", refusal=OutcomeRefusal.INTERVAL_NOT_DECLARED)
        if _preregistration_digest(identifiers, frozen, self.interval) != self.preregistration_digest:
            raise OutcomeSelectionError("preregistration digest does not match commitment", refusal=OutcomeRefusal.PREREGISTRATION_DIGEST_MISMATCH)


@dataclass(frozen=True)
class CellOutcome:
    """One retained cell record derived from one exact L4a inference packet."""
    candidate_id: str
    packet: Any
    value: float | None = None

    def __post_init__(self) -> None:
        packet_type, _ = _packet_types()
        if type(self.candidate_id) is not str:
            raise OutcomeSelectionError("outcome candidate ID must be an exact string", refusal=OutcomeRefusal.OUTCOME_CANDIDATE_ID_NOT_STRING)
        if not self.candidate_id:
            raise OutcomeSelectionError("outcome candidate ID must be non-empty", refusal=OutcomeRefusal.OUTCOME_CANDIDATE_ID_EMPTY)
        if type(self.packet) is not packet_type:
            raise OutcomeSelectionError("outcome must carry an exact L4a inference packet", refusal=OutcomeRefusal.OUTCOME_PACKET_NOT_PACKET)
        object.__setattr__(self, "_chain_states", tuple(chain.status.state for chain in self.packet.chains))

    @property
    def chain_states(self) -> tuple[Enum, ...]:
        return self._chain_states

    @property
    def independent_sampler_inputs(self) -> Mapping[str, Any]:
        return self.packet.independent_sampler_inputs

    @property
    def reportability(self) -> Reportability:
        _, chain_state = _packet_types()
        return Reportability.REPORTABLE_AS_COMPLETE if all(state is chain_state.COMPLETED for state in self.chain_states) else Reportability.UNREPORTABLE_AS_COMPLETE

    @property
    def unreportable_witness(self) -> tuple[Enum, ...]:
        _, chain_state = _packet_types()
        return tuple(state for state in self.chain_states if state is not chain_state.COMPLETED)


@dataclass(frozen=True)
class OutcomeLedger:
    """Committed complete outcome set with all outcomes retained, never dropped."""
    commitment: SelectionCommitment
    outcomes: tuple[CellOutcome, ...]

    @classmethod
    def attach(cls, commitment: SelectionCommitment, outcomes: Sequence[CellOutcome]) -> "OutcomeLedger":
        commitment.verify()
        records = tuple(outcomes)
        if not all(type(record) is CellOutcome for record in records):
            raise OutcomeSelectionError("outcomes must contain exact cell records", refusal=OutcomeRefusal.OUTCOME_NOT_CHAIN_COMPLETE_RECORD)
        by_id = {record.candidate_id: record for record in records}
        if len(by_id) != len(records) or set(by_id) != set(commitment.candidate_ids):
            raise OutcomeSelectionError("outcomes must attach exactly the committed candidate set", refusal=OutcomeRefusal.OUTCOMES_NOT_COMPLETE_SET)
        return cls(commitment, records)

    def select_complete(self) -> tuple[CellOutcome, ...]:
        self.commitment.verify()
        return tuple(outcome for outcome in self.outcomes if outcome.candidate_id in self.commitment.candidate_ids and outcome.reportability is Reportability.REPORTABLE_AS_COMPLETE)

    def unreportable(self) -> tuple[CellOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.reportability is Reportability.UNREPORTABLE_AS_COMPLETE)
