"""Pre-registered complete-outcome selection and interval reporting contracts.

This module deliberately does not rank numerical outcomes.  A selection is
defined only by a commitment's candidate identifiers and each candidate's
recorded chain completion state.  Numeric values may attach after commitment
for later estimation, but cannot change complete-outcome eligibility.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any


class OutcomeSelectionError(ValueError):
    """A preregistration, outcome attachment, or reporting boundary failed."""


class Reportability(str, Enum):
    """Whether a cell may be represented as complete in a report."""

    REPORTABLE_AS_COMPLETE = "reportable-as-complete"
    UNREPORTABLE_AS_COMPLETE = "unreportable-as-complete"


def _freeze(value: Any) -> Any:
    """Freeze JSON-shaped data without changing its scientific contents."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(nested) for nested in value)
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    """Return a content address for an ex-ante JSON-shaped declaration."""

    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise OutcomeSelectionError("commitment criteria must be canonical JSON data") from error
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IntervalContract:
    """The estimator and interval property declared before outcomes attach.

    Parameters
    ----------
    estimator
        Name of the estimator that later reporting must use.
    interval_form
        Name of the interval construction.
    coverage
        Declared coverage in the open unit interval.
    multiplicity_method
        Declared multiplicity adjustment; it cannot be selected after results.
    """

    estimator: str
    interval_form: str
    coverage: float
    multiplicity_method: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            self.estimator, self.interval_form, self.multiplicity_method,
        )):
            raise OutcomeSelectionError("interval contract names must be non-empty")
        if type(self.coverage) not in (int, float) or not 0.0 < self.coverage < 1.0:
            raise OutcomeSelectionError("interval coverage must lie strictly between zero and one")


@dataclass(frozen=True)
class SelectionCommitment:
    """Tamper-evident selection and interval declaration made before outcomes."""

    candidate_ids: tuple[str, ...]
    criteria: Mapping[str, Any]
    interval: IntervalContract
    criteria_digest: str

    @classmethod
    def create(
        cls,
        candidate_ids: Sequence[str],
        criteria: Mapping[str, Any],
        interval: IntervalContract,
    ) -> "SelectionCommitment":
        """Create an immutable, content-addressed declaration before attachment."""

        identifiers = tuple(candidate_ids)
        if not identifiers or any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise OutcomeSelectionError("committed candidate IDs must be non-empty strings")
        if len(set(identifiers)) != len(identifiers):
            raise OutcomeSelectionError("committed candidate IDs must be unique")
        if not isinstance(criteria, Mapping):
            raise OutcomeSelectionError("selection criteria must be a mapping")
        criteria_copy = dict(criteria)
        return cls(identifiers, _freeze(criteria_copy), interval, _canonical_digest(criteria_copy))

    def verify(self) -> None:
        """Detect a changed criteria payload before any outcome use."""

        if _canonical_digest(dict(self.criteria)) != self.criteria_digest:
            raise OutcomeSelectionError("selection criteria digest does not match commitment")


@dataclass(frozen=True)
class CellOutcome:
    """One retained cell record, including its reportability state and witness."""

    candidate_id: str
    chain_states: tuple[str, ...]
    independent_sampler_inputs: Mapping[str, Any]
    value: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise OutcomeSelectionError("outcome candidate ID must be non-empty")
        if not self.chain_states or any(not isinstance(state, str) for state in self.chain_states):
            raise OutcomeSelectionError("outcome must carry every recorded chain state")
        if not isinstance(self.independent_sampler_inputs, Mapping):
            raise OutcomeSelectionError("outcome must retain immutable independent sampler inputs")
        object.__setattr__(self, "chain_states", tuple(self.chain_states))
        object.__setattr__(self, "independent_sampler_inputs", _freeze(dict(self.independent_sampler_inputs)))

    @property
    def reportability(self) -> Reportability:
        """Return an explicit state; incomplete/dead cells are never absent."""

        return (
            Reportability.REPORTABLE_AS_COMPLETE
            if all(state == "completed" for state in self.chain_states)
            else Reportability.UNREPORTABLE_AS_COMPLETE
        )

    @property
    def unreportable_witness(self) -> tuple[str, ...]:
        """Name every non-completed chain state when completeness is impossible."""

        return tuple(state for state in self.chain_states if state != "completed")


@dataclass(frozen=True)
class OutcomeLedger:
    """Committed complete outcome set with all outcomes retained, never dropped."""

    commitment: SelectionCommitment
    outcomes: tuple[CellOutcome, ...]

    @classmethod
    def attach(cls, commitment: SelectionCommitment, outcomes: Sequence[CellOutcome]) -> "OutcomeLedger":
        """Attach the entire committed set only after verifying the declaration."""

        commitment.verify()
        records = tuple(outcomes)
        by_id = {record.candidate_id: record for record in records}
        if len(by_id) != len(records) or set(by_id) != set(commitment.candidate_ids):
            raise OutcomeSelectionError("outcomes must attach exactly the committed candidate set")
        return cls(commitment, records)

    def select_complete(self) -> tuple[CellOutcome, ...]:
        """Select solely by committed membership and completion status, never values."""

        self.commitment.verify()
        return tuple(
            outcome
            for outcome in self.outcomes
            if outcome.candidate_id in self.commitment.candidate_ids
            and outcome.reportability is Reportability.REPORTABLE_AS_COMPLETE
        )

    def unreportable(self) -> tuple[CellOutcome, ...]:
        """Retain, and explicitly expose, every cell that cannot be complete."""

        return tuple(
            outcome for outcome in self.outcomes
            if outcome.reportability is Reportability.UNREPORTABLE_AS_COMPLETE
        )
