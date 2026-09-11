"""Reference-free qualification for the HI cheap ranking evaluator.

Blinding is a property of this boundary: reference values are refused from
every input route and are never emitted.  This module only qualifies mechanics
and reports whether a checkpoint is usable for later ranking; it never ranks
or makes an accuracy decision.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class EvaluatorQualificationError(ValueError):
    """A mechanics-only evaluator input or oracle disagreement is invalid."""


class IncompleteRankArtifactError(EvaluatorQualificationError):
    """A rank artifact does not account for every required checkpoint."""


class InferenceStatus(Enum):
    """Declared evaluator state, independent of the trainer seam policy."""

    READY = "ready"
    NONFINITE_LOCAL_ENERGY = "nonfinite_local_energy"


@dataclass(frozen=True)
class LocalEnergyRow:
    """One local-energy observation with the provenance needed to witness it."""

    checkpoint_id: str
    chain_id: str
    topology: Mapping[str, Any]
    local_energy: float


@dataclass(frozen=True)
class InferenceState:
    """Visible state for one checkpoint; nonfinite rows are retained as witnesses."""

    checkpoint_id: str
    status: InferenceStatus
    rows: tuple[LocalEnergyRow, ...]


@dataclass(frozen=True)
class FactorQualification:
    """Agreement of one mechanics factor against independently named oracles."""

    factor_id: str
    value: Any
    oracle_values: Mapping[str, Any]


_FORBIDDEN_BLINDING_TOKENS = frozenset({"reference", "reference_energy", "e_ref", "accuracy"})


def qualify_factor(
    factor_id: str,
    value: Any,
    calculator: Callable[[Any], Any],
    *,
    analytic_oracle: Callable[[Any], Any] | None = None,
    naive_oracle: Callable[[Any], Any] | None = None,
    slow_oracle: Callable[[Any], Any] | None = None,
) -> FactorQualification:
    """Qualify a calculator result against all supplied mechanics-only oracles.

    At least one oracle is required.  A disagreement names the factor and the
    oracle that witnessed it, so callers cannot mistake a generic assertion for
    a qualification result.
    """

    _refuse_blinding_content(value, "factor input")
    supplied = {
        name: oracle
        for name, oracle in (
            ("analytic", analytic_oracle),
            ("naive", naive_oracle),
            ("slow", slow_oracle),
        )
        if oracle is not None
    }
    if not supplied:
        raise EvaluatorQualificationError("factor qualification requires an analytic, naive, or slow oracle")
    observed = calculator(value)
    _refuse_blinding_content(observed, "calculator output")
    oracle_values: dict[str, Any] = {}
    for name, oracle in supplied.items():
        expected = oracle(value)
        _refuse_blinding_content(expected, f"{name} oracle output")
        oracle_values[name] = expected
        if observed != expected:
            raise EvaluatorQualificationError(
                f"factor {factor_id!r} disagrees with {name} oracle: {observed!r} != {expected!r}"
            )
    return FactorQualification(factor_id, observed, oracle_values)


def evaluate_local_energy_rows(
    checkpoint_id: str, rows: Sequence[LocalEnergyRow], *, expected_chain_ids: Sequence[str]
) -> InferenceState:
    """Expose every nonfinite local-energy row with chain and topology provenance.

    The declared policy is ``NONFINITE_LOCAL_ENERGY``: retain every row and
    make the checkpoint unusable to a rank artifact.  It intentionally does
    not inherit trainer ruling 377f6b0f's raise-always policy, because this is
    an evaluator state boundary rather than the trainer seam.
    """

    if not rows:
        raise EvaluatorQualificationError("local-energy evaluation requires at least one row")
    actual = tuple(row.chain_id for row in rows)
    required = tuple(expected_chain_ids)
    if len(actual) != len(set(actual)) or set(actual) != set(required):
        raise IncompleteRankArtifactError("local-energy rows must contain each expected chain exactly once")
    if any(row.checkpoint_id != checkpoint_id for row in rows):
        raise EvaluatorQualificationError("local-energy row checkpoint provenance does not match evaluation")
    if any(not row.topology for row in rows):
        raise EvaluatorQualificationError("local-energy rows require nonempty topology provenance")
    status = (
        InferenceStatus.NONFINITE_LOCAL_ENERGY
        if any(not math.isfinite(row.local_energy) for row in rows)
        else InferenceStatus.READY
    )
    return InferenceState(checkpoint_id, status, tuple(rows))


def require_complete_rank_artifact(states: Sequence[InferenceState], *, expected_checkpoint_ids: Sequence[str]) -> None:
    """Refuse partial or nonfinite checkpoint artifacts before any ranking step."""

    actual = tuple(state.checkpoint_id for state in states)
    expected = tuple(expected_checkpoint_ids)
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise IncompleteRankArtifactError("rank artifact must contain each expected checkpoint exactly once")
    nonfinite = [state.checkpoint_id for state in states if state.status is not InferenceStatus.READY]
    if nonfinite:
        raise IncompleteRankArtifactError(f"rank artifact contains nonfinite local-energy state: {nonfinite!r}")


def _refuse_blinding_content(value: Any, label: str) -> None:
    """Reject reference-bearing fields recursively before they reach the evaluator."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise EvaluatorQualificationError(f"{label} keys must be strings")
            normalized = key.lower().replace("-", "_")
            if normalized in _FORBIDDEN_BLINDING_TOKENS or "reference" in normalized:
                raise EvaluatorQualificationError(f"{label} contains forbidden blinding field {key!r}")
            _refuse_blinding_content(nested, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _refuse_blinding_content(nested, f"{label}[{index}]")
