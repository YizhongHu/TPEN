"""Reference-free qualification for the HI cheap ranking evaluator.

Blinding is a property of this boundary: reference values are refused from
every input route and are never emitted.  This module only qualifies mechanics
and reports whether a checkpoint is usable for later ranking; it never ranks
or makes an accuracy decision.

The non-finite policy is declared independently at each evaluator seam. L3a
retains rows with provenance and rejects the resulting artifact, while the
adjacent L3b ranking evaluator declares ``defer``. This declaration is
deliberate; L3a does not inherit trainer ruling 377f6b0f's raise-always policy.
Artifact checkpoint and chain identifiers are screened at the rank-artifact
guard before duplicate, completeness, or non-finite checks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any


_WALKER_NAME = "he_importance_blinding_traversal"
_WALKER_PATH = Path(__file__).with_name("blinding_traversal.py")
_WALKER_SPEC = importlib.util.spec_from_file_location(_WALKER_NAME, _WALKER_PATH)
assert _WALKER_SPEC is not None and _WALKER_SPEC.loader is not None
_WALKER = sys.modules.get(_WALKER_NAME)
if _WALKER is None:
    _WALKER = importlib.util.module_from_spec(_WALKER_SPEC)
    sys.modules[_WALKER_NAME] = _WALKER
    _WALKER_SPEC.loader.exec_module(_WALKER)


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

    _refuse_blinding_content(factor_id, "factor id")
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

    Raises
    ------
    EvaluatorQualificationError
        If the checkpoint, expected chain IDs, row chain IDs, or topology
        provenance contains blinding-bearing content; if
        ``expected_chain_ids`` contains duplicated IDs; if a row has
        mismatched checkpoint provenance; or if topology provenance is empty.
    IncompleteRankArtifactError
        If the rows contain missing, extra, or duplicated actual chain IDs.
    """

    if not rows:
        raise EvaluatorQualificationError("local-energy evaluation requires at least one row")
    _refuse_blinding_content(checkpoint_id, "local-energy checkpoint id")
    required = tuple(expected_chain_ids)
    for index, chain_id in enumerate(required):
        _refuse_blinding_content(chain_id, f"local-energy expected chain id[{index}]")
    actual = tuple(row.chain_id for row in rows)
    for index, chain_id in enumerate(actual):
        _refuse_blinding_content(chain_id, f"local-energy row[{index}] chain id")
    duplicate_expected = tuple(
        dict.fromkeys(chain_id for chain_id in required if required.count(chain_id) > 1)
    )
    if duplicate_expected:
        raise EvaluatorQualificationError(
            f"expected chain ids must be unique; duplicated ids: {duplicate_expected!r}"
        )
    if len(actual) != len(set(actual)) or set(actual) != set(required):
        raise IncompleteRankArtifactError("local-energy rows must contain each expected chain exactly once")
    if any(row.checkpoint_id != checkpoint_id for row in rows):
        raise EvaluatorQualificationError("local-energy row checkpoint provenance does not match evaluation")
    for index, row in enumerate(rows):
        _refuse_blinding_content(row.topology, f"local-energy row[{index}] topology")
    if any(not row.topology for row in rows):
        raise EvaluatorQualificationError("local-energy rows require nonempty topology provenance")
    status = (
        InferenceStatus.NONFINITE_LOCAL_ENERGY
        if any(not math.isfinite(row.local_energy) for row in rows)
        else InferenceStatus.READY
    )
    return InferenceState(checkpoint_id, status, tuple(rows))


def require_complete_rank_artifact(states: Sequence[InferenceState], *, expected_checkpoint_ids: Sequence[str]) -> None:
    """Refuse blinded, partial, or nonfinite checkpoint artifacts before ranking.

    Checkpoint and row identifiers are screened shallowly at this boundary;
    payload traversal remains owned by the producer seams.

    Raises
    ------
    EvaluatorQualificationError
        If an expected, state, or row checkpoint/chain identifier contains
        blinding-bearing content, or if ``expected_checkpoint_ids`` contains
        duplicated IDs.
    IncompleteRankArtifactError
        If the artifact contains missing, extra, or duplicated actual
        checkpoint IDs, or contains a nonfinite local-energy state.
    """

    expected = tuple(expected_checkpoint_ids)
    for index, checkpoint_id in enumerate(expected):
        _refuse_blinding_content(checkpoint_id, f"rank-artifact expected checkpoint id[{index}]")
    for state_index, state in enumerate(states):
        _refuse_blinding_content(state.checkpoint_id, f"rank-artifact state[{state_index}] checkpoint id")
        for row_index, row in enumerate(state.rows):
            _refuse_blinding_content(
                row.checkpoint_id,
                f"rank-artifact state[{state_index}] row[{row_index}] checkpoint id",
            )
            _refuse_blinding_content(
                row.chain_id,
                f"rank-artifact state[{state_index}] row[{row_index}] chain id",
            )
    actual = tuple(state.checkpoint_id for state in states)
    duplicate_expected = tuple(
        dict.fromkeys(checkpoint_id for checkpoint_id in expected if expected.count(checkpoint_id) > 1)
    )
    if duplicate_expected:
        raise EvaluatorQualificationError(
            f"expected checkpoint ids must be unique; duplicated ids: {duplicate_expected!r}"
        )
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise IncompleteRankArtifactError("rank artifact must contain each expected checkpoint exactly once")
    nonfinite = [state.checkpoint_id for state in states if state.status is not InferenceStatus.READY]
    if nonfinite:
        raise IncompleteRankArtifactError(f"rank artifact contains nonfinite local-energy state: {nonfinite!r}")


def _refuse_blinding_content(value: Any, label: str) -> None:
    """Reject reference-bearing fields recursively before they reach the evaluator."""

    def is_forbidden(text: str) -> bool:
        normalized = text.lower().replace("-", "_")
        return normalized in _FORBIDDEN_BLINDING_TOKENS or "reference" in normalized

    def error_factory(path: str, content: Any, reason: str) -> Exception:
        if reason == "key-type":
            return EvaluatorQualificationError(f"{label} keys must be strings")
        if reason == "key":
            parent = path.rsplit(".", 1)[0]
            return EvaluatorQualificationError(
                f"{parent} contains forbidden blinding field {content!r}"
            )
        if reason == "decode":
            return EvaluatorQualificationError(f"{path} contains undecodable blinding content")
        if reason == "cycle":
            return EvaluatorQualificationError(f"{path} contains a cyclic blinding value")
        return EvaluatorQualificationError(f"{path} contains forbidden blinding value {content!r}")

    _WALKER.walk_blinding_content(
        value,
        label,
        is_forbidden=is_forbidden,
        error_factory=error_factory,
    )
