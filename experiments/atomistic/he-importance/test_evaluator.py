"""Contract tests for the mechanics-only HI ranking evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "he_importance_evaluator", Path(__file__).with_name("evaluator.py")
)
assert _SPEC is not None and _SPEC.loader is not None
evaluator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = evaluator
_SPEC.loader.exec_module(evaluator)


def _rows(*values: float) -> tuple[object, ...]:
    return tuple(
        evaluator.LocalEnergyRow("checkpoint-a", f"chain-{index}", {"rank": index}, value)
        for index, value in enumerate(values)
    )


@pytest.mark.parametrize("oracle_name", ["analytic", "naive", "slow"])
def test_factor_qualification_accepts_each_required_oracle(oracle_name: str) -> None:
    oracle = lambda value: value * 2
    result = evaluator.qualify_factor("cusp", 3, oracle, **{f"{oracle_name}_oracle": oracle})
    assert result.value == 6
    assert result.oracle_values == {oracle_name: 6}


def test_factor_qualification_witnesses_disagreeing_oracle() -> None:
    with pytest.raises(
        evaluator.EvaluatorQualificationError,
        match="factor 'cusp' disagrees with slow oracle: 6 != 7",
    ):
        evaluator.qualify_factor("cusp", 3, lambda value: value * 2, slow_oracle=lambda value: 7)


@pytest.mark.parametrize("route", ["input", "calculator output", "analytic oracle output"])
@pytest.mark.parametrize("forbidden_key", ["reference_error", "e_ref"])
def test_blinding_refuses_reference_field_on_every_evaluator_route(
    route: str, forbidden_key: str
) -> None:
    value = {"coordinate": 2}
    calculator = lambda item: item
    oracle = lambda item: item
    if route == "input":
        value = {forbidden_key: 0.01}
    elif route == "calculator output":
        calculator = lambda item: {forbidden_key: 0.01}
    else:
        oracle = lambda item: {forbidden_key: 0.01}
    with pytest.raises(evaluator.EvaluatorQualificationError, match=f"forbidden blinding field '{forbidden_key}'"):
        evaluator.qualify_factor("cusp", value, calculator, analytic_oracle=oracle)


def test_nonfinite_row_is_visible_with_its_chain_and_topology_witness() -> None:
    rows = _rows(1.0, float("nan"))
    state = evaluator.evaluate_local_energy_rows(
        "checkpoint-a", rows, expected_chain_ids=("chain-0", "chain-1")
    )
    assert state.status is evaluator.InferenceStatus.NONFINITE_LOCAL_ENERGY
    assert state.rows == rows
    assert state.rows[1].chain_id == "chain-1"
    assert state.rows[1].topology == {"rank": 1}


@pytest.mark.parametrize(
    ("rows", "expected", "message"),
    [
        (_rows(1.0), ("chain-0", "chain-1"), "each expected chain exactly once"),
        (_rows(1.0, 2.0), ("chain-0",), "each expected chain exactly once"),
    ],
)
def test_local_energy_chain_completeness_has_a_discriminating_witness(
    rows: tuple[object, ...], expected: tuple[str, ...], message: str
) -> None:
    with pytest.raises(evaluator.IncompleteRankArtifactError, match=message):
        evaluator.evaluate_local_energy_rows("checkpoint-a", rows, expected_chain_ids=expected)


def test_rank_artifact_rejects_nonfinite_state_instead_of_silently_dropping_it() -> None:
    ready = evaluator.evaluate_local_energy_rows("checkpoint-a", _rows(1.0), expected_chain_ids=("chain-0",))
    bad_rows = tuple(
        evaluator.LocalEnergyRow("checkpoint-b", row.chain_id, row.topology, float("inf"))
        for row in _rows(1.0)
    )
    nonfinite = evaluator.evaluate_local_energy_rows(
        "checkpoint-b", bad_rows, expected_chain_ids=("chain-0",)
    )
    with pytest.raises(
        evaluator.IncompleteRankArtifactError,
        match=r"rank artifact contains nonfinite local-energy state: \['checkpoint-b'\]",
    ):
        evaluator.require_complete_rank_artifact(
            (ready, nonfinite), expected_checkpoint_ids=("checkpoint-a", "checkpoint-b")
        )


@pytest.mark.parametrize(
    ("states", "expected"),
    [([], ("checkpoint-a",)),],
)
def test_rank_artifact_rejects_partial_or_duplicate_checkpoint_sets(
    states: list[object], expected: tuple[str, ...]
) -> None:
    with pytest.raises(evaluator.IncompleteRankArtifactError, match="each expected checkpoint exactly once"):
        evaluator.require_complete_rank_artifact(states, expected_checkpoint_ids=expected)
