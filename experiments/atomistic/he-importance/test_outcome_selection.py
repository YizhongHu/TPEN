"""Contract tests for preregistered complete-outcome selection."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "he_importance_outcome_selection", Path(__file__).with_name("outcome_selection.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_TRAVERSAL_SPEC = importlib.util.spec_from_file_location(
    "content_traversal", Path(__file__).with_name("content_traversal.py")
)
assert _TRAVERSAL_SPEC is not None and _TRAVERSAL_SPEC.loader is not None
traversal = importlib.util.module_from_spec(_TRAVERSAL_SPEC)
sys.modules[_TRAVERSAL_SPEC.name] = traversal
_TRAVERSAL_SPEC.loader.exec_module(traversal)
selection = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = selection
_SPEC.loader.exec_module(selection)


def _commitment() -> object:
    return selection.SelectionCommitment.create(
        ("cell-a", "cell-b"),
        {"eligibility": "all chains completed", "candidate_set": "committed"},
        selection.IntervalContract("mean", "two-sided-t", 0.95, "holm"),
    )


def _outcome(identifier: str, states: tuple[str, ...], value: float | None) -> object:
    return selection.CellOutcome(identifier, states, {"sampler": {"walkers": 4096}}, value)


@dataclass
class SelectionRow:
    reference_energy: object


def test_outcome_screen_reaches_reference_name_inside_its_own_record_type() -> None:
    with pytest.raises(selection.OutcomeSelectionError, match="reference-bearing route"):
        selection.CellOutcome("cell-a", ("completed",), {"sampler": SelectionRow(3.0)}, 1.0)


def test_commitment_screen_reaches_reference_name_inside_its_own_record_type() -> None:
    with pytest.raises(selection.OutcomeSelectionError, match="reference-bearing route"):
        selection.SelectionCommitment.create(
            ("cell-a",), {"criteria": SelectionRow(3.0)}, selection.IntervalContract("mean", "t", 0.95, "holm")
        )


def test_complete_selection_is_invariant_to_outcome_values_and_kills_peek_mutant() -> None:
    commitment = _commitment()
    first = selection.OutcomeLedger.attach(
        commitment, (_outcome("cell-a", ("completed",), -1.0), _outcome("cell-b", ("completed",), 100.0))
    )
    second = selection.OutcomeLedger.attach(
        commitment, (_outcome("cell-a", ("completed",), 100.0), _outcome("cell-b", ("completed",), -1.0))
    )
    assert tuple(record.candidate_id for record in first.select_complete()) == ("cell-a", "cell-b")
    assert tuple(record.candidate_id for record in second.select_complete()) == ("cell-a", "cell-b")


@pytest.mark.parametrize("state", ["idle", "running", "error", "cancelled", "dead"])
def test_every_noncompleted_chain_carries_unreportable_state_and_witness(state: str) -> None:
    ledger = selection.OutcomeLedger.attach(
        _commitment(),
        (_outcome("cell-a", ("completed", state), 3.0), _outcome("cell-b", ("completed",), 4.0)),
    )
    unreportable = ledger.unreportable()
    assert [(record.candidate_id, record.reportability, record.unreportable_witness) for record in unreportable] == [
        ("cell-a", selection.Reportability.UNREPORTABLE_AS_COMPLETE, (state,))
    ]
    assert [record.candidate_id for record in ledger.select_complete()] == ["cell-b"]


def test_outcomes_after_commitment_are_complete_set_not_silently_droppable() -> None:
    ledger = selection.OutcomeLedger.attach(
        _commitment(),
        (_outcome("cell-a", ("dead",), None), _outcome("cell-b", ("completed",), 4.0)),
    )
    assert len(ledger.outcomes) == 2
    assert ledger.unreportable()[0].candidate_id == "cell-a"
    with pytest.raises(selection.OutcomeSelectionError, match="exactly the committed candidate set"):
        selection.OutcomeLedger.attach(_commitment(), (_outcome("cell-b", ("completed",), 4.0),))


def test_tampered_criteria_is_detected_before_outcomes_attach() -> None:
    commitment = _commitment()
    object.__setattr__(commitment, "criteria", {"eligibility": "choose the lowest value"})
    with pytest.raises(selection.OutcomeSelectionError, match="digest"):
        selection.OutcomeLedger.attach(
            commitment, (_outcome("cell-a", ("completed",), 1.0), _outcome("cell-b", ("completed",), 2.0))
        )


@pytest.mark.parametrize(
    "estimator,interval_form,coverage,multiplicity",
    [("mean", "two-sided-t", 0.95, "holm"), ("median", "percentile-bootstrap", 0.9, "bonferroni")],
)
def test_interval_contract_requires_the_full_declared_property(
    estimator: str, interval_form: str, coverage: float, multiplicity: str
) -> None:
    contract = selection.IntervalContract(estimator, interval_form, coverage, multiplicity)
    assert (contract.estimator, contract.interval_form, contract.coverage, contract.multiplicity_method) == (
        estimator, interval_form, coverage, multiplicity
    )


@pytest.mark.parametrize("field", ["estimator", "interval_form", "multiplicity_method"])
def test_interval_contract_rejects_each_missing_declaration(field: str) -> None:
    fields = {"estimator": "mean", "interval_form": "two-sided-t", "coverage": 0.95, "multiplicity_method": "holm"}
    fields[field] = ""
    with pytest.raises(selection.OutcomeSelectionError, match="names"):
        selection.IntervalContract(**fields)
