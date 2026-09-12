"""Contract tests for preregistered selection declarations."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import pytest

def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

selection = _load("he_importance_outcome_selection", "outcome_selection.py")
traversal = _load("he_importance_content_traversal_for_selection", "content_traversal.py")

def test_criteria_digest_keeps_criteria_only_scope() -> None:
    interval = selection.IntervalContract("mean", "two-sided-t", 0.95, "holm")
    first = selection.SelectionCommitment.create(("cell-a",), {"eligibility": "x"}, interval)
    second = selection.SelectionCommitment.create(("cell-b",), {"eligibility": "x"}, interval)
    assert first.criteria_digest == second.criteria_digest
    assert first.preregistration_digest != second.preregistration_digest

def test_literal_envelope_oracle_is_independent() -> None:
    interval = selection.IntervalContract("mean", "two-sided-t", 0.95, "holm")
    commitment = selection.SelectionCommitment.create(("cell-a", "cell-b"), {"eligibility": "x"}, interval)
    assert commitment.criteria_digest == hashlib.sha256(b'{"eligibility":"x"}').hexdigest()
    envelope = b'{"candidate_ids":["cell-a","cell-b"],"criteria":{"eligibility":"x"},"interval":{"coverage":0.95,"estimator":"mean","interval_form":"two-sided-t","multiplicity_method":"holm"},"schema":"he-importance/preregistration/v1"}'
    assert commitment.preregistration_digest == hashlib.sha256(envelope).hexdigest()
    commitment.verify()

def test_criteria_tamper_is_checked_before_envelope() -> None:
    commitment = selection.SelectionCommitment.create(("cell-a",), {"eligibility": "x"}, selection.IntervalContract("mean", "two-sided-t", 0.95, "holm"))
    object.__setattr__(commitment, "criteria", {"eligibility": "y"})
    with pytest.raises(selection.OutcomeSelectionError) as caught:
        commitment.verify()
    assert caught.value.refusal is selection.OutcomeRefusal.CRITERIA_DIGEST_MISMATCH

def test_candidate_tamper_uses_distinct_envelope_member() -> None:
    commitment = selection.SelectionCommitment.create(("cell-a", "cell-b"), {"eligibility": "x"}, selection.IntervalContract("mean", "two-sided-t", 0.95, "holm"))
    object.__setattr__(commitment, "candidate_ids", ("cell-a",))
    with pytest.raises(selection.OutcomeSelectionError) as caught:
        commitment.verify()
    assert caught.value.refusal is selection.OutcomeRefusal.PREREGISTRATION_DIGEST_MISMATCH

def test_canonical_key_order_is_raw_code_point_order() -> None:
    projected = traversal.project_content(traversal.freeze_content({"a": 1, chr(0x00E9): 2}))
    assert selection._canonical_bytes(projected) == b'{"a":1,"\\u00e9":2}'

def test_surrogate_pair_spelling_is_refused_while_astral_scalar_is_admitted() -> None:
    astral, pair = chr(0x1F600), chr(0xD83D) + chr(0xDE00)
    assert astral != pair and (len(astral), len(pair)) == (1, 2)
    traversal.freeze_content({"k": astral})
    with pytest.raises(traversal.ContentTraversalError) as caught:
        traversal.freeze_content({"k": pair})
    assert caught.value.refusal is traversal.ContentRefusal.STRING_HAS_SURROGATE_CODEPOINT

@pytest.mark.parametrize("value", [chr(0xD800), chr(0xDBFF), chr(0xDC00), chr(0xDFFF)])
def test_surrogate_boundaries_are_refused(value: str) -> None:
    with pytest.raises(traversal.ContentTraversalError) as caught:
        traversal.freeze_content({"k": value})
    assert caught.value.refusal is traversal.ContentRefusal.STRING_HAS_SURROGATE_CODEPOINT

@pytest.mark.parametrize("value", [chr(0xD7FF), chr(0xE000), chr(0x1F600), chr(0x10FFFF)])
def test_unicode_scalar_boundaries_remain_admitted(value: str) -> None:
    traversal.freeze_content({"k": value})

def test_unknown_leaf_is_refused_by_primitive() -> None:
    with pytest.raises(traversal.ContentTraversalError) as caught:
        traversal.freeze_content({"k": object()})
    assert caught.value.refusal is traversal.ContentRefusal.CONTENT_KIND_UNDECLARED
