"""Contract tests for preregistered selection declarations."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from datetime import UTC, datetime, timedelta
import pytest

def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    if name in sys.modules:
        return sys.modules[name]
    target = Path(__file__).with_name(filename).resolve()
    for candidate in tuple(sys.modules.values()):
        candidate_path = getattr(candidate, "__file__", None)
        if candidate_path is not None and Path(candidate_path).resolve() == target:
            sys.modules.update({name: candidate})
            return candidate
    module = importlib.util.module_from_spec(spec)
    sys.modules.update({name: module})
    spec.loader.exec_module(module)
    return module

selection = _load("he_importance_outcome_selection", "outcome_selection.py")
traversal = _load("he_importance_content_traversal_for_selection", "content_traversal.py")


def _completed_packet(tmp_path: Path, index: int = 0):
    packet = importlib.import_module("experiments.atomistic.he-importance.inference_packet")
    created = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
    parent = (tmp_path / f"cell-{index}" / ("a" * 64)).resolve()
    checkpoint = parent / "checkpoints" / "update-0001"
    checkpoint_ref = packet.CheckpointReference(parent, checkpoint, "a" * 64, {"world_size": 1})
    status = packet.ChainStatus(
        state=packet.ChainState.COMPLETED,
        created_at=created,
        started_at=created + timedelta(seconds=1),
        finished_at=created + timedelta(seconds=3),
        last_activity_at=created + timedelta(seconds=3),
    )
    chain = packet.IndependentChain(
        chain_id=f"chain-{index}",
        seeds=packet.ChainSeedProvenance(100 + index, 200 + index, 300 + index, 400 + index),
        interval=packet.SamplingInterval(10_000, 10, 32_768),
        status=status,
    )
    return packet.InferencePacket(
        checkpoint_ref,
        {"walkers": 4096, "burn_in_sweeps": 10, "sampler": {"walkers": 4096}},
        (chain,),
        None,
    )

def test_criteria_digest_keeps_criteria_only_scope() -> None:
    interval = selection.IntervalContract("mean", "two-sided-t", 0.95, "holm")
    first = selection.SelectionCommitment.create(("cell-a",), {"eligibility": "x"}, interval)
    second = selection.SelectionCommitment.create(("cell-b",), {"eligibility": "x"}, interval)
    assert first.criteria_digest == second.criteria_digest
    assert first.preregistration_digest != second.preregistration_digest


@pytest.mark.parametrize("schema", ["he-importance/train/v1", "he-importance/evaluation/v1"])
def test_topology_rank_is_part_of_complete_criteria_digest(schema: str, tmp_path: Path) -> None:
    interval = selection.IntervalContract("mean", "two-sided-t", 0.95, "holm")
    first_criteria = {"schema": schema, "topology": {"ranks": 8}, "eligibility": "x"}
    second_criteria = {"schema": schema, "topology": {"ranks": 16}, "eligibility": "x"}
    assert set(first_criteria) == set(second_criteria)
    assert first_criteria["schema"] == second_criteria["schema"]
    assert first_criteria["eligibility"] == second_criteria["eligibility"]
    assert first_criteria["topology"] != second_criteria["topology"]
    first = selection.SelectionCommitment.create(("cell-a",), first_criteria, interval)
    second = selection.SelectionCommitment.create(("cell-a",), second_criteria, interval)
    first.verify()
    second.verify()
    first_bytes = b'{"eligibility":"x","schema":"' + schema.encode("ascii") + b'","topology":{"ranks":8}}'
    second_bytes = b'{"eligibility":"x","schema":"' + schema.encode("ascii") + b'","topology":{"ranks":16}}'
    assert first.criteria_digest == hashlib.sha256(first_bytes).hexdigest()
    assert second.criteria_digest == hashlib.sha256(second_bytes).hexdigest()
    assert first.criteria_digest != second.criteria_digest
    first.verify()
    second.verify()
    selection.OutcomeLedger.attach(first, (selection.CellOutcome("cell-a", _completed_packet(tmp_path, 0), 1.0),))
    selection.OutcomeLedger.attach(second, (selection.CellOutcome("cell-a", _completed_packet(tmp_path, 1), 1.0),))


def test_equal_content_verifies_both_digests() -> None:
    interval = selection.IntervalContract("mean", "two-sided-t", 0.95, "holm")
    commitment = selection.SelectionCommitment.create(("cell-a",), {"a": [1, 2], "b": "x"}, interval)
    object.__setattr__(commitment, "criteria", {"b": "x", "a": (1, 2)})
    commitment.verify()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("estimator", "median"), ("interval_form", "percentile-bootstrap"), ("coverage", 0.9), ("multiplicity_method", "bonferroni")],
)
def test_interval_tamper_after_attach_is_preregistration_mismatch(field: str, replacement: object, tmp_path: Path) -> None:
    commitment = selection.SelectionCommitment.create(("cell-a",), {"eligibility": "x"}, selection.IntervalContract("mean", "two-sided-t", 0.95, "holm"))
    ledger = selection.OutcomeLedger.attach(commitment, (selection.CellOutcome("cell-a", _completed_packet(tmp_path), 1.0),))
    original = commitment.interval
    values = {name: getattr(original, name) for name in ("estimator", "interval_form", "coverage", "multiplicity_method")}
    values[field] = replacement
    object.__setattr__(commitment, "interval", selection.IntervalContract(**values))
    with pytest.raises(selection.OutcomeSelectionError) as caught:
        ledger.select_complete()
    assert caught.value.refusal is selection.OutcomeRefusal.PREREGISTRATION_DIGEST_MISMATCH


def test_criteria_readdressed_envelope_stale_is_detected() -> None:
    commitment = selection.SelectionCommitment.create(("cell-a",), {"eligibility": "x"}, selection.IntervalContract("mean", "two-sided-t", 0.95, "holm"))
    object.__setattr__(commitment, "criteria", {"eligibility": "y"})
    object.__setattr__(commitment, "criteria_digest", hashlib.sha256(b'{"eligibility":"y"}').hexdigest())
    with pytest.raises(selection.OutcomeSelectionError) as caught:
        commitment.verify()
    assert caught.value.refusal is selection.OutcomeRefusal.PREREGISTRATION_DIGEST_MISMATCH


def test_criteria_address_only_corrupted_reports_only_criteria_member() -> None:
    commitment = selection.SelectionCommitment.create(("cell-a",), {"eligibility": "x"}, selection.IntervalContract("mean", "two-sided-t", 0.95, "holm"))
    object.__setattr__(commitment, "criteria_digest", "0" * 64)
    with pytest.raises(selection.OutcomeSelectionError) as caught:
        commitment.verify()
    assert caught.value.refusal is selection.OutcomeRefusal.CRITERIA_DIGEST_MISMATCH


def test_preregistration_address_only_corrupted_reports_only_envelope_member() -> None:
    commitment = selection.SelectionCommitment.create(("cell-a",), {"eligibility": "x"}, selection.IntervalContract("mean", "two-sided-t", 0.95, "holm"))
    object.__setattr__(commitment, "preregistration_digest", "0" * 64)
    with pytest.raises(selection.OutcomeSelectionError) as caught:
        commitment.verify()
    assert caught.value.refusal is selection.OutcomeRefusal.PREREGISTRATION_DIGEST_MISMATCH

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


def test_equal_maps_have_identical_bytes_across_insertion_orders() -> None:
    left = traversal.project_content(traversal.freeze_content({"z": 0, "a": 1}))
    right = traversal.project_content(traversal.freeze_content({"a": 1, "z": 0}))
    assert selection._canonical_bytes(left) == selection._canonical_bytes(right) == b'{"a":1,"z":0}'


def test_separator_oracle_is_compact_and_sorted() -> None:
    value = {"b": 2, "a": 1}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    assert encoded == b'{"a":1,"b":2}'


def test_canonical_encoder_failure_has_its_own_refusal() -> None:
    value = 10 ** 4300
    frozen = traversal.freeze_content({"k": value})
    assert traversal.project_content(frozen)["k"] == value
    previous = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(4300)
        with pytest.raises(selection.OutcomeSelectionError) as caught:
            selection._canonical_bytes({"k": value})
    finally:
        sys.set_int_max_str_digits(previous)
    assert caught.value.refusal is selection.OutcomeRefusal.CANONICAL_ENCODING_FAILED
    assert isinstance(caught.value.__cause__, ValueError)

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
