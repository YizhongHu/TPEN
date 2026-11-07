"""Round-7 closure for the 2026-11-07 operator-review amendment of the H-v2
contract (TO note ``operator-review-contract-proposals-20261107`` and its
item-3/item-8 corrections on item ``fffc313a``).

The operator answered all eight pending ``lane_proposals`` on PR 587: items 1
and 2 (primary measure, margins) were delegated to the lane and resolved from
He-v1 fresh-evaluation evidence; items 3-7 were decided directly; item 8 (the
four-stream design) was answered after one round of clarification. This file
tests the amendment's structural consequences, not the frozen-snapshot
comparison (``test_contract_review_amend_r5.py`` covers that every leaf is
intentionally changed).
"""

import json
from pathlib import Path

INPUTS = Path(__file__).with_name("contract_inputs.json")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_exceedance_margins_are_per_threshold_and_directional_contradiction_is_recorded() -> None:
    """Item 2: 0.002 Ha at the tail thresholds, 0.01 Ha at the bulk thresholds."""
    inputs = load_inputs()
    parameters = inputs["decision_rules"]["parameters"]
    margins = parameters["exceedance_margins_hartree"]
    thresholds = set(inputs["measurement"]["fixed_exceedance_thresholds_hartree"])
    assert set(float(key) for key in margins) == thresholds
    for tail in ("0.5", "1"):
        assert margins[tail] == 0.002
    for bulk in ("0.01", "0.05", "0.1"):
        assert margins[bulk] == 0.01
    assert "exceedance_absolute_margin" not in parameters

    recurrence = inputs["decision_rules"]["recurrence"]
    assert "exceedance_margins_hartree" in recurrence
    # The directional fix: a bulk-threshold decrease alongside a W rise (what
    # He-v1 itself shows) must not be read as a contradiction.
    assert "is not a contradiction" in recurrence
    assert "bulk-threshold" in recurrence

    bounded_null = inputs["decision_rules"]["bounded_null"]
    assert "exceedance_margins_hartree" in bounded_null
    assert "0.002" in bounded_null and "0.01" in bounded_null


def test_all_decided_lane_proposals_cite_the_operator_review_note() -> None:
    """Every decided proposal names the 2026-11-07 operator-review note."""
    proposals = load_inputs()["lane_proposals"]
    decided = [p for p in proposals if p["id"] != "one_electron_pipeline_check"]
    assert len(decided) == 7
    for proposal in decided:
        assert proposal["status"] == "decided_2026-11-07"
        assert "operator-review-contract-proposals-20261107" in proposal["decided_by"]


def test_nuclear_curvature_matches_he_v1_primary_source_and_normalizability_gate() -> None:
    """Item 3: match He-v1's trainable curvature law, not an independent choice."""
    proposal = next(
        p
        for p in load_inputs()["lane_proposals"]
        if p["id"] == "learned_model_optimizer_and_training_sampler"
    )
    assert "418accf153368aab45586dc2a2cc97c18472691c" in proposal["proposal"]
    assert "curvature_coefficient=0.01" in proposal["proposal"]
    assert "curvature_range=1.0" in proposal["proposal"]
    assert "trainable=true" in proposal["proposal"]
    # The correction: item 3 is NOT an independent 'trainable' choice; it is
    # whatever He-v1 did, with the correction quoted verbatim.
    assert "whatever He-v1 did we do in Hydrogen experiment" in proposal["decided_by"]
    gate = load_inputs()["measurement"]["normalizability_gate"]
    assert gate["exact_hydrogen_answer"] == "c=0"


def test_one_electron_forced_deviations_from_he_v1_are_recorded() -> None:
    """Item 5: deviations forced by the one-electron system must be named."""
    proposal = next(
        p
        for p in load_inputs()["lane_proposals"]
        if p["id"] == "learned_model_optimizer_and_training_sampler"
    )
    for phrase in (
        "ElectronElectronCusp is omitted from the H-v2 model configuration",
        "n_up=1, n_down=0",
        "nuclear_charge is 1",
        "readout device rather than a second electron",
    ):
        assert phrase in proposal["proposal"], phrase


def test_science_venue_names_both_cannon_gpu_partitions_with_wall_limits() -> None:
    """Item 7: Cannon with long jobs, not packed Polaris; partitions cited."""
    proposal = next(
        p for p in load_inputs()["lane_proposals"] if p["id"] == "science_venue_and_ceiling"
    )
    for phrase in (
        "kozinsky_gpu",
        "7-00:00:00",
        "seas_gpu",
        "2-00:00:00",
        "--constraint=a100",
    ):
        assert phrase in proposal["proposal"], phrase
    assert "Polaris is not used for H-v2 science" in proposal["proposal"]


def test_four_stream_design_kept_with_fallback_condition_recorded() -> None:
    """Item 8: keep four streams; fallback is conditional and unconfirmed."""
    decision = load_inputs()["measurement"]["fresh_evaluation"]["stream_design_decision"]
    assert decision["status"] == "decided_2026-11-07"
    assert "too difficult" in decision["fallback"]
    assert "confirmed with the operator through the manager" in decision["fallback"]
    assert "5 streams" in decision["fallback"]


def test_science_lineages_stay_prohibited_pending_remaining_gates() -> None:
    """The amendment settles items 1, 2 and 8 but does not lift the science hold."""
    science_lineages = load_inputs()["admission"]["science_lineages"]
    # Unchanged invariants from round 4 (test_contract_review_amend_r4.py).
    assert science_lineages["current_status"] == "prohibited"
    assert science_lineages["status"] == "prohibited_until_operator_reviews_science_plan"
    disposition = science_lineages["operator_review_20261107_disposition"]
    assert "operator-review-contract-proposals-20261107" in disposition
    assert "5cfcca3e" in disposition
    assert "406194c5" in disposition
    assert "2f1b2921" in disposition
