"""Semantic contract checks for the H-v2 documentation slice.

These checks validate invariants of the contract, not pending operator choices
or any scientific result.
"""

import json
from pathlib import Path


INPUTS = Path(__file__).with_name("contract_inputs.json")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_primary_statistic_drives_rules_and_mad_is_secondary() -> None:
    inputs = load_inputs()
    measurement = inputs["measurement"]
    rules = inputs["decision_rules"]
    assert "W" in measurement["primary_statistic"]
    assert "W" in rules["recurrence"]
    assert "W" in rules["bounded_null"]
    assert "MAD" not in rules["recurrence"]
    assert "median absolute deviation" not in rules["bounded_null"].lower()
    assert "median_absolute_deviation" in measurement["secondary_dispersion"]


def test_tail_contract_never_promotes_hill_or_missing_diagnostics() -> None:
    tail = load_inputs()["measurement"]["tail_rule"]
    assert tail["hill"] == "context_only"
    assert "never replaced by a Hill" in tail["diagnostic"]
    assert tail["absence_or_inconclusive"] == (
        "unresolved; an absent admissible tail diagnostic is never a pass"
    )


def test_science_hold_and_pipeline_scope_are_separate() -> None:
    admission = load_inputs()["admission"]
    assert admission["pipeline_check"]["status"].startswith("admitted_")
    assert [part["id"] for part in admission["pipeline_check"]["parts"]] == list(
        "abcdefg"
    )
    assert admission["science_lineages"]["status"] == (
        "prohibited_until_operator_reviews_science_plan"
    )


def test_seed_rosters_are_disjoint_and_result_categories_closed() -> None:
    inputs = load_inputs()
    seeds = inputs["measurement"]["seed_roster"]
    groups = [
        set(seeds["training_initialization_seeds"]),
        set(seeds["training_sampler_seeds"]),
        set(seeds["pilot_seeds"]),
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(groups)
        for right in groups[index + 1 :]
    )
    assert (
        "1000000 + 10000*lineage_index + 10*evaluated_checkpoint_ordinal + stream_index"
        in seeds["evaluation_stream_seed_formula"]
    )
    assert set(inputs["decision_rules"]["categories"]) == {
        "recurrence",
        "bounded_null",
        "unresolved",
    }


def test_all_eight_lane_proposals_are_visible() -> None:
    proposals = load_inputs()["lane_proposals"]
    assert len(proposals) == 8
    assert all(item["status"].startswith("proposed_by_lane_") for item in proposals)
    assert all(
        item["proposal"] and item["rationale"] and item["alternatives"]
        for item in proposals
    )


def test_retention_and_evaluation_sets_are_distinct() -> None:
    measurement = load_inputs()["measurement"]
    assert measurement["checkpoint_retention"]["count_per_lineage"] == 51
    assert measurement["evaluated_checkpoints"]["count_per_lineage"] == 27
    assert measurement["checkpoint_retention"]["updates"].startswith(
        "0 and every 1000"
    )
