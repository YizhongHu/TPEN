"""Semantic contract checks for the H-v2 documentation slice.

These checks validate invariants of the contract, not pending operator choices
or any scientific result.
"""

import json
import math
import re
from pathlib import Path


INPUTS = Path(__file__).with_name("contract_inputs.json")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def _integer_seeds(value: object) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, dict):
        return [seed for item in value.values() for seed in _integer_seeds(item)]
    if isinstance(value, list):
        return [seed for item in value for seed in _integer_seeds(item)]
    return []


def _evaluation_seeds(inputs: dict) -> set[int]:
    roster = inputs["measurement"]["seed_roster"]
    match = re.search(
        r"(\d+) \+ (\d+)\*lineage_index \+ "
        r"(\d+)\*evaluated_checkpoint_ordinal \+ stream_index",
        roster["evaluation_stream_seed_formula"],
    )
    assert match is not None
    base, lineage_stride, checkpoint_stride = map(int, match.groups())
    n_checkpoints = len(inputs["measurement"]["evaluated_checkpoints"]["updates"])
    return {
        base + lineage_stride * lineage + checkpoint_stride * ordinal + stream
        for lineage in range(3)
        for ordinal in range(n_checkpoints)
        for stream in range(4)
    }


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
        "an absent or inconclusive gating diagnostic is never a pass"
    )
    assert tail["gating"]["statistics"] == [
        "W_99_9",
        "high_quantiles",
        "exceedance_probabilities",
    ]
    assert tail["non_gating_interpretation"]["statistics"] == [
        "beta survival-slope curvature",
        "generalized-Pareto xi",
        "Hill",
    ]


def test_science_hold_and_pipeline_scope_are_separate() -> None:
    admission = load_inputs()["admission"]
    assert admission["pipeline_check"]["status"].startswith("admitted_")
    assert [part["id"] for part in admission["pipeline_check"]["parts"]] == list(
        "abcdefg"
    )
    assert admission["science_lineages"]["status"] == (
        "prohibited_until_operator_reviews_science_plan"
    )


def test_all_seed_namespaces_and_evaluation_seeds_are_globally_distinct() -> None:
    inputs = load_inputs()
    roster = inputs["measurement"]["seed_roster"]
    explicit_seed_values = _integer_seeds(roster)
    explicit_seeds = set(explicit_seed_values)
    evaluation_seeds = _evaluation_seeds(inputs)
    assert len(evaluation_seeds) == 324
    assert len(explicit_seed_values) == len(explicit_seeds)
    assert explicit_seeds.isdisjoint(evaluation_seeds)
    assert len(explicit_seeds | evaluation_seeds) == len(explicit_seeds) + 324
    assert inputs["decision_rules"]["categories"] == [
        "recurrence",
        "bounded_null",
        "unresolved",
    ]


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


def test_bounded_null_reference_set_and_late_count_are_machine_readable() -> None:
    inputs = load_inputs()
    measurement = inputs["measurement"]
    parameters = inputs["decision_rules"]["parameters"]
    evaluated = measurement["evaluated_checkpoints"]["updates"]
    assert parameters["bounded_null_reference_updates"] == [
        update for update in evaluated if update <= 20000
    ]
    assert parameters["bounded_null_late_count"] == parameters["late_required_count"]


def test_training_and_fresh_evaluation_sampler_settings_are_exact() -> None:
    inputs = load_inputs()
    proposal = inputs["lane_proposals"][7]["proposal"]
    criteria = next(
        part["pass_criteria"]
        for part in inputs["admission"]["pipeline_check"]["parts"]
        if part["id"] == "e"
    )
    assert criteria["training_sampler"] == {
        "walkers": 256,
        "burn_in": 100,
        "metropolis_steps_per_update": 10,
        "proposal_scale_bohr": 0.5,
    }
    assert criteria["fresh_evaluation"] == {
        "walkers": 1024,
        "draws_per_stream": 64,
        "streams": 2,
        "burn_in": 1000,
        "stride": 20,
    }
    assert criteria["fresh_evaluation"]["burn_in"] != criteria["training_sampler"]["burn_in"]
    assert "burn-in 100" in proposal
    assert "10 Metropolis steps per update" in proposal
    assert "proposal scale 0.5 bohr" in proposal


def test_normalizability_gate_is_explicit() -> None:
    gate = load_inputs()["measurement"]["normalizability_gate"]
    assert gate["outer_slope"] == "-1 + c/d"
    assert gate["kappa_min"] == 0.5
    assert gate["condition"] == "outer slope <= -kappa_min at every checkpoint"
    assert gate["violation"] == "checkpoint is unresolved; no records are filtered"
