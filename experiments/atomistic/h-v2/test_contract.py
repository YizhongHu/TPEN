"""Semantic contract checks for the H-v2 documentation slice.

These checks validate invariants of the contract, not pending operator choices
or any scientific result.
"""

import json
import math
import re
import statistics
from pathlib import Path

import pytest


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
    # consumed_outside_roster records blocks the spike used outside this
    # roster (including a deliberate replay of pipeline_cpu_learned_e's own
    # seeds); it is not itself an assignment made by the roster, so it is
    # excluded from the roster's own-seed distinctness sweep.
    own_assignments = {k: v for k, v in roster.items() if k != "consumed_outside_roster"}
    explicit_seed_values = _integer_seeds(own_assignments)
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
    pipeline_check = next(p for p in proposals if p["id"] == "one_electron_pipeline_check")
    assert pipeline_check["status"] == "proposed_by_lane_admitted_non_science_scope"
    decided = [p for p in proposals if p["id"] != "one_electron_pipeline_check"]
    # Operator review 2026-11-07 (operator-review-contract-proposals-20261107) settled
    # every remaining lane proposal; none is still pending operator review.
    assert all(item["status"] == "decided_2026-11-07" for item in decided)
    assert all("decided_by" in item and item["decided_by"] for item in decided)
    assert not any(
        item["status"].startswith("proposed_by_lane_pending_operator_review")
        for item in proposals
    )
    horizon = next(p for p in proposals if p["id"] == "horizon_and_checkpoints")
    assert not any("51" in alternative for alternative in horizon["alternatives"])
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


def test_checkpoint_retention_and_evaluation_counts_are_decided() -> None:
    measurement = load_inputs()["measurement"]
    retention = measurement["checkpoint_retention"]
    evaluated = measurement["evaluated_checkpoints"]
    assert retention["status"] == "decided"
    assert evaluated["status"] == "decided"
    assert retention["count_per_lineage"] == 51
    assert evaluated["count_per_lineage"] == 27
    assert "operator" in retention["decided_by"].lower()
    assert "operator" in evaluated["decided_by"].lower()


def test_amplitude_risk_definition_and_thresholds_are_decided() -> None:
    inputs = load_inputs()
    amplitude_risk = inputs["measurement"]["amplitude_risk"]
    assert amplitude_risk["status"] == "definition_and_thresholds_decided"
    assert "held for the operator" not in amplitude_risk["blocks"]
    assert "descriptive only" in amplitude_risk["interpretation"]
    assert "fraction" in amplitude_risk["definition"]
    assert "Pfaffian" in amplitude_risk["naming_rule"]
    thresholds = amplitude_risk["thresholds"]
    assert thresholds["frozen_tau"] == [-3, -4, -5, -7, -10]
    assert thresholds["primary"] == -5
    assert thresholds["labels"]["-5"] == "primary descriptive scale"
    assert "marginal resolution" in thresholds["labels"]["-7"]
    assert "not a healthy zero" in thresholds["labels"]["-10"]
    assert set(str(tau) for tau in thresholds["frozen_tau"]) == set(thresholds["labels"])
    assert amplitude_risk["evidence"]["spike_sha"] == (
        "39135fda1388ef319c2325c8ad934de3b3c513db"
    )
    assert amplitude_risk["evidence"]["cannon_jobs"] == [58383577, 58388053, 58388109]
    # The amplitude-risk thresholds are descriptive only; no gate on science.
    assert "no gate" in amplitude_risk["must_fix_before_science"]
    assert inputs["admission"]["science_lineages"]["status"] == (
        "prohibited_until_operator_reviews_science_plan"
    )


def test_logging_is_an_optional_module_not_an_always_on_core_metric() -> None:
    logging_block = load_inputs()["logging"]
    assert logging_block["status"] == "decided"
    assert "optional module" in logging_block["design_rule"]
    assert "does not compute or log" in logging_block["design_rule"]
    assert logging_block["core_logs_unconditionally"] is False
    module_names = {module["name"] for module in logging_block["modules"]}
    assert "amplitude_risk_fraction" in module_names
    assert "raw_training_batch_recorder" in module_names
    held = [m for m in logging_block["modules"] if m["name"] == "amplitude_risk_fraction"]
    assert held[0]["status"] == "thresholds_frozen_implementation_pending"
    kinds = {"training_callback", "evaluation_generator", "one_shot_callback"}
    assert all(module["kind"] in kinds for module in logging_block["modules"])


def test_module_status_agrees_with_the_amplitude_risk_decision() -> None:
    inputs = load_inputs()
    risk_status = inputs["measurement"]["amplitude_risk"]["status"]
    module = next(m for m in inputs["logging"]["modules"] if m["name"] == "amplitude_risk_fraction")
    assert risk_status == "definition_and_thresholds_decided"
    assert "pending_spike" not in module.get("status", "")


def test_raw_training_batch_cadence_and_format_are_decided() -> None:
    cadence = load_inputs()["logging"]["cadence"]["raw_training_batches"]
    assert cadence["capture_coordinate"] == "iteration_step"
    assert cadence["every_n_steps"] == 50
    assert cadence["includes_step_zero"] is True
    assert cadence["partition_size_steps"] == 1000
    assert cadence["format"] == "numpy_column_bundle_with_json_integrity_sidecar"
    assert len(cadence["out_of_memory_checks"]) == 2


def test_storage_row_counts_match_their_declared_formulas() -> None:
    figures = load_inputs()["logging"]["storage_figures"]
    scale = figures["scale"]
    fresh = figures["fresh_evaluation"]
    raw = figures["raw_training_batches"]
    assert fresh["rows"] == 3 * 27 * 4 * scale["training_walkers"] * 256
    assert raw["batches_per_lineage"] == 1001
    assert raw["rows"] == scale["lineages"] * raw["batches_per_lineage"] * scale["training_walkers"]
    assert fresh["compact_gib"] == pytest.approx(
        fresh["rows"] * fresh["bytes_per_row_compact"] / 2**30, rel=0.01
    )
    assert raw["compact_gib"] == pytest.approx(
        raw["rows"] * raw["bytes_per_row_compact"] / 2**30, rel=0.01
    )
    cadence = load_inputs()["logging"]["cadence"]["raw_training_batches"]
    full_partitions, remainder = divmod(raw["batches_per_lineage"], 20)
    assert raw["partitions_per_lineage"] == full_partitions + (1 if remainder else 0)
    assert raw["data_objects_total"] == scale["lineages"] * raw["partitions_per_lineage"]
    assert raw["data_objects_total"] == 153
    assert cadence["partition_size_steps"] == 1000
    assert figures["compression_ratio"] == (
        "unmeasured; a pilot block must measure it before a Holystore budget promises it"
    )


def test_storage_lifecycle_requires_a_disposition_before_removal() -> None:
    lifecycle = load_inputs()["logging"]["storage_lifecycle"]
    assert lifecycle["working_location"] == "Cannon netscratch, not scratch"
    assert lifecycle["purge_window_days"] == 90
    assert "lifecycle disposition" in lifecycle["removal_rule"]
    assert "recovery verification" in lifecycle["removal_rule"]


def test_sampler_tail_qualification_grid_is_sorted_and_disjoint_from_the_stub() -> None:
    tail = load_inputs()["tail_characterization"]["sampler_tail_qualification"]
    radial_levels = tail["radial_levels"]
    assert radial_levels == sorted(radial_levels, reverse=True)
    assert len(radial_levels) == 11
    assert radial_levels[-1] == 0.0001
    assert tail["radial_sides"] == ["lower_tail_cdf", "upper_tail_survival"]
    exceedance_levels = tail["alpha_1_2_only"]["exceedance_levels"]
    assert exceedance_levels == sorted(exceedance_levels, reverse=True)
    assert 0.001 in exceedance_levels
    assert exceedance_levels[0] == 0.5
    assert exceedance_levels[-1] == 0.0001
    assert tail["alpha_1_2_only"]["w_level"] == 0.001
    # alpha=1.2's large positive local-energy tail is driven by the inner
    # radial tail, not by the outer radius the superseded stub checked.
    assert "inner radial tail" in tail["targets"]["alpha_1_2"]["tail_origin"]
    assert "r=3" in tail["superseded_stub"]


def test_sampler_tail_qualification_support_rule_matches_the_declared_shape() -> None:
    tail = load_inputs()["tail_characterization"]["sampler_tail_qualification"]
    support = tail["support_rule"]
    shape = tail["shape_and_seeds"]["fresh_evaluation_sampler"]
    rows_per_stream = shape["walkers"] * shape["draws_per_stream"]
    assert rows_per_stream == support["fresh_evaluation_rows_per_stream"]
    deepest = support["deepest_gated_level"]
    assert deepest == min(tail["radial_levels"])
    expected_count = rows_per_stream * deepest
    assert support["minimum_expected_count_per_stream"] == 100
    assert expected_count >= support["minimum_expected_count_per_stream"]
    assert round(expected_count) == support["deepest_gated_level_expected_count"]


def test_sampler_tail_qualification_k_and_c_match_the_formula() -> None:
    tail = load_inputs()["tail_characterization"]["sampler_tail_qualification"]
    band = tail["band"]
    alpha_only = tail["alpha_1_2_only"]
    streams = tail["shape_and_seeds"]["fresh_evaluation_sampler"]["streams"]
    per_stream = (
        len(tail["radial_levels"]) * len(tail["radial_sides"]) * 2
        + len(alpha_only["fixed_exceedance_thresholds_hartree"])
        + len(alpha_only["exceedance_levels"])
    )
    k_computed = per_stream * streams
    assert k_computed == band["k_value"]
    c_computed = statistics.NormalDist().inv_cdf(
        1 - band["family_wise_false_fail_rate"] / (2 * band["k_value"])
    )
    assert c_computed == pytest.approx(band["c_value"], abs=1e-12)
    assert band["centering"].startswith("per-stream sample median")
    assert "recomputed inside every whole-walker resample" in band["se_method"]


def test_sampler_tail_qualification_seed_namespace_is_disjoint() -> None:
    inputs = load_inputs()
    roster = inputs["measurement"]["seed_roster"]
    namespace = roster["non_science_seed_namespaces"]["sampler_tail_qualification"]
    namespace_seeds = set(_integer_seeds(namespace))
    assert len(namespace_seeds) == 4
    other_seeds = set(
        _integer_seeds(roster["training_initialization_seeds"])
        + _integer_seeds(roster["training_sampler_seeds"])
    )
    for key, value in roster["non_science_seed_namespaces"].items():
        if key != "sampler_tail_qualification":
            other_seeds |= set(_integer_seeds(value))
    other_seeds |= _evaluation_seeds(inputs)
    assert namespace_seeds.isdisjoint(other_seeds)


def test_roster_seeds_avoid_blocks_consumed_outside_the_roster() -> None:
    inputs = load_inputs()
    roster = inputs["measurement"]["seed_roster"]
    consumed: set[int] = set()
    for block in roster["consumed_outside_roster"]:
        lo, hi = block["range"]
        consumed |= set(range(lo, hi + 1))
    assert {910001, 910002, 920001} <= consumed  # the spike's known collisions are recorded
    roster_seeds = set(_integer_seeds(roster["training_initialization_seeds"]))
    roster_seeds |= set(_integer_seeds(roster["training_sampler_seeds"]))
    for name, namespace in roster["non_science_seed_namespaces"].items():
        if name == "pipeline_cpu_learned_e":
            # Only training and sampler were replayed by the spike (recorded
            # above); the evaluation seeds must still be disjoint from every
            # consumed block.
            roster_seeds |= set(_integer_seeds(namespace["evaluation"]))
            continue
        roster_seeds |= set(_integer_seeds(namespace))
    roster_seeds |= _evaluation_seeds(inputs)
    assert roster_seeds.isdisjoint(consumed)


def test_pr_560_incorporation_maps_every_finding_to_a_layer() -> None:
    incorporation = load_inputs()["pr_560_incorporation"]
    findings = incorporation["findings"]
    assert [item["finding"] for item in findings] == list(range(1, 8))
    assert all(item["layer"] for item in findings)
    assert incorporation["preserved_tip"] == "0da2b23d628a15b18ac2031ff5f72dff0c010280"
    assert len(incorporation["hooke_only_handed_to_manager"]) == 4


def test_green_tail_qualification_targets_follow_from_alpha() -> None:
    """R ~ Gamma(3, 2*alpha) and E_L = -alpha^2/2 + (alpha-1)/r for psi = exp(-alpha r)."""

    targets = load_inputs()["tail_characterization"]["sampler_tail_qualification"]["targets"]
    for key, alpha in (("alpha_1", 1.0), ("alpha_1_2", 1.2)):
        target = targets[key]
        match = re.fullmatch(r"Gamma\(shape=(\d+), rate=([\d.]+)\)", target["radius_distribution"])
        assert match is not None, key
        assert int(match.group(1)) == 3
        assert float(match.group(2)) == pytest.approx(2 * alpha, abs=1e-12)
        energy = target["local_energy_hartree"]
        if isinstance(energy, (int, float)):
            assert alpha == 1.0
            assert energy == pytest.approx(-alpha * alpha / 2, abs=1e-12)
        else:
            match = re.fullmatch(r"(-?[\d.]+) \+ ([\d.]+)/r", energy)
            assert match is not None, key
            assert float(match.group(1)) == pytest.approx(-alpha * alpha / 2, abs=1e-12)
            assert float(match.group(2)) == pytest.approx(alpha - 1, abs=1e-12)


def test_experiment_boundary_rule_and_run_module_trap_are_stated() -> None:
    boundary = load_inputs()["experiment_boundary"]
    assert "run_from_config" in boundary["rule"]
    assert "tpen command" in boundary["rule"]
    assert "core tests/" in boundary["rule"]
    assert boundary["allowed_core_surfaces"] == [
        "configuration",
        "tpen.run.run_from_config",
        "the installed tpen command",
    ]
    assert "python -m tpen.run" in boundary["run_module_trap"]
    assert "exits 0 and does nothing" in boundary["run_module_trap"]
    assert "tpen.run.main" not in boundary["run_module_trap"]
