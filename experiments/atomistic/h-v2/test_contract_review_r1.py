"""Independent review tests for the H-v2 contract revision."""

import json
import math
import random
import re
from pathlib import Path

import pytest


INPUTS = Path(__file__).with_name("contract_inputs.json")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_evaluation_seed_formula_is_injective_and_disjoint() -> None:
    inputs = load_inputs()
    measurement = inputs["measurement"]
    roster = measurement["seed_roster"]
    match = re.search(
        r"(\d+) \+ (\d+)\*lineage_index \+ "
        r"(\d+)\*evaluated_checkpoint_ordinal \+ stream_index",
        roster["evaluation_stream_seed_formula"],
    )
    assert match is not None
    base, lineage_stride, checkpoint_stride = map(int, match.groups())
    n_ordinals = len(measurement["evaluated_checkpoints"]["updates"])
    seeds = {
        base + lineage_stride * lineage + checkpoint_stride * ordinal + stream
        for lineage in range(3)
        for ordinal in range(n_ordinals)
        for stream in range(4)
    }
    assert len(seeds) == 3 * n_ordinals * 4 == 324
    for name in (
        "training_initialization_seeds",
        "training_sampler_seeds",
        "pilot_seeds",
    ):
        assert seeds.isdisjoint(roster[name])
    assert 10 * (n_ordinals - 1) + 3 < 10000
    assert 3 < 10


def test_evaluated_set_is_subset_of_retained_set_and_covers_rule_windows() -> None:
    measurement = load_inputs()["measurement"]
    retained = set(range(0, 50001, 1000))
    assert len(retained) == measurement["checkpoint_retention"]["count_per_lineage"]
    evaluated = measurement["evaluated_checkpoints"]["updates"]
    assert evaluated == sorted(set(evaluated))
    assert len(evaluated) == measurement["evaluated_checkpoints"]["count_per_lineage"]
    assert set(evaluated).issubset(retained)
    assert {0, 1000}.issubset(evaluated)
    assert {40000, 45000, 50000}.issubset(evaluated)
    for centre in range(3000, 19001, 1000):
        assert {centre - 1000, centre, centre + 1000}.issubset(evaluated)


def test_hydrogen_alpha_state_closed_form_w_oracle() -> None:
    alpha = 1.2

    def cdf(radius: float) -> float:
        t = 2 * alpha * radius
        return 1 - math.exp(-t) * (1 + t + t * t / 2)

    def quantile(probability: float) -> float:
        low, high = 0.0, 1.0
        while cdf(high) < probability:
            high *= 2
        for _ in range(100):
            middle = (low + high) / 2
            if cdf(middle) < probability:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    def local_energy(radius: float, parameter: float = alpha) -> float:
        return -parameter * parameter / 2 + (parameter - 1) / radius

    r_median = quantile(0.5)
    median_energy = local_energy(r_median)
    coefficient = abs(alpha - 1)

    def survival_abs_deviation(threshold: float) -> float:
        left_inverse = 1 / r_median + threshold / coefficient
        right_inverse = 1 / r_median - threshold / coefficient
        lower_radius = 1 / left_inverse
        upper_mass = 1 - cdf(1 / right_inverse) if right_inverse > 0 else 0
        return cdf(lower_radius) + upper_mass

    def w_quantile(probability: float) -> float:
        target_survival = 1 - probability
        low, high = 0.0, 1.0
        while survival_abs_deviation(high) > target_survival:
            high *= 2
        for _ in range(100):
            middle = (low + high) / 2
            if survival_abs_deviation(middle) > target_survival:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    assert alpha * alpha / 2 - alpha == -0.48
    assert alpha * alpha * (alpha - 1) * (alpha - 1) == pytest.approx(0.0576, abs=1e-12)
    assert r_median == pytest.approx(1.114192, abs=1e-5)
    assert median_energy == pytest.approx(-0.540498, abs=1e-5)
    assert w_quantile(0.999) == pytest.approx(2.339741, abs=1e-4)
    assert survival_abs_deviation(1.0) == pytest.approx(0.008304, abs=1e-5)
    assert survival_abs_deviation(0.01) == pytest.approx(0.926514, abs=1e-5)

    rng = random.Random(20261105)
    energies = [
        local_energy(-(math.log(rng.random()) + math.log(rng.random()) + math.log(rng.random())) / (2 * alpha))
        for _ in range(400000)
    ]
    sorted_energies = sorted(energies)
    empirical_median = (sorted_energies[199999] + sorted_energies[200000]) / 2
    deviations = sorted(abs(energy - empirical_median) for energy in energies)
    w_mc = deviations[int(0.999 * len(deviations))]
    assert abs(w_mc - 2.339741) / 2.339741 < 0.05
    assert abs(sum(energies) / len(energies) + 0.48) < 0.003

    exact_energies = [local_energy(radius, parameter=1.0) for radius in (0.1, 0.5, 1.0, 2.0)]
    assert all(energy == pytest.approx(-0.5, abs=1e-12) for energy in exact_energies)
    exact_deviations = [abs(energy - (-0.5)) for energy in exact_energies]
    assert max(exact_deviations) == 0
    for threshold in (0.01, 0.05, 0.1, 0.5, 1.0):
        assert sum(deviation > threshold for deviation in exact_deviations) / len(exact_deviations) == 0


def test_decision_rule_parameters_are_machine_readable() -> None:
    rules = load_inputs()["decision_rules"]
    assert "parameters" in rules
    parameters = rules["parameters"]
    assert parameters["w_factor_margin"] == 2
    assert parameters["w_absolute_margin_hartree"] == 0.01
    assert parameters["exceedance_absolute_margin"] == 0.002
    assert parameters["basin_center_updates"] == list(range(3000, 19001, 1000))
    assert parameters["early_updates"] == [0, 1000]
    assert parameters["late_updates"] == [40000, 45000, 50000]
    assert parameters["late_required_count"] == 2
    assert parameters["selection_streams"] == [0, 1]
    assert parameters["confirmation_streams"] == [2, 3]
    assert parameters["lineages_required_for_recurrence"] == 2
