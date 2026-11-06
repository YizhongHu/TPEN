"""Pre-review round 1 tests for the 2026-11-06 amendment of the H-v2 contract.

Proposed by the amendment pre-reviewer. Each test states in its name whether
it is expected to pass on head 4badcd4d (GREEN) or to fail until the matching
finding is resolved (RED). Tests read only contract_inputs.json; they use the
standard library and pytest only, and never import tpen.
"""

import json
import math
import re
import statistics
from pathlib import Path

import pytest


INPUTS = Path(__file__).with_name("contract_inputs.json")
HEX8 = re.compile(r"^[0-9a-f]{8}$")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def _gamma3_cdf(radius: float, rate: float) -> float:
    # Closed-form CDF of Gamma(shape=3, rate): 1 - e^{-x}(1 + x + x^2/2), x = rate*r.
    x = rate * radius
    return 1 - math.exp(-x) * (1 + x + x * x / 2)


def _gamma3_quantile(probability: float, rate: float) -> float:
    low, high = 0.0, 1.0
    while _gamma3_cdf(high, rate) < probability:
        high *= 2
    for _ in range(200):
        middle = (low + high) / 2
        if _gamma3_cdf(middle, rate) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _exact_exceedance_alpha_1_2(threshold: float) -> float:
    # alpha = 1.2: R ~ Gamma(3, 2.4), E_L = -0.72 + 0.2/r, A = |E_L - median(E_L)|.
    alpha, rate = 1.2, 2.4
    r_median = _gamma3_quantile(0.5, rate)
    coefficient = alpha - 1
    # A > t  <=>  |1/r - 1/r_median| > t / coefficient
    lower_radius = 1 / (1 / r_median + threshold / coefficient)
    mass = _gamma3_cdf(lower_radius, rate)
    right_inverse = 1 / r_median - threshold / coefficient
    if right_inverse > 0:
        mass += 1 - _gamma3_cdf(1 / right_inverse, rate)
    return mass


# ---------------------------------------------------------------- GREEN tests


def test_green_tail_qualification_reuses_the_contract_thresholds_and_shape() -> None:
    """The qualification grid must track the science contract, not a private copy."""
    inputs = load_inputs()
    measurement = inputs["measurement"]
    tail = inputs["tail_characterization"]["sampler_tail_qualification"]
    alpha_only = tail["alpha_1_2_only"]
    assert alpha_only["fixed_exceedance_thresholds_hartree"] == (
        measurement["fixed_exceedance_thresholds_hartree"]
    )
    shape = tail["shape_and_seeds"]["fresh_evaluation_sampler"]
    science = measurement["fresh_evaluation"]
    for key in ("burn_in", "stride", "walkers", "draws_per_stream"):
        assert shape[key] == science[key], key
    # W is the 99.9th percentile, so the W exceedance level is 1 - 0.999.
    assert alpha_only["w_level"] == pytest.approx(1 - 0.999, abs=1e-12)
    assert alpha_only["w_level"] in alpha_only["exceedance_levels"]


def test_green_every_comparison_counted_in_k_is_gated_by_the_support_rule() -> None:
    """K counts every declared comparison; each must clear the 100-count floor."""
    tail = load_inputs()["tail_characterization"]["sampler_tail_qualification"]
    support = tail["support_rule"]
    floor = support["minimum_expected_count_per_stream"]
    rows = support["fresh_evaluation_rows_per_stream"]
    alpha_only = tail["alpha_1_2_only"]
    # Level-based comparisons: expected count is level * rows by construction.
    for level in tail["radial_levels"] + alpha_only["exceedance_levels"]:
        assert level * rows >= floor, level
    # Fixed thresholds: expected count follows from the exact alpha=1.2 law.
    for threshold in alpha_only["fixed_exceedance_thresholds_hartree"]:
        expected = _exact_exceedance_alpha_1_2(threshold) * rows
        assert expected >= floor, (threshold, expected)
    # Cross-check the oracle against the contract's own analytic controls.
    controls = load_inputs()["measurement"]["analytic_controls"]["alpha_1_2"]
    assert _exact_exceedance_alpha_1_2(1.0) == pytest.approx(
        controls["exceedance_probability_abs_deviation_gt_1_hartree"], abs=5e-7
    )
    assert _exact_exceedance_alpha_1_2(0.01) == pytest.approx(
        controls["exceedance_probability_abs_deviation_gt_0_01_hartree"], abs=5e-7
    )


def test_green_pr_560_map_agrees_with_the_logging_module_layers() -> None:
    """Findings 3, 4 and 5 and the logging modules name the same core layers."""
    inputs = load_inputs()
    modules = {m["name"]: m["layer"] for m in inputs["logging"]["modules"]}
    findings = {f["finding"]: f["layer"] for f in inputs["pr_560_incorporation"]["findings"]}
    assert findings[3] == modules["initial_checkpoint_callback"]
    assert findings[4] == modules["parameter_activity_callback"]
    assert findings[5] == modules["readout_trajectory_generator"]
    # Finding 1 (tail-calibrated sampler bands) lands in the tail-qualification layer.
    tail_layer = inputs["tail_characterization"]["sampler_tail_qualification"]["layer"]
    assert tail_layer.startswith(findings[1])
    for layer in list(modules.values()) + list(findings.values()):
        assert HEX8.match(layer), layer


# ------------------------------------------------------------------ RED tests
# Each RED test encodes the reviewer's proposed resolution of a finding. It is
# expected to FAIL on head 4badcd4d and to pass once the finding is accepted
# and the contract is amended accordingly.


def test_red_finding_2_k_counts_every_stream_the_pass_rule_scans() -> None:
    """The pass rule takes max|z| over both streams, so K must count both."""
    tail = load_inputs()["tail_characterization"]["sampler_tail_qualification"]
    band = tail["band"]
    alpha_only = tail["alpha_1_2_only"]
    streams = tail["shape_and_seeds"]["fresh_evaluation_sampler"]["streams"]
    per_stream = (
        len(tail["radial_levels"]) * len(tail["radial_sides"]) * 2
        + len(alpha_only["fixed_exceedance_thresholds_hartree"])
        + len(alpha_only["exceedance_levels"])
    )
    assert band["k_value"] == per_stream * streams
    c_computed = statistics.NormalDist().inv_cdf(
        1 - band["family_wise_false_fail_rate"] / (2 * band["k_value"])
    )
    assert c_computed == pytest.approx(band["c_value"], abs=1e-12)


def test_red_finding_5_horizon_proposal_does_not_read_fully_pending() -> None:
    """Checkpoint counts are decided; the proposal that carries them must say so."""
    inputs = load_inputs()
    assert inputs["measurement"]["checkpoint_retention"]["status"] == "decided"
    proposal = next(
        p for p in inputs["lane_proposals"] if p["id"] == "horizon_and_checkpoints"
    )
    assert proposal["status"] != "proposed_by_lane_pending_operator_review"
    # The operator ruled out evaluating all 51; it is no longer a live alternative.
    assert not any("51" in alternative for alternative in proposal["alternatives"])


def test_red_finding_1_amplitude_risk_is_not_held_for_the_operator() -> None:
    """Operator 2026-11-06 19:56Z handed the amplitude-risk call to the lane."""
    inputs = load_inputs()
    risk = inputs["measurement"]["amplitude_risk"]
    assert "held for the operator" not in risk["blocks"]
    held = [m for m in inputs["logging"]["modules"] if m["name"] == "amplitude_risk_fraction"]
    assert held[0]["status"] != "held_for_operator"
