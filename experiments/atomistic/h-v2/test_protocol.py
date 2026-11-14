"""Unit controls for the frozen H-v2 pipeline source."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).parent


def _module(name: str):
    spec = importlib.util.spec_from_file_location(f"h_v2_{name}", HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


oracle = _module("analytic_oracle")
protocol = _module("protocol")


def test_exact_and_imperfect_analytic_controls() -> None:
    exact = oracle.analytic_control(1.0)
    imperfect = oracle.analytic_control(1.2)
    assert exact["born_mean_hartree"] == pytest.approx(-0.5)
    assert exact["variance_hartree2"] == pytest.approx(0.0)
    assert exact["w_p99_9_hartree"] == 0.0
    assert imperfect["born_mean_hartree"] == pytest.approx(-0.48)
    assert imperfect["variance_hartree2"] == pytest.approx(0.0576)
    assert imperfect["median_local_energy_hartree"] == pytest.approx(-0.540498, abs=2e-6)
    assert imperfect["w_p99_9_hartree"] == pytest.approx(2.339741, abs=2e-6)
    assert imperfect["exceedance_gt_1"] == pytest.approx(0.008304, abs=2e-6)
    assert imperfect["exceedance_gt_0.01"] == pytest.approx(0.926514, abs=2e-6)


def test_zero_radius_is_rejected_without_regularization() -> None:
    with pytest.raises(ValueError, match="radius"):
        oracle.local_energy(0.0)
    with pytest.raises(ValueError, match="radius"):
        oracle.local_energy(float("nan"))


def test_independent_gamma_draws_and_zero_event_bound() -> None:
    radii = oracle.sample_born_radii(20000, 910004)
    assert all(radius > 0.0 and math.isfinite(radius) for radius in radii)
    assert sum(radius <= 1.0 for radius in radii) / len(radii) == pytest.approx(
        oracle.gamma3_cdf(1.0), abs=0.015
    )
    assert 0.0 < oracle.zero_event_upper_bound(4096) < 0.01


def test_frozen_configs_are_hydrogen_and_model_only_eval() -> None:
    train = protocol.training_config()
    protocol.assert_hydrogen_contract(train)
    assert train["runtime"]["seed"] != train["sampler"]["seed"]
    evaluation = protocol.evaluation_config(Path("/tmp/checkpoint"), stream=1)
    protocol.assert_hydrogen_contract(evaluation, evaluation=True)
    assert evaluation["runtime"]["seed"] != train["runtime"]["seed"]
    assert evaluation["load"]["mode"] == "model_only"


def test_frozen_training_config_selects_the_contract_sampler_literals() -> None:
    """Pin the frozen sampler binding the moved core tests rely on as literals.

    ``tests/unit/sampling/test_hydrogen_sampler_controls.py`` builds
    ``MetropolisSampler`` directly from these same numeric literals (not from
    this configuration) because the experiment boundary forbids that core
    test from importing ``tpen`` or reading this file. This test is what
    keeps the two in sync: a drift in the frozen sampler settings fails here.
    """

    sampler = protocol.training_config()["sampler"]
    assert sampler["_target_"] == "tpen.sampling.metropolis.MetropolisSampler"
    assert sampler["n_walkers"] == 256
    assert sampler["burn_in"] == 100
    assert sampler["n_steps"] == 10
    assert sampler["proposal_scale"] == 0.5
    assert sampler["seed"] == 910002


def test_frozen_evaluation_model_block_equals_training_model_block() -> None:
    """Pin that training and fresh evaluation select the identical model block.

    The pipeline check does not re-derive the evaluation model from training;
    both configs are frozen independently. This keeps them from drifting apart
    (for example the nuclear cusp law) without either config-identity check
    above catching the mismatch.
    """

    train_model = protocol.training_config()["model"]
    evaluation_model = protocol.evaluation_config(Path("/tmp/checkpoint"), stream=0)["model"]
    assert train_model == evaluation_model


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda cfg: cfg["system"]["nuclei"].update(charges=[2.0]), "charge-one"),
        (lambda cfg: cfg["hamiltonian_terms"].update(electron_electron={}), "only kinetic"),
        (lambda cfg: cfg["hamiltonian_terms"]["kinetic"].update(_target_="wrong.Kinetic"), "-1/2"),
        (lambda cfg: cfg["hamiltonian_terms"]["electron_nucleus"].update(eps=0.1), "raw"),
        (lambda cfg: cfg["sampler"].update(_target_="wrong.Sampler"), "Metropolis sampler"),
        (lambda cfg: cfg["model"]["factors"].append({"_target_": "tpen.nn.ElectronElectronCusp"}), "only the nuclear"),
        (lambda cfg: cfg["model"]["factors"][0].pop("law"), "must declare a law"),
        (lambda cfg: cfg["model"]["factors"][0]["law"].update(_target_="wrong.Law"), "contract curvature law"),
        (lambda cfg: cfg["model"]["factors"][0]["law"].update(curvature_coefficient=0.02), "coefficient must be 0.01"),
        (lambda cfg: cfg["model"]["factors"][0]["law"].update(curvature_range=2.0), "range must be 1.0"),
        (lambda cfg: cfg["model"]["factors"][0]["law"].update(trainable=False), "must be trainable"),
        (lambda cfg: cfg["model"]["factors"][0]["law"].update(eps=0.5), "unexpected keys"),
        (lambda cfg: cfg["model"]["factors"][0].update(law=None), "must be a mapping"),
    ],
)
def test_physics_mutations_are_rejected(mutation, message) -> None:
    config = protocol.training_config()
    mutation(config)
    with pytest.raises(ValueError, match=message):
        protocol.assert_hydrogen_contract(config)


# -----------------------------------------------------------------------------
# Independent numeric-integration controls for the sampler tail qualification
# (028867dc) oracle extensions: `radial_survival`, `radial_lower_tail_quantile`,
# `radial_upper_tail_quantile`, `energy_exceedance_threshold`.  Each check below
# recomputes the target quantity from the Gamma(shape=3, rate=2*alpha) density
# by composite Simpson's rule, never by calling the closed-form functions
# under test, so an error in the closed form (not just a typo in a literal)
# would be caught here.
# -----------------------------------------------------------------------------


def _simpson(f, low: float, high: float, n: int) -> float:
    """Composite Simpson's rule with ``n`` (even) subintervals."""

    assert n % 2 == 0
    h = (high - low) / n
    total = f(low) + f(high)
    for i in range(1, n):
        total += f(low + i * h) * (4.0 if i % 2 else 2.0)
    return total * h / 3.0


def _gamma3_pdf(radius: float, alpha: float) -> float:
    rate = 2.0 * alpha
    return (rate**3) * radius * radius * math.exp(-rate * radius) / 2.0


def _numeric_gamma3_cdf(radius: float, alpha: float) -> float:
    if radius <= 0.0:
        return 0.0
    return _simpson(lambda r: _gamma3_pdf(r, alpha), 0.0, radius, 20000)


def _numeric_exceedance_probability(threshold: float, alpha: float) -> float:
    median_radius = oracle.gamma3_quantile(0.5, alpha)
    median_energy = oracle.local_energy(median_radius, alpha)

    def integrand(radius: float) -> float:
        if radius <= 0.0:
            return 0.0
        energy = -(alpha**2) / 2.0 + (alpha - 1.0) / radius
        if abs(energy - median_energy) <= threshold:
            return 0.0
        return _gamma3_pdf(radius, alpha)

    return _simpson(integrand, 1e-9, 60.0 / alpha, 500000)


@pytest.mark.parametrize("radius,alpha", [(0.1, 1.0), (1.0, 1.0), (3.0, 1.0), (0.2, 1.2), (2.5, 1.2)])
def test_gamma3_cdf_matches_independent_numeric_integration(radius, alpha) -> None:
    assert oracle.gamma3_cdf(radius, alpha) == pytest.approx(_numeric_gamma3_cdf(radius, alpha), abs=1e-6)


@pytest.mark.parametrize("level,alpha", [(0.2, 1.0), (0.01, 1.0), (0.001, 1.2), (0.1, 1.2)])
def test_radial_quantiles_invert_the_survival_functions(level, alpha) -> None:
    lower = oracle.radial_lower_tail_quantile(level, alpha)
    assert oracle.gamma3_cdf(lower, alpha) == pytest.approx(level, abs=1e-8)
    upper = oracle.radial_upper_tail_quantile(level, alpha)
    assert oracle.radial_survival(upper, alpha) == pytest.approx(level, abs=1e-8)
    assert oracle.radial_survival(lower, alpha) == pytest.approx(1.0 - level, abs=1e-8)


@pytest.mark.parametrize("threshold", [0.01, 1.0])
def test_exceedance_probability_matches_independent_numeric_integration(threshold) -> None:
    assert oracle.exceedance_probability(threshold, 1.2) == pytest.approx(
        _numeric_exceedance_probability(threshold, 1.2), abs=2e-3
    )


@pytest.mark.parametrize("level", [0.5, 0.1, 0.001])
def test_energy_exceedance_threshold_inverts_exceedance_probability(level) -> None:
    threshold = oracle.energy_exceedance_threshold(level, 1.2)
    assert oracle.exceedance_probability(threshold, 1.2) == pytest.approx(level, abs=1e-8)


def _contract_inputs() -> dict:
    import json

    return json.loads((HERE / "contract_inputs.json").read_text(encoding="utf-8"))


def test_amplitude_risk_thresholds_agree_with_the_frozen_contract() -> None:
    risk = _contract_inputs()["measurement"]["amplitude_risk"]["thresholds"]
    assert list(protocol.AMPLITUDE_RISK_THRESHOLDS) == risk["frozen_tau"]
    assert protocol.AMPLITUDE_RISK_PRIMARY_THRESHOLD == risk["primary"]
    assert protocol.AMPLITUDE_RISK_LABELS == risk["labels"]


def test_amplitude_risk_module_yaml_matches_the_frozen_thresholds() -> None:
    config = protocol.load_yaml(HERE / "configs" / "amplitude_risk_module.yaml")
    entry = config["amplitude_risk_callback"]
    assert entry["_target_"] == "tpen.callback.AmplitudeRiskFraction"
    assert entry["thresholds"] == list(protocol.AMPLITUDE_RISK_THRESHOLDS)


def test_assert_amplitude_risk_thresholds_is_a_noop_without_the_callback() -> None:
    protocol.assert_amplitude_risk_thresholds(protocol.training_config())
    protocol.assert_hydrogen_contract(protocol.training_config())


def test_assert_amplitude_risk_thresholds_accepts_the_frozen_list() -> None:
    config, _ = protocol.science_training_config("L1", 0)
    for target in protocol.candidate_modules_to_append():
        protocol.append_candidate_module(config, target)
    protocol.assert_amplitude_risk_thresholds(config)
    protocol.assert_hydrogen_contract(config)


def test_assert_amplitude_risk_thresholds_refuses_a_drifted_list() -> None:
    config, _ = protocol.science_training_config("L1", 0)
    for target in protocol.candidate_modules_to_append():
        protocol.append_candidate_module(config, target)
    amplitude = next(
        callback for callback in config["callbacks"]
        if callback["_target_"] == "tpen.callback.AmplitudeRiskFraction"
    )
    amplitude["thresholds"] = [-1, -2]
    with pytest.raises(ValueError, match="frozen"):
        protocol.assert_amplitude_risk_thresholds(config)
    with pytest.raises(ValueError, match="frozen"):
        protocol.assert_hydrogen_contract(config)
