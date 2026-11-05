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


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda cfg: cfg["system"]["nuclei"].update(charges=[2.0]), "charge-one"),
        (lambda cfg: cfg["hamiltonian_terms"].update(electron_electron={}), "only kinetic"),
        (lambda cfg: cfg["hamiltonian_terms"]["kinetic"].update(_target_="wrong.Kinetic"), "-1/2"),
        (lambda cfg: cfg["hamiltonian_terms"]["electron_nucleus"].update(eps=0.1), "raw"),
        (lambda cfg: cfg["model"]["factors"].append({"_target_": "tpen.nn.ElectronElectronCusp"}), "only the nuclear"),
    ],
)
def test_physics_mutations_are_rejected(mutation, message) -> None:
    config = protocol.training_config()
    mutation(config)
    with pytest.raises(ValueError, match=message):
        protocol.assert_hydrogen_contract(config)
