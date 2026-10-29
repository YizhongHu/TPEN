"""Independent scalar oracles for the one-electron study's instruments.

The expected kinetic/potential values use Python scalar mathematics, without
calling either reference forward, learned forwards, or automatic derivatives.
The selected coordinate grids qualify pointwise evaluation only; their
variance is never used as a learned-distribution variance estimate.
"""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest
import torch

from tpen.data.batch import ElectronBatch
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.one_electron import OneElectronExponential, OneElectronGaussian
from tpen.physics.potential import ElectronNucleusInteraction, HarmonicTrap


def _grid(radii: tuple[float, ...]) -> torch.Tensor:
    directions = [tuple(sign if i == axis else 0 for i in range(3))
                  for axis in range(3) for sign in (-1, 1)]
    directions += list(itertools.product((-1, 1), repeat=3))
    coordinates = []
    for radius in radii:
        for direction in directions:
            length = math.sqrt(sum(x*x for x in direction))
            coordinates.append([[radius*x/length for x in direction]])
    return torch.tensor(coordinates, dtype=torch.float64)


def _batch(system: str, positions: torch.Tensor) -> ElectronBatch:
    if system == "hooke":
        return ElectronBatch(positions=positions)
    return ElectronBatch(positions=positions,
                         nuclear_positions=torch.zeros((1, 3), dtype=torch.float64),
                         nuclear_charges=torch.ones(1, dtype=torch.float64))


def _model_and_terms(system: str, scale: float):
    if system == "hooke":
        return OneElectronGaussian(alpha=scale), {
            "kinetic": KineticEnergy(), "harmonic_trap": HarmonicTrap(omega=0.5)}
    return OneElectronExponential(decay=scale), {
        "kinetic": KineticEnergy(), "electron_nucleus": ElectronNucleusInteraction(eps=0.0)}


def _oracle(system: str, scale: float, positions: torch.Tensor):
    kinetic, potential, logabs = [], [], []
    for (point,) in positions.tolist():
        radius_squared = math.fsum(x*x for x in point)
        radius = math.sqrt(radius_squared)
        if system == "hooke":
            kinetic.append(1.5*scale - 0.5*scale**2*radius_squared)
            potential.append(0.125*radius_squared)
            logabs.append(math.log((scale/math.pi)**0.75*math.exp(-scale*radius_squared/2)))
        else:
            kinetic.append(-scale**2/2 + scale/radius)
            potential.append(-1/radius)
            logabs.append(math.log(math.sqrt(scale**3/math.pi)*math.exp(-scale*radius)))
    return tuple(torch.tensor(values, dtype=torch.float64) for values in (kinetic, potential, logabs))


@pytest.mark.parametrize("system,scale", [("hooke", 0.5), ("hooke", 0.6),
                                           ("hydrogen", 1.0), ("hydrogen", 1.2)])
@pytest.mark.parametrize("stress", [False, True])
def test_pointwise_scalar_oracle(tmp_path, system: str, scale: float, stress: bool) -> None:
    radii = (1e-8, 1e-6, 1e-4, 12., 20.) if stress else (1e-3, 1e-2, .1, .5, 1., 2., 4., 8.)
    positions = _grid(radii)
    batch = _batch(system, positions)
    model, terms = _model_and_terms(system, scale)
    result = local_energy(terms, model, batch, return_terms=True)
    assert isinstance(result, LocalEnergyResult)
    kinetic, potential, logabs = _oracle(system, scale, positions)
    budget = torch.maximum(torch.full_like(kinetic, 1e-7), 1e-14*(kinetic.abs()+potential.abs()))
    assert torch.isfinite(result.total).all()
    assert torch.all((result.terms["kinetic"]-kinetic).abs() <= budget)
    potential_name = "harmonic_trap" if system == "hooke" else "electron_nucleus"
    assert torch.all((result.terms[potential_name]-potential).abs() <= budget)
    assert torch.all((result.total-kinetic-potential).abs() <= budget)
    assert torch.allclose(model(batch).logabs, logabs, atol=1e-12, rtol=1e-14)
    (tmp_path / "pointwise-control.json").write_text(json.dumps({
        "system": system, "scale": scale, "stress": stress,
        "maximum_energy_error": float((result.total-kinetic-potential).abs().max()),
        "variance": float(result.total.var(unbiased=False)), "rows": len(positions),
        "oracle": "independent_python_scalar_math", "learned_forward_used": False,
    }, indent=2))
    if not stress and ((system, scale) in (("hooke", .5), ("hydrogen", 1.))):
        # These literal eigenvalues belong to ONE electron, not the pair oracle.
        expected_energy = .75 if system == "hooke" else -.5
        assert (result.total-expected_energy).abs().max().item() <= 1e-7
        assert result.total.var(unbiased=False).item() <= 1e-10


@pytest.mark.parametrize("system,scale", [("hooke", .5), ("hydrogen", 1.)])
def test_scalar_control_detects_wrong_kinetic_prefactor_and_missing_potential(system, scale):
    class WrongKinetic:
        def local_energy(self, wavefunction, batch):
            result = KineticEnergy().local_energy(wavefunction, batch)
            return result.total / 2

    positions = _grid((.1, .5, 1., 2., 4.))
    model, terms = _model_and_terms(system, scale)
    batch = _batch(system, positions)
    kinetic, potential, _ = _oracle(system, scale, positions)
    expected = kinetic + potential
    true_result = local_energy(terms, model, batch, return_terms=True)
    assert torch.allclose(true_result.total, expected, atol=1e-7, rtol=0)
    missing_potential = local_energy({"kinetic": terms["kinetic"]}, model, batch)
    assert (missing_potential-expected).abs().max().item() > .1
    wrong_prefactor = local_energy({**terms, "kinetic": WrongKinetic()}, model, batch)
    assert (wrong_prefactor-expected).abs().max().item() > .1


@pytest.mark.parametrize("system,scale", [("hooke", .6), ("hydrogen", 1.2)])
def test_imperfect_reference_defeats_constant_energy_evaluator(system, scale):
    positions = _grid((.1, .5, 1., 2., 4.))
    kinetic, potential, _ = _oracle(system, scale, positions)
    energy = kinetic + potential
    # A constant ground-energy stub cannot satisfy the independent oracle.
    assert energy.var(unbiased=False).item() > 1e-4
    ground = .75 if system == "hooke" else -.5
    assert (energy-ground).abs().max().item() > .1


@pytest.mark.parametrize("model", [OneElectronGaussian(), OneElectronExponential()])
@pytest.mark.parametrize("shape", [(2, 0, 3), (2, 2, 3), (2, 1, 2)])
def test_references_refuse_pair_or_wrong_dimension(model, shape):
    with pytest.raises(ValueError, match="one-electron references"):
        model(ElectronBatch(positions=torch.ones(shape, dtype=torch.float64)))


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf")])
def test_reference_scales_must_be_normalizable(value):
    with pytest.raises(ValueError, match="finite and positive"):
        OneElectronGaussian(alpha=value)
    with pytest.raises(ValueError, match="finite and positive"):
        OneElectronExponential(decay=value)


@pytest.mark.parametrize("system,scale", [("hooke", .6), ("hydrogen", 1.2)])
def test_imperfect_reference_empirical_moments_against_independent_draws(tmp_path, system, scale):
    rng = np.random.default_rng(64001 if system == "hooke" else 64002)
    sample_count = 16384

    def independent_positions(count):
        if system == "hooke":
            return rng.normal(size=(count, 1, 3)) / math.sqrt(2 * scale)
        radius = -np.log(rng.uniform(size=(count, 3))).sum(axis=-1) / (2 * scale)
        direction = rng.normal(size=(count, 3))
        return (radius[:, None] * direction / np.linalg.norm(direction, axis=-1)[:, None])[:, None, :]

    # Empirical calibration quantiles, not a Gaussian variance error bar.
    moments = []
    for _ in range(1024):
        positions = independent_positions(sample_count)
        r2 = np.square(positions).sum(axis=(1, 2))
        energy = (1.5 * scale + (.25 - scale**2) * r2 / 2 if system == "hooke"
                  else -scale**2 / 2 + (scale - 1) / np.sqrt(r2))
        moments.append((float(energy.mean()), float(energy.var())))
    bands = np.quantile(np.asarray(moments), [.001, .999], axis=0)
    expected_mean = 3 * (scale + .25 / scale) / 4 if system == "hooke" else scale**2 / 2 - scale
    expected_variance = 3 * (.25 - scale**2)**2 / (8 * scale**2) if system == "hooke" else scale**2 * (scale - 1)**2
    (tmp_path / "moment-bands-before-evaluation.json").write_text(json.dumps({
        "bands": bands.tolist(), "population_mean": expected_mean,
        "population_variance": expected_variance, "replicates": 1024, "sample_count": sample_count,
    }, indent=2))
    positions = torch.tensor(independent_positions(sample_count), dtype=torch.float64)
    model, terms = _model_and_terms(system, scale)
    evaluated = torch.cat([local_energy(terms, model, _batch(system, part)).detach()
                           for part in positions.split(4096)]).numpy()
    observed = np.array([evaluated.mean(), evaluated.var()])
    assert np.all(observed >= bands[0]) and np.all(observed <= bands[1])
    assert bands[0, 0] < expected_mean < bands[1, 0]
    assert bands[0, 1] < expected_variance < bands[1, 1]
    (tmp_path / "moment-control.json").write_text(json.dumps({"observed": observed.tolist(), "status": "passed"}))
