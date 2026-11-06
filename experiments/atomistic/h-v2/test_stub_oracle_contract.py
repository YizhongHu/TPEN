"""Closed-form hydrogen controls through TPEN's real operators and sampler."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import ElectronBatch, WavefunctionOutput

HERE = Path(__file__).parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load("h_v2_stub_protocol", "protocol.py")
oracle = _load("h_v2_stub_oracle", "analytic_oracle.py")


class _ExactHydrogenState(torch.nn.Module):
    """psi = exp(-alpha r): the closed-form state analytic_oracle.py describes."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, batch):
        positions = batch.positions
        r = positions.reshape(positions.shape[0], -1).norm(dim=-1)
        return WavefunctionOutput(logabs=-self.alpha * r, sign=torch.ones_like(r))


def _terms_and_sampler():
    full = OmegaConf.create(protocol.training_config())
    kinetic = instantiate(full.hamiltonian_terms.kinetic)
    potential = instantiate(full.hamiltonian_terms.electron_nucleus)
    sampler = instantiate(full.sampler)
    return kinetic, potential, sampler


def _batch_from_radii(radii: torch.Tensor, directions: torch.Tensor) -> ElectronBatch:
    return ElectronBatch(positions=(radii[:, None] * directions)[:, None, :], spins=torch.ones(radii.shape[0], 1, dtype=radii.dtype))


@pytest.mark.parametrize("alpha", [1.0, 1.2])
def test_stub_exact_state_pointwise_local_energy_matches_oracle(alpha: float, record_property) -> None:
    kinetic, potential, _ = _terms_and_sampler()
    generator = torch.Generator().manual_seed(910100 + int(alpha * 10))
    directions = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    ordinary = torch.tensor([0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    stress = torch.tensor([1e-8, 1e-6, 1e-4], dtype=torch.float64)
    model = _ExactHydrogenState(alpha).double()
    errors = []
    for radii in (ordinary, stress):
        batch = _batch_from_radii(radii, directions[: radii.numel()])
        k = kinetic.local_energy(model, batch).total
        v = potential.local_energy(None, batch).total
        expected = torch.tensor([oracle.local_energy(float(r), alpha) for r in radii], dtype=torch.float64)
        error = (k + v - expected).abs()
        errors.extend(error.tolist())
        if radii is stress:
            k_oracle = -0.5 * alpha * alpha + alpha / radii
            v_oracle = -1.0 / radii
            tolerance = torch.maximum(torch.full_like(radii, 1e-9), 128 * torch.finfo(torch.float64).eps * (k_oracle.abs() + v_oracle.abs()))
            assert torch.all(error <= tolerance)
        else:
            assert torch.allclose(k + v, expected, atol=1e-8, rtol=0.0)
    record_property("max_local_energy_error", max(errors))
    print(f"alpha={alpha} max_local_energy_error={max(errors)}")


def test_stub_imperfect_state_is_not_reported_exact(record_property) -> None:
    kinetic, potential, _ = _terms_and_sampler()
    radii = torch.tensor(oracle.sample_born_radii(20000, 910004), dtype=torch.float64) / 1.2
    generator = torch.Generator().manual_seed(910005)
    directions = torch.randn(20000, 3, generator=generator, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    energies = kinetic.local_energy(_ExactHydrogenState(1.2).double(), _batch_from_radii(radii, directions)).total
    energies = energies + potential.local_energy(None, _batch_from_radii(radii, directions)).total
    mean = float(energies.mean())
    variance = float(energies.var(unbiased=True))
    record_property("alpha_1_2_mean", mean)
    record_property("alpha_1_2_variance", variance)
    print(f"alpha=1.2 mean={mean} variance={variance}")
    assert abs(mean + 0.48) <= 0.01
    assert abs(variance - 0.0576) <= 0.1 * 0.0576
    exact_radii = torch.tensor([0.001, 0.1, 1.0, 3.0], dtype=torch.float64)
    exact_dirs = directions[:4]
    exact_batch = _batch_from_radii(exact_radii, exact_dirs)
    exact = kinetic.local_energy(_ExactHydrogenState(1.0).double(), exact_batch).total + potential.local_energy(None, exact_batch).total
    assert torch.allclose(exact, torch.full_like(exact, -0.5), atol=1e-9, rtol=0.0)


def test_real_sampler_radii_match_gamma32_from_narrow_and_diffuse_starts(record_property) -> None:
    """The predeclared empirical-CDF absolute-difference band is 0.02."""

    kinetic, potential, _ = _terms_and_sampler()
    del kinetic, potential
    model = _ExactHydrogenState(1.0).double()
    radii_at = (0.5, 1.0, 1.5, 2.0, 3.0)
    for scale in (0.1, 5.0):
        full = OmegaConf.create(protocol.training_config())
        sampler = instantiate(full.sampler, initial_scale=scale)
        walkers = sampler.reset()
        for _ in range(1000):
            walkers = sampler.step(model, walkers)
        retained = []
        accepted = []
        for _ in range(200):
            for _ in range(20):
                walkers = sampler.step(model, walkers)
                accepted.append(float(walkers.aux["accepted"].double().mean()))
            retained.append(walkers.positions.reshape(walkers.positions.shape[0], -1).norm(dim=-1))
        radii = torch.cat(retained)
        differences = [abs(float((radii <= r).double().mean()) - oracle.gamma3_cdf(r)) for r in radii_at]
        record_property(f"scale_{scale}_cdf_differences", differences)
        record_property(f"scale_{scale}_acceptance_rate", sum(accepted) / len(accepted))
        print(f"scale={scale} cdf_differences={differences} acceptance_rate={sum(accepted) / len(accepted)}")
        assert max(differences) < 0.02
