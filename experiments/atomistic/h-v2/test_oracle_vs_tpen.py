"""Independent finite-difference and exponent controls for TPEN's real local-energy path.

These checks exercise the REAL ``tpen.physics.kinetic.KineticEnergy``,
``tpen.physics.potential.ElectronNucleusPotential``, and
``tpen.sampling.metropolis.MetropolisSampler`` code through the frozen H-v2
configs, and compare their output to independently-coded references computed
in this module -- not to ``tpen.physics`` itself and not to a grep over
``tpen/sampling/metropolis.py``. They replace the uncaught kinetic-prefactor
and nuclear-potential-sign mutation controls, and the grep-only sampler-target
control, found by the r1 verification receipt.

Scope note: a genuine closed-form-oracle comparison -- TPEN's wavefunction
pinned, via config alone, to the exact ``psi = exp(-alpha r)`` analytic
hydrogen state used by ``analytic_oracle.py`` -- is out of scope here. No
config-only route exists to make a Hydra-constructed `tpen.nn.TPENWaveFunction`
contribute an exact, input-independent constant from its embedding/readout:
`tpen.nn.embedding.Embedding` only ever builds `tpen.nn.mlp.MLP`/
`tpen.nn.initialization.SeededLinear` layers, whose only initializers
(`TorchInitializer.uniform_`/`normal_`/`xavier_uniform_`/
`linear_kaiming_uniform_`) are random, with no zero/constant scheme; and
`tpen.nn.readout.pfaffian.PfaffianReadout` always folds a learned order-1
feature into the odd-electron border, even with ``trainable=False``. Pinning
the amplitude to a known closed form would need a new core capability (a
null/fixed-amplitude readout, or a zero-initialization scheme) that is out of
scope for this slice; see the H-v2 README.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data.batch import ElectronBatch, Walkers

HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("h_v2_protocol_oracle", HERE / "protocol.py")
assert _spec is not None and _spec.loader is not None
protocol = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = protocol
_spec.loader.exec_module(protocol)


def _hydrogen_model_and_terms(seed: int):
    """Instantiate the frozen H-v2 model and Hamiltonian terms at one seed."""

    torch.manual_seed(seed)
    full = OmegaConf.create(protocol.training_config())
    # Generated MLP/readout weights default to float32; the runner normally
    # casts the model to ``runtime.dtype`` (see PfaffianReadout's docstring).
    # Reproduce that cast here so float64 positions do not hit a dtype
    # mismatch in the embedding's Linear layers.
    model = instantiate(full.model).to(dtype=torch.float64)
    kinetic = instantiate(full.hamiltonian_terms.kinetic)
    potential = instantiate(full.hamiltonian_terms.electron_nucleus)
    return model, kinetic, potential


def _hydrogen_batch(positions: torch.Tensor) -> ElectronBatch:
    spins = torch.ones(positions.shape[0], 1, dtype=positions.dtype)
    return ElectronBatch(positions=positions, spins=spins)


def _finite_difference_kinetic(model, positions: torch.Tensor, h: float = 1e-3) -> torch.Tensor:
    """Return an independently-coded central-difference kinetic local energy.

    Reimplements ``-0.5 * (laplacian(log|psi|) + |grad log|psi||^2)`` from
    scratch with central finite differences instead of autograd, so a wrong
    kinetic prefactor in `tpen.physics.kinetic.KineticEnergy` (e.g. ``-1.0``
    instead of ``-0.5``) disagrees with this reference regardless of the
    embedding/readout weights in play.
    """

    def logabs_at(probe: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(_hydrogen_batch(probe)).logabs

    base = logabs_at(positions)
    n_electrons, spatial_dim = positions.shape[1], positions.shape[2]
    laplacian = torch.zeros(positions.shape[0], dtype=positions.dtype)
    grad_sq = torch.zeros(positions.shape[0], dtype=positions.dtype)
    for electron in range(n_electrons):
        for coordinate in range(spatial_dim):
            plus = positions.clone()
            plus[:, electron, coordinate] += h
            minus = positions.clone()
            minus[:, electron, coordinate] -= h
            f_plus = logabs_at(plus)
            f_minus = logabs_at(minus)
            grad_sq = grad_sq + ((f_plus - f_minus) / (2.0 * h)).square()
            laplacian = laplacian + (f_plus - 2.0 * base + f_minus) / (h * h)
    return -0.5 * (laplacian + grad_sq)


def _raw_coulomb_potential(positions: torch.Tensor, charge: float) -> torch.Tensor:
    """Return an independently-coded raw ``-charge / r`` electron-nucleus potential."""

    distance = positions.reshape(positions.shape[0], -1).norm(dim=-1)
    return -charge / distance


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_kinetic_matches_independent_finite_difference(seed: int) -> None:
    """Catch a wrong kinetic prefactor behaviorally, at several random weight draws."""

    model, kinetic, _ = _hydrogen_model_and_terms(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    positions = torch.rand(4, 1, 3, generator=generator, dtype=torch.float64) * 1.5 + 0.5
    batch = _hydrogen_batch(positions)
    actual = kinetic.local_energy(model, batch).total
    expected = _finite_difference_kinetic(model, positions)
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_potential_matches_raw_coulomb(seed: int) -> None:
    """Catch a sign-flipped (or otherwise wrong) nuclear potential behaviorally."""

    _, _, potential = _hydrogen_model_and_terms(seed)
    generator = torch.Generator().manual_seed(seed + 2000)
    positions = torch.rand(8, 1, 3, generator=generator, dtype=torch.float64) * 2.0 + 0.1
    batch = _hydrogen_batch(positions)
    actual = potential.local_energy(None, batch).total
    charge = float(protocol.training_config()["system"]["nuclei"]["charges"][0])
    expected = _raw_coulomb_potential(positions, charge)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_sampler_accept_exponent_matches_independent_rederivation() -> None:
    """Behaviorally pin the sampler's |psi|^2 (not |psi|) acceptance exponent.

    Replaces the source-text grep `test_sampler_target_is_born_density_not_amplitude`
    relied on: an independent recomputation of the accept-ratio exponent from
    the model's own forward output, using the exact proposal and uniform draws
    the sampler itself consumed (recovered by saving/restoring the sampler's
    public generator state), must equal the sampler's own recorded
    ``aux["log_accept_ratio"]``. Changing the exponent coefficient in
    `tpen.sampling.metropolis.MetropolisSampler.step` from ``2.0`` (|psi|^2)
    to ``1.0`` (|psi|) breaks this equality regardless of what the grep sees.
    """

    model, _, _ = _hydrogen_model_and_terms(seed=7)
    full = OmegaConf.create(protocol.training_config())
    sampler = instantiate(full.sampler)
    walkers = sampler.reset()

    saved_state = sampler._generator.get_state().clone()
    current_logabs, _ = sampler._evaluate(model, walkers)
    proposals, log_q_ratio = sampler._propose(model, walkers)
    proposal_walkers = Walkers(
        positions=proposals,
        spins=walkers.spins,
        atomic_configuration=walkers.atomic_configuration,
        aux=dict(walkers.aux),
    )
    proposed_logabs, _ = sampler._evaluate(model, proposal_walkers)
    expected_log_accept_ratio = torch.nan_to_num(
        2.0 * (proposed_logabs - current_logabs) + log_q_ratio, nan=-torch.inf
    )

    sampler._generator.set_state(saved_state)
    stepped = sampler.step(model, walkers)
    assert torch.equal(stepped.aux["log_accept_ratio"], expected_log_accept_ratio)
