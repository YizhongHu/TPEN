"""Independent reference machinery for the H-v2 charge-one hydrogen contract.

These helpers are deliberately independent of ``tpen.physics``/``tpen.sampling``
internals: a smooth hand-coded wavefunction for finite-difference checks, a
closed-form ``psi = exp(-alpha r)`` state, and standard-library closed-form
hydrogen formulas. They let core tests behaviorally pin TPEN's real kinetic,
potential, and sampler code against a charge-one, one-electron hydrogen setup
without reading any experiment configuration file.

The frozen-contract numeric literals below (charge, sampler settings) are
intentionally duplicated from ``experiments/atomistic/h-v2/configs/train.yaml``
rather than read from it, so these tests never import or read a file under
``experiments/``. ``experiments/atomistic/h-v2/test_protocol.py`` separately
pins the frozen config to the same literals.
"""

from __future__ import annotations

import math
import random

import torch
from torch import nn

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, WavefunctionOutput

#: Frozen H-v2 contract literals (see module docstring).
HYDROGEN_CHARGE = 1.0
SAMPLER_SETTINGS: dict[str, float | int] = {
    "n_walkers": 256,
    "burn_in": 100,
    "n_steps": 10,
    "proposal_scale": 0.5,
    "seed": 910002,
}


def hydrogen_atoms(dtype: torch.dtype = torch.float64) -> AtomicConfiguration:
    """Return the frozen charge-one nucleus at the origin."""

    return AtomicConfiguration(
        positions=torch.zeros(1, 3, dtype=dtype),
        charges=torch.full((1,), HYDROGEN_CHARGE, dtype=dtype),
    )


def hydrogen_batch(positions: torch.Tensor) -> ElectronBatch:
    """Wrap one-electron positions ``[batch, 1, 3]`` into an ``ElectronBatch``."""

    spins = torch.ones(positions.shape[0], 1, dtype=positions.dtype)
    return ElectronBatch(positions=positions, spins=spins)


class RandomSmoothWaveFunction(nn.Module):
    """A seeded, infinitely-differentiable stand-in wavefunction.

    Not the trained H-v2 architecture: a small ``tanh`` network over one
    electron's Cartesian position, used only to give
    ``tpen.physics.kinetic.KineticEnergy`` a real, nontrivial autograd path to
    differentiate at several independent random weight draws, so a wrong
    kinetic prefactor is caught regardless of the particular smooth function
    in play.
    """

    def __init__(self, seed: int, hidden: int = 16) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(0.3 * torch.randn(3, hidden, generator=generator, dtype=torch.float64))
        self.b1 = nn.Parameter(0.1 * torch.randn(hidden, generator=generator, dtype=torch.float64))
        self.w2 = nn.Parameter(0.3 * torch.randn(hidden, generator=generator, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        positions = batch.positions.reshape(batch.positions.shape[0], -1)
        hidden = torch.tanh(positions.to(self.w1.dtype) @ self.w1 + self.b1)
        logabs = hidden @ self.w2
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class ExactHydrogenState(nn.Module):
    """The closed-form ``psi = exp(-alpha r)`` amplitude as a wavefunction module."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        positions = batch.positions
        r = positions.reshape(positions.shape[0], -1).norm(dim=-1)
        return WavefunctionOutput(logabs=-self.alpha * r, sign=torch.ones_like(r))


def independent_central_difference_kinetic(model: nn.Module, positions: torch.Tensor, h: float = 1e-3) -> torch.Tensor:
    """Return an independently-coded central-difference kinetic local energy.

    Reimplements ``-0.5 * (laplacian(log|psi|) + |grad log|psi||^2)`` from
    scratch with central finite differences instead of autograd, so a wrong
    kinetic prefactor in ``tpen.physics.kinetic.KineticEnergy`` disagrees with
    this reference regardless of the model's weights.
    """

    def logabs_at(probe: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(hydrogen_batch(probe)).logabs

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


def raw_coulomb_potential(positions: torch.Tensor, charge: float) -> torch.Tensor:
    """Return an independently-coded raw ``-charge / r`` electron-nucleus potential."""

    distance = positions.reshape(positions.shape[0], -1).norm(dim=-1)
    return -charge / distance


def closed_form_local_energy(radius: float, alpha: float = 1.0, charge: float = HYDROGEN_CHARGE) -> float:
    """Return the scalar local energy for ``psi=exp(-alpha*r)`` and ``-charge/r``."""

    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("hydrogen local energy requires a finite radius > 0")
    return -0.5 * alpha * alpha + (alpha - charge) / radius


def gamma3_cdf(radius: float, alpha: float = 1.0) -> float:
    """Return the Born radial CDF for ``Gamma(shape=3, rate=2*alpha)``."""

    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and nonnegative")
    t = 2.0 * alpha * radius
    return 1.0 - math.exp(-t) * (1.0 + t + 0.5 * t * t)


def sample_born_radii(count: int, seed: int) -> list[float]:
    """Draw independent Born radii as sums of three rate-two exponentials."""

    rng = random.Random(seed)
    return [-math.log(rng.random()) / 2.0 - math.log(rng.random()) / 2.0 - math.log(rng.random()) / 2.0 for _ in range(count)]


__all__ = [
    "HYDROGEN_CHARGE",
    "SAMPLER_SETTINGS",
    "ExactHydrogenState",
    "RandomSmoothWaveFunction",
    "closed_form_local_energy",
    "gamma3_cdf",
    "hydrogen_atoms",
    "hydrogen_batch",
    "independent_central_difference_kinetic",
    "raw_coulomb_potential",
    "sample_born_radii",
]
