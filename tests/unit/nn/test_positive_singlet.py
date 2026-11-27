"""Tests for the opt-in Hooke-2e positive singlet ansatz."""

from __future__ import annotations

import pytest
import torch

from tpen.data.batch import ElectronBatch
from tpen.data.real import Feature
from tpen.nn import HookeSingletTailFactor, PositiveSingletReadout
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
from tpen.physics.system import ParticleSpec, build_system_descriptor


_DTYPE = torch.float64
_SYSTEM = build_system_descriptor(
    ParticleSpec(n_electrons=2, n_spin_up=1, n_spin_down=1, spatial_dim=3),
    {
        "kinetic": KineticEnergy(),
        "harmonic_trap": HarmonicTrap(omega=0.5),
        "electron_electron": ElectronElectronInteraction(),
    },
)


def _batch(positions: torch.Tensor, *, spins: tuple[float, float] = (1.0, -1.0)) -> ElectronBatch:
    labels = torch.tensor([spins] * positions.shape[0], dtype=positions.dtype, device=positions.device)
    return ElectronBatch(positions=positions, spins=labels)


def _features(batch_size: int = 2, channels: int = 3) -> Feature:
    generator = torch.Generator().manual_seed(17)
    pair = torch.randn(batch_size, channels, 2, 2, generator=generator, dtype=_DTYPE)
    return Feature(
        [
            torch.empty(batch_size, 0, dtype=_DTYPE),
            torch.zeros(batch_size, channels, 2, dtype=_DTYPE),
            pair,
        ]
    )


def test_tail_matches_closed_form_gradient_and_laplacian() -> None:
    positions = torch.tensor([[[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2]]], dtype=_DTYPE, requires_grad=True)
    kappa = 0.7
    value = HookeSingletTailFactor(kappa=kappa, system_descriptor=_SYSTEM)(_batch(positions)).sum()
    gradient = torch.autograd.grad(value, positions, create_graph=True)[0]
    delta = positions[0, 0] - positions[0, 1]
    r2 = delta.square().sum()
    denominator = 1.0 + kappa * r2
    expected_gradient = torch.stack((kappa * delta / denominator, -kappa * delta / denominator)).unsqueeze(0)
    torch.testing.assert_close(gradient, expected_gradient)
    laplacian = torch.zeros((), dtype=_DTYPE)
    for index in range(positions.numel()):
        second = torch.autograd.grad(gradient.reshape(-1)[index], positions, retain_graph=True)[0]
        laplacian = laplacian + second.reshape(-1)[index]
    expected_laplacian = 2.0 * kappa * (3.0 + kappa * r2) / denominator.square()
    torch.testing.assert_close(laplacian, expected_laplacian)


def test_readout_is_exchange_symmetric_and_positive() -> None:
    positions = torch.tensor([[[0.1, 0.2, -0.3], [-0.4, 0.5, 0.6]]], dtype=_DTYPE)
    batch = _batch(positions)
    features = _features(batch_size=1)
    readout = PositiveSingletReadout(channels=3, system_descriptor=_SYSTEM).to(dtype=_DTYPE)
    exchanged = _batch(positions[:, [1, 0]], spins=(-1.0, 1.0))
    exchanged_features = Feature([features[0], features[1], features[2].transpose(-1, -2)])
    first = readout(features, batch)
    second = readout(exchanged_features, exchanged)
    torch.testing.assert_close(first.logabs, second.logabs, rtol=0.0, atol=1.0e-14)
    assert torch.all(first.sign == 1)
    assert torch.isfinite(torch.exp(first.logabs)).all()


def test_tail_is_zero_slope_at_coalescence_and_tiny_kappa_matches_comparator() -> None:
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0e-6, 0.0, 0.0]]], dtype=_DTYPE, requires_grad=True)
    tail = HookeSingletTailFactor(kappa=1.0, system_descriptor=_SYSTEM)
    value = tail(_batch(positions)).sum()
    gradient = torch.autograd.grad(value, positions)[0]
    assert torch.max(gradient.abs()) < 1.0e-6
    tiny = HookeSingletTailFactor(kappa=1.0e-14, system_descriptor=_SYSTEM)(_batch(positions.detach()))
    torch.testing.assert_close(tiny, torch.zeros_like(tiny), atol=1.0e-12, rtol=0.0)


@pytest.mark.parametrize("kappa", [0.0, -1.0])
def test_tail_rejects_nonpositive_kappa(kappa: float) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        HookeSingletTailFactor(kappa=kappa, system_descriptor=_SYSTEM)


def test_ansatz_rejects_non_hooke_and_triplet_sectors() -> None:
    with pytest.raises(ValueError, match="configured SystemDescriptor"):
        PositiveSingletReadout(channels=1, system_descriptor=object())
    readout = PositiveSingletReadout(channels=1, system_descriptor=_SYSTEM)
    features = _features(batch_size=1, channels=1)
    positions = torch.zeros(1, 2, 3, dtype=_DTYPE)
    with pytest.raises(ValueError, match="one up and one down"):
        readout(features, _batch(positions, spins=(1.0, 1.0)))
    with pytest.raises(ValueError, match="exactly two"):
        readout(
            Feature(
                [
                    torch.empty(1, 0, dtype=_DTYPE),
                    torch.zeros(1, 1, 1, dtype=_DTYPE),
                    torch.zeros(1, 1, 1, 1, dtype=_DTYPE),
                ]
            ),
            ElectronBatch(torch.zeros(1, 1, 3, dtype=_DTYPE), spins=torch.ones(1, 1, dtype=_DTYPE)),
        )
