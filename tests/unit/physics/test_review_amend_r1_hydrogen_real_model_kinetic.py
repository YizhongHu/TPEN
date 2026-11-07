"""Kinetic oracle control through the real H-v2 TPEN architecture."""

from __future__ import annotations

import torch
import pytest

from tpen.nn import (
    ElectronNucleusCusp,
    Embedding,
    EquivariantMixing,
    PathAggregation,
    ResidualUpdater,
    TPENLayer,
    TPENWaveFunction,
)
from tpen.nn.readout.pfaffian import PfaffianReadout
from tpen.physics.kinetic import KineticEnergy

from tests.helpers.hydrogen_contract import hydrogen_atoms, hydrogen_batch, independent_central_difference_kinetic


def _model() -> TPENWaveFunction:
    """Build the one-electron model from the frozen YAML component literals."""

    embedding = Embedding(max_order=2, spatial_dim=3, out_channels=8, hidden_channels=32, num_hidden_layers=1, include_spins=True)
    layer = TPENLayer(
        mixing=EquivariantMixing(max_order=2, max_virtual_order=2, channels=8, implementation="vectorized", initial_weight=0.5, activation=torch.nn.SiLU()),
        path_aggregation=PathAggregation(max_order=2, max_virtual_order=2, channels=8, activation=torch.nn.SiLU()),
        update=ResidualUpdater(),
    )
    return TPENWaveFunction(
        embedding=embedding,
        layers=[layer],
        factors=[ElectronNucleusCusp(atoms=hydrogen_atoms())],
        readout=PfaffianReadout(channels=8, trainable=True),
        trace_name="tpen",
    ).to(dtype=torch.float64)


def _check_real_architecture_kinetic(seed: int, record_property) -> None:
    """Differentiate the real architecture and compare two finite differences."""

    torch.manual_seed(seed)
    model = _model()
    generator = torch.Generator().manual_seed(seed + 1000)
    positions = torch.rand(4, 1, 3, generator=generator, dtype=torch.float64) * 1.5 + 0.5
    actual = KineticEnergy().local_energy(model, hydrogen_batch(positions)).total
    expected = independent_central_difference_kinetic(model, positions)
    d_h = independent_central_difference_kinetic(model, positions, h=1e-3)
    d_half = independent_central_difference_kinetic(model, positions, h=5e-4)
    discrepancy = float((actual - (4.0 * d_half - d_h) / 3.0).abs().max())
    record_property(f"seed_{seed}_richardson_discrepancy", discrepancy)
    print(f"seed={seed} richardson_discrepancy={discrepancy}")
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_real_architecture_kinetic_for_seed(seed: int, record_property) -> None:
    """Run the real-architecture oracle for each requested initialization."""

    _check_real_architecture_kinetic(seed, record_property)
