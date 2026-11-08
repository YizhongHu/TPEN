"""Unit tests for the exact one-electron hydrogen wavefunction.

These exercise ``local_energy`` over the kinetic + electron-nucleus terms
against the analytic hydrogen ground state (alpha=1, constant at -0.5
hartree) and the known-imperfect alpha=1.2 control (pointwise local energy
``-alpha^2/2 + (alpha-1)/r``, not constant).
"""

from __future__ import annotations

import math

import pytest
import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.physics.hydrogen import HydrogenExact
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronNucleusPotential

DTYPE = torch.float64
BATCH_SIZE = 64
ENERGY_ATOL = 1e-5
ENERGY_RTOL = 1e-5
VARIANCE_ATOL = 1e-8


def _origin_atoms() -> AtomicConfiguration:
    return AtomicConfiguration(
        positions=torch.zeros(1, 3, dtype=DTYPE),
        charges=torch.ones(1, dtype=DTYPE),
    )


def _hydrogen_terms(atoms: AtomicConfiguration) -> dict:
    return {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(atoms=atoms, eps=0.0),
    }


def _positions(seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    pos = torch.empty(BATCH_SIZE, 1, 3, dtype=DTYPE).normal_(generator=g) * 0.8
    # Keep away from the nuclear coalescence point so 1/r is well-conditioned.
    pos[:, 0, 0] += 1.5
    return pos


def test_exact_ground_state_local_energy_is_constant_at_minus_half() -> None:
    atoms = _origin_atoms()
    wf = HydrogenExact(atoms=atoms, alpha=1.0)
    terms = _hydrogen_terms(atoms)
    batch = ElectronBatch(positions=_positions())

    result = local_energy(terms, wf, batch, return_terms=True)
    assert isinstance(result, LocalEnergyResult)
    eloc = result.total

    assert eloc.shape == (BATCH_SIZE,)
    assert torch.all(torch.isfinite(eloc))
    assert torch.allclose(eloc, torch.full_like(eloc, wf.exact_energy), atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
    assert eloc.var().item() < VARIANCE_ATOL


def test_imperfect_state_local_energy_matches_closed_form_and_is_not_constant() -> None:
    atoms = _origin_atoms()
    alpha = 1.2
    wf = HydrogenExact(atoms=atoms, alpha=alpha)
    terms = _hydrogen_terms(atoms)
    positions = _positions(seed=1)
    batch = ElectronBatch(positions=positions)

    eloc = local_energy(terms, wf, batch)
    assert isinstance(eloc, torch.Tensor)

    r = positions[:, 0, :].norm(dim=-1)
    expected = -0.5 * alpha * alpha + (alpha - 1.0) / r
    assert torch.allclose(eloc, expected, atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
    # Not an eigenstate: the local energy varies with r.
    assert eloc.var().item() > VARIANCE_ATOL


def test_exact_energy_is_only_defined_at_alpha_one() -> None:
    atoms = _origin_atoms()
    assert HydrogenExact(atoms=atoms, alpha=1.0).exact_energy == pytest.approx(-0.5)


def test_rejects_non_atomic_configuration() -> None:
    with pytest.raises(TypeError):
        HydrogenExact(atoms="not-an-atomic-configuration")


def test_rejects_more_than_one_nucleus() -> None:
    atoms = AtomicConfiguration(
        positions=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=DTYPE),
        charges=torch.ones(2, dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="exactly one nucleus"):
        HydrogenExact(atoms=atoms)


def test_rejects_non_unit_nuclear_charge() -> None:
    atoms = AtomicConfiguration(
        positions=torch.zeros(1, 3, dtype=DTYPE),
        charges=torch.full((1,), 2.0, dtype=DTYPE),
    )
    with pytest.raises(ValueError, match="charge-one"):
        HydrogenExact(atoms=atoms)


def test_forward_sign_is_always_positive() -> None:
    atoms = _origin_atoms()
    wf = HydrogenExact(atoms=atoms, alpha=1.0)
    batch = ElectronBatch(positions=_positions())
    output = wf(batch)
    assert torch.all(output.sign == 1.0)
    assert torch.all(torch.isfinite(output.logabs))


def test_has_no_trainable_parameters() -> None:
    wf = HydrogenExact(atoms=_origin_atoms(), alpha=1.0)
    assert list(wf.parameters()) == []


def test_displaced_nucleus_local_energy_is_constant_at_minus_half() -> None:
    """Every fixture above puts the nucleus at the origin, so a wavefunction
    that ignored `AtomicConfiguration.positions` would still pass them. Place
    the nucleus away from the origin and require the same exact-ground-state
    result, directly exercising the `atoms.positions` convention."""

    nucleus = torch.tensor([[0.3, -0.2, 0.5]], dtype=DTYPE)
    atoms = AtomicConfiguration(positions=nucleus, charges=torch.ones(1, dtype=DTYPE))
    wf = HydrogenExact(atoms=atoms, alpha=1.0)
    generator = torch.Generator().manual_seed(0)
    relative = torch.empty(BATCH_SIZE, 1, 3, dtype=DTYPE).normal_(generator=generator) * 0.8
    positions = relative + nucleus
    batch = ElectronBatch(positions=positions)
    terms = _hydrogen_terms(atoms)

    eloc = local_energy(terms, wf, batch)
    assert torch.allclose(eloc, torch.full_like(eloc, -0.5), atol=ENERGY_ATOL)
    output = wf(batch)
    assert torch.allclose(output.logabs, -relative[:, 0, :].norm(dim=-1), atol=1e-12, rtol=0.0)
