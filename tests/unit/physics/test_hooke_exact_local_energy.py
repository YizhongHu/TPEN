"""Unit tests for the Hamiltonian: exact Hooke local energies are constant.

These exercise ``local_energy`` over the kinetic + harmonic-trap +
electron-electron terms against the analytic Hooke eigenstates, asserting the
local energy is the exact eigenvalue everywhere (and has near-zero variance).
"""

from __future__ import annotations

import math

import pytest
import torch

from tpen.data.batch import ElectronBatch
from tpen.physics.hamiltonian import LocalEnergyResult, local_energy
from tpen.physics.hooke import HookeSingletExact, HookeTripletExact, OneElectronHookeExact
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronElectronInteraction, HarmonicTrap
from tpen.physics.terms import summarize_physical_terms

DTYPE = torch.float64
BATCH_SIZE = 64
ENERGY_ATOL = 1e-5
ENERGY_RTOL = 1e-5
VARIANCE_ATOL = 1e-8


def _hooke_terms(wf) -> dict:
    """Named kinetic + harmonic trap + electron-electron terms for a Hooke pair."""
    return {
        "kinetic": KineticEnergy(),
        "harmonic_trap": HarmonicTrap(omega=wf.omega),
        "electron_electron": ElectronElectronInteraction(),
    }


def _hooke_one_electron_terms(wf) -> dict:
    """Named kinetic + harmonic-trap terms for a one-electron Hooke state."""
    return {"kinetic": KineticEnergy(), "harmonic_trap": HarmonicTrap(omega=wf.omega)}


def _singlet_positions(seed: int = 0) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    pos = torch.empty(BATCH_SIZE, 2, 3, dtype=DTYPE).normal_(generator=g) * 0.8
    # Displace electron 2 along x by 1.5 to guarantee r12 >= 1.5 > min_pair_distance
    pos[:, 1, 0] = pos[:, 0, 0] + 1.5
    return pos


def _triplet_positions(seed: int = 1) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    pos = torch.empty(BATCH_SIZE, 2, 3, dtype=DTYPE).normal_(generator=g) * 0.8
    # Displace electron 2 along x by 1.5 to guarantee r12 >= 1.5 > min_pair_distance
    pos[:, 1, 0] = pos[:, 0, 0] + 1.5
    # Set z2 = z1 + 0.5 to guarantee |z1 - z2| = 0.5 > min_triplet_node_distance
    pos[:, 1, 2] = pos[:, 0, 2] + 0.5
    return pos


def _one_electron_positions(seed: int = 2) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.empty(BATCH_SIZE, 1, 3, dtype=DTYPE).normal_(generator=g) * 0.8


def test_one_electron_local_energy_is_constant_at_three_omega_over_two() -> None:
    wf = OneElectronHookeExact()
    result = local_energy(
        _hooke_one_electron_terms(wf), wf, ElectronBatch(positions=_one_electron_positions()), return_terms=True
    )

    assert isinstance(result, LocalEnergyResult)
    assert torch.allclose(result.total, torch.full_like(result.total, 0.75), atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
    assert torch.allclose(result.total, torch.full_like(result.total, wf.exact_energy), atol=ENERGY_ATOL)


def test_one_electron_gaussian_imperfect_width_is_position_dependent() -> None:
    wf = OneElectronHookeExact(omega=0.4)
    positions = _one_electron_positions()
    result = local_energy(
        {"kinetic": KineticEnergy(), "harmonic_trap": HarmonicTrap(omega=0.5)},
        wf,
        ElectronBatch(positions=positions),
    )

    assert isinstance(result, torch.Tensor)
    assert result.std().item() > 1e-3


def test_one_electron_trap_coefficient_control_is_observable() -> None:
    wf = OneElectronHookeExact()
    batch = ElectronBatch(positions=_one_electron_positions())
    exact = local_energy(_hooke_one_electron_terms(wf), wf, batch)
    wrong_trap = local_energy(
        {"kinetic": KineticEnergy(), "harmonic_trap": HarmonicTrap(omega=wf.omega * 1.1)}, wf, batch
    )

    assert isinstance(exact, torch.Tensor)
    assert isinstance(wrong_trap, torch.Tensor)
    assert not torch.allclose(wrong_trap, exact)


def test_singlet_local_energy_constant_at_exact_energy() -> None:
    wf = HookeSingletExact()
    terms = _hooke_terms(wf)
    positions = _singlet_positions()
    batch = ElectronBatch(positions=positions)

    result = local_energy(terms, wf, batch, return_terms=True)
    assert isinstance(result, LocalEnergyResult)
    eloc = result.total

    assert eloc.shape == (BATCH_SIZE,)
    assert torch.all(torch.isfinite(eloc))
    assert torch.allclose(eloc, torch.full_like(eloc, wf.exact_energy), atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
    assert eloc.std().item() < VARIANCE_ATOL**0.5


def test_singlet_local_energy_variance_near_zero() -> None:
    wf = HookeSingletExact()
    terms = _hooke_terms(wf)
    batch = ElectronBatch(positions=_singlet_positions())

    eloc = local_energy(terms, wf, batch)
    assert isinstance(eloc, torch.Tensor)
    assert eloc.var().item() < VARIANCE_ATOL


def test_exact_singlet_physical_term_summary_reports_virial_diagnostic() -> None:
    wf = HookeSingletExact()
    result = local_energy(_hooke_terms(wf), wf, ElectronBatch(positions=_singlet_positions()), return_terms=True)
    metrics = summarize_physical_terms(result.terms)

    # This diagnostic vanishes only for an exact eigenstate averaged under
    # |psi|^2. Neither this fixed point set nor a finite walker sample provides
    # that exact expectation, so do not assert that it vanishes here.
    assert math.isfinite(metrics["virial_residual"])
    assert math.isfinite(metrics["virial_relative_residual"])


def test_physical_term_summary_matches_virial_formula() -> None:
    metrics = summarize_physical_terms(
        {
            "kinetic": torch.tensor([1.0, 3.0], dtype=DTYPE),
            "harmonic_trap": torch.tensor([2.0, 4.0], dtype=DTYPE),
            "electron_electron": torch.tensor([0.5, 1.5], dtype=DTYPE),
        }
    )

    assert metrics["virial_residual"] == pytest.approx(-1.0)
    assert metrics["virial_relative_residual"] == pytest.approx(1.0 / 11.0)


def test_triplet_local_energy_constant_at_exact_energy() -> None:
    wf = HookeTripletExact()
    terms = _hooke_terms(wf)
    positions = _triplet_positions()
    batch = ElectronBatch(positions=positions)

    result = local_energy(terms, wf, batch, return_terms=True)
    assert isinstance(result, LocalEnergyResult)
    eloc = result.total

    assert eloc.shape == (BATCH_SIZE,)
    assert torch.all(torch.isfinite(eloc))
    assert torch.allclose(eloc, torch.full_like(eloc, wf.exact_energy), atol=ENERGY_ATOL, rtol=ENERGY_RTOL)
    assert eloc.std().item() < VARIANCE_ATOL**0.5


def test_triplet_local_energy_variance_near_zero() -> None:
    wf = HookeTripletExact()
    terms = _hooke_terms(wf)
    batch = ElectronBatch(positions=_triplet_positions())

    eloc = local_energy(terms, wf, batch)
    assert isinstance(eloc, torch.Tensor)
    assert eloc.var().item() < VARIANCE_ATOL


def test_local_energy_term_decomposition_sums_to_total() -> None:
    wf = HookeSingletExact()
    terms = _hooke_terms(wf)
    batch = ElectronBatch(positions=_singlet_positions())

    result = local_energy(terms, wf, batch, return_terms=True)
    assert isinstance(result, LocalEnergyResult)
    assert set(result.terms.keys()) == {"kinetic", "harmonic_trap", "electron_electron"}

    reconstructed = sum(result.terms.values())
    assert torch.allclose(result.total, reconstructed)
