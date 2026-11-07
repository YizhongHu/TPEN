"""Behavioral hydrogen-oracle controls for ``KineticEnergy``/``ElectronNucleusPotential``.

Moved here from ``experiments/atomistic/h-v2/test_oracle_vs_tpen.py`` and
``test_stub_oracle_contract.py`` (contract amendment 2026-11-06): these checks
reach ``tpen.physics`` internals directly, which the experiment boundary
restricts to configuration/``run_from_config``/the ``tpen`` command. The
checks themselves, and the mutants they catch, are unchanged -- only the
charge-one hydrogen setup is now built inline with literals (see
``tests/helpers/hydrogen_contract.py``) instead of via the frozen Hydra
config. This module does not duplicate
``tests/unit/physics/test_finite_difference_kinetic.py`` (the optional
Numdifftools Hessdiag oracle, a different differentiation method gated behind
the ``finite-difference`` extra) or ``tests/unit/physics/test_atomic_potential.py``
(hand-calculated potential algebra unrelated to this contract's mutant
controls).
"""

from __future__ import annotations

import pytest
import torch

from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronNucleusPotential

from tests.helpers.hydrogen_contract import (
    HYDROGEN_CHARGE,
    ExactHydrogenState,
    RandomSmoothWaveFunction,
    closed_form_local_energy,
    hydrogen_atoms,
    hydrogen_batch,
    independent_central_difference_kinetic,
    raw_coulomb_potential,
    sample_born_radii,
)

#: Richardson-extrapolation tolerance for
#: `test_kinetic_finite_difference_richardson_headroom`. Measured on Cannon
#: (job 58571622) for `RandomSmoothWaveFunction` at seeds 1-3:
#: 2.0584259644018665e-09, 3.3302093530723775e-09, 3.81966344664697e-09 --
#: about 260x headroom here. The real-architecture arm
#: (`tests/unit/physics/test_review_amend_r1_hydrogen_real_model_kinetic.py`)
#: measured 6.6e-8 to 8.0e-8 at the same bound, about 12x headroom, so one
#: constant serves both. The extrapolated discrepancy is recorded via
#: `record_property` so a future tightening is a measured change, not a guess.
RICHARDSON_TOLERANCE = 1e-6


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_kinetic_matches_independent_finite_difference(seed: int) -> None:
    """Catch a wrong kinetic prefactor behaviorally, at several random weight draws."""

    model = RandomSmoothWaveFunction(seed)
    kinetic = KineticEnergy()
    generator = torch.Generator().manual_seed(seed + 1000)
    positions = torch.rand(4, 1, 3, generator=generator, dtype=torch.float64) * 1.5 + 0.5
    batch = hydrogen_batch(positions)
    actual = kinetic.local_energy(model, batch).total
    expected = independent_central_difference_kinetic(model, positions)
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_potential_matches_raw_coulomb(seed: int) -> None:
    """Catch a sign-flipped (or otherwise wrong) nuclear potential behaviorally."""

    potential = ElectronNucleusPotential(hydrogen_atoms())
    generator = torch.Generator().manual_seed(seed + 2000)
    positions = torch.rand(8, 1, 3, generator=generator, dtype=torch.float64) * 2.0 + 0.1
    batch = hydrogen_batch(positions)
    actual = potential.local_energy(None, batch).total
    expected = raw_coulomb_potential(positions, HYDROGEN_CHARGE)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_kinetic_finite_difference_richardson_headroom(seed: int, record_property) -> None:
    """Richardson-extrapolate the finite-difference kinetic check's own two step sizes."""

    model = RandomSmoothWaveFunction(seed)
    kinetic = KineticEnergy()
    generator = torch.Generator().manual_seed(seed + 5000)
    positions = torch.rand(16, 1, 3, generator=generator, dtype=torch.float64) * 2.8 + 0.2
    actual = kinetic.local_energy(model, hydrogen_batch(positions)).total
    d_h = independent_central_difference_kinetic(model, positions, h=1e-3)
    d_half_h = independent_central_difference_kinetic(model, positions, h=5e-4)
    richardson = (4.0 * d_half_h - d_h) / 3.0
    discrepancy = float((actual - richardson).abs().max())
    record_property(f"seed_{seed}_richardson_discrepancy", discrepancy)
    print(f"seed={seed} richardson_discrepancy={discrepancy}")
    assert discrepancy <= RICHARDSON_TOLERANCE


@pytest.mark.parametrize("alpha", [1.0, 1.2])
def test_stub_exact_state_pointwise_local_energy_matches_oracle(alpha: float, record_property) -> None:
    """Pin TPEN's real kinetic+potential code against the closed-form oracle."""

    kinetic = KineticEnergy()
    potential = ElectronNucleusPotential(hydrogen_atoms())
    generator = torch.Generator().manual_seed(910100 + int(alpha * 10))
    directions = torch.randn(8, 3, generator=generator, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    ordinary = torch.tensor([0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    stress = torch.tensor([1e-8, 1e-6, 1e-4], dtype=torch.float64)
    model = ExactHydrogenState(alpha).double()
    errors = []
    for radii in (ordinary, stress):
        batch = hydrogen_batch((radii[:, None] * directions[: radii.numel()])[:, None, :])
        k = kinetic.local_energy(model, batch).total
        v = potential.local_energy(None, batch).total
        expected = torch.tensor([closed_form_local_energy(float(r), alpha) for r in radii], dtype=torch.float64)
        error = (k + v - expected).abs()
        errors.extend(error.tolist())
        if radii is stress:
            k_oracle = -0.5 * alpha * alpha + alpha / radii
            v_oracle = -1.0 / radii
            tolerance = torch.maximum(
                torch.full_like(radii, 1e-9), 128 * torch.finfo(torch.float64).eps * (k_oracle.abs() + v_oracle.abs())
            )
            assert torch.all(error <= tolerance)
        else:
            assert torch.allclose(k + v, expected, atol=1e-8, rtol=0.0)
    record_property("max_local_energy_error", max(errors))
    print(f"alpha={alpha} max_local_energy_error={max(errors)}")


def test_stub_imperfect_state_is_not_reported_exact(record_property) -> None:
    """A non-eigenstate's local energy must show real dispersion, not a constant."""

    kinetic = KineticEnergy()
    potential = ElectronNucleusPotential(hydrogen_atoms())
    radii = torch.tensor(sample_born_radii(20000, 910004), dtype=torch.float64) / 1.2
    generator = torch.Generator().manual_seed(910005)
    directions = torch.randn(20000, 3, generator=generator, dtype=torch.float64)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    batch = (radii[:, None] * directions)[:, None, :]
    model_1_2 = ExactHydrogenState(1.2).double()
    energies = kinetic.local_energy(model_1_2, hydrogen_batch(batch)).total
    energies = energies + potential.local_energy(None, hydrogen_batch(batch)).total
    mean = float(energies.mean())
    variance = float(energies.var(unbiased=True))
    record_property("alpha_1_2_mean", mean)
    record_property("alpha_1_2_variance", variance)
    print(f"alpha=1.2 mean={mean} variance={variance}")
    assert abs(mean + 0.48) <= 0.01
    assert abs(variance - 0.0576) <= 0.1 * 0.0576
    exact_radii = torch.tensor([0.001, 0.1, 1.0, 3.0], dtype=torch.float64)
    exact_batch = hydrogen_batch((exact_radii[:, None] * directions[:4])[:, None, :])
    model_1_0 = ExactHydrogenState(1.0).double()
    exact = kinetic.local_energy(model_1_0, exact_batch).total + potential.local_energy(None, exact_batch).total
    assert torch.allclose(exact, torch.full_like(exact, -0.5), atol=1e-9, rtol=0.0)
