"""Typed local-energy evaluator seam (LOCAL-ENERGY-ARCHITECTURE.md active slice).

Pins the acceptance criteria of the typed-interface slice: `local_energy` is
an explicit delegate to `NaiveLocalEnergyEvaluator`; the evaluator statically
and dynamically consumes a `NaiveLocalEnergyContext` (never an arbitrary
mapping); term outputs, names, and validation behavior are unchanged; and a
protocol-only custom term unknown to TPEN core evaluates successfully.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from typeguard import TypeCheckError, suppress_type_checks
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.data import AtomicConfiguration
from tpen.data.batch import ElectronBatch, FactorizedLocalEnergyInput, WavefunctionOutput
from tpen.nn import Embedding, TPENWaveFunction
from tpen.nn.cusp import CurvatureElectronNucleusCuspLaw, ElectronNucleusCusp
from tpen.nn.readout import PfaffianReadout
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.potential import ElectronNucleusInteraction, ElectronNucleusPotential
from tpen.physics.potential import HarmonicTrap
from tpen.physics.hamiltonian import (
    AnalyticCuspContext,
    AnalyticCuspEvaluator,
    LocalEnergyResult,
    NaiveLocalEnergyContext,
    NaiveLocalEnergyEvaluator,
    local_energy,
)


_MAPPING_SPEC = importlib.util.spec_from_file_location(
    "he_importance_context_mapping_analytic_hessian",
    Path(__file__).parents[3] / "experiments" / "atomistic" / "he-importance" / "context_mapping.py",
)
assert _MAPPING_SPEC is not None and _MAPPING_SPEC.loader is not None
_MAPPING = importlib.util.module_from_spec(_MAPPING_SPEC)
sys.modules[_MAPPING_SPEC.name] = _MAPPING
_MAPPING_SPEC.loader.exec_module(_MAPPING)

_DTYPE = torch.float64


def _batch(n_walkers: int = 3) -> ElectronBatch:
    generator = torch.Generator().manual_seed(11)
    return ElectronBatch(positions=torch.randn(n_walkers, 2, 3, generator=generator, dtype=_DTYPE))


class ConstantTerm:
    """Protocol-only term unknown to TPEN core."""

    name = "constant"

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        flat = batch.flatten_samples()
        total = torch.full((flat.batch_size,), self.value, dtype=flat.dtype, device=flat.device)
        return LocalEnergyResult(total=total)


class BrokenShapeTerm:
    name = "broken"

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        return LocalEnergyResult(total=torch.zeros((2, 2), dtype=_DTYPE))


def test_local_energy_delegates_to_naive_evaluator() -> None:
    # A position-dependent physics term (HarmonicTrap) makes the equality a
    # real pin rather than a constant-term tautology.
    batch = _batch()
    terms = {"trap": HarmonicTrap(omega=0.5), "shift": ConstantTerm(-0.25)}

    via_entry_point = local_energy(terms, None, batch, return_terms=True)
    via_evaluator = NaiveLocalEnergyEvaluator().evaluate(
        terms, NaiveLocalEnergyContext(wavefunction=None, batch=batch), return_terms=True
    )

    torch.testing.assert_close(via_entry_point.total, via_evaluator.total)
    torch.testing.assert_close(via_entry_point.terms["trap"], via_evaluator.terms["trap"])
    assert list(via_entry_point.terms) == list(via_evaluator.terms) == ["trap", "shift"]
    expected_trap = 0.5 * 0.5**2 * batch.positions.square().sum(dim=(1, 2))
    torch.testing.assert_close(via_entry_point.terms["trap"], expected_trap)


def test_naive_evaluator_rejects_untyped_context() -> None:
    # Acceptance pin: the evaluator does not accept arbitrary mappings or
    # positional (wavefunction, batch) shims — only the typed context. Under
    # the instrumented suite typeguard raises first (TypeCheckError); the
    # evaluator's own TypeError guard covers uninstrumented production runs.
    with pytest.raises((TypeError, TypeCheckError), match="NaiveLocalEnergyContext"):
        NaiveLocalEnergyEvaluator().evaluate(
            {"a": ConstantTerm(1.0)},
            {"wavefunction": None, "batch": _batch()},
        )
    # Pin the evaluator's own guard too — production runs are uninstrumented,
    # so the TypeError branch must stay live without typeguard.
    with suppress_type_checks():
        with pytest.raises(TypeError, match="NaiveLocalEnergyContext"):
            NaiveLocalEnergyEvaluator().evaluate(
                {"a": ConstantTerm(1.0)},
                {"wavefunction": None, "batch": _batch()},
            )


def test_empty_hamiltonian_returns_zero_energy() -> None:
    batch = _batch()
    total = NaiveLocalEnergyEvaluator().evaluate(
        {}, NaiveLocalEnergyContext(wavefunction=None, batch=batch)
    )
    torch.testing.assert_close(total, torch.zeros(3, dtype=_DTYPE))


def test_validation_failures_surface_through_evaluator() -> None:
    context = NaiveLocalEnergyContext(wavefunction=None, batch=_batch())
    with pytest.raises(ValueError, match="must have shape"):
        NaiveLocalEnergyEvaluator().evaluate({"broken": BrokenShapeTerm()}, context)


def test_sequence_terms_keep_snake_case_naming_through_delegation() -> None:
    batch = _batch()
    result = local_energy([ConstantTerm(2.0)], None, batch, return_terms=True)
    assert list(result.terms) == ["constant_term"]
    torch.testing.assert_close(result.terms["constant_term"], torch.full((3,), 2.0, dtype=_DTYPE))


class _AnalyticWavefunction:
    def __init__(self, provider, curvature: float | torch.Tensor = 0.0):
        self.analytic_cusp_provider = provider
        self.curvature = curvature

    def factorized_local_energy_input(self, batch):
        flat = batch.flatten_samples()
        logabs = -self.curvature * flat.positions.square().sum(dim=(1, 2))
        output = WavefunctionOutput(
            logabs=logabs,
            sign=torch.ones_like(logabs),
            aux={"source": "regular"},
        )
        return FactorizedLocalEnergyInput(
            output,
            self.analytic_cusp_provider.analytic_evaluation(flat),
        )


def _analytic_setup(*, z=1.0, nuclear_position=None, eps=0.0, law=None):
    position = (
        torch.zeros((1, 3), dtype=_DTYPE)
        if nuclear_position is None
        else torch.tensor([nuclear_position], dtype=_DTYPE)
    )
    atoms = AtomicConfiguration(position, torch.tensor([z], dtype=_DTYPE))
    provider = ElectronNucleusCusp(atoms, law=law)
    wavefunction = _AnalyticWavefunction(provider)
    batch = ElectronBatch(
        positions=torch.tensor(
            [[[0.4, 0.0, 0.0]], [[0.4, 0.0, 0.0]]],
            dtype=_DTYPE,
        )
    )
    terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(atoms, eps=eps),
    }
    return atoms, wavefunction, batch, terms


def _softplus_fragment_analytic_wavefunction() -> tuple[TPENWaveFunction, AtomicConfiguration]:
    """Build the analytic evaluator model from the production layer fragment."""

    atoms = AtomicConfiguration(
        torch.zeros((1, 3), dtype=_DTYPE), torch.tensor([2.0], dtype=_DTYPE)
    )
    cusp = ElectronNucleusCusp(atoms)
    fragment = _MAPPING.context_fragment(
        "tensor", activation="Softplus beta2,Softplus beta2", channels=1
    )
    config = OmegaConf.create({"layer": fragment})
    OmegaConf.resolve(config)
    layer = instantiate(config.layer)
    model = TPENWaveFunction(
        embedding=Embedding(max_order=2, spatial_dim=3, out_channels=1, hidden_channels=4, num_hidden_layers=1),
        layers=(layer,),
        readout=PfaffianReadout(channels=1),
        factors=(cusp,),
        analytic_cusp_provider=cusp,
        layout=layer.layout,
    ).to(dtype=_DTYPE)
    return model, atoms


def test_analytic_hessian_softplus_fragment_batched_matches_loop() -> None:
    model, atoms = _softplus_fragment_analytic_wavefunction()
    batch = ElectronBatch(
        positions=torch.tensor(
            [
                [[0.4, 0.1, 0.0], [-0.3, 0.2, 0.1]],
                [[0.5, -0.2, 0.1], [-0.2, 0.3, -0.1]],
            ],
            dtype=_DTYPE,
        ),
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
        spins=torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=_DTYPE),
    )
    terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(atoms, eps=0.0),
    }
    context = AnalyticCuspContext(model, batch)
    evaluator = AnalyticCuspEvaluator()
    batched = evaluator.evaluate(terms, context)
    loop = evaluator.evaluate_reference(terms, context)

    assert torch.isfinite(batched).all()
    assert torch.isfinite(loop).all()
    torch.testing.assert_close(batched, loop, rtol=1.0e-11, atol=1.0e-11)


def test_analytic_evaluator_fuses_hydrogen_and_returns_full_output() -> None:
    _, wavefunction, batch, terms = _analytic_setup()
    result = AnalyticCuspEvaluator().evaluate(
        terms, AnalyticCuspContext(wavefunction, batch), return_terms=True
    )

    torch.testing.assert_close(result.total, torch.full((2,), -0.5, dtype=_DTYPE))
    assert list(result.terms) == ["kinetic_plus_electron_nucleus"]
    assert result.wavefunction_output is not None
    torch.testing.assert_close(
        result.wavefunction_output.logabs,
        torch.full((2,), -0.4, dtype=_DTYPE),
    )
    assert result.wavefunction_output.aux == {"source": "regular"}
    assert result.per_electron_kinetic is None


def test_analytic_evaluator_fuses_legacy_electron_nucleus_interaction() -> None:
    law = CurvatureElectronNucleusCuspLaw(
        curvature_coefficient=0.23,
        curvature_range=0.7,
        trainable=False,
        eps=0.0,
    )
    atoms, wavefunction, batch, canonical_terms = _analytic_setup(law=law)
    transported_batch = ElectronBatch(
        positions=batch.positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
    )
    terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusInteraction(eps=0.0),
    }

    evaluator = AnalyticCuspEvaluator()
    canonical = evaluator.evaluate(
        canonical_terms, AnalyticCuspContext(wavefunction, batch)
    )
    result = evaluator.evaluate(
        terms, AnalyticCuspContext(wavefunction, transported_batch), return_terms=True
    )

    evaluation = wavefunction.analytic_cusp_provider.analytic_evaluation(transported_batch)
    cusp_gradient = (
        evaluation.radial_first_derivative.unsqueeze(-1)
        * evaluation.displacement
        / evaluation.distance.unsqueeze(-1)
    ).sum(dim=2)
    expected = -0.5 * cusp_gradient.square().sum(dim=(1, 2))
    expected = expected + evaluation.local_energy_pair().sum(dim=(1, 2))
    torch.testing.assert_close(result.total, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result.total, canonical, rtol=0.0, atol=0.0)
    assert torch.any(result.total != torch.full_like(result.total, -0.5))
    assert list(result.terms) == ["kinetic_plus_electron_nucleus"]


def test_analytic_evaluator_rejects_canonical_geometry_mismatch() -> None:
    atoms, wavefunction, batch, _ = _analytic_setup()
    mismatched = AtomicConfiguration(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=_DTYPE),
        torch.tensor([1.0], dtype=_DTYPE),
    )
    terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(mismatched, eps=0.0),
    }
    transported_batch = ElectronBatch(
        positions=batch.positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
    )

    with pytest.raises(ValueError, match="geometry.*agree exactly"):
        AnalyticCuspEvaluator().evaluate(
            terms, AnalyticCuspContext(wavefunction, transported_batch)
        )


def test_analytic_evaluator_rejects_cross_dtype_geometry_mismatches() -> None:
    base_positions = torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float32)
    base_charges = torch.tensor([1.5], dtype=torch.float32)
    cases = [
        (
            AtomicConfiguration(base_positions, base_charges),
            AtomicConfiguration(
                base_positions.to(torch.float64) + 2**-25,
                base_charges.to(torch.float64),
            ),
        ),
        (
            AtomicConfiguration(
                base_positions.to(torch.float64) + 2**-25,
                base_charges.to(torch.float64),
            ),
            AtomicConfiguration(base_positions, base_charges),
        ),
        (
            AtomicConfiguration(base_positions, base_charges),
            AtomicConfiguration(
                base_positions.to(torch.float64),
                base_charges.to(torch.float64) + 2**-25,
            ),
        ),
        (
            AtomicConfiguration(
                base_positions.to(torch.float64),
                base_charges.to(torch.float64) + 2**-25,
            ),
            AtomicConfiguration(base_positions, base_charges),
        ),
        (
            AtomicConfiguration(
                torch.tensor([[0, 1, 2]], dtype=torch.int64),
                torch.tensor([2], dtype=torch.int64),
            ),
            AtomicConfiguration(
                torch.tensor([[0.25, 1.0, 2.0]], dtype=torch.float64),
                torch.tensor([2.0], dtype=torch.float64),
            ),
        ),
        (
            AtomicConfiguration(
                torch.tensor([[0.25, 1.0, 2.0]], dtype=torch.float64),
                torch.tensor([2.0], dtype=torch.float64),
            ),
            AtomicConfiguration(
                torch.tensor([[0, 1, 2]], dtype=torch.int64),
                torch.tensor([2], dtype=torch.int64),
            ),
        ),
    ]

    batch = ElectronBatch(
        positions=torch.tensor(
            [[[0.4, 0.5, 0.6]], [[0.5, 0.6, 0.7]]], dtype=torch.float64
        )
    )
    for provider_atoms, declared_atoms in cases:
        provider = ElectronNucleusCusp(provider_atoms)
        wavefunction = _AnalyticWavefunction(provider)
        terms = {
            "kinetic": KineticEnergy(),
            "electron_nucleus": ElectronNucleusPotential(declared_atoms, eps=0.0),
        }
        with pytest.raises(ValueError, match="dtype mismatch.*configuration error"):
            AnalyticCuspEvaluator().evaluate(
                terms, AnalyticCuspContext(wavefunction, batch)
            )


def test_analytic_evaluator_rejects_exact_values_with_different_dtypes() -> None:
    provider_atoms = AtomicConfiguration(
        torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float32),
    )
    declared_atoms = AtomicConfiguration(
        provider_atoms.positions.to(torch.float64),
        provider_atoms.charges.to(torch.float64),
    )
    provider = ElectronNucleusCusp(provider_atoms)
    batch = ElectronBatch(
        positions=torch.tensor(
            [[[0.4, 0.5, 0.6]], [[0.5, 0.6, 0.7]]], dtype=torch.float64
        )
    )
    terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(declared_atoms, eps=0.0),
    }

    with pytest.raises(ValueError, match="dtype mismatch.*positions"):
        AnalyticCuspEvaluator().evaluate(
            terms, AnalyticCuspContext(_AnalyticWavefunction(provider), batch)
        )


def test_analytic_evaluator_rejects_legacy_geometry_mismatch() -> None:
    atoms, wavefunction, batch, _ = _analytic_setup()
    mismatched = AtomicConfiguration(
        torch.tensor([[1.0, 0.0, 0.0]], dtype=_DTYPE),
        torch.tensor([1.0], dtype=_DTYPE),
    )
    transported_batch = ElectronBatch(
        positions=batch.positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
    )
    terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusInteraction(
            mismatched.positions, mismatched.charges, eps=0.0
        ),
    }

    with pytest.raises(ValueError, match="geometry.*agree exactly"):
        AnalyticCuspEvaluator().evaluate(
            terms, AnalyticCuspContext(wavefunction, transported_batch)
        )


def test_analytic_evaluator_rejects_duplicate_electron_nucleus_operator_identity() -> None:
    atoms, wavefunction, batch, _ = _analytic_setup()
    transported_batch = ElectronBatch(
        positions=batch.positions,
        nuclear_positions=atoms.positions,
        nuclear_charges=atoms.charges,
    )
    terms = {
        "canonical": ElectronNucleusPotential(atoms, eps=0.0),
        "legacy": ElectronNucleusInteraction(eps=0.0),
        "kinetic": KineticEnergy(),
    }

    with pytest.raises(ValueError, match="electron_nucleus_coulomb"):
        AnalyticCuspEvaluator().evaluate(
            terms, AnalyticCuspContext(wavefunction, transported_batch)
        )


def test_analytic_fast_kernel_matches_reference_in_values_and_parameter_gradients() -> None:
    law = CurvatureElectronNucleusCuspLaw(
        curvature_coefficient=0.23,
        curvature_range=0.7,
        trainable=False,
        eps=0.0,
    )
    _, wavefunction, _, terms = _analytic_setup(law=law)
    batch = ElectronBatch(
        positions=torch.tensor(
            [
                [[0.4, 0.0, 0.0], [0.0, 0.4, 0.0]],
                [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.6, 0.0, 0.0], [0.0, 0.6, 0.0]],
            ],
            dtype=_DTYPE,
        )
    )
    curvature = torch.nn.Parameter(torch.tensor(0.23, dtype=_DTYPE))
    wavefunction.curvature = curvature
    evaluator = AnalyticCuspEvaluator()
    fast = evaluator.evaluate(terms, AnalyticCuspContext(wavefunction, batch), return_terms=True)
    fast_gradient = torch.autograd.grad(fast.total.sum(), curvature, retain_graph=True)[0]
    reference = evaluator.evaluate_reference(
        terms, AnalyticCuspContext(wavefunction, batch), return_terms=True
    )
    reference_gradient = torch.autograd.grad(reference.total.sum(), curvature)[0]

    torch.testing.assert_close(fast.total, reference.total, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(fast_gradient, reference_gradient, rtol=1.0e-12, atol=1.0e-12)
    assert tuple(fast.total.shape) == (3,)
    assert tuple(reference.total.shape) == (3,)


def test_analytic_evaluator_accumulates_terms_before_and_after_fused_group() -> None:
    class CountingTerm:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def local_energy(self, wavefunction, batch):
            self.calls += 1
            size = batch.flatten_samples().batch_size
            return LocalEnergyResult(
                total=torch.full((size,), self.value, dtype=batch.positions.dtype)
            )

    before = CountingTerm(2.25)
    after = CountingTerm(-0.75)
    atoms, wavefunction, batch, _ = _analytic_setup()
    terms = {
        "before": before,
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(atoms, eps=0.0),
        "after": after,
    }
    result = AnalyticCuspEvaluator().evaluate(
        terms, AnalyticCuspContext(wavefunction, batch), return_terms=True
    )

    assert before.calls == after.calls == 1
    assert list(result.terms) == ["before", "kinetic_plus_electron_nucleus", "after"]
    torch.testing.assert_close(result.total, sum(result.terms.values()))


def test_analytic_evaluator_uses_cancelled_slope_residual_at_tiny_radius() -> None:
    law = CurvatureElectronNucleusCuspLaw(
        curvature_coefficient=0.23,
        curvature_range=0.7,
        trainable=False,
        eps=0.0,
    )
    _, wavefunction, _, terms = _analytic_setup(law=law)
    batch = ElectronBatch(
        positions=torch.tensor(
            [[[1.0e-10, 0.0, 0.0]], [[1.0e-10, 0.0, 0.0]]],
            dtype=_DTYPE,
        )
    )
    result = AnalyticCuspEvaluator().evaluate(
        terms, AnalyticCuspContext(wavefunction, batch), return_terms=True
    )
    evaluation = wavefunction.analytic_cusp_provider.analytic_evaluation(batch)
    grad_u = (
        evaluation.radial_first_derivative.unsqueeze(-1)
        * evaluation.displacement
        / evaluation.distance.unsqueeze(-1)
    ).sum(dim=2)
    expected = -0.5 * evaluation.radial_second_derivative - evaluation.slope_residual
    expected = expected.sum(dim=(1, 2)) - 0.5 * grad_u.square().sum(dim=(1, 2))
    torch.testing.assert_close(result.total, expected, rtol=1.0e-12, atol=1.0e-12)


def test_analytic_evaluator_retains_all_nuclear_centres_in_grad_u() -> None:
    atoms = AtomicConfiguration(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=_DTYPE),
        torch.tensor([1.0, 2.0], dtype=_DTYPE),
    )
    provider = ElectronNucleusCusp(atoms)
    wavefunction = _AnalyticWavefunction(provider)
    batch = ElectronBatch(
        positions=torch.tensor(
            [[[0.2, 0.3, 0.0]], [[0.2, 0.3, 0.0]]],
            dtype=_DTYPE,
        )
    )
    terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(atoms, eps=0.0),
    }
    result = AnalyticCuspEvaluator().evaluate(
        terms, AnalyticCuspContext(wavefunction, batch), return_terms=True
    )
    evaluation = provider.analytic_evaluation(batch)
    grad_u = (
        evaluation.radial_first_derivative.unsqueeze(-1)
        * evaluation.displacement
        / evaluation.distance.unsqueeze(-1)
    ).sum(dim=2)
    expected = -0.5 * grad_u.square().sum(dim=(1, 2)) + evaluation.local_energy_pair().sum(dim=(1, 2))
    torch.testing.assert_close(result.total, expected, rtol=1.0e-12, atol=1.0e-12)


def test_hydrogenic_near_cusp_ladder_uses_derived_z_limit() -> None:
    z = 1.7
    _, wavefunction, _, terms = _analytic_setup(z=z)
    radii = torch.tensor([10.0 ** -power for power in range(2, 11)], dtype=_DTYPE)
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=_DTYPE,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)
    positions = (radii[:, None, None] * directions[None, :, :]).reshape(-1, 1, 3)
    batch = ElectronBatch(positions=positions)
    result = AnalyticCuspEvaluator().evaluate(terms, AnalyticCuspContext(wavefunction, batch))
    expected_limit = -(z**2) / 2.0
    torch.testing.assert_close(
        result,
        torch.full_like(result, expected_limit),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    values = result.reshape(len(radii), len(directions))
    spread = values.amax(dim=1) - values.amin(dim=1)
    assert torch.isfinite(spread).all()
