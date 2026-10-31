"""Physics and local-energy sanity tests."""

from __future__ import annotations

import gc
import math
import weakref

import pytest
import torch
from torch import nn

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.permutation import Permutation
from tpen.nn import ElectronElectronCusp
from tpen.physics.hamiltonian import (
    LocalEnergyResult,
    evaluate_local_energy_in_chunks,
    local_energy,
    normalize_hamiltonian_terms,
    summarize_local_energy,
)
from tpen.physics.hooke import HookeSingletExact, HookeTripletExact
from tpen.physics.kinetic import KineticEnergy, kinetic_energy_from_logabs
from tpen.physics.potential import (
    ElectronElectronInteraction,
    ElectronNucleusInteraction,
    HarmonicTrap,
    NucleusNucleusInteraction,
)


class GaussianOutputModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float64)

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = -self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class TrainableGaussianOutputModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float64))

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = -self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class CuspGaussianOutputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self.cusp = ElectronElectronCusp(
            same_spin_coefficient=0.25,
            opposite_spin_coefficient=0.5,
            range_parameter=0.5,
            eps=1.0e-12,
        )

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = self.cusp(batch) - self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


class _ConstantTerm:
    """Hamiltonian term returning fixed values, for helper tests."""

    def __init__(self, name: str, value: torch.Tensor) -> None:
        self.name = name
        self._value = value

    def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
        return LocalEnergyResult(total=self._value, terms={self.name: self._value})


class _GraphLifetimeEvaluator:
    """Return graph-bearing values and observe their release between chunks."""

    def __init__(self) -> None:
        self._prior_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        self.calls = 0

    def validate_for_generator(self, terms, wavefunction, generator):
        del terms, wavefunction, generator

    def make_context(self, wavefunction, batch):
        return batch

    def evaluate(self, terms, context, *, return_terms=False):
        assert all(reference() is None for reference in self._prior_refs)
        self.calls += 1
        total = context.positions.square().sum(dim=(1, 2))
        self._prior_refs = [weakref.ref(total)]
        if not return_terms:
            return total
        term = total * 2.0
        logabs = total + 1.0
        sign = torch.ones_like(logabs)
        per_electron = total.unsqueeze(-1)
        self._prior_refs.extend(
            weakref.ref(value)
            for value in (term, logabs, sign, per_electron)
        )
        return LocalEnergyResult(
            total=total,
            terms={"energy": term},
            wavefunction_output=WavefunctionOutput(logabs=logabs, sign=sign),
            per_electron_kinetic=per_electron,
        )


def test_potential_terms_match_direct_hand_calculations() -> None:
    positions = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[0.5, -1.0], [2.5, 1.0]],
        ],
        dtype=torch.float64,
    )
    nuclei = torch.tensor([[0.0, -1.0], [2.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([2.0, 0.5], dtype=torch.float64)
    batch = ElectronBatch(positions=positions, nuclear_positions=nuclei, nuclear_charges=charges)

    harmonic = HarmonicTrap(omega=1.5).local_energy(None, batch).total
    repulsion = ElectronElectronInteraction().local_energy(None, batch).total
    attraction = ElectronNucleusInteraction(nuclei, charges).local_energy(None, batch).total

    expected_harmonic = 0.5 * (1.5**2) * positions.square().sum(dim=(1, 2))
    expected_repulsion = torch.linalg.norm(positions[:, 0] - positions[:, 1], dim=-1).reciprocal()
    expected_attraction = -(
        charges.view(1, 1, -1)
        / torch.linalg.norm(positions.unsqueeze(2) - nuclei.view(1, 1, 2, 2), dim=-1)
    ).sum(dim=(1, 2))

    assert torch.allclose(harmonic, expected_harmonic)
    assert torch.allclose(repulsion, expected_repulsion)
    assert torch.allclose(attraction, expected_attraction)


def test_autograd_kinetic_matches_gaussian_logabs_formula() -> None:
    positions = torch.tensor(
        [
            [[1.0, 2.0], [0.5, -1.0]],
            [[-1.5, 0.25], [2.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    alpha = 0.3
    batch = ElectronBatch(positions=positions)

    kinetic = kinetic_energy_from_logabs(GaussianOutputModel(alpha), batch)

    n_electrons = positions.shape[1]
    spatial_dim = positions.shape[2]
    expected = alpha * n_electrons * spatial_dim - 2.0 * alpha**2 * positions.square().sum(dim=(1, 2))
    assert torch.allclose(kinetic, expected)


def test_zero_electron_kinetic_preserves_empty_attribution_shape() -> None:
    batch = ElectronBatch(positions=torch.empty(3, 0, 2, dtype=torch.float64))

    result = local_energy([KineticEnergy()], GaussianOutputModel(alpha=0.3), batch, return_terms=True)

    assert isinstance(result, LocalEnergyResult)
    assert result.total.shape == (3,)
    assert torch.equal(result.total, torch.zeros(3, dtype=torch.float64))
    assert result.per_electron_kinetic is not None
    assert result.per_electron_kinetic.shape == (3, 0)


def test_harmonic_oscillator_ground_state_has_constant_local_energy() -> None:
    omega = 1.7
    positions = torch.tensor(
        [
            [[1.0, 0.0, -1.0], [0.5, 2.0, 0.25]],
            [[-0.5, 1.5, 2.5], [1.0, -2.0, 0.5]],
        ],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)
    terms = [KineticEnergy(), HarmonicTrap(omega=omega)]

    energy = local_energy(terms, GaussianOutputModel(alpha=omega / 2.0), batch)

    n_electrons, spatial_dim = positions.shape[1], positions.shape[2]
    expected = torch.full((2,), n_electrons * spatial_dim * omega / 2.0, dtype=torch.float64)
    assert torch.allclose(energy, expected)


def test_local_energy_accepts_wavefunction_output_and_preserves_parameter_gradients() -> None:
    batch = ElectronBatch(
        positions=torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[-1.0, 1.0], [2.0, -0.5]]], dtype=torch.float64),
    )
    model = TrainableGaussianOutputModel(alpha=0.25)
    terms = [KineticEnergy(), HarmonicTrap(omega=1.0)]

    energy = local_energy(terms, model, batch)
    energy.mean().backward()

    assert energy.shape == (2,)
    assert torch.all(torch.isfinite(energy))
    assert model.alpha.grad is not None
    assert torch.isfinite(model.alpha.grad)


def test_cusp_local_energy_has_finite_second_derivatives_with_pair_diagonal() -> None:
    batch = ElectronBatch(
        positions=torch.tensor([[[0.25, -0.1, 0.3], [-0.35, 0.4, -0.2]]], dtype=torch.float64),
        spins=torch.tensor([[1.0, 1.0]], dtype=torch.float64),
    )
    model = CuspGaussianOutputModel()
    terms = [KineticEnergy(), HarmonicTrap(omega=0.5), ElectronElectronInteraction()]

    energy = local_energy(terms, model, batch)
    energy.mean().backward()

    assert energy.shape == (1,)
    assert torch.all(torch.isfinite(energy))
    assert model.alpha.grad is not None
    assert torch.isfinite(model.alpha.grad)


@pytest.mark.parametrize("return_terms", [False, True])
def test_chunked_local_energy_releases_previous_graph_before_next_chunk(
    return_terms: bool,
) -> None:
    positions = torch.arange(12, dtype=torch.float64).reshape(4, 1, 3).requires_grad_()
    batch = ElectronBatch(positions=positions)
    evaluator = _GraphLifetimeEvaluator()

    result = evaluate_local_energy_in_chunks(
        {},
        None,
        batch,
        return_terms=return_terms,
        chunk_size=2,
        evaluator=evaluator,
    )
    gc.collect()

    assert evaluator.calls == 2
    assert all(reference() is None for reference in evaluator._prior_refs)
    if return_terms:
        assert isinstance(result, LocalEnergyResult)
    else:
        assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("chunk_size", [1, 2, 5, 16])
def test_chunked_local_energy_preserves_ordered_structured_values(chunk_size: int) -> None:
    batch = ElectronBatch(
        positions=torch.tensor(
            [
                [[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2]],
                [[0.3, 0.2, -0.5], [0.6, -0.1, 0.4]],
                [[-0.2, 0.7, 0.1], [0.5, 0.3, -0.6]],
                [[0.4, -0.3, 0.2], [-0.7, 0.1, 0.5]],
                [[0.1, 0.6, -0.4], [0.2, -0.5, 0.3]],
            ],
            dtype=torch.float64,
        ),
        spins=torch.tensor([[1.0, -1.0]] * 5, dtype=torch.float64),
    )
    terms = [KineticEnergy(), HarmonicTrap(omega=0.7)]
    full = local_energy(terms, GaussianOutputModel(alpha=0.23), batch, return_terms=True)
    chunked = evaluate_local_energy_in_chunks(
        terms,
        GaussianOutputModel(alpha=0.23),
        batch,
        return_terms=True,
        chunk_size=chunk_size,
    )

    assert isinstance(full, LocalEnergyResult)
    assert isinstance(chunked, LocalEnergyResult)
    assert torch.equal(chunked.total, full.total)
    assert list(chunked.terms) == list(full.terms)
    for name in full.terms:
        assert torch.equal(chunked.terms[name], full.terms[name])
    assert chunked.wavefunction_output is not None
    assert full.wavefunction_output is not None
    assert torch.equal(chunked.wavefunction_output.logabs, full.wavefunction_output.logabs)
    assert torch.equal(chunked.wavefunction_output.sign, full.wavefunction_output.sign)
    assert chunked.per_electron_kinetic is not None
    assert full.per_electron_kinetic is not None
    assert torch.equal(chunked.per_electron_kinetic, full.per_electron_kinetic)


@pytest.mark.parametrize("terms", [[KineticEnergy()], {"kinetic": KineticEnergy()}])
def test_chunked_local_energy_accepts_sequence_and_mapping_terms(terms) -> None:
    batch = ElectronBatch(
        positions=torch.tensor(
            [
                [[0.2, -0.1, 0.3], [-0.4, 0.5, -0.2]],
                [[0.3, 0.2, -0.5], [0.6, -0.1, 0.4]],
                [[-0.2, 0.7, 0.1], [0.5, 0.3, -0.6]],
                [[0.4, -0.3, 0.2], [-0.7, 0.1, 0.5]],
            ],
            dtype=torch.float64,
        )
    )
    model = GaussianOutputModel(alpha=0.3)
    full = local_energy(terms, model, batch, return_terms=True)
    chunked = evaluate_local_energy_in_chunks(
        terms, model, batch, return_terms=True, chunk_size=2
    )
    assert isinstance(full, LocalEnergyResult)
    assert isinstance(chunked, LocalEnergyResult)
    assert torch.equal(chunked.total, full.total)
    assert list(chunked.terms) == list(full.terms)
    for name in full.terms:
        assert torch.equal(chunked.terms[name], full.terms[name])
    assert chunked.wavefunction_output is not None
    assert full.wavefunction_output is not None
    assert torch.equal(chunked.wavefunction_output.logabs, full.wavefunction_output.logabs)
    assert torch.equal(chunked.wavefunction_output.sign, full.wavefunction_output.sign)
    assert chunked.per_electron_kinetic is not None
    assert full.per_electron_kinetic is not None
    assert torch.equal(chunked.per_electron_kinetic, full.per_electron_kinetic)


def test_chunked_local_energy_preserves_mapped_batch_metadata() -> None:
    positions = torch.arange(24, dtype=torch.float64).reshape(4, 2, 3) / 10
    batch = ElectronBatch(
        positions=positions,
        spins=torch.tensor([[1.0, -1.0]] * 4),
        nuclear_positions=torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0]],
                [[0.2, 0.0, 0.0]],
                [[0.3, 0.0, 0.0]],
            ],
            dtype=torch.float64,
        ),
        nuclear_charges=torch.ones(4, 1, dtype=torch.float64),
        aux={"sample_id": torch.arange(4)},
    )
    seen: list[ElectronBatch] = []

    class _MetadataEvaluator:
        def validate_for_generator(self, terms, wavefunction, generator):
            del terms, wavefunction, generator

        def make_context(self, wavefunction, batch):
            del wavefunction
            seen.append(batch)
            return batch

        def evaluate(self, terms, context, *, return_terms=False):
            del terms, return_terms
            return context.positions.sum(dim=(1, 2))

    evaluate_local_energy_in_chunks(
        {}, None, batch, chunk_size=2, evaluator=_MetadataEvaluator()
    )
    assert [chunk.spins.tolist() for chunk in seen] == [[[1.0, -1.0]], [[1.0, -1.0]]]
    assert [chunk.nuclear_positions.shape for chunk in seen] == [(2, 1, 3), (2, 1, 3)]
    assert [chunk.aux["sample_id"].tolist() for chunk in seen] == [[0, 1], [2, 3]]


# --- normalize_hamiltonian_terms ---


def test_normalize_hamiltonian_terms_dict_is_used_directly() -> None:
    kinetic = KineticEnergy()
    trap = HarmonicTrap(omega=1.0)
    normalized = normalize_hamiltonian_terms({"ke": kinetic, "trap": trap})

    assert list(normalized) == ["ke", "trap"]
    assert normalized["ke"] is kinetic
    assert normalized["trap"] is trap


def test_normalize_hamiltonian_terms_list_uses_snake_case_class_names() -> None:
    normalized = normalize_hamiltonian_terms(
        [KineticEnergy(), HarmonicTrap(omega=1.0), ElectronElectronInteraction()]
    )

    assert list(normalized) == ["kinetic_energy", "harmonic_trap", "electron_electron_interaction"]


def test_normalize_hamiltonian_terms_disambiguates_repeated_classes_by_index() -> None:
    normalized = normalize_hamiltonian_terms([HarmonicTrap(omega=1.0), HarmonicTrap(omega=2.0)])

    assert list(normalized) == ["harmonic_trap_0", "harmonic_trap_1"]


def test_normalize_hamiltonian_terms_rejects_non_string_keys() -> None:
    with pytest.raises(TypeError, match="must be strings"):
        normalize_hamiltonian_terms({0: KineticEnergy()})


def test_normalize_hamiltonian_terms_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_hamiltonian_terms({"  ": KineticEnergy()})


def test_normalize_hamiltonian_terms_rejects_invalid_term_spec() -> None:
    with pytest.raises(TypeError, match="local_energy"):
        normalize_hamiltonian_terms({"bad": object()})


# --- Local-energy helper over a list of terms ---


def test_local_energy_helper_sums_terms() -> None:
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    b = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    terms = [_ConstantTerm("a", a), _ConstantTerm("b", b)]
    batch = ElectronBatch(positions=torch.zeros(3, 2, 1, dtype=torch.float64))

    total = local_energy(terms, None, batch)

    assert isinstance(total, torch.Tensor)
    assert torch.equal(total, a + b)


def test_local_energy_helper_return_terms_true_with_named_dict() -> None:
    a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    b = torch.tensor([3.0, 4.0], dtype=torch.float64)
    terms = {"alpha": _ConstantTerm("alpha", a), "beta": _ConstantTerm("beta", b)}
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    result = local_energy(terms, None, batch, return_terms=True)

    assert isinstance(result, LocalEnergyResult)
    assert torch.equal(result.total, a + b)
    # dict keys become the decomposition names
    assert torch.equal(result.terms["alpha"], a)
    assert torch.equal(result.terms["beta"], b)


def test_local_energy_helper_return_terms_true_with_list_uses_snake_class_names() -> None:
    a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    b = torch.tensor([3.0, 4.0], dtype=torch.float64)
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    # A list of two same-class terms falls back to snake-case class names,
    # disambiguated by index to keep names unique.
    result = local_energy([_ConstantTerm("a", a), _ConstantTerm("b", b)], None, batch, return_terms=True)

    assert isinstance(result, LocalEnergyResult)
    assert set(result.terms) == {"constant_term_0", "constant_term_1"}
    assert torch.equal(result.terms["constant_term_0"], a)
    assert torch.equal(result.terms["constant_term_1"], b)


def test_local_energy_rejects_empty_configured_term_name() -> None:
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="non-empty"):
        local_energy({"": _ConstantTerm("ignored", torch.zeros(2, dtype=torch.float64))}, None, batch)


def test_local_energy_rejects_term_object_without_local_energy() -> None:
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(TypeError, match="local_energy"):
        local_energy({"bad": object()}, None, batch)


def test_local_energy_rejects_term_returning_tensor_instead_of_result() -> None:
    class TensorTerm:
        def local_energy(self, wavefunction, batch: ElectronBatch) -> torch.Tensor:
            return torch.zeros(batch.batch_size, dtype=batch.dtype)

    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(TypeError, match="LocalEnergyResult"):
        local_energy({"bad": TensorTerm()}, None, batch)


def test_local_energy_rejects_term_total_shape_mismatch() -> None:
    term = _ConstantTerm("bad", torch.zeros(2, 1, dtype=torch.float64))
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="total.*shape"):
        local_energy({"bad": term}, None, batch)


def test_local_energy_rejects_term_decomposition_shape_mismatch() -> None:
    class BadDecompositionTerm:
        def local_energy(self, wavefunction, batch: ElectronBatch) -> LocalEnergyResult:
            total = torch.zeros(batch.batch_size, dtype=batch.dtype)
            return LocalEnergyResult(total=total, terms={"bad": torch.zeros(batch.batch_size, 1)})

    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="decomposition.*shape"):
        local_energy({"bad": BadDecompositionTerm()}, None, batch)


def test_local_energy_return_terms_preserves_configured_names() -> None:
    a = torch.tensor([1.0, 2.0], dtype=torch.float64)
    b = torch.tensor([3.0, 4.0], dtype=torch.float64)
    batch = ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64))

    result = local_energy(
        {
            "kinetic_custom": _ConstantTerm("ignored_a", a),
            "trap_custom": _ConstantTerm("ignored_b", b),
        },
        None,
        batch,
        return_terms=True,
    )

    assert isinstance(result, LocalEnergyResult)
    assert set(result.terms) == {"kinetic_custom", "trap_custom"}


# --- Term classes return decomposed LocalEnergyResult objects ---


def test_harmonic_trap_term_returns_local_energy_result() -> None:
    positions = torch.tensor(
        [[[1.0, 0.0, -1.0], [0.5, 2.0, 0.25]], [[-0.5, 1.5, 2.5], [1.0, -2.0, 0.5]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)
    omega = 0.75

    result = HarmonicTrap(omega=omega).local_energy(None, batch)

    expected = 0.5 * (omega**2) * positions.square().sum(dim=(1, 2))
    assert isinstance(result, LocalEnergyResult)
    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.terms["harmonic_trap"], expected)


def test_electron_electron_interaction_term_returns_local_energy_result() -> None:
    positions = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)

    result = ElectronElectronInteraction().local_energy(None, batch)

    expected = torch.linalg.norm(positions[:, 0] - positions[:, 1], dim=-1).reciprocal()
    assert isinstance(result, LocalEnergyResult)
    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.terms["electron_electron"], expected)


def test_electron_nucleus_interaction_term_returns_local_energy_result() -> None:
    positions = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[-1.0, 0.5, 0.0], [0.0, 0.0, 2.0]]],
        dtype=torch.float64,
    )
    nuclei = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([2.0], dtype=torch.float64)
    batch = ElectronBatch(positions=positions, nuclear_positions=nuclei, nuclear_charges=charges)

    result = ElectronNucleusInteraction(nuclei, charges).local_energy(None, batch)

    expected = -(
        charges.view(1, 1, -1)
        / torch.linalg.norm(positions.unsqueeze(2) - nuclei.view(1, 1, 1, 3), dim=-1)
    ).sum(dim=(1, 2))
    assert isinstance(result, LocalEnergyResult)
    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.terms["electron_nucleus"], expected)


def test_electron_nucleus_interaction_uses_batch_context_and_rejects_legacy_mismatch() -> None:
    positions = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float64)
    nuclei = torch.tensor([[0.0, 0.0], [3.0, 0.0]], dtype=torch.float64)
    charges = torch.tensor([2.0, 1.0], dtype=torch.float64)
    batch = ElectronBatch(positions=positions, nuclear_positions=nuclei, nuclear_charges=charges)

    result = ElectronNucleusInteraction().local_energy(None, batch)
    permuted = ElectronNucleusInteraction().local_energy(None, batch.permute(Permutation((1, 0))))
    assert torch.equal(result.total, permuted.total)

    mismatched = ElectronNucleusInteraction(nuclei + 1.0, charges)
    with pytest.raises(ValueError, match="agree exactly"):
        mismatched.local_energy(None, batch)

    with pytest.raises(ValueError, match="batch.nuclear_positions"):
        ElectronNucleusInteraction().local_energy(None, ElectronBatch(positions=positions))

    with pytest.raises(ValueError, match="batch.nuclear_charges"):
        ElectronNucleusInteraction().local_energy(
            None,
            ElectronBatch(positions=positions, nuclear_positions=nuclei),
        )


@pytest.mark.parametrize(
    ("nuclei", "charges", "expected"),
    [
        ([[0.0, 0.0]], [2.0], [0.0]),
        ([[0.0, 0.0], [2.0, 0.0]], [2.0, 3.0], [3.0]),
        ([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], [1.0, 2.0, 3.0], [2.0 / 3.0 + 3.0 / 4.0 + 6.0 / 5.0]),
    ],
)
def test_nucleus_nucleus_interaction_exact_pair_sums(nuclei, charges, expected) -> None:
    batch = ElectronBatch(
        positions=torch.zeros(1, 1, 2, dtype=torch.float64),
        nuclear_positions=torch.tensor(nuclei, dtype=torch.float64),
        nuclear_charges=torch.tensor(charges, dtype=torch.float64),
    )

    result = NucleusNucleusInteraction().local_energy(None, batch)

    assert torch.allclose(result.total, torch.tensor(expected, dtype=torch.float64))
    assert torch.equal(result.total, result.terms["nucleus_nucleus"])


def test_nucleus_nucleus_interaction_is_invariant_to_identical_nucleus_permutation() -> None:
    batch = ElectronBatch(
        positions=torch.zeros(2, 1, 3, dtype=torch.float64),
        nuclear_positions=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0, 2.0, 1.0], dtype=torch.float64),
    )
    permuted = ElectronBatch(
        positions=batch.positions,
        nuclear_positions=batch.nuclear_positions[[1, 0, 2]],
        nuclear_charges=batch.nuclear_charges[[1, 0, 2]],
    )

    torch.testing.assert_close(
        NucleusNucleusInteraction().local_energy(None, batch).total,
        NucleusNucleusInteraction().local_energy(None, permuted).total,
    )


@pytest.mark.parametrize(
    ("nuclei", "charges", "message"),
    [
        ([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0], "colliding"),
        ([[0.0, float("nan")], [1.0, 0.0]], [1.0, 1.0], "finite"),
        ([[0.0, 0.0], [1.0, 0.0]], [1.0, float("inf")], "finite"),
    ],
)
def test_nucleus_nucleus_interaction_rejects_collisions_and_nonfinite_metadata(nuclei, charges, message) -> None:
    batch = ElectronBatch(
        positions=torch.zeros(1, 1, 2, dtype=torch.float64),
        nuclear_positions=torch.tensor(nuclei, dtype=torch.float64),
        nuclear_charges=torch.tensor(charges, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match=message):
        NucleusNucleusInteraction().local_energy(None, batch)


def test_nucleus_nucleus_interaction_preserves_dtype_device_and_hamiltonian_aggregation() -> None:
    batch = ElectronBatch(
        positions=torch.zeros(2, 1, 2, dtype=torch.float32),
        nuclear_positions=torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32),
        nuclear_charges=torch.tensor([2.0, 3.0], dtype=torch.float32),
    )
    term = NucleusNucleusInteraction()

    result = local_energy({"nn": term, "shift": HarmonicTrap(omega=0.0)}, None, batch, return_terms=True)

    assert isinstance(result, LocalEnergyResult)
    assert result.total.dtype == batch.positions.dtype
    assert result.total.device == batch.positions.device
    torch.testing.assert_close(result.terms["nn"], torch.tensor([3.0, 3.0], dtype=torch.float32))
    torch.testing.assert_close(result.total, result.terms["nn"] + result.terms["shift"])


def test_electron_nucleus_interaction_respects_batch_dtype() -> None:
    positions = torch.tensor([[[1.0], [2.0]]], dtype=torch.float32)
    batch = ElectronBatch(
        positions=positions,
        nuclear_positions=torch.tensor([[0.0]], dtype=torch.float64),
        nuclear_charges=torch.tensor([2.0], dtype=torch.float64),
    )

    result = ElectronNucleusInteraction().local_energy(None, batch)
    assert result.total.dtype is torch.float32
    assert result.total.device == positions.device


def test_kinetic_energy_term_returns_local_energy_result() -> None:
    positions = torch.tensor(
        [[[1.0, 2.0], [0.5, -1.0]], [[-1.5, 0.25], [2.0, 0.0]]],
        dtype=torch.float64,
    )
    batch = ElectronBatch(positions=positions)
    model = GaussianOutputModel(0.3)

    result = KineticEnergy().local_energy(model, batch)

    expected = kinetic_energy_from_logabs(model, batch)
    assert isinstance(result, LocalEnergyResult)
    assert torch.allclose(result.total, expected)
    assert torch.allclose(result.terms["kinetic"], expected)


# --- Symmetry sanity tests for exact Hooke references ---


def _two_particle_3d_batch() -> tuple[ElectronBatch, ElectronBatch]:
    """Return a batch and its particle-swapped version."""
    positions = torch.tensor(
        [
            [[0.3, -0.2, 0.8], [-0.1, 0.5, 0.3]],
            [[1.0, 0.0, 0.5], [0.0, -0.5, 1.5]],
            [[-0.4, 0.7, -0.3], [0.6, -0.1, 0.9]],
        ],
        dtype=torch.float64,
    )
    swapped = positions[:, [1, 0], :]
    return ElectronBatch(positions=positions), ElectronBatch(positions=swapped)


def test_singlet_logabs_invariant_under_particle_swap() -> None:
    wf = HookeSingletExact()
    batch, swapped = _two_particle_3d_batch()

    out = wf(batch)
    out_swapped = wf(swapped)

    assert torch.allclose(out.logabs, out_swapped.logabs)


def test_singlet_sign_invariant_under_particle_swap() -> None:
    wf = HookeSingletExact()
    batch, swapped = _two_particle_3d_batch()

    out = wf(batch)
    out_swapped = wf(swapped)

    assert torch.equal(out.sign, out_swapped.sign)


def test_triplet_logabs_invariant_under_particle_swap() -> None:
    wf = HookeTripletExact()
    # Use positions where z1 != z2 for all configs
    positions = torch.tensor(
        [
            [[0.3, -0.2, 0.8], [-0.1, 0.5, 0.3]],
            [[1.0, 0.0, 0.5], [0.0, -0.5, -0.5]],
            [[-0.4, 0.7, -0.3], [0.6, -0.1, 0.9]],
        ],
        dtype=torch.float64,
    )
    swapped = positions[:, [1, 0], :]
    batch = ElectronBatch(positions=positions)
    batch_swapped = ElectronBatch(positions=swapped)

    out = wf(batch)
    out_swapped = wf(batch_swapped)

    assert torch.allclose(out.logabs, out_swapped.logabs)


def test_triplet_sign_flips_under_particle_swap() -> None:
    wf = HookeTripletExact()
    positions = torch.tensor(
        [
            [[0.3, -0.2, 0.8], [-0.1, 0.5, 0.3]],
            [[1.0, 0.0, 0.5], [0.0, -0.5, -0.5]],
            [[-0.4, 0.7, -0.3], [0.6, -0.1, 0.9]],
        ],
        dtype=torch.float64,
    )
    swapped = positions[:, [1, 0], :]
    batch = ElectronBatch(positions=positions)
    batch_swapped = ElectronBatch(positions=swapped)

    out = wf(batch)
    out_swapped = wf(batch_swapped)

    assert torch.equal(out.sign, -out_swapped.sign)


# --- summarize_local_energy ---


def test_summarize_local_energy_all_finite() -> None:
    eloc = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)

    metrics = summarize_local_energy(eloc)

    assert metrics["local_energy_n_total"] == 3
    assert metrics["local_energy_n_finite"] == 3
    assert metrics["local_energy_finite_fraction"] == 1.0
    assert metrics["local_energy_nonfinite_count"] == 0
    assert metrics["energy"] == pytest.approx(3.0)
    assert metrics["energy_variance"] == pytest.approx(8.0 / 3.0)
    assert metrics["energy_std"] == pytest.approx(math.sqrt(8.0 / 3.0))
    assert metrics["energy_stderr"] == pytest.approx(math.sqrt(8.0 / 3.0) / math.sqrt(3))
    assert "expected_energy" not in metrics


def test_summarize_local_energy_some_nonfinite() -> None:
    eloc = torch.tensor([1.0, float("inf"), 3.0, float("nan")], dtype=torch.float64)

    metrics = summarize_local_energy(eloc)

    assert metrics["local_energy_n_total"] == 4
    assert metrics["local_energy_n_finite"] == 2
    assert metrics["local_energy_nonfinite_count"] == 2
    assert metrics["local_energy_finite_fraction"] == pytest.approx(0.5)
    assert metrics["energy"] == pytest.approx(2.0)
    assert math.isfinite(metrics["energy_variance"])


def test_summarize_local_energy_all_nonfinite() -> None:
    eloc = torch.tensor([float("inf"), float("nan")], dtype=torch.float64)

    metrics = summarize_local_energy(eloc)

    assert metrics["local_energy_n_finite"] == 0
    assert metrics["local_energy_n_total"] == 2
    assert metrics["local_energy_nonfinite_count"] == 2
    assert metrics["local_energy_finite_fraction"] == 0.0
    assert math.isnan(metrics["energy"])
    assert math.isnan(metrics["energy_variance"])
    assert math.isnan(metrics["energy_std"])
    assert metrics["energy_stderr"] == float("inf")


def test_summarize_local_energy_empty() -> None:
    metrics = summarize_local_energy(torch.empty(0, dtype=torch.float64))

    assert metrics["local_energy_n_total"] == 0
    assert metrics["local_energy_n_finite"] == 0
    assert metrics["local_energy_nonfinite_count"] == 0
    assert metrics["local_energy_finite_fraction"] == 0.0
    assert math.isnan(metrics["energy"])
    assert math.isnan(metrics["energy_variance"])
    assert math.isnan(metrics["energy_std"])
    assert metrics["energy_stderr"] == float("inf")


def test_summarize_local_energy_is_reference_free() -> None:
    eloc = torch.tensor([1.5, 2.5], dtype=torch.float64)

    metrics = summarize_local_energy(eloc)

    assert "reference_energy" not in metrics
    assert "energy_error" not in metrics
    assert "energy_abs_error" not in metrics


def test_summarize_local_energy_result_with_finite_terms() -> None:
    result = LocalEnergyResult(
        total=torch.tensor([2.0, 2.0], dtype=torch.float64),
        terms={
            "kinetic": torch.tensor([1.0, 1.0], dtype=torch.float64),
            "trap": torch.tensor([1.0, 1.0], dtype=torch.float64),
        },
    )

    metrics = summarize_local_energy(result)

    assert metrics["energy_term_kinetic"] == pytest.approx(1.0)
    assert metrics["energy_term_kinetic_nonfinite_count"] == 0
    assert metrics["energy_term_kinetic_finite_fraction"] == 1.0
    assert metrics["energy_term_trap"] == pytest.approx(1.0)
    assert metrics["energy_term_trap_nonfinite_count"] == 0


def test_summarize_local_energy_result_with_nonfinite_term_values() -> None:
    result = LocalEnergyResult(
        total=torch.tensor([2.0, 2.0], dtype=torch.float64),
        terms={"kinetic": torch.tensor([1.0, float("nan")], dtype=torch.float64)},
    )

    metrics = summarize_local_energy(result)

    assert metrics["energy_term_kinetic"] == pytest.approx(1.0)
    assert metrics["energy_term_kinetic_nonfinite_count"] == 1
    assert metrics["energy_term_kinetic_finite_fraction"] == pytest.approx(0.5)


def test_summarize_local_energy_term_with_no_finite_values() -> None:
    result = LocalEnergyResult(
        total=torch.tensor([2.0], dtype=torch.float64),
        terms={"kinetic": torch.tensor([float("nan")], dtype=torch.float64)},
    )

    metrics = summarize_local_energy(result)

    assert math.isnan(metrics["energy_term_kinetic"])
    assert math.isnan(metrics["energy_term_kinetic_variance"])
    assert metrics["energy_term_kinetic_nonfinite_count"] == 1
    assert metrics["energy_term_kinetic_finite_fraction"] == 0.0
