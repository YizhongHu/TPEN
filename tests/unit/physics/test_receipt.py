"""Shared local-energy execution receipt and result transport (S1-E)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.physics.hamiltonian import (
    DeclaresExecutionReceipt,
    LocalEnergyResult,
    NaiveLocalEnergyContext,
    NaiveLocalEnergyEvaluator,
    RecordingLocalEnergyEvaluator,
    evaluate_local_energy_in_chunks,
)
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.operators import (
    ELECTRON_ELECTRON_COULOMB,
    ELECTRON_NUCLEUS_COULOMB,
    HARMONIC_TRAP,
    KINETIC_ENERGY,
)
from tpen.physics.potential import ElectronElectronInteraction, ElectronNucleusPotential, HarmonicTrap
from tpen.physics.receipt import DiagnosticDeclaration, ExecutionReceipt, require_matching_system
from tpen.physics.system import ParticleSpec, build_system_descriptor

_DTYPE = torch.float64


class GaussianOutputModel(nn.Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=_DTYPE)

    def forward(self, batch: ElectronBatch) -> WavefunctionOutput:
        logabs = -self.alpha * batch.positions.square().sum(dim=(1, 2))
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def _terms() -> dict:
    return {"kinetic": KineticEnergy(), "harmonic_trap": HarmonicTrap(omega=0.5)}


def _batch(n_walkers: int = 5) -> ElectronBatch:
    generator = torch.Generator().manual_seed(7)
    return ElectronBatch(positions=torch.randn(n_walkers, 1, 3, generator=generator, dtype=_DTYPE))


def _receipt(**overrides) -> ExecutionReceipt:
    fields = dict(
        system_id="system-a",
        implementation="tpen.physics.hamiltonian.NaiveLocalEnergyEvaluator",
        version=1,
        numerical_policy={"term_order": "configured_declaration_order"},
        manifest={KINETIC_ENERGY: ("kinetic",)},
    )
    fields.update(overrides)
    return ExecutionReceipt(**fields)


# --- ExecutionReceipt validation and identity --------------------------------


def test_receipt_rejects_empty_manifest() -> None:
    with pytest.raises(ValueError):
        _receipt(manifest={})


def test_receipt_rejects_non_operator_id_manifest_key() -> None:
    with pytest.raises(TypeError):
        _receipt(manifest={"kinetic": ("kinetic",)})


def test_diagnostic_declaration_requires_non_empty_fields() -> None:
    with pytest.raises(ValueError):
        DiagnosticDeclaration(name="", available=True, derivation="x")
    with pytest.raises(ValueError):
        DiagnosticDeclaration(name="x", available=True, derivation="")


def test_diagnostics_key_must_match_declaration_name() -> None:
    declaration = DiagnosticDeclaration(name="kinetic_split", available=False, derivation="unavailable: fused")
    with pytest.raises(ValueError):
        _receipt(diagnostics={"other_name": declaration})


def test_method_id_identical_for_identical_settings() -> None:
    assert _receipt().method_id() == _receipt().method_id()
    assert _receipt() == _receipt()


def test_method_id_distinct_for_changed_numerical_policy() -> None:
    changed = _receipt(numerical_policy={"term_order": "other"})
    assert changed.method_id() != _receipt().method_id()


def test_method_id_distinct_for_changed_manifest() -> None:
    changed = _receipt(manifest={KINETIC_ENERGY: ("kinetic",), HARMONIC_TRAP: ("harmonic_trap",)})
    assert changed.method_id() != _receipt().method_id()


def test_method_id_distinct_for_changed_version() -> None:
    changed = _receipt(version=2)
    assert changed.method_id() != _receipt().method_id()


def test_method_id_numerical_policy_boundary_shift_no_longer_collides() -> None:
    # Regression for the v1 collision found in review: concatenating an
    # unframed key directly against an unframed string value let the
    # key/value boundary shift without changing the overall bytes --
    # {"a": "sb"} and {"as": "b"} both encoded to b"asb" under v1 and
    # therefore hashed to the same method_id. v2's length-prefixed framing
    # (see `_encode_length_prefixed`) makes the boundary unambiguous.
    left = _receipt(numerical_policy={"a": "sb"})
    right = _receipt(numerical_policy={"as": "b"})
    assert left.method_id() != right.method_id()


def test_method_id_manifest_names_boundary_shift_no_longer_collides() -> None:
    # Analogous boundary-shift case for the manifest's per-operator
    # contribution-name list: without a length prefix on each name, the
    # boundary between adjacent names in the list can shift without
    # changing the overall bytes.
    left = _receipt(manifest={KINETIC_ENERGY: ("ab", "c")})
    right = _receipt(manifest={KINETIC_ENERGY: ("a", "bc")})
    assert left.method_id() != right.method_id()


def test_method_id_reproducible_across_independent_instances() -> None:
    # Stands in for "stable across processes": two independently constructed
    # receipts, from independently constructed (but equal) inputs, agree.
    policy_a = {"term_order": "configured_declaration_order"}
    policy_b = {"term_order": "configured_declaration_order"}
    manifest_a = {KINETIC_ENERGY: ("kinetic",)}
    manifest_b = {KINETIC_ENERGY: ("kinetic",)}
    left = _receipt(numerical_policy=policy_a, manifest=manifest_a)
    right = _receipt(numerical_policy=policy_b, manifest=manifest_b)
    assert left.method_id() == right.method_id()
    assert left is not right


def test_require_matching_system_passes_through() -> None:
    receipt = _receipt(system_id="sys-123")
    assert require_matching_system(receipt, "sys-123") is receipt


def test_require_matching_system_raises_on_mismatch() -> None:
    receipt = _receipt(system_id="sys-123")
    with pytest.raises(ValueError):
        require_matching_system(receipt, "sys-456")


# --- Naive reference: receipt declaration ------------------------------------


def test_naive_describe_execution_receipt_manifest_matches_terms() -> None:
    evaluator = NaiveLocalEnergyEvaluator()
    receipt = evaluator.describe_execution_receipt(_terms(), "sys-xyz")
    assert receipt.system_id == "sys-xyz"
    assert dict(receipt.manifest) == {
        KINETIC_ENERGY: ("kinetic",),
        HARMONIC_TRAP: ("harmonic_trap",),
    }
    assert receipt.diagnostics == {}


def test_naive_describe_execution_receipt_changed_settings_distinct() -> None:
    evaluator = NaiveLocalEnergyEvaluator()
    full = evaluator.describe_execution_receipt(_terms(), "sys-xyz")
    partial = evaluator.describe_execution_receipt({"kinetic": KineticEnergy()}, "sys-xyz")
    assert full.method_id() != partial.method_id()
    assert full != partial


def test_naive_describe_execution_receipt_identical_settings_identical() -> None:
    evaluator = NaiveLocalEnergyEvaluator()
    left = evaluator.describe_execution_receipt(_terms(), "sys-xyz")
    right = evaluator.describe_execution_receipt(_terms(), "sys-xyz")
    assert left == right
    assert left.method_id() == right.method_id()


# --- RecordingLocalEnergyEvaluator: narrow adapter ---------------------------


def test_recording_adapter_rejects_delegate_without_receipt_capability() -> None:
    # A structural stand-in that implements only the LocalEnergyEvaluator
    # surface, with no describe_execution_receipt, proves the adapter
    # refuses a delegate that cannot declare a receipt rather than silently
    # substituting another evaluator's receipt.
    class NoReceiptEvaluator:
        def make_context(self, wavefunction, batch):
            return NaiveLocalEnergyContext(wavefunction=wavefunction, batch=batch)

        def validate_for_generator(self, terms, wavefunction, generator):
            del terms, wavefunction, generator

        def evaluate(self, terms, context, *, return_terms=False):
            return NaiveLocalEnergyEvaluator().evaluate(terms, context, return_terms=return_terms)

    assert not isinstance(NoReceiptEvaluator(), DeclaresExecutionReceipt)
    with pytest.raises(TypeError):
        RecordingLocalEnergyEvaluator(NoReceiptEvaluator(), system_id="sys-xyz")


def test_recording_adapter_requires_non_empty_system_id() -> None:
    with pytest.raises(ValueError):
        RecordingLocalEnergyEvaluator(NaiveLocalEnergyEvaluator(), system_id="")


def test_recording_adapter_preserves_naive_arithmetic_bit_identical() -> None:
    terms = _terms()
    model = GaussianOutputModel(alpha=0.5)
    batch = _batch()
    naive = NaiveLocalEnergyEvaluator()
    context = naive.make_context(model, batch)

    plain = naive.evaluate(terms, context, return_terms=True)
    wrapped_evaluator = RecordingLocalEnergyEvaluator(naive, system_id="sys-xyz")
    recorded = wrapped_evaluator.evaluate(terms, context, return_terms=True)

    assert isinstance(plain, LocalEnergyResult)
    assert isinstance(recorded, LocalEnergyResult)
    assert torch.equal(plain.total, recorded.total)
    assert set(plain.terms) == set(recorded.terms)
    for name in plain.terms:
        assert torch.equal(plain.terms[name], recorded.terms[name])
    assert plain.receipt is None
    assert recorded.receipt is not None
    assert recorded.receipt.system_id == "sys-xyz"


def test_recording_adapter_scalar_result_unaffected() -> None:
    terms = _terms()
    model = GaussianOutputModel(alpha=0.5)
    batch = _batch()
    naive = NaiveLocalEnergyEvaluator()
    context = naive.make_context(model, batch)

    plain_total = naive.evaluate(terms, context, return_terms=False)
    wrapped_evaluator = RecordingLocalEnergyEvaluator(naive, system_id="sys-xyz")
    recorded_total = wrapped_evaluator.evaluate(terms, context, return_terms=False)

    assert isinstance(recorded_total, torch.Tensor)
    assert torch.equal(plain_total, recorded_total)


# --- Fused contribution manifest (stub producer) -----------------------------


class _StubFusedEvaluator(NaiveLocalEnergyEvaluator):
    """Stub fused producer: one contribution claims both operator ids."""

    fused_term_name = "kinetic_plus_harmonic_trap"

    def evaluate(self, terms, context, *, return_terms=False):
        result = super().evaluate(terms, context, return_terms=True)
        fused_total = result.terms["kinetic"] + result.terms["harmonic_trap"]
        fused = LocalEnergyResult(
            total=fused_total,
            terms={self.fused_term_name: fused_total},
            wavefunction_output=result.wavefunction_output,
            per_electron_kinetic=result.per_electron_kinetic,
        )
        return fused if return_terms else fused.total

    def describe_execution_receipt(self, terms, system_id):
        del terms
        return ExecutionReceipt(
            system_id=system_id,
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            version=1,
            numerical_policy={"fusion": "kinetic_plus_harmonic_trap"},
            manifest={
                KINETIC_ENERGY: (self.fused_term_name,),
                HARMONIC_TRAP: (self.fused_term_name,),
            },
        )


def test_fused_stub_manifest_covers_both_ids_without_fabricated_diagnostics() -> None:
    terms = _terms()
    model = GaussianOutputModel(alpha=0.5)
    batch = _batch()
    stub = _StubFusedEvaluator()
    context = stub.make_context(model, batch)
    adapter = RecordingLocalEnergyEvaluator(stub, system_id="sys-fused")

    result = adapter.evaluate(terms, context, return_terms=True)

    assert isinstance(result, LocalEnergyResult)
    assert list(result.terms) == [_StubFusedEvaluator.fused_term_name]
    receipt = result.receipt
    assert receipt is not None
    assert receipt.manifest[KINETIC_ENERGY] == (_StubFusedEvaluator.fused_term_name,)
    assert receipt.manifest[HARMONIC_TRAP] == (_StubFusedEvaluator.fused_term_name,)
    assert receipt.diagnostics == {}


# --- Chunked evaluation: receipt binding -------------------------------------


def test_chunked_evaluation_carries_one_receipt() -> None:
    terms = _terms()
    model = GaussianOutputModel(alpha=0.5)
    batch = _batch(n_walkers=8)
    naive = NaiveLocalEnergyEvaluator()
    adapter = RecordingLocalEnergyEvaluator(naive, system_id="sys-chunked")

    result = evaluate_local_energy_in_chunks(
        terms,
        model,
        batch,
        return_terms=True,
        chunk_size=3,
        evaluator=adapter,
    )

    assert isinstance(result, LocalEnergyResult)
    assert result.receipt is not None
    assert result.receipt.system_id == "sys-chunked"


class _AlternatingReceiptEvaluator(NaiveLocalEnergyEvaluator):
    """Stub evaluator that disagrees with itself across calls."""

    def __init__(self) -> None:
        self._calls = 0

    def describe_execution_receipt(self, terms, system_id):
        del terms
        self._calls += 1
        return ExecutionReceipt(
            system_id=f"{system_id}-{self._calls}",
            implementation="tests.stub.AlternatingReceiptEvaluator",
            version=1,
            numerical_policy={},
            manifest={KINETIC_ENERGY: ("kinetic",)},
        )


def test_chunked_evaluation_raises_on_receipt_disagreement() -> None:
    terms = _terms()
    model = GaussianOutputModel(alpha=0.5)
    batch = _batch(n_walkers=8)
    unstable = _AlternatingReceiptEvaluator()
    adapter = RecordingLocalEnergyEvaluator(unstable, system_id="sys-unstable")

    with pytest.raises(ValueError, match="disagreed between chunks"):
        evaluate_local_energy_in_chunks(
            terms,
            model,
            batch,
            return_terms=True,
            chunk_size=3,
            evaluator=adapter,
        )


# --- Integration: receipt bound to a real SystemDescriptor --------------------


def _batch_with_electrons(n_electrons: int, n_walkers: int = 3) -> ElectronBatch:
    generator = torch.Generator().manual_seed(11)
    return ElectronBatch(positions=torch.randn(n_walkers, n_electrons, 3, generator=generator, dtype=_DTYPE))


def _manifest_from_descriptor(descriptor) -> dict:
    manifest: dict = {}
    for term in descriptor.terms:
        manifest.setdefault(term.operator_id, []).append(term.slot)
    return {operator_id: tuple(names) for operator_id, names in manifest.items()}


def test_receipt_system_id_and_manifest_match_real_hooke_descriptor() -> None:
    particle = ParticleSpec(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=3)
    terms = {"kinetic": KineticEnergy(), "trap_energy": HarmonicTrap(omega=0.5)}
    descriptor = build_system_descriptor(particle, terms)
    assert [term.slot for term in descriptor.terms] == ["kinetic", "trap_energy"]

    naive = NaiveLocalEnergyEvaluator()
    context = naive.make_context(GaussianOutputModel(alpha=0.5), _batch_with_electrons(1))
    adapter = RecordingLocalEnergyEvaluator(naive, system_id=descriptor.system_id())
    result = adapter.evaluate(terms, context, return_terms=True)

    assert result.receipt.system_id == descriptor.system_id()
    assert dict(result.receipt.manifest) == _manifest_from_descriptor(descriptor)
    require_matching_system(result.receipt, descriptor.system_id())
    with pytest.raises(ValueError, match="does not match"):
        require_matching_system(result.receipt, "a-different-system-id")


def test_receipt_system_id_and_manifest_match_real_helium_descriptor() -> None:
    atoms = AtomicConfiguration(torch.zeros(1, 3), torch.tensor([2.0]))
    particle = ParticleSpec(n_electrons=2, n_spin_up=1, n_spin_down=1, spatial_dim=3)
    terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(atoms),
        "electron_electron": ElectronElectronInteraction(),
    }
    descriptor = build_system_descriptor(particle, terms)
    assert {term.operator_id for term in descriptor.terms} == {
        KINETIC_ENERGY,
        ELECTRON_NUCLEUS_COULOMB,
        ELECTRON_ELECTRON_COULOMB,
    }

    naive = NaiveLocalEnergyEvaluator()
    context = naive.make_context(GaussianOutputModel(alpha=0.5), _batch_with_electrons(2))
    adapter = RecordingLocalEnergyEvaluator(naive, system_id=descriptor.system_id())
    result = adapter.evaluate(terms, context, return_terms=True)

    assert result.receipt.system_id == descriptor.system_id()
    assert dict(result.receipt.manifest) == _manifest_from_descriptor(descriptor)
    require_matching_system(result.receipt, descriptor.system_id())

    other_descriptor = build_system_descriptor(
        ParticleSpec(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=3),
        {"kinetic": KineticEnergy(), "harmonic_trap": HarmonicTrap(omega=0.5)},
    )
    with pytest.raises(ValueError, match="does not match"):
        require_matching_system(result.receipt, other_descriptor.system_id())
