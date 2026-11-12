"""Physical-system descriptor and semantic term declarations (S1-D)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tpen.data.atomic_configuration import AtomicConfiguration
from tpen.physics.context import ReadOnlyContext
from tpen.physics.kinetic import KineticEnergy
from tpen.physics.operators import OperatorId, register_operator
from tpen.physics.potential import (
    ElectronElectronInteraction,
    ElectronNucleusPotential,
    HarmonicTrap,
    NucleusNucleusPotential,
)
from tpen.physics.system import (
    SYSTEM_DESCRIPTOR_KEY,
    HamiltonianTermDescriptor,
    ParticleSpec,
    SystemDescriptor,
    SystemDescriptorProvider,
    build_system_descriptor,
)


def _hooke_1e_terms() -> dict:
    return {"kinetic": KineticEnergy(), "harmonic_trap": HarmonicTrap(omega=0.5)}


def _hooke_1e_particle() -> ParticleSpec:
    return ParticleSpec(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=3)


def _helium_atoms() -> AtomicConfiguration:
    return AtomicConfiguration(torch.zeros(1, 3), torch.tensor([2.0]))


def _helium_terms(atoms: AtomicConfiguration | None = None) -> dict:
    atoms = atoms if atoms is not None else _helium_atoms()
    return {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(atoms),
        "electron_electron": ElectronElectronInteraction(),
    }


def _helium_particle() -> ParticleSpec:
    return ParticleSpec(n_electrons=2, n_spin_up=1, n_spin_down=1, spatial_dim=3)


def _h2_atoms(separation: float = 1.4) -> AtomicConfiguration:
    half = separation / 2.0
    return AtomicConfiguration(
        torch.tensor([[-half, 0.0, 0.0], [half, 0.0, 0.0]]),
        torch.tensor([1.0, 1.0]),
    )


def _h2_terms(atoms: AtomicConfiguration | None = None, *, include_nuclear_repulsion: bool = True) -> dict:
    atoms = atoms if atoms is not None else _h2_atoms()
    terms = {
        "kinetic": KineticEnergy(),
        "electron_nucleus": ElectronNucleusPotential(atoms),
        "electron_electron": ElectronElectronInteraction(),
    }
    if include_nuclear_repulsion:
        terms["nucleus_nucleus"] = NucleusNucleusPotential(atoms)
    return terms


def _h2_particle() -> ParticleSpec:
    return ParticleSpec(n_electrons=2, n_spin_up=1, n_spin_down=1, spatial_dim=3)


# --- Hooke-1e: builds with no fabricated nuclear resource -------------------


def test_hooke_1e_builds_with_no_nuclear_resource() -> None:
    descriptor = build_system_descriptor(_hooke_1e_particle(), _hooke_1e_terms())

    assert descriptor.particle.n_electrons == 1
    assert len(descriptor.terms) == 2
    assert all(term.resource_id is None for term in descriptor.terms)


# --- Golden, pinned content ids (cross-process stability evidence) ---------


GOLDEN_SYSTEM_IDS = {
    "hooke_1e_omega_0.5": "d8408fa70440c94c7f1a5826891917aff82651a1cab041a44268655682cb27e0",
    "helium": "8d41df68cc4e3285ff5e2fda1b09f8050430ac593a1c25b3abb56b1ba2d0ea8a",
    "h2_r1.4": "8daa15da651c797bdb3b91c6bfd61c5645f18d798944df9ae671a7ed585d86a3",
}


def _compute_golden_ids() -> dict[str, str]:
    return {
        "hooke_1e_omega_0.5": build_system_descriptor(_hooke_1e_particle(), _hooke_1e_terms()).system_id(),
        "helium": build_system_descriptor(_helium_particle(), _helium_terms()).system_id(),
        "h2_r1.4": build_system_descriptor(_h2_particle(), _h2_terms()).system_id(),
    }


def test_golden_system_ids_are_pinned() -> None:
    """Pin one golden system_id per canonical system.

    This is cross-process stability evidence for `system_id`: the hard-coded
    hex digests above were computed once and must keep matching. A change
    here means the canonical encoding changed, which is a breaking identity
    change requiring explicit review, not an incidental refactor.
    """

    computed = _compute_golden_ids()
    assert computed == GOLDEN_SYSTEM_IDS, (
        "system_id canonical encoding changed; if intentional, update "
        f"GOLDEN_SYSTEM_IDS to {computed!r}"
    )


# --- Discrimination ----------------------------------------------------------


def test_trap_omega_change_discriminates() -> None:
    base = build_system_descriptor(_hooke_1e_particle(), _hooke_1e_terms())
    changed = build_system_descriptor(
        _hooke_1e_particle(), {"kinetic": KineticEnergy(), "harmonic_trap": HarmonicTrap(omega=0.6)}
    )

    assert base.system_id() != changed.system_id()


def test_particle_count_discriminates() -> None:
    base = build_system_descriptor(_helium_particle(), _helium_terms())
    one_electron = build_system_descriptor(
        ParticleSpec(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=3),
        {"kinetic": KineticEnergy(), "electron_nucleus": ElectronNucleusPotential(_helium_atoms())},
    )

    assert base.system_id() != one_electron.system_id()


def test_spin_split_discriminates() -> None:
    singlet = build_system_descriptor(ParticleSpec(2, 1, 1, 3), _helium_terms())
    triplet_like = build_system_descriptor(ParticleSpec(2, 2, 0, 3), _helium_terms())

    assert singlet.system_id() != triplet_like.system_id()


def test_nuclear_charge_discriminates() -> None:
    helium = build_system_descriptor(_helium_particle(), _helium_terms())
    lithium_like_atoms = AtomicConfiguration(torch.zeros(1, 3), torch.tensor([3.0]))
    changed_charge = build_system_descriptor(_helium_particle(), _helium_terms(lithium_like_atoms))

    assert helium.system_id() != changed_charge.system_id()


def test_nuclear_geometry_discriminates() -> None:
    h2_equilibrium = build_system_descriptor(_h2_particle(), _h2_terms(_h2_atoms(1.4)))
    h2_stretched = build_system_descriptor(_h2_particle(), _h2_terms(_h2_atoms(2.0)))

    assert h2_equilibrium.system_id() != h2_stretched.system_id()


def test_h2_with_and_without_nuclear_repulsion_discriminates() -> None:
    with_repulsion = build_system_descriptor(_h2_particle(), _h2_terms(include_nuclear_repulsion=True))
    without_repulsion = build_system_descriptor(_h2_particle(), _h2_terms(include_nuclear_repulsion=False))

    assert with_repulsion.system_id() != without_repulsion.system_id()
    assert len(with_repulsion.terms) == len(without_repulsion.terms) + 1


# --- Compositional: trap-plus-nuclei ----------------------------------------


def test_trap_plus_nuclei_composes_and_differs_from_either_alone() -> None:
    atoms = _helium_atoms()
    trap_only = build_system_descriptor(_hooke_1e_particle(), _hooke_1e_terms())
    nuclei_only = build_system_descriptor(
        ParticleSpec(1, 1, 0, 3),
        {"kinetic": KineticEnergy(), "electron_nucleus": ElectronNucleusPotential(atoms)},
    )
    both = build_system_descriptor(
        ParticleSpec(1, 1, 0, 3),
        {
            "kinetic": KineticEnergy(),
            "harmonic_trap": HarmonicTrap(omega=0.5),
            "electron_nucleus": ElectronNucleusPotential(atoms),
        },
    )

    assert both.system_id() not in (trap_only.system_id(), nuclei_only.system_id())
    assert len(both.terms) == 3


# --- Registered custom term vs. unregistered/undeclared term ---------------


class _UnregisteredTerm:
    """A term class that was never passed through `register_operator`."""

    name = "mystery"

    def local_energy(self, wavefunction, batch):  # pragma: no cover - never reached
        raise AssertionError("local_energy should not run before the descriptor guard fires")


def test_unregistered_term_raises_before_any_descriptor_is_produced() -> None:
    with pytest.raises(ValueError, match="not a registered operator"):
        build_system_descriptor(_hooke_1e_particle(), {"mystery": _UnregisteredTerm()})


def test_registered_term_without_description_raises_before_any_descriptor_is_produced() -> None:
    custom_id = OperatorId("test.system", "undeclared_custom_term")

    @register_operator(custom_id)
    class _UndeclaredCustomTerm:
        name = "undeclared_custom"

        def local_energy(self, wavefunction, batch):  # pragma: no cover - never reached
            raise AssertionError("local_energy should not run before the descriptor guard fires")

    with pytest.raises(ValueError, match="does not implement describe_physical_operator"):
        build_system_descriptor(_hooke_1e_particle(), {"undeclared_custom": _UndeclaredCustomTerm()})


def test_registered_custom_term_declares_itself() -> None:
    custom_id = OperatorId("test.system", "custom_confinement")

    @register_operator(custom_id)
    class _CustomConfinement:
        name = "custom_confinement"

        def __init__(self, strength: float) -> None:
            self.strength = strength

        def local_energy(self, wavefunction, batch):  # pragma: no cover - not exercised here
            raise AssertionError("local_energy is not exercised by this descriptor-only test")

        def describe_physical_operator(self) -> HamiltonianTermDescriptor:
            return HamiltonianTermDescriptor(
                operator_id=self.operator_id,
                slot=self.name,
                parameters={"strength": float(self.strength)},
            )

    descriptor = build_system_descriptor(
        _hooke_1e_particle(),
        {"kinetic": KineticEnergy(), "custom_confinement": _CustomConfinement(strength=1.5)},
    )

    custom_terms = [term for term in descriptor.terms if term.operator_id == custom_id]
    assert len(custom_terms) == 1
    assert custom_terms[0].parameters["strength"] == 1.5


def test_describe_physical_operator_must_agree_with_registration() -> None:
    custom_id = OperatorId("test.system", "mismatched_term")
    other_id = OperatorId("test.system", "other_operator")

    @register_operator(custom_id)
    class _MismatchedTerm:
        name = "mismatched"

        def local_energy(self, wavefunction, batch):  # pragma: no cover - never reached
            raise AssertionError("local_energy should not run before the descriptor guard fires")

        def describe_physical_operator(self) -> HamiltonianTermDescriptor:
            return HamiltonianTermDescriptor(operator_id=other_id, slot=self.name)

    with pytest.raises(ValueError, match="disagrees with its registered operator"):
        build_system_descriptor(_hooke_1e_particle(), {"mismatched": _MismatchedTerm()})


# --- Immutability and canonical content id ----------------------------------


def test_same_physics_different_construction_order_gives_identical_id() -> None:
    atoms = _h2_atoms()
    forward = build_system_descriptor(
        _h2_particle(),
        {
            "kinetic": KineticEnergy(),
            "electron_nucleus": ElectronNucleusPotential(atoms),
            "electron_electron": ElectronElectronInteraction(),
            "nucleus_nucleus": NucleusNucleusPotential(atoms),
        },
    )
    reversed_order = build_system_descriptor(
        _h2_particle(),
        {
            "nucleus_nucleus": NucleusNucleusPotential(atoms),
            "electron_electron": ElectronElectronInteraction(),
            "electron_nucleus": ElectronNucleusPotential(atoms),
            "kinetic": KineticEnergy(),
        },
    )
    from_sequence = build_system_descriptor(
        _h2_particle(),
        [
            ElectronElectronInteraction(),
            KineticEnergy(),
            NucleusNucleusPotential(atoms),
            ElectronNucleusPotential(atoms),
        ],
    )

    assert forward.system_id() == reversed_order.system_id() == from_sequence.system_id()
    # Configured order is still preserved for execution/record binding.
    assert [term.slot for term in forward.terms] == ["kinetic", "electron_nucleus", "electron_electron", "nucleus_nucleus"]
    assert [term.slot for term in reversed_order.terms] == ["nucleus_nucleus", "electron_electron", "electron_nucleus", "kinetic"]


def test_particle_spec_is_immutable() -> None:
    particle = ParticleSpec(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=3)

    with pytest.raises(dataclasses.FrozenInstanceError):
        particle.n_electrons = 2  # type: ignore[misc]


def test_term_descriptor_is_immutable() -> None:
    descriptor = KineticEnergy().describe_physical_operator()

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.slot = "renamed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        descriptor.parameters["new"] = 1  # type: ignore[index]


def test_system_descriptor_is_immutable() -> None:
    descriptor = build_system_descriptor(_hooke_1e_particle(), _hooke_1e_terms())

    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.units = "si"  # type: ignore[misc]


def test_system_descriptor_rejects_duplicate_operator_identity() -> None:
    with pytest.raises(ValueError, match="more than one term"):
        build_system_descriptor(
            _hooke_1e_particle(),
            {"a": HarmonicTrap(omega=0.5), "b": HarmonicTrap(omega=0.5)},
        )


@pytest.mark.parametrize(
    "bad_particle_kwargs",
    [
        dict(n_electrons=-1, n_spin_up=0, n_spin_down=0, spatial_dim=3),
        dict(n_electrons=2, n_spin_up=1, n_spin_down=0, spatial_dim=3),
        dict(n_electrons=1, n_spin_up=1, n_spin_down=0, spatial_dim=0),
    ],
)
def test_particle_spec_rejects_invalid_values(bad_particle_kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ParticleSpec(**bad_particle_kwargs)


def test_term_parameters_reject_non_scalar_values() -> None:
    with pytest.raises(TypeError):
        HamiltonianTermDescriptor(
            operator_id=OperatorId("test.system", "bad_params"),
            slot="bad",
            parameters={"vector": [1, 2, 3]},  # type: ignore[dict-item]
        )


# --- Published through one physics-owned typed context key/provider --------


def test_system_descriptor_published_through_one_context_key_and_provider() -> None:
    descriptor = build_system_descriptor(_hooke_1e_particle(), _hooke_1e_terms())
    provider = SystemDescriptorProvider(descriptor)

    assert provider.key is SYSTEM_DESCRIPTOR_KEY
    assert provider.dependencies == frozenset()

    context = ReadOnlyContext({SYSTEM_DESCRIPTOR_KEY: provider.build(ReadOnlyContext({}))})
    assert context[SYSTEM_DESCRIPTOR_KEY] is descriptor


def test_provider_rejects_a_foreign_key() -> None:
    from tpen.physics.context import ContextKey

    foreign_key = ContextKey("system_descriptor", SystemDescriptor)
    descriptor = build_system_descriptor(_hooke_1e_particle(), _hooke_1e_terms())

    with pytest.raises(ValueError, match="SYSTEM_DESCRIPTOR_KEY"):
        SystemDescriptorProvider(descriptor, key=foreign_key)
