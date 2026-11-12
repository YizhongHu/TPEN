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
    assert descriptor.resources == {}


# --- Golden, pinned content ids (cross-process stability evidence) ---------


#: r1 fix bumped the canonical encoding to v2 (versioned length/count
#: framing; see `tpen.physics.system._DESCRIPTOR_SCHEMA_VERSION`), so every
#: v1 golden value above is stale. These v2 values were discovered by the
#: same two-step process used to pin the original v1 goldens: commit
#: deliberately-wrong placeholders, run on Cannon (`test` partition, job
#: 59922776, SHA 6fe7dd1dda7af00a600c91d08d068accaa5ba934), and take the
#: real computed values from the failing assertion's message.
GOLDEN_SYSTEM_IDS = {
    "hooke_1e_omega_0.5": "1ea17d39bfbf291f014d1e5f35cfc759ebe6a609ba8bc8f39ebde8c82b4eb62f",
    "helium": "093086f1343c96add6ca23d7a2af404db274ba200cfa772dff6f1f68b6b27a18",
    "h2_r1.4": "802be996a23facb3e1940cd8bca195131148f7c0b1d6f25de262356ddafbbe2d",
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


# --- Framing: variable-length parameter fields must not collide ------------


def _string_param_term(custom_id: OperatorId, params: dict) -> HamiltonianTermDescriptor:
    class _StringParamTerm:
        name = "string_param"

        def local_energy(self, wavefunction, batch):  # pragma: no cover - never reached
            raise AssertionError("local_energy is not exercised by this descriptor-only test")

    registered = register_operator(custom_id)(_StringParamTerm)
    term = registered()
    return HamiltonianTermDescriptor(operator_id=term.operator_id, slot=term.name, parameters=params)


def test_unframed_parameter_boundaries_no_longer_collide() -> None:
    """r1 regression: ``{"a": "bs", "b": ""}`` vs ``{"a": "", "b": "bs"}``.

    Under the v1 encoding these two distinct parameter maps both serialized
    to ``b"asbsbs"`` (no boundary between a key's bytes and its value's
    bytes, and no boundary between successive key/value pairs), so they
    hashed to the same `HamiltonianTermDescriptor.canonical_bytes`. The v2
    length-framed encoding (`_encode_length_prefixed`) must distinguish them.
    """

    left = _string_param_term(OperatorId("test.system", "framing_left"), {"a": "bs", "b": ""})
    right = _string_param_term(OperatorId("test.system", "framing_left"), {"a": "", "b": "bs"})

    assert left.canonical_bytes() != right.canonical_bytes()


def test_reversed_parameter_insertion_order_is_still_equal() -> None:
    """Sorting by key (not insertion order) keeps this green, unlike the collision above."""

    custom_id = OperatorId("test.system", "framing_order")
    forward = _string_param_term(custom_id, {"a": "bs", "b": "cs"})
    reversed_insertion = HamiltonianTermDescriptor(
        operator_id=forward.operator_id, slot=forward.slot, parameters={"b": "cs", "a": "bs"}
    )

    assert forward.canonical_bytes() == reversed_insertion.canonical_bytes()


# --- Resource payloads: immutable typed snapshots, deduplicated by reference -


def test_resource_payload_survives_input_mutation() -> None:
    positions = torch.tensor([[0.0, 0.0, 0.0]])
    charges = torch.tensor([2.0])
    atoms = AtomicConfiguration(positions, charges)
    descriptor = build_system_descriptor(
        ParticleSpec(1, 1, 0, 3),
        {"kinetic": KineticEnergy(), "electron_nucleus": ElectronNucleusPotential(atoms)},
    )
    resource_id = atoms.content_id()
    before = descriptor.resources[resource_id].positions.clone()

    # Mutate the term's own geometry object's tensor in place, after the
    # descriptor exists. `atoms.positions` is itself an owned clone (never
    # the caller's original tensor), so this exercises the *second*,
    # descriptor-level clone: `SystemDescriptor.resources` must not alias
    # `atoms.positions` either.
    atoms.positions.add_(100.0)

    assert torch.equal(descriptor.resources[resource_id].positions, before)
    assert not torch.equal(atoms.positions, before)


def test_shared_en_nn_resource_references_resolve_to_one_payload() -> None:
    atoms = _h2_atoms()
    descriptor = build_system_descriptor(
        _h2_particle(),
        {
            "kinetic": KineticEnergy(),
            "electron_nucleus": ElectronNucleusPotential(atoms),
            "nucleus_nucleus": NucleusNucleusPotential(atoms),
        },
    )

    en_term = next(t for t in descriptor.terms if t.slot == "electron_nucleus")
    nn_term = next(t for t in descriptor.terms if t.slot == "nucleus_nucleus")
    assert en_term.resource_id == nn_term.resource_id
    assert len(descriptor.resources) == 1
    assert descriptor.resources[en_term.resource_id].positions.shape == atoms.positions.shape


def test_nn_only_geometry_discriminates() -> None:
    """`NucleusNucleusPotential` alone, varying only separation."""

    equilibrium = build_system_descriptor(_h2_particle(), {"nucleus_nucleus": NucleusNucleusPotential(_h2_atoms(1.4))})
    stretched = build_system_descriptor(_h2_particle(), {"nucleus_nucleus": NucleusNucleusPotential(_h2_atoms(2.0))})

    assert equilibrium.system_id() != stretched.system_id()


def test_term_resource_survives_input_mutation() -> None:
    """r2 regression: `HamiltonianTermDescriptor.resource` must not alias its input.

    The r1 fix decoupled `SystemDescriptor.resources` from the caller's
    atoms (see `test_resource_payload_survives_input_mutation`), but
    `HamiltonianTermDescriptor.resource` itself still held the caller's own
    `AtomicConfiguration` object: mutating `atoms` after construction changed
    what `term.resource` reported, even though the `resources` table stayed
    correct (the two access paths disagreed about the same resource_id).
    """

    positions = torch.tensor([[0.0, 0.0, 0.0]])
    charges = torch.tensor([2.0])
    atoms = AtomicConfiguration(positions, charges)
    descriptor = build_system_descriptor(
        ParticleSpec(1, 1, 0, 3),
        {"kinetic": KineticEnergy(), "electron_nucleus": ElectronNucleusPotential(atoms)},
    )
    resource_id = atoms.content_id()
    term = next(t for t in descriptor.terms if t.slot == "electron_nucleus")

    atoms.positions.add_(100.0)
    atoms.charges.add_(1.0)

    assert term.resource.content_id() == resource_id
    assert descriptor.resources[resource_id].content_id() == resource_id


def test_exposed_table_resource_mutation_does_not_persist() -> None:
    """Mutating a tensor obtained from `d.resources[rid]` must not corrupt later reads."""

    atoms = AtomicConfiguration(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([2.0]))
    descriptor = build_system_descriptor(
        ParticleSpec(1, 1, 0, 3),
        {"kinetic": KineticEnergy(), "electron_nucleus": ElectronNucleusPotential(atoms)},
    )
    resource_id = atoms.content_id()

    exposed = descriptor.resources[resource_id]
    exposed.positions.add_(100.0)
    exposed.charges.add_(1.0)

    assert descriptor.resources[resource_id].content_id() == resource_id
    assert descriptor.system_id() == build_system_descriptor(
        ParticleSpec(1, 1, 0, 3),
        {"kinetic": KineticEnergy(), "electron_nucleus": ElectronNucleusPotential(atoms)},
    ).system_id()


def test_exposed_term_resource_mutation_does_not_persist() -> None:
    """Mutating a tensor obtained from `term.resource` must not corrupt later reads."""

    atoms = AtomicConfiguration(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([2.0]))
    descriptor = build_system_descriptor(
        ParticleSpec(1, 1, 0, 3),
        {"kinetic": KineticEnergy(), "electron_nucleus": ElectronNucleusPotential(atoms)},
    )
    resource_id = atoms.content_id()
    term = next(t for t in descriptor.terms if t.slot == "electron_nucleus")

    exposed = term.resource
    exposed.positions.add_(100.0)
    exposed.charges.add_(1.0)

    assert term.resource.content_id() == resource_id
    assert descriptor.resources[resource_id].content_id() == resource_id


def test_stale_term_declaration_does_not_leak_into_aggregated_resource() -> None:
    """A term's declaration is snapshotted at its own construction, not re-read live.

    Declare a term, then mutate its source atoms *before* the declaration is
    aggregated into a `SystemDescriptor`: the aggregated payload must still
    match the resource_id the term declared, not the atoms' current (changed)
    content.
    """

    atoms = AtomicConfiguration(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([2.0]))
    term_obj = ElectronNucleusPotential(atoms)
    declaration = term_obj.describe_physical_operator()
    resource_id = declaration.resource_id

    atoms.positions.add_(100.0)
    atoms.charges.add_(1.0)

    system = SystemDescriptor(
        particle=ParticleSpec(1, 1, 0, 3),
        units="atomic_units",
        terms=(declaration,),
    )

    assert system.resources[resource_id].content_id() == resource_id


def test_en_and_nn_distinct_resources_stay_distinct() -> None:
    atoms_a = _h2_atoms(1.4)
    atoms_b = _h2_atoms(2.0)
    descriptor = build_system_descriptor(
        _h2_particle(),
        {
            "kinetic": KineticEnergy(),
            "electron_nucleus": ElectronNucleusPotential(atoms_a),
            "nucleus_nucleus": NucleusNucleusPotential(atoms_b),
        },
    )

    en_term = next(t for t in descriptor.terms if t.slot == "electron_nucleus")
    nn_term = next(t for t in descriptor.terms if t.slot == "nucleus_nucleus")
    assert en_term.resource_id != nn_term.resource_id
    assert len(descriptor.resources) == 2


# --- Slot normalization: slot is the resolved execution name ---------------


def test_mapping_slot_is_the_configured_key_not_the_term_label() -> None:
    descriptor = build_system_descriptor(_hooke_1e_particle(), {"trap_energy": HarmonicTrap(omega=0.5)})

    assert [term.slot for term in descriptor.terms] == ["trap_energy"]


def test_sequence_slot_is_the_normalized_class_name() -> None:
    descriptor = build_system_descriptor(_hooke_1e_particle(), [KineticEnergy()])

    assert [term.slot for term in descriptor.terms] == ["kinetic_energy"]


def test_renamed_slot_preserves_system_id() -> None:
    renamed = build_system_descriptor(_hooke_1e_particle(), {"trap_energy": HarmonicTrap(omega=0.5)})
    original = build_system_descriptor(_hooke_1e_particle(), {"harmonic_trap": HarmonicTrap(omega=0.5)})

    assert renamed.system_id() == original.system_id()
    assert renamed.terms[0].slot != original.terms[0].slot


def test_renamed_slot_on_resource_bearing_term_preserves_resource() -> None:
    """The slot-rename path in `build_system_descriptor` must carry `resource` forward.

    `dataclasses.replace` cannot be used for the rename (see the comment in
    `build_system_descriptor`): it would fall back to `resource`'s `InitVar`
    default (`None`) instead of the original payload.
    """

    atoms = _helium_atoms()
    descriptor = build_system_descriptor(
        _helium_particle(), {"nuclear_potential": ElectronNucleusPotential(atoms)}
    )

    term = next(t for t in descriptor.terms if t.slot == "nuclear_potential")
    assert term.resource is not None
    assert term.resource.content_id() == atoms.content_id()
    assert descriptor.resources[term.resource_id].content_id() == atoms.content_id()


# --- Finite parameters and signed-zero canonicalization ---------------------


@pytest.mark.parametrize("bad_omega", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_trap_parameter_is_rejected(bad_omega: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        HarmonicTrap(omega=bad_omega).describe_physical_operator()


def test_signed_zero_eps_is_canonicalized_for_electron_electron() -> None:
    positive = build_system_descriptor(_hooke_1e_particle(), {"electron_electron": ElectronElectronInteraction(eps=0.0)})
    negative = build_system_descriptor(_hooke_1e_particle(), {"electron_electron": ElectronElectronInteraction(eps=-0.0)})

    assert positive.system_id() == negative.system_id()


def test_signed_zero_eps_is_canonicalized_for_electron_nucleus() -> None:
    atoms = _helium_atoms()
    positive = build_system_descriptor(
        _helium_particle(), {"electron_nucleus": ElectronNucleusPotential(atoms, eps=0.0)}
    )
    negative = build_system_descriptor(
        _helium_particle(), {"electron_nucleus": ElectronNucleusPotential(atoms, eps=-0.0)}
    )

    assert positive.system_id() == negative.system_id()


def test_eps_discriminates_independently_for_electron_electron() -> None:
    zero = build_system_descriptor(_hooke_1e_particle(), {"electron_electron": ElectronElectronInteraction(eps=0.0)})
    nonzero = build_system_descriptor(
        _hooke_1e_particle(), {"electron_electron": ElectronElectronInteraction(eps=1e-4)}
    )

    assert zero.system_id() != nonzero.system_id()


def test_eps_discriminates_independently_for_electron_nucleus() -> None:
    atoms = _helium_atoms()
    zero = build_system_descriptor(
        _helium_particle(), {"electron_nucleus": ElectronNucleusPotential(atoms, eps=0.0)}
    )
    nonzero = build_system_descriptor(
        _helium_particle(), {"electron_nucleus": ElectronNucleusPotential(atoms, eps=1e-4)}
    )

    assert zero.system_id() != nonzero.system_id()


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
