"""Schema tests for the baseline comparison-system registry.

These tests are deliberately about *shape and evidence discipline*, not physics.
They guard the two failure modes that would silently corrupt the baseline
comparison: a malformed entry, and a reference energy that is present without a
citation (or absent while still claiming to be transcribed).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml

# The path comes from the loader, not a second copy of the literal: a test that
# validates a different file than the loader reads is a test that can pass while
# production code reads something broken.
from experiments.baselines.systems import (
    REGISTRY_PATH,
    RegistryError,
    known_system_ids,
    load_registry,
    system_ids,
)

# Keys every registry entry must carry. Missing any of these is a hard failure:
# downstream collection joins on `id` and reports against
# `reference_energy_hartree`, so a partial entry is worse than no entry.
REQUIRED_KEYS = frozenset(
    {
        "id",
        "n_up",
        "n_down",
        "spatial_dim",
        "hamiltonian",
        "reference_energy_hartree",
        "reference_source",
        "confidence",
    }
)

ALLOWED_CONFIDENCE = frozenset({"transcribed", "unverified"})


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    """Return the parsed registry document.

    Returns
    -------
    dict
        The full ``systems.yaml`` document.
    """

    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    assert isinstance(document, dict), "systems.yaml must parse to a mapping"
    return document


@pytest.fixture(scope="module")
def entries(registry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the registry's system entries.

    Returns
    -------
    list of dict
        One mapping per comparison system.
    """

    systems = registry.get("systems")
    assert isinstance(systems, list) and systems, "systems must be a non-empty list"
    for entry in systems:
        assert isinstance(entry, dict), "each system entry must be a mapping"
    return systems


def _walk_floats(value: Any) -> list[float]:
    """Return every float reachable from ``value``.

    Parameters
    ----------
    value : Any
        Parsed YAML fragment: mapping, sequence, or scalar.

    Returns
    -------
    list of float
        All floats found, in document order.
    """

    if isinstance(value, dict):
        return [found for item in value.values() for found in _walk_floats(item)]
    if isinstance(value, (list, tuple)):
        return [found for item in value for found in _walk_floats(item)]
    # bool is a subclass of int but never a physical quantity here.
    if isinstance(value, float):
        return [value]
    return []


def test_document_header(registry: dict[str, Any]) -> None:
    """The registry declares its schema version and unit convention."""

    assert registry.get("schema_version") == 1
    units = registry.get("units")
    assert units == {"energy": "hartree", "length": "bohr"}


def test_required_keys_present(entries: list[dict[str, Any]]) -> None:
    """Every entry carries the full required key set."""

    for entry in entries:
        missing = REQUIRED_KEYS - set(entry)
        assert not missing, f"{entry.get('id', '<no id>')} is missing keys: {sorted(missing)}"


def test_ids_are_unique_non_empty_strings(entries: list[dict[str, Any]]) -> None:
    """Entry ids are usable as join keys."""

    ids = [entry["id"] for entry in entries]
    for system_id in ids:
        assert isinstance(system_id, str) and system_id.strip(), "id must be a non-empty string"
    assert len(set(ids)) == len(ids), "system ids must be unique"


def test_no_nan_or_infinite_values(registry: dict[str, Any]) -> None:
    """No NaN or infinity anywhere in the document.

    YAML happily parses ``.nan`` and ``.inf``; either would poison an energy
    comparison without raising, so reject them at load time.
    """

    for value in _walk_floats(registry):
        assert math.isfinite(value), f"non-finite value in registry: {value!r}"


def test_spin_counts_are_positive_integers(entries: list[dict[str, Any]]) -> None:
    """Spin sectors are integer counts with at least one spin-up electron.

    The majority sector is required to be positive; ``n_down`` may be zero to
    allow a fully spin-polarized system.
    """

    for entry in entries:
        n_up, n_down = entry["n_up"], entry["n_down"]
        for name, count in (("n_up", n_up), ("n_down", n_down)):
            assert isinstance(count, int) and not isinstance(count, bool), (
                f"{entry['id']}.{name} must be an int"
            )
        assert n_up >= 1, f"{entry['id']}.n_up must be positive"
        assert n_down >= 0, f"{entry['id']}.n_down must be non-negative"
        assert n_up >= n_down, f"{entry['id']} must put the majority sector in n_up"


def test_spatial_dim_is_positive_integer(entries: list[dict[str, Any]]) -> None:
    """Spatial dimension is a positive integer."""

    for entry in entries:
        spatial_dim = entry["spatial_dim"]
        assert isinstance(spatial_dim, int) and not isinstance(spatial_dim, bool)
        assert spatial_dim >= 1, f"{entry['id']}.spatial_dim must be positive"


def test_hamiltonian_terms_are_named(entries: list[dict[str, Any]]) -> None:
    """Each entry lists at least one named Hamiltonian term."""

    for entry in entries:
        hamiltonian = entry["hamiltonian"]
        assert isinstance(hamiltonian, dict), f"{entry['id']}.hamiltonian must be a mapping"
        terms = hamiltonian.get("terms")
        assert isinstance(terms, list) and terms, f"{entry['id']} must declare Hamiltonian terms"
        for term in terms:
            assert isinstance(term, dict), f"{entry['id']} term entries must be mappings"
            name = term.get("term")
            assert isinstance(name, str) and name.strip(), f"{entry['id']} term needs a name"


def test_reference_energy_is_null_or_finite_float(entries: list[dict[str, Any]]) -> None:
    """Reference energies are either absent or real numbers."""

    for entry in entries:
        energy = entry["reference_energy_hartree"]
        if energy is None:
            continue
        assert isinstance(energy, (int, float)) and not isinstance(energy, bool), (
            f"{entry['id']}.reference_energy_hartree must be numeric or null"
        )
        assert math.isfinite(float(energy)), f"{entry['id']} reference energy must be finite"


def test_confidence_matches_evidence(entries: list[dict[str, Any]]) -> None:
    """Confidence labels agree with what the entry actually carries.

    The evidence rule for this program: a number may only be present when it was
    transcribed from a cited source, and a missing number may never be labelled
    as transcribed.
    """

    for entry in entries:
        confidence = entry["confidence"]
        assert confidence in ALLOWED_CONFIDENCE, (
            f"{entry['id']}.confidence must be one of {sorted(ALLOWED_CONFIDENCE)}"
        )
        energy = entry["reference_energy_hartree"]
        source = entry["reference_source"]
        if energy is None:
            assert confidence == "unverified", (
                f"{entry['id']} has no reference energy and cannot be transcribed"
            )
        else:
            assert confidence == "transcribed", (
                f"{entry['id']} carries a reference energy so it must be transcribed"
            )
            assert isinstance(source, str) and source.strip(), (
                f"{entry['id']} carries a reference energy without a citation"
            )


def test_no_facility_absolute_paths() -> None:
    """The registry never hard-codes a facility filesystem path."""

    text = REGISTRY_PATH.read_text(encoding="utf-8")
    assert "/n/netscratch" not in text
    assert "/n/holy" not in text


def _nuclei(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every nucleus declared by an entry's Hamiltonian terms.

    Parameters
    ----------
    entry : dict
        One registry entry.

    Returns
    -------
    list of dict
        Nucleus mappings, in declaration order.
    """

    return [
        nucleus
        for term in entry["hamiltonian"]["terms"]
        for nucleus in term.get("nuclei", [])
    ]


def test_multinuclear_entries_declare_nuclear_repulsion(entries: list[dict[str, Any]]) -> None:
    """Any entry with more than one nucleus includes ``nucleus_nucleus``.

    The reference energies are total energies, which include the nuclear
    repulsion at the stated geometry. An entry that omits the term is not a
    usable run specification: a run built from it would be compared against a
    reference that is a constant away from what the run computes.
    """

    for entry in entries:
        nuclei = _nuclei(entry)
        if len(nuclei) < 2:
            continue
        terms = [term["term"] for term in entry["hamiltonian"]["terms"]]
        assert "nucleus_nucleus" in terms, (
            f"{entry['id']} declares {len(nuclei)} nuclei but no nucleus_nucleus term"
        )


def test_electron_count_matches_total_nuclear_charge(entries: list[dict[str, Any]]) -> None:
    """Every entry with nuclei describes a neutral system.

    Nothing in this registry is an ion, so the electron count must equal the
    summed nuclear charge. This is the cheapest available check on a mistyped
    charge or a mistyped spin sector -- either one silently changes which
    physical system a reference energy is being claimed for.
    """

    for entry in entries:
        nuclei = _nuclei(entry)
        if not nuclei:
            continue
        total_charge = sum(nucleus["charge"] for nucleus in nuclei)
        n_electrons = entry["n_up"] + entry["n_down"]
        assert n_electrons == total_charge, (
            f"{entry['id']} has {n_electrons} electrons against nuclear charge "
            f"{total_charge}; it is an ion or a typo"
        )


def test_loader_ids_match_the_document(entries: list[dict[str, Any]]) -> None:
    """``known_system_ids`` reports exactly the ids the file declares."""

    assert known_system_ids() == frozenset(entry["id"] for entry in entries)


def test_hooke_one_electron_reference_is_distinct_from_pair(entries: list[dict[str, Any]]) -> None:
    """The one-electron analytic reference cannot silently reuse the pair value."""

    by_id = {entry["id"]: entry for entry in entries}
    assert by_id["hooke_one_electron_omega0.5"]["reference_energy_hartree"] == 0.75
    assert by_id["hooke_pair_singlet_omega0.5"]["reference_energy_hartree"] == 2.0
    assert by_id["hooke_one_electron_omega0.5"]["n_up"] + by_id["hooke_one_electron_omega0.5"]["n_down"] == 1


def test_loader_covers_every_reproduced_system() -> None:
    """Every system a baseline run has been emitted for is registered.

    These ids are named by records already emitted on the cluster. Dropping one
    from the registry would not break any adapter -- it would make the record
    uncomparable, which is why the set is pinned here rather than left implicit.
    """

    emitted = {
        "he_atom",
        "li_atom",
        "be_atom",
        "b_atom",
        "n_atom",
        "lih_molecule",
        "h2_molecule",
        "n2_molecule",
    }
    assert emitted <= known_system_ids(), f"unregistered: {sorted(emitted - known_system_ids())}"


def test_loader_rejects_a_missing_registry(tmp_path: Path) -> None:
    """A missing registry raises rather than yielding an empty id set."""

    with pytest.raises(RegistryError, match="cannot read"):
        load_registry(tmp_path / "absent.yaml")


def test_loader_rejects_a_registry_without_systems(tmp_path: Path) -> None:
    """A registry with no ``systems`` list raises."""

    path = tmp_path / "systems.yaml"
    path.write_text("schema_version: 1\nsystems: []\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="non-empty 'systems' list"):
        load_registry(path)


def test_loader_rejects_an_entry_without_an_id(tmp_path: Path) -> None:
    """An entry missing its id raises instead of being skipped."""

    path = tmp_path / "systems.yaml"
    path.write_text("systems:\n  - description: nameless\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="non-empty string id"):
        system_ids(path)
