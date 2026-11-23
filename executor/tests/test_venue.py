"""Strict VenueProfile loader tests using real on-disk JSON fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen_executor.venue import VenueProfileRefusal, load_venue_profile


ROOT = Path(__file__).parents[2]
VENUES = ROOT / "experiments" / "venues"
FIXTURES = Path(__file__).parent / "fixtures" / "venues"


MUTANTS = (
    {
        "id": "M1",
        "target": "executor/src/tpen_executor/venue.py",
        "old": 'if mapping.get("schema") != cls.schema:',
        "new": "if False:",
        "killer": "test_wrong_schema_and_unknown_keys_are_refused",
    },
    {
        "id": "M2",
        "target": "executor/src/tpen_executor/venue.py",
        "old": "if len(values) != len(set(values)):",
        "new": "if False:",
        "killer": "test_duplicate_queue_and_stratum_names_are_refused",
    },
    {
        "id": "M3",
        "target": "executor/src/tpen_executor/venue.py",
        "old": "if set(data) != expected:",
        "new": "if False:",
        "killer": "test_wrong_schema_and_unknown_keys_are_refused",
    },
    {
        "id": "M4",
        "target": "executor/src/tpen_executor/venue.py",
        "old": "if type(self.memory_mib) is not int or self.memory_mib <= 0:",
        "new": "if type(self.memory_mib) is not int:",
        "killer": "test_nonpositive_or_boolean_memory_is_refused",
    },
    {
        "id": "M5",
        "target": "executor/src/tpen_executor/venue.py",
        "old": "object_pairs_hook=_reject_duplicate_keys,",
        "new": "object_pairs_hook=dict,",
        "killer": "test_duplicate_json_key_is_refused",
    },
    {
        "id": "M6",
        "target": "executor/src/tpen_executor/venue.py",
        "old": 'if candidate.suffix != ".json":',
        "new": "if False:",
        "killer": "test_unsupported_suffix_is_refused",
    },
    {
        "id": "M7",
        "target": "executor/src/tpen_executor/venue.py",
        "old": 'fragment.encode("utf-8")',
        "new": "# no encoding validation",
        "killer": "test_lone_surrogates_are_refused",
    },
)


def test_cannon_profile_decodes_immutable_facts() -> None:
    profile = load_venue_profile(VENUES / "cannon.json")

    assert profile.venue == "cannon"
    assert profile.scheduler == "slurm"
    assert profile.queues == ("test", "gpu_test", "seas_gpu", "kozinsky_gpu", "gpu")
    assert profile.gpu_strata[0].name == "a100"
    assert profile.gpu_strata[0].accelerator == "cuda"
    assert profile.gpu_strata[0].gpu_name_contains == ("A100",)
    assert profile.gpu_strata[0].memory_mib == 81920
    assert profile.gpu_strata[0].mig_mode == "disabled"
    with pytest.raises((AttributeError, TypeError)):
        profile.queues += ("extra",)


def test_equivalent_json_has_equal_canonical_mapping_and_digest() -> None:
    cannon = load_venue_profile(VENUES / "cannon.json")
    equivalent = load_venue_profile(FIXTURES / "equivalent.json")

    assert equivalent.to_dict() == cannon.to_dict()
    assert equivalent.canonical_bytes() == cannon.canonical_bytes()
    assert equivalent.digest() == cannon.digest()


def test_valid_unicode_fragments_canonicalize() -> None:
    for fixture in ("non-ascii-fragment.json", "surrogate-pair-fragment.json"):
        profile = load_venue_profile(FIXTURES / fixture)
        assert profile.canonical_bytes()
        assert len(profile.digest()) == 64


def test_lone_surrogates_are_refused() -> None:
    for fixture in ("lone-high-surrogate.json", "lone-low-surrogate.json"):
        with pytest.raises(VenueProfileRefusal):
            load_venue_profile(FIXTURES / fixture)


@pytest.mark.parametrize(
    "fixture",
    [
        "wrong-schema.json",
        "unknown-key.json",
        "missing-key.json",
        "invalid-scheduler.json",
        "duplicate-queue.json",
        "duplicate-stratum.json",
        "cpu-cross-field.json",
        "cuda-cross-field.json",
        "zero-memory.json",
        "boolean-memory.json",
        "duplicate-key.json",
        "multi-document.json",
    ],
)
def test_invalid_fixtures_are_refused_without_fallback(fixture: str) -> None:
    with pytest.raises(VenueProfileRefusal):
        load_venue_profile(FIXTURES / fixture)


def test_wrong_schema_and_unknown_keys_are_refused() -> None:
    for fixture in ("wrong-schema.json", "unknown-key.json", "missing-key.json"):
        with pytest.raises(VenueProfileRefusal):
            load_venue_profile(FIXTURES / fixture)


def test_duplicate_queue_and_stratum_names_are_refused() -> None:
    for fixture in ("duplicate-queue.json", "duplicate-stratum.json"):
        with pytest.raises(VenueProfileRefusal):
            load_venue_profile(FIXTURES / fixture)


def test_nonpositive_or_boolean_memory_is_refused() -> None:
    for fixture in ("zero-memory.json", "boolean-memory.json"):
        with pytest.raises(VenueProfileRefusal):
            load_venue_profile(FIXTURES / fixture)


def test_duplicate_json_key_is_refused() -> None:
    with pytest.raises(VenueProfileRefusal):
        load_venue_profile(FIXTURES / "duplicate-key.json")


def test_unsupported_suffix_is_refused() -> None:
    for fixture in ("unsupported.toml", "plain.txt"):
        with pytest.raises(VenueProfileRefusal):
            load_venue_profile(FIXTURES / fixture)


def test_multi_document_and_unreadable_paths_are_refused() -> None:
    with pytest.raises(VenueProfileRefusal):
        load_venue_profile(FIXTURES / "multi-document.json")
    with pytest.raises(VenueProfileRefusal):
        load_venue_profile(FIXTURES / "missing.json")


def test_mutant_table_is_product_only_and_named_killers_exist() -> None:
    product = (ROOT / "executor" / "src" / "tpen_executor" / "venue.py").read_text(encoding="utf-8")
    tests = Path(__file__).read_text(encoding="utf-8")
    assert len({entry["id"] for entry in MUTANTS}) == len(MUTANTS)
    for entry in MUTANTS:
        assert entry["target"] == "executor/src/tpen_executor/venue.py"
        assert product.count(entry["old"]) == 1
        assert product.count(entry["new"]) == 0
        assert f"def {entry['killer']}" in tests
