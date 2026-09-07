"""Tests for the HI v2 stage authority and closed manifest schemas."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "he_importance_stage_coordinate", Path(__file__).with_name("stage_coordinate.py")
)
assert _SPEC is not None and _SPEC.loader is not None
stage_coordinate = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stage_coordinate
_SPEC.loader.exec_module(stage_coordinate)


def _train_manifest() -> dict[str, object]:
    return {
        "schema": stage_coordinate.TRAIN_MANIFEST_SCHEMA,
        "stage": "O1",
        "scientific_identity": {"architecture": "control", "optimizer": "adam"},
        "seed_identity": {"stage_seed": 820001, "lineage": 0},
        "payload": {"updates": 50_000},
    }


def _evaluation_manifest() -> dict[str, object]:
    return {
        **_train_manifest(),
        "schema": stage_coordinate.EVALUATION_MANIFEST_SCHEMA,
        "reference": {"energy": -2.903724377034119598},
        "accuracy": {"conventional_band": 0.0016},
    }


def test_all_35_transcribed_stage_fields_are_pinned_against_committed_authority() -> None:
    expected = [
        {"code": "Q", "purpose": "mechanics", "seeds_per_point": 2, "maximum_lineages": 480, "updates": 2_000},
        {"code": "O1", "purpose": "scientific", "seeds_per_point": 8, "maximum_lineages": 1_920, "updates": 50_000},
        {"code": "O2", "purpose": "scientific", "seeds_per_point": 8, "maximum_lineages": 320, "updates": 50_000},
        {"code": "A", "purpose": "scientific", "seeds_per_point": 8, "maximum_lineages": 4_064, "updates": 50_000},
        {"code": "B", "purpose": "scientific", "seeds_per_point": 8, "maximum_lineages": 960, "updates": 50_000},
        {"code": "R", "purpose": "scientific", "seeds_per_point": 12, "maximum_lineages": 768, "updates": 50_000},
        {"code": "F", "purpose": "confirmation", "seeds_per_point": 48, "maximum_lineages": 96, "updates": None},
    ]
    authority = json.loads(Path(__file__).with_name("stage_coordinate_authority.json").read_text())
    assert authority == expected
    assert [definition.__dict__ for definition in stage_coordinate.STAGE_DEFINITIONS] == expected


@pytest.mark.parametrize(
    "spelling",
    [
        "worldSize", "deviceId", "nodeCount", "workerIndex", "referenceEnergy", "accuracyBand",
        "ranks", "devices", "gpus", "references", "accuracies",
        "ref_energy", "exact_energy", "baseline_energy", "target_energy",
        "e0", "target", "E_exact", "benchmark", "gold", "threshold",
    ],
)
def test_train_manifest_refuses_realized_falsifier_spellings_at_nested_levels(spelling: str) -> None:
    manifest = _train_manifest()
    manifest["payload"] = {"updates": 50_000, spelling: -2.903724377034119598}
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="payload keys mismatch"):
        stage_coordinate.validate_train_manifest(manifest)


@pytest.mark.parametrize("container", ["scientific_identity", "seed_identity", "payload"])
def test_train_manifest_refuses_unknown_keys_at_every_nested_level(container: str) -> None:
    manifest = _train_manifest()
    nested = dict(manifest[container])
    nested["unknown"] = "refuses"
    manifest[container] = nested
    with pytest.raises(stage_coordinate.ManifestSchemaError, match=f"{container} keys mismatch"):
        stage_coordinate.validate_train_manifest(manifest)


@pytest.mark.parametrize(
    ("container", "missing_key"),
    [
        ("scientific_identity", "architecture"),
        ("seed_identity", "lineage"),
        ("payload", "updates"),
    ],
)
def test_train_manifest_refuses_missing_keys_at_every_nested_level(
    container: str, missing_key: str
) -> None:
    manifest = _train_manifest()
    nested = dict(manifest[container])
    del nested[missing_key]
    manifest[container] = nested
    with pytest.raises(stage_coordinate.ManifestSchemaError, match=f"{container} keys mismatch"):
        stage_coordinate.validate_train_manifest(manifest)


def test_train_manifest_refuses_unknown_top_level_key() -> None:
    manifest = _train_manifest()
    manifest["reference"] = {"energy": -2.903724377034119598}
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="manifest keys mismatch"):
        stage_coordinate.validate_train_manifest(manifest)


@pytest.mark.parametrize(
    "missing_key",
    ["schema", "stage", "scientific_identity", "seed_identity", "payload"],
)
def test_train_manifest_requires_each_top_level_key_with_typed_failure(missing_key: str) -> None:
    """Kill the exact-membership-to-subset mutant with every common key."""

    manifest = _train_manifest()
    del manifest[missing_key]
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="manifest keys mismatch"):
        stage_coordinate.validate_train_manifest(manifest)


def test_train_manifest_refuses_an_unknown_stage() -> None:
    manifest = _train_manifest()
    manifest["stage"] = "legacy-256-breadth"
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="unknown HI stage"):
        stage_coordinate.validate_train_manifest(manifest)


@pytest.mark.parametrize("container", ["reference", "accuracy"])
def test_evaluation_manifest_refuses_unknown_keys_at_every_nested_level(container: str) -> None:
    manifest = _evaluation_manifest()
    nested = dict(manifest[container])
    nested["unknown"] = "refuses"
    manifest[container] = nested
    with pytest.raises(stage_coordinate.ManifestSchemaError, match=f"{container} keys mismatch"):
        stage_coordinate.validate_evaluation_manifest(manifest)


@pytest.mark.parametrize("container", ["reference", "accuracy"])
def test_evaluation_manifest_refuses_missing_keys_at_every_nested_level(container: str) -> None:
    manifest = _evaluation_manifest()
    manifest[container] = {}
    with pytest.raises(stage_coordinate.ManifestSchemaError, match=f"{container} keys mismatch"):
        stage_coordinate.validate_evaluation_manifest(manifest)


@pytest.mark.parametrize("invalid_updates", [[50_000], (50_000,)])
def test_train_manifest_refuses_list_and_tuple_payloads(invalid_updates: object) -> None:
    manifest = _train_manifest()
    manifest["payload"] = {"updates": invalid_updates}
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="updates must be int"):
        stage_coordinate.validate_train_manifest(manifest)


def test_evaluation_manifest_is_distinct_and_owns_reference_inputs() -> None:
    stage_coordinate.validate_evaluation_manifest(_evaluation_manifest())
    assert stage_coordinate.EVALUATION_MANIFEST_SCHEMA != stage_coordinate.TRAIN_MANIFEST_SCHEMA


def test_schema_string_equality_is_enforced() -> None:
    manifest = _train_manifest()
    manifest["schema"] = "he-importance/train/v0"
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="expected schema"):
        stage_coordinate.validate_train_manifest(manifest)
