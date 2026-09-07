"""Unit tests for the HI v2 stage coordinate and manifest firewall."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "he_importance_stage_coordinate",
    Path(__file__).with_name("stage_coordinate.py"),
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
        "reference": {"energy": -2.9037243770341195},
        "accuracy": {"conventional_band": 0.0016},
    }


def test_consolidated_stage_coordinate_has_no_legacy_breadth_stage() -> None:
    definitions = {definition.code: definition for definition in stage_coordinate.STAGE_DEFINITIONS}
    assert tuple(definitions) == ("Q", "O1", "O2", "A", "B", "R", "F")
    assert definitions["Q"].updates == 2_000
    assert definitions["O1"].maximum_lineages == 1_920
    assert definitions["A"].maximum_lineages == 4_064
    assert definitions["R"].seeds_per_point == 12
    assert definitions["F"].updates is None


def test_train_manifest_is_closed_and_reference_free() -> None:
    stage_coordinate.validate_train_manifest(_train_manifest())

    reference = _train_manifest()
    reference["payload"] = {"reference_energy": -2.9037243770341195}
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="reference_energy"):
        stage_coordinate.validate_train_manifest(reference)

    accuracy = _train_manifest()
    accuracy["payload"] = {"decision": {"accuracy_band": 0.0016}}
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="accuracy_band"):
        stage_coordinate.validate_train_manifest(accuracy)

    extra = _train_manifest()
    extra["reference"] = {"energy": -2.9037243770341195}
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="keys mismatch"):
        stage_coordinate.validate_train_manifest(extra)


def test_evaluation_manifest_is_a_distinct_schema_that_owns_reference_inputs() -> None:
    stage_coordinate.validate_evaluation_manifest(_evaluation_manifest())
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="keys mismatch"):
        stage_coordinate.validate_train_manifest(_evaluation_manifest())


@pytest.mark.parametrize("identity_key", ["rank", "device", "gpu_count", "world_size"])
def test_physical_topology_cannot_change_a_scientific_or_seed_identity(identity_key: str) -> None:
    manifest = _train_manifest()
    manifest["scientific_identity"] = {"architecture": "control", identity_key: 0}
    with pytest.raises(stage_coordinate.ManifestSchemaError, match=identity_key):
        stage_coordinate.validate_train_manifest(manifest)

    manifest = _train_manifest()
    manifest["seed_identity"] = {"stage_seed": 820001, identity_key: 0}
    with pytest.raises(stage_coordinate.ManifestSchemaError, match=identity_key):
        stage_coordinate.validate_train_manifest(manifest)
