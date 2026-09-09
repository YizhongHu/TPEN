"""Tests for the HI v2 stage authority and closed manifest schemas."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import MappingProxyType

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
        "seed_identity": {"stage": "O1", "label": "000001", "namespace": "hi-v2"},
        "payload": {"updates": 50_000, "configuration": {"model": ("control",)}},
        "topology": MappingProxyType({"launcher": ("single",)}),
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
    manifest["payload"] = {
        "updates": 50_000,
        "configuration": MappingProxyType({"nested": ({spelling: -2.903724377034119598},)}),
    }
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="reference energy|forbidden train content"):
        stage_coordinate.validate_train_manifest(manifest)


@pytest.mark.parametrize("container", ["payload"])
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
        ("payload", "updates"),
        ("payload", "configuration"),
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


def test_train_manifest_accepts_mapping_proxies_at_each_enumerated_level() -> None:
    manifest = _train_manifest()
    proxied = MappingProxyType(
        {
            **manifest,
            "scientific_identity": MappingProxyType(manifest["scientific_identity"]),
            "seed_identity": MappingProxyType(manifest["seed_identity"]),
            "payload": MappingProxyType(manifest["payload"]),
            "topology": MappingProxyType(manifest["topology"]),
        }
    )
    stage_coordinate.validate_train_manifest(proxied)


def test_designated_topology_subtree_accepts_generated_spellings_without_lexical_policy() -> None:
    """Topology is structurally isolated; L2b owns its wholesale hash exclusion."""

    manifest = _train_manifest()
    generated = {
        f"execution_fact_{index:02d}_{chr(65 + index % 26)}": index
        for index in range(32)
    }
    generated.update(
        {
            "local_rank": 3,
            "num_gpus": 8,
            "device_mesh": ("x", "y"),
            "visible_devices": MappingProxyType({"CUDA_VISIBLE_DEVICES": "0,1"}),
            "global_rank": 7,
            "tp_size": 2,
            "node_rank": 1,
            "gpu_id": 0,
        }
    )
    manifest["topology"] = MappingProxyType(generated)
    stage_coordinate.validate_train_manifest(manifest)


def test_topology_is_required_and_mesh_outside_it_remains_caller_declared_science() -> None:
    manifest = _train_manifest()
    del manifest["topology"]
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="manifest keys mismatch"):
        stage_coordinate.validate_train_manifest(manifest)

    coarse = _train_manifest()
    fine = _train_manifest()
    coarse["payload"] = {"updates": 50_000, "configuration": {"mesh": "coarse"}}
    fine["payload"] = {"updates": 50_000, "configuration": {"mesh": "fine"}}
    stage_coordinate.validate_train_manifest(coarse)
    stage_coordinate.validate_train_manifest(fine)


@pytest.mark.parametrize(
    "topology",
    [
        {"energy": -2.903724377034119598},
        {"energy": "-2.903724377034119598"},
        MappingProxyType({"nested": (MappingProxyType({"energy": -2.903724377034119598}),)}),
        {"nested": {-2.903724377034119598}},
        {"referenceEnergy": "forbidden"},
    ],
)
def test_topology_subtree_keeps_the_train_content_screen(topology: object) -> None:
    """Pin the content firewall inside the L2b-excluded topology subtree."""

    manifest = _train_manifest()
    manifest["topology"] = topology
    with pytest.raises(stage_coordinate.ManifestSchemaError):
        stage_coordinate.validate_train_manifest(manifest)


def test_delegated_subtrees_accept_l2b_vocabulary_without_closing_it() -> None:
    manifest = _train_manifest()
    manifest["scientific_identity"] = MappingProxyType({"caller_choice": ("A", "B")})
    manifest["seed_identity"] = MappingProxyType({"stage": "O1", "label": "one", "namespace": "n"})
    manifest["payload"] = {
        "updates": 50_000,
        "configuration": MappingProxyType({"caller_config": (MappingProxyType({"depth": 2}),)}),
    }
    stage_coordinate.validate_train_manifest(manifest)


def test_delegated_subtrees_reject_non_string_keys_and_accuracy_content() -> None:
    manifest = _train_manifest()
    manifest["scientific_identity"] = MappingProxyType({1: "not structural"})
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="keys must be strings"):
        stage_coordinate.validate_train_manifest(manifest)

    manifest = _train_manifest()
    manifest["payload"] = {
        "updates": 50_000,
        "configuration": MappingProxyType({"nested": ({"accuracyBand": "forbidden"},)}),
    }
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="forbidden train content"):
        stage_coordinate.validate_train_manifest(manifest)


class _AttributeCarrier:
    """Unsupported caller object used to pin the D9 structural backstop."""

    def __init__(self, value: object) -> None:
        self.value = value


@pytest.mark.parametrize(
    "smuggled",
    [
        "-2.903724377034119598",
        {-2.903724377034119598},
        _AttributeCarrier(-2.903724377034119598),
    ],
)
def test_delegated_content_screen_rejects_string_set_and_attribute_smuggling(smuggled: object) -> None:
    manifest = _train_manifest()
    manifest["payload"] = {
        "updates": 50_000,
        "configuration": MappingProxyType({"nested": (smuggled,)}),
    }
    with pytest.raises(stage_coordinate.ManifestSchemaError):
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


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest.__setitem__("stage", ["O1"]),
        lambda manifest: manifest.__setitem__("scientific_identity", ("not", "a", "mapping")),
        lambda manifest: manifest.__setitem__("seed_identity", ("not", "a", "mapping")),
        lambda manifest: manifest.__setitem__("topology", ("not", "a", "mapping")),
        lambda manifest: manifest.__setitem__("payload", {"updates": "50k", "configuration": {}}),
        lambda manifest: manifest.__setitem__("payload", {"updates": 50_000, "configuration": ()}),
    ],
)
def test_every_structural_guard_raises_the_typed_error(mutate: object) -> None:
    manifest = _train_manifest()
    mutate(manifest)
    with pytest.raises(stage_coordinate.ManifestSchemaError):
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
    manifest["payload"] = {"updates": invalid_updates, "configuration": {}}
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
