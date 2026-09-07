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


def test_seed_namespaces_encode_the_new_8_12_48_policy() -> None:
    assert stage_coordinate.seed_labels("O1") == tuple(range(820_001, 820_009))
    assert stage_coordinate.seed_labels("R") == tuple(range(850_001, 850_013))
    assert stage_coordinate.seed_labels("F") == tuple(range(860_001, 860_049))
    streams = stage_coordinate.seed_namespace("O1", 820_001)
    assert tuple(streams) == stage_coordinate.SEED_STREAMS
    assert len(set(streams.values())) == len(streams)
    assert all(0 < value < 2**63 for value in streams.values())


@pytest.mark.parametrize("stage", ["O1", "R", "F"])
def test_each_seed_label_has_pairwise_distinct_rng_streams(stage: str) -> None:
    namespaces = stage_coordinate.seed_namespaces(stage)
    signatures = {tuple(streams.items()) for streams in namespaces.values()}
    assert len(namespaces) == stage_coordinate.stage_definition(stage).seeds_per_point
    assert len(signatures) == len(namespaces)


def test_materialized_union_deduplicates_only_resolved_scientific_identity(tmp_path: Path) -> None:
    control = {"scientific_identity": {"model": "control"}, "payload": {"updates": 50_000}}
    duplicate = {"scientific_identity": {"model": "control"}, "payload": {"updates": 50_000}}
    variant = {"scientific_identity": {"model": "hybrid"}, "payload": {"updates": 50_000}}
    cells = stage_coordinate.materialize_stage(
        "O1",
        [control, duplicate, variant],
        stage_coordinate.OptimizerCell("adam", "available"),
        tmp_path,
    )
    assert len(cells) == 16
    assert len({cell.content_hash for cell in cells}) == 16
    assert len({cell.output_path for cell in cells}) == 16
    assert all(cell.output_path.is_absolute() for cell in cells)
    assert all(
        cell.manifest["scientific_identity"]["optimizer_cell"] == {"method": "adam", "status": "available"}
        for cell in cells
    )
    with pytest.raises(TypeError):
        cells[0].manifest["stage"] = "F"  # type: ignore[index]
    assert stage_coordinate.content_hash(cells[0].manifest) == cells[0].content_hash


def test_topology_projection_keeps_hash_and_path_invariant_in_open_routes(tmp_path: Path) -> None:
    base = {
        "scientific_identity": {"architecture": "control", "width": 16},
        "payload": {"updates": 50_000, "input_features": ("r12",)},
    }
    topology_variant = {
        "scientific_identity": {"architecture": "control", "width": 16, "deviceId": 7},
        "payload": {"updates": 50_000, "input_features": ("r12",), "hostName": "node-1"},
    }
    optimizer = stage_coordinate.OptimizerCell("adam", "available")
    baseline = stage_coordinate.materialize_stage("O1", [base], optimizer, tmp_path)
    varied = stage_coordinate.materialize_stage("O1", [topology_variant], optimizer, tmp_path)
    assert [cell.content_hash for cell in varied] == [cell.content_hash for cell in baseline]
    assert [cell.output_path for cell in varied] == [cell.output_path for cell in baseline]


def test_materializer_rejects_conflicting_resolved_scientific_identity(tmp_path: Path) -> None:
    first = {"scientific_identity": {"model": "control"}, "payload": {"updates": 50_000}}
    conflict = {"scientific_identity": {"model": "control"}, "payload": {"updates": 25_000}}
    with pytest.raises(stage_coordinate.MaterializationError, match="conflicting"):
        stage_coordinate.materialize_stage(
            "O1", [first, conflict], stage_coordinate.OptimizerCell("adam", "available"), tmp_path
        )


def test_unavailable_optimizer_is_explicit_and_never_becomes_adam(tmp_path: Path) -> None:
    unavailable = stage_coordinate.OptimizerCell("linear_method", "unavailable", "full tangent unavailable")
    cells = stage_coordinate.materialize_stage(
        "O1",
        [{"scientific_identity": {"model": "control"}, "payload": {"updates": 50_000}}],
        unavailable,
        tmp_path,
    )
    assert len(cells) == 8
    assert all(
        cell.manifest["scientific_identity"]["optimizer_cell"]
        == {
            "method": "linear_method",
            "status": "unavailable",
            "unavailable_reason": "full tangent unavailable",
        }
        for cell in cells
    )
    with pytest.raises(stage_coordinate.MaterializationError, match="need a reason"):
        stage_coordinate.OptimizerCell("linear_method", "unavailable")


def test_l2a_content_screen_reaches_frozen_delegated_configuration(tmp_path: Path) -> None:
    forbidden = {
        "scientific_identity": MappingProxyType({"architecture": "control"}),
        "payload": {
            "updates": 50_000,
            "nested": MappingProxyType({"values": (-2.903724377034119598,)}),
        },
    }
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="reference energy"):
        stage_coordinate.materialize_stage(
            "O1", [forbidden], stage_coordinate.OptimizerCell("adam", "available"), tmp_path
        )


def test_l2b_closes_its_seed_identity_vocabulary() -> None:
    manifest = {
        "schema": stage_coordinate.TRAIN_MANIFEST_SCHEMA,
        "stage": "O1",
        "scientific_identity": {"architecture": "control"},
        "seed_identity": {"stage": "O1", "label": 820_001, "namespace": "fresh-training", "rank": 0},
        "payload": {"updates": 50_000, "configuration": {"updates": 50_000}},
    }
    with pytest.raises(stage_coordinate.ManifestSchemaError, match="seed_identity keys mismatch"):
        stage_coordinate.validate_materialized_manifest(manifest)


def test_content_hash_is_canonical_and_rejects_nonfinite_values() -> None:
    assert stage_coordinate.content_hash({"a": 1, "b": [2, 3]}) == (
        "efbd0040190fb0871831e606c581f8a66db79d8e2bb836745a70051306956070"
    )
    assert stage_coordinate.content_hash(MappingProxyType({"b": (2, 3), "a": 1})) == (
        "efbd0040190fb0871831e606c581f8a66db79d8e2bb836745a70051306956070"
    )
    with pytest.raises(stage_coordinate.MaterializationError, match="finite JSON"):
        stage_coordinate.content_hash({"bad": float("nan")})


def test_committed_intended_configuration_inventory_materializes_completely(tmp_path: Path) -> None:
    inventory = json.loads(Path(__file__).with_name("intended_configurations.json").read_text())
    cells = stage_coordinate.materialize_intended_configurations(tmp_path)
    expected = sum(
        len(entry["configurations"]) * stage_coordinate.stage_definition(entry["stage"]).seeds_per_point
        for entry in inventory
    )
    assert len(cells) == expected
    assert {cell.manifest["stage"] for cell in cells} == {entry["stage"] for entry in inventory}


def test_real_hi_namespace_family_exposes_the_materialization_api() -> None:
    from tpen.hi.train import v1

    assert "materialize_stage" in v1.__all__
    assert "validate_materialized_manifest" in v1.__all__
    assert v1.content_hash({"a": 1, "b": [2, 3]}) == stage_coordinate.content_hash({"a": 1, "b": [2, 3]})
