"""Tests for the HI v2 stage authority and closed manifest schemas."""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from dataclasses import fields
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


def _configuration(
    scientific_identity: dict[str, object],
    payload: dict[str, object],
    topology: object = None,
) -> dict[str, object]:
    """Build the complete caller-owned materializer input shape."""

    return {
        "scientific_identity": scientific_identity,
        "payload": payload,
        "topology": {} if topology is None else topology,
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
    control = _configuration({"model": "control"}, {"updates": 50_000})
    duplicate = _configuration({"model": "control"}, {"updates": 50_000})
    variant = _configuration({"model": "hybrid"}, {"updates": 50_000})
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


def _topology_probe_keys() -> tuple[str, ...]:
    """Generate a spelling-independent structural-exclusion corpus.

    These names are deliberately derived from an index rather than a topology
    vocabulary.  The property is positional: any key under the designated
    subtree is execution data, and the same key outside it is science.
    """

    return (
        *(f"generated_execution_fact_{index:02d}" for index in range(28)),
        # Both are required same-repository collisions: execution names inside
        # the topology boundary and scientific facts in the paired outer arm.
        "rank",
        "node",
    )


def _paired_configuration(route: str, key: str, *, inside_topology: bool) -> dict[str, object]:
    scientific_identity: dict[str, object] = {"architecture": "control", "width": 16}
    payload: dict[str, object] = {"updates": 50_000, "input_features": ("r12",)}
    topology: dict[str, object] = {}
    if inside_topology:
        topology[key] = ("execution", 1)
    elif route == "scientific_identity":
        scientific_identity[key] = ("science", 1)
    else:
        payload[key] = ("science", 1)
    return _configuration(scientific_identity, payload, MappingProxyType(topology))


@pytest.mark.parametrize("route", ["scientific_identity", "payload.configuration"])
@pytest.mark.parametrize("key", _topology_probe_keys())
def test_topology_exclusion_is_structural_and_open_routes_remain_science(
    tmp_path: Path, route: str, key: str
) -> None:
    """Each generated key runs in both positions, without lexical policy."""

    optimizer = stage_coordinate.OptimizerCell("adam", "available")
    baseline = stage_coordinate.materialize_stage(
        "O1", [_configuration({"architecture": "control", "width": 16}, {"updates": 50_000, "input_features": ("r12",)})], optimizer, tmp_path / "base"
    )
    inside = stage_coordinate.materialize_stage(
        "O1", [_paired_configuration(route, key, inside_topology=True)], optimizer, tmp_path / "inside"
    )
    outside = stage_coordinate.materialize_stage(
        "O1", [_paired_configuration(route, key, inside_topology=False)], optimizer, tmp_path / "outside"
    )
    assert [cell.content_hash for cell in inside] == [cell.content_hash for cell in baseline]
    assert [cell.output_path.name for cell in inside] == [cell.output_path.name for cell in baseline]
    assert [cell.content_hash for cell in outside] != [cell.content_hash for cell in baseline]
    assert [cell.output_path.name for cell in outside] != [cell.output_path.name for cell in baseline]


_SCIENCE_FACT_CASES = (
    # Repository evidence: these are caller-declared scientific coordinates.
    ("max_order", "rg: 130 hits across 8 files; tpen/hi_schema.py:755"),
    ("max_virtual_order", "rg: 36 hits across 4 files; tpen/hi_schema.py:756"),
    ("channels", "rg: 127 hits across 18 files; tpen/hi_schema.py:743"),
    ("n_walkers", "rg: 115 hits across 13 files; tpen/sampling/mala.py:55"),
    ("stage", "intended_configurations.json:3 committed HI coordinate"),
    # The word is topology-related in distributed execution, but this position
    # is caller-declared science: `tpen/config.py:52` names config-tree nodes.
    ("node", "tpen/config.py:52 config-tree node collision"),
    (
        "rank",
        "rg: 15 tpen/distributed.py hits; tensor/matrix rank at tpen/nn/readout/pfaffian.py:128",
    ),
    ("topology", "L2a V6 contract: only the manifest-root member is execution data"),
    # `rg --fixed-strings mesh|shard tpen experiments` found no HI-coordinate
    # declaration; retain these explicitly labeled speculative science probes.
    ("mesh", "speculative: no HI-coordinate repository hit"),
    ("shard", "speculative: no HI-coordinate repository hit"),
)


@pytest.mark.parametrize(("key", "classification_evidence"), _SCIENCE_FACT_CASES)
def test_facts_outside_designated_topology_fork_identity(
    tmp_path: Path, key: str, classification_evidence: str
) -> None:
    """Outside topology, every caller-supplied fact is identity-significant."""

    assert classification_evidence
    optimizer = stage_coordinate.OptimizerCell("adam", "available")
    base = stage_coordinate.materialize_stage(
        "O1", [_configuration({"architecture": "control"}, {"updates": 50_000})], optimizer, tmp_path / "base"
    )
    science = stage_coordinate.materialize_stage(
        "O1",
        [_configuration({"architecture": "control", key: "science-value"}, {"updates": 50_000})],
        optimizer,
        tmp_path / "science",
    )
    assert [cell.content_hash for cell in science] != [cell.content_hash for cell in base]
    assert [cell.output_path.name for cell in science] != [cell.output_path.name for cell in base]


def test_materializer_rejects_conflicting_resolved_scientific_identity(tmp_path: Path) -> None:
    first = _configuration({"model": "control"}, {"updates": 50_000})
    conflict = _configuration({"model": "control"}, {"updates": 25_000})
    with pytest.raises(stage_coordinate.MaterializationError, match="conflicting"):
        stage_coordinate.materialize_stage(
            "O1", [first, conflict], stage_coordinate.OptimizerCell("adam", "available"), tmp_path
        )


def test_unavailable_optimizer_is_explicit_and_never_becomes_adam(tmp_path: Path) -> None:
    unavailable = stage_coordinate.OptimizerCell("linear_method", "unavailable", "full tangent unavailable")
    cells = stage_coordinate.materialize_stage(
        "O1",
        [_configuration({"model": "control"}, {"updates": 50_000})],
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
        "topology": MappingProxyType({}),
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
        "topology": {},
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


def test_canonical_hash_recurses_through_frozen_sequences_and_keeps_them_immutable() -> None:
    plain = {"nested": ({"tuple": [1, {"leaf": 2}]},)}
    frozen = stage_coordinate._freeze(plain)
    assert isinstance(frozen, MappingProxyType)
    assert isinstance(frozen["nested"], tuple)
    assert isinstance(frozen["nested"][0], MappingProxyType)
    assert stage_coordinate.content_hash(frozen) == stage_coordinate.content_hash(plain)
    with pytest.raises(TypeError):
        frozen["nested"][0]["tuple"] = ()  # type: ignore[index]


def test_canonical_hash_preserves_the_typed_non_string_key_failure() -> None:
    with pytest.raises(stage_coordinate.MaterializationError, match="identity mapping keys must be strings"):
        stage_coordinate.content_hash({1: "not-json"})


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
    assert "materialize_job_packets" in v1.__all__
    assert v1.RankingStatistic.LOGABS_VARIANCE.value == "logabs_variance"
    assert v1.content_hash({"a": 1, "b": [2, 3]}) == stage_coordinate.content_hash({"a": 1, "b": [2, 3]})


def _packet_source_cells(tmp_path: Path) -> tuple[object, ...]:
    # Source-cell construction has no launch or allocation inputs.  This fixture
    # therefore records deliberately empty execution topology; a future
    # caller that knows rank/device/world-size facts must supply them here,
    # under L2a's designated non-scientific subtree.
    return stage_coordinate.materialize_stage(
        "O1",
        [
            {
                "scientific_identity": {"model": "control"},
                "payload": {"updates": 50_000},
                "topology": {},
            }
        ],
        stage_coordinate.OptimizerCell("adam", "available"),
        tmp_path,
    )


def test_job_packets_are_separate_and_checkpoint_cadence_is_science_input(tmp_path: Path) -> None:
    cadence = stage_coordinate.CheckpointCadence(1_000, (1_000, 2_000))
    packets = stage_coordinate.materialize_job_packets(
        _packet_source_cells(tmp_path),
        cadence,
        {"statistic": "logabs_variance"},
        (
            stage_coordinate.RankingStatistic.LOGABS_VARIANCE,
            stage_coordinate.RankingStatistic.ACCEPTANCE_RATE,
        ),
        {"walkers": 4_096, "burn_in_sweeps": 100},
        ddp_provenance={"launcher": "operator-supplied"},
    )

    assert len(packets.train) == len(_packet_source_cells(tmp_path / "count"))
    assert len(packets.ranking) == len(packets.train) * len(cadence.selected_updates)
    assert len(packets.independent_sampler_test) == len(packets.ranking)
    assert tuple(field.name for field in fields(stage_coordinate.TrainingPacket)) == (
        "cell",
        "checkpoint_cadence",
        "ddp_provenance",
    )
    assert cadence.science_parameters() == {
        "every_n_updates": 1_000,
        "selected_updates": (1_000, 2_000),
    }
    cell_paths = {packet.cell.content_hash: packet.cell.output_path for packet in packets.train}
    for packet in (*packets.ranking, *packets.independent_sampler_test):
        assert packet.checkpoint_path.parent.name == "checkpoints"
        assert packet.checkpoint_path.parent.parent == cell_paths[packet.source_content_hash]


@pytest.mark.parametrize(
    "metric",
    [
        "eloc",
        "E_local",
        "elocal",
        "e_loc",
        "localEnergy",
        "variationalEnergy",
        "energies",
        "mean_E",
        "E_var",
        "E0",
        "hamiltonian",
    ],
)
def test_ranking_packet_refuses_untyped_energy_spellings(tmp_path: Path, metric: str) -> None:
    with pytest.raises(stage_coordinate.MaterializationError, match="RankingStatistic"):
        stage_coordinate.materialize_job_packets(
            _packet_source_cells(tmp_path),
            stage_coordinate.CheckpointCadence(1_000, (1_000,)),
            {"statistic": "logabs_variance"},
            (metric,),  # type: ignore[arg-type]
            {"walkers": 4_096},
            ddp_provenance={"launcher": "operator-supplied"},
        )


def test_expensive_packet_receives_immutable_independent_sampler_inputs(tmp_path: Path) -> None:
    packets = stage_coordinate.materialize_job_packets(
        _packet_source_cells(tmp_path),
        stage_coordinate.CheckpointCadence(1_000, (1_000,)),
        {"statistic": "logabs_variance"},
        (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
        {"sampler": {"walkers": 4_096}, "burn_in_sweeps": 100},
        ddp_provenance={"launcher": "operator-supplied"},
    )
    inputs = packets.independent_sampler_test[0].independent_sampler_inputs
    assert inputs["sampler"]["walkers"] == 4_096
    with pytest.raises(TypeError):
        inputs["sampler"]["walkers"] = 1  # type: ignore[index]


@pytest.mark.parametrize("packet_route", ["ranking", "independent"])
def test_job_inputs_refuse_unknown_keys(tmp_path: Path, packet_route: str) -> None:
    ranking_inputs = {"statistic": "logabs_variance"}
    independent_inputs = {"walkers": 4_096}
    (ranking_inputs if packet_route == "ranking" else independent_inputs)["unknown"] = "value"
    with pytest.raises(stage_coordinate.MaterializationError, match="unknown input key"):
        stage_coordinate.materialize_job_packets(
            _packet_source_cells(tmp_path),
            stage_coordinate.CheckpointCadence(1_000, (1_000,)),
            ranking_inputs,
            (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
            independent_inputs,
            ddp_provenance={"launcher": "operator-supplied"},
        )


@pytest.mark.parametrize(
    ("packet_route", "reference_value"),
    [
        ("ranking", -2.903724377034119598),
        ("ranking", "-2.903724377034119598"),
        ("ranking", str(-2.903724377034119598)),
        ("ranking", repr(-2.903724377034119598)),
        ("ranking", "-2.903724"),
        ("independent", -2.903724377034119598),
        ("independent", "-2.903724377034119598"),
        ("independent", str(-2.903724377034119598)),
        ("independent", repr(-2.903724377034119598)),
        ("independent", "-2.903724"),
    ],
)
def test_job_inputs_refuse_reference_representations_at_frozen_depth(
    tmp_path: Path, packet_route: str, reference_value: object
) -> None:
    ranking_inputs: dict[str, object] = {"statistic": "logabs_variance"}
    independent_inputs: dict[str, object] = {"walkers": 4_096}
    if packet_route == "ranking":
        ranking_inputs["statistic"] = MappingProxyType({"nested": (reference_value,)})
    else:
        independent_inputs["sampler"] = MappingProxyType({"walkers": (reference_value,)})
    with pytest.raises(stage_coordinate.MaterializationError, match="reference energy"):
        stage_coordinate.materialize_job_packets(
            _packet_source_cells(tmp_path),
            stage_coordinate.CheckpointCadence(1_000, (1_000,)),
            ranking_inputs,
            (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
            independent_inputs,
            ddp_provenance={"launcher": "operator-supplied"},
        )


@pytest.mark.parametrize("packet_route", ["ranking", "independent"])
def test_decimal_rule_reaches_reference_value_at_legal_top_level_key(
    tmp_path: Path, packet_route: str
) -> None:
    ranking_inputs: dict[str, object] = {"statistic": "logabs_variance"}
    independent_inputs: dict[str, object] = {"walkers": 4_096}
    (ranking_inputs if packet_route == "ranking" else independent_inputs)[
        "statistic" if packet_route == "ranking" else "walkers"
    ] = -2.9037244
    with pytest.raises(stage_coordinate.MaterializationError, match="reference energy"):
        stage_coordinate.materialize_job_packets(
            _packet_source_cells(tmp_path),
            stage_coordinate.CheckpointCadence(1_000, (1_000,)),
            ranking_inputs,
            (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
            independent_inputs,
            ddp_provenance={},
        )


def test_decimal_rule_reaches_reference_value_at_legal_sampler_depth(tmp_path: Path) -> None:
    with pytest.raises(stage_coordinate.MaterializationError, match="reference energy"):
        stage_coordinate.materialize_job_packets(
            _packet_source_cells(tmp_path),
            stage_coordinate.CheckpointCadence(1_000, (1_000,)),
            {"statistic": "logabs_variance"},
            (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
            {"sampler": {"walkers": -2.9037244}},
            ddp_provenance={},
        )


@pytest.mark.parametrize(
    "key", ["e0", "target", "E_exact", "benchmark", "gold", "threshold", "reference_energy", "energy"]
)
def test_embedded_cell_manifest_refuses_reference_content(tmp_path: Path, key: str) -> None:
    cell = _packet_source_cells(tmp_path)[0]
    contaminated = stage_coordinate.MaterializedCell(
        manifest={
            **cell.manifest,
            "scientific_identity": {**cell.manifest["scientific_identity"], key: -2.903724377034119598},
        },
        content_hash=cell.content_hash,
        output_path=cell.output_path,
        seed_streams=cell.seed_streams,
    )
    with pytest.raises(stage_coordinate.MaterializationError, match="reference energy"):
        stage_coordinate.materialize_job_packets(
            (contaminated,),
            stage_coordinate.CheckpointCadence(1_000, (1_000,)),
            {"statistic": "logabs_variance"},
            (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
            {"walkers": 4_096},
            ddp_provenance={},
        )


def test_job_packet_refuses_callable_payload(tmp_path: Path) -> None:
    with pytest.raises(stage_coordinate.MaterializationError, match="callable"):
        stage_coordinate.materialize_job_packets(
            _packet_source_cells(tmp_path),
            stage_coordinate.CheckpointCadence(1_000, (1_000,)),
            {"statistic": lambda: None},
            (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
            {"walkers": 4_096},
            ddp_provenance={},
        )


def test_packet_refuses_checkpoints_beyond_stage_horizon(tmp_path: Path) -> None:
    with pytest.raises(stage_coordinate.MaterializationError, match="stage horizon"):
        stage_coordinate.materialize_job_packets(
            _packet_source_cells(tmp_path),
            stage_coordinate.CheckpointCadence(1_000, (999_000_000,)),
            {"statistic": "logabs_variance"},
            (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
            {"walkers": 4_096},
            ddp_provenance={},
        )


def test_packet_accepts_checkpoint_at_exact_stage_horizon(tmp_path: Path) -> None:
    packets = stage_coordinate.materialize_job_packets(
        _packet_source_cells(tmp_path), stage_coordinate.CheckpointCadence(1, (50_000,)),
        {"statistic": "logabs_variance"}, (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
        {"walkers": 4_096}, ddp_provenance={},
    )
    assert packets.ranking


@pytest.mark.parametrize("significant_figures", range(7, 17))
def test_packet_refuses_decimal_agreement_after_rounding(tmp_path: Path, significant_figures: int) -> None:
    rounded = format(-2.903724377034119598, f".{significant_figures}g")
    with pytest.raises(stage_coordinate.MaterializationError, match="reference energy"):
        stage_coordinate.materialize_job_packets(
            _packet_source_cells(tmp_path), stage_coordinate.CheckpointCadence(1_000, (1_000,)),
            {"statistic": rounded}, (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
            {"walkers": 4_096}, ddp_provenance={},
        )


def test_packet_refuses_unknown_nested_sampler_key(tmp_path: Path) -> None:
    with pytest.raises(stage_coordinate.MaterializationError, match="unknown input key 'unknown'"):
        stage_coordinate.materialize_job_packets(
            _packet_source_cells(tmp_path), stage_coordinate.CheckpointCadence(1_000, (1_000,)),
            {"statistic": "logabs_variance"}, (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
            {"sampler": {"unknown": 1}}, ddp_provenance={},
        )


def test_topology_collision_is_reported_but_scientific_control_is_silent(tmp_path: Path) -> None:
    optimizer = stage_coordinate.OptimizerCell("adam", "available")
    base = {"scientific_identity": {"model": "control"}, "payload": {"updates": 50_000}}
    with pytest.warns(UserWarning, match="execution-topology collision"):
        stage_coordinate.materialize_stage(
            "O1", [{**base, "topology": {"world_size": 8}}, {**base, "topology": {"world_size": 16}}], optimizer, tmp_path
        )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        stage_coordinate.materialize_stage(
            "O1", [{**base, "scientific_identity": {"model": "eight"}, "topology": {}}, {**base, "scientific_identity": {"model": "sixteen"}, "topology": {}}], optimizer, tmp_path / "control"
        )
    assert not captured


def test_ddp_provenance_is_preserved_across_distinct_packet_arms(tmp_path: Path) -> None:
    first_cells = _packet_source_cells(tmp_path / "first")
    second_cells = _packet_source_cells(tmp_path / "second")
    cadence = stage_coordinate.CheckpointCadence(1_000, (1_000,))
    first = stage_coordinate.materialize_job_packets(
        first_cells,
        cadence,
        {"statistic": "logabs_variance"},
        (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
        {"walkers": 4_096},
        ddp_provenance={"world_size": 1, "launcher": "first"},
    )
    second = stage_coordinate.materialize_job_packets(
        second_cells,
        cadence,
        {"statistic": "logabs_variance"},
        (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
        {"walkers": 4_096},
        ddp_provenance={"world_size": 8, "launcher": "second"},
    )
    assert [packet.cell.content_hash for packet in first.train] == [packet.cell.content_hash for packet in second.train]
    for packet in (*first.train, *first.ranking, *first.independent_sampler_test):
        assert packet.ddp_provenance == {"world_size": 1, "launcher": "first"}
    for packet in (*second.train, *second.ranking, *second.independent_sampler_test):
        assert packet.ddp_provenance == {"world_size": 8, "launcher": "second"}
