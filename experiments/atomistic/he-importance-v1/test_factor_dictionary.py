"""Contract tests for the He-importance S1a dictionary and resolver."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = Path(__file__).with_name("factor_dictionary.py")
SPEC = importlib.util.spec_from_file_location("he_importance_factor_dictionary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
dictionary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dictionary
SPEC.loader.exec_module(dictionary)


def _assignment(**overrides):
    signs = dictionary.derive_breadth_columns({column: -1 for column in "ABCDEFGH"})
    value = {
        "mixing": "tensor",
        "activation": "pointwise",
        "optimizer": "Adam",
        "lr_multiplier": 1.0,
        **signs,
    }
    value.update(overrides)
    return value


def test_canonical_dictionary_is_byte_stable_and_closed() -> None:
    payload = dictionary.load_dictionary()
    assert dictionary.canonical_yaml(payload) == dictionary.DICTIONARY_PATH.read_text(encoding="utf-8")
    assert tuple(payload["vocabularies"]) == (
        "mixing_packages",
        "activation_packages",
        "optimizer_families",
        "lr_multipliers",
        "binary_factors",
        "anchors",
    )


@pytest.mark.parametrize("field", ["mixing_packages", "activation_packages", "optimizer_families", "lr_multipliers", "binary_factors", "anchors"])
def test_unknown_missing_extra_duplicate_and_reordered_vocabularies_fail(field: str) -> None:
    payload = dictionary.load_dictionary()
    expected = list(payload["vocabularies"][field])
    mutations = [
        expected[:-1],
        [*expected, "unknown"],
        [*expected, expected[-1]],
        [expected[1], expected[0], *expected[2:]],
    ]
    for mutation in mutations:
        broken = deepcopy(payload)
        broken["vocabularies"][field] = mutation
        with pytest.raises(ValueError):
            dictionary.validate_dictionary(broken)


def test_stage_0_values_and_non_adam_centres_are_exact_and_fail_closed() -> None:
    stage_0 = dictionary.load_dictionary()["stage_0"]
    assert stage_0["families"]["Adam"]["lr_candidates"] == [0.00125, 0.0025, 0.005, 0.01]
    assert stage_0["families"]["RAdam"]["fixed"] == {"betas": [0.9, 0.999], "eps": 1e-8}
    assert stage_0["families"]["RMSprop"]["fixed"] == {"alpha": 0.99, "eps": 1e-8, "momentum": 0.0}
    assert stage_0["families"]["SGD-Nesterov"]["fixed"] == {"momentum": 0.9, "nesterov": True}
    assert dictionary.resolve_center_times_multiplier("Adam", 2.0) == 0.01
    with pytest.raises(ValueError, match="signed Stage 0 centre"):
        dictionary.resolve_center_times_multiplier("RAdam", 1.0)
    assert dictionary.resolve_center_times_multiplier("RAdam", 2.0, signed_centers={"RAdam": 0.0025}) == 0.005


def test_breadth_resolution_is_256_records_and_c_only_changes_virtual_order() -> None:
    records = dictionary.iter_breadth_signs()
    assert len(records) == 256
    assert records[0] == {**{column: -1 for column in "ABCDEFGH"}, "I": 1, "J": 1, "K": 1, "L": 1}
    assert records[-1] == {column: 1 for column in "ABCDEFGHIJKL"}
    low = dictionary.breadth_levels(records[0])
    high_signs = dictionary.derive_breadth_columns({**{column: -1 for column in "ABCDEFGH"}, "C": 1})
    high = dictionary.breadth_levels(high_signs)
    assert low["C"] == 1
    assert high["C"] == 2
    assert low["A"] == high["A"] and low["B"] == high["B"]


def test_out_of_fraction_control_resolves_without_becoming_a_fraction_row() -> None:
    control = dict(zip("ABCDEFGHIJKL", (1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1)))
    assert dictionary.is_fraction_row(control) is False
    levels = dictionary.breadth_levels(control)
    assert levels["J"] == "none"
    assert levels["K"] == 0.5
    assignment = {
        "mixing": "tensor",
        "activation": "pointwise",
        "optimizer": "Adam",
        "lr_multiplier": 1.0,
        **control,
    }
    resolved = dictionary.resolve_assignment(
        assignment,
        absolute_lr=0.005,
        runtime_seed=620001,
        science_assignment_id="named-control",
    )
    assert resolved.config["model"]["gradient_clip"] == "none"
    assert resolved.config["model"]["proposal_scale"] == 0.5


def test_fraction_rows_still_satisfy_all_four_generator_relations() -> None:
    rows = dictionary.iter_breadth_signs()
    assert len(rows) == 256
    assert all(dictionary.is_fraction_row(row) for row in rows)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: {key: value for key, value in row.items() if key != "L"},
        lambda row: {**row, "M": 1},
        lambda row: {**row, "A": 0},
    ],
)
def test_malformed_complete_assignments_still_fail(mutation) -> None:
    row = dictionary.iter_breadth_signs()[0]
    with pytest.raises(ValueError):
        dictionary.breadth_levels(mutation(row))


def test_c_amendment_reaches_only_virtual_order_in_resolved_configs() -> None:
    low_signs = dictionary.derive_breadth_columns({column: -1 for column in "ABCDEFGH"})
    high_signs = dictionary.derive_breadth_columns({**{column: -1 for column in "ABCDEFGH"}, "C": 1})
    low = dictionary.resolve_assignment(low_signs | {"mixing": "tensor", "activation": "pointwise", "optimizer": "Adam", "lr_multiplier": 1.0}, absolute_lr=0.005, runtime_seed=1)
    high = dictionary.resolve_assignment(high_signs | {"mixing": "tensor", "activation": "pointwise", "optimizer": "Adam", "lr_multiplier": 1.0}, absolute_lr=0.005, runtime_seed=1)
    assert low.config["model"]["body_max_order"] == high.config["model"]["body_max_order"] == 2
    assert low.config["model"]["max_virtual_order"] == 1
    assert high.config["model"]["max_virtual_order"] == 2
    assert low.config["control_reference"] == "CONTROL_COORDINATES=PENDING-USER-DETAILED-REVIEW"


@pytest.mark.parametrize("package, expected", [
    ("tensor", ("tensor_product",)),
    ("k-GNN-like-linear", ("linear",)),
    ("hybrid-additive", ("linear", "tensor_product")),
])
def test_choice_surface_maps_landed_producers_and_supports_c1_c2(package, expected) -> None:
    pytest.importorskip("torch")
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from tpen.nn import CompositeMixing, EquivariantMixing, LinearEquivariantMixing, PathAggregation

    for max_virtual_order in (1, 2):
        surface = dictionary.build_shared_hooke_choice_surface(
            channels=16,
            max_virtual_order=max_virtual_order,
            mixing_package=package,
            activation_package="pointwise",
        )
        selected = surface["choices"]["interaction"][package]
        layout = instantiate(OmegaConf.create(selected["layout"]))
        mixing = instantiate(OmegaConf.create(selected["mixing"]))
        aggregation = instantiate(OmegaConf.create(selected["path_aggregation"]))
        assert isinstance(mixing, CompositeMixing)
        assert isinstance(aggregation, PathAggregation)
        assert tuple(slice_.family for slice_ in layout.family_slices) == expected
        assert layout.output_orders.values == (1, 2)
        assert layout.fingerprint == mixing.layout.fingerprint == aggregation.layout.fingerprint
        producers = tuple(mixing.producers)
        assert tuple(type(producer) for producer in producers) == {
            "tensor": (EquivariantMixing,),
            "k-GNN-like-linear": (LinearEquivariantMixing,),
            "hybrid-additive": (LinearEquivariantMixing, EquivariantMixing),
        }[package]
        for producer in producers:
            if isinstance(producer, EquivariantMixing):
                assert producer.max_order == 2
                assert producer.max_virtual_order == max_virtual_order


@pytest.mark.parametrize("package", dictionary.ACTIVATION_PACKAGES)
def test_cpmlp_slots_have_correct_axes_and_one_hidden_layer(package) -> None:
    pytest.importorskip("torch")
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from tpen.nn import ChannelPreservingMLPActivation

    surface = dictionary.build_shared_hooke_choice_surface(
        channels=32,
        mixing_package="tensor",
        activation_package=package,
    )
    slots = surface["choices"]["activation"]
    if package in ("MLP-mixing", "MLP-both"):
        mixing = instantiate(OmegaConf.create(slots["mixing"]))
        assert isinstance(mixing, ChannelPreservingMLPActivation)
        assert mixing.layout.axes.tuple_axes_start == 3
        assert all(spec.hidden_channels == 64 and spec.num_hidden_layers == 1 for spec in mixing.layout.specs)
    if package in ("MLP-aggregation", "MLP-both"):
        aggregation = instantiate(OmegaConf.create(slots["aggregation"]))
        assert isinstance(aggregation, ChannelPreservingMLPActivation)
        assert aggregation.layout.axes.tuple_axes_start == 2
        assert all(spec.hidden_channels == 64 and spec.num_hidden_layers == 1 for spec in aggregation.layout.specs)


def test_runtime_seed_is_one_source_with_non_aliasing_layer_slot_streams_and_standalone_defaults() -> None:
    pytest.importorskip("torch")
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    defaults = dictionary.cpmlp_initializer_streams(None)
    assert defaults["mixing"]["seed"] == 701
    assert defaults["aggregation"]["seed"] == 702
    assert defaults["mixing"]["stream"] != defaults["aggregation"]["stream"]
    assert "layer_3" in dictionary.cpmlp_initializer_streams(10, layer_index=3)["mixing"]["stream"]

    def state(seed: int):
        surface = dictionary.build_shared_hooke_choice_surface(
            channels=16,
            mixing_package="tensor",
            activation_package="MLP-both",
            runtime_seed=seed,
            layer_index=2,
        )
        activation = instantiate(OmegaConf.create(surface["choices"]["activation"]["mixing"]))
        return tuple(parameter.detach().clone() for parameter in activation.parameters())

    first = state(12345)
    same = state(12345)
    other = state(12346)
    assert all(left.equal(right) for left, right in zip(first, same))
    assert any(not left.equal(right) for left, right in zip(first, other))


def test_existing_hooke_fragment_exposes_the_shared_override_surface() -> None:
    from omegaconf import OmegaConf

    presets = OmegaConf.load(ROOT / "experiments" / "hooke" / "choices" / "tpen_presets.yaml")
    default = OmegaConf.to_container(presets.choices.shared, resolve=True)
    assert default["channels"] == 2
    assert default["max_order"] == default["max_virtual_order"] == 2
    assert default["cpmlp"]["mixing_seed"] == 701
    assert default["cpmlp"]["aggregation_seed"] == 702
    override = OmegaConf.merge(
        presets,
        OmegaConf.create(
            {
                "runtime": {"seed": 9876},
                "choices": {
                    "shared": {
                        "channels": 32,
                        "max_virtual_order": 1,
                        "path_family": "full",
                        "aggregation": "sum",
                        "initial_weight": 0.25,
                    }
                },
            }
        ),
    )
    resolved = OmegaConf.to_container(override.choices, resolve=True)
    assert resolved["shared"]["cpmlp"]["mixing_seed"] == 9876
    assert resolved["shared"]["cpmlp"]["aggregation_seed"] == 9876
    assert resolved["interaction"]["tensor_product"]["tensor_product_metadata"]["max_virtual_order"] == 1
    assert resolved["interaction"]["tensor_product"]["tensor_product_metadata"]["output_embedding"] == "full"


def test_resolved_config_keeps_science_identity_separate_from_collision_signature() -> None:
    first = dictionary.resolve_assignment(
        _assignment(), absolute_lr=0.005, runtime_seed=620001, science_assignment_id="breadth-cell-0001"
    )
    second = dictionary.resolve_assignment(
        _assignment(), absolute_lr=0.005, runtime_seed=620001, science_assignment_id="bridge-cell-0007"
    )
    assert first.science_assignment_id != second.science_assignment_id
    assert first.structural_signature == second.structural_signature
    assert first.config["model"]["body_max_order"] == 2
    assert first.config["evaluation"]["reference_energy"] is None
    assert first.config["parameter_count"]["kind"] == "endpoint"
