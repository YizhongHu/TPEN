"""Tests for the literal HI context-to-Hydra mapping."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from hydra.utils import instantiate
from omegaconf import OmegaConf
import pytest


_SPEC = importlib.util.spec_from_file_location(
    "he_importance_context_mapping_test", Path(__file__).with_name("context_mapping.py")
)
assert _SPEC is not None and _SPEC.loader is not None
context_mapping = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = context_mapping
_SPEC.loader.exec_module(context_mapping)


PATH_LABELS = (
    "tensor",
    "linear/coordinate-neighbor",
    "linear/orbit-complete",
    "hybrid/coordinate-neighbor",
    "hybrid/orbit-complete",
)
ACTIVATION_LABELS = (
    "SiLU,SiLU",
    "tanh,tanh",
    "exact GELU,exact GELU",
    "Softplus beta2,Softplus beta2",
    "channel-MLP,SiLU",
    "SiLU,channel-MLP",
    "channel-MLP,channel-MLP",
)


def test_unknown_labels_refuse_without_a_default() -> None:
    with pytest.raises(context_mapping.UnknownContextLabel):
        context_mapping.context_fragment("linear/unknown")
    with pytest.raises(context_mapping.UnknownContextLabel):
        context_mapping.context_fragment("tensor", activation="unknown")


def test_linear_oracles_are_literal_and_support_bounded() -> None:
    coordinate = context_mapping.context_fragment("linear/coordinate-neighbor")
    orbit = context_mapping.context_fragment("linear/orbit-complete")
    coordinate_paths = coordinate["layout"]["linear"]["explicit"]["_args_"][0]
    orbit_paths = orbit["layout"]["linear"]["explicit"]["_args_"][0]

    assert [(p["output_order"], p["input_order"], p["tau_in"]) for p in coordinate_paths] == [
        (1, 1, [0]),
        (1, 1, [1]),
        (2, 2, [0, 1]),
    ]
    assert len(orbit_paths) == 8
    assert all(
        p["output_order"] + p["input_order"] - len(set(p["tau_out"]) & set(p["tau_in"])) <= 2
        for p in orbit_paths
    )


def test_layout_and_explicit_execution_controls_are_written() -> None:
    for label in PATH_LABELS:
        fragment = context_mapping.context_fragment(label)
        assert fragment["layout"]["input_orders"]["values"] == [1, 2]
        assert fragment["mixing"]["producers"][-1].get("activation") is None
        assert fragment["path_aggregation"]["initializer"]["stream"] == "layer0/path_aggregation"
        for producer in fragment["mixing"]["producers"]:
            assert producer["implementation"] == "vectorized"
            if producer["_target_"].endswith("LinearEquivariantMixing"):
                assert producer["aggregation"] == "sum"


@pytest.mark.parametrize("path_label", PATH_LABELS)
@pytest.mark.parametrize("activation_label", ACTIVATION_LABELS)
def test_all_35_fragments_have_declared_targets(path_label: str, activation_label: str) -> None:
    fragment = context_mapping.context_fragment(path_label, activation=activation_label)
    assert fragment["mixing"]["_target_"] == "tpen.nn.CompositeMixing"
    assert fragment["path_aggregation"]["_target_"] == "tpen.nn.PathAggregation"
    assert fragment["update"]["_target_"] == "tpen.nn.ResidualUpdater"


def test_activation_parameters_and_named_streams() -> None:
    fragment = context_mapping.context_fragment("hybrid/orbit-complete", activation="channel-MLP,channel-MLP")
    producer = fragment["mixing"]["activation"]
    aggregation = fragment["path_aggregation"]["activation"]
    assert producer["layout"]["axes"]["tuple_axes_start"] == 3
    assert aggregation["layout"]["axes"]["tuple_axes_start"] == 2
    for activation in (producer, aggregation):
        specs = activation["layout"]["specs"]
        assert [(spec["channels"], spec["hidden_channels"], spec["num_hidden_layers"]) for spec in specs] == [
            (32, 128, 1),
            (32, 128, 1),
        ]
        assert all(spec["activation"]["_target_"] == "torch.nn.SiLU" for spec in specs)
        assert all(spec["bias"] is True for spec in specs)
    assert producer["initializer"]["stream"] != aggregation["initializer"]["stream"]


def test_exact_gelu_and_softplus_beta_two_are_explicit() -> None:
    gelu = context_mapping.context_fragment("tensor", activation="exact GELU,exact GELU")
    assert gelu["mixing"]["activation"] == {"_target_": "torch.nn.GELU", "approximate": "none"}
    softplus = context_mapping.context_fragment("tensor", activation="Softplus beta2,Softplus beta2")
    assert softplus["mixing"]["activation"]["beta"] == 2.0
    assert softplus["mixing"]["activation"]["threshold"] == float("inf")


@pytest.mark.parametrize("path_label", PATH_LABELS)
@pytest.mark.parametrize("activation_label", ACTIVATION_LABELS)
def test_every_fragment_instantiates(path_label: str, activation_label: str) -> None:
    config = OmegaConf.create(
        context_mapping.context_fragment(path_label, activation=activation_label)
    )
    layout = instantiate(config.layout)
    mixing = instantiate(config.mixing)
    aggregation = instantiate(config.path_aggregation)
    layer = instantiate(config)
    assert mixing.layout.fingerprint == layout.fingerprint
    assert aggregation.layout.fingerprint == layout.fingerprint
    assert layer.layout.fingerprint == layout.fingerprint
