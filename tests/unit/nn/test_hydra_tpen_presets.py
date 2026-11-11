"""Hydra choice-library contracts for activation slots and producer modes."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tpen.data.real import Feature, zero_block
from tpen.nn import (
    ChannelPreservingMLPActivation,
    CompositeMixing,
    EquivariantMixing,
    InteractionMode,
    LinearEquivariantMixing,
    PathAggregation,
    ResolvedInteractionConfig,
    TPENLayer,
    normalize_interaction_mode,
)


ROOT = Path(__file__).resolve().parents[3]
PRESETS = ROOT / "experiments" / "hooke" / "choices" / "tpen_presets.yaml"
DEFAULT_CONFIG = ROOT / "experiments" / "atomistic" / "h2-v1" / "configs" / "train.yaml"
MODES = tuple(InteractionMode)


def _config() -> DictConfig:
    """Load the same merged fragment shape used by the choice libraries."""

    return OmegaConf.load(PRESETS)


def _mode_config(config: DictConfig, mode: InteractionMode) -> DictConfig:
    return config.choices.interaction[mode.value]


def _feature(batch: int = 2, particles: int = 3) -> Feature:
    generator = torch.Generator().manual_seed(709)
    return Feature(
        [
            zero_block(batch_size=batch, dtype=torch.float64),
            torch.randn(batch, 2, particles, generator=generator, dtype=torch.float64),
            torch.randn(batch, 2, particles, particles, generator=generator, dtype=torch.float64),
        ]
    )


def test_fragment_uses_the_existing_shared_choice_library_location() -> None:
    """The fragment is co-located with the established Hooke choice table."""

    assert PRESETS.parent == ROOT / "experiments" / "hooke" / "choices"
    assert PRESETS.exists()
    assert "choices:" in PRESETS.read_text(encoding="utf-8")


def test_hydra_selects_independent_activation_slots_and_frozen_axes() -> None:
    config = _config()
    mixing = instantiate(config.choices.activation.channel_mlp.mixing)
    aggregation = instantiate(config.choices.activation.channel_mlp.aggregation)

    assert type(mixing) is type(aggregation) is ChannelPreservingMLPActivation
    assert mixing.layout.axes.channel_axis == aggregation.layout.axes.channel_axis == 1
    assert mixing.layout.axes.tuple_axes_start == 3
    assert aggregation.layout.axes.tuple_axes_start == 2
    assert tuple(spec.order for spec in mixing.layout.specs) == (1, 2)
    assert tuple(spec.order for spec in aggregation.layout.specs) == (1, 2)
    assert mixing is not aggregation
    assert {id(parameter) for parameter in mixing.parameters()}.isdisjoint(
        {id(parameter) for parameter in aggregation.parameters()}
    )

    layer = instantiate(_mode_config(config, InteractionMode.HYBRID).layer)
    keys = tuple(layer.state_dict())
    assert any(key.startswith("mixing.activation.mlps.") for key in keys)
    assert any(key.startswith("path_aggregation.activation.mlps.") for key in keys)


@pytest.mark.parametrize("mode", MODES)
def test_hydra_mode_selects_concrete_producers_and_layout(mode: InteractionMode) -> None:
    config = _config()
    selected = _mode_config(config, mode)
    mixing = instantiate(selected.mixing)
    aggregation = instantiate(selected.path_aggregation)
    layout = instantiate(selected.layout)

    assert isinstance(mixing, CompositeMixing)
    assert isinstance(aggregation, PathAggregation)
    assert aggregation.layout.fingerprint == layout.fingerprint
    assert mixing.layout.fingerprint == layout.fingerprint
    assert normalize_interaction_mode(selected.mode) is mode
    expected = {
        InteractionMode.LINEAR: ("linear",),
        InteractionMode.HYBRID: ("linear", "tensor_product"),
        InteractionMode.TENSOR_PRODUCT: ("tensor_product",),
    }[mode]
    assert tuple(slice_.family for slice_ in layout.family_slices) == expected
    assert tuple(type(producer) for producer in mixing.producers) == {
        InteractionMode.LINEAR: (LinearEquivariantMixing,),
        InteractionMode.HYBRID: (LinearEquivariantMixing, EquivariantMixing),
        InteractionMode.TENSOR_PRODUCT: (EquivariantMixing,),
    }[mode]
    resolved = ResolvedInteractionConfig.from_layout(layout)
    assert tuple(family.value for family in resolved.producer_order) == expected
    assert resolved.fingerprint == layout.fingerprint


@pytest.mark.parametrize("mode", MODES)
def test_hydra_mode_layer_forward_backward_and_static_state(mode: InteractionMode) -> None:
    selected = _mode_config(_config(), mode)
    layer = instantiate(selected.layer).to(dtype=torch.float64)
    assert isinstance(layer, TPENLayer)
    assert layer.mixing.activation is not layer.path_aggregation.activation
    assert {id(parameter) for parameter in layer.mixing.activation.parameters()}.isdisjoint(
        {id(parameter) for parameter in layer.path_aggregation.activation.parameters()}
    )
    before = tuple(layer.state_dict())
    output = layer(_feature())
    loss = sum(block.square().sum() for block in output.blocks[1:])
    loss.backward()
    assert all(parameter.grad is not None for parameter in layer.parameters())
    assert tuple(layer.state_dict()) == before
    assert sum(parameter.numel() for parameter in layer.mixing.activation.parameters()) == 44
    assert sum(parameter.numel() for parameter in layer.path_aggregation.activation.parameters()) == 44


@pytest.mark.parametrize("mode", MODES)
def test_hydra_mode_activation_state_roundtrips_strictly(mode: InteractionMode) -> None:
    selected = _mode_config(_config(), mode)
    source = instantiate(selected.layer).to(dtype=torch.float64)
    target = instantiate(selected.layer).to(dtype=torch.float64)
    state = source.state_dict()
    result = target.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert tuple(target.state_dict()) == tuple(state)


def test_default_config_remains_legacy_and_legacy_checkpoint_roundtrips() -> None:
    """The opt-in fragment must not change the checked-in default config."""

    config = OmegaConf.load(DEFAULT_CONFIG)
    mixing_cfg = config.model.layers[0].mixing
    aggregation_cfg = config.model.layers[0].path_aggregation
    assert mixing_cfg.activation._target_ == "torch.nn.SiLU"
    assert aggregation_cfg.activation._target_ == "torch.nn.SiLU"
    assert "ChannelPreservingMLPActivation" not in OmegaConf.to_yaml(config.model)

    # This is the real control for the L7 restore gate: the legacy TP-only
    # model has no layout fingerprint and no activation-owned parameter keys.
    source = instantiate(config.model)
    legacy = OrderedDict(source.state_dict())
    assert "_tpen_layout_fingerprint" not in legacy
    assert not any("activation.mlps." in key for key in legacy)
    restored = instantiate(config.model)
    result = restored.load_state_dict(legacy, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_activation_does_not_enter_layout_fingerprint() -> None:
    config = _config()
    selected = _mode_config(config, InteractionMode.HYBRID)
    layout = instantiate(selected.layout)
    with_activation = instantiate(selected.layer)
    without_activation = instantiate(
        OmegaConf.merge(
            selected.layer,
            {
                "mixing": {"activation": None},
                "path_aggregation": {"activation": None},
            },
        )
    )
    assert with_activation.layout.fingerprint == layout.fingerprint
    assert without_activation.layout.fingerprint == layout.fingerprint
    assert with_activation.layout.fingerprint == without_activation.layout.fingerprint
