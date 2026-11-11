"""Model-owned layout identity and pre-mutation restore gates."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from tpen.data.paths import (
    LinearPathMetadata,
    NormalizedChannels,
    NormalizedOrders,
    PathMetadata,
    compose_path_layout,
)
from tpen.nn import TPENWaveFunction


def _layout(*, channels: int = 1):
    metadata = PathMetadata.generate(max_order=1, max_virtual_order=1, output_embedding="canonical")
    return compose_path_layout(
        linear=None,
        tensor_product=metadata,
        input_orders=NormalizedOrders((1,)),
        output_orders=NormalizedOrders((1,)),
        input_channels=NormalizedChannels(((1, channels),)),
        output_channels=NormalizedChannels(((1, channels),)),
    )


def _model(layout):
    return TPENWaveFunction(
        embedding=torch.nn.Linear(1, 1, bias=False),
        layers=(),
        readout=torch.nn.Identity(),
        layout=layout,
    )


def test_wrong_layout_refuses_before_parameter_mutation() -> None:
    source = _model(_layout())
    target = _model(_layout(channels=2))
    before = OrderedDict((key, value.detach().clone()) for key, value in target.state_dict().items())

    with pytest.raises(ValueError, match="layout fingerprint"):
        target.load_state_dict(source.state_dict(), strict=True)

    after = target.state_dict()
    assert tuple(after) == tuple(before)
    for key in before:
        torch.testing.assert_close(after[key], before[key], rtol=0.0, atol=0.0)


def test_missing_layout_identity_is_legacy_compatible_only_for_tp_only() -> None:
    source = _model(_layout())
    legacy = OrderedDict(source.state_dict())
    legacy.pop(source._LAYOUT_STATE_KEY)
    restored = _model(_layout())
    restored.load_state_dict(legacy, strict=True)


def test_missing_layout_identity_is_rejected_for_non_tp_checkpoint() -> None:
    source = _model(_layout())
    legacy = OrderedDict(source.state_dict())
    legacy.pop(source._LAYOUT_STATE_KEY)
    hybrid_layout = compose_path_layout(
        linear=LinearPathMetadata.generate(max_order=1),
        tensor_product=PathMetadata.generate(max_order=1, max_virtual_order=1, output_embedding="canonical"),
        input_orders=NormalizedOrders((1,)),
        output_orders=NormalizedOrders((1,)),
        input_channels=NormalizedChannels(((1, 1),)),
        output_channels=NormalizedChannels(((1, 1),)),
    )
    with pytest.raises(ValueError, match="no layout fingerprint"):
        _model(hybrid_layout).load_state_dict(legacy, strict=True)
