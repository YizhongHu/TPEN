"""Torch-free tests for the Hydra interaction preset boundary."""

from __future__ import annotations

import pytest

from tpen.nn.interaction_config import (
    InteractionMode,
    ProducerFamily,
    normalize_interaction_mode,
    normalize_producer_order,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("linear", (ProducerFamily.LINEAR,)),
        ("hybrid", (ProducerFamily.LINEAR, ProducerFamily.TENSOR_PRODUCT)),
        ("tensor_product", (ProducerFamily.TENSOR_PRODUCT,)),
    ],
)
def test_hydra_modes_have_typed_static_producer_orders(mode: str, expected: tuple[ProducerFamily, ...]) -> None:
    """The three external mode values normalize to immutable producer tuples."""

    normalized = normalize_interaction_mode(mode)
    assert normalized.producer_order == expected
    assert isinstance(normalized, InteractionMode)


def test_producer_order_rejects_reordered_hybrid() -> None:
    """The union axis contract is linear first, tensor-product second."""

    with pytest.raises(ValueError, match="linear then tensor_product"):
        normalize_producer_order((ProducerFamily.TENSOR_PRODUCT, ProducerFamily.LINEAR))


def test_mode_and_producer_policy_reject_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported interaction mode"):
        normalize_interaction_mode("not-a-mode")
    with pytest.raises(ValueError):
        normalize_producer_order(("not-a-family",))
