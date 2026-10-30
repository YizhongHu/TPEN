"""Tests for explicit TPEN neural initializers."""

from __future__ import annotations

import torch

from tpen.nn import Embedding, MLP, PathAggregation, SeededLinear, TorchInitializer


def _state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def _assert_state_dicts_equal(left: torch.nn.Module, right: torch.nn.Module) -> None:
    left_state = _state_dict(left)
    right_state = _state_dict(right)
    assert left_state.keys() == right_state.keys()
    for key, left_value in left_state.items():
        torch.testing.assert_close(left_value, right_state[key])


def test_seeded_linear_is_deterministic_without_global_rng_mutation() -> None:
    initializer = TorchInitializer(seed=17)
    torch.manual_seed(999)
    before = torch.get_rng_state()

    first = SeededLinear(3, 4, initializer=initializer)

    torch.testing.assert_close(torch.get_rng_state(), before)
    torch.manual_seed(123)
    second = SeededLinear(3, 4, initializer=TorchInitializer(seed=17))

    _assert_state_dicts_equal(first, second)


def test_initializer_seed_changes_initialized_parameters() -> None:
    first = SeededLinear(3, 4, initializer=TorchInitializer(seed=17))
    second = SeededLinear(3, 4, initializer=TorchInitializer(seed=18))

    with torch.no_grad():
        assert not torch.equal(first.weight, second.weight)


def test_fresh_named_stream_generators_repeat_the_same_normal_draws() -> None:
    first = torch.empty(7, dtype=torch.float64)
    second = torch.empty(7, dtype=torch.float64)

    TorchInitializer(seed=37, stream="readout/a12").normal_(first)
    TorchInitializer(seed=37, stream="readout/a12").normal_(second)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_mlp_uses_explicit_initializer_without_global_rng_mutation() -> None:
    torch.manual_seed(999)
    before = torch.get_rng_state()

    first = MLP(3, 2, hidden_channels=5, num_hidden_layers=2, initializer=TorchInitializer(seed=23))

    torch.testing.assert_close(torch.get_rng_state(), before)
    torch.manual_seed(123)
    second = MLP(3, 2, hidden_channels=5, num_hidden_layers=2, initializer=TorchInitializer(seed=23))

    _assert_state_dicts_equal(first, second)


def test_embedding_passes_initializer_to_generated_order_mlps() -> None:
    first = Embedding(
        max_order=2,
        spatial_dim=3,
        out_channels=4,
        hidden_channels=5,
        num_hidden_layers=1,
        include_spins=False,
        initializer=TorchInitializer(seed=29),
    )
    second = Embedding(
        max_order=2,
        spatial_dim=3,
        out_channels=4,
        hidden_channels=5,
        num_hidden_layers=1,
        include_spins=False,
        initializer=TorchInitializer(seed=29),
    )

    _assert_state_dicts_equal(first, second)


def test_path_aggregation_uses_explicit_initializer_without_global_rng_mutation() -> None:
    torch.manual_seed(999)
    before = torch.get_rng_state()

    first = PathAggregation(
        max_order=2,
        channels=2,
        path_counts_by_order={1: 2, 2: 3},
        initializer=TorchInitializer(seed=31),
    )

    torch.testing.assert_close(torch.get_rng_state(), before)
    torch.manual_seed(123)
    second = PathAggregation(
        max_order=2,
        channels=2,
        path_counts_by_order={1: 2, 2: 3},
        initializer=TorchInitializer(seed=31),
    )

    _assert_state_dicts_equal(first, second)


def test_path_aggregation_initializer_seed_changes_path_weights() -> None:
    first = PathAggregation(
        max_order=1,
        channels=2,
        path_counts_by_order={1: 2},
        initializer=TorchInitializer(seed=31),
    )
    second = PathAggregation(
        max_order=1,
        channels=2,
        path_counts_by_order={1: 2},
        initializer=TorchInitializer(seed=32),
    )

    with torch.no_grad():
        assert not torch.equal(first.weights[first.key(1)], second.weights[second.key(1)])
