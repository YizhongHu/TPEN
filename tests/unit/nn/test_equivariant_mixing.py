"""Equivariance tests for slow real-space mixing."""

from __future__ import annotations

import torch
import pytest

from tpen.data.real import Feature, zero_block
from tpen.nn import EquivariantMixing
from tpen.nn.mixing_kernel import Aggregation, MixingImplementation
from tpen.data.paths import VirtualPath, load_default_path_metadata
from tests.helpers.equivariance import assert_equivariant_all
from tpen.nn.mixing_kernel import build_binary_index_plan


def _one_channel_feature(values: torch.Tensor) -> Feature:
    batch, n_particles = values.shape
    return Feature(
        [
            zero_block(batch_size=batch, device=values.device, dtype=values.dtype),
            values.unsqueeze(1),
        ]
    )


def test_slow_mixing_matches_one_body_product_formula() -> None:
    values = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    feature = _one_channel_feature(values)
    mixing = EquivariantMixing(max_order=1, max_virtual_order=1, channels=1, initial_weight=1.0).to(dtype=torch.float64)

    output = mixing(feature)

    assert output.blocks[1].shape == (1, 1, 1, 3)
    torch.testing.assert_close(output.blocks[1][:, 0, 0], values.square())


def test_completion_mean_averages_compatible_virtual_tuples() -> None:
    values = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    feature = _one_channel_feature(values)
    mixing = EquivariantMixing(
        max_order=1,
        max_virtual_order=2,
        aggregation="completion_mean",
        channels=1,
        initial_weight=1.0,
    ).to(dtype=torch.float64)

    output = mixing(feature)

    torch.testing.assert_close(output.blocks[1][:, 0, 0], values.square())
    expected_order_two_completion = torch.tensor([[3.0, 5.0, 6.0]], dtype=torch.float64)
    torch.testing.assert_close(output.blocks[1][:, 0, 1], expected_order_two_completion)
    torch.testing.assert_close(output.blocks[1][:, 0, 2], expected_order_two_completion)


@pytest.mark.parametrize("implementation", ["slow", "vectorized"])
def test_mixing_handles_zero_particles_without_nan(implementation: str) -> None:
    feature = Feature(
        [
            zero_block(batch_size=1, dtype=torch.float64),
            torch.empty(1, 1, 0, dtype=torch.float64),
            torch.empty(1, 1, 0, 0, dtype=torch.float64),
        ]
    )
    mixing = EquivariantMixing(
        max_order=2,
        max_virtual_order=2,
        channels=1,
        aggregation="completion_mean",
        implementation=implementation,
    ).to(dtype=torch.float64)

    output = mixing(feature)

    assert output.blocks[1].shape[-1:] == (0,)
    assert output.blocks[2].shape[-2:] == (0, 0)
    assert torch.isfinite(output.blocks[1]).all()
    assert torch.isfinite(output.blocks[2]).all()


@pytest.mark.parametrize("implementation", ["slow", "vectorized"])
def test_mixing_zeroes_orders_without_distinct_virtual_tuples(implementation: str) -> None:
    feature = Feature(
        [
            zero_block(batch_size=1, dtype=torch.float64),
            torch.ones(1, 1, 1, dtype=torch.float64),
            torch.ones(1, 1, 1, 1, dtype=torch.float64),
        ]
    )
    mixing = EquivariantMixing(
        max_order=2,
        max_virtual_order=2,
        channels=1,
        aggregation="completion_mean",
        implementation=implementation,
    ).to(dtype=torch.float64)

    output = mixing(feature)

    torch.testing.assert_close(output.blocks[2], torch.zeros_like(output.blocks[2]))


def test_mixing_default_paths_come_from_saved_metadata() -> None:
    mixing = EquivariantMixing(max_order=2, max_virtual_order=2, output_embedding="full", channels=1)
    metadata = load_default_path_metadata("full")
    expected = [
        path
        for path in metadata.all_paths()
        if path.s <= 2 and path.m <= 2 and path.m1 <= 2 and path.m2 <= 2
    ]

    assert [path.as_tuple() for path in mixing.paths] == [path.as_tuple() for path in expected]


def test_mixing_normalizes_closed_choices_without_changing_parameter_keys() -> None:
    mixing = EquivariantMixing(
        max_order=2,
        max_virtual_order=2,
        channels=1,
        aggregation="completion_mean",
        implementation="vectorized",
    )

    assert mixing.aggregation is Aggregation.COMPLETION_MEAN
    assert mixing.implementation is MixingImplementation.VECTORIZED
    assert tuple(mixing.state_dict()) == tuple(f"weights.g{path.global_id}" for path in mixing.paths)


@pytest.mark.parametrize("implementation", ["slow", "vectorized"])
@pytest.mark.parametrize("aggregation", ["sum", "completion_mean"])
def test_mixing_named_parameters_are_static_across_runtime_particle_counts(
    implementation: str, aggregation: str
) -> None:
    mixing = EquivariantMixing(
        max_order=2,
        max_virtual_order=2,
        channels=1,
        aggregation=aggregation,
        implementation=implementation,
    ).to(dtype=torch.float64)
    before = tuple(name for name, _ in mixing.named_parameters())

    for n_particles in (2, 4):
        feature = Feature(
            [
                zero_block(batch_size=1, dtype=torch.float64),
                torch.ones(1, 1, n_particles, dtype=torch.float64),
                torch.ones(1, 1, n_particles, n_particles, dtype=torch.float64),
            ]
        )
        mixing(feature)

    assert tuple(name for name, _ in mixing.named_parameters()) == before


def test_binary_index_plan_pins_repeated_output_targets() -> None:
    path = VirtualPath(
        s=2,
        m=1,
        m1=1,
        m2=1,
        local_id=0,
        global_id=0,
        tau=(0,),
        tau1=(0,),
        tau2=(1,),
    )

    plan = build_binary_index_plan(path, 3)

    assert tuple(plan.supports.shape) == (6, 2)
    assert plan.output_indices.reshape(-1).tolist() == [0, 0, 1, 1, 2, 2]
    assert plan.flat_output_indices.tolist() == [0, 0, 1, 1, 2, 2]


def test_binary_index_plan_pins_empty_support_cache() -> None:
    path = VirtualPath(
        s=2,
        m=1,
        m1=1,
        m2=1,
        local_id=0,
        global_id=0,
        tau=(0,),
        tau1=(0,),
        tau2=(1,),
    )

    plan = build_binary_index_plan(path, 0)

    assert tuple(plan.supports.shape) == (0, 2)
    assert tuple(plan.output_indices.shape) == (0, 1)
    assert tuple(plan.flat_output_indices.shape) == (0,)


def test_vectorized_mixing_matches_slow_reference_gradients_tightly() -> None:
    generator = torch.Generator().manual_seed(707)
    slow_feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, 4, generator=generator, dtype=torch.float64, requires_grad=True),
            torch.randn(1, 2, 4, 4, generator=generator, dtype=torch.float64, requires_grad=True),
        ]
    )
    vectorized_feature = slow_feature.clone()
    vectorized_feature = Feature(
        [vectorized_feature.blocks[0]]
        + [block.detach().clone().requires_grad_() for block in vectorized_feature.blocks[1:]]
    )
    slow = EquivariantMixing(
        max_order=2,
        max_virtual_order=2,
        channels=2,
        aggregation="completion_mean",
        implementation="slow",
        initial_weight=0.5,
    ).to(dtype=torch.float64)
    vectorized = EquivariantMixing(
        max_order=2,
        max_virtual_order=2,
        channels=2,
        aggregation="completion_mean",
        implementation="vectorized",
        initial_weight=0.5,
    ).to(dtype=torch.float64)
    vectorized.load_state_dict(slow.state_dict())

    slow_loss = sum(block.sum() for block in slow(slow_feature).blocks[1:])
    vectorized_loss = sum(block.sum() for block in vectorized(vectorized_feature).blocks[1:])
    slow_loss.backward()
    vectorized_loss.backward()

    torch.testing.assert_close(slow_loss, vectorized_loss, atol=1.0e-12, rtol=1.0e-12)
    for (slow_name, slow_parameter), (vectorized_name, vectorized_parameter) in zip(
        slow.named_parameters(), vectorized.named_parameters()
    ):
        assert slow_name == vectorized_name
        torch.testing.assert_close(
            slow_parameter.grad,
            vectorized_parameter.grad,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    for slow_block, vectorized_block in zip(slow_feature.blocks[1:], vectorized_feature.blocks[1:]):
        torch.testing.assert_close(slow_block.grad, vectorized_block.grad, atol=1.0e-12, rtol=1.0e-12)


def test_slow_mixing_passes_forced_runtime_equivariance_check() -> None:
    generator = torch.Generator().manual_seed(4321)
    feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, 3, generator=generator, dtype=torch.float64),
            torch.randn(1, 2, 3, 3, generator=generator, dtype=torch.float64),
        ]
    )
    mixing = EquivariantMixing(
        max_order=2,
        max_virtual_order=2,
        channels=2,
    ).to(dtype=torch.float64)

    output = mixing(feature)

    assert output.validate() is output
    assert_equivariant_all(mixing, feature)


def test_vectorized_mixing_matches_slow_reference_for_all_aggregations() -> None:
    generator = torch.Generator().manual_seed(2026)
    feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, 4, generator=generator, dtype=torch.float64),
            torch.randn(1, 2, 4, 4, generator=generator, dtype=torch.float64),
        ]
    )
    other = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 3, 4, generator=generator, dtype=torch.float64),
            torch.randn(1, 3, 4, 4, generator=generator, dtype=torch.float64),
        ]
    )
    out_channels = {1: 4, 2: 5}

    for output_embedding in ("canonical", "full"):
        for aggregation in ("sum", "completion_mean"):
            slow = EquivariantMixing(
                max_order=2,
                max_virtual_order=2,
                output_embedding=output_embedding,
                aggregation=aggregation,
                channels={1: 2, 2: 2},
                right_channels={1: 3, 2: 3},
                out_channels=out_channels,
                implementation="slow",
                initial_weight=0.5,
            ).to(dtype=torch.float64)
            vectorized = EquivariantMixing(
                max_order=2,
                max_virtual_order=2,
                output_embedding=output_embedding,
                aggregation=aggregation,
                channels={1: 2, 2: 2},
                right_channels={1: 3, 2: 3},
                out_channels=out_channels,
                implementation="vectorized",
                initial_weight=0.5,
            ).to(dtype=torch.float64)

            slow_output = slow(feature, other)
            vectorized_output = vectorized(feature, other)

            torch.testing.assert_close(vectorized_output.blocks, slow_output.blocks)


def test_vectorized_mixing_passes_forced_runtime_equivariance_check() -> None:
    generator = torch.Generator().manual_seed(31415)
    feature = Feature(
        [
            zero_block(dtype=torch.float64),
            torch.randn(1, 2, 3, generator=generator, dtype=torch.float64),
            torch.randn(1, 2, 3, 3, generator=generator, dtype=torch.float64),
            torch.randn(1, 2, 3, 3, 3, generator=generator, dtype=torch.float64),
        ]
    )
    mixing = EquivariantMixing(
        max_order=3,
        max_virtual_order=3,
        aggregation="completion_mean",
        channels=2,
        implementation="vectorized",
    ).to(dtype=torch.float64)

    output = mixing(feature)

    assert output.validate() is output
    assert_equivariant_all(mixing, feature)
