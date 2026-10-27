"""Tests for the slow unary linear support-path reference."""

from __future__ import annotations

import pytest
import torch
from typeguard import TypeCheckError, suppress_type_checks

from tpen.data.paths import LinearPathMetadata, LinearPathPolicy, SupportPath
from tpen.data.real import Feature, zero_block
from tpen.nn import LinearEquivariantMixing
from tpen.nn.mixing_kernel import Aggregation, MixingImplementation, execute_unary


def _feature(values: torch.Tensor, order: int = 1) -> Feature:
    blocks = [zero_block(batch_size=values.shape[0], dtype=values.dtype)]
    blocks.extend(torch.zeros(values.shape[0], 1, *(values.shape[-1:] * current_order), dtype=values.dtype) for current_order in range(1, order))
    blocks.append(values.unsqueeze(1))
    return Feature(blocks)


def test_coordinate_neighbor_metadata_has_static_paths() -> None:
    metadata = LinearPathMetadata.generate(max_order=2)

    assert metadata.policy is LinearPathPolicy.COORDINATE_NEIGHBOR
    assert [len(metadata.paths_for_output_order(order)) for order in (1, 2)] == [2, 3]
    assert metadata.paths_for_output_order(2)[1].tau_in == (2, 1)
    assert metadata.paths_for_output_order(2)[2].tau_in == (0, 2)


@pytest.mark.parametrize("aggregation", [Aggregation.SUM, Aggregation.COMPLETION_MEAN])
def test_linear_one_body_formula_and_parameter_static(aggregation: Aggregation) -> None:
    values = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    mixing = LinearEquivariantMixing(max_order=1, channels=1, aggregation=aggregation).to(dtype=torch.float64)
    before = tuple(mixing.state_dict())
    output = mixing(_feature(values))
    paths = mixing.metadata.paths_for_output_order(1)
    identity_index = paths.index(SupportPath(1, 1, (0,), (0,)))
    disjoint_index = paths.index(SupportPath(1, 1, (0,), (1,)))

    assert output.blocks[1].shape == (1, 1, 2, 3)
    torch.testing.assert_close(output.blocks[1][0, 0, identity_index], values[0])
    expected = torch.tensor([6.0, 5.0, 3.0], dtype=values.dtype)
    if aggregation is Aggregation.COMPLETION_MEAN:
        expected = torch.tensor([3.0, 2.5, 1.5], dtype=values.dtype)
    torch.testing.assert_close(output.blocks[1][0, 0, disjoint_index], expected)
    assert tuple(mixing.state_dict()) == before


def test_linear_empty_orbit_and_zero_particles_are_zero_safe() -> None:
    metadata = LinearPathMetadata.generate(max_order=1, policy=LinearPathPolicy.ORBIT_COMPLETE)
    mixing = LinearEquivariantMixing(max_order=1, channels=1, metadata=metadata, aggregation="completion_mean")
    paths = metadata.paths_for_output_order(1)
    disjoint_index = paths.index(SupportPath(1, 1, (0,), (1,)))
    identity_index = paths.index(SupportPath(1, 1, (0,), (0,)))
    one_particle = mixing(Feature([zero_block(batch_size=1), torch.ones(1, 1, 1)]))
    assert one_particle.blocks[1].shape == (1, 1, 2, 1)
    assert torch.isfinite(one_particle.blocks[1]).all()
    torch.testing.assert_close(one_particle.blocks[1][0, 0, identity_index], torch.ones(1))
    torch.testing.assert_close(one_particle.blocks[1][0, 0, disjoint_index], torch.zeros(1))

    output = mixing(Feature([zero_block(batch_size=1), torch.empty(1, 1, 0)]))

    assert output.blocks[1].shape == (1, 1, 2, 0)
    assert torch.isfinite(output.blocks[1]).all()
    torch.testing.assert_close(output.blocks[1], torch.zeros_like(output.blocks[1]))


def test_linear_explicit_metadata_preserves_declared_order() -> None:
    paths = (
        SupportPath(1, 1, (0,), (1,), "sum"),
        SupportPath(1, 1, (0,), (0,), "sum"),
    )
    metadata = LinearPathMetadata.generate(max_order=1, policy="explicit", explicit=paths)
    assert metadata.all_paths() == paths


def test_linear_explicit_metadata_accepts_sequences_and_stores_tuple() -> None:
    paths = [
        SupportPath(1, 1, (0,), (1,), "sum"),
        SupportPath(1, 1, (0,), (0,), "sum"),
    ]
    from_list = LinearPathMetadata.generate(max_order=1, policy="explicit", explicit=paths)
    from_tuple = LinearPathMetadata.generate(max_order=1, policy="explicit", explicit=tuple(paths))

    assert from_list == from_tuple
    assert from_list.fingerprint == from_tuple.fingerprint
    assert from_list.all_paths() == tuple(paths)
    assert isinstance(from_list.outputs[0].entries, tuple)

    assert LinearPathMetadata.generate(max_order=1, policy="explicit").all_paths() == ()


@pytest.mark.parametrize(
    "explicit", [("not a SupportPath",), "not a sequence of paths", b"not a sequence of paths"]
)
def test_linear_explicit_metadata_rejects_boundary_invalid_values(explicit: object) -> None:
    with pytest.raises((TypeError, TypeCheckError), match="explicit"):
        LinearPathMetadata.generate(max_order=1, policy="explicit", explicit=explicit)  # type: ignore[arg-type]


def test_linear_explicit_metadata_rejects_nonfirst_invalid_element_in_body() -> None:
    explicit = (SupportPath(1, 1, (0,), (0,)), "not a SupportPath")
    with pytest.raises(TypeError, match=r"explicit\[1\]"):
        LinearPathMetadata.generate(max_order=1, policy="explicit", explicit=explicit)  # type: ignore[arg-type]


@pytest.mark.parametrize("explicit", ["not a sequence of paths", b"not a sequence of paths"])
def test_linear_explicit_metadata_body_rejects_text_with_typeguard_suppressed(explicit: object) -> None:
    with suppress_type_checks():
        with pytest.raises(TypeError, match="sequence"):
            LinearPathMetadata.generate(max_order=1, policy="explicit", explicit=explicit)  # type: ignore[arg-type]


@pytest.mark.parametrize("implementation", [MixingImplementation.SLOW, MixingImplementation.VECTORIZED])
@pytest.mark.parametrize("aggregation", [Aggregation.SUM, Aggregation.COMPLETION_MEAN])
def test_unary_hand_computed_one_body_values(
    implementation: MixingImplementation, aggregation: Aggregation
) -> None:
    """Anchor execution to the defining equation, independently of references."""

    paths = (
        SupportPath(1, 1, (0,), (0,), "sum"),
        SupportPath(1, 1, (0,), (1,), "completion_mean"),
    )
    weights = (
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([[3.0]], dtype=torch.float64),
    )
    source = torch.tensor([[[1.0, 2.0, 4.0]]], dtype=torch.float64)
    actual = execute_unary(
        paths,
        weights,
        source,
        n_particles=3,
        output_order=1,
        batch_size=1,
        output_channels=1,
        aggregation=aggregation,
        implementation=implementation,
    )
    expected = torch.tensor(
        [[[[2.0, 4.0, 8.0], [9.0, 7.5, 4.5]]]], dtype=torch.float64
    )
    if aggregation is Aggregation.SUM:
        expected[:, :, 1] *= 2.0
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("implementation", ["slow", "vectorized"])
def test_linear_implementation_is_static_and_selectable(implementation: str) -> None:
    mixing = LinearEquivariantMixing(max_order=1, channels=1, implementation=implementation)
    assert mixing.implementation is MixingImplementation(implementation)
    before = tuple(mixing.state_dict())
    mixing(_feature(torch.ones(1, 3)))
    assert tuple(mixing.state_dict()) == before
