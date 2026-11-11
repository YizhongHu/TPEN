"""Tests for immutable wavefunction packet and parameter-layout contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from tpen.data.batch import (
    CoordinateForwardPacket,
    CoordinateLogGradient,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterScore,
    ParameterScoreForwardPacket,
    ParameterSlot,
    WavefunctionOutput,
    WavefunctionPacket,
)
from tpen.data.indices import permute_particle_axis
from tpen.data.permutation import Permutation


def _output(sample_shape: tuple[int, ...] = (2, 3), *, requires_grad: bool = False) -> WavefunctionOutput:
    logabs = torch.arange(1, 1 + torch.tensor(sample_shape).prod().item(), dtype=torch.float64).reshape(sample_shape)
    logabs.requires_grad_(requires_grad)
    return WavefunctionOutput(logabs=logabs, sign=torch.ones(sample_shape, dtype=torch.float64))


def _layout() -> ParameterLayout:
    return ParameterLayout(
        slots=(
            ParameterSlot(ordinal=0, shape=(2,), numel=2, dtype=torch.float64),
            ParameterSlot(ordinal=1, shape=(), numel=1, dtype=torch.float64),
        )
    )


def _scores(sample_shape: tuple[int, ...] = (2, 3), *, requires_grad: bool = False) -> MaterializedParameterLogScores:
    first = torch.arange(12, dtype=torch.float64).reshape(*sample_shape, 2)
    second = torch.arange(6, dtype=torch.float64).reshape(sample_shape)
    first.requires_grad_(requires_grad)
    second.requires_grad_(requires_grad)
    return MaterializedParameterLogScores(layout=_layout(), blocks=(first, second))


def test_coordinate_log_gradient_has_exact_shape_and_semantic_operations() -> None:
    values = torch.arange(2 * 3 * 4 * 2, dtype=torch.float64).reshape(2, 3, 4, 2)
    values.requires_grad_()
    gradient = CoordinateLogGradient(values=values)

    assert gradient.validate(sample_shape=(2, 3), n_electrons=4, spatial_dim=2) is gradient
    assert gradient.sample_shape == (2, 3)
    assert gradient.n_electrons == 4
    assert gradient.spatial_dim == 2

    detached = gradient.detach()
    assert not detached.values.requires_grad
    torch.testing.assert_close(detached.values, gradient.values)

    moved = gradient.to(dtype=torch.float32)
    assert moved.dtype == torch.float32
    permutation = Permutation((2, 0, 3, 1))
    permuted = gradient.permute(permutation)
    torch.testing.assert_close(
        permuted.values,
        permute_particle_axis(values, permutation, axis=-2),
    )
    assert gradient.compare(CoordinateLogGradient(values=values.clone()))[0]


def test_coordinate_log_gradient_rejects_non_real_or_incomplete_shapes() -> None:
    with pytest.raises(ValueError, match="shape"):
        CoordinateLogGradient(values=torch.zeros(3))
    with pytest.raises(TypeError, match="real floating"):
        CoordinateLogGradient(values=torch.zeros(2, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="sample shape"):
        CoordinateLogGradient(values=torch.zeros(2, 3, 4)).validate(sample_shape=(3,))


def test_parameter_layout_is_static_dense_and_exact() -> None:
    layout = _layout()

    assert layout.validate() is layout
    assert layout.total_numel == 3
    assert isinstance(layout.slots, tuple)
    assert layout.detach() is layout
    assert layout.permute(Permutation((1, 0))) is layout
    cast = layout.to(dtype=torch.float32)
    assert all(slot.dtype == torch.float32 for slot in cast.slots)
    assert not layout.compare(cast)[0]

    with pytest.raises(ValueError, match="prod"):
        ParameterSlot(ordinal=0, shape=(2, 3), numel=5, dtype=torch.float64)
    with pytest.raises(ValueError, match="dense"):
        ParameterLayout(
            slots=(ParameterSlot(ordinal=1, shape=(), numel=1, dtype=torch.float64),)
        )
    with pytest.raises(FrozenInstanceError):
        layout.slots = ()


def test_parameter_binding_keeps_direct_live_references_and_exact_order() -> None:
    layout = ParameterLayout(
        slots=(
            ParameterSlot(ordinal=0, shape=(2,), numel=2, dtype=torch.float64),
            ParameterSlot(ordinal=1, shape=(2,), numel=2, dtype=torch.float64),
        )
    )
    first = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))
    second = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    binding = ParameterBinding(layout=layout, parameters=(first, second))

    assert binding.validate() is binding
    assert binding.parameters[0] is first
    assert binding.parameters[1] is second
    assert binding.to(device=first.device, dtype=first.dtype) is binding
    assert binding.compare(ParameterBinding(layout=layout, parameters=(first, second)))[0]
    assert not binding.compare(ParameterBinding(layout=layout, parameters=(second, first)))[0]
    with pytest.raises(RuntimeError, match="cannot detach"):
        binding.detach()


def test_parameter_binding_rejects_copies_and_layout_disagreement() -> None:
    slot = ParameterSlot(ordinal=0, shape=(2,), numel=2, dtype=torch.float64)
    layout = ParameterLayout(slots=(slot,))

    with pytest.raises(TypeError, match="direct torch.nn.Parameter"):
        ParameterBinding(layout=layout, parameters=(torch.ones(2, dtype=torch.float64),))
    with pytest.raises(ValueError, match="shape"):
        ParameterBinding(
            layout=layout,
            parameters=(torch.nn.Parameter(torch.ones(3, dtype=torch.float64)),),
        )
    with pytest.raises(ValueError, match="require gradients"):
        ParameterBinding(
            layout=layout,
            parameters=(torch.nn.Parameter(torch.ones(2, dtype=torch.float64), requires_grad=False),),
        )


def test_materialized_parameter_scores_use_sample_plus_parameter_shapes() -> None:
    scores = _scores(requires_grad=True)

    assert isinstance(scores, ParameterScore)
    assert scores.validate(sample_shape=(2, 3)) is scores
    assert scores.sample_shape == (2, 3)
    assert scores.blocks[0].shape == (2, 3, 2)
    assert scores.blocks[1].shape == (2, 3)
    assert not any(block.requires_grad for block in scores.detach().blocks)

    cast = scores.to(dtype=torch.float32)
    assert all(block.dtype == torch.float32 for block in cast.blocks)
    assert all(slot.dtype == torch.float32 for slot in cast.layout.slots)
    permuted = scores.permute(Permutation((1, 0)))
    assert scores.compare(permuted)[0]
    assert all(left is not right for left, right in zip(scores.blocks, permuted.blocks))


def test_materialized_parameter_scores_reject_missing_or_misshaped_blocks() -> None:
    layout = _layout()

    with pytest.raises(ValueError, match="one block"):
        MaterializedParameterLogScores(
            layout=layout,
            blocks=(torch.zeros(2, 3, 2, dtype=torch.float64),),
        )
    with pytest.raises(ValueError, match="trailing shape"):
        MaterializedParameterLogScores(
            layout=layout,
            blocks=(
                torch.zeros(2, 3, 3, dtype=torch.float64),
                torch.zeros(2, 3, dtype=torch.float64),
            ),
        )
    with pytest.raises(ValueError, match="sample shape"):
        MaterializedParameterLogScores(
            layout=layout,
            blocks=(
                torch.zeros(2, 3, 2, dtype=torch.float64),
                torch.zeros(6, dtype=torch.float64),
            ),
        )


def test_coordinate_packet_cannot_omit_or_mismatch_derivative_payload() -> None:
    output = _output()
    coordinates = CoordinateLogGradient(values=torch.zeros(2, 3, 4, 2, dtype=torch.float64))
    packet = CoordinateForwardPacket(output=output, coordinates=coordinates)

    assert isinstance(packet, WavefunctionPacket)
    assert packet.as_output() is output
    assert packet.validate() is packet
    with pytest.raises(TypeError):
        CoordinateForwardPacket(output=output)
    with pytest.raises(ValueError, match="sample shape"):
        CoordinateForwardPacket(
            output=output,
            coordinates=CoordinateLogGradient(values=torch.zeros(6, 4, 2, dtype=torch.float64)),
        )
    with pytest.raises(FrozenInstanceError):
        packet.coordinates = coordinates.detach()


def test_coordinate_packet_operations_include_value_and_derivative_fields() -> None:
    output = _output(requires_grad=True)
    values = torch.arange(2 * 3 * 4 * 2, dtype=torch.float64).reshape(2, 3, 4, 2)
    values.requires_grad_()
    packet = CoordinateForwardPacket(
        output=output,
        coordinates=CoordinateLogGradient(values=values),
    )

    detached = packet.detach()
    assert not detached.output.logabs.requires_grad
    assert not detached.coordinates.values.requires_grad
    cast = packet.to(dtype=torch.float32)
    assert cast.output.logabs.dtype == torch.float32
    assert cast.coordinates.values.dtype == torch.float32

    permutation = Permutation((1, 3, 0, 2))
    permuted = packet.permute(permutation)
    torch.testing.assert_close(permuted.output.sign, -packet.output.sign)
    torch.testing.assert_close(
        permuted.coordinates.values,
        permute_particle_axis(packet.coordinates.values, permutation, axis=-2),
    )
    assert packet.compare(
        CoordinateForwardPacket(
            output=_output(requires_grad=True),
            coordinates=CoordinateLogGradient(values=values.clone()),
        )
    )[0]


def test_parameter_score_packet_operations_include_every_score_block() -> None:
    output = _output(requires_grad=True)
    packet = ParameterScoreForwardPacket(output=output, parameter_scores=_scores(requires_grad=True))

    assert packet.as_output() is output
    assert packet.validate() is packet
    detached = packet.detach()
    assert not detached.output.logabs.requires_grad
    assert not any(block.requires_grad for block in detached.parameter_scores.blocks)
    cast = packet.to(dtype=torch.float32)
    assert cast.output.logabs.dtype == torch.float32
    assert all(block.dtype == torch.float32 for block in cast.parameter_scores.blocks)

    permuted = packet.permute(Permutation((1, 0)))
    torch.testing.assert_close(permuted.output.sign, -packet.output.sign)
    assert packet.parameter_scores.compare(permuted.parameter_scores)[0]
    assert packet.compare(ParameterScoreForwardPacket(output=_output(), parameter_scores=_scores()))[0]


def test_parameter_score_packet_rejects_sample_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="sample shape"):
        ParameterScoreForwardPacket(output=_output(), parameter_scores=_scores(sample_shape=(6,)))
