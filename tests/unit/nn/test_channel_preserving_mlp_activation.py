"""Tests for the typed, order-wise channel-preserving MLP activation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import product

import pytest
import torch
from torch import nn

from tpen.nn import (
    ChannelActivationAxes,
    ChannelPreservingMLPActivation,
    MLP,
    OrderMLPLayout,
    OrderMLPSpec,
    TorchInitializer,
)


def _layout(*, tuple_axes_start: int = 2, channel_axis: int = 1) -> OrderMLPLayout:
    return OrderMLPLayout(
        axes=ChannelActivationAxes(
            channel_axis=channel_axis, tuple_axes_start=tuple_axes_start
        ),
        specs=(
            OrderMLPSpec(
                order=1,
                channels=3,
                hidden_channels=5,
                num_hidden_layers=1,
                activation=nn.Tanh(),
            ),
            OrderMLPSpec(
                order=2,
                channels=3,
                hidden_channels=5,
                num_hidden_layers=1,
                activation=nn.Tanh(),
            ),
        ),
    )


def _oracle(activation: ChannelPreservingMLPActivation, inputs: torch.Tensor) -> torch.Tensor:
    """Apply the selected MLP one channel vector at a time at inert positions."""

    axes = activation.layout.axes
    order = inputs.ndim - axes.tuple_axes_start
    index = next(index for index, spec in enumerate(activation.layout.specs) if spec.order == order)
    output = torch.empty_like(inputs)
    inert_axes = tuple(axis for axis in range(inputs.ndim) if axis != axes.channel_axis)
    inert_shape = tuple(int(inputs.shape[axis]) for axis in inert_axes)
    for position in product(*(range(size) for size in inert_shape)):
        tensor_index = [slice(None)] * inputs.ndim
        for axis, value in zip(inert_axes, position, strict=True):
            tensor_index[axis] = value
        tensor_index_tuple = tuple(tensor_index)
        output[tensor_index_tuple] = activation.mlps[index](inputs[tensor_index_tuple])
    return output


@pytest.mark.parametrize(
    ("tuple_axes_start", "channel_axis", "shape"),
    [
        (2, 0, (3, 2, 4)),
        (2, 0, (3, 2, 2, 2)),
        (2, 1, (2, 3, 4)),
        (2, 1, (2, 3, 2, 2)),
        (3, 0, (3, 2, 5, 4)),
        (3, 0, (3, 2, 5, 2, 2)),
        (3, 2, (2, 5, 3, 4)),
        (3, 2, (2, 5, 3, 2, 2)),
    ],
)
def test_movedim_fast_path_matches_inert_position_oracle(
    tuple_axes_start: int, channel_axis: int, shape: tuple[int, ...]
) -> None:
    activation = ChannelPreservingMLPActivation(
        _layout(tuple_axes_start=tuple_axes_start, channel_axis=channel_axis),
        initializer=TorchInitializer(seed=123),
    ).to(dtype=torch.float64)
    inputs = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float64).reshape(shape) / 17

    actual = activation(inputs)
    expected = _oracle(activation, inputs)

    assert actual.shape == inputs.shape
    # Scalar and batched linear kernels can round in different orders on
    # float64; 1e-12 is tight relative to the values here and still catches
    # an incorrect movedim axis by many orders of magnitude.
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_layout_and_specs_are_frozen_and_mlps_are_eagerly_registered() -> None:
    layout = _layout()
    with pytest.raises(FrozenInstanceError):
        layout.axes.channel_axis = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        layout.specs = ()  # type: ignore[misc]

    activation = ChannelPreservingMLPActivation(layout, initializer=TorchInitializer(seed=4))
    assert isinstance(activation.mlps, nn.ModuleList)
    assert len(activation.mlps) == len(layout.specs)
    assert len(tuple(activation.parameters())) > 0
    optimizer = torch.optim.SGD(activation.parameters(), lr=0.1)
    parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert parameter_ids == {id(parameter) for parameter in activation.parameters()}


def test_torch_initializer_uses_stable_per_order_streams() -> None:
    seed = 919
    activation = ChannelPreservingMLPActivation(_layout(), initializer=TorchInitializer(seed=seed))
    repeated = ChannelPreservingMLPActivation(_layout(), initializer=TorchInitializer(seed=seed))
    assert all(
        torch.equal(left, right)
        for left, right in zip(activation.parameters(), repeated.parameters(), strict=True)
    )

    for index, spec in enumerate(_layout().specs):
        expected = MLP(
            in_channels=spec.channels,
            out_channels=spec.channels,
            hidden_channels=spec.hidden_channels,
            num_hidden_layers=spec.num_hidden_layers,
            activation=spec.activation,
            bias=spec.bias,
            initializer=TorchInitializer(seed=seed).spawn(f"order_{spec.order}"),
        )
        for actual_parameter, expected_parameter in zip(
            activation.mlps[index].parameters(), expected.parameters(), strict=True
        ):
            torch.testing.assert_close(actual_parameter, expected_parameter, atol=0.0, rtol=0.0)


def test_dtype_device_and_gradients_are_preserved() -> None:
    activation = ChannelPreservingMLPActivation(
        _layout(), initializer=TorchInitializer(seed=77)
    ).to(dtype=torch.float64)
    for shape in ((2, 3, 2), (2, 3, 2, 2)):
        inputs = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        outputs = activation(inputs)
        outputs.square().sum().backward()

        assert outputs.shape == inputs.shape
        assert outputs.dtype == inputs.dtype
        assert outputs.device == inputs.device
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in activation.parameters())


def test_float32_forward_preserves_dtype_under_standard_module_convention() -> None:
    activation = ChannelPreservingMLPActivation(
        _layout(), initializer=TorchInitializer(seed=78)
    )
    inputs = torch.randn(2, 3, 2, dtype=torch.float32)

    outputs = activation(inputs)

    assert outputs.shape == inputs.shape
    assert outputs.dtype == torch.float32
    assert outputs.device == inputs.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_cuda_forward_preserves_device_dtype_and_gradients() -> None:
    activation = ChannelPreservingMLPActivation(
        _layout(), initializer=TorchInitializer(seed=79)
    ).to(device="cuda", dtype=torch.float64)
    inputs = torch.randn(2, 3, 2, device="cuda", dtype=torch.float64, requires_grad=True)

    outputs = activation(inputs)
    outputs.square().sum().backward()

    assert outputs.shape == inputs.shape
    assert outputs.dtype == inputs.dtype
    assert outputs.device == inputs.device
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_invalid_axes_fail_loudly() -> None:
    with pytest.raises(ValueError, match="tuple_axes_start"):
        ChannelActivationAxes(channel_axis=2, tuple_axes_start=2)
    with pytest.raises(ValueError, match="nonnegative"):
        ChannelActivationAxes(channel_axis=-1)


def test_invalid_type_arguments_fail_loudly() -> None:
    with pytest.raises(TypeError, match="channel_axis must be an integer"):
        ChannelActivationAxes(channel_axis="1")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tuple_axes_start must be an integer"):
        ChannelActivationAxes(tuple_axes_start="2")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="activation must be"):
        OrderMLPSpec(order=1, channels=3, activation=lambda value: value)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bias must be a bool"):
        OrderMLPSpec(order=1, channels=3, bias=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ordered sequence"):
        OrderMLPLayout(axes=ChannelActivationAxes(), specs={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="only OrderMLPSpec"):
        OrderMLPLayout(axes=ChannelActivationAxes(), specs=(object(),))  # type: ignore[arg-type]


def test_invalid_orders_fail_loudly() -> None:
    with pytest.raises(ValueError, match="order must be positive"):
        OrderMLPSpec(order=0, channels=3)
    with pytest.raises(ValueError, match="strictly increasing"):
        OrderMLPLayout(
            axes=ChannelActivationAxes(),
            specs=(OrderMLPSpec(order=2, channels=3), OrderMLPSpec(order=1, channels=3)),
        )
    activation = ChannelPreservingMLPActivation(_layout())
    with pytest.raises(ValueError, match="not configured"):
        activation(torch.randn(2, 3, 2, 2, 2))


def test_invalid_channels_fail_loudly() -> None:
    activation = ChannelPreservingMLPActivation(_layout())
    with pytest.raises(ValueError, match="input channels"):
        activation(torch.randn(2, 4, 3))


def test_invalid_ranks_fail_loudly() -> None:
    activation = ChannelPreservingMLPActivation(_layout())
    with pytest.raises(ValueError, match="input rank"):
        activation(torch.randn(2, 3))
