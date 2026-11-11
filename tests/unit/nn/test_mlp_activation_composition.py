"""Black-box gates for composing the channel-preserving MLP activation.

The two typed operator boundaries deliberately receive the same callable
interface as the existing pointwise activations. These tests observe that
interface through call counts, argument shapes, substitution, independent
slow references, and typed permutation checks; no operator-specific MLP path
is allowed or needed.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import pytest
import torch
from torch import nn

from tpen.data.permutation import all_permutations
from tpen.data.real import Feature, Interaction, Update, zero_block
from tpen.nn import (
    ChannelActivationAxes,
    ChannelPreservingMLPActivation,
    EquivariantMixing,
    GaussianActivation,
    OrderMLPLayout,
    OrderMLPSpec,
    PathAggregation,
    TorchInitializer,
)

_DTYPE = torch.float64
_PATHS_BY_ORDER = {1: 3, 2: 4}


def _layout(*, tuple_axes_start: int, channel_axis: int = 1, bias: bool = True) -> OrderMLPLayout:
    return OrderMLPLayout(
        axes=ChannelActivationAxes(
            channel_axis=channel_axis,
            tuple_axes_start=tuple_axes_start,
        ),
        specs=(
            OrderMLPSpec(
                order=1,
                channels=2,
                hidden_channels=4,
                num_hidden_layers=1,
                activation=nn.Tanh(),
                bias=bias,
            ),
            OrderMLPSpec(
                order=2,
                channels=2,
                hidden_channels=4,
                num_hidden_layers=1,
                activation=nn.Tanh(),
                bias=bias,
            ),
        ),
    )


def _mlp_activation(*, tuple_axes_start: int, seed: int = 31, bias: bool = True) -> ChannelPreservingMLPActivation:
    return ChannelPreservingMLPActivation(
        _layout(tuple_axes_start=tuple_axes_start, bias=bias),
        initializer=TorchInitializer(seed=seed),
    ).to(dtype=_DTYPE)


def _feature(*, n_particles: int = 3, batch: int = 2, seed: int = 101) -> Feature:
    generator = torch.Generator().manual_seed(seed)
    return Feature(
        [
            zero_block(batch_size=batch, dtype=_DTYPE),
            torch.randn(batch, 2, n_particles, generator=generator, dtype=_DTYPE),
            torch.randn(batch, 2, n_particles, n_particles, generator=generator, dtype=_DTYPE),
        ]
    )


def _interaction(*, n_particles: int = 3, batch: int = 2, seed: int = 103) -> Interaction:
    generator = torch.Generator().manual_seed(seed)
    return Interaction(
        [
            zero_block(batch_size=batch, paths=max(_PATHS_BY_ORDER.values()), dtype=_DTYPE),
            torch.randn(batch, 2, 3, n_particles, generator=generator, dtype=_DTYPE),
            torch.randn(batch, 2, 4, n_particles, n_particles, generator=generator, dtype=_DTYPE),
        ]
    )


def _mixing(activation: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None) -> EquivariantMixing:
    return EquivariantMixing(
        max_order=2,
        max_virtual_order=2,
        channels=2,
        implementation="vectorized",
        initial_weight=0.5,
        activation=activation,
    ).to(dtype=_DTYPE)


def _aggregation(activation: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None) -> PathAggregation:
    return PathAggregation(
        max_order=2,
        channels=2,
        path_counts_by_order=_PATHS_BY_ORDER,
        activation=activation,
        initializer=TorchInitializer(seed=47),
    ).to(dtype=_DTYPE)


def _slow_channel_activation(activation: ChannelPreservingMLPActivation, inputs: torch.Tensor) -> torch.Tensor:
    """Apply the configured MLP by indexing inert positions, never movedim."""

    axes = activation.layout.axes
    order = inputs.ndim - axes.tuple_axes_start
    spec_index = next(index for index, spec in enumerate(activation.layout.specs) if spec.order == order)
    output = torch.empty_like(inputs)
    non_channel_axes = tuple(axis for axis in range(inputs.ndim) if axis != axes.channel_axis)
    ranges = tuple(range(int(inputs.shape[axis])) for axis in non_channel_axes)
    for positions in product(*ranges):
        index = [slice(None)] * inputs.ndim
        for axis, position in zip(non_channel_axes, positions, strict=True):
            index[axis] = position
        tensor_index = tuple(index)
        output[tensor_index] = activation.mlps[spec_index](inputs[tensor_index])
    return output


def _apply_expected_activation(
    activation: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None,
    block: torch.Tensor,
) -> torch.Tensor:
    if activation is None:
        return block
    if isinstance(activation, ChannelPreservingMLPActivation):
        return _slow_channel_activation(activation, block)
    return activation(block)


def _slow_aggregation(interaction: Interaction, module: PathAggregation, activation: object) -> Update:
    """Literal per-channel/path contraction independent of ``einsum``."""

    blocks: list[torch.Tensor] = [
        zero_block(
            batch_size=interaction.batch_size,
            device=interaction.blocks[0].device,
            dtype=interaction.blocks[0].dtype,
        )
    ]
    for order in range(1, len(interaction.blocks)):
        tensor = interaction.blocks[order]
        weight = module.weights[module.key(order)].detach()
        contracted = torch.zeros(
            (tensor.shape[0], tensor.shape[1], *tensor.shape[3:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for channel in range(tensor.shape[1]):
            for path in range(tensor.shape[2]):
                contracted[:, channel] += weight[channel, path] * tensor[:, channel, path]
        blocks.append(_apply_expected_activation(activation, contracted))
    return Update(blocks)


def test_equivariant_mixing_mlp_matches_an_independent_slow_reference() -> None:
    feature = _feature()
    activation = _mlp_activation(tuple_axes_start=3)
    actual = _mixing(activation)(feature)
    raw = EquivariantMixing(
        max_order=2,
        max_virtual_order=2,
        channels=2,
        implementation="slow",
        initial_weight=0.5,
        activation=None,
    ).to(dtype=_DTYPE)(feature)
    expected = Interaction(
        [raw.blocks[0]] + [_slow_channel_activation(activation, block) for block in raw.blocks[1:]]
    )

    matches, stats = actual.compare(expected, atol=1.0e-12, rtol=1.0e-12)
    assert matches, stats


def test_path_aggregation_mlp_matches_an_independent_slow_reference() -> None:
    interaction = _interaction()
    activation = _mlp_activation(tuple_axes_start=2)
    module = _aggregation(activation)
    actual = module(interaction)
    expected = _slow_aggregation(interaction, module, activation)

    matches, stats = actual.compare(expected, atol=1.0e-12, rtol=1.0e-12)
    assert matches, stats


class _RecordingActivation(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.calls: list[tuple[tuple[object, ...], dict[str, object], tuple[int, ...]]] = []

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        tensor = args[0]
        assert isinstance(tensor, torch.Tensor)
        self.calls.append((args, kwargs, tuple(tensor.shape)))
        return self.inner(tensor)


@pytest.mark.parametrize("boundary", ["mixing", "aggregation"])
def test_operator_treats_activation_as_black_box_single_tensor_call(boundary: str) -> None:
    inner = _mlp_activation(tuple_axes_start=3 if boundary == "mixing" else 2)
    recording = _RecordingActivation(inner)
    if boundary == "mixing":
        output = _mixing(recording)(_feature())
    else:
        output = _aggregation(recording)(_interaction())

    positive_blocks = output.blocks[1:]
    assert len(recording.calls) == len(positive_blocks) == 2
    assert [call[2] for call in recording.calls] == [tuple(block.shape) for block in positive_blocks]
    assert all(len(call[0]) == 1 and call[1] == {} for call in recording.calls)


@pytest.mark.parametrize("boundary", ["mixing", "aggregation"])
def test_operator_substitutes_none_pointwise_gaussian_and_mlp(boundary: str) -> None:
    if boundary == "mixing":
        inputs = _feature()
        factory = _mixing
        tuple_axes_start = 3
    else:
        inputs = _interaction()
        factory = _aggregation
        tuple_axes_start = 2

    activations: list[nn.Module | Callable[[torch.Tensor], torch.Tensor] | None] = [
        None,
        nn.SiLU(),
        GaussianActivation(sigma=0.7),
        _mlp_activation(tuple_axes_start=tuple_axes_start, seed=67),
    ]
    raw = factory(None)(inputs)
    for activation in activations:
        actual = factory(activation)(inputs)
        expected_blocks = [raw.blocks[0]] + [
            _apply_expected_activation(activation, block) for block in raw.blocks[1:]
        ]
        expected = type(raw)(expected_blocks)
        matches, stats = actual.compare(expected, atol=1.0e-12, rtol=1.0e-12)
        assert matches, f"{boundary} disagreed for {type(activation).__name__ if activation else 'None'}: {stats}"


@pytest.mark.parametrize("boundary", ["mixing", "aggregation"])
def test_mlp_activation_preserves_typed_boundary_equivariance(boundary: str) -> None:
    if boundary == "mixing":
        module = _mixing(_mlp_activation(tuple_axes_start=3, seed=71))
        inputs = _feature(n_particles=3, seed=107)
    else:
        module = _aggregation(_mlp_activation(tuple_axes_start=2, seed=71))
        inputs = _interaction(n_particles=3, seed=109)

    baseline = module(inputs)
    for permutation in all_permutations(3):
        permuted_first = module(inputs.permute(permutation))
        permuted_last = baseline.permute(permutation)
        matches, stats = permuted_first.compare(permuted_last, atol=1.0e-12, rtol=1.0e-12)
        assert matches, f"{boundary} failed typed equivariance for {permutation}: {stats}"


@pytest.mark.parametrize("boundary", ["mixing", "aggregation"])
def test_mlp_activation_parameters_are_eager_optimizer_visible_and_updated(boundary: str) -> None:
    if boundary == "mixing":
        module = _mixing(_mlp_activation(tuple_axes_start=3, seed=73))
        inputs = _feature(seed=113)
    else:
        module = _aggregation(_mlp_activation(tuple_axes_start=2, seed=73))
        inputs = _interaction(seed=127)

    parameters = tuple(module.parameters())
    activation_parameters = tuple(module.activation.parameters())
    assert parameters
    assert activation_parameters
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    before = tuple(parameter.detach().clone() for parameter in activation_parameters)
    loss = sum(block.square().sum() for block in module(inputs).blocks[1:])
    loss.backward()
    assert all(parameter.grad is not None for parameter in activation_parameters)
    optimizer.step()
    assert any(not torch.equal(old, new) for old, new in zip(before, activation_parameters, strict=True))


@pytest.mark.parametrize("boundary", ["mixing", "aggregation"])
def test_mlp_activation_state_roundtrips_strictly_at_typed_boundary(boundary: str) -> None:
    if boundary == "mixing":
        activation = _mlp_activation(tuple_axes_start=3, seed=79)
        module = _mixing(activation)
        inputs = _feature(seed=131)
        clone = _mixing(_mlp_activation(tuple_axes_start=3, seed=999))
    else:
        activation = _mlp_activation(tuple_axes_start=2, seed=79)
        module = _aggregation(activation)
        inputs = _interaction(seed=137)
        clone = _aggregation(_mlp_activation(tuple_axes_start=2, seed=999))

    state = module.state_dict()
    assert any(key.startswith("activation.mlps.") for key in state)
    result = clone.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    original_output = module(inputs)
    cloned_output = clone(inputs)
    matches, stats = original_output.compare(cloned_output, atol=0.0, rtol=0.0)
    assert matches, stats


def _make_zero_response(activation: ChannelPreservingMLPActivation, value: float = 0.25) -> None:
    with torch.no_grad():
        for mlp in activation.mlps:
            for parameter in mlp.parameters():
                parameter.zero_()
            mlp.layers[-1].bias.fill_(value)


def _zero_channel_response(activation: ChannelPreservingMLPActivation, order: int) -> torch.Tensor:
    """Evaluate one order MLP at its zero channel vector."""

    spec_index = next(index for index, spec in enumerate(activation.layout.specs) if spec.order == order)
    return activation.mlps[spec_index](torch.zeros(2, dtype=_DTYPE))


def test_equivariant_mixing_bias_maps_unwritten_zero_tuples_to_invariant_constant() -> None:
    feature = _feature(n_particles=3, seed=149)
    raw_module = _mixing(None)
    activation = _mlp_activation(tuple_axes_start=3, bias=True, seed=83)
    _make_zero_response(activation)
    activated = _mixing(activation)
    raw = raw_module(feature)
    output = activated(feature)

    raw_diagonal = torch.diagonal(raw.blocks[2], dim1=-2, dim2=-1)
    output_diagonal = torch.diagonal(output.blocks[2], dim1=-2, dim2=-1)
    assert torch.equal(raw_diagonal, torch.zeros_like(raw_diagonal))
    expected = _zero_channel_response(activation, order=2)
    torch.testing.assert_close(output_diagonal, expected.reshape(1, 2, 1, 1).expand_as(output_diagonal))
    assert torch.all(output_diagonal != 0)


def test_path_aggregation_bias_maps_zero_contractions_to_nonzero_output() -> None:
    interaction = Interaction(
        [
            zero_block(batch_size=1, paths=4, dtype=_DTYPE),
            torch.zeros(1, 2, 3, 3, dtype=_DTYPE),
            torch.zeros(1, 2, 4, 3, 3, dtype=_DTYPE),
        ]
    )
    activation = _mlp_activation(tuple_axes_start=2, bias=True, seed=89)
    _make_zero_response(activation)
    output = _aggregation(activation)(interaction)

    for order, block in enumerate(output.blocks[1:], start=1):
        expected = _zero_channel_response(activation, order=order)
        torch.testing.assert_close(block, expected.reshape(1, 2, *([1] * (block.ndim - 2))).expand_as(block))
        assert torch.all(block != 0)
