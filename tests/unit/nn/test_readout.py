"""Tests for readout scaffold trainability."""

from __future__ import annotations

import math

import pytest
import torch
from typeguard import TypeCheckError

from tpen.data.batch import ElectronBatch
from tpen.data.permutation import all_permutations
from tpen.data.real import Feature, zero_block
from tpen.nn import TorchInitializer
from tpen.nn.readout import PfaffianReadout
from tpen.nn.readout.pfaffian import _ODD_PADDING_IRREP, _pfaffian_single, pfaffian


class BlockContainer:
    def __init__(self, blocks: list[torch.Tensor]) -> None:
        self.blocks = blocks

    def __contains__(self, order: int) -> bool:
        return 0 <= order < len(self.blocks)


def _batch(n_electrons: int = 2) -> ElectronBatch:
    return ElectronBatch(positions=torch.zeros(1, n_electrons, 1, dtype=torch.float64))


def _pfaffian_features(n_electrons: int = 2) -> Feature:
    pair = torch.zeros(1, 2, n_electrons, n_electrons, dtype=torch.float64)
    pair[:, :, 0, 1] = torch.tensor([2.0, 4.0], dtype=torch.float64)
    pair[:, :, 1, 0] = -pair[:, :, 0, 1]
    one_body = torch.zeros(1, 1, n_electrons, dtype=torch.float64)
    return Feature([zero_block(dtype=torch.float64), one_body, pair])


def test_pfaffian_matches_known_four_by_four_formula() -> None:
    matrix = torch.tensor(
        [
            [0.0, 2.0, 3.0, 7.0],
            [-2.0, 0.0, 5.0, 11.0],
            [-3.0, -5.0, 0.0, 13.0],
            [-7.0, -11.0, -13.0, 0.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        pfaffian(matrix),
        torch.tensor(2.0 * 13.0 - 3.0 * 11.0 + 7.0 * 5.0, dtype=torch.float64),
    )


def _random_skew(shape: tuple[int, ...], n: int, seed: int) -> torch.Tensor:
    """Draw skew-symmetric matrices with leading batch shape ``shape``."""

    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(*shape, n, n, generator=generator, dtype=torch.float64)
    return raw - raw.transpose(-1, -2)


def _reference_pfaffian(matrix: torch.Tensor) -> torch.Tensor:
    """Apply the slow unbatched reference to every matrix in a batch.

    This is the oracle CLAUDE.md requires the vectorized routine to reproduce:
    one `_pfaffian_single` call per matrix, reassembled into the batch shape.
    """

    leading = tuple(matrix.shape[:-2])
    # Spelled out rather than inferred with -1: an n == 0 matrix has zero
    # elements, which makes an inferred dimension ambiguous and raises.
    count = math.prod(leading)
    flat = matrix.reshape(count, matrix.shape[-2], matrix.shape[-1])
    return torch.stack([_pfaffian_single(item) for item in flat], dim=0).reshape(leading)


# Distinct batch and channel extents so a transposed leading axis cannot pass.
_FAST_SLOW_SHAPES = [(), (5,), (3, 2)]


@pytest.mark.parametrize("n", [0, 2, 3, 4, 6])
@pytest.mark.parametrize("leading", _FAST_SLOW_SHAPES)
def test_pfaffian_batched_matches_slow_reference(n: int, leading: tuple[int, ...]) -> None:
    # CLAUDE.md, "Implement slow reference versions first": fast(x) == slow(x).
    # Covers n=2 (the production Hooke pair), n=3 (odd, Pf == 0), n=4 and n=6
    # (deeper recursion), unbatched and batched, in float64.
    matrix = _random_skew(leading, n, seed=1000 + n + 7 * len(leading))

    result = pfaffian(matrix)

    assert result.shape == leading
    torch.testing.assert_close(result, _reference_pfaffian(matrix))


def test_pfaffian_two_by_two_is_bitwise_identical_to_reference() -> None:
    # n == 2 must be the same gather the reference performs, not merely a close
    # numeric answer: the acceptance contract forbids any numerics change there.
    matrix = _random_skew((3, 2), 2, seed=17)

    assert torch.equal(pfaffian(matrix), _reference_pfaffian(matrix))


def test_pfaffian_batched_shape_is_the_leading_batch_shape() -> None:
    # Guards the [batch, channels, n, n] -> [batch, channels] contract the
    # readout depends on, with batch != channels so an axis swap is visible.
    matrix = _random_skew((3, 2), 4, seed=23)

    assert pfaffian(matrix).shape == (3, 2)


def test_pfaffian_rejects_malformed_matrix_ranks_and_shapes() -> None:
    with pytest.raises(ValueError, match="rank at least 2"):
        pfaffian(torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="must be square"):
        pfaffian(torch.zeros(3, 4, dtype=torch.float64))
    with pytest.raises(ValueError, match="must be square"):
        pfaffian(torch.zeros(2, 3, 4, dtype=torch.float64))


@pytest.mark.parametrize("n", [2, 4])
def test_pfaffian_first_and_second_derivatives_match_slow_reference(n: int) -> None:
    # The readout runs inside the autodiff Laplacian, which is a DOUBLE
    # backward, so first-order agreement is not sufficient evidence.
    raw = _random_skew((3, 2), n, seed=31).requires_grad_(True)

    def gradients(routine) -> tuple[torch.Tensor, torch.Tensor]:
        skew = raw - raw.transpose(-1, -2)
        value = routine(skew).sum()
        (first,) = torch.autograd.grad(value, raw, create_graph=True)
        # `first` must stay attached to `raw`, so this second grad genuinely
        # exercises the double-backward graph. At n == 2 the Pfaffian is linear
        # in the kernel and its true Hessian is zero, so the functional is
        # chosen to stay nonzero there too rather than pass vacuously.
        (second,) = torch.autograd.grad((first * raw).sum(), raw)
        return first, second

    fast_first, fast_second = gradients(pfaffian)
    slow_first, slow_second = gradients(_reference_pfaffian)

    torch.testing.assert_close(fast_first, slow_first)
    torch.testing.assert_close(fast_second, slow_second)
    assert fast_first.abs().sum() > 0
    assert fast_second.abs().sum() > 0


def test_pfaffian_readout_channel_pfaffians_match_slow_reference() -> None:
    # End-to-end at the readout boundary: the batched call must reproduce the
    # reference per (walker, channel) entry, including the odd-n border.
    generator = torch.Generator().manual_seed(41)
    for n_electrons in (2, 3, 4):
        pair = torch.randn(3, 2, n_electrons, n_electrons, generator=generator, dtype=torch.float64)
        one_body = torch.randn(3, 2, n_electrons, generator=generator, dtype=torch.float64)
        features = Feature([zero_block(batch_size=3, dtype=torch.float64), one_body, pair])
        readout = PfaffianReadout(channels=2)
        batch = ElectronBatch(positions=torch.zeros(3, n_electrons, 1, dtype=torch.float64))

        output = readout(features, batch)

        kernel = readout.build_skew_kernel(features, batch)
        torch.testing.assert_close(
            output.aux["channel_pfaffians"], _reference_pfaffian(kernel)
        )


def test_pfaffian_readout_second_derivative_reaches_positions() -> None:
    # T12 extended to second order: the Laplacian path differentiates the
    # readout twice, so a batched routine that silently detaches a graph must
    # fail here rather than only in a training run.
    generator = torch.Generator().manual_seed(53)
    raw = torch.randn(3, 2, 4, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    features = Feature([zero_block(batch_size=3, dtype=torch.float64), torch.empty(3, 0, 4, dtype=torch.float64), raw])
    readout = PfaffianReadout(channels=2)

    batch = ElectronBatch(positions=torch.zeros(3, 4, 1, dtype=torch.float64))

    logabs = readout(features, batch).logabs.sum()
    (first,) = torch.autograd.grad(logabs, raw, create_graph=True)
    (second,) = torch.autograd.grad(first.square().sum(), raw)

    assert torch.all(torch.isfinite(second))
    assert second.abs().sum() > 0


def test_pfaffian_readout_weights_are_fixed_by_default() -> None:
    readout = PfaffianReadout(pair_channels=2)

    output = readout(_pfaffian_features(), _batch())

    assert output.logabs.shape == (1,)
    assert "channel_weights" not in dict(readout.named_parameters())
    assert "channel_weight_buffer" in dict(readout.named_buffers())
    assert "border_weight_buffer" not in dict(readout.named_buffers())


def test_pfaffian_readout_trainable_flag_registers_weights() -> None:
    readout = PfaffianReadout(pair_channels=2, trainable=True)

    readout(_pfaffian_features(), _batch())

    parameters = dict(readout.named_parameters())
    assert set(parameters) == {"channel_weights"}
    assert parameters["channel_weights"].requires_grad


def test_pfaffian_readout_is_antisymmetric_under_even_particle_permutations() -> None:
    pair = torch.tensor(
        [
            [
                [
                    [0.0, 2.0, 3.0, 7.0],
                    [-2.0, 0.0, 5.0, 11.0],
                    [-3.0, -5.0, 0.0, 13.0],
                    [-7.0, -11.0, -13.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float64,
    )
    features = Feature([zero_block(dtype=torch.float64), torch.empty(1, 0, 4, dtype=torch.float64), pair])
    batch = _batch(n_electrons=4)
    readout = PfaffianReadout(channels=1)

    output = readout(features, batch)
    for permutation in all_permutations(4):
        permuted_output = readout(features.permute(permutation), batch.permute(permutation))

        torch.testing.assert_close(permuted_output.logabs, output.logabs)
        torch.testing.assert_close(permuted_output.sign, output.sign * permutation.sign)


def test_pfaffian_readout_uses_one_irrep_padding_block_for_odd_electrons() -> None:
    one_body = torch.tensor([[[7.0, 11.0, 13.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    pair[:, :, 0, 1] = 2.0
    pair[:, :, 0, 2] = 3.0
    pair[:, :, 1, 2] = 5.0
    pair = pair - pair.transpose(-1, -2)
    features = Feature([zero_block(dtype=torch.float64), one_body, pair])

    assert _ODD_PADDING_IRREP.parts == (1,)
    output = PfaffianReadout(channels=1)(features, _batch(n_electrons=3))

    expected = torch.tensor([2.0 * 13.0 - 3.0 * 11.0 + 7.0 * 5.0], dtype=torch.float64)
    torch.testing.assert_close(output.aux["pfaffian"], expected)
    torch.testing.assert_close(output.logabs, expected.log())
    torch.testing.assert_close(output.sign, torch.ones_like(expected))
    torch.testing.assert_close(output.aux["K"] + output.aux["K"].transpose(-1, -2), torch.zeros_like(output.aux["K"]))


def test_pfaffian_readout_is_antisymmetric_under_odd_particle_permutations() -> None:
    one_body = torch.tensor([[[7.0, 11.0, 13.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    pair[:, :, 0, 1] = 2.0
    pair[:, :, 0, 2] = 3.0
    pair[:, :, 1, 2] = 5.0
    pair = pair - pair.transpose(-1, -2)
    features = Feature([zero_block(dtype=torch.float64), one_body, pair])
    batch = _batch(n_electrons=3)
    readout = PfaffianReadout(channels=1)

    output = readout(features, batch)
    for permutation in all_permutations(3):
        permuted_output = readout(features.permute(permutation), batch.permute(permutation))

        torch.testing.assert_close(permuted_output.logabs, output.logabs)
        torch.testing.assert_close(permuted_output.sign, output.sign * permutation.sign)


def test_pfaffian_readout_builds_per_channel_bordered_kernels() -> None:
    # B1: each channel keeps its own skew kernel; odd-n padding borders every
    # channel with that channel's order-1 block (no channel mixing in the
    # kernel).
    one_body = torch.tensor([[[2.0, 4.0, 6.0], [10.0, 14.0, 18.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
    pair[:, 0, 0, 1] = 2.0
    pair[:, 0, 0, 2] = 3.0
    pair[:, 0, 1, 2] = 5.0
    pair[:, 1, 0, 1] = 4.0
    pair[:, 1, 0, 2] = 9.0
    pair[:, 1, 1, 2] = 11.0
    pair = pair - pair.transpose(-1, -2)
    features = Feature([zero_block(dtype=torch.float64), one_body, pair])

    kernel = PfaffianReadout(channels=2).build_skew_kernel(features, _batch(n_electrons=3))

    expected = torch.zeros(1, 2, 4, 4, dtype=torch.float64)
    expected[:, :, :-1, :-1] = pair
    expected[:, :, :-1, -1] = one_body
    expected[:, :, -1, :-1] = -one_body
    torch.testing.assert_close(kernel, expected)


def test_pfaffian_readout_rejects_per_channel_padding_channel_mismatch() -> None:
    # Per-channel odd-n padding is only defined when the order-1 block
    # carries one border per pair channel.
    one_body = torch.tensor([[[2.0, 4.0, 6.0]]], dtype=torch.float64)
    pair = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
    features = Feature([zero_block(dtype=torch.float64), one_body, pair])

    with pytest.raises(ValueError, match="match pair channels"):
        PfaffianReadout(channels=2)(features, _batch(n_electrons=3))


def test_pfaffian_readout_requires_one_irrep_padding_for_odd_electron_systems() -> None:
    pair = torch.zeros(1, 1, 3, 3, dtype=torch.float64)
    features_without_border = Feature([zero_block(dtype=torch.float64), torch.empty(1, 0, 3, dtype=torch.float64), pair])

    with pytest.raises(KeyError, match=r"irrep \(1\)"):
        PfaffianReadout(channels=1)(features_without_border, _batch(n_electrons=3))


def test_pfaffian_readout_does_not_expose_odd_padding_toggle() -> None:
    with pytest.raises(TypeError, match="allow_odd_electron_bordered"):
        PfaffianReadout(allow_odd_electron_bordered=False, channels=1)  # type: ignore[call-arg]


def test_pfaffian_readout_does_not_own_harmonic_confinement() -> None:
    with pytest.raises(TypeError, match="envelope"):
        PfaffianReadout(envelope=object(), channels=1)  # type: ignore[call-arg]


def test_pfaffian_readout_rejects_malformed_kernel_inputs() -> None:
    readout = PfaffianReadout(pair_channels=2)
    malformed_pair = BlockContainer([zero_block(dtype=torch.float64), torch.empty(1, 0, 3), torch.zeros(1, 1, 3)])
    feature = _pfaffian_features()
    border_mismatch = BlockContainer(
        [
            zero_block(dtype=torch.float64),
            torch.ones(1, 1, 4, dtype=torch.float64),
            torch.zeros(1, 1, 3, 3, dtype=torch.float64),
        ]
    )

    with pytest.raises(TypeCheckError, match="Feature"):
        readout.build_skew_kernel(malformed_pair)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Feature batch size"):
        readout.build_skew_kernel(feature, ElectronBatch(positions=torch.zeros(2, 2, 1, dtype=torch.float64)))
    with pytest.raises(TypeCheckError, match="Feature"):
        readout.build_skew_kernel(border_mismatch)  # type: ignore[arg-type]


def test_pfaffian_readout_returns_empty_pfaffian_for_zero_electrons() -> None:
    features = Feature(
        [
            zero_block(batch_size=1, dtype=torch.float64),
            torch.empty(1, 1, 0, dtype=torch.float64),
            torch.empty(1, 2, 0, 0, dtype=torch.float64),
        ]
    )
    batch = _batch(n_electrons=0)

    output = PfaffianReadout(pair_channels=2)(features, batch)

    torch.testing.assert_close(output.logabs, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(output.sign, torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(output.aux["pfaffian"], torch.ones(1, dtype=torch.float64))


def _two_channel_distinguishing_features() -> tuple[Feature, torch.Tensor, torch.Tensor]:
    """T6 (B1) two-channel 4x4 case: per-channel Pfaffians with hand values."""

    def skew_from_upper(a12, a13, a14, a23, a24, a34):
        matrix = torch.zeros(4, 4, dtype=torch.float64)
        matrix[0, 1], matrix[0, 2], matrix[0, 3] = a12, a13, a14
        matrix[1, 2], matrix[1, 3], matrix[2, 3] = a23, a24, a34
        return matrix - matrix.T

    # Pf[skew] = a12*a34 - a13*a24 + a14*a23.
    kernel_a = skew_from_upper(2.0, 3.0, 7.0, 5.0, 11.0, 13.0)
    kernel_b = skew_from_upper(1.0, -4.0, 2.0, 6.0, 3.0, -5.0)
    pf_a = torch.tensor(2.0 * 13.0 - 3.0 * 11.0 + 7.0 * 5.0, dtype=torch.float64)
    pf_b = torch.tensor(1.0 * -5.0 - -4.0 * 3.0 + 2.0 * 6.0, dtype=torch.float64)
    pair = torch.stack([kernel_a, kernel_b]).unsqueeze(0)
    features = Feature([zero_block(dtype=torch.float64), torch.empty(1, 0, 4, dtype=torch.float64), pair])
    return features, pf_a, pf_b


def test_pfaffian_readout_matches_per_channel_reference_and_differs_from_mixed_kernel() -> None:
    # T6 channel-semantics case (B1): the module must compute
    # Psi = sum_c w_c Pf[skew_c] and NOT Pf[sum_c w_c skew_c]. Non-uniform
    # weights make the two function classes numerically distinct here.
    from tests.helpers.tpen_reference import mixed_kernel_pfaffian_readout, per_channel_pfaffian_readout

    features, pf_a, pf_b = _two_channel_distinguishing_features()
    weights = torch.tensor([0.75, 0.25], dtype=torch.float64)
    readout = PfaffianReadout(channels=2, trainable=True).to(dtype=torch.float64)
    with torch.no_grad():
        readout.channel_weights.copy_(weights)

    output = readout(features, _batch(n_electrons=4))

    expected_psi = (weights[0] * pf_a + weights[1] * pf_b).reshape(1)
    reference_psi = per_channel_pfaffian_readout(features, weights)
    rejected_psi = mixed_kernel_pfaffian_readout(features, weights)
    torch.testing.assert_close(output.aux["pfaffian"], expected_psi)
    torch.testing.assert_close(output.aux["pfaffian"], reference_psi)
    torch.testing.assert_close(output.logabs, expected_psi.abs().log())
    torch.testing.assert_close(output.sign, torch.sign(expected_psi))
    torch.testing.assert_close(
        output.aux["channel_pfaffians"], torch.stack([pf_a, pf_b]).reshape(1, 2)
    )
    assert not torch.allclose(output.aux["pfaffian"], rejected_psi)


def test_pfaffian_readout_stays_antisymmetric_with_distinct_channel_weights() -> None:
    # T2/T6: per-channel readout must keep sign equivariance even when the
    # channel weights are non-uniform (each Pf flips with det(P)).
    features, _pf_a, _pf_b = _two_channel_distinguishing_features()
    batch = _batch(n_electrons=4)
    readout = PfaffianReadout(channels=2, trainable=True).to(dtype=torch.float64)
    with torch.no_grad():
        readout.channel_weights.copy_(torch.tensor([0.75, 0.25], dtype=torch.float64))

    output = readout(features, batch)
    for permutation in all_permutations(4):
        permuted_output = readout(features.permute(permutation), batch.permute(permutation))

        torch.testing.assert_close(permuted_output.logabs, output.logabs)
        torch.testing.assert_close(permuted_output.sign, output.sign * permutation.sign)


def test_pfaffian_readout_gradients_reach_channel_weights() -> None:
    # T12: backward through logabs must reach the readout weights with a
    # finite, not-identically-zero gradient.
    features, _pf_a, _pf_b = _two_channel_distinguishing_features()
    readout = PfaffianReadout(channels=2, trainable=True).to(dtype=torch.float64)

    readout(features, _batch(n_electrons=4)).logabs.sum().backward()

    gradient = readout.channel_weights.grad
    assert gradient is not None
    assert torch.all(torch.isfinite(gradient))
    assert gradient.abs().sum() > 0


def test_pfaffian_readout_default_initialization_preserves_control_surface() -> None:
    readout = PfaffianReadout(pair_channels=3)

    assert readout.initialization == "equal"
    torch.testing.assert_close(readout.channel_weight_buffer, torch.full((3,), 1.0 / 3))
    assert set(readout.state_dict()) == set()
    assert set(dict(readout.named_parameters())) == set()
    assert set(dict(readout.named_buffers())) == {"channel_weight_buffer"}


@pytest.mark.parametrize("initialization", ["signed_l1", "random_direction_l2"])
def test_pfaffian_readout_random_initializations_match_closed_form(
    initialization: str,
) -> None:
    initializer = TorchInitializer(seed=810012, stream="readout/a12")
    readout = PfaffianReadout(
        pair_channels=4,
        initialization=initialization,
        initializer=initializer,
    )

    z = torch.empty(4, dtype=torch.float64)
    TorchInitializer(seed=810012, stream="readout/a12").normal_(z)
    if initialization == "signed_l1":
        expected = z / z.abs().sum()
        assert torch.isclose(
            readout.channel_weight_buffer.abs().sum(),
            torch.ones((), dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )
    else:
        expected = z / (math.sqrt(4) * torch.linalg.vector_norm(z))
        assert torch.isclose(
            math.sqrt(4) * torch.linalg.vector_norm(readout.channel_weight_buffer),
            torch.ones((), dtype=torch.float64),
            rtol=1e-6,
            atol=1e-6,
        )
    torch.testing.assert_close(
        readout.channel_weight_buffer,
        expected.to(dtype=torch.get_default_dtype()),
        rtol=0,
        atol=0,
    )


def test_pfaffian_readout_a12_random_modes_share_paired_direction() -> None:
    seed_stream = TorchInitializer(seed=810013, stream="readout/a12")
    l1 = PfaffianReadout(pair_channels=4, initialization="signed_l1", initializer=seed_stream)
    l2 = PfaffianReadout(
        pair_channels=4,
        initialization="random_direction_l2",
        initializer=TorchInitializer(seed=810013, stream="readout/a12"),
    )

    ratio = l1.channel_weight_buffer / l2.channel_weight_buffer
    torch.testing.assert_close(ratio, ratio[0].expand_as(ratio), rtol=1e-6, atol=1e-7)
    assert torch.all(ratio > 0)


def test_pfaffian_readout_named_stream_is_independent_of_global_rng_and_construction_order() -> None:
    torch.manual_seed(1)
    first = PfaffianReadout(
        pair_channels=4,
        initialization="signed_l1",
        initializer=TorchInitializer(seed=810014, stream="readout/a12"),
    )
    torch.manual_seed(2)
    _ = PfaffianReadout(pair_channels=9)
    second = PfaffianReadout(
        pair_channels=4,
        initialization="signed_l1",
        initializer=TorchInitializer(seed=810014, stream="readout/a12"),
    )

    torch.testing.assert_close(first.channel_weight_buffer, second.channel_weight_buffer, rtol=0, atol=0)
    assert not torch.equal(
        first.channel_weight_buffer,
        PfaffianReadout(
            pair_channels=4,
            initialization="signed_l1",
            initializer=TorchInitializer(seed=810015, stream="readout/a12"),
        ).channel_weight_buffer,
    )
    assert not torch.equal(
        first.channel_weight_buffer,
        PfaffianReadout(
            pair_channels=4,
            initialization="signed_l1",
            initializer=TorchInitializer(seed=810014, stream="readout/other"),
        ).channel_weight_buffer,
    )


@pytest.mark.parametrize("initialization", ["equal", "signed_l1", "random_direction_l2"])
@pytest.mark.parametrize("trainable", [False, True])
def test_pfaffian_readout_initialization_preserves_trainability(
    initialization: str,
    trainable: bool,
) -> None:
    kwargs = {} if initialization == "equal" else {
        "initializer": TorchInitializer(seed=810016, stream="readout/a12")
    }
    readout = PfaffianReadout(
        pair_channels=2,
        initialization=initialization,
        trainable=trainable,
        **kwargs,
    )
    if trainable:
        readout(_pfaffian_features(), _batch()).aux["pfaffian"].sum().backward()
        assert isinstance(readout.channel_weights, torch.nn.Parameter)
        assert readout.channel_weights.grad is not None
        assert readout.channel_weights.grad.abs().sum() > 0
    else:
        assert "channel_weights" not in dict(readout.named_parameters())
        assert "channel_weight_buffer" in dict(readout.named_buffers())


def test_pfaffian_readout_rejects_unknown_initialization() -> None:
    with pytest.raises(ValueError, match="Unknown PfaffianReadout initialization"):
        PfaffianReadout(pair_channels=2, initialization="normal")
