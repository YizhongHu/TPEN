"""Focused coordinate-gradient request tests for :class:`TPENWaveFunction`."""

from __future__ import annotations

import pytest
import torch

from tpen.data.batch import CoordinateForwardPacket, ElectronBatch
from tpen.data.permutation import Permutation
from tpen.nn import CoordinateGradientRequest
from tests.helpers.hooke_models import build_tiny_spenn, tiny_pair_batch


def test_coordinate_request_preserves_value_path_and_skips_default_derivative_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_tiny_spenn()
    batch = tiny_pair_batch(n_walkers=2)
    primal = model(batch)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("default forward must not request coordinate derivatives")

    with monkeypatch.context() as context:
        context.setattr(torch.autograd, "grad", fail_if_called)
        default_again = model(batch)
    torch.testing.assert_close(default_again.logabs, primal.logabs)
    torch.testing.assert_close(default_again.sign, primal.sign)

    packet = model(batch, request=CoordinateGradientRequest())
    assert isinstance(packet, CoordinateForwardPacket)
    assert packet.as_output() is packet.output
    torch.testing.assert_close(packet.as_output().logabs, primal.logabs)
    torch.testing.assert_close(packet.as_output().sign, primal.sign)
    assert packet.coordinates.values.shape == (2, 2, 3)


def test_coordinate_request_agrees_with_slow_autograd_oracle() -> None:
    model = build_tiny_spenn()
    batch = tiny_pair_batch(n_walkers=2)
    positions = batch.positions.detach().requires_grad_(True)
    oracle_output = model(ElectronBatch(positions=positions, spins=batch.spins))
    oracle_gradient = torch.autograd.grad(oracle_output.logabs.sum(), positions)[0]

    packet = model(batch, request=CoordinateGradientRequest())
    torch.testing.assert_close(packet.coordinates.values, oracle_gradient)


def test_coordinate_request_primal_is_value_only_after_vjp() -> None:
    model = build_tiny_spenn()
    batch = tiny_pair_batch(n_walkers=1)

    packet = model(batch, request=CoordinateGradientRequest())

    assert not packet.output.logabs.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        packet.output.logabs.sum().backward()


def test_coordinate_request_rejects_inference_mode() -> None:
    model = build_tiny_spenn()
    batch = tiny_pair_batch(n_walkers=1)

    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference mode"):
        model(batch, request=CoordinateGradientRequest())


def test_coordinate_request_preserves_electron_permutation_semantics() -> None:
    model = build_tiny_spenn()
    batch = tiny_pair_batch(n_walkers=2)
    permutation = Permutation((1, 0))

    packet = model(batch, request=CoordinateGradientRequest())
    permuted_packet = model(batch.permute(permutation), request=CoordinateGradientRequest())

    matches, stats = permuted_packet.compare(packet.permute(permutation), atol=1.0e-10, rtol=1.0e-10)
    assert matches, stats


def test_coordinate_request_matches_concatenated_samples() -> None:
    model = build_tiny_spenn()
    flat = tiny_pair_batch(n_walkers=4)
    batched = ElectronBatch(
        positions=flat.positions.reshape(2, 2, 2, 3),
        spins=flat.spins.reshape(2, 2, 2),
    )

    batched_packet = model(batched, request=CoordinateGradientRequest())
    flat_packet = model(flat, request=CoordinateGradientRequest())

    torch.testing.assert_close(batched_packet.output.logabs.reshape(-1), flat_packet.output.logabs)
    torch.testing.assert_close(batched_packet.coordinates.values.reshape(-1, 2, 3), flat_packet.coordinates.values)
