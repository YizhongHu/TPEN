"""Contract tests for exact per-tensor block-diagonal natural gradients."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers.hooke_models import build_tiny_spenn
from tests.unit.nn.test_tpen_wavefunction_parameter_scores import _build_model
from tpen.data.batch import ElectronBatch, ParameterScoreForwardPacket
from tpen.nn import InteractionMode, MaterializedParameterScoreRequest
from tpen.training.block_ng import (
    BlockDiagonalNaturalGradientUpdate,
    BlockNGPolicy,
    build_block_ng_directions,
)
from tpen.training.update import ModelParameterBinding, ScoreUpdateInput


def _connected_model():
    """Return the real three-electron TPEN fixture accepted by the score seam."""

    torch.manual_seed(0)
    return _build_model(InteractionMode.TENSOR_PRODUCT)


def _batch(n_walkers: int = 6) -> ElectronBatch:
    """Return a deterministic three-electron score batch."""

    generator = torch.Generator().manual_seed(5)
    return ElectronBatch(
        positions=torch.randn(n_walkers, 3, 3, generator=generator, dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0, 1.0]] * n_walkers, dtype=torch.float64),
    )


def _emit(model, batch: ElectronBatch):
    """Materialize raw per-sample score blocks through the public score seam."""

    packet = model.evaluate_materialized_parameter_score_request(
        request=MaterializedParameterScoreRequest(), batch=batch
    )
    assert isinstance(packet, ParameterScoreForwardPacket)
    return packet


def _input(model, batch, packet, energies: torch.Tensor) -> ScoreUpdateInput:
    """Assemble one live score-method input record."""

    return ScoreUpdateInput(
        batch=batch,
        wavefunction=packet.output,
        local_energy=energies,
        step=0,
        parameter_scores=packet.parameter_scores,
        parameter_binding=model.parameter_binding,
    )


def _numpy_block_oracle(scores: np.ndarray, energies: np.ndarray, damping: float) -> np.ndarray:
    """Independently solve one dense block using NumPy float64."""

    centered_scores = scores - scores.mean(axis=0, keepdims=True)
    centered_energy = energies - energies.mean()
    n_samples = scores.shape[0]
    fisher = centered_scores.T @ centered_scores / n_samples
    gradient = 2.0 * centered_scores.T @ centered_energy / n_samples
    return np.linalg.solve(fisher + damping * np.eye(fisher.shape[0]), gradient)


def test_tensor_blocks_cover_the_parameter_layout_once() -> None:
    """Every layout slot maps to exactly one block with no overlap or gap."""

    model = _connected_model()
    packet = _emit(model, _batch())
    layout = model.parameter_binding.layout

    covered = [slot.ordinal for slot, block in zip(layout.slots, packet.parameter_scores.blocks, strict=True)
               if tuple(block.shape[-len(slot.shape) :]) == slot.shape]
    assert covered == list(range(len(layout.slots)))
    assert len(set(covered)) == len(layout.slots)
    assert sum(slot.numel for slot in layout.slots) == layout.total_numel


@pytest.mark.parametrize(
    ("solve_dtype", "rtol", "atol"),
    [(torch.float64, 1.0e-11, 1.0e-11), (torch.float32, 2.0e-4, 2.0e-4)],
)
def test_three_electron_block_ng_float64_oracle_and_float32_sentinels(
    solve_dtype: torch.dtype, rtol: float, atol: float
) -> None:
    """Check float64 correctness and frozen-fixture float32 sentinels.

    The float32 arm is not an oracle-equivalence claim.  Its all-block error
    sentinels are specific to this frozen fixture; changing its seeds, sample
    count, damping, width, or arithmetic sequence invalidates them and
    requires re-measurement.
    """

    model = _connected_model()
    batch = _batch()
    packet = _emit(model, batch)
    energies = torch.randn(6, generator=torch.Generator().manual_seed(17), dtype=torch.float64)
    damping = 1.0e-2
    before = [parameter.detach().clone() for parameter in model.parameters()]
    method = BlockDiagonalNaturalGradientUpdate(
        torch.optim.SGD(model.parameters(), lr=1.0),
        model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
        policy=BlockNGPolicy(damping=damping, learning_rate=1.0, solve_dtype=solve_dtype),
    )

    assert method.update(_input(model, batch, packet, energies)).applied
    # Build reference and observed pairings independently by slot ordinal. A
    # same-shape block permutation must therefore disagree with its owner.
    scores_by_ordinal = {
        slot.ordinal: block
        for slot, block in zip(
            packet.parameter_scores.layout.slots,
            packet.parameter_scores.blocks,
            strict=True,
        )
    }
    updates_by_ordinal = {
        slot.ordinal: (before[slot.ordinal] - tuple(model.parameters())[slot.ordinal].detach())
        for slot in model.parameter_binding.layout.slots
    }
    absolute_errors = []
    relative_errors = []
    for slot in model.parameter_binding.layout.slots:
        expected = _numpy_block_oracle(
            scores_by_ordinal[slot.ordinal].detach().numpy().reshape(6, -1),
            energies.numpy(),
            damping,
        )
        actual = updates_by_ordinal[slot.ordinal].numpy().reshape(-1)
        if solve_dtype == torch.float64:
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        else:
            error = np.abs(actual - expected)
            absolute_errors.append(float(error.max()))
            relative_errors.append(
                float(
                    np.divide(
                        error,
                        np.abs(expected),
                        out=np.zeros_like(error),
                        where=np.abs(expected) > 0.0,
                    ).max()
                )
            )
    if solve_dtype == torch.float32:
        # Measured independently across ALL blocks: max abs lives in a P=4
        # block, while max relative lives in a different P=32 block. The
        # full-precision relative measurement was 0.008931180836996507; 9e-3
        # rounds upward with enough portability headroom while still rejecting
        # the 0.01021 regression caused by removing energy centering.
        # The absolute calibration was 1.663939e-4, so this bound also rounds
        # upward rather than excluding its own observation. It does not catch
        # the no-centering mutant: its absolute error falls to 1.6209e-4.
        assert max(absolute_errors) <= 1.664e-4
        # The relative limb is the discriminating guard: the no-centering
        # mutant reaches 0.01021381958083481 and must exceed this sentinel.
        assert max(relative_errors) <= 9.0e-3
    assert method.last_telemetry is not None
    assert method.last_telemetry.solve_dtype == str(solve_dtype)


def test_zero_score_column_stays_exactly_zero_with_a_live_block_direction() -> None:
    """A dead coordinate is zero while its tensor's live route exercises the solve."""

    scores = (
        torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [8.0, 0.0]], dtype=torch.float64
        ),
    )
    directions, gradients = build_block_ng_directions(
        scores,
        torch.tensor([2.0, -1.0, 3.0, 0.5], dtype=torch.float64),
        damping=1.0e-2,
    )
    assert torch.equal(gradients[0][1:], torch.zeros_like(gradients[0][1:]))
    assert torch.equal(directions[0][1:], torch.zeros_like(directions[0][1:]))
    # The first coordinate has a genuine Fisher solve, so replacing the whole
    # method with gradient/damping cannot satisfy this fixture.
    assert not torch.equal(directions[0][:1], gradients[0][:1] / 1.0e-2)


def test_identical_block_builds_are_bitwise_deterministic() -> None:
    """Repeated identical inputs yield bitwise-identical block directions."""

    scores = (torch.randn((7, 4), generator=torch.Generator().manual_seed(11), dtype=torch.float64),)
    energies = torch.randn(7, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
    first, _ = build_block_ng_directions(scores, energies, damping=1.0e-3)
    second, _ = build_block_ng_directions(scores, energies, damping=1.0e-3)
    assert torch.equal(first[0], second[0])


def test_one_score_request_supplies_one_block_ng_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """One score request supplies the update; the method never re-evaluates it."""

    model = _connected_model()
    batch = _batch()
    method = BlockDiagonalNaturalGradientUpdate(
        torch.optim.SGD(model.parameters(), lr=1.0e-3),
        model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
        policy=BlockNGPolicy(damping=1.0e-2, learning_rate=1.0e-3),
    )
    original = model.evaluate_materialized_parameter_score_request
    requests = 0

    def counted_request(*, request, batch):
        nonlocal requests
        requests += 1
        return original(request=request, batch=batch)

    monkeypatch.setattr(model, "evaluate_materialized_parameter_score_request", counted_request)
    # This mirrors the trainer's one score-bearing forward followed by update.
    packet = _emit(model, batch)
    energies = torch.randn(6, generator=torch.Generator().manual_seed(19), dtype=torch.float64)
    assert method.update(_input(model, batch, packet, energies)).applied
    assert requests == 1


def test_two_electron_tpen_runs_when_the_score_seam_allows_it() -> None:
    """Exercise two electrons once item 68711cfd supplies zero inactive scores."""

    model = build_tiny_spenn()
    batch = ElectronBatch(
        positions=torch.zeros((4, 2, 3), dtype=torch.float64),
        spins=torch.tensor([[1.0, -1.0]] * 4, dtype=torch.float64),
    )
    try:
        packet = _emit(model, batch)
    except RuntimeError as error:
        if "unused or disconnected" in str(error):
            pytest.skip("score seam item 68711cfd has not landed: two-electron scores disconnect")
        raise
    method = BlockDiagonalNaturalGradientUpdate(
        torch.optim.SGD(model.parameters(), lr=1.0e-3),
        model_parameters=ModelParameterBinding(parameters=tuple(model.parameters())),
        policy=BlockNGPolicy(damping=1.0e-2, learning_rate=1.0e-3),
    )
    energies = torch.tensor([0.5, -1.0, 1.5, 0.0], dtype=torch.float64)
    assert method.update(_input(model, batch, packet, energies)).applied
