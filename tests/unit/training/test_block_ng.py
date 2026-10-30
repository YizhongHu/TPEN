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


def _block_ng_oracle_pairs(solve_dtype: torch.dtype, energy_shift: float = 0.0):
    """Apply one block-NG update and pair each block with its float64 oracle.

    The update sees ``energies + energy_shift``; the oracle always sees the
    unshifted energies.  Centering makes the exact update invariant to a
    constant energy shift, so both sides describe the same mathematical
    answer for every shift.
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

    assert method.update(_input(model, batch, packet, energies + energy_shift)).applied
    assert method.last_telemetry is not None
    assert method.last_telemetry.solve_dtype == str(solve_dtype)
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
    pairs = []
    for slot in model.parameter_binding.layout.slots:
        expected = _numpy_block_oracle(
            scores_by_ordinal[slot.ordinal].detach().numpy().reshape(6, -1),
            energies.numpy(),
            damping,
        )
        actual = updates_by_ordinal[slot.ordinal].numpy().reshape(-1)
        pairs.append((actual, expected))
    return pairs


def _max_oracle_errors(pairs) -> tuple[float, float]:
    """Return the all-block maximum absolute and relative errors.

    Relative error is zero wherever the oracle entry is exactly zero, so only
    the absolute error sees those entries.
    """

    absolute = 0.0
    relative = 0.0
    for actual, expected in pairs:
        error = np.abs(actual - expected)
        absolute = max(absolute, float(error.max()))
        relative = max(
            relative,
            float(
                np.divide(
                    error,
                    np.abs(expected),
                    out=np.zeros_like(error),
                    where=np.abs(expected) > 0.0,
                ).max()
            ),
        )
    return absolute, relative


def test_three_electron_block_ng_matches_the_float64_oracle() -> None:
    """A float64 solve reproduces the independent NumPy oracle block by block."""

    for actual, expected in _block_ng_oracle_pairs(torch.float64):
        np.testing.assert_allclose(actual, expected, rtol=1.0e-11, atol=1.0e-11)


def test_three_electron_float32_block_ng_is_invariant_to_an_energy_shift() -> None:
    """A constant energy shift must not change the float32 error against the oracle.

    This is a same-run comparison, not a calibrated threshold.  Energy
    centering makes the exact update invariant to a constant shift ``C``.
    Without centering, the float32 gradient picks up a term proportional to
    ``mean(E + C)`` times the rounding residue of the centered score column
    sums (zero in exact arithmetic), so the error grows with ``C``.  Both
    sides of the ratio come from the same process on the same machine, so the
    machine's float32 rounding level divides out.

    Do not replace the ratio with a difference or a fixed error threshold: a
    difference needs a magnitude, and a magnitude is a hardware calibration
    again.  The previous fixed thresholds were correct on the machine they
    were measured on and failed on the next one.
    """

    shift = 1.0e3
    # Bound on (shifted error) / (unshifted error), all-block maxima, measured
    # at this shift with torch 2.12.0+cpu.  Each facility was run at two
    # torch thread counts (Cannon 1 and 4, Polaris 1 and 32), which agreed to
    # within 0.1% at this shift:
    #
    #                                   relative ratio   absolute ratio
    #   healthy  FASRC Cannon (8480CL)       0.887            1.044
    #   healthy  ALCF Polaris (7543P)        1.022            1.022
    #   mutant   FASRC Cannon (8480CL)     159.5            221.3
    #   mutant   ALCF Polaris (7543P)      164.0            114.5
    #
    # "mutant" drops ``energies - energies.mean()`` in
    # ``build_block_ng_directions``.  The bound sits near the geometric mean
    # of the worst healthy ratio (1.044) and the weakest mutant ratio (114.5),
    # leaving roughly 10x on each side.  Across every shift tried (-10 to
    # 1000) the healthy ratios stayed within 0.67-1.04 on Cannon and
    # 0.997-1.022 on Polaris; the weakest mutant ratio at this shift is the
    # Polaris absolute 114.5.  Smaller shifts separate less: at a
    # shift of 10 the mutant ratio was only about 1.6 to 2.2 on Cannon.  For
    # contrast, the UNSHIFTED mutant error sits only 14% above healthy on
    # Cannon and only 1% above it on Polaris, which is why a fixed error
    # threshold could not separate them portably.
    ratio_bound = 10.0
    base_absolute, base_relative = _max_oracle_errors(_block_ng_oracle_pairs(torch.float32))
    shifted_absolute, shifted_relative = _max_oracle_errors(
        _block_ng_oracle_pairs(torch.float32, energy_shift=shift)
    )
    report = (
        f"shift {shift!r}: max relative error {base_relative!r} -> {shifted_relative!r}; "
        f"max absolute error {base_absolute!r} -> {shifted_absolute!r}"
    )
    # Relative limb first, then absolute; at this shift either limb alone
    # separates the mutant on both facilities.
    assert shifted_relative <= ratio_bound * base_relative, f"relative limb failed: {report}"
    # Relative error is masked to zero where the oracle entry is exactly
    # zero, so only the absolute limb would see such entries.  This fixture
    # has NONE (measured on both facilities), so that role is currently
    # vacuous; the absolute limb is kept because it separates the mutant on
    # its own, not to protect zero entries.
    assert shifted_absolute <= ratio_bound * base_absolute, f"absolute limb failed: {report}"
    # Deliberately UNCALIBRATED backstop.  It catches an unshifted float32
    # solve that is wrong by an order of magnitude, which the ratio above
    # cannot see because both of its sides would be wrong together.  These
    # numbers were chosen as round orders of magnitude well clear of any
    # healthy observation, not fitted to one; do not tighten them toward a
    # measured value, because that recreates a hardware-calibrated bound.
    assert base_relative <= 1.0e-1, f"backstop failed: {report}"
    assert base_absolute <= 1.0e-2, f"backstop failed: {report}"


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
