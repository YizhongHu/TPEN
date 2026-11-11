"""Contract tests for the frozen VMC score geometry.

Tolerances here are float64 and the problems are tiny, so agreement should be
at the level of a few rounding steps.  ``1e-12`` is therefore a real bound and
not a shrug; anything looser would stop discriminating a convention error.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers.sr_dense_oracle import design_matrix, energy_gradient, energy_residual
from tpen.data.batch import (
    MaterializedParameterLogScores,
    ParameterLayout,
    ParameterSlot,
)
from tpen.training.score_geometry import (
    SCORE_CONVENTION_VERSION,
    ScoreConventions,
    ScoreGeometry,
    build_energy_residual,
    build_score_geometry,
    build_score_geometry_from_rows,
    flatten_parameter_score_blocks,
    layout_convention_fingerprint,
    unflatten_to_layout,
)
from tpen.training.statistics import IdentityStatisticsReducer, StatisticsReducer

TOLERANCE = 1.0e-12


def _layout(shapes: tuple[tuple[int, ...], ...]) -> ParameterLayout:
    """Build a float64 layout from parameter shapes."""

    slots = []
    for ordinal, shape in enumerate(shapes):
        numel = 1
        for size in shape:
            numel *= size
        slots.append(
            ParameterSlot(ordinal=ordinal, shape=shape, numel=numel, dtype=torch.float64)
        )
    return ParameterLayout(slots=tuple(slots))


def _scores(
    layout: ParameterLayout,
    *,
    n_samples: int,
    seed: int = 0,
) -> MaterializedParameterLogScores:
    """Build reproducible pseudo-random raw score blocks."""

    generator = torch.Generator().manual_seed(seed)
    blocks = tuple(
        torch.randn(
            (n_samples, *slot.shape),
            generator=generator,
            dtype=torch.float64,
        )
        for slot in layout.slots
    )
    return MaterializedParameterLogScores(layout=layout, blocks=blocks)


class _DoublingReducer(StatisticsReducer):
    """A reducer that is provably not the identity.

    Its only purpose is to prove that the geometry actually *asks* the reducer
    for its count and total instead of computing a local mean.  A test using
    only :class:`IdentityStatisticsReducer` could not tell those two apart,
    which is exactly the distributed-correctness hazard the seam exists for.
    """

    def reduce_count(self, count: int) -> int:
        return 2 * count

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        return 2.0 * value


def test_flatten_uses_layout_slot_order_then_row_major() -> None:
    """Column order is slot order, then each slot's row-major elements."""

    layout = _layout(((2,), (2, 2)))
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    second = torch.tensor(
        [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]],
        dtype=torch.float64,
    )
    scores = MaterializedParameterLogScores(layout=layout, blocks=(first, second))

    rows = flatten_parameter_score_blocks(scores)

    expected = torch.tensor(
        [
            [1.0, 2.0, 10.0, 20.0, 30.0, 40.0],
            [3.0, 4.0, 50.0, 60.0, 70.0, 80.0],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(rows, expected)


def test_unflatten_inverts_flatten_for_every_slot_shape() -> None:
    """A flat vector round-trips back into exact parameter-shaped blocks."""

    layout = _layout(((3,), (2, 2), ()))
    vector = torch.arange(8, dtype=torch.float64)

    blocks = unflatten_to_layout(vector, layout=layout)

    assert tuple(tuple(block.shape) for block in blocks) == ((3,), (2, 2), ())
    assert torch.equal(blocks[0], torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64))
    assert torch.equal(
        blocks[1], torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    )
    assert torch.equal(blocks[2], torch.tensor(7.0, dtype=torch.float64))

    # Flattening the blocks back reproduces the original vector exactly.
    rebuilt = torch.cat([block.reshape(-1) for block in blocks])
    assert torch.equal(rebuilt, vector)


def test_design_matrix_matches_independent_numpy_reference() -> None:
    """Centering and 1/sqrt(N) normalization match the NumPy oracle."""

    layout = _layout(((3,), (2,)))
    scores = _scores(layout, n_samples=7, seed=11)
    raw = flatten_parameter_score_blocks(scores).numpy()

    geometry = build_score_geometry(scores)

    np.testing.assert_allclose(
        geometry.design.numpy(),
        design_matrix(raw),
        rtol=TOLERANCE,
        atol=TOLERANCE,
    )
    assert geometry.count == 7
    assert geometry.n_parameters == 5


def test_design_columns_are_centered() -> None:
    """Every design column sums to zero, which is what centering means."""

    layout = _layout(((4,),))
    geometry = build_score_geometry(_scores(layout, n_samples=9, seed=3))

    column_sums = geometry.design.sum(dim=0)
    assert torch.allclose(
        column_sums,
        torch.zeros_like(column_sums),
        rtol=0.0,
        atol=TOLERANCE,
    )


def test_geometry_is_invariant_to_a_constant_score_shift() -> None:
    """Adding a per-parameter constant to every raw score changes nothing.

    A constant shift is exactly the gauge freedom centering removes, so this
    is the sharpest single check that centering happens at all.
    """

    layout = _layout(((3,), (2,)))
    scores = _scores(layout, n_samples=6, seed=5)
    shifted = MaterializedParameterLogScores(
        layout=layout,
        blocks=tuple(
            block + (7.5 * (index + 1))
            for index, block in enumerate(scores.blocks)
        ),
    )

    baseline = build_score_geometry(scores)
    shifted_geometry = build_score_geometry(shifted)

    assert torch.allclose(
        baseline.design,
        shifted_geometry.design,
        rtol=TOLERANCE,
        atol=TOLERANCE,
    )


def test_energy_residual_reproduces_the_ordinary_vmc_energy_gradient() -> None:
    """``A^T eps`` equals the NumPy reference energy gradient.

    This is the identity the Euclidean-limit acceptance rests on: the residual
    scale must match the objective's factor of two exactly.
    """

    layout = _layout(((4,),))
    scores = _scores(layout, n_samples=8, seed=17)
    raw = flatten_parameter_score_blocks(scores).numpy()
    energies = torch.tensor(
        [-1.2, 0.4, 3.1, -0.7, 2.2, 0.05, -3.3, 1.9], dtype=torch.float64
    )

    geometry = build_score_geometry(scores)
    residual = build_energy_residual(energies, geometry=geometry)
    gradient = geometry.design.transpose(0, 1) @ residual

    np.testing.assert_allclose(
        residual.numpy(),
        energy_residual(energies.numpy()),
        rtol=TOLERANCE,
        atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        gradient.numpy(),
        energy_gradient(raw, energies.numpy()),
        rtol=TOLERANCE,
        atol=TOLERANCE,
    )


def test_energy_gradient_matches_autograd_through_the_vmc_objective() -> None:
    """The reference gradient equals what ``compute_vmc_objective`` backpropagates.

    The oracle is checked against TPEN's *existing* objective rather than
    against the SR code, so this pins the shared convention rather than
    comparing the new engine with itself.
    """

    from tpen.training.vmc import compute_vmc_objective

    generator = torch.Generator().manual_seed(23)
    parameter = torch.nn.Parameter(torch.randn(4, generator=generator, dtype=torch.float64))
    features = torch.randn((6, 4), generator=generator, dtype=torch.float64)
    energies = torch.randn(6, generator=generator, dtype=torch.float64)

    # A linear log-amplitude makes the exact per-sample score the feature row,
    # so the score matrix is known independently of any TPEN score machinery.
    logabs = features @ parameter
    objective = compute_vmc_objective(logabs, energies)
    objective.loss.backward()

    np.testing.assert_allclose(
        parameter.grad.numpy(),
        energy_gradient(features.numpy(), energies.numpy()),
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_geometry_takes_its_count_and_total_from_the_reducer() -> None:
    """A non-identity reducer visibly changes centering and normalization.

    With :class:`IdentityStatisticsReducer` a reduced total is
    indistinguishable from a local ``.mean()``.  Doubling both the count and
    the sum leaves the mean unchanged but changes the normalization by
    ``sqrt(2)``, so only a geometry that genuinely consults the reducer for
    its count can produce this result.
    """

    layout = _layout(((3,),))
    scores = _scores(layout, n_samples=5, seed=29)

    baseline = build_score_geometry(scores, reducer=IdentityStatisticsReducer())
    doubled = build_score_geometry(scores, reducer=_DoublingReducer())

    assert baseline.count == 5
    assert doubled.count == 10
    assert torch.allclose(
        doubled.design * np.sqrt(2.0),
        baseline.design,
        rtol=TOLERANCE,
        atol=TOLERANCE,
    )


def test_energy_residual_rejects_a_count_disagreeing_with_the_geometry() -> None:
    """Score and energy centering must share one count, or the step is wrong."""

    layout = _layout(((2,),))
    scores = _scores(layout, n_samples=4, seed=31)
    geometry = build_score_geometry(scores, reducer=IdentityStatisticsReducer())
    energies = torch.zeros(4, dtype=torch.float64)

    with pytest.raises(ValueError, match="reduced count must match"):
        build_energy_residual(energies, geometry=geometry, reducer=_DoublingReducer())


def test_solve_dtype_promotes_the_design_matrix() -> None:
    """A float32 score block is promoted when the conventions request float64."""

    slot = ParameterSlot(ordinal=0, shape=(2,), numel=2, dtype=torch.float32)
    layout = ParameterLayout(slots=(slot,))
    scores = MaterializedParameterLogScores(
        layout=layout,
        blocks=(torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float32),),
    )

    geometry = build_score_geometry(
        scores,
        conventions=ScoreConventions(solve_dtype=torch.float64),
    )

    assert geometry.dtype == torch.float64


def test_fingerprint_separates_layouts_and_conventions() -> None:
    """The digest moves when either the layout or a convention changes."""

    layout = _layout(((2,), (3,)))
    other_layout = _layout(((3,), (2,)))
    conventions = ScoreConventions()

    baseline = layout_convention_fingerprint(layout, conventions)
    reordered = layout_convention_fingerprint(other_layout, conventions)
    rescaled = layout_convention_fingerprint(
        layout, ScoreConventions(energy_gradient_scale=1.0)
    )

    assert baseline["total_numel"] == 5
    assert baseline["conventions"]["version"] == SCORE_CONVENTION_VERSION
    # Same total parameter count, different slot shapes: the digest must still
    # separate them, or a reshaped model would silently reuse stale state.
    assert reordered["total_numel"] == baseline["total_numel"]
    assert reordered["digest"] != baseline["digest"]
    assert rescaled["digest"] != baseline["digest"]
    assert layout_convention_fingerprint(layout, ScoreConventions())["digest"] == (
        baseline["digest"]
    )


def test_geometry_and_conventions_reject_invalid_configuration() -> None:
    """Loud rejection of the shapes a caller could plausibly get wrong."""

    layout = _layout(((2,),))

    with pytest.raises(ValueError, match="nonzero"):
        ScoreConventions(energy_gradient_scale=0.0)
    with pytest.raises(TypeError, match="real floating"):
        ScoreConventions(solve_dtype=torch.int64)
    with pytest.raises(ValueError, match="must be a matrix"):
        build_score_geometry_from_rows(
            torch.zeros(3, dtype=torch.float64),
            layout=layout,
        )
    with pytest.raises(ValueError, match="one element per flat layout coordinate"):
        unflatten_to_layout(torch.zeros(3, dtype=torch.float64), layout=layout)
    with pytest.raises(ValueError, match="one column per flat layout coordinate"):
        ScoreGeometry(
            design=torch.zeros((2, 3), dtype=torch.float64),
            layout=layout,
            count=2,
            conventions=ScoreConventions(),
        )
    with pytest.raises(ValueError, match="positive integer"):
        ScoreGeometry(
            design=torch.zeros((2, 2), dtype=torch.float64),
            layout=layout,
            count=0,
            conventions=ScoreConventions(),
        )
