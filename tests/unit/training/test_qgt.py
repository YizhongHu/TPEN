"""Contract tests for QGT actions, dense forms, damping, and dense solves.

The central claim under test is that the parameter-space and sample-space
routes are the *same* linear algebra reached two structurally different ways,
and that both agree with an independent NumPy reference.  A test that only
compared the two routes to each other would pass even if both shared one wrong
centering convention, so the NumPy oracle anchors them.

Direct algebra is compared at ``1e-12`` (float64, a handful of rounding
steps).  Solves are compared at ``1e-9``: an LU solve and a symmetric
eigendecomposition of the same matrix do not round identically, and the gap
grows with the condition number, which damping bounds but does not remove.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers.sr_dense_oracle import damping_shift, sr_direction
from tpen.data.batch import ParameterLayout, ParameterSlot
from tpen.training.qgt import (
    DampingPolicy,
    QGTOperator,
    solve_parameter_space,
    solve_sample_space,
)
from tpen.training.score_geometry import (
    ScoreConventions,
    build_energy_residual,
    build_score_geometry_from_rows,
)

ALGEBRA_TOLERANCE = 1.0e-12
SOLVE_TOLERANCE = 1.0e-9


def _layout(n_parameters: int) -> ParameterLayout:
    """Build a single-slot float64 layout of the requested width."""

    slot = ParameterSlot(
        ordinal=0,
        shape=(n_parameters,),
        numel=n_parameters,
        dtype=torch.float64,
    )
    return ParameterLayout(slots=(slot,))


def _problem(
    *,
    n_samples: int,
    n_parameters: int,
    seed: int = 0,
    duplicate_last_column: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reproducible raw score rows and local energies."""

    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn((n_samples, n_parameters), generator=generator, dtype=torch.float64)
    if duplicate_last_column:
        # An exactly repeated column makes the QGT singular by construction,
        # which is the rank-deficient case the acceptance contract names.
        rows[:, -1] = rows[:, 0]
    energies = torch.randn(n_samples, generator=generator, dtype=torch.float64)
    return rows, energies


def _operator(rows: torch.Tensor) -> QGTOperator:
    """Build an operator over the centered, normalized design matrix."""

    geometry = build_score_geometry_from_rows(
        rows,
        layout=_layout(int(rows.shape[1])),
        conventions=ScoreConventions(),
    )
    return QGTOperator(geometry)


def test_dense_parameter_qgt_matches_the_numpy_reference() -> None:
    """``S = A^T A`` agrees with the independent oracle and is symmetric."""

    rows, _ = _problem(n_samples=9, n_parameters=4, seed=1)
    operator = _operator(rows)

    qgt = operator.dense_parameter_qgt()

    np.testing.assert_allclose(
        qgt.numpy(),
        sr_direction(rows.numpy(), np.zeros(9)).qgt,
        rtol=ALGEBRA_TOLERANCE,
        atol=ALGEBRA_TOLERANCE,
    )
    assert torch.equal(qgt, qgt.transpose(0, 1))


def test_actions_agree_with_the_dense_matrices_they_avoid_forming() -> None:
    """``Jv``, ``J^T u``, and both Gram actions match their dense counterparts."""

    rows, _ = _problem(n_samples=7, n_parameters=5, seed=2)
    operator = _operator(rows)
    design = operator.geometry.design
    generator = torch.Generator().manual_seed(99)
    parameter_vector = torch.randn(5, generator=generator, dtype=torch.float64)
    sample_vector = torch.randn(7, generator=generator, dtype=torch.float64)

    assert torch.allclose(
        operator.jv(parameter_vector),
        design @ parameter_vector,
        rtol=ALGEBRA_TOLERANCE,
        atol=ALGEBRA_TOLERANCE,
    )
    assert torch.allclose(
        operator.jt_u(sample_vector),
        design.transpose(0, 1) @ sample_vector,
        rtol=ALGEBRA_TOLERANCE,
        atol=ALGEBRA_TOLERANCE,
    )
    assert torch.allclose(
        operator.qgt_action(parameter_vector),
        operator.dense_parameter_qgt() @ parameter_vector,
        rtol=ALGEBRA_TOLERANCE,
        atol=ALGEBRA_TOLERANCE,
    )
    assert torch.allclose(
        operator.sample_gram_action(sample_vector),
        operator.dense_sample_gram() @ sample_vector,
        rtol=ALGEBRA_TOLERANCE,
        atol=ALGEBRA_TOLERANCE,
    )


def test_trace_equals_both_matrix_traces() -> None:
    """``trace(S) == trace(T) == ||A||_F^2`` is what anchors relative damping."""

    rows, _ = _problem(n_samples=6, n_parameters=3, seed=4)
    operator = _operator(rows)

    trace = float(operator.trace().item())

    assert trace == pytest.approx(
        float(torch.diagonal(operator.dense_parameter_qgt()).sum().item()),
        rel=ALGEBRA_TOLERANCE,
    )
    assert trace == pytest.approx(
        float(torch.diagonal(operator.dense_sample_gram()).sum().item()),
        rel=ALGEBRA_TOLERANCE,
    )


def test_damping_shift_matches_the_reference_definition() -> None:
    """The shift is ``absolute + relative * trace(S) / P``, floored by ``minimum``."""

    rows, _ = _problem(n_samples=8, n_parameters=4, seed=6)
    operator = _operator(rows)
    policy = DampingPolicy(absolute=1.0e-4, relative=2.0e-2, minimum=0.0)

    shift = policy.shift(trace=float(operator.trace().item()), n_parameters=4)

    assert shift == pytest.approx(
        damping_shift(rows.numpy(), absolute=1.0e-4, relative=2.0e-2),
        rel=ALGEBRA_TOLERANCE,
    )
    # The floor wins when it is larger than the computed shift.
    floored = DampingPolicy(absolute=0.0, relative=0.0, minimum=0.25)
    assert floored.shift(trace=1.0e6, n_parameters=4) == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("n_samples", "n_parameters"),
    [(12, 4), (4, 12), (6, 6)],
    ids=["overdetermined", "minsr-regime", "square"],
)
def test_parameter_space_solve_matches_the_numpy_oracle(
    n_samples: int,
    n_parameters: int,
) -> None:
    """The dense route reproduces an independent LU solve in every shape regime."""

    rows, energies = _problem(n_samples=n_samples, n_parameters=n_parameters, seed=7)
    operator = _operator(rows)
    residual = build_energy_residual(energies, geometry=operator.geometry)
    policy = DampingPolicy(absolute=0.0, relative=1.0e-2)

    direction, diagnostics = solve_parameter_space(operator, residual, damping=policy)

    expected = sr_direction(rows.numpy(), energies.numpy(), absolute=0.0, relative=1.0e-2)
    np.testing.assert_allclose(
        direction.numpy(),
        expected.direction,
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )
    assert diagnostics.space == "parameter"
    assert diagnostics.n_modes == n_parameters
    assert diagnostics.shift == pytest.approx(expected.shift, rel=ALGEBRA_TOLERANCE)


@pytest.mark.parametrize(
    ("n_samples", "n_parameters"),
    [(12, 4), (4, 12), (6, 6)],
    ids=["overdetermined", "minsr-regime", "square"],
)
def test_sample_space_solve_equals_parameter_space_and_the_oracle(
    n_samples: int,
    n_parameters: int,
) -> None:
    """The push-through identity holds, and both routes match the reference.

    This is the acceptance criterion that SR and minSR agree where they are
    mathematically equivalent.  It is checked against the oracle too, so a
    shared convention error cannot hide inside a two-route agreement.
    """

    rows, energies = _problem(n_samples=n_samples, n_parameters=n_parameters, seed=8)
    operator = _operator(rows)
    residual = build_energy_residual(energies, geometry=operator.geometry)
    policy = DampingPolicy(absolute=0.0, relative=1.0e-2)

    parameter_direction, parameter_diagnostics = solve_parameter_space(
        operator, residual, damping=policy
    )
    sample_direction, sample_diagnostics = solve_sample_space(
        operator, residual, damping=policy
    )

    np.testing.assert_allclose(
        sample_direction.numpy(),
        parameter_direction.numpy(),
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        sample_direction.numpy(),
        sr_direction(
            rows.numpy(), energies.numpy(), absolute=0.0, relative=1.0e-2
        ).direction,
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )
    assert sample_diagnostics.space == "sample"
    assert sample_diagnostics.n_modes == n_samples
    # The identical scalar shift is what makes the two routes the same
    # algorithm rather than two similar ones.
    assert sample_diagnostics.shift == pytest.approx(
        parameter_diagnostics.shift, rel=ALGEBRA_TOLERANCE
    )


def test_rank_deficient_qgt_is_solved_consistently_by_both_routes() -> None:
    """A singular QGT stays well posed and both routes still agree."""

    rows, energies = _problem(
        n_samples=10, n_parameters=4, seed=9, duplicate_last_column=True
    )
    operator = _operator(rows)
    residual = build_energy_residual(energies, geometry=operator.geometry)
    policy = DampingPolicy(absolute=1.0e-6, relative=0.0)

    eigenvalues = torch.linalg.eigvalsh(operator.dense_parameter_qgt())
    assert float(eigenvalues[0].item()) < 1.0e-14, "the duplicate column should be singular"

    parameter_direction, _ = solve_parameter_space(operator, residual, damping=policy)
    sample_direction, _ = solve_sample_space(operator, residual, damping=policy)

    np.testing.assert_allclose(
        parameter_direction.numpy(),
        sample_direction.numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        parameter_direction.numpy(),
        sr_direction(
            rows.numpy(), energies.numpy(), absolute=1.0e-6, relative=0.0
        ).direction,
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_rank_truncation_discards_modes_and_preserves_route_agreement() -> None:
    """A relative cutoff removes modes, is reported, and keeps both routes equal.

    The cutoff is chosen large enough to discard a genuinely nonzero mode, not
    merely the exact null mode, so the test discriminates a real truncation
    from a no-op.
    """

    rows, energies = _problem(
        n_samples=10, n_parameters=4, seed=10, duplicate_last_column=True
    )
    operator = _operator(rows)
    residual = build_energy_residual(energies, geometry=operator.geometry)
    policy = DampingPolicy(absolute=1.0e-8, relative=0.0)

    # Derive the cutoff from the actual spectrum so it provably discards the
    # smallest NONZERO mode, not just the exact null one. A fixed constant
    # could silently degrade into a no-op if the random spectrum shifted, and
    # a no-op cutoff would make the route-agreement assertion below vacuous.
    eigenvalues = torch.linalg.eigvalsh(operator.dense_parameter_qgt()).clamp_min(0.0)
    nonzero = eigenvalues[eigenvalues > 1.0e-12 * float(eigenvalues[-1].item())]
    assert nonzero.numel() >= 2, "need at least two nonzero modes to truncate one"
    smallest_nonzero = float(nonzero[0].item())
    next_nonzero = float(nonzero[1].item())
    cutoff = 0.5 * (smallest_nonzero + next_nonzero) / float(eigenvalues[-1].item())

    untruncated, plain_diagnostics = solve_parameter_space(
        operator, residual, damping=policy, rank_cutoff=0.0
    )
    parameter_direction, parameter_diagnostics = solve_parameter_space(
        operator, residual, damping=policy, rank_cutoff=cutoff
    )
    sample_direction, sample_diagnostics = solve_sample_space(
        operator, residual, damping=policy, rank_cutoff=cutoff
    )

    assert plain_diagnostics.truncated_modes == 0
    assert parameter_diagnostics.retained_modes < parameter_diagnostics.n_modes
    assert parameter_diagnostics.truncated_modes >= 1
    assert sample_diagnostics.retained_modes == parameter_diagnostics.retained_modes
    # Truncation must actually change the answer, or the cutoff was inert and
    # the agreement below would prove nothing.
    assert not np.allclose(
        parameter_direction.numpy(), untruncated.numpy(), rtol=1.0e-3, atol=1.0e-3
    )
    np.testing.assert_allclose(
        sample_direction.numpy(),
        parameter_direction.numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        parameter_direction.numpy(),
        sr_direction(
            rows.numpy(),
            energies.numpy(),
            absolute=1.0e-8,
            relative=0.0,
            rank_cutoff=cutoff,
        ).direction,
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_solves_reject_undamped_singular_systems_and_bad_inputs() -> None:
    """Degenerate configurations fail loudly instead of returning a number."""

    rows, energies = _problem(
        n_samples=8, n_parameters=3, seed=12, duplicate_last_column=True
    )
    operator = _operator(rows)
    residual = build_energy_residual(energies, geometry=operator.geometry)

    with pytest.raises(ValueError, match="singular"):
        solve_parameter_space(
            operator, residual, damping=DampingPolicy(absolute=0.0, relative=0.0)
        )
    with pytest.raises(ValueError, match="rank_cutoff"):
        solve_parameter_space(
            operator,
            residual,
            damping=DampingPolicy(absolute=1.0e-3, relative=0.0),
            rank_cutoff=1.0,
        )
    with pytest.raises(ValueError, match="non-finite"):
        solve_parameter_space(
            operator,
            torch.full((8,), float("nan"), dtype=torch.float64),
            damping=DampingPolicy(absolute=1.0e-3, relative=0.0),
        )
    with pytest.raises(ValueError, match="flat vector of length 3"):
        operator.jv(torch.zeros(4, dtype=torch.float64))
    with pytest.raises(ValueError, match="non-negative"):
        DampingPolicy(absolute=-1.0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_diagnostics_report_the_dtype_the_solve_actually_ran_in(dtype) -> None:
    """The reported dtype is populated on the CONFIGURED path, both routes.

    Covers the ordinary production configuration, in which `solve_dtype` IS
    set -- the shipped presets set `float64` -- so the reported value is
    exercised on the path real runs take.

    WHAT THIS TEST DOES NOT DO, corrected after reviewer round 2. An earlier
    docstring here claimed that parametrizing over two dtypes meant "an
    implementation echoing a hardcoded or configured string" could not satisfy
    it. Only the first half was true. Both parametrizations set
    `ScoreConventions(solve_dtype=dtype)` on rows cast to that SAME dtype, so
    configured and observed are aliases in every arm and an implementation
    returning `str(conventions.solve_dtype)` passes both. The reviewer proved
    it with a mutant: this test stayed green under a configured-echo, while
    `test_sr_review_probes.py::test_t6_reported_dtype_is_observed_not_an_echo_of_the_configuration`
    went red.

    That is not a bad choice of dtypes; it could not be fixed by choosing
    different ones. `build_score_geometry_from_rows` casts the rows TO any
    configured `solve_dtype`, so configured == observed BY CONSTRUCTION
    whenever it is set. Leaving it unset is the only way at this seam to make
    the two diverge, which is exactly what T6 does and why the discriminating
    assertion lives there rather than here.

    Keep both: T6 owns the discrimination, this test owns the configured path
    that T6 by construction cannot cover.
    """

    rows, energies = _problem(n_samples=8, n_parameters=3, seed=21)
    geometry = build_score_geometry_from_rows(
        rows.to(dtype=dtype),
        layout=_layout(3),
        conventions=ScoreConventions(solve_dtype=dtype),
    )
    operator = QGTOperator(geometry)
    residual = build_energy_residual(energies, geometry=geometry)
    policy = DampingPolicy(absolute=0.0, relative=1.0e-2)

    _, parameter_diagnostics = solve_parameter_space(operator, residual, damping=policy)
    _, sample_diagnostics = solve_sample_space(operator, residual, damping=policy)

    assert parameter_diagnostics.dtype == str(dtype)
    assert sample_diagnostics.dtype == str(dtype)
    assert parameter_diagnostics.as_metrics()["qgt_dtype"] == str(dtype)
    # It must describe the matrix that was actually factorized, so it agrees
    # with the design matrix rather than with the request that produced it.
    assert parameter_diagnostics.dtype == str(geometry.design.dtype)
