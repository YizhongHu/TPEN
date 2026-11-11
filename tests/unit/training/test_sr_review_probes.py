"""Reviewer probes for SR Lane N, review round 1 (PR 472 / PR 476).

ADOPTED FROM THE DURABLE REVIEWER. Written by the reviewer's test-implementation
lane on `claude/vmc-sr/review-n-r1` @ `fdc460f218d1082ea0dcd362e976ab1672e40ee5`,
accepted in full by the implementation lane and landed here.

The reviewer's single file is SPLIT ACROSS THE TWO STACK LAYERS, by dependency
rather than by preference: T1-T3 and T6 exercise only the score geometry and the
QGT, which live in this layer (PR 472), so they land here. T4-T5 need the trainer
and import helpers from `test_sr_trainer_integration`, which exist only in the N3
layer, so they are appended to THIS SAME FILE there (PR 476). Test bodies are
adopted unchanged; only the module docstring and the import list differ from the
reviewer's original, because a layer cannot import what it does not yet contain.

* T1 -- an asymmetric reducer proves score/energy centering is built from the
  reducer's REDUCED total, not from a local `.mean()` that only borrows the
  reducer's count. Reviewer finding F1: the pre-existing `_DoublingReducer`
  probe doubles count AND sum, leaving the reduced mean equal to the local
  mean, so a local-mean mutant passed the whole suite. Proven by reviewer
  mutant M1.
* T2 -- the QGT singular guard is bidirectional: it refuses a matrix truly
  singular at the tolerance boundary, and does NOT refuse a well-conditioned
  matrix whose smallest eigenvalue is merely small but legitimate. Reviewer
  finding F2: the guard was tested only in the refuse direction, so an
  over-closed guard (mutant M2) passed.
* T3 -- probes this lane's own receipt claim that a fixed score seam emitting
  exactly-zero score columns for structurally inactive parameters needs no
  change on the SR side. Reviewer finding F3. The claim HOLDS; landing the test
  keeps the future 2-electron unblock (item `68711cfd`) from regressing here.
* T4 -- a checkpoint state carrying a stateful SR method's envelope is
  `json.dumps`-serializable and round-trips exactly. Reviewer finding F6: the
  checkpoint regression this lane self-caught (job 44572457, Adam optimizer
  tensors leaking into `trainer.json`) had no direct JSON pin, so a future
  warm-start tensor in the envelope would reintroduce it undetected.
* T5 -- resuming a stateful method from state missing the `update_method` key
  must RAISE rather than silently leave the method's counters unrestored.
  Reviewer finding F4; this probe was RED on delivery and is green here against
  the fix in `VMCTrainer.load_state_dict`.
* T6 -- round 2. The reported solve dtype is OBSERVED from the factorized
  matrix, not an echo of the configured `ScoreConventions.solve_dtype`. The
  lane's own `test_qgt.py::test_diagnostics_report_the_dtype_the_solve_actually_ran_in`
  parametrizes over float64/float32 but always sets `solve_dtype` equal to the
  incoming dtype, so the configured value and the observed matrix dtype never
  diverge across its parametrizations and a configured-echo implementation
  passes both undetected. T6 leaves `solve_dtype` unset (`None`) so the two
  can disagree: a configured-echo implementation reports the string `"None"`
  and goes red.

Tolerances mirror the sibling contract-test modules this file draws its helpers
and conventions from: float64 problems, `1e-9`-`1e-12` for well-conditioned
direct comparisons, with an explicitly looser and explicitly justified tolerance
in T2(a) where the construction sits deliberately near the edge of float64's
numerical rank.
"""
from __future__ import annotations

import json


import numpy as np
import pytest
import torch

from tests.helpers.sr_dense_oracle import sr_direction
from tests.unit.training.test_sr_trainer_integration import _fit, _sr_method
from tpen.data.batch import (
    MaterializedParameterLogScores,
    ParameterLayout,
    ParameterSlot,
)
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
    flatten_parameter_score_blocks,
)
from tpen.training.statistics import StatisticsReducer
from tpen.training.trainer import VMCTrainer

TIGHT_TOLERANCE = 1.0e-12
SOLVE_TOLERANCE = 1.0e-9


def _layout(shapes: tuple[tuple[int, ...], ...]) -> ParameterLayout:
    """Build a float64 layout from parameter shapes, matching the sibling module."""

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
    seed: int,
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


def _problem_with_duplicate_column(
    *, n_samples: int, n_params: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw score rows made exactly rank-deficient by a duplicated column."""

    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn((n_samples, n_params), generator=generator, dtype=torch.float64)
    rows[:, -1] = rows[:, 0]
    energies = torch.randn(n_samples, generator=generator, dtype=torch.float64)
    return rows, energies


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------


class _AsymmetricReducer(StatisticsReducer):
    """Doubles the count, TRIPLES the sum: the reduced mean is 1.5x the local mean.

    `_DoublingReducer` in `test_score_geometry.py` doubles both count and sum,
    so its reduced mean is identical to the local mean -- a mutant that centers
    with `rows.mean(dim=0)` while taking only the COUNT from the reducer
    (discarding the reducer's sum entirely) still passes that test, because the
    two means happen to coincide. This reducer makes the reduced mean provably
    different from the local mean, so the same mutant cannot pass here.
    """

    def reduce_count(self, count: int) -> int:
        return 2 * count

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        return 3.0 * value


def test_t1_centering_uses_the_reduced_total_not_a_local_mean() -> None:
    """T1: centering must consult the reducer's TOTAL, not a local `.mean()`.

    Feeds known rows through an asymmetric reducer (count x2, sum x3, so the
    reduced mean is 1.5x the local mean) and checks the design matrix against
    an independent NumPy computation of
    `(rows - reduced_total / reduced_count) / sqrt(reduced_count)`. The same
    asymmetric reducer is reused for `build_energy_residual`, both because the
    task requires a consistent reduced count between the two calls (the
    function raises otherwise) and because it is the same hazard applied to
    the energy residual.

    Expected to PASS against current code.
    """

    layout = _layout(((4,),))
    scores = _scores(layout, n_samples=6, seed=41)
    rows = flatten_parameter_score_blocks(scores)

    geometry = build_score_geometry_from_rows(
        rows,
        layout=layout,
        reducer=_AsymmetricReducer(),
    )

    rows_np = rows.numpy()
    reduced_count = 2 * rows_np.shape[0]
    reduced_total = 3.0 * rows_np.sum(axis=0)
    expected_design = (rows_np - reduced_total / reduced_count) / np.sqrt(reduced_count)

    np.testing.assert_allclose(
        geometry.design.numpy(),
        expected_design,
        rtol=TIGHT_TOLERANCE,
        atol=TIGHT_TOLERANCE,
    )
    assert geometry.count == reduced_count

    energies = torch.tensor(
        [-1.1, 0.7, 2.3, -0.4, 1.9, -2.2], dtype=torch.float64
    )
    residual = build_energy_residual(
        energies, geometry=geometry, reducer=_AsymmetricReducer()
    )

    energies_np = energies.numpy()
    reduced_energy_total = 3.0 * energies_np.sum()
    expected_residual = (
        2.0
        * (energies_np - reduced_energy_total / reduced_count)
        / np.sqrt(reduced_count)
    )
    np.testing.assert_allclose(
        residual.numpy(),
        expected_residual,
        rtol=TIGHT_TOLERANCE,
        atol=TIGHT_TOLERANCE,
    )


# ---------------------------------------------------------------------------
# T2
# ---------------------------------------------------------------------------


def test_t2_singular_guard_is_bidirectional() -> None:
    """T2: the rank guard refuses true singularity, not a legitimate small spectrum.

    (a) Builds a design matrix by synthesizing its QGT spectrum directly (via
        an orthonormal sample-space basis and an independent parameter-space
        rotation), placing the smallest eigenvalue at roughly 1e3x the
        numerical-rank tolerance `P * eps * max_eigenvalue`. The margin
        assumption is measured and asserted from the ACTUAL spectrum (not
        merely assumed from the construction), so it cannot silently degrade.
        Solving with zero damping and zero rank cutoff must succeed and agree
        with the independent NumPy oracle.

    (b) Builds an exactly singular matrix (duplicated score column), then
        damps it with a shift strictly between the measured smallest
        eigenvalue and the measured tolerance -- guaranteed to keep
        `min_retained + shift` below the tolerance regardless of exactly where
        the near-zero eigenvalue's rounding noise landed (the `+/-1e-17`
        hazard the guard's own module docstring documents). This must raise
        'singular'.

    Both sub-tests are expected to PASS against current code.
    """

    eps64 = torch.finfo(torch.float64).eps

    # --- (a) over-restriction side: legitimate small-but-nonzero spectrum ---
    n_samples, n_params = 20, 6
    generator = torch.Generator().manual_seed(211)

    raw = torch.randn(n_samples, n_params, generator=generator, dtype=torch.float64)
    raw = raw - raw.mean(dim=0, keepdim=True)
    orthonormal_columns, _ = torch.linalg.qr(raw)
    rotation_raw = torch.randn(
        n_params, n_params, generator=generator, dtype=torch.float64
    )
    rotation, _ = torch.linalg.qr(rotation_raw)

    max_eig_target = 10.0
    approx_tolerance = n_params * eps64 * max_eig_target
    min_eig_target = 1.0e3 * approx_tolerance
    target_eigenvalues = torch.tensor(
        [max_eig_target, 8.0, 6.0, 4.0, 2.0, min_eig_target], dtype=torch.float64
    )
    singular_values = torch.sqrt(float(n_samples) * target_eigenvalues)
    rows = orthonormal_columns @ torch.diag(singular_values) @ rotation.transpose(0, 1)
    energies = torch.randn(n_samples, generator=generator, dtype=torch.float64)

    layout = _layout(((n_params,),))
    geometry = build_score_geometry_from_rows(
        rows, layout=layout, conventions=ScoreConventions()
    )
    operator = QGTOperator(geometry)

    measured_eigenvalues = torch.linalg.eigvalsh(operator.dense_parameter_qgt()).clamp_min(
        0.0
    )
    max_eig = float(measured_eigenvalues[-1])
    min_eig = float(measured_eigenvalues[0])
    tolerance = n_params * eps64 * max_eig
    ratio = min_eig / tolerance
    assert 1.0e2 <= ratio <= 1.0e4, (
        "construction did not land in the 'legitimate small spectrum' regime: "
        f"measured min eigenvalue is {ratio:.1f}x the rank tolerance, expected ~1e3x"
    )

    residual = build_energy_residual(energies, geometry=geometry)
    policy = DampingPolicy(absolute=0.0, relative=0.0, minimum=0.0)
    direction, _ = solve_parameter_space(operator, residual, damping=policy, rank_cutoff=0.0)

    # The solve residual is a backward-stability check, robust to the
    # matrix's condition number: an accepted 'solve' must actually satisfy the
    # linear system it claims to solve, independent of how sensitive the
    # solution itself is to rounding.
    gradient = operator.energy_gradient(residual)
    solved = operator.dense_parameter_qgt() @ direction
    residual_norm = float(torch.linalg.vector_norm(solved - gradient).item())
    gradient_norm = float(torch.linalg.vector_norm(gradient).item())
    assert residual_norm <= 1.0e-6 * gradient_norm, (
        "the accepted solve does not actually satisfy (S + shift I) delta = g"
    )

    oracle = sr_direction(
        rows.numpy(), energies.numpy(), absolute=0.0, relative=0.0, minimum=0.0
    )
    relative_gap = float(
        np.linalg.norm(direction.numpy() - oracle.direction)
        / np.linalg.norm(oracle.direction)
    )
    # This matrix is deliberately ill-conditioned (~1e12 by construction: the
    # min/max eigenvalue ratio is fixed at ~1e3 * P * eps regardless of overall
    # scale). A symmetric eigendecomposition (subject) and an independent LU
    # solve (oracle) are both backward-stable, so their FORWARD error bound is
    # O(condition_number * eps) ~ 1e12 * 2e-16 ~ 2e-4. A tight per-element
    # tolerance here would be either vacuous or flaky depending on where each
    # backend's rounding lands in the near-null direction; a loose
    # relative-norm bound is the honest comparison for this regime.
    assert relative_gap < 5.0e-2, f"direction diverged from the oracle: {relative_gap}"

    # --- (b) refusal side: strictly positive shift BELOW the tolerance ---
    duplicate_rows, dup_energies = _problem_with_duplicate_column(
        n_samples=10, n_params=4, seed=213
    )
    dup_layout = _layout(((4,),))
    dup_geometry = build_score_geometry_from_rows(
        duplicate_rows, layout=dup_layout, conventions=ScoreConventions()
    )
    dup_operator = QGTOperator(dup_geometry)
    dup_measured_eigenvalues = torch.linalg.eigvalsh(
        dup_operator.dense_parameter_qgt()
    ).clamp_min(0.0)
    dup_max_eig = float(dup_measured_eigenvalues[-1])
    dup_min_eig = float(dup_measured_eigenvalues[0])
    dup_tolerance = 4 * eps64 * dup_max_eig
    assert dup_min_eig < dup_tolerance, (
        "the duplicate-column construction is not singular enough to probe the "
        "boundary: the measured smallest eigenvalue already exceeds the tolerance"
    )
    # Strictly between the measured smallest eigenvalue and the tolerance:
    # guaranteed min_retained + shift < tolerance regardless of the exact
    # rounding-noise value of the near-zero eigenvalue.
    below_tolerance_shift = 0.5 * (dup_min_eig + dup_tolerance)
    assert dup_min_eig < below_tolerance_shift < dup_tolerance

    dup_residual = build_energy_residual(dup_energies, geometry=dup_geometry)
    with pytest.raises(ValueError, match="singular"):
        solve_parameter_space(
            dup_operator,
            dup_residual,
            damping=DampingPolicy(absolute=below_tolerance_shift, relative=0.0, minimum=0.0),
        )


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------


def test_t3_dead_score_columns_solve_to_an_exact_zero_direction() -> None:
    """T3: probes Lane N's own claim about structurally inactive parameters.

    Lane N's receipt (see `test_score_seam_blocks_sr_on_a_two_electron_tpen_model`
    in `test_sr_trainer_integration.py`, which pins the current blocker) asserts
    that once the score seam is fixed, structurally inactive parameters --
    the g0..g14-type dead coordinates at 2 electrons -- will emit exactly-zero
    score columns, and that this needs NO change on the SR side. This builds a
    synthetic engine-level problem with a contiguous block of all-zero score
    columns mixed with live ones, under positive damping, and checks:

    * both solve routes succeed;
    * the returned direction is exactly zero (<= 1e-14) on the dead
      coordinates;
    * the live-coordinate sub-direction matches the NumPy oracle computed on
      the REDUCED (live-columns-only) problem;
    * the two routes agree with each other.

    Damping uses `relative=0.0` deliberately: the shift would otherwise be
    anchored to `trace(S) / P` with P counting the dead coordinates, which
    would make the full problem's shift disagree with the reduced problem's
    shift purely from a different P, confounding the comparison with an
    artifact of the damping formula rather than the claim under test.

    Expected to PASS. If it fails, that is a first-class finding about the
    reviewed claim -- report it; do not fix production code.
    """

    n_samples = 12
    n_live = 4
    n_dead = 3
    n_params = n_live + n_dead
    generator = torch.Generator().manual_seed(71)

    live_rows = torch.randn(n_samples, n_live, generator=generator, dtype=torch.float64)
    dead_rows = torch.zeros(n_samples, n_dead, dtype=torch.float64)
    rows = torch.cat([live_rows, dead_rows], dim=1)
    energies = torch.randn(n_samples, generator=generator, dtype=torch.float64)

    layout = _layout(((n_params,),))
    geometry = build_score_geometry_from_rows(
        rows, layout=layout, conventions=ScoreConventions()
    )
    residual = build_energy_residual(energies, geometry=geometry)
    operator = QGTOperator(geometry)
    damping = DampingPolicy(absolute=1.0e-6, relative=0.0, minimum=0.0)

    param_direction, param_diagnostics = solve_parameter_space(
        operator, residual, damping=damping
    )
    sample_direction, sample_diagnostics = solve_sample_space(
        operator, residual, damping=damping
    )

    assert param_diagnostics.space == "parameter"
    assert sample_diagnostics.space == "sample"

    dead_slice = slice(n_live, n_params)
    for direction, label in (
        (param_direction, "parameter-space"),
        (sample_direction, "sample-space"),
    ):
        dead_component = direction[dead_slice]
        assert torch.all(dead_component.abs() <= 1.0e-14), (
            f"{label} route left a nonzero component on a dead coordinate: "
            f"{dead_component.tolist()}"
        )

    live_oracle = sr_direction(
        live_rows.numpy(), energies.numpy(), absolute=1.0e-6, relative=0.0, minimum=0.0
    )
    np.testing.assert_allclose(
        param_direction[:n_live].numpy(),
        live_oracle.direction,
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        sample_direction[:n_live].numpy(),
        live_oracle.direction,
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        param_direction.numpy(),
        sample_direction.numpy(),
        rtol=SOLVE_TOLERANCE,
        atol=SOLVE_TOLERANCE,
    )


# ---------------------------------------------------------------------------
# T4
# ---------------------------------------------------------------------------


def test_t4_trainer_state_with_a_stateful_sr_method_is_json_serializable() -> None:
    """T4: a checkpoint state carrying a stateful SR method must be JSON-safe.

    Drives one real SR step through `VMCTrainer` exactly like
    `test_method_state_round_trips_through_the_trainer_checkpoint_state` in
    `test_sr_trainer_integration.py`, then asserts `json.dumps(trainer.state_dict())`
    succeeds and round-trips through `json.loads` with the `update_method`
    payload intact.

    Rationale: the checkpoint regression Lane N self-caught (job 44572457,
    Adam optimizer tensors leaking into `trainer.json`) has no direct JSON pin
    today. A future warm-start tensor added to the SR method-state envelope
    would reintroduce exactly that regression -- `json.dumps` raises on a
    `torch.Tensor` -- and nothing in the existing suite would catch it before
    it reached a checkpoint on disk. This test makes that assumption
    load-bearing rather than implicit.

    Expected to PASS against current code.
    """

    trainer, _, _, _ = _fit(solve_space="parameter", max_steps=1)

    state = trainer.state_dict()
    assert "update_method" in state, "a stateful method must contribute checkpoint state"

    serialized = json.dumps(state)
    round_tripped = json.loads(serialized)

    assert round_tripped == state
    assert round_tripped["update_method"] == state["update_method"]


# ---------------------------------------------------------------------------
# T5 -- RED DEMONSTRATION, expected to FAIL against current code
# ---------------------------------------------------------------------------


def test_t5_resume_without_update_method_key_should_raise_red_demonstration() -> None:
    """T5: RED DEMONSTRATION of reviewer finding F4. EXPECTED TO FAIL.

    Do not xfail this test. Let it fail, and invoke it separately from T1-T4 in
    the Cannon run (`-k` deselecting/selecting on the literal substring `t5`)
    so its failure does not pollute their pass/fail accounting.

    Sets up: run 2 SR steps through a trainer, take `state = trainer.state_dict()`,
    delete the `'update_method'` key, build a FRESH trainer and a fresh SR
    method over the same model, `resolve_update_state`, then
    `load_state_dict(state-without-key)`.

    CURRENT behavior (verified by reading `VMCTrainer.load_state_dict`):
    `"update_method" in state` is False, so `load_method_state_dict` is never
    called -- the fresh method's own `completed_updates` counter silently
    stays at its initial value while `trainer.completed_updates` and
    `trainer.next_iteration` are restored from the (still-present) top-level
    counters in `state`. Method state and trainer counters silently diverge:
    resuming looks like a clean, successful restore even though the SR
    method's internal state was NOT restored at all.

    DESIRED behavior, asserted here: a method whose `method_state_dict()` is
    non-empty, when asked to restore from a state that is missing the
    `'update_method'` key, should raise rather than silently leaving its own
    state unrestored. This is the reviewer's finding, not yet implemented.
    """

    trainer, model, _, _ = _fit(solve_space="parameter", max_steps=2)

    state = trainer.state_dict()
    assert "update_method" in state, "sanity: a stateful method must contribute state"
    del state["update_method"]

    resumed_method = _sr_method(model)
    resumed = VMCTrainer(max_steps=2, update_method=resumed_method)
    resumed.resolve_update_state(model=model, optimizer=resumed_method.optimizer)

    with pytest.raises(ValueError, match="update_method"):
        resumed.load_state_dict(state)
# T6 -- round 2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_t6_reported_dtype_is_observed_not_an_echo_of_the_configuration(dtype) -> None:
    """T6: the reported solve dtype must be OBSERVED from the matrix, not an
    echo of the configured convention.

    Round-2 finding on the lane's own dtype test,
    `test_qgt.py::test_diagnostics_report_the_dtype_the_solve_actually_ran_in`.
    That test parametrizes over float64/float32, but in EVERY parametrization
    it sets `ScoreConventions(solve_dtype=dtype)` equal to the incoming dtype
    of the rows it builds. The configured convention and the matrix's actual
    dtype are therefore never allowed to diverge across that test's
    parametrizations -- they are aliases of one another throughout -- so an
    implementation that reports `dtype=str(operator.geometry.conventions.solve_dtype)`
    instead of the correct `dtype=str(matrix.dtype)` agrees with both
    parametrizations for the wrong reason and passes undetected.

    This test breaks the alias. `ScoreConventions()` here is built with
    `solve_dtype` LEFT AS `None`. Per `build_score_geometry_from_rows`, `None`
    means "keep the incoming score dtype": the geometry's design matrix ends
    up in `dtype` because the ROWS were constructed in `dtype`, while the
    CONFIGURED convention itself carries no dtype at all. A configured-echo
    implementation must therefore report the Python string `"None"` in both
    parametrizations -- disagreeing with `str(dtype)` and going red -- while
    an implementation that reads the dtype off the actually-factorized matrix
    still agrees, because nothing about the matrix itself changed.

    No numerical comparison is needed or performed here: this test
    discriminates entirely on the reported dtype STRING, not on any solved
    value, so no oracle is required.
    """

    n_samples, n_parameters = 8, 3
    generator = torch.Generator().manual_seed(21)
    rows = torch.randn((n_samples, n_parameters), generator=generator, dtype=dtype)
    energies = torch.randn(n_samples, generator=generator, dtype=dtype)

    layout = _layout(((n_parameters,),))
    geometry = build_score_geometry_from_rows(
        rows, layout=layout, conventions=ScoreConventions()
    )
    assert geometry.conventions.solve_dtype is None, (
        "sanity: solve_dtype must be left unset for this test to discriminate"
    )
    assert geometry.design.dtype == dtype, (
        "sanity: with solve_dtype=None the geometry must keep the incoming rows dtype"
    )

    operator = QGTOperator(geometry)
    residual = build_energy_residual(energies, geometry=geometry)
    policy = DampingPolicy(absolute=0.0, relative=1.0e-2)

    _, parameter_diagnostics = solve_parameter_space(operator, residual, damping=policy)
    _, sample_diagnostics = solve_sample_space(operator, residual, damping=policy)

    assert parameter_diagnostics.dtype == str(dtype)
    assert sample_diagnostics.dtype == str(dtype)
    assert parameter_diagnostics.as_metrics()["qgt_dtype"] == str(dtype)
    assert sample_diagnostics.as_metrics()["qgt_dtype"] == str(dtype)
