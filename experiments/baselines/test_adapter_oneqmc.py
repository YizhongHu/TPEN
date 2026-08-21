"""Tests for the OneQMC (Orbformer) run-directory adapter.

Almost every test goes through :func:`record_from_series` or one of the pure
helpers, so it needs no HDF5 file and no ``h5py``. That split matters: ``h5py``
is not a TPEN dependency, and a test skipped for a missing dependency protects
nothing. Only the round-trip test is gated on ``h5py``, and that one must be run
in the OneQMC virtualenv on the cluster.

The failure this file guards hardest is the slot-versus-molecule confusion.
``metrics/E_loc/mean_elec`` column ``j`` is a mol-batch *slot*, not molecule
``j``, so an adapter that indexes by position produces a number that looks
entirely reasonable and belongs to the wrong system. Several tests below are
built so that a positional read returns a plausible energy rather than an error.
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path

import pytest

from experiments.baselines.errors import AdapterError
from experiments.baselines.statistics import (
    MIN_BLOCKS,
    MIN_TAIL_STEPS,
    blocking_inflation,
    blocking_stderr,
    select_tail,
)
from experiments.baselines.adapters.oneqmc import (
    DEFAULT_TAIL_FRACTION,
    ENERGY_DATASET,
    HUBER_DELTA_HARTREE,
    MOL_INDEX_DATASET,
    NATIVE_ANSATZES,
    SPREAD_DATASET,
    build_record,
    gather_molecule,
    huber_mean,
    main,
    metadata_from_attrs,
    read_attrs,
    read_series,
    record_from_series,
    result_path,
    write_record,
)

#: Exact non-relativistic helium ground state, for orientation only. No test
#: asserts agreement with it: this adapter reports what a run produced, and a
#: test that demanded physical correctness would fail on a deliberately bad run.
HE_EXACT_HARTREE = -2.903724

# Below MIN_TAIL_STEPS, so tests must pass allow_short_tail. That is faithful to
# the lane: an Orbformer evaluation pass is a few thousand steps, not 10000.
SHORT_KWARGS = dict(allow_short_tail=True)


def _series(count: int = 200, mean: float = HE_EXACT_HARTREE, seed: int = 5) -> list[float]:
    """A plateaued energy series: noise about a fixed mean, no drift."""

    rng = random.Random(seed)
    return [mean + rng.gauss(0.0, 1e-5) for _ in range(count)]


def _record(energies, spreads=None, **overrides):
    """Build a record with lane-typical arguments, overridable per test."""

    energies = list(energies)
    kwargs = dict(
        system_id="he_atom",
        electron_batch_size=1024,
        ansatz="orbformer-se",
        estimator="inference",
        training_provenance="from-scratch",
        run_id="test-run",
        allow_short_tail=True,
    )
    kwargs.update(overrides)
    return record_from_series(
        energies,
        [0.05] * len(energies) if spreads is None else list(spreads),
        **kwargs,
    )


def _huber_loss(mu: float, values, delta: float = HUBER_DELTA_HARTREE) -> float:
    """Vendor Huber objective, transcribed from ``oneqmc.analysis.energy``."""

    total = 0.0
    for value in values:
        residual = abs(value - mu)
        total += (
            0.5 * residual**2 if residual <= delta else delta * (residual - delta / 2)
        )
    return total


# --------------------------------------------------------------------------
# slot versus molecule -- the defect this adapter exists to avoid
# --------------------------------------------------------------------------


def test_gather_follows_mol_idx_not_column_position() -> None:
    """Molecule identity comes from ``mol_idx``, never from the column index.

    Here molecule 0 sits in column 1 and molecule 1 in column 0, which is what a
    shuffled mol batch looks like. A positional read returns the *other*
    molecule's energy: a finite, plausible number attached to the wrong system.
    Nothing downstream can detect that, which is why it is tested first.
    """

    mol_idx = [[1, 0], [1, 0], [1, 0]]
    energies = [[-7.5, -2.9], [-7.6, -2.8], [-7.4, -2.9]]
    spreads = [[0.4, 0.05], [0.4, 0.05], [0.4, 0.05]]

    series = gather_molecule(mol_idx, energies, spreads, 0)

    assert series.energies == (-2.9, -2.8, -2.9)
    assert series.spreads == (0.05, 0.05, 0.05)
    assert series.mol_idx == 0


def test_gather_skips_absent_steps_without_forward_filling() -> None:
    """A step the molecule missed is dropped, not filled with the last value.

    The vendor's reader forward-fills, which is right for a plot and wrong for an
    estimator: a repeated value adds a sample without adding information, so
    every variance estimate downstream narrows. Molecule 0 appears on three of
    six steps here, so a forward-filling implementation returns six values.
    """

    mol_idx = [[0], [1], [0], [1], [0], [1]]
    energies = [[-2.90], [-7.5], [-2.91], [-7.5], [-2.92], [-7.5]]
    spreads = [[0.05], [0.4], [0.05], [0.4], [0.05], [0.4]]

    series = gather_molecule(mol_idx, energies, spreads, 0)

    assert series.energies == (-2.90, -2.91, -2.92)
    assert series.logged_steps == 6, "coverage denominator is the file's step count"


def test_gather_rejects_a_molecule_in_two_slots() -> None:
    """Two slots holding one molecule raises rather than picking a weighting.

    Each slot is a separate electron-batch mean whose weight depends on how the
    sampler assigned walkers. Averaging them unweighted is a guess, and a silent
    guess is exactly what this adapter refuses to make.
    """

    with pytest.raises(AdapterError, match="2 slots"):
        gather_molecule([[0, 0]], [[-2.9, -2.8]], [[0.05, 0.05]], 0)


def test_gather_rejects_an_absent_molecule() -> None:
    """A molecule that never appears raises instead of returning nothing.

    An empty series would otherwise reach the estimator as a very short run, and
    the error would surface as a window complaint rather than as the real
    problem: ``--mol-idx`` naming a molecule this run never sampled.
    """

    with pytest.raises(AdapterError, match="never appears"):
        gather_molecule([[0], [1]], [[-2.9], [-7.5]], [[0.05], [0.4]], 7)


def test_gather_drops_non_finite_steps_and_counts_them() -> None:
    """NaN and infinite steps are dropped, and the count is reported.

    A diverged step must not enter the mean, but the drop must also be visible:
    a run producing many of them is not a healthy run, and silence would let it
    pass as clean.
    """

    mol_idx = [[0], [0], [0], [0]]
    energies = [[-2.90], [float("nan")], [-2.91], [float("inf")]]
    spreads = [[0.05], [0.05], [0.05], [0.05]]

    series = gather_molecule(mol_idx, energies, spreads, 0)

    assert series.energies == (-2.90, -2.91)
    assert series.nonfinite_dropped == 2


def test_gather_drops_a_step_whose_spread_is_non_finite() -> None:
    """A finite energy with a NaN spread is dropped too.

    The local-energy variance is built from ``std_elec``; letting a NaN through
    there poisons the variance column while the energy column looks fine.
    """

    series = gather_molecule(
        [[0], [0]], [[-2.90], [-2.91]], [[0.05], [float("nan")]], 0
    )

    assert series.energies == (-2.90,)
    assert series.nonfinite_dropped == 1


def test_gather_rejects_an_all_non_finite_molecule() -> None:
    """Every step non-finite raises, rather than yielding an empty series."""

    with pytest.raises(AdapterError, match="non-finite"):
        gather_molecule([[0], [0]], [[float("nan")], [float("nan")]], [[0.1], [0.1]], 0)


def test_gather_rejects_mismatched_row_counts() -> None:
    """Datasets of different lengths raise; they cannot be zipped meaningfully."""

    with pytest.raises(AdapterError, match="shape mismatch"):
        gather_molecule([[0], [0]], [[-2.9]], [[0.05], [0.05]], 0)


def test_gather_rejects_mismatched_slot_counts() -> None:
    """A step whose datasets disagree on slot count raises.

    Truncating to the shorter row would silently realign columns, which is the
    same class of error as reading by position.
    """

    with pytest.raises(AdapterError, match="slots"):
        gather_molecule([[0, 1]], [[-2.9]], [[0.05, 0.4]], 0)


# --------------------------------------------------------------------------
# the Huber estimator
# --------------------------------------------------------------------------


def test_huber_mean_matches_a_brute_force_minimizer() -> None:
    """The bisection solution minimizes the vendor's objective.

    Checked against the transcribed loss on a grid rather than against the
    arithmetic mean, so the test still discriminates on a series where the two
    differ.
    """

    values = [-2.9, -2.8, -2.95, 4.0, -2.85]
    estimate, _ = huber_mean(values)

    # Compared by objective value rather than by argument, so the assertion does
    # not depend on the grid resolution: the exact minimizer here is -2.625,
    # which a 20001-point grid over a 6.95-wide bracket cannot land on.
    grid = [min(values) + i * (max(values) - min(values)) / 20000 for i in range(20001)]
    assert _huber_loss(estimate, values) <= min(_huber_loss(mu, values) for mu in grid)
    assert estimate == pytest.approx(-2.625, abs=1e-9)


def test_huber_mean_equals_arithmetic_mean_when_nothing_clips() -> None:
    """Inside delta the objective is quadratic, so the estimate is the mean.

    This is why the clipped count is reported: without it a reader cannot tell a
    genuinely robust estimate from a plain average wearing the word "robust".
    """

    values = _series(200)
    estimate, clipped = huber_mean(values)

    assert clipped == 0
    assert estimate == pytest.approx(sum(values) / len(values), abs=1e-12)


def test_huber_mean_resists_an_outlier_the_arithmetic_mean_cannot() -> None:
    """One gross outlier moves the Huber estimate far less than the mean."""

    values = _series(200) + [50.0]
    estimate, clipped = huber_mean(values)

    assert clipped == 1
    assert abs(estimate - HE_EXACT_HARTREE) < 1e-2
    assert abs(sum(values) / len(values) - HE_EXACT_HARTREE) > 0.2


def test_huber_mean_handles_a_constant_series() -> None:
    """A zero-spread series returns its own value and clips nothing."""

    assert huber_mean([-2.9, -2.9, -2.9]) == (-2.9, 0)


def test_huber_mean_rejects_empty_and_non_positive_delta() -> None:
    """Degenerate arguments raise rather than returning a meaningless number."""

    with pytest.raises(AdapterError, match="at least one value"):
        huber_mean([])
    with pytest.raises(AdapterError, match="positive"):
        huber_mean([-2.9, -2.8], delta=0.0)


def test_record_energy_is_the_huber_estimate_of_the_window() -> None:
    """The record's energy is the window's Huber estimate, not the full series.

    The first half here is deliberately far off, so an implementation that
    averaged everything would land visibly away from the plateau.
    """

    energies = [0.0] * 100 + _series(100)
    # The floor comes down with the fraction so the window is exactly the
    # requested quarter. Leaving the 10000-step default in place on a 200-step
    # run makes the window whatever `select_tail` resolves a sub-floor run to,
    # which is not what this test is about.
    record = _record(energies, tail_fraction=0.25, min_tail_steps=2)

    expected, _ = huber_mean(energies[-50:])
    assert record.energy_hartree == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------
# provenance -- the claim the record makes about how it was produced
# --------------------------------------------------------------------------


def test_code_is_oneqmc_and_ansatz_carries_the_model() -> None:
    """``code`` names the codebase we ran; ``ansatz`` names the model.

    Writing ``code="orbformer"`` would name a codebase that does not exist:
    Orbformer is a model, OneQMC is the repository, and the comparison table is
    grouped by code.
    """

    record = _record(_series(100))

    assert record.code == "oneqmc"
    assert record.ansatz == "orbformer-se"


@pytest.mark.parametrize("ansatz", sorted(NATIVE_ANSATZES))
def test_native_ansatz_is_labelled_native(ansatz: str) -> None:
    """An ``orbformer-*`` row is OneQMC's own work, and says so."""

    record = _record(_series(100), ansatz=ansatz)

    assert "native" in record.notes
    assert "REIMPLEMENTATION" not in record.notes


@pytest.mark.parametrize("ansatz", ["psiformer", "psiformer-new", "envnet"])
def test_foreign_ansatz_is_labelled_a_reimplementation(ansatz: str) -> None:
    """OneQMC's Psiformer is not the Psiformer authors' code, and must say so.

    Without this, a OneQMC ``psiformer`` row reads as a Psiformer measurement
    and any deficit gets attributed to the wrong group's method.
    """

    record = _record(_series(100), ansatz=ansatz)

    assert "REIMPLEMENTATION" in record.notes


def test_finetuned_record_denies_being_a_reproduction() -> None:
    """A fine-tuned row states that it is not a from-scratch reproduction.

    A few thousand fine-tuning steps from a released checkpoint and the paper's
    ~11200 A100-hour pretraining are different claims. Nothing in ``result.h5``
    distinguishes them, so the record has to.
    """

    record = _record(
        _series(100),
        training_provenance="finetune-from-release",
        checkpoint_provenance="lac.chkpt sha256:7c140f15",
    )

    assert "NOT a from-scratch reproduction" in record.notes
    assert "lac.chkpt sha256:7c140f15" in record.notes


def test_finetuned_record_requires_checkpoint_provenance() -> None:
    """A fine-tuned row with no named checkpoint is refused.

    A checkpoint with no recorded provenance cannot be audited against the
    release it claims to start from, so the row is not usable as a baseline.
    """

    with pytest.raises(AdapterError, match="checkpoint_provenance"):
        _record(_series(100), training_provenance="finetune-from-release")


def test_from_scratch_record_says_from_scratch() -> None:
    """A from-scratch row states that too, rather than staying silent."""

    record = _record(_series(100), training_provenance="from-scratch")

    assert "trained from scratch" in record.notes
    assert "NOT a from-scratch reproduction" not in record.notes


def test_unknown_provenance_tier_is_refused() -> None:
    """Provenance is a closed vocabulary; a free-text tier raises."""

    with pytest.raises(AdapterError, match="training_provenance"):
        _record(_series(100), training_provenance="pretrained-ish")


# --------------------------------------------------------------------------
# the error bar, the variance, and the sample count
# --------------------------------------------------------------------------


def test_notes_disclaim_the_paper_error_bar() -> None:
    """The bar is a blocked standard error, and the notes refuse the other name.

    Orbformer's published bars are an across-chain spread (arXiv:2506.19960
    Appendix H). Per-walker local energies are never logged, so that quantity
    cannot be rebuilt from this file, and letting the two share a column would
    silently compare unlike uncertainties.
    """

    record = _record(_series(200))

    assert record.energy_stderr_hartree > 0.0
    assert "NOT the across-chain variance" in record.notes
    assert "Appendix H" in record.notes


def test_short_window_never_emits_a_zero_error_bar() -> None:
    """A sub-32-step window still gets a positive bar, marked as naive.

    ``blocking_stderr`` stops before its first blocking level when the window
    holds fewer than 32 values and returns 0.0. The record schema accepts a zero
    bar as non-negative, so it would be published as infinite precision on a
    20-step estimate. This is the reachable case, not a hypothetical: a
    2000-step evaluation at ``--metric-logger-period 25`` logs 80 rows.
    """

    record = _record(_series(20), tail_fraction=1.0)

    assert record.energy_stderr_hartree > 0.0
    assert "NAIVE standard error" in record.notes
    assert "UNDERSTATES" in record.notes


def test_long_window_bar_is_blocked_not_naive() -> None:
    """Above the block floor the bar is the blocked one, with no naive caveat."""

    record = _record(_series(200), tail_fraction=1.0)

    assert record.energy_stderr_hartree > 0.0
    assert "NAIVE standard error" not in record.notes


def test_local_energy_variance_is_the_mean_squared_spread() -> None:
    """``std_elec`` is an across-walker standard deviation, so square it.

    Reporting the spread itself, or the variance of the mean, would put a
    quantity off by a factor of the batch size into the column used to compare
    ansatz quality.
    """

    energies = _series(100)
    record = _record(energies, spreads=[0.2] * 50 + [0.4] * 50, tail_fraction=1.0)

    assert record.local_energy_variance_hartree2 == pytest.approx(
        (0.2**2 + 0.4**2) / 2, abs=1e-12
    )


def test_samples_counts_walkers_times_steps() -> None:
    """``samples`` is steps times electron batch size."""

    record = _record(_series(100), electron_batch_size=1024)

    assert record.steps == 100
    assert record.samples == 100 * 1024


def test_logger_period_scales_steps_but_not_the_window() -> None:
    """Logged steps times the logger period recovers real steps.

    Walkers are propagated on unlogged steps too, so a period-25 run of 100
    logged rows performed 2500 steps. Counting rows would understate the cost by
    the period and make the run look 25 times cheaper than it was.
    """

    record = _record(_series(100), metric_logger_period=25, electron_batch_size=1024)

    assert record.steps == 2500
    assert record.samples == 2500 * 1024
    assert "Logger period 25" in record.notes


def test_non_positive_batch_size_and_period_are_refused() -> None:
    """A zero or negative scale factor raises rather than zeroing ``samples``."""

    with pytest.raises(AdapterError, match="electron_batch_size"):
        _record(_series(100), electron_batch_size=0)
    with pytest.raises(AdapterError, match="metric_logger_period"):
        _record(_series(100), metric_logger_period=0)


def test_mismatched_energy_and_spread_lengths_are_refused() -> None:
    """Energies and spreads must be gathered together, or they are misaligned."""

    with pytest.raises(AdapterError, match="but"):
        record_from_series(
            [-2.9, -2.8],
            [0.05],
            system_id="he_atom",
            electron_batch_size=1024,
            ansatz="orbformer-se",
            estimator="inference",
            training_provenance="from-scratch",
            run_id="r",
            allow_short_tail=True,
        )


# --------------------------------------------------------------------------
# window selection and convergence reporting
# --------------------------------------------------------------------------


def test_short_run_is_refused_unless_explicitly_allowed() -> None:
    """A sub-floor run raises by default, and is marked provisional when allowed.

    The floor exists because a short window cannot average out the slow mode.
    Allowing it must therefore be a decision that shows up in the record, not a
    silent downgrade.
    """

    energies = _series(200)
    with pytest.raises(AdapterError, match="minimum"):
        _record(energies, allow_short_tail=False)

    record = _record(energies, allow_short_tail=True)
    assert "provisional" in record.notes


def test_whole_trace_window_mixes_in_relaxation_and_a_fraction_does_not() -> None:
    """A full-trace window averages the relaxation steps; a fraction excludes them.

    An evaluation pass launched with ``--discard-sampler-state`` spends its first
    steps relaxing, and those steps are logged like any other. Asking for the
    whole trace therefore folds relaxation into the published energy. That is why
    the lane emits with an explicit fraction and an explicit floor rather than
    letting the window fall out of how a sub-floor run happens to be resolved --
    a resolution that is ``select_tail``'s business and has changed there before.
    Both halves are pinned so the difference cannot be rediscovered on a
    published number.
    """

    energies = [0.0] * 150 + _series(50)

    whole = _record(energies, tail_fraction=1.0)
    assert whole.notes.count("last 200 of 200 logged steps") == 1
    assert whole.energy_hartree == pytest.approx(huber_mean(energies)[0])

    genuine = _record(energies, tail_fraction=0.25, min_tail_steps=2)
    assert "last 50 of 200 logged steps" in genuine.notes
    assert genuine.energy_hartree == pytest.approx(huber_mean(energies[-50:])[0])


#: Shape of this lane's Orbformer pilot: 2000 logged evaluation steps, the
#: default fraction, and the standard floor it cannot reach. Named constants
#: because the tripwire below reads as arbitrary numbers otherwise.
PILOT_LOGGED_STEPS = 2000
PILOT_WINDOW_AT_PIN = 2000
# The window PR #291's sub-floor fallback returns for this shape, measured by
# the statistics lane at its tip. Named rather than folded into the message so
# the tripwire says which value it discriminates against. On 2026-08-21 the
# program ruled that this fallback -- the requested fraction, not the whole
# trace -- is the correct one, and left #291 unchanged; see DECISION_ITEM_ID.
# The pin below is nevertheless still 2000, because this branch descends from
# dev, where select_tail clips the floor to the run length. Flipping the pin
# before #291 merges would make the test red today for a reason a reader would
# misread as a regression, so the flip belongs in the commit that rebases this
# branch onto a dev containing #291.
PILOT_WINDOW_AFTER_291 = 500
DECISION_ITEM_ID = "573509bb-58ef-45f7-a34d-3f5b110597e0"
DECISION_DATE = "2026-08-21"

#: Exact objects the pilot expectation was measured against, so a future reader
#: can tell a stale expectation from a regression. The blob is the more precise
#: of the two: it is the file whose rule produced the number.
STATISTICS_BLOB_AT_PIN = "fb0cec1ae1afc4795a1ba7a18c84b9481f0a226d"
DEV_COMMIT_AT_PIN = "e139a10f33c8866460264db0323887e4a38dbf26"

#: The window sentence the adapter writes, e.g. "the last 50 of 200 logged
#: steps". Anchored on "last " because the coverage sentence also ends in
#: "logged steps" and would otherwise match.
_WINDOW_SENTENCE = re.compile(r"last (\d+) of (\d+) logged steps")


def _window_from_notes(notes: str) -> tuple[int, int]:
    """Return the (window, total) the notes report, requiring exactly one match.

    Raises rather than returning a best guess when the count is not one: an
    ambiguous parse here would silently compare the wrong number, and this
    check is meant to fail loudly instead.
    """

    matches = _WINDOW_SENTENCE.findall(notes)
    assert len(matches) == 1, f"expected one window sentence, found {matches}: {notes}"
    window, total = matches[0]
    return int(window), int(total)


@pytest.mark.parametrize(
    "total, fraction, min_tail_steps",
    [
        # Floor reachable, fraction governs.
        (200, 0.25, 2),
        # Whole trace requested explicitly.
        (200, 1.0, 2),
        # Sub-floor run: how select_tail resolves this is its business and has
        # changed there, so nothing here pins the value -- only the invariants.
        (200, 0.25, MIN_TAIL_STEPS),
        (PILOT_LOGGED_STEPS, DEFAULT_TAIL_FRACTION, MIN_TAIL_STEPS),
    ],
)
def test_notes_window_is_the_window_actually_averaged(
    total: int, fraction: float, min_tail_steps: int
) -> None:
    """The reported window is the averaged window, whatever select_tail resolves to.

    These are the properties this adapter owns, stated without pinning
    ``select_tail``'s resolution rule: the window lies inside the run, the
    sentence in the notes names the window the energy was actually computed
    from, and the provisional caveat appears exactly when the window is below
    the floor the caller asked for. Written this way so a change in
    ``select_tail`` moves the numbers without falsifying the assertions -- the
    previous version of these tests pinned nothing at all here, which is why a
    help string could go stale unnoticed.

    The energy comparison is what makes the parsed window trustworthy: a notes
    sentence that disagreed with the slice would pass a substring check and fail
    this one.
    """

    # A relaxation prefix half a hartree above the plateau, so a window that
    # mis-reports its own length disagrees with the energy by far more than
    # float noise. On a flat series the difference between a quarter-tail mean
    # and a whole-trace mean is ~1e-6 relative, which pytest.approx's default
    # tolerance accepts -- that looseness let a mutant that reported the whole
    # trace while averaging the tail pass this test.
    plateau = total - max(2, round(0.25 * total))
    energies = [HE_EXACT_HARTREE + 0.5] * plateau + _series(total - plateau)
    record = _record(
        energies,
        tail_fraction=fraction,
        min_tail_steps=min_tail_steps,
        allow_short_tail=True,
    )

    window, reported_total = _window_from_notes(record.notes)

    assert reported_total == total
    assert 2 <= window <= total
    # Exact rather than approximate: the same slice through the same estimator
    # is the same float.
    assert record.energy_hartree == pytest.approx(
        huber_mean(energies[-window:])[0], rel=0.0, abs=1e-12
    )
    assert ("provisional" in record.notes) == (window < min_tail_steps)


@pytest.mark.parametrize(
    "total, min_tail_steps",
    # The last case is the short-window regime, where the block floor is
    # lowered and any "no blocks measured" sentinel would come from.
    [(200, MIN_TAIL_STEPS), (PILOT_LOGGED_STEPS, MIN_TAIL_STEPS), (200, 2), (20, 2)],
)
def test_notes_never_render_a_literal_none(total: int, min_tail_steps: int) -> None:
    """No notes field may contain the literal string "None".

    A bare replacement field holding ``None`` -- ``f"from {count} blocks"`` --
    renders "from None blocks" without raising, which reads as a forgotten
    count rather than as "blocking never ran". That silent branch is the shared
    failure mode across the adapters in this program, so it is asserted at the
    output rather than trusted to the call sites.
    """

    record = _record(
        _series(total),
        min_tail_steps=min_tail_steps,
        allow_short_tail=True,
    )

    assert "None" not in record.notes


def test_lowering_the_block_floor_is_what_keeps_the_ladder_running() -> None:
    """``min(MIN_BLOCKS, len(tail))`` is load-bearing; do not simplify it away.

    A short Orbformer window is shorter than ``MIN_BLOCKS``, so with the default
    floor blocking's loop condition is false before its first level and the
    function never measures anything -- it either returns a degenerate bar or
    refuses, depending on the ``statistics`` version. Lowering the floor to the
    window length makes the first level run, which is what gives this adapter a
    real number and what makes any "no blocks measured" sentinel unreachable
    from here.

    Both halves are asserted, because the second is the reason for the first.
    """

    tail = _series(20)
    assert len(tail) < MIN_BLOCKS

    lowered = min(MIN_BLOCKS, len(tail))
    stderr, blocks = blocking_stderr(tail, min_blocks=lowered)
    assert stderr > 0.0
    assert blocks == len(tail)
    assert blocking_inflation(tail, min_blocks=lowered) >= 1.0

    # The unlowered floor, for contrast. Which way it fails is the shared
    # module's business; that it yields nothing usable is the point.
    try:
        degenerate, _ = blocking_stderr(tail, min_blocks=MIN_BLOCKS)
    except AdapterError:
        pass
    else:
        assert degenerate <= 0.0, (
            "blocking measured something at the unlowered floor, so this test no "
            "longer demonstrates why the adapter lowers it"
        )

    # And the adapter's own output on such a window: a numeric inflation ratio,
    # never a sentinel.
    # tail_fraction=1.0 so the record's window is the same 20 steps the
    # blocking assertions above were made on.
    record = _record(tail, min_tail_steps=2, tail_fraction=1.0)
    assert _window_from_notes(record.notes) == (len(tail), len(tail))
    assert re.search(r"inflation \d+\.\d\dx", record.notes)
    assert "None" not in record.notes


def test_pilot_window_is_still_the_whole_trace() -> None:
    """Deliberate merge-order tripwire on ``select_tail``'s sub-floor resolution.

    This is the only test here that pins a resolved window, and it is pinned on
    purpose: the ``--allow-short-tail`` help used to describe the resolution
    rule, went stale when the rule changed, and nothing caught it because
    nothing pinned the value. The failure message below carries the whole
    diagnosis, so a future reader does not have to reconstruct it.
    """

    window = select_tail(
        PILOT_LOGGED_STEPS,
        DEFAULT_TAIL_FRACTION,
        min_steps=MIN_TAIL_STEPS,
        allow_below_floor=True,
    )

    assert window == PILOT_WINDOW_AT_PIN, (
        f"select_tail resolved the pilot shape (total_steps={PILOT_LOGGED_STEPS}, "
        f"fraction={DEFAULT_TAIL_FRACTION}, min_steps={MIN_TAIL_STEPS}, "
        f"allow_below_floor=True) to {window}, not {PILOT_WINDOW_AT_PIN}. "
        "This is a deliberate merge-order alarm, not a flaky test, and as of "
        f"{DECISION_DATE} the expected response to it is to CHANGE THIS LINE. "
        "The expectation was pinned against experiments/baselines/statistics.py "
        f"at blob {STATISTICS_BLOB_AT_PIN} (origin/dev commit "
        f"{DEV_COMMIT_AT_PIN}), where the floor is clipped to the run length so "
        "a sub-floor run keeps its whole trace. PR #291 "
        "(claude/statistics-short-window, tip "
        "4e10ad9c5055976dc31b45c8b01a565028d2c143) returns the requested "
        f"fraction instead, which is {PILOT_WINDOW_AFTER_291} for this shape. "
        "This lane argued for the whole trace, on the grounds that it maximises "
        "information in the one regime where the run is definitionally short; "
        f"the program ruled against that on {DECISION_DATE} and left #291 as "
        "written, in part because select_tail takes no estimator argument, so a "
        "single hard-coded fallback is necessarily wrong for one of its two "
        f"callers. A window of {PILOT_WINDOW_AFTER_291} here therefore does NOT "
        "mean statistics.py is broken: it means #291 reached dev and this "
        "branch was rebased onto it. Correct response: flip "
        f"PILOT_WINDOW_AT_PIN to {PILOT_WINDOW_AFTER_291} in the rebase commit "
        "itself and name the flip in that commit message, so it reads as the "
        "sanctioned consequence of a merge rather than as drift. The decision "
        f"and its reasoning are on Task Orchestrator item {DECISION_ITEM_ID}. "
        "Any OTHER value is a real regression in select_tail and must be fixed "
        "there, not here."
    )


def test_short_tail_help_does_not_state_a_resolution_rule(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The flag's help must describe the decision, not select_tail's arithmetic.

    It previously promised that "the floor is clipped to the run length, so on
    a short run this widens the window to the whole trace", which is a claim
    about ``select_tail`` rather than about this flag, and would become false
    the moment that rule changed. Help text is the one part of this adapter an
    operator reads instead of the code, so a false sentence there is acted on.
    """

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0

    # argparse re-wraps help to the terminal width, so compare on collapsed
    # whitespace rather than on the emitted line breaks.
    help_text = " ".join(capsys.readouterr().out.split())

    assert "--allow-short-tail" in help_text
    # Asserted before the positive checks so that a reverted help string fails
    # on the false claim itself rather than on a missing word.
    for stale in (
        "clipped to the run length",
        "widens the window to the whole trace",
    ):
        assert stale not in help_text, f"help still asserts {stale!r}: {help_text}"
    assert "provisional" in help_text
    assert "select_tail" in help_text


def test_drifting_series_is_flagged_monotone() -> None:
    """A still-descending run is reported as possibly unconverged.

    The verdict describes; it never changes the number. A rule that altered the
    estimate would make the emitted energy depend on a noisy test.
    """

    drifting = [HE_EXACT_HARTREE + 1e-3 * (200 - i) for i in range(200)]
    record = _record(drifting)

    assert "MONOTONE" in record.notes
    assert "may not have converged" in record.notes


def test_plateaued_series_is_not_flagged() -> None:
    """A converged run is not accused of drifting."""

    record = _record(_series(200))

    assert "not monotone" in record.notes
    assert "MONOTONE" not in record.notes


def test_window_too_short_for_the_sign_test_says_unassessed() -> None:
    """Fewer steps than windows reports UNASSESSED, not silent success.

    A missing check must read as missing. Printing nothing would let a
    four-step estimate look as vetted as a converged one.
    """

    record = _record(_series(4))

    assert "UNASSESSED" in record.notes


def test_zero_variance_window_is_refused_rather_than_given_a_zero_bar() -> None:
    """A constant window emits no record at all.

    Every step carrying the identical energy means the sampler stopped moving or
    the series was forward-filled upstream. Such a window has no inflation ratio
    and no standard error, and the record schema only requires the bar to be
    non-negative -- so a zero would be published as infinite precision on a
    number that was never measured. This is the same stance the reader takes on a
    forward-filled trace, applied to the estimator.
    """

    with pytest.raises(AdapterError, match="zero-variance window"):
        _record([HE_EXACT_HARTREE] * 200)


def test_estimator_text_distinguishes_inference_from_training_tail() -> None:
    """The two estimators are different quantities and are named differently."""

    inference = _record(_series(100), estimator="inference")
    training = _record(_series(100), estimator="training_tail")

    assert inference.estimator == "inference"
    assert "Fixed-parameter inference pass" in inference.notes
    assert training.estimator == "training_tail"
    assert "Training-tail average" in training.notes


def test_coverage_sentence_reports_the_gathered_fraction() -> None:
    """The notes say how many of the file's steps carried this molecule.

    Twelve logged steps of which three carried molecule 2 is a mol batch of
    four, and a reader has to be able to see that the estimate rests on three
    points rather than twelve.
    """

    record = _record(
        _series(3), logged_steps=12, mol_idx=2, nonfinite_dropped=1, tail_fraction=1.0
    )

    assert "molecule index 2" in record.notes
    assert "3 of 12" in record.notes
    assert "forward-filled" in record.notes
    assert "1 step(s) dropped as non-finite" in record.notes


# --------------------------------------------------------------------------
# operator notes and file attributes
# --------------------------------------------------------------------------


def test_operator_note_is_appended_never_substituted() -> None:
    """An operator caveat is added to the generated notes, not in place of them."""

    record = _record(_series(100), note="He is outside the LAC-supported atom set.")

    assert "He is outside the LAC-supported atom set." in record.notes
    assert "NOT the across-chain variance" in record.notes


def test_blank_operator_note_is_refused() -> None:
    """A whitespace-only note raises: a caveat that vanishes is worse than none."""

    with pytest.raises(AdapterError, match="non-empty"):
        _record(_series(100), note="   ")


def test_metadata_from_attrs_parses_bytes_and_spans() -> None:
    """Attrs may arrive as bytes; the span is stop minus start, in seconds."""

    metadata = metadata_from_attrs(
        {
            "start_time": b"2026-08-20 10:00:00.000000",
            "stop_time": "2026-08-20 11:30:00.000000",
            "num_gpus": 4,
            "gpu_type": b"NVIDIA A100-SXM4-40GB",
        }
    )

    assert metadata == {
        "device_type": "cuda",
        "gpu_model": "NVIDIA A100-SXM4-40GB",
        "n_gpus": 4,
        "wall_clock_seconds": 5400.0,
    }


def test_metadata_reports_a_cpu_run_as_cpu() -> None:
    """JAX's ``device_kind`` of ``cpu`` is a real CPU run, not missing data.

    Mapping it to ``cuda`` would put a CPU timing into a GPU-hours comparison.
    """

    metadata = metadata_from_attrs({"gpu_type": "cpu", "num_gpus": 1})

    assert metadata["device_type"] == "cpu"
    assert metadata["gpu_model"] is None


def test_missing_stop_time_yields_no_duration() -> None:
    """Without ``stop_time`` the duration is None, never a fabricated number.

    ``stop_time`` is rewritten on every logged scalar, so it is normally present
    even for a killed run; when it is absent something unusual happened and a
    guessed wall clock would enter a cost comparison as fact.
    """

    metadata = metadata_from_attrs({"start_time": "2026-08-20 10:00:00.000000"})

    assert metadata["wall_clock_seconds"] is None
    assert metadata["device_type"] is None


def test_unparseable_and_negative_spans_yield_no_duration() -> None:
    """A malformed or reversed stamp pair gives None rather than nonsense."""

    assert (
        metadata_from_attrs({"start_time": "yesterday", "stop_time": "today"})[
            "wall_clock_seconds"
        ]
        is None
    )
    assert (
        metadata_from_attrs(
            {
                "start_time": "2026-08-20 11:00:00.000000",
                "stop_time": "2026-08-20 10:00:00.000000",
            }
        )["wall_clock_seconds"]
        is None
    )


def test_wall_clock_note_states_it_is_a_lower_bound() -> None:
    """The recorded span is the logger's lifetime, and the notes say so.

    It starts after job startup and ends at the last logged scalar, which for a
    scheduler-killed run precedes the job's death. Calling it job wall clock
    would understate cost by an unknown amount.
    """

    record = _record(_series(100), wall_clock_seconds=5400.0)

    assert record.wall_clock_seconds == 5400.0
    assert "lower bound" in record.notes


def test_record_leaves_run_dir_for_the_collector() -> None:
    """``run_dir`` stays empty; the adapter cannot know the collector's root."""

    assert _record(_series(100)).run_dir is None


# --------------------------------------------------------------------------
# paths, output, and the HDF5 round trip
# --------------------------------------------------------------------------


def test_result_path_accepts_the_run_root_or_the_training_subdir(tmp_path: Path) -> None:
    """OneQMC nests output under ``training/``; both spellings resolve."""

    nested = tmp_path / "training"
    nested.mkdir()
    (nested / "result.h5").write_bytes(b"")
    assert result_path(tmp_path) == nested / "result.h5"

    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "result.h5").write_bytes(b"")
    assert result_path(flat) == flat / "result.h5"


def test_write_record_round_trips_through_json(tmp_path: Path) -> None:
    """The emitted file is the record, readable back without loss."""

    record = _record(_series(100))
    path = write_record(record, tmp_path)

    assert path.name == "baseline_record.json"
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["code"] == "oneqmc"
    assert reloaded["energy_hartree"] == pytest.approx(record.energy_hartree)


def test_missing_result_file_raises(tmp_path: Path) -> None:
    """A run directory with no ``result.h5`` raises a named error."""

    with pytest.raises(AdapterError, match="result.h5"):
        read_series(tmp_path)


class _BlockH5py:
    """Meta-path finder that makes ``import h5py`` fail and nothing else.

    A finder rather than a patched :func:`builtins.__import__`, because the
    latter intercepts every import in the process for the duration of the test,
    including pytest's own. A finder is consulted only for the module it claims.

    Returning ``None`` from ``find_spec`` is not enough on its own -- that just
    defers to the next finder -- so this one raises, which surfaces as the
    :class:`ModuleNotFoundError` the adapter catches. Assigning ``None`` into
    ``sys.modules`` would NOT work here: that raises plain
    :class:`ImportError`, which is the parent class, so the adapter's
    ``except ModuleNotFoundError`` would not catch it and the test would pass
    for the wrong reason.
    """

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname == "h5py" or fullname.startswith("h5py."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def _block_h5py(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import h5py`` raise for the rest of the test, installed or not.

    Both directions matter. On an interpreter without h5py the block is a no-op
    and the test still asserts the right thing; on one with h5py it is what
    makes the assertion meaningful at all. That is the point: the ordering
    property must hold in both closures, and it was the closure difference that
    hid the defect.
    """

    monkeypatch.delitem(sys.modules, "h5py", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockH5py(), *sys.meta_path])


@pytest.mark.parametrize("reader", [read_series, read_attrs])
def test_missing_file_is_diagnosed_before_the_missing_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: object
) -> None:
    """With h5py unimportable, an empty run directory still reports the file.

    Both errors are correct, so the only question is which one the caller sees
    first, and the ordering is not cosmetic. A caller who pointed at the wrong
    directory needs to be told the file is missing; telling them to install
    h5py sends them to fix an environment that was never the problem.

    This test previously did not exist, and its absence let the reverse ordering
    through. The two closures disagreed: the local interpreter carried h5py so
    ``test_missing_result_file_raises`` passed, while the project venv does not
    ship h5py, so the same test failed on the cluster with the dependency
    message instead of the filename. Blocking the import here pins the ordering
    in BOTH closures, which is the property that was actually wanted.
    """

    _block_h5py(monkeypatch)

    with pytest.raises(AdapterError, match="result.h5"):
        reader(tmp_path)  # type: ignore[operator]


@pytest.mark.parametrize("reader", [read_series, read_attrs])
def test_missing_dependency_is_reported_when_the_file_does_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: object
) -> None:
    """The h5py branch stays reachable: a present file plus no h5py says h5py.

    Guards the other half of the reordering. Moving the existence check first
    would be a regression if it made the dependency message unreachable, so this
    asserts the branch still fires on the input that should reach it. The file
    is created empty on purpose -- it is never opened, because the import fails
    before any read.
    """

    _block_h5py(monkeypatch)
    training = tmp_path / "training"
    training.mkdir()
    (training / "result.h5").write_bytes(b"")

    with pytest.raises(AdapterError, match="h5py"):
        reader(tmp_path)  # type: ignore[operator]


@pytest.mark.parametrize("mol_idx", [0, 1])
def test_build_record_reads_a_real_hdf5_file(tmp_path: Path, mol_idx: int) -> None:
    """End to end against a file shaped exactly like OneQMC's output.

    Both molecules are asserted from one fixture: the two-molecule mol batch is
    what makes the slot-versus-molecule bug reachable, and a fixture with a
    single molecule would pass under a positional read.
    """

    h5py = pytest.importorskip("h5py", reason="h5py is not a TPEN dependency")

    steps = 40
    energies, spreads, indices = [], [], []
    for step in range(steps):
        # Molecule 0 in the right-hand slot, molecule 1 in the left: a
        # positional read would swap helium for lithium.
        indices.append([1, 0])
        energies.append([-7.478 + 1e-4 * (step % 3), -2.9037 + 1e-5 * (step % 3)])
        spreads.append([0.4, 0.05])

    training = tmp_path / "training"
    training.mkdir()
    with h5py.File(training / "result.h5", "w", libver="v110") as handle:
        handle.attrs["start_time"] = "2026-08-20 10:00:00.000000"
        handle.attrs["stop_time"] = "2026-08-20 10:30:00.000000"
        handle.attrs["num_gpus"] = 1
        handle.attrs["gpu_type"] = "NVIDIA A100-SXM4-40GB"
        handle.create_dataset(MOL_INDEX_DATASET, data=indices)
        handle.create_dataset(ENERGY_DATASET, data=energies)
        handle.create_dataset(SPREAD_DATASET, data=spreads)

    record = build_record(
        tmp_path,
        system_id="he_atom" if mol_idx == 0 else "li_atom",
        electron_batch_size=1024,
        ansatz="orbformer-se",
        estimator="inference",
        training_provenance="finetune-from-release",
        checkpoint_provenance="lac.chkpt sha256:7c140f15",
        mol_idx=mol_idx,
        tail_fraction=1.0,
        allow_short_tail=True,
    )

    column = [row[0 if mol_idx == 1 else 1] for row in energies]
    assert record.energy_hartree == pytest.approx(huber_mean(column)[0], abs=1e-9)
    assert record.n_gpus == 1
    assert record.device_type == "cuda"
    assert record.wall_clock_seconds == 1800.0
    assert record.steps == steps


def test_hdf5_file_missing_a_dataset_raises(tmp_path: Path) -> None:
    """A file without ``mol_idx`` raises rather than falling back to position.

    A positional fallback is the whole failure mode; there must be no path to it,
    including "the map was missing".
    """

    h5py = pytest.importorskip("h5py", reason="h5py is not a TPEN dependency")

    with h5py.File(tmp_path / "result.h5", "w", libver="v110") as handle:
        handle.create_dataset(ENERGY_DATASET, data=[[-2.9], [-2.9]])
        handle.create_dataset(SPREAD_DATASET, data=[[0.05], [0.05]])

    with pytest.raises(AdapterError, match=MOL_INDEX_DATASET):
        read_series(tmp_path)


def test_defaults_are_the_documented_ones() -> None:
    """Guard the two constants a caller is most likely to rely on implicitly."""

    assert DEFAULT_TAIL_FRACTION == 0.25
    assert HUBER_DELTA_HARTREE == 1.0
    assert math.isfinite(HUBER_DELTA_HARTREE)
