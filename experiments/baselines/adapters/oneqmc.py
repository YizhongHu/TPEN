"""Translate a OneQMC (Orbformer) run directory into a :class:`BaselineRecord`.

OneQMC writes ``<workdir>/training/result.h5`` through
``oneqmc.log.H5MetricLogStream``. Neither of the existing adapters can read it,
so without this module every Orbformer measurement is stranded outside
``results.jsonl``.

Five properties of this module are deliberate and must survive any refactor.

**The column axis is a mol-batch slot, not a molecule.**
``metrics/E_loc/mean_elec`` has shape ``(logged_steps, mol_batch_size)`` and
``metrics/mol_idx`` -- same shape -- says which molecule occupied each slot at
each logged step. Taking column ``j`` as "molecule ``j``" is correct only when
``mol_batch_size == 1``, and silently mixes molecules otherwise. Every read here
gathers through ``mol_idx``; see :func:`gather_molecule`.

**Gaps are skipped, never forward-filled.** The vendor's own reader
(``oneqmc.analysis.h5_io.read_result``) scatters by ``mol_idx`` into a dense
``(steps, n_mol)`` array and then forward-fills the holes. That is right for
plotting a curve and wrong for an estimator: a repeated value is not a new
sample, so filling deflates the variance and invents steps that were never
sampled. This adapter drops the steps a molecule did not appear in and reports
the coverage in the record's notes.

**The energy estimator is a Huber M-estimate, not a plain mean.** The vendor
protocol (README, "Evaluation of the energy") is a fresh fixed-parameter
``--test`` run followed by ``oneqmc.analysis.energy.robust_mean``, which
minimizes a Huber loss with ``delta = 1`` hartree. :func:`huber_mean` computes
that same estimand without importing ``oneqmc`` or ``scipy``, by solving the
first-order condition exactly rather than by numerical minimization. On a
well-behaved series no residual reaches one hartree and the estimate *is* the
arithmetic mean; the notes say how many steps were clipped so a reader can tell
the two cases apart.

**The error bar is not the paper's.** Orbformer's published bars come from the
spread across independent chains (arXiv:2506.19960, Appendix H). That is not
reconstructible from this file: per-walker local energies are never written,
only the electron-batch mean ``E_loc/mean_elec`` and its across-walker standard
deviation ``E_loc/std_elec``. This adapter reports a blocked standard error over
the step series instead, and says so in the notes rather than letting the two
quantities share a column silently.

**Device and timing metadata are file attrs, not a job log.** The logger writes
``start_time``, ``num_gpus`` and ``gpu_type`` at construction and rewrites
``stop_time`` on *every* scalar log. So ``stop_time`` exists even for a run the
scheduler killed, and it means "time of the last logged scalar", not process
exit. :func:`metadata_from_attrs` keeps that semantics; the notes carry it.

Examples
--------
::

    uv run python -m experiments.baselines.adapters.oneqmc \\
        --run-dir path/to/orbformer-he-eval --system-id he_atom \\
        --ansatz orbformer-se --electron-batch-size 1024 \\
        --estimator inference --training-provenance finetune-from-release \\
        --checkpoint-provenance "lac.chkpt sha256:7c140f15..."
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.baselines.errors import AdapterError
from experiments.baselines.records import BaselineRecord
from experiments.baselines.statistics import (
    MIN_BLOCKS,
    MIN_TAIL_STEPS,
    SIGN_TEST_WINDOWS,
    blocking_inflation,
    blocking_stderr,
    select_tail,
    sign_test,
)

RESULT_FILENAME = "result.h5"
TRAINING_SUBDIR = "training"
RECORD_FILENAME = "baseline_record.json"

#: Electron-batch mean local energy, one row per logged step and one column per
#: mol-batch slot. The per-walker energies behind it are never written.
ENERGY_DATASET = "metrics/E_loc/mean_elec"

#: Across-walker standard deviation of the local energy, same shape. Squared and
#: averaged it gives the local-energy variance, the ansatz-quality signal.
SPREAD_DATASET = "metrics/E_loc/std_elec"

#: Slot-to-molecule map, same shape. Without it the columns are meaningless.
MOL_INDEX_DATASET = "metrics/mol_idx"

#: Huber loss width used by ``oneqmc.analysis.energy.robust_mean``, in hartree.
#: One hartree is enormous next to the step-to-step spread of a converged He
#: series, so on such a series the estimate coincides with the plain mean; the
#: value is kept because matching the vendor's estimand is the point.
HUBER_DELTA_HARTREE = 1.0

#: Trailing fraction averaged by default. Matches the DeepQMC adapter, and is
#: wanted here for an extra reason: the recommended evaluation run starts from
#: ``--discard-sampler-state``, so its early steps are still relaxing towards
#: the stationary distribution and are not part of the measurement.
DEFAULT_TAIL_FRACTION = 0.25

#: Ansatzes that are OneQMC's own work. OneQMC is Orbformer's codebase, so an
#: ``orbformer-*`` row is native. Everything else it ships -- ``psiformer``,
#: ``psiformer-new``, ``envnet`` -- is OneQMC's reimplementation of someone
#: else's method, and no claim about that method may rest on such a row. The set
#: is deliberately narrow: mislabelling a reimplementation as native overstates,
#: while the reverse only understates.
NATIVE_ANSATZES = frozenset({"orbformer-se", "orbformer-se-small"})

#: How the parameters being measured came to exist. Required on every record,
#: never inferred from a path or a step count: fine-tuning a released checkpoint
#: for a few thousand steps and reproducing a 400000-step pretraining run are
#: different scientific claims, and nothing in ``result.h5`` distinguishes them.
PROVENANCE_TIERS = ("from-scratch", "finetune-from-release")


@dataclasses.dataclass(frozen=True)
class MoleculeSeries:
    """One molecule's step series, gathered out of the slot-shaped datasets.

    Attributes
    ----------
    energies : tuple of float
        Electron-batch mean local energy, one entry per logged step in which the
        molecule was actually sampled.
    spreads : tuple of float
        Matching across-walker standard deviations.
    mol_idx : int
        Molecule index these entries belong to.
    logged_steps : int
        Rows in the file, i.e. how many logged steps the run wrote at all.
    nonfinite_dropped : int
        Steps discarded because the energy was NaN or infinite. Reported rather
        than hidden: the vendor's ``robust_mean`` filters these too, and a run
        that produced many of them is not a healthy run.
    """

    energies: tuple[float, ...]
    spreads: tuple[float, ...]
    mol_idx: int
    logged_steps: int
    nonfinite_dropped: int


def result_path(run_dir: Path) -> Path:
    """Return the HDF5 path for a run directory.

    OneQMC nests its output under ``training/``; accept either the run root or
    that subdirectory so a caller need not remember which.
    """

    direct = run_dir / RESULT_FILENAME
    if direct.is_file():
        return direct
    return run_dir / TRAINING_SUBDIR / RESULT_FILENAME


def gather_molecule(
    mol_indices: Sequence[Sequence[float]],
    energies: Sequence[Sequence[float]],
    spreads: Sequence[Sequence[float]],
    mol_idx: int,
) -> MoleculeSeries:
    """Gather one molecule's series out of slot-shaped rows.

    Pure function over nested sequences so that the slot-versus-molecule logic
    -- the part that is easy to get silently wrong -- is testable without
    ``h5py`` and without a run directory.

    Parameters
    ----------
    mol_indices : sequence of sequence of float
        ``metrics/mol_idx`` rows: which molecule sat in each slot at each step.
    energies, spreads : sequence of sequence of float
        ``metrics/E_loc/mean_elec`` and ``metrics/E_loc/std_elec`` rows.
    mol_idx : int
        Molecule to extract.

    Returns
    -------
    MoleculeSeries
        The gathered series, with coverage counts.

    Raises
    ------
    AdapterError
        If the three inputs disagree in shape; if ``mol_idx`` never appears (an
        empty series would otherwise look like a short run); if a single step
        carries the molecule in more than one slot; or if every step was
        dropped as non-finite.

    Notes
    -----
    A step in which the molecule is absent is skipped. It is *not* forward
    filled, unlike ``oneqmc.analysis.h5_io.read_result``: repeating the previous
    value adds no information but does add a sample, which narrows every
    variance estimate downstream.

    A step carrying the molecule in several slots raises rather than averaging.
    Those slots are separate batch means whose relative weight depends on how
    the sampler assigned walkers, and guessing a weighting is exactly the kind
    of silent choice this adapter exists to prevent.
    """

    if len(energies) != len(mol_indices) or len(spreads) != len(mol_indices):
        raise AdapterError(
            f"shape mismatch: {len(mol_indices)} mol_idx rows, {len(energies)} energy "
            f"rows, {len(spreads)} spread rows"
        )

    kept_energies: list[float] = []
    kept_spreads: list[float] = []
    seen = False
    nonfinite = 0

    for step, slots in enumerate(mol_indices):
        if len(energies[step]) != len(slots) or len(spreads[step]) != len(slots):
            raise AdapterError(
                f"step {step} has {len(slots)} mol_idx slots but "
                f"{len(energies[step])} energy and {len(spreads[step])} spread slots"
            )
        hits = [column for column, value in enumerate(slots) if int(value) == mol_idx]
        if not hits:
            continue
        seen = True
        if len(hits) > 1:
            raise AdapterError(
                f"step {step} carries molecule {mol_idx} in {len(hits)} slots; this "
                "adapter refuses to choose a weighting between them"
            )
        energy = float(energies[step][hits[0]])
        spread = float(spreads[step][hits[0]])
        if not _is_finite(energy) or not _is_finite(spread):
            nonfinite += 1
            continue
        kept_energies.append(energy)
        kept_spreads.append(spread)

    if not seen:
        raise AdapterError(
            f"molecule index {mol_idx} never appears in {MOL_INDEX_DATASET}; "
            "check --mol-idx against the dataset the run was launched on"
        )
    if not kept_energies:
        raise AdapterError(
            f"every one of the {nonfinite} steps carrying molecule {mol_idx} had a "
            "non-finite energy or spread"
        )

    return MoleculeSeries(
        energies=tuple(kept_energies),
        spreads=tuple(kept_spreads),
        mol_idx=mol_idx,
        logged_steps=len(mol_indices),
        nonfinite_dropped=nonfinite,
    )


def read_series(run_dir: Path, mol_idx: int = 0) -> MoleculeSeries:
    """Read one molecule's step series from a OneQMC run directory.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run root, or its ``training`` subdirectory.
    mol_idx : int, optional
        Molecule index to extract, matching ``metrics/mol_idx``.

    Returns
    -------
    MoleculeSeries
        Gathered series; see :func:`gather_molecule` for the gather rules.

    Raises
    ------
    AdapterError
        If ``h5py`` is unavailable, the file or any required dataset is missing,
        or the file holds no rows.

    Notes
    -----
    The file is opened with ``swmr=True, libver="v110"``, the same way the
    vendor's reader opens it and the same way the writer created it. That
    matters for a run the scheduler killed: the writer dies without closing, so
    the superblock stays flagged open-for-write and a plain open fails with
    *"file is already open for write"*. SWMR reads it as-is. ``h5clear -s``
    would also clear the flag, but it writes to the file, and run data is not to
    be modified.

    A successful read does not prove the producing job finished -- a live writer
    looks identical. Ask the scheduler for that.
    """

    try:
        import h5py
    except ModuleNotFoundError as error:  # pragma: no cover - environment-dependent
        raise AdapterError(
            "reading OneQMC output needs h5py, which is not a TPEN dependency; run "
            "this adapter with the OneQMC virtualenv that already provides it"
        ) from error

    path = result_path(run_dir)
    if not path.is_file():
        raise AdapterError(f"no {RESULT_FILENAME} under {run_dir}")

    try:
        with h5py.File(path, "r", swmr=True, libver="v110") as handle:
            rows = {}
            for dataset in (MOL_INDEX_DATASET, ENERGY_DATASET, SPREAD_DATASET):
                if dataset not in handle:
                    raise AdapterError(f"{path} has no '{dataset}' dataset")
                rows[dataset] = [
                    [float(value) for value in _as_row(row)] for row in handle[dataset][:]
                ]
    except OSError as error:
        raise AdapterError(f"cannot open {path}: {error}") from error

    if not rows[MOL_INDEX_DATASET]:
        raise AdapterError(f"{path} has no logged steps")

    # Attrs are provenance, not measurement, and reach the caller through
    # read_attrs instead of riding along here; only the series feeds the
    # estimator.
    return gather_molecule(
        rows[MOL_INDEX_DATASET],
        rows[ENERGY_DATASET],
        rows[SPREAD_DATASET],
        mol_idx,
    )


def read_attrs(run_dir: Path) -> dict[str, Any]:
    """Return the HDF5 file attributes of a OneQMC run.

    Separate from :func:`read_series` so a caller can inspect provenance without
    reading the trace.
    """

    try:
        import h5py
    except ModuleNotFoundError as error:  # pragma: no cover - environment-dependent
        raise AdapterError("reading OneQMC output needs h5py") from error

    path = result_path(run_dir)
    if not path.is_file():
        raise AdapterError(f"no {RESULT_FILENAME} under {run_dir}")
    try:
        with h5py.File(path, "r", swmr=True, libver="v110") as handle:
            return dict(handle.attrs)
    except OSError as error:
        raise AdapterError(f"cannot open {path}: {error}") from error


def metadata_from_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Return device and wall-clock metadata from HDF5 file attributes.

    Parameters
    ----------
    attrs : mapping
        The file's attribute mapping. Values may be ``bytes`` or ``str``
        depending on how ``h5py`` stored them.

    Returns
    -------
    dict
        ``device_type``, ``gpu_model``, ``n_gpus`` and ``wall_clock_seconds``,
        each ``None`` when the corresponding attribute is missing or
        unparseable. Nothing is guessed: a missing ``stop_time`` yields no
        duration rather than a fabricated one.

    Notes
    -----
    ``gpu_type`` is JAX's ``device_kind``, so ``"cpu"`` there is a genuine CPU
    run rather than missing data, and it is reported as such.

    ``start_time``/``stop_time`` are ``str(datetime.now())`` -- naive local time,
    no zone. The duration between them measures the metric logger's lifetime:
    it starts when the logger is constructed, which is after job startup, and
    ends at the last logged scalar, which for a killed run is before the job
    died. It is therefore a lower bound on job wall clock, and the record's
    notes say so.
    """

    gpu_type = _decode(attrs.get("gpu_type"))
    device_type: str | None
    if gpu_type is None:
        device_type = None
    elif gpu_type.strip().lower() == "cpu":
        device_type, gpu_type = "cpu", None
    else:
        device_type = "cuda"

    n_gpus = attrs.get("num_gpus")
    try:
        n_gpus = int(n_gpus) if n_gpus is not None else None
    except (TypeError, ValueError):
        n_gpus = None

    start, stop = _decode(attrs.get("start_time")), _decode(attrs.get("stop_time"))
    wall_clock: float | None = None
    if start is not None and stop is not None:
        try:
            wall_clock = (
                datetime.fromisoformat(stop) - datetime.fromisoformat(start)
            ).total_seconds()
        except ValueError:
            wall_clock = None
        if wall_clock is not None and wall_clock < 0.0:
            # Cannot happen from a single writer, and a negative duration would
            # fail record validation; drop it rather than pass it on.
            wall_clock = None

    return {
        "device_type": device_type,
        "gpu_model": gpu_type,
        "n_gpus": n_gpus,
        "wall_clock_seconds": wall_clock,
    }


def huber_mean(
    values: Sequence[float], delta: float = HUBER_DELTA_HARTREE
) -> tuple[float, int]:
    """Return the Huber M-estimate of a series, and how many points it clipped.

    This is the estimand of ``oneqmc.analysis.energy.robust_mean``: the
    minimizer of ``sum(huber(v - mu, delta))``. The vendor reaches it with
    ``scipy.optimize.minimize`` from the arithmetic mean; this implementation
    solves the first-order condition ``sum(clip(v - mu, -delta, +delta)) = 0``
    by bisection instead, which needs neither ``scipy`` nor a convergence
    tolerance argument and cannot stop early on a flat gradient.

    Parameters
    ----------
    values : sequence of float
        The series, normally an already-selected estimator window.
    delta : float, optional
        Huber width in hartree. Residuals beyond it contribute linearly, which
        is what bounds an outlier's pull on the estimate.

    Returns
    -------
    tuple of (float, int)
        The estimate, and the number of points whose residual exceeded
        ``delta``. **A count of zero means the estimate is exactly the
        arithmetic mean**, because inside ``delta`` the objective is quadratic;
        reporting it is how a reader tells a robust estimate from a plain
        average.

    Raises
    ------
    AdapterError
        If ``values`` is empty or ``delta`` is not positive.

    Notes
    -----
    The objective's derivative is non-increasing in ``mu`` and changes sign
    inside ``[min(values), max(values)]``, so bisection is exact to machine
    precision in a fixed 100 iterations -- no tolerance to tune and no
    dependence on a starting point.
    """

    data = [float(value) for value in values]
    if not data:
        raise AdapterError("huber mean needs at least one value")
    if not delta > 0.0:
        raise AdapterError(f"huber delta must be positive, got {delta}")

    low, high = min(data), max(data)
    if low == high:
        return low, 0

    def gradient(mu: float) -> float:
        # d/d mu of the loss, negated: non-increasing in mu, positive at `low`
        # and negative at `high`, so the root is bracketed.
        return sum(min(max(value - mu, -delta), delta) for value in data)

    for _ in range(100):
        middle = 0.5 * (low + high)
        if gradient(middle) > 0.0:
            low = middle
        else:
            high = middle

    estimate = 0.5 * (low + high)
    clipped = sum(1 for value in data if abs(value - estimate) > delta)
    return estimate, clipped


def record_from_series(
    energies: Sequence[float],
    spreads: Sequence[float],
    *,
    system_id: str,
    electron_batch_size: int,
    ansatz: str,
    estimator: str,
    training_provenance: str,
    checkpoint_provenance: str | None = None,
    logged_steps: int | None = None,
    nonfinite_dropped: int = 0,
    mol_idx: int = 0,
    metric_logger_period: int = 1,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    min_tail_steps: int = MIN_TAIL_STEPS,
    allow_short_tail: bool = False,
    run_id: str,
    code_commit: str | None = None,
    optimizer: str = "kfac",
    dtype: str | None = None,
    seed: int | None = None,
    parameter_count: int | None = None,
    device_type: str | None = None,
    gpu_model: str | None = None,
    n_gpus: int | None = None,
    wall_clock_seconds: float | None = None,
    note: str | None = None,
) -> BaselineRecord:
    """Build a record from an already-gathered series.

    Separated from :func:`build_record` so that every decision this adapter
    makes -- window selection, the Huber estimate, the error bar, the
    provenance text -- is exercisable without an HDF5 file, and therefore
    without ``h5py`` installed.

    Parameters
    ----------
    energies, spreads : sequence of float
        One electron-batch mean local energy and one across-walker standard
        deviation per logged step that carried this molecule.
    electron_batch_size : int
        Walkers per step, needed for the ``samples`` denominator. Required, not
        inferred: the file records the batch mean, not the batch size.
    ansatz : str
        OneQMC ansatz name, e.g. ``"orbformer-se"``. Required and never
        inferred; a run directory does not reveal it, and it decides whether the
        row is native code or a reimplementation.
    estimator : str
        ``"inference"`` for a fixed-parameter ``--test`` run, which is the
        protocol the vendor recommends, or ``"training_tail"`` for an average
        over the end of a fine-tuning run. Required: the two are different
        quantities and the file looks the same either way.
    training_provenance : str
        One of :data:`PROVENANCE_TIERS`. ``"finetune-from-release"`` additionally
        requires ``checkpoint_provenance``.
    checkpoint_provenance : str or None, optional
        Identity of the released checkpoint -- URL, hash, or both. Mandatory for
        a fine-tuned row: without it the row cannot be reproduced and cannot be
        audited against the release it claims to start from.
    logged_steps : int or None, optional
        Rows the run wrote, which exceeds ``len(energies)`` when other molecules
        shared the mol batch. Reported in the notes as coverage.
    metric_logger_period : int, optional
        ``--metric-logger-period`` of the run. Logged steps are multiplied by it
        to recover optimizer/evaluation steps, since walkers are propagated on
        unlogged steps too.
    note : str or None, optional
        Operator caveat, appended to the generated notes. Never replaces any
        generated sentence.

    Returns
    -------
    BaselineRecord
        Validated record carrying ``code="oneqmc"``.

    Raises
    ------
    AdapterError
        On a shape mismatch, an unknown provenance tier, a fine-tuned row with
        no checkpoint provenance, a blank ``note``, a non-positive
        ``electron_batch_size`` or ``metric_logger_period``, or a window that
        cannot be selected.
    """

    energy_series = [float(value) for value in energies]
    spread_series = [float(value) for value in spreads]
    if len(energy_series) != len(spread_series):
        raise AdapterError(
            f"{len(energy_series)} energies but {len(spread_series)} spreads; the two "
            "datasets must be gathered together"
        )
    if training_provenance not in PROVENANCE_TIERS:
        raise AdapterError(
            f"training_provenance must be one of {PROVENANCE_TIERS}, got "
            f"{training_provenance!r}"
        )
    if training_provenance == "finetune-from-release" and not (
        checkpoint_provenance or ""
    ).strip():
        raise AdapterError(
            "a fine-tuned row requires checkpoint_provenance: a record that cannot "
            "name the released checkpoint it started from is not reproducible"
        )
    if note is not None and not note.strip():
        raise AdapterError("note must be non-empty when given; a caveat that vanishes is worse than none")
    if electron_batch_size <= 0:
        raise AdapterError(f"electron_batch_size must be positive, got {electron_batch_size}")
    if metric_logger_period <= 0:
        raise AdapterError(f"metric_logger_period must be positive, got {metric_logger_period}")

    window = select_tail(
        len(energy_series),
        tail_fraction,
        min_steps=min_tail_steps,
        allow_below_floor=allow_short_tail,
    )
    tail = energy_series[-window:]
    tail_spreads = spread_series[-window:]

    energy, clipped = huber_mean(tail)

    # `blocking_stderr` stops before its first level when the window holds fewer
    # than MIN_BLOCKS values, and then returns 0.0 -- a zero bar, which the
    # record schema accepts as non-negative and which reads as infinite
    # precision. An Orbformer evaluation pass logs every `--metric-logger-period`
    # step, so a 2000-step run at period 25 yields 80 rows and a quarter-tail of
    # 20, i.e. exactly that case. Lower the block floor to the window length so
    # level one runs, and say in the notes that the resulting bar is the naive
    # one and understates.
    block_floor = min(MIN_BLOCKS, len(tail))
    stderr, _ = blocking_stderr(tail, min_blocks=block_floor)
    # `std_elec` is the across-walker spread of the local energy, so its square
    # is the local-energy variance itself -- not the variance of the mean.
    local_variance = statistics.fmean(spread**2 for spread in tail_spreads)

    # Convergence and autocorrelation are reported, never used to alter the
    # number. A verdict that could change the estimate would be a selection
    # rule; this one only describes.
    try:
        signs, monotone = sign_test(tail, windows=SIGN_TEST_WINDOWS)
        verdict = (
            f"windowed sign test over {SIGN_TEST_WINDOWS} windows gives '{signs}': "
            + (
                "MONOTONE, so the series was still drifting at the end of this trace "
                "and the run may not have converged"
                if monotone
                else "not monotone, consistent with noise rather than drift"
            )
        )
    except AdapterError:
        verdict = (
            f"tail of {len(tail)} steps is too short for a {SIGN_TEST_WINDOWS}-window "
            "sign test, so convergence is UNASSESSED"
        )

    try:
        inflation = f"{blocking_inflation(tail, min_blocks=block_floor):.2f}x"
    except AdapterError:
        inflation = "undefined"

    unblocked = (
        f" The window of {len(tail)} steps is below the {MIN_BLOCKS}-block floor, so "
        "blocking could not run and this bar is the NAIVE standard error: it ignores "
        "autocorrelation and therefore UNDERSTATES the uncertainty."
        if block_floor < MIN_BLOCKS
        else ""
    )

    steps = len(energy_series) * metric_logger_period

    short = (
        " Window is BELOW the standard minimum, so this estimate is provisional."
        if len(tail) < min_tail_steps
        else ""
    )
    estimator_text = (
        f"Fixed-parameter inference pass over the last {len(tail)} of "
        f"{len(energy_series)} logged steps."
        if estimator == "inference"
        else f"Training-tail average over the last {len(tail)} of "
        f"{len(energy_series)} logged steps."
    ) + short

    huber_text = (
        f"Energy is the Huber M-estimate matching oneqmc robust_mean at delta="
        f"{HUBER_DELTA_HARTREE} hartree; "
        + (
            "no step in the window exceeded delta, so it equals the arithmetic mean"
            if clipped == 0
            else f"{clipped} of {len(tail)} window steps exceeded delta and were clipped"
        )
    )

    coverage = (
        f"Gathered through {MOL_INDEX_DATASET} for molecule index {mol_idx}: "
        f"{len(energy_series)} of {logged_steps if logged_steps is not None else len(energy_series)} "
        "logged steps carried it, gaps skipped rather than forward-filled"
        + (f", {nonfinite_dropped} step(s) dropped as non-finite" if nonfinite_dropped else "")
        + "."
    )
    if metric_logger_period != 1:
        coverage += (
            f" Logger period {metric_logger_period}, so `steps` counts "
            "optimizer/evaluation steps while the window is in logged steps."
        )

    bar_text = (
        "Error bar is a blocked (Flyvbjerg-Petersen) standard error over the step "
        f"series, autocorrelation inflation {inflation} over the naive estimate. This "
        "is NOT the across-chain variance the Orbformer paper reports (arXiv:2506.19960 "
        "Appendix H): per-walker local energies are not logged, so that quantity is not "
        "reconstructible from result.h5." + unblocked
    )

    native = (
        " OneQMC is Orbformer's own codebase, so this ansatz is native rather than a "
        "reimplementation."
        if ansatz in NATIVE_ANSATZES
        else f" '{ansatz}' here is OneQMC's REIMPLEMENTATION, not the {ansatz} authors' "
        "own code; no claim about that method may rest on this row."
    )

    if training_provenance == "finetune-from-release":
        provenance_text = (
            f" PROVENANCE: fine-tuned from a released checkpoint ({checkpoint_provenance}). "
            "This is NOT a from-scratch reproduction of the paper's training run, whose "
            "pretraining alone is ~11200 A100-hours over 800000 steps."
        )
    else:
        provenance_text = " PROVENANCE: trained from scratch in this program."

    clock_text = (
        " Wall clock is the metric logger's start_time-to-stop_time span, a lower bound "
        "on job wall clock: it excludes job startup before the logger exists and ends at "
        "the last logged scalar, which for a killed run precedes the job's death."
        if wall_clock_seconds is not None
        else ""
    )

    notes = (
        f"{estimator_text} {huber_text}. {bar_text} {coverage} {verdict}."
        f"{native}{provenance_text}{clock_text}"
    )
    if note is not None:
        notes = f"{notes} {note.strip()}"

    return BaselineRecord(
        system_id=system_id,
        code="oneqmc",
        code_commit=code_commit,
        ansatz=ansatz,
        energy_hartree=energy,
        energy_stderr_hartree=stderr,
        local_energy_variance_hartree2=local_variance,
        steps=steps,
        samples=steps * electron_batch_size,
        wall_clock_seconds=wall_clock_seconds,
        estimator=estimator,
        device_type=device_type,
        gpu_model=gpu_model,
        n_gpus=n_gpus,
        dtype=dtype,
        optimizer=optimizer,
        parameter_count=parameter_count,
        seed=seed,
        run_id=run_id,
        # The adapter cannot know the collector's scan root. Leave this blank so
        # collect() stamps the collision-free path relative to that root.
        run_dir=None,
        collected_at=None,
        notes=notes,
    )


def build_record(
    run_dir: Path,
    *,
    system_id: str,
    electron_batch_size: int,
    ansatz: str,
    estimator: str,
    training_provenance: str,
    checkpoint_provenance: str | None = None,
    mol_idx: int = 0,
    metric_logger_period: int = 1,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    min_tail_steps: int = MIN_TAIL_STEPS,
    allow_short_tail: bool = False,
    code_commit: str | None = None,
    optimizer: str = "kfac",
    dtype: str | None = None,
    seed: int | None = None,
    parameter_count: int | None = None,
    run_id: str | None = None,
    note: str | None = None,
) -> BaselineRecord:
    """Build one comparison record from a OneQMC run directory.

    Reads the gathered series and the device/timing attrs from ``result.h5``,
    then delegates every decision to :func:`record_from_series`.
    """

    series = read_series(run_dir, mol_idx=mol_idx)
    metadata = metadata_from_attrs(read_attrs(run_dir))

    return record_from_series(
        series.energies,
        series.spreads,
        system_id=system_id,
        electron_batch_size=electron_batch_size,
        ansatz=ansatz,
        estimator=estimator,
        training_provenance=training_provenance,
        checkpoint_provenance=checkpoint_provenance,
        logged_steps=series.logged_steps,
        nonfinite_dropped=series.nonfinite_dropped,
        mol_idx=series.mol_idx,
        metric_logger_period=metric_logger_period,
        tail_fraction=tail_fraction,
        min_tail_steps=min_tail_steps,
        allow_short_tail=allow_short_tail,
        run_id=run_id or run_dir.name,
        code_commit=code_commit,
        optimizer=optimizer,
        dtype=dtype,
        seed=seed,
        parameter_count=parameter_count,
        note=note,
        **metadata,
    )


def write_record(record: BaselineRecord, run_dir: Path) -> Path:
    """Write ``baseline_record.json`` into ``run_dir`` and return its path."""

    path = run_dir / RECORD_FILENAME
    path.write_text(json.dumps(record.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _as_row(row: Any) -> Sequence[Any]:
    """Return a 1-D row as a sequence, wrapping a scalar into a one-slot row."""

    if hasattr(row, "__len__") and not isinstance(row, (str, bytes)):
        return row
    return [row]


def _decode(value: Any) -> str | None:
    """Return an HDF5 string attribute as ``str``, or None if absent."""

    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _is_finite(value: float) -> bool:
    """Return True for a real, finite float."""

    return math.isfinite(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit code."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument(
        "--electron-batch-size",
        type=int,
        required=True,
        help="walkers per step; result.h5 records the batch mean, not the batch size",
    )
    # Required, not defaulted. Nothing in a OneQMC run directory reveals the
    # ansatz, and the value decides whether the row is native code or a
    # reimplementation of someone else's method.
    parser.add_argument("--ansatz", required=True)
    parser.add_argument(
        "--estimator",
        choices=("training_tail", "inference"),
        required=True,
        help="'inference' for a --test run, the protocol the vendor recommends",
    )
    parser.add_argument(
        "--training-provenance",
        choices=PROVENANCE_TIERS,
        required=True,
        help="how the measured parameters came to exist; never inferred",
    )
    parser.add_argument(
        "--checkpoint-provenance",
        default=None,
        help="URL and/or hash of the released checkpoint; required when fine-tuned",
    )
    parser.add_argument("--mol-idx", type=int, default=0)
    parser.add_argument("--metric-logger-period", type=int, default=1)
    parser.add_argument("--tail-fraction", type=float, default=DEFAULT_TAIL_FRACTION)
    parser.add_argument(
        "--min-tail-steps",
        type=int,
        default=MIN_TAIL_STEPS,
        help="absolute floor on the estimator window, in logged steps",
    )
    parser.add_argument(
        "--allow-short-tail",
        action="store_true",
        help=(
            "accept a window below the floor for a short run; the record says so. "
            "Note the floor is clipped to the run length, so on a short run this "
            "widens the window to the whole trace unless --min-tail-steps is lowered "
            "too"
        ),
    )
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--optimizer", default="kfac")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--parameter-count", type=int, default=None)
    parser.add_argument(
        "--note",
        default=None,
        help="operator caveat appended to the generated notes, never replacing them",
    )
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args(argv)

    try:
        record = build_record(
            args.run_dir,
            system_id=args.system_id,
            electron_batch_size=args.electron_batch_size,
            ansatz=args.ansatz,
            estimator=args.estimator,
            training_provenance=args.training_provenance,
            checkpoint_provenance=args.checkpoint_provenance,
            mol_idx=args.mol_idx,
            metric_logger_period=args.metric_logger_period,
            tail_fraction=args.tail_fraction,
            min_tail_steps=args.min_tail_steps,
            allow_short_tail=args.allow_short_tail,
            code_commit=args.code_commit,
            optimizer=args.optimizer,
            dtype=args.dtype,
            seed=args.seed,
            parameter_count=args.parameter_count,
            note=args.note,
        )
    except AdapterError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(record.to_json_dict(), indent=2))
        return 0

    print(write_record(record, args.run_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
