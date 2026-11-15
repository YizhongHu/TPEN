"""Translate a FermiNet run directory into a :class:`BaselineRecord`.

A FermiNet run leaves ``train_stats.csv`` (columns ``step,energy,ewmean,ewvar,
pmove``) in its save path, and the job script leaves a stdout log carrying the
device identity and the run's start/end stamps. This module reads both and
emits the one ``baseline_record.json`` the collector expects.

Two properties of the energy estimate are deliberate and must survive any
refactor.

**The error bar comes from blocking, not from the naive standard error.**
Successive VMC steps are correlated, so ``sigma/sqrt(n)`` understates the
uncertainty. :func:`blocking_stderr` repeatedly pair-averages the series and
takes the largest standard error seen while enough blocks remain to estimate
one reliably.

**The estimator is a training-tail average, which is not what FermiNet's paper
reports.** Their table values come from a separate post-training evaluation
phase. The difference is recorded in the record's ``notes`` so a later
comparison cannot silently treat the two as equivalent.

Examples
--------
::

    uv run python -m experiments.baselines.adapters.ferminet \\
        --run-dir path/to/run --system-id li_atom --batch-size 4096
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from experiments.baselines.errors import AdapterError
from experiments.baselines.records import BaselineRecord

# Re-exported so that callers importing these from this module keep working;
# both now live in the statistics module, which owns the concept.
from experiments.baselines.statistics import (  # noqa: F401
    MIN_TAIL_STEPS,
    blocking_stderr,
    select_tail,
)

TRAIN_STATS_FILENAME = "train_stats.csv"
RECORD_FILENAME = "baseline_record.json"

#: ``NVIDIA A100-SXM4-40GB, GPU-39166c9d-..., 40960 MiB``
_NVIDIA_SMI = re.compile(r"^\s*(NVIDIA [^,]+),\s*(GPU-[0-9a-f-]+)", re.MULTILINE)
_START_STAMP = re.compile(r"start=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})")
_END_STAMP = re.compile(r"end=(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2})")


class _TrajectoryRows(list[dict[str, int | float | None]]):
    """List-compatible rows carrying optimizer-step provenance."""

    step_basis = "optimizer_step"


def read_trajectory(run_dir: Path) -> list[dict[str, int | float | None]]:
    """Read FermiNet's loss and time-weighted variance per optimizer step.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory containing ``train_stats.csv``.

    Returns
    -------
    list of dict
        Rows with ``step``, ``energy``, ``walker_local_energy_variance`` and
        ``energy_time_ewvar`` fields. FermiNet does not write walker variance;
        its ``ewvar`` is deliberately kept under the separate time-variance
        field.

    Raises
    ------
    AdapterError
        If the file is missing, required columns are absent, or it has no rows.
        An empty run is an error, not an empty result.
    """

    stats_path = run_dir / TRAIN_STATS_FILENAME
    if not stats_path.is_file():
        raise AdapterError(f"no {TRAIN_STATS_FILENAME} in {run_dir}")

    with stats_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", "energy", "ewvar"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            if "energy" in missing:
                raise AdapterError(f"{stats_path} has no 'energy' column")
            raise AdapterError(f"{stats_path} is missing required column(s): {', '.join(missing)}")
        rows = _TrajectoryRows()
        for row_number, row in enumerate(reader, 2):
            try:
                rows.append({
                    "step": int(row["step"]),
                    "energy": float(row["energy"]),
                    "walker_local_energy_variance": None,
                    "energy_time_ewvar": float(row["ewvar"]),
                    "step_basis": "optimizer_step",
                })
            except (TypeError, ValueError) as error:
                raise AdapterError(f"{stats_path}:{row_number} has invalid trajectory values") from error

    if not rows:
        raise AdapterError(f"{stats_path} has no data rows")
    return rows


def read_energies(run_dir: Path) -> list[float]:
    """Return the per-step energy column of a FermiNet run."""

    return [float(row["energy"]) for row in read_trajectory(run_dir)]


def parse_device(log_text: str) -> tuple[str | None, str | None]:
    """Return ``(device_type, gpu_model)`` parsed from a job log.

    The model is read from the ``nvidia-smi`` line the job scripts print inside
    the allocation, never inferred from the partition name: the delivered
    device has been observed to disagree with the partition's advertised GRES.
    """

    match = _NVIDIA_SMI.search(log_text)
    if match is None:
        return None, None
    return "cuda", match.group(1).strip()


def parse_wall_clock_seconds(log_text: str) -> float | None:
    """Return elapsed seconds from the ``start=``/``end=`` stamps, or None."""

    start, end = _START_STAMP.search(log_text), _END_STAMP.search(log_text)
    if start is None or end is None:
        return None
    delta = datetime.fromisoformat(end.group(1)) - datetime.fromisoformat(start.group(1))
    return delta.total_seconds()


def build_record(
    run_dir: Path,
    *,
    system_id: str,
    batch_size: int,
    tail_fraction: float = 0.1,
    min_tail_steps: int = MIN_TAIL_STEPS,
    allow_short_tail: bool = False,
    ansatz: str,
    estimator: str = "training_tail",
    log_path: Path | None = None,
    code_commit: str | None = None,
    optimizer: str = "kfac",
    dtype: str | None = None,
    seed: int | None = None,
    parameter_count: int | None = None,
    run_id: str | None = None,
) -> BaselineRecord:
    """Build one comparison record from a completed FermiNet run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Directory holding ``train_stats.csv``.
    system_id : str
        Key into ``experiments/baselines/systems.yaml``.
    batch_size : int
        Walkers per step, needed to convert steps into local-energy evaluations.
    ansatz : str
        Which network actually ran, e.g. ``"ferminet"`` or ``"psiformer"``.
        Required rather than defaulted: this was previously hardcoded, and the
        resulting records claimed FermiNet for Psiformer runs.
    estimator : str, optional
        ``"training_tail"`` (default) or ``"inference"``. A fixed-parameter
        evaluation pass must be recorded as ``"inference"``; it is a different
        quantity from a training-tail average and the two must not be mixed
        silently.
    tail_fraction : float, optional
        Fraction of the trailing run averaged for the energy estimate. Use
        ``1.0`` for an inference run, where the whole trace is the estimate and
        there is no optimization transient to discard.
    log_path : pathlib.Path or None, optional
        Job stdout log, used for device identity and wall clock.
    dtype : str or None, optional
        Left ``None`` unless known. FermiNet does not state its dtype, so
        guessing one would fabricate provenance.

    Returns
    -------
    BaselineRecord
        Validated record, ready to serialise.

    Raises
    ------
    AdapterError
        If the run has no usable energy series, ``tail_fraction`` selects
        fewer than two samples, or the selected tail is constant and so carries
        no error bar.
    """

    energies = read_energies(run_dir)
    # Absolute floor, not fraction alone. This adapter's 0.1 default is MORE
    # exposed than the DeepQMC adapter's 0.25: on a 20000-step run the
    # fraction asks for 2000 steps and MIN_TAIL_STEPS overrides it to 10000,
    # so the realized window is 5x the requested one. That gap is why the
    # notes string below renders the measured window and never the requested
    # fraction. See statistics.MIN_TAIL_STEPS.
    window = select_tail(
        len(energies),
        tail_fraction,
        min_steps=min_tail_steps,
        allow_below_floor=allow_short_tail,
    )
    tail = energies[-window:]

    stderr, n_blocks = blocking_stderr(tail, allow_below_floor=allow_short_tail)
    # Second line of defence on the publication boundary, deliberately
    # independent of what statistics.blocking_stderr currently does. Today that
    # function raises on a degenerate window, so this branch is unreachable
    # through it; before that change it returned exactly 0.0, and records.py
    # rejects only NEGATIVE stderr, so a degenerate run published a baseline row
    # claiming zero uncertainty - the most authoritative-looking number in the
    # table produced by the least informative series. Keep the guard: the cost
    # is one comparison, and it is what holds if the layer below is reverted or
    # grows a new zero-valued route.
    if stderr == 0.0:
        raise AdapterError(
            f"{run_dir}: the selected tail of {len(tail)} steps yields a zero error "
            "bar, so its spread is unmeasured rather than zero; refusing to emit a record"
        )
    # blocking_stderr returns None for the block count when the window was too
    # short to block. None only breaks a caller loudly if the caller does
    # arithmetic on it: the notes below interpolate it with no format spec, and
    # f"{None}" renders the string "None" without complaint. A record reading
    # "from None blocks" looks like a forgotten field, not like "blocking never
    # ran", so refuse instead of formatting it.
    #
    # This branch is the refusal that fires under --allow-short-tail, because
    # the call above now forwards that opt-in: a window under the 32-block
    # minimum comes back as (naive stderr, None) instead of raising one layer
    # down. Before the forwarding it was dead code. Measured on a varying
    # series, tail lengths 2..399 through the bare call gave 368 returns and 30
    # raises and zero Nones, and deleting these six lines left the whole
    # ferminet suite green - the guard read as a second line of defence while
    # guarding nothing reachable.
    #
    # With the flag OFF the 32-block floor is still enforced by statistics, but
    # only by way of a lowered --min-tail-steps: at the default 10000-step floor
    # select_tail refuses first, so statistics' block raise is unreachable on
    # the default path. Also measured, rather than assumed from reading the two
    # floors. Either route ends in a refusal; only the message and the layer
    # that emits it differ.
    #
    # This raise dominates the notes block below, so the bare {n_blocks}
    # interpolations there cannot render "None". The guard is what makes them
    # safe - not a format-time branch, which would be unreachable in turn.
    if n_blocks is None:
        raise AdapterError(
            f"{run_dir}: the selected tail of {len(tail)} steps was never blocked, so "
            "the error bar is an uncorrected naive estimate that understates the "
            "uncertainty; refusing to emit a record that cannot say so in a number"
        )
    device_type, gpu_model, wall_clock = None, None, None
    if log_path is not None and log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        device_type, gpu_model = parse_device(log_text)
        wall_clock = parse_wall_clock_seconds(log_text)

    return BaselineRecord(
        system_id=system_id,
        code="ferminet",
        code_commit=code_commit,
        ansatz=ansatz,
        energy_hartree=statistics.fmean(tail),
        energy_stderr_hartree=stderr,
        local_energy_variance_hartree2=None,
        steps=len(energies),
        samples=len(energies) * batch_size,
        wall_clock_seconds=wall_clock,
        estimator=estimator,
        device_type=device_type,
        gpu_model=gpu_model,
        n_gpus=1 if device_type == "cuda" else None,
        dtype=dtype,
        optimizer=optimizer,
        parameter_count=parameter_count,
        seed=seed,
        run_id=run_id or run_dir.name,
        # The adapter cannot know the collector's scan root. Leave this blank
        # so collect() stamps the collision-free path relative to that root.
        run_dir=None,
        collected_at=None,
        notes=(
            (
                f"Training-tail average over the last {len(tail)} of "
                f"{len(energies)} steps, blocked standard error from {n_blocks} "
                "blocks. NOT the estimator FermiNet's published table uses: those "
                "values come from a separate post-training evaluation phase, so "
                "this number is expected to sit slightly high."
            )
            if estimator == "training_tail"
            else (
                f"Fixed-parameter inference pass over {len(tail)} steps, blocked "
                f"standard error from {n_blocks} blocks. This matches the estimator "
                "behind FermiNet's published table."
            )
        ),
    )


def write_record(record: BaselineRecord, run_dir: Path) -> Path:
    """Write ``baseline_record.json`` into ``run_dir`` and return its path."""

    path = run_dir / RECORD_FILENAME
    path.write_text(json.dumps(record.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit code."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    # Required, not defaulted: this was hardcoded to "ferminet", so Psiformer
    # runs emitted records claiming FermiNet. A wrong value should now take
    # deliberate effort rather than inattention.
    parser.add_argument("--ansatz", required=True)
    parser.add_argument(
        "--estimator",
        choices=("training_tail", "inference"),
        default="training_tail",
        help="use 'inference' for a fixed-parameter evaluation pass",
    )
    parser.add_argument("--tail-fraction", type=float, default=0.1)
    parser.add_argument(
        "--min-tail-steps",
        type=int,
        default=MIN_TAIL_STEPS,
        help="absolute floor on the estimator window, in steps",
    )
    parser.add_argument(
        "--allow-short-tail",
        action="store_true",
        help=(
            "lower the STEP floor on the estimator window for a short run; the "
            "record says the window was short. A window too short to BLOCK is "
            "still refused: the opt-in is forwarded to the blocking estimator "
            "and this adapter then rejects the uncorrected naive error bar "
            "itself, so a sufficiently small window plus this flag is an error, "
            "not an emission"
        ),
    )
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--code-commit", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = parser.parse_args(argv)

    try:
        record = build_record(
            args.run_dir,
            system_id=args.system_id,
            batch_size=args.batch_size,
            ansatz=args.ansatz,
            estimator=args.estimator,
            tail_fraction=args.tail_fraction,
            min_tail_steps=args.min_tail_steps,
            allow_short_tail=args.allow_short_tail,
            log_path=args.log_path,
            code_commit=args.code_commit,
            seed=args.seed,
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
