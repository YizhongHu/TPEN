"""Read and summarize baseline training trajectories.

The adapters own the source formats. This module owns the common table and the
Stage-1 questions: does the loss move, are values finite, and does the
walker-local-energy variance move? FermiNet's ``ewvar`` is retained as a
separate time-weighted energy diagnostic and is never promoted to walker
variance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from experiments.baselines.errors import AdapterError


@dataclass(frozen=True)
class TrajectoryRow:
    """One optimizer step in the code-neutral trajectory table."""

    step: int
    energy: float
    walker_local_energy_variance: float | None
    energy_time_ewvar: float | None
    finite: bool
    step_basis: str
    source_code: str
    source_file: str


class TrajectoryTable(list[TrajectoryRow]):
    """Rows plus source metadata needed to interpret their step labels."""

    def __init__(self, rows: Iterable[TrajectoryRow]):
        super().__init__(rows)
        self.step_basis = _step_basis_from_rows(self)


def _step_basis_from_rows(rows: Sequence[TrajectoryRow]) -> str:
    """Return the sole step basis represented by rows."""

    bases = {row.step_basis for row in rows}
    if len(bases) != 1:
        raise AdapterError("trajectory rows have mixed step_basis values")
    return bases.pop()


def read_trajectory(code: str, run_dir: Path) -> list[TrajectoryRow]:
    """Read one run through its format-specific adapter."""

    if code == "ferminet":
        from experiments.baselines.adapters import ferminet

        raw_rows = ferminet.read_trajectory(run_dir)
        source_file = str(run_dir / ferminet.TRAIN_STATS_FILENAME)
    elif code == "deepqmc":
        from experiments.baselines.adapters import deepqmc

        raw_rows = deepqmc.read_trajectory(run_dir)
        source_file = str(deepqmc.result_path(run_dir))
    else:
        raise AdapterError(f"trajectory reader is not implemented for code {code!r}")

    rows: list[TrajectoryRow] = []
    for raw in raw_rows:
        values = [raw["energy"]]
        if raw["walker_local_energy_variance"] is not None:
            values.append(raw["walker_local_energy_variance"])
        if raw["energy_time_ewvar"] is not None:
            values.append(raw["energy_time_ewvar"])
        rows.append(TrajectoryRow(
            step=int(raw["step"]),
            energy=float(raw["energy"]),
            walker_local_energy_variance=(
                None if raw["walker_local_energy_variance"] is None
                else float(raw["walker_local_energy_variance"])
            ),
            energy_time_ewvar=(
                None if raw["energy_time_ewvar"] is None else float(raw["energy_time_ewvar"])
            ),
            finite=all(math.isfinite(float(value)) for value in values),
            step_basis=str(raw["step_basis"]),
            source_code=code,
            source_file=source_file,
        ))
    if not rows:
        raise AdapterError(f"{source_file} has no trajectory rows")
    return TrajectoryTable(rows)


def _finite_mean(rows: Iterable[TrajectoryRow], field: str) -> float | None:
    values = [
        float(value)
        for row in rows
        if (value := getattr(row, field)) is not None and math.isfinite(float(value))
    ]
    return math.fsum(values) / len(values) if values else None


def summarize(rows: Sequence[TrajectoryRow], *, window_fraction: float = 0.1) -> dict[str, object]:
    """Return a Stage-1 summary for a trajectory.

    The first and last windows use ``max(1, floor(n_steps * fraction))`` rows.
    Trend means ignore non-finite values, while ``non_finite_count`` remains an
    explicit quality signal. A variance ratio is reported only when both window
    means exist and the first mean is non-zero. The endpoint ratio is not a
    verdict that the trajectory avoided a late rise or recurrence.
    """

    if not rows:
        raise AdapterError("cannot summarize an empty trajectory")
    if not 0 < window_fraction <= 0.5:
        raise ValueError("window_fraction must be in (0, 0.5]")
    window = max(1, int(len(rows) * window_fraction))
    first, last = rows[:window], rows[-window:]
    nonfinite = [row for row in rows if not row.finite]

    first_variance = _finite_mean(first, "walker_local_energy_variance")
    last_variance = _finite_mean(last, "walker_local_energy_variance")
    variance_available = first_variance is not None and last_variance is not None
    if not variance_available:
        variance_ratio: float | None = None
        variance_reason = "walker local-energy variance is unavailable in the source trajectory"
    elif first_variance == 0.0:
        variance_ratio = None
        variance_reason = "first-window walker variance is zero; ratio is undefined"
    else:
        variance_ratio = last_variance / first_variance
        variance_reason = None

    first_ewvar = _finite_mean(first, "energy_time_ewvar")
    last_ewvar = _finite_mean(last, "energy_time_ewvar")
    ewvar_ratio = None if first_ewvar in (None, 0.0) or last_ewvar is None else last_ewvar / first_ewvar
    return {
        "source_code": rows[0].source_code,
        "source_file": rows[0].source_file,
        "step_basis": _step_basis_from_rows(rows),
        "n_steps": len(rows),
        "first_step": rows[0].step,
        "last_step": rows[-1].step,
        "window_fraction": window_fraction,
        "window_size": window,
        "non_finite_count": len(nonfinite),
        "first_non_finite_step": nonfinite[0].step if nonfinite else None,
        "energy_first_window_mean": _finite_mean(first, "energy"),
        "energy_last_window_mean": _finite_mean(last, "energy"),
        "walker_variance_first_window_mean": first_variance,
        "walker_variance_last_window_mean": last_variance,
        "walker_variance_ratio_last_over_first": variance_ratio,
        "walker_variance_unavailable_reason": variance_reason,
        "energy_time_ewvar_first_window_mean": first_ewvar,
        "energy_time_ewvar_last_window_mean": last_ewvar,
        "energy_time_ewvar_ratio_last_over_first": ewvar_ratio,
    }


def write_csv(rows: Sequence[TrajectoryRow], path: Path) -> None:
    """Write trajectory rows to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", choices=("ferminet", "deepqmc"), required=True)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="trajectory CSV output")
    parser.add_argument("--json", dest="json_path", type=Path, help="summary JSON output")
    parser.add_argument("--window-fraction", type=float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the trajectory reader and write only explicitly requested files."""

    args = _parser().parse_args(argv)
    rows = read_trajectory(args.code, args.run_dir)
    summary = summarize(rows, window_fraction=args.window_fraction)
    if args.out is not None:
        write_csv(rows, args.out)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["TrajectoryRow", "TrajectoryTable", "main", "read_trajectory", "summarize", "write_csv"]
