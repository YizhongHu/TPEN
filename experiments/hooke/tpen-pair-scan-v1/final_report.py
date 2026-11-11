"""Render the ``09_final_report`` attempt for the TPEN pair scan.

This is a NEW minimal report, not a port of ``pair_stability_v3/final_report.py``
(72 KB of v3-specific plotting whose figures were keyed on v3's
``update_normalization`` / ``feature_normalization`` axes and on the
``feature_trace_stability`` / ``readout_trace_stability`` metrics, neither of
which exists in the TPEN stack). Retargeting it would have cost more than
replacing it.

Scope, deliberately small
-------------------------
It reads only the compact CSV tables written by ``final_collect.py`` -- never a
raw run artifact -- and writes four reduced tables, two figures, and one
markdown summary organized around the study's selection contract:

1. PRIMARY: ``eval/mcmc_energy/local_energy_mean``, the variational energy
   estimator. The Hooke omega=0.5 singlet's exact energy is 2.0 Ha, and the
   variational principle bounds the estimator from below by it, so ``energy_error``
   is a signed distance whose sign is itself a diagnostic: a *negative* value is
   not a better wavefunction, it is evidence the estimator is not variational
   (a broken sampler, a stale checkpoint, or a mis-wired restore).
2. SECONDARY: fixed-prior local-energy VARIANCE. Var_q[E_L] = 0 iff psi is an
   eigenstate, for any prior q, so the variance is valid on a fixed geometry
   prior where the mean is not.
3. INVARIANTS: ``full_model_antisymmetry`` and ``trace_equivariance``. These are
   pass/fail properties, not scores; the report surfaces worst-case error and
   mismatch counts so a violated invariant cannot hide inside an axis mean.
4. COST and FAILURES, passed through for budgeting and triage.

Stage inputs and outputs::

    08_final_collect/{attempt_id}/*.csv
      -> 09_final_report/{attempt_id}/{report.md, final_report.json,
                                       tables/*.csv, figures/*.png}
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

# Siblings are loaded study-scoped, not by bare import: experiments/ has several
# same-named modules and the first study loaded would otherwise own the bare name
# for every study after it. See experiments/toolkit/study_imports.py.
#
# The loader is reached BY PATH rather than by putting the repository root on
# sys.path. A study directory that mutates sys.path is the mechanism behind the
# very defect this import exists to fix, and he-cutover's gateway test forbids it
# outright -- so the fix must not reintroduce it in order to install itself.
import importlib.util as _tpen_importlib  # noqa: E402
import sys as _tpen_sys  # noqa: E402
from pathlib import Path as _TpenPath  # noqa: E402

if "_tpen_study_imports" not in _tpen_sys.modules:
    _tpen_spec = _tpen_importlib.spec_from_file_location(
        "_tpen_study_imports",
        _TpenPath(__file__).resolve().parents[3] / "experiments" / "toolkit" / "study_imports.py",
    )
    _tpen_module = _tpen_importlib.module_from_spec(_tpen_spec)
    _tpen_sys.modules["_tpen_study_imports"] = _tpen_module
    _tpen_spec.loader.exec_module(_tpen_module)
sibling = _tpen_sys.modules["_tpen_study_imports"].sibling

plot = sibling(__file__, 'plot')
_tpen_stats = sibling(__file__, 'stats')
_as_float = _tpen_stats.as_float
_format_number = _tpen_stats.format_number
_median = _tpen_stats.median
_quantile = _tpen_stats.quantile
_tpen_utils_io = sibling(__file__, 'utils.io')
write_json = _tpen_utils_io.write_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_FINAL_COLLECT = _tpen_utils_layout.STAGE_FINAL_COLLECT
STAGE_FINAL_REPORT = _tpen_utils_layout.STAGE_FINAL_REPORT
latest_attempt_id = _tpen_utils_layout.latest_attempt_id
stage_dir = _tpen_utils_layout.stage_dir
write_latest = _tpen_utils_layout.write_latest
_tpen_utils_naming = sibling(__file__, 'utils.naming')
log_prefix = _tpen_utils_naming.log_prefix
study_name = _tpen_utils_naming.study_name
_tpen_utils_time = sibling(__file__, 'utils.time')
new_attempt_id = _tpen_utils_time.new_attempt_id

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.toolkit.artifacts import read_csv as _read_csv, write_csv as _write_csv  # noqa: E402

DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"

# Exact energy of the Hooke omega=0.5 singlet (Taut 1993). The variational
# principle bounds the MCMC estimator from below by this value.
EXACT_HOOKE_ENERGY = 2.0
PRIMARY_METRIC = "eval/mcmc_energy/local_energy_mean"
SECONDARY_METRIC = "eval/stratified_geometry/local_energy_variance"

# Compact tables this report consumes. A missing table is reported as a warning
# and renders an explicit "no data" panel rather than crashing, because a
# partially collected lineage is a normal intermediate state.
SOURCE_TABLES = (
    "run_index.csv",
    "architecture_summary.csv",
    "energy_by_run.csv",
    "symmetry_summary.csv",
    "trace_summary.csv",
    "training_curve_summary.csv",
    "failure_modes.csv",
    "cost_by_axis.csv",
)

REPORT_TABLES = (
    "energy_by_axis.csv",
    "invariants_by_axis.csv",
    "failures_by_axis.csv",
    "cost_by_axis.csv",
)
REPORT_FIGURES = (
    "energy_error_heatmap.png",
    "training_energy_curves.png",
)

ENERGY_BY_AXIS_COLUMNS = [
    "report_row",
    "report_col",
    "n_runs",
    "energy_median",
    "energy_q25",
    "energy_q75",
    "energy_error_median",
    "n_below_exact",
    "local_energy_var_median",
]
INVARIANTS_BY_AXIS_COLUMNS = [
    "report_row",
    "report_col",
    "antisymmetry_logabs_error_max",
    "antisymmetry_sign_mismatch_total",
    "trace_equivariance_error_max",
    "trace_comparison_error_total",
]
FAILURES_BY_AXIS_COLUMNS = [
    "report_row",
    "report_col",
    "n_failure_rows",
    "dominant_failure_mode",
    "dominant_failure_count",
]


# ---------------------------------------------------------------------------
# Attempt resolution
# ---------------------------------------------------------------------------
def resolve_final_collect_attempt_id(
    results_root: str | Path,
    requested: str | None = None,
) -> str:
    """Return the source ``08_final_collect`` attempt id.

    Parameters
    ----------
    results_root
        Study results root.
    requested
        Explicit attempt id, or ``None`` to use the stage's latest pointer.

    Returns
    -------
    str
        The resolved attempt id.

    Raises
    ------
    FileNotFoundError
        When no attempt is requested and the stage has none.
    """

    if requested:
        return str(requested)
    stage = stage_dir(results_root, STAGE_FINAL_COLLECT)
    latest = latest_attempt_id(stage)
    if latest is None:
        raise FileNotFoundError(f"no {STAGE_FINAL_COLLECT} attempts under {stage}")
    return latest


def _axis_key(row: dict[str, Any]) -> tuple[str, str]:
    """Return one row's (report row, report column) grouping key."""

    return str(row.get("report_row", "")), str(row.get("report_col", ""))


def _grouped_by_axis(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group compact-table rows by their two report axes."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_axis_key(row)].append(row)
    return grouped


def _max_float(values: Sequence[Any]) -> float | None:
    """Return the largest finite value, or ``None`` when there is none."""

    finite = [value for value in (_as_float(item) for item in values) if value is not None]
    return max(finite) if finite else None


def _sum_int(values: Sequence[Any]) -> int:
    """Return the integer sum of parseable count columns."""

    return int(sum(value for value in (_as_float(item) for item in values) if value is not None))


# ---------------------------------------------------------------------------
# Reduced tables
# ---------------------------------------------------------------------------
def energy_by_axis_rows(energy_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate ``energy_by_run.csv`` over seed replicates per report cell.

    ``n_below_exact`` counts runs whose variational estimate came out *below*
    the exact 2.0 Ha. That count is reported rather than hidden in a median
    because it cannot happen for a correct variational estimator, so a non-zero
    value invalidates the cell instead of merely ranking it low.
    """

    out: list[dict[str, Any]] = []
    for (report_row, report_col), rows in sorted(_grouped_by_axis(energy_rows).items()):
        energies = [_as_float(row.get("energy_mean")) for row in rows]
        finite_energies = [value for value in energies if value is not None]
        errors = [_as_float(row.get("energy_error")) for row in rows]
        variances = [_as_float(row.get("local_energy_var")) for row in rows]
        out.append(
            {
                "report_row": report_row,
                "report_col": report_col,
                "n_runs": len(rows),
                "energy_median": _format_number(_median(finite_energies)),
                "energy_q25": _format_number(_quantile(finite_energies, 0.25)),
                "energy_q75": _format_number(_quantile(finite_energies, 0.75)),
                "energy_error_median": _format_number(_median(errors)),
                "n_below_exact": sum(1 for value in finite_energies if value < EXACT_HOOKE_ENERGY),
                "local_energy_var_median": _format_number(_median(variances)),
            }
        )
    return out


def invariants_by_axis_rows(
    symmetry_rows: Sequence[dict[str, Any]],
    trace_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate the two invariant tables per report cell using worst case.

    Invariants are aggregated by MAXIMUM, never by mean or median: an
    antisymmetry violation on one seed is a violation of the model class, and
    averaging it against nine clean seeds would report it as small.
    """

    symmetry = _grouped_by_axis(symmetry_rows)
    trace = _grouped_by_axis(trace_rows)
    out: list[dict[str, Any]] = []
    for key in sorted(set(symmetry) | set(trace)):
        report_row, report_col = key
        symmetry_cell = symmetry.get(key, [])
        trace_cell = trace.get(key, [])
        out.append(
            {
                "report_row": report_row,
                "report_col": report_col,
                "antisymmetry_logabs_error_max": _format_number(
                    _max_float([row.get("logabs_error_max") for row in symmetry_cell])
                ),
                "antisymmetry_sign_mismatch_total": _sum_int(
                    [row.get("sign_mismatch_count") for row in symmetry_cell]
                ),
                "trace_equivariance_error_max": _format_number(
                    _max_float([row.get("max_equivariance_error") for row in trace_cell])
                ),
                "trace_comparison_error_total": _sum_int(
                    [row.get("comparison_error_count") for row in trace_cell]
                ),
            }
        )
    return out


def failures_by_axis_rows(failure_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count failure rows per report cell and name the dominant mode."""

    out: list[dict[str, Any]] = []
    for (report_row, report_col), rows in sorted(_grouped_by_axis(failure_rows).items()):
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            mode = str(row.get("failure_mode", "")) or "unlabelled"
            task = str(row.get("task", ""))
            counts[f"{task}:{mode}" if task else mode] += 1
        dominant, dominant_count = ("", 0)
        if counts:
            dominant, dominant_count = max(sorted(counts.items()), key=lambda item: item[1])
        out.append(
            {
                "report_row": report_row,
                "report_col": report_col,
                "n_failure_rows": len(rows),
                "dominant_failure_mode": dominant,
                "dominant_failure_count": dominant_count,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_energy_error_figure(
    path: Path,
    energy_axis_rows: Sequence[dict[str, Any]],
    *,
    row_label: str,
    col_label: str,
) -> None:
    """Render the median energy-error heatmap over the two report axes."""

    heatmap_rows = [
        {
            row_label: row["report_row"],
            col_label: row["report_col"],
            "energy_error_median": _as_float(row.get("energy_error_median")),
        }
        for row in energy_axis_rows
        if _as_float(row.get("energy_error_median")) is not None
    ]
    if not heatmap_rows:
        plot.save_no_data(path, "median energy error (Ha)")
        return
    plot.save_heatmap(
        path,
        heatmap_rows,
        row_key=row_label,
        col_key=col_label,
        value_key="energy_error_median",
        title=f"median {PRIMARY_METRIC} - {EXACT_HOOKE_ENERGY} (Ha)",
    )


def save_training_curve_figure(
    path: Path,
    training_rows: Sequence[dict[str, Any]],
    *,
    row_label: str,
    col_label: str,
) -> None:
    """Render training energy against step, panelled by the report row axis."""

    series = [
        {
            "panel_key": str(row.get("report_row", "")),
            "line_key": str(row.get("report_col", "")),
            "x": _as_float(row.get("step")),
            "y": _as_float(row.get("energy_mean")),
        }
        for row in training_rows
    ]
    series = [point for point in series if point["x"] is not None and point["y"] is not None]
    if not series:
        plot.save_no_data(path, "training energy")
        return
    plot.save_grouped_line_grid(
        path,
        series,
        x_label="step",
        y_label="train/energy (Ha)",
        title="training energy by report axis",
        panel_keys=sorted({point["panel_key"] for point in series}),
        # `panel_title` is a key -> title callable, not a label string.
        panel_title=lambda key: f"{row_label}={key}",
        legend_title=col_label,
    )


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _markdown_table(columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> list[str]:
    """Return a GitHub-flavoured markdown table, or a placeholder when empty."""

    if not rows:
        return ["_no rows_", ""]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines += ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    lines.append("")
    return lines


def report_markdown(
    *,
    study: str,
    attempt_id: str,
    collect_attempt_id: str,
    row_label: str,
    col_label: str,
    energy_axis_rows: Sequence[dict[str, Any]],
    invariant_rows: Sequence[dict[str, Any]],
    failure_axis_rows: Sequence[dict[str, Any]],
    cost_rows: Sequence[dict[str, Any]],
    run_index_rows: Sequence[dict[str, Any]],
    warnings: Sequence[str],
) -> str:
    """Assemble the report markdown from the reduced tables."""

    below_exact = sum(int(_as_float(row.get("n_below_exact")) or 0) for row in energy_axis_rows)
    lines = [
        f"# {study} final report",
        "",
        f"- report attempt: `{attempt_id}`",
        f"- source `{STAGE_FINAL_COLLECT}` attempt: `{collect_attempt_id}`",
        f"- report axes: rows = `{row_label}`, columns = `{col_label}`",
        f"- final runs collected: {len(run_index_rows)}",
        "",
        "## Primary metric: variational energy",
        "",
        f"`{PRIMARY_METRIC}` against the exact {EXACT_HOOKE_ENERGY} Ha of the Hooke "
        "omega=0.5 singlet. The variational principle bounds this estimator from "
        "below by the exact value, so `energy_error_median` should be positive and "
        "small. `n_below_exact` counts runs that came out below the bound; any "
        "non-zero count is an estimator defect, not a better wavefunction.",
        "",
    ]
    if below_exact:
        lines += [
            f"**{below_exact} run(s) reported an energy below the {EXACT_HOOKE_ENERGY} Ha "
            "variational bound. Treat those cells as invalid until the cause is found.**",
            "",
        ]
    lines += _markdown_table(ENERGY_BY_AXIS_COLUMNS, energy_axis_rows)
    lines += [
        f"`local_energy_var_median` is the secondary metric (`{SECONDARY_METRIC}`): "
        "Var_q[E_L] vanishes iff psi is an eigenstate, for any fixed prior q.",
        "",
        "## Invariants",
        "",
        "Aggregated by worst case across seeds, not by average: an antisymmetry or "
        "equivariance violation on one seed is a violation of the model class.",
        "",
    ]
    lines += _markdown_table(INVARIANTS_BY_AXIS_COLUMNS, invariant_rows)
    lines += ["## Failure modes", ""]
    lines += _markdown_table(FAILURES_BY_AXIS_COLUMNS, failure_axis_rows)
    lines += ["## Cost", ""]
    cost_columns = list(cost_rows[0].keys()) if cost_rows else []
    lines += _markdown_table(cost_columns, cost_rows)
    lines += ["## Figures", ""]
    lines += [f"- `figures/{name}`" for name in REPORT_FIGURES]
    lines.append("")
    if warnings:
        lines += ["## Warnings", ""]
        lines += [f"- {warning}" for warning in warnings]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------
def build_report(
    *,
    results_root: str | Path,
    attempt_id: str | None = None,
    collect_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Render one ``09_final_report`` attempt and return its provenance record."""

    results_root = Path(results_root)
    collect_attempt_id = resolve_final_collect_attempt_id(results_root, collect_attempt_id)
    collect_dir = stage_dir(results_root, STAGE_FINAL_COLLECT) / collect_attempt_id
    if not collect_dir.is_dir():
        raise FileNotFoundError(f"final collect attempt does not exist: {collect_dir}")

    warnings: list[str] = []
    tables: dict[str, list[dict[str, Any]]] = {}
    for name in SOURCE_TABLES:
        path = collect_dir / name
        if not path.is_file():
            warnings.append(f"missing source table: {path}")
        tables[name] = _read_csv(path)

    # The collect stage records which planned axes its report columns hold, so
    # the figures are labelled with real axis names instead of "report_row".
    manifest_path = collect_dir / "manifest.yaml"
    manifest_scalars: dict[str, str] = {}
    if manifest_path.is_file():
        for line in manifest_path.read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and not key.startswith(" ") and value.strip():
                manifest_scalars[key.strip()] = value.strip()
    else:
        warnings.append(f"missing collect manifest: {manifest_path}")
    row_label = manifest_scalars.get("report_row_key") or "report_row"
    col_label = manifest_scalars.get("report_col_key") or "report_col"

    attempt_id = attempt_id or new_attempt_id()
    attempt_dir = stage_dir(results_root, STAGE_FINAL_REPORT) / attempt_id
    (attempt_dir / "tables").mkdir(parents=True, exist_ok=True)
    (attempt_dir / "figures").mkdir(parents=True, exist_ok=True)

    energy_axis_rows = energy_by_axis_rows(tables["energy_by_run.csv"])
    invariant_rows = invariants_by_axis_rows(
        tables["symmetry_summary.csv"], tables["trace_summary.csv"]
    )
    failure_axis_rows = failures_by_axis_rows(tables["failure_modes.csv"])
    cost_rows = tables["cost_by_axis.csv"]

    _write_csv(attempt_dir / "tables" / "energy_by_axis.csv", energy_axis_rows, ENERGY_BY_AXIS_COLUMNS)
    _write_csv(
        attempt_dir / "tables" / "invariants_by_axis.csv", invariant_rows, INVARIANTS_BY_AXIS_COLUMNS
    )
    _write_csv(
        attempt_dir / "tables" / "failures_by_axis.csv", failure_axis_rows, FAILURES_BY_AXIS_COLUMNS
    )
    _write_csv(
        attempt_dir / "tables" / "cost_by_axis.csv",
        cost_rows,
        list(cost_rows[0].keys()) if cost_rows else [],
    )

    save_energy_error_figure(
        attempt_dir / "figures" / "energy_error_heatmap.png",
        energy_axis_rows,
        row_label=row_label,
        col_label=col_label,
    )
    save_training_curve_figure(
        attempt_dir / "figures" / "training_energy_curves.png",
        tables["training_curve_summary.csv"],
        row_label=row_label,
        col_label=col_label,
    )

    study = study_name(manifest_scalars.get("study"))
    markdown = report_markdown(
        study=study,
        attempt_id=attempt_id,
        collect_attempt_id=collect_attempt_id,
        row_label=row_label,
        col_label=col_label,
        energy_axis_rows=energy_axis_rows,
        invariant_rows=invariant_rows,
        failure_axis_rows=failure_axis_rows,
        cost_rows=cost_rows,
        run_index_rows=tables["run_index.csv"],
        warnings=warnings,
    )
    (attempt_dir / "report.md").write_text(markdown)

    record = {
        "study": study,
        "stage": STAGE_FINAL_REPORT,
        "attempt_id": attempt_id,
        "attempt_dir": str(attempt_dir),
        "final_collect_attempt_id": collect_attempt_id,
        "final_collect_attempt_dir": str(collect_dir),
        "report_row_key": row_label,
        "report_col_key": col_label,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "n_final_runs": len(tables["run_index.csv"]),
        "tables": list(REPORT_TABLES),
        "figures": list(REPORT_FIGURES),
        "warnings": warnings,
    }
    write_json(attempt_dir / "final_report.json", record)
    write_latest(stage_dir(results_root, STAGE_FINAL_REPORT), attempt_id)
    return record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse report command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--final-collect-attempt-id",
        default=None,
        help=f"Override source collect attempt; defaults to {STAGE_FINAL_COLLECT}/latest.json.",
    )
    parser.add_argument("--attempt-id", default=None, help="Report attempt id (defaults to now).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the report for the latest collected final lineage."""

    args = parse_args(argv)
    record = build_report(
        results_root=args.results_root,
        attempt_id=args.attempt_id,
        collect_attempt_id=args.final_collect_attempt_id,
    )
    prefix = log_prefix(record["study"])
    print(
        f"{prefix} wrote {STAGE_FINAL_REPORT} attempt {record['attempt_id']} "
        f"from {STAGE_FINAL_COLLECT}/{record['final_collect_attempt_id']} "
        f"-> {record['attempt_dir']}"
    )
    for warning in record["warnings"]:
        print(f"{prefix} warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
