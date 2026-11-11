"""Render final reports from compact ``08_final_collect`` tables.

This stage is intentionally report-oriented and fast. It consumes only compact
tables written by ``final_collect.py``; it does not read raw final-eval task
records, final-train metrics, or checkpoints.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
study_name_from_manifest = _tpen_utils_naming.study_name_from_manifest
_tpen_utils_time = sibling(__file__, 'utils.time')
new_attempt_id = _tpen_utils_time.new_attempt_id
_tpen_stats = sibling(__file__, 'stats')
_as_float = _tpen_stats.as_float
_crop_bar_series_to_weighted_quantiles = _tpen_stats.crop_bar_series_to_weighted_quantiles
_finite_values = _tpen_stats.finite_values
_format_number = _tpen_stats.format_number
_mean = _tpen_stats.mean
_median = _tpen_stats.median
_quantile = _tpen_stats.quantile
_variance = _tpen_stats.variance

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.toolkit.artifacts import (  # noqa: E402
    read_csv as _read_csv,
    write_csv as _write_csv_columns,
)
from experiments.toolkit.virial import derive_virial_metrics

DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
EXACT_HOOKE_ENERGY = 2.0
WINNER_KINDS = ("energy", "stability")
PLOT_WINNER_KINDS = ("energy",)
NARROW_WINNER_HEATMAP_WIDTH_SCALE = 0.75
FULL_GRID_REPORT_AXES = ("basis_update", "feature_normalization")
SYMMETRY_METRICS = (
    "logabs_error_max",
    "logabs_error_median",
    "sign_mismatch_count",
    "parity_mismatch_count",
    "finite_fraction",
)
SYMMETRY_LOG_METRICS = (
    "logabs_error_max",
    "logabs_error_median",
    "sign_mismatch_count",
    "parity_mismatch_count",
)
FEATURE_TRACE_METRICS = (
    "rms_q95",
    "max_abs",
    "nonfinite_count",
)
FEATURE_TRACE_EXCLUDED_LAYERS = {
    "embedding.embedding_normalization",
    "layers.0.feature_normalization",
    "layers.0.update_normalization",
}
ENERGY_COMPONENT_QUANTITIES = (
    ("kinetic", "kinetic_mean"),
    ("harmonic_trap", "harmonic_trap_mean"),
    ("electron_electron", "electron_electron_mean"),
    ("total_energy", "energy_mean"),
    ("virial_residual", "virial_residual"),
    ("virial_relative_residual", "virial_relative_residual"),
)
ENERGY_COMPONENT_COLUMNS = (
    "winner_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "quantity",
    "n",
    "mean",
    "median",
    "min",
    "max",
)
VIRIAL_RESIDUAL_STATS = ("mean", "median", "min", "max")

COMPACT_TABLES = (
    "run_index.csv",
    "architecture_summary.csv",
    "energy_by_run.csv",
    "local_energy_histograms.csv",
    "cusp_profile_summary.csv",
    "tail_profile_summary.csv",
    "stratified_summary.csv",
    "hooke_orbital_summary.csv",
    "symmetry_summary.csv",
    "trace_summary.csv",
    "training_curve_summary.csv",
    "resource_summary.csv",
    "failure_modes.csv",
    "cost_by_run.csv",
    "cost_by_axis.csv",
    "cost_by_task.csv",
)


def _energy_component_value(row: dict[str, Any], key: str) -> float | None:
    value = _as_float(row.get(key))
    if value is None and key in {"virial_residual", "virial_relative_residual"}:
        derived = derive_virial_metrics(
            _as_float(row.get("kinetic_mean")),
            _as_float(row.get("harmonic_trap_mean")),
            _as_float(row.get("electron_electron_mean")),
        )
        return derived["residual" if key == "virial_residual" else "relative_residual"]
    return value


def _winner_id(basis_class: str, normalization: str, winner_kind: str) -> str:
    return _safe_label(f"{basis_class}_{normalization}_{winner_kind}")


def _aggregate_component_row(
    *,
    basis_class: str,
    normalization: str,
    winner_kind: str,
    quantity: str,
    values: Sequence[float],
    axis_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "winner_id": _winner_id(basis_class, normalization, winner_kind),
        "basis_class": basis_class,
        "normalization": normalization,
        "winner_kind": winner_kind,
        "quantity": quantity,
        "n": len(values),
        "mean": _format_number(_mean(values)),
        "median": _format_number(_median(values)),
        "min": _format_number(min(values) if values else None),
        "max": _format_number(max(values) if values else None),
    }
    if axis_values:
        row.update({key: value for key, value in axis_values.items() if key})
    return row


def _energy_component_rows_for_group(
    rows: Sequence[dict[str, Any]],
    *,
    basis_class: str,
    normalization: str,
    winner_kind: str,
    axis_values: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    output = []
    for quantity, key in ENERGY_COMPONENT_QUANTITIES:
        values = _finite_values(_energy_component_value(row, key) for row in rows)
        output.append(
            _aggregate_component_row(
                basis_class=basis_class,
                normalization=normalization,
                winner_kind=winner_kind,
                quantity=quantity,
                values=values,
                axis_values=axis_values,
            )
        )
    return output


def _energy_component_axis_values(row: dict[str, Any], axis_keys: Sequence[str]) -> dict[str, str]:
    """Return report-axis values carried through derived virial tables."""

    axis_values = {}
    for index, key in enumerate(axis_keys):
        fallback = "basis_class" if index == 0 else "normalization"
        value = _row_axis_value(row, key, fallback)
        axis_values[str(key)] = value or "all"
    return axis_values


def _energy_component_tables_by_winner(rows: Sequence[dict[str, Any]], *, axis_keys: Sequence[str] = ()) -> dict[str, list[dict[str, Any]]]:
    """Return validation-style energy-component tables for each winner family."""

    groups: dict[tuple[tuple[str, ...], str, str, str], list[dict[str, Any]]] = defaultdict(list)
    axis_values_by_group: dict[tuple[tuple[str, ...], str, str, str], dict[str, str]] = {}
    for row in rows:
        basis_class = _architecture_label(row)
        normalization = str(row.get("normalization", "")) or "all"
        winner_kind = "energy" if str(row.get("winner_kind", "")).strip() == "energy" else "stability"
        axis_values = _energy_component_axis_values(row, axis_keys)
        axis_tuple = tuple(axis_values.get(str(key), "") for key in axis_keys)
        group_key = (axis_tuple, basis_class, normalization, winner_kind)
        groups[group_key].append(row)
        axis_values_by_group[group_key] = axis_values

    tables = {}
    for group_key, group_rows in sorted(groups.items()):
        _axis_tuple, basis_class, normalization, winner_kind = group_key
        winner_id = _winner_id(basis_class, normalization, winner_kind)
        if winner_id in tables:
            winner_id = _safe_label(f"{winner_id}_{'_'.join(_axis_tuple)}")
        tables[winner_id] = _energy_component_rows_for_group(
            group_rows,
            basis_class=basis_class,
            normalization=normalization,
            winner_kind=winner_kind,
            axis_values=axis_values_by_group[group_key],
        )
    return tables


def _combined_energy_component_rows(rows: Sequence[dict[str, Any]], *, axis_keys: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Return combined energy-component rows in stable winner-id order."""

    by_winner = _energy_component_tables_by_winner(rows, axis_keys=axis_keys)
    return [row for winner_id in sorted(by_winner) for row in by_winner[winner_id]]


def _virial_residual_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return virial-residual summary rows for heatmaps."""

    return [row for row in rows if row.get("quantity") == "virial_residual"]


def _resolve_collect_attempt_id(results_root: Path, requested: str | None) -> str:
    if requested is not None:
        return requested
    collect_stage = stage_dir(results_root, STAGE_FINAL_COLLECT)
    attempt_id = latest_attempt_id(collect_stage)
    if attempt_id is None:
        raise FileNotFoundError(f"no final-collect attempts under {stage_dir(results_root, STAGE_FINAL_COLLECT)}")
    return attempt_id


def _load_collect_manifest(collect_dir: Path) -> dict[str, Any]:
    """Read top-level scalar fields from final_collect's manifest.yaml."""

    manifest_path = collect_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return {}
    manifest: dict[str, Any] = {}
    current_list_key: str | None = None
    for line in manifest_path.read_text().splitlines():
        if line.startswith(" "):
            if current_list_key is not None:
                stripped = line.strip()
                if stripped.startswith("- "):
                    values = manifest.setdefault(current_list_key, [])
                    if isinstance(values, list):
                        values.append(stripped[2:].strip())
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value:
            manifest[key.strip()] = value
        else:
            current_list_key = key.strip()
    return manifest


def _load_collect_tables(
    results_root: Path,
    collect_attempt_id: str,
) -> tuple[Path, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    collect_dir = stage_dir(results_root, STAGE_FINAL_COLLECT) / collect_attempt_id
    if not collect_dir.is_dir():
        raise FileNotFoundError(f"missing final-collect attempt: {collect_dir}")
    return collect_dir, _load_collect_manifest(collect_dir), {name: _read_csv(collect_dir / name) for name in COMPACT_TABLES}


def _column_has_values(rows: Sequence[dict[str, Any]], key: str) -> bool:
    return any(str(row.get(key, "")).strip() for row in rows)


def _manifest_axis_names(collect_manifest: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = collect_manifest.get(key, [])
    if not isinstance(raw, list | tuple):
        return ()
    return tuple(str(axis).strip() for axis in raw if str(axis).strip())


def _report_axis_keys(
    collect_manifest: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
) -> tuple[str, str]:
    """Return the primary and secondary comparison axes for final reporting."""

    summary_rows = tables.get("architecture_summary.csv", [])
    major_axes = _manifest_axis_names(collect_manifest, "major_axes")
    minor_axes = _manifest_axis_names(collect_manifest, "minor_axes")
    configured = (
        str(collect_manifest.get("report_row_key", "")).strip(),
        str(collect_manifest.get("report_col_key", "")).strip(),
    )
    candidates: list[tuple[str | None, str | None]] = []
    if {"basis", "update_normalization", "feature_normalization"}.issubset(set(major_axes)):
        candidates.append(("basis_update", "feature_normalization"))
    elif len(major_axes) >= 2:
        candidates.append((major_axes[0], major_axes[1]))
    elif major_axes and minor_axes:
        candidates.append((major_axes[0], minor_axes[0]))
    candidates.extend(
        [
            configured,
            ("basis_update", "feature_normalization"),
            ("basis", "mechanism"),
            ("basis_class", "normalization"),
        ]
    )
    for row_key, col_key in candidates:
        if row_key and col_key and _column_has_values(summary_rows, row_key) and _column_has_values(summary_rows, col_key):
            return row_key, col_key
    return "basis_class", "normalization"


def _row_axis_value(row: dict[str, Any], key: str, fallback: str) -> str:
    return str(row.get(key, "")).strip() or str(row.get(fallback, "")).strip()


def _architecture_label(row: dict[str, Any]) -> str:
    return str(row.get("basis_update", row.get("basis_class", row.get("architecture", row.get("basis", ""))))) or "all"


def _winner_rows(rows: Sequence[dict[str, Any]], winner_kind: str) -> list[dict[str, Any]]:
    if winner_kind == "energy":
        return [row for row in rows if str(row.get("winner_kind", "")).strip() == "energy"]
    return [row for row in rows if str(row.get("winner_kind", "")).strip() not in {"", "energy"}]


def _winner_title(winner_kind: str) -> str:
    return f"{winner_kind} winners"


def _winner_filename(prefix: str, winner_kind: str, suffix: str) -> str:
    return f"{prefix}_{winner_kind}_winner_{suffix}"


def _save_winner_pair_heatmap(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    title: str,
    transform: str | None = None,
    width_scale: float = 1.0,
) -> None:
    """Save configured winner heatmaps side by side with one scale."""

    plot.save_winner_pair_heatmap(
        path,
        {winner: _winner_rows(rows, winner) for winner in PLOT_WINNER_KINDS},
        row_key=row_key,
        col_key=col_key,
        value_key=value_key,
        title=title,
        panel_titles={winner: _winner_title(winner) for winner in PLOT_WINNER_KINDS},
        transform=transform,
        width_scale=width_scale,
    )


def _figure_label(section: str, index: int) -> str:
    return f"{section}{chr(ord('A') + index)}"


def _safe_label(value: Any) -> str:
    label = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label) or "all"


def _unique_in_order(values: Sequence[Any]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        label = str(value)
        if label == "" or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _figure_1b_splits_basis(row_key: str, col_key: str) -> bool:
    """Return whether 1B expands the merged full-grid basis/update axis."""

    return (row_key, col_key) == FULL_GRID_REPORT_AXES


def _energy_variance_points(rows: Sequence[dict[str, Any]], *, row_key: str, col_key: str) -> list[dict[str, Any]]:
    """Return positive log-log points for the 1B winner scatter."""

    split_basis = _figure_1b_splits_basis(row_key, col_key)
    points = []
    for row in rows:
        energy_error = _as_float(row.get("energy_error"))
        variance = _as_float(row.get("local_energy_var"))
        if energy_error is None or variance is None:
            continue
        abs_error = abs(energy_error)
        if abs_error <= 0.0 or variance <= 0.0:
            continue
        winner_kind = "energy" if str(row.get("winner_kind", "")).strip() == "energy" else "stability"
        primary_axis = _row_axis_value(row, row_key, "basis_class")
        if split_basis and str(row.get("update_normalization", "")).strip():
            primary_axis = str(row["update_normalization"])
        basis = str(row.get("basis", "")).strip() or "all"
        points.append(
            {
                "abs_energy_error": abs_error,
                "local_energy_var": variance,
                "panel_axis": (winner_kind, basis) if split_basis else winner_kind,
                "primary_axis": primary_axis,
                "secondary_axis": _row_axis_value(row, col_key, "normalization"),
                "winner_kind": winner_kind,
                "basis": basis,
            }
        )
    return points


def _save_energy_variance_scatter(path: Path, rows: Sequence[dict[str, Any]], *, row_key: str, col_key: str, title: str) -> None:
    points = [
        point
        for point in _energy_variance_points(rows, row_key=row_key, col_key=col_key)
        if point["winner_kind"] in PLOT_WINNER_KINDS
    ]
    if not points:
        plot.save_no_data(path, title)
        return
    winner_kinds = [
        winner
        for winner in PLOT_WINNER_KINDS
        if any(point["winner_kind"] == winner for point in points)
    ]
    split_basis = _figure_1b_splits_basis(row_key, col_key)
    if split_basis:
        basis_values = sorted({str(point["basis"]) for point in points})
        panel_keys: Sequence[Any] = [
            (winner, basis)
            for winner in winner_kinds
            for basis in basis_values
            if any(point["panel_axis"] == (winner, basis) for point in points)
        ]
        panel_titles: Mapping[Any, str] = {
            panel: f"basis={panel[1]}\n{_winner_title(panel[0])}"
            for panel in panel_keys
        }
        marker_title = "update_normalization"
        layout_description = "Panels split basis choices within winner type"
    else:
        panel_keys = winner_kinds
        panel_titles = {winner: _winner_title(winner) for winner in winner_kinds}
        marker_title = row_key
        layout_description = "Panels separate winner type"
    plot.save_loglog_scatter_grid(
        path,
        points,
        panel_key="panel_axis",
        panel_keys=panel_keys,
        panel_titles=panel_titles,
        x_key="abs_energy_error",
        y_key="local_energy_var",
        color_key="secondary_axis",
        marker_key="primary_axis",
        x_label="abs energy error |E - 2|",
        y_label="local-energy variance",
        title=f"{title}\n{layout_description}; marker shape encodes {marker_title}; color encodes {col_key}.",
        color_title=col_key,
        marker_title=marker_title,
    )


def _local_energy_distribution_groups(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[str], list[str], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Group compact local-energy histograms by normalization and architecture."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        normalization = str(row.get("normalization", ""))
        architecture = _architecture_label(row)
        groups[(normalization, architecture)].append(row)
    normalizations = sorted({key[0] for key in groups})
    architectures = sorted({key[1] for key in groups})
    return normalizations, architectures, groups


def _local_energy_bar_series(rows: Sequence[dict[str, Any]]) -> tuple[list[float], list[float], list[float]]:
    """Aggregate run-level histogram bins for one displayed distribution."""

    counts_by_center: dict[float, float] = defaultdict(float)
    widths_by_center: dict[float, float] = {}
    for row in rows:
        center = _as_float(row.get("bin_center"))
        count = _as_float(row.get("count")) or 0.0
        if center is None or count <= 0.0:
            continue
        left = _as_float(row.get("bin_left"))
        right = _as_float(row.get("bin_right"))
        counts_by_center[center] += count
        widths_by_center[center] = (right - left) if left is not None and right is not None else 1.0
    centers = sorted(counts_by_center)
    counts = [counts_by_center[center] for center in centers]
    widths = [widths_by_center[center] for center in centers]
    return _crop_bar_series_to_weighted_quantiles(centers, counts, widths)


def _save_local_energy_distribution_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    title: str,
    row_axis_label: str,
    col_axis_label: str,
) -> None:
    normalizations, architectures, groups = _local_energy_distribution_groups(rows)
    if not groups:
        plot.save_no_data(path, title)
        return

    bars = []
    for (normalization, architecture), values in groups.items():
        centers, counts, widths = _local_energy_bar_series(values)
        for center, count, width in zip(centers, counts, widths, strict=True):
            bars.append(
                {
                    "panel_key": (normalization, architecture),
                    "x": center,
                    "height": count,
                    "width": width,
                    "color": "#4C78A8",
                }
            )
    if not bars:
        plot.save_no_data(path, title)
        return
    plot.save_grouped_bar_grid(
        path,
        bars,
        row_keys=normalizations,
        col_keys=architectures,
        x_label="local_energy",
        y_label="count",
        title=title,
        figsize=(max(4.0, 3.2 * len(architectures)), max(3.0, 2.4 * len(normalizations))),
        rect=(0.0, 0.0, 1.0, 0.94),
        suptitle_y=0.98,
        bbox_inches=None,
        row_axis_label=row_axis_label,
        col_axis_label=col_axis_label,
    )


def _com_label(raw: Any) -> str:
    value = str(raw).strip()
    return f"CoM {value}" if value else "CoM all"


def _cusp_profile_points(
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    value_key: str,
) -> dict[tuple[str, str, str], list[dict[str, float | int]]]:
    """Return cusp profile points with one line per center of mass.

    Cusp diagnostics evaluate multiple directions for each center of mass. The
    local energy and log amplitude should agree across directions at fixed CoM,
    so Figure 2 collapses directions and seeds into one CoM line. Values are
    means over all compact direction/seed records; error bars show their
    variance.
    """

    cells: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("winner_kind", "")) != winner_kind:
            continue
        r12 = _as_float(row.get("r12"))
        value = _as_float(row.get(value_key))
        if r12 is None or value is None:
            continue
        basis = str(row.get("basis_class", row.get("basis", "")))
        normalization = str(row.get("normalization", ""))
        com = _com_label(row.get("com_id", row.get("center_of_mass_id", "")))
        cells[(basis, normalization, com, r12)].append(value)

    out: dict[tuple[str, str, str], list[dict[str, float | int]]] = defaultdict(list)
    for (basis, normalization, com, r12), values in sorted(cells.items()):
        mean = _mean(values)
        variance = _variance(values)
        if mean is None or variance is None:
            continue
        out[(basis, normalization, com)].append(
            {
                "r12": r12,
                "mean": mean,
                "variance": variance,
                "n_records": len(values),
            }
        )
    return out


def _save_cusp_winner_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    value_key: str,
    y_label: str,
    title: str,
    row_axis_label: str,
    col_axis_label: str,
) -> None:
    """Save one cusp profile metric as a winner-specific architecture grid."""

    profiles = _cusp_profile_points(rows, winner_kind=winner_kind, value_key=value_key)
    if not profiles:
        plot.save_no_data(path, title)
        return

    architectures = sorted({cell[0] for cell in profiles})
    normalizations = sorted({cell[1] for cell in profiles})
    line_labels = sorted({cell[2] for cell in profiles})
    series = [
        {
            "panel_key": (normalization, architecture),
            "line_key": com,
            "x": point["r12"],
            "y": point["mean"],
            "yerr": point["variance"],
            "linewidth": 1.0,
            "marker": "o",
        }
        for (architecture, normalization, com), points in profiles.items()
        for point in points
    ]
    plot.save_grouped_line_grid(
        path,
        series,
        row_keys=normalizations,
        col_keys=architectures,
        line_keys=line_labels,
        x_label="r12",
        y_label=y_label,
        title=f"{title}\nLines are CoM groups; means and variances pool all direction/seed records.",
        legend_title="CoM",
        rect=(0.0, 0.0, 0.86, 0.94),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.1 * len(normalizations))),
        row_axis_label=row_axis_label,
        col_axis_label=col_axis_label,
    )


def _cusp_derivative_profiles(
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
) -> tuple[
    dict[tuple[str, str, str], list[dict[str, float | int]]],
    dict[tuple[str, str, str], list[dict[str, float | int]]],
]:
    """Return seed-aggregated model and target cusp derivative profiles."""

    model_cells: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    target_cells: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        row_winner = "energy" if str(row.get("winner_kind", "")).strip() == "energy" else "stability"
        if row_winner != winner_kind:
            continue
        r12 = _as_float(row.get("r12"))
        value = _as_float(row.get("d_logabs_dr_median"))
        if r12 is None or value is None:
            continue
        architecture = _architecture_label(row)
        normalization = str(row.get("normalization", "")) or "all"
        com = _com_label(row.get("com_id", row.get("center_of_mass_id", "")))
        model_cells[(architecture, normalization, com, r12)].append(value)
        target = _as_float(row.get("target_d_logabs_dr"))
        if target is not None:
            target_cells[(architecture, normalization, com, r12)].append(target)

    def profiles(cells: dict[tuple[str, str, str, float], list[float]]) -> dict[tuple[str, str, str], list[dict[str, float | int]]]:
        out: dict[tuple[str, str, str], list[dict[str, float | int]]] = defaultdict(list)
        for (architecture, normalization, com, r12), values in sorted(cells.items()):
            median = _median(values)
            if median is None:
                continue
            out[(architecture, normalization, com)].append({"r12": r12, "median": median, "n_records": len(values)})
        return out

    return profiles(model_cells), profiles(target_cells)


def _save_cusp_derivative_winner_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    title: str,
    row_axis_label: str,
    col_axis_label: str,
) -> None:
    """Save Figure 2D for one winner type."""

    path.parent.mkdir(parents=True, exist_ok=True)
    model_profiles, target_profiles = _cusp_derivative_profiles(rows, winner_kind=winner_kind)
    if not model_profiles:
        plot.save_no_data(path, title)
        return

    architectures = sorted({key[0] for key in model_profiles})
    normalizations = sorted({key[1] for key in model_profiles})
    com_labels = sorted({key[2] for key in model_profiles})
    plt = plot.pyplot()
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(index % cmap.N) for index, label in enumerate(com_labels)}
    fig, axes = plt.subplots(
        len(normalizations),
        len(architectures),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    legend_handles: dict[str, Any] = {}
    target_handle = None
    for row_index, normalization in enumerate(normalizations):
        for col_index, architecture in enumerate(architectures):
            ax = axes[row_index][col_index]
            plotted = False
            for com in com_labels:
                points = sorted(model_profiles.get((architecture, normalization, com), []), key=lambda item: float(item["r12"]))
                if not points:
                    continue
                (line,) = ax.plot(
                    [float(point["r12"]) for point in points],
                    [float(point["median"]) for point in points],
                    linewidth=1.1,
                    marker="o",
                    markersize=2.8,
                    color=colors[com],
                    label=com,
                )
                legend_handles.setdefault(com, line)
                target_points = sorted(target_profiles.get((architecture, normalization, com), []), key=lambda item: float(item["r12"]))
                if target_points:
                    (target_line,) = ax.plot(
                        [float(point["r12"]) for point in target_points],
                        [float(point["median"]) for point in target_points],
                        linewidth=1.0,
                        linestyle="--",
                        color=colors[com],
                        alpha=0.7,
                    )
                    target_handle = target_handle or target_line
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            if row_index == 0:
                ax.set_title(architecture, fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{normalization}\nd logabs / dr")
            if row_index == len(normalizations) - 1:
                ax.set_xlabel("r12")
            ax.grid(True, linewidth=0.35, alpha=0.35)
    if legend_handles:
        handles = list(legend_handles.values())
        labels = list(legend_handles.keys())
        if target_handle is not None:
            handles.append(target_handle)
            labels.append("target")
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.99, 0.5),
            fontsize=7,
            title="CoM",
            title_fontsize=8,
            borderaxespad=0.0,
        )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 0.86 if legend_handles else 1.0, 0.92))
    plot.add_major_axis_labels(fig, axes, row_label=row_axis_label, col_label=col_axis_label, col_position="top")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _tail_seed_profile_points(
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    value_key: str,
) -> dict[tuple[str, str, str], list[dict[str, float | int]]]:
    """Return tail profile points with seed variance for each CoM line.

    Tail records may contain multiple radial paths for the same CoM, radius, and
    seed. Those paths are averaged inside a seed first; the plotted uncertainty
    is then the variance of those seed-level values.
    """

    per_seed: dict[tuple[str, str, str, float, str], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("winner_kind", "")) != winner_kind:
            continue
        radius = _as_float(row.get("radius"))
        value = _as_float(row.get(value_key))
        if radius is None or value is None:
            continue
        basis = str(row.get("basis_class", row.get("basis", "")))
        normalization = str(row.get("normalization", ""))
        com = _com_label(row.get("com_id", row.get("center_of_mass_id", "")))
        seed = str(row.get("seed_index", row.get("final_run_id", "")))
        per_seed[(basis, normalization, com, radius, seed)].append(value)

    by_radius: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for (basis, normalization, com, radius, _seed), values in per_seed.items():
        mean = _mean(values)
        if mean is not None:
            by_radius[(basis, normalization, com, radius)].append(mean)

    out: dict[tuple[str, str, str], list[dict[str, float | int]]] = defaultdict(list)
    for (basis, normalization, com, radius), seed_values in sorted(by_radius.items()):
        mean = _mean(seed_values)
        variance = _variance(seed_values)
        if mean is None or variance is None:
            continue
        out[(basis, normalization, com)].append(
            {
                "radius": radius,
                "mean": mean,
                "variance": variance,
                "n_seeds": len(seed_values),
            }
        )
    return out


def _tail_local_energy_bar_points(
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
) -> dict[tuple[str, str, str], list[dict[str, float | int]]]:
    """Return tail local-energy bar points with q5-q85 ranges."""

    cells: dict[tuple[str, str, str, float], list[tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        if str(row.get("winner_kind", "")) != winner_kind:
            continue
        radius = _as_float(row.get("radius"))
        median = _as_float(row.get("local_energy_median"))
        if radius is None or median is None:
            continue
        low = _as_float(row.get("local_energy_q05"))
        if low is None:
            low = _as_float(row.get("local_energy_q25"))
        high = _as_float(row.get("local_energy_q85"))
        if high is None:
            high = _as_float(row.get("local_energy_q75"))
        low = median if low is None else low
        high = median if high is None else high
        basis = str(row.get("basis_class", row.get("basis", "")))
        normalization = str(row.get("normalization", ""))
        com = _com_label(row.get("com_id", row.get("center_of_mass_id", "")))
        cells[(basis, normalization, com, radius)].append((median, low, high))

    out: dict[tuple[str, str, str], list[dict[str, float | int]]] = defaultdict(list)
    for (basis, normalization, com, radius), values in sorted(cells.items()):
        medians = [value[0] for value in values]
        lows = [value[1] for value in values]
        highs = [value[2] for value in values]
        median = _median(medians)
        low = _median(lows)
        high = _median(highs)
        if median is None or low is None or high is None:
            continue
        out[(basis, normalization, com)].append(
            {
                "radius": radius,
                "median": median,
                "low": min(low, median),
                "high": max(high, median),
                "n_records": len(values),
            }
        )
    return out


def _save_tail_local_energy_line_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    title: str,
    row_axis_label: str,
    col_axis_label: str,
) -> None:
    """Save tail local-energy medians as CoM lines with q5-q85 ranges."""

    profiles = _tail_local_energy_bar_points(rows, winner_kind=winner_kind)
    if not profiles:
        plot.save_no_data(path, title)
        return

    architectures = sorted({cell[0] for cell in profiles})
    normalizations = sorted({cell[1] for cell in profiles})
    com_labels = sorted({cell[2] for cell in profiles})

    series = [
        {
            "panel_key": (normalization, architecture),
            "line_key": com,
            "x": point["radius"],
            "y": point["median"],
            "yerr_low": max(0.0, float(point["median"]) - float(point["low"])),
            "yerr_high": max(0.0, float(point["high"]) - float(point["median"])),
            "marker": "o",
        }
        for (architecture, normalization, com), points in profiles.items()
        for point in points
    ]
    plot.save_grouped_line_grid(
        path,
        series,
        row_keys=normalizations,
        col_keys=architectures,
        line_keys=com_labels,
        x_label="radius",
        y_label="local energy",
        title=f"{title}\nLines are CoM groups; points show median local energy and error bars show q5-q85.",
        legend_title="CoM",
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
        rect=(0.0, 0.0, 0.88, 0.94),
        row_axis_label=row_axis_label,
        col_axis_label=col_axis_label,
    )


def _save_tail_logabs_line_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    title: str,
    row_axis_label: str,
    col_axis_label: str,
) -> None:
    """Save tail logabs profiles as lines."""

    profiles = _tail_seed_profile_points(rows, winner_kind=winner_kind, value_key="logabs_median")
    if not profiles:
        plot.save_no_data(path, title)
        return

    architectures = sorted({cell[0] for cell in profiles})
    normalizations = sorted({cell[1] for cell in profiles})
    com_labels = sorted({cell[2] for cell in profiles})
    series = [
        {
            "panel_key": (normalization, architecture),
            "line_key": com,
            "x": point["radius"],
            "y": point["mean"],
            "yerr": point["variance"],
            "marker": "o",
        }
        for (architecture, normalization, com), points in profiles.items()
        for point in points
    ]
    plot.save_grouped_line_grid(
        path,
        series,
        row_keys=normalizations,
        col_keys=architectures,
        line_keys=com_labels,
        x_label="radius",
        y_label="logabs",
        title=f"{title}\nLines are CoM groups; error bars are seed variance over final replicates.",
        legend_title="CoM",
        rect=(0.0, 0.0, 0.88, 0.94),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
        row_axis_label=row_axis_label,
        col_axis_label=col_axis_label,
    )


def _group_label(row: dict[str, Any], group_keys: Sequence[str]) -> str:
    parts = [str(row.get(key, "")) for key in group_keys if row.get(key, "") != ""]
    return "/".join(parts) if parts else "all"


def _save_line_plot(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    group_keys: Sequence[str],
    title: str,
    legend: str = "auto",
    legend_title: str | None = None,
) -> None:
    series = []
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        series.append({"line_key": _group_label(row, group_keys), "x": x, "y": y})
    if not series:
        plot.save_no_data(path, title)
        return
    plot.save_grouped_line_plot(
        path,
        series,
        x_label=x_key,
        y_label=y_key,
        title=title,
        legend=legend,
        legend_title=legend_title,
    )


def _save_architecture_line_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    group_keys: Sequence[str],
    title: str,
    legend_title: str,
) -> None:
    series = []
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        architecture = _architecture_label(row)
        series.append(
            {
                "panel_key": architecture,
                "line_key": _group_label(row, group_keys),
                "x": x,
                "y": y,
            }
        )
    if not series:
        plot.save_no_data(path, title)
        return

    architectures = sorted({str(row["panel_key"]) for row in series})
    labels = sorted({str(row["line_key"]) for row in series})
    plot.save_grouped_line_grid(
        path,
        series,
        panel_keys=architectures,
        panel_title=lambda key: str(key),
        line_keys=labels,
        x_label=x_key,
        y_label=y_key,
        title=title,
        legend_title=legend_title,
        rect=(0.0, 0.0, 0.83, 0.94),
    )


def _save_architecture_normalization_line_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    group_keys: Sequence[str],
    title: str,
    legend_title: str,
    row_axis_label: str,
    col_axis_label: str,
) -> None:
    """Save a line grid with normalization rows and architecture columns."""

    series = []
    for row in rows:
        x = _as_float(row.get(x_key))
        y = _as_float(row.get(y_key))
        if x is None or y is None:
            continue
        architecture = _architecture_label(row)
        normalization = str(row.get("normalization", "")) or "all"
        series.append(
            {
                "panel_key": (normalization, architecture),
                "line_key": _group_label(row, group_keys),
                "x": x,
                "y": y,
            }
        )
    if not series:
        plot.save_no_data(path, title)
        return

    x_label = x_key.replace("_", " ")
    y_label = y_key.replace("_", " ")
    architectures = sorted({str(row["panel_key"][1]) for row in series})
    normalizations = sorted({str(row["panel_key"][0]) for row in series})
    labels = sorted({str(row["line_key"]) for row in series})
    plot.save_grouped_line_grid(
        path,
        series,
        row_keys=normalizations,
        col_keys=architectures,
        line_keys=labels,
        x_label=x_label,
        y_label=y_label,
        title=title,
        legend_title=legend_title,
        rect=(0.0, 0.0, 0.84, 0.94),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
        row_axis_label=row_axis_label,
        col_axis_label=col_axis_label,
    )


def _training_curve_value(row: dict[str, Any], value_mode: str) -> float | None:
    energy = _as_float(row.get("energy_mean"))
    if energy is None:
        return None
    if value_mode == "abs_energy_error":
        return abs(energy - EXACT_HOOKE_ENERGY)
    if value_mode == "energy_mean":
        return energy
    raise ValueError(f"unknown training curve value mode: {value_mode}")


def _smooth_training_points(points: Sequence[dict[str, Any]], *, window: int) -> list[dict[str, Any]]:
    """Return a centered rolling mean for one training run."""

    sorted_points = sorted(points, key=lambda point: float(point["step"]))
    if window <= 1 or len(sorted_points) <= 1:
        return [dict(point) for point in sorted_points]
    radius = max(1, window // 2)
    smoothed = []
    for index, point in enumerate(sorted_points):
        low = max(0, index - radius)
        high = min(len(sorted_points), index + radius + 1)
        value = _mean([float(item["value"]) for item in sorted_points[low:high]])
        if value is not None:
            smoothed.append({**point, "value": value})
    return smoothed


def _training_run_curves(
    rows: Sequence[dict[str, Any]],
    *,
    value_mode: str = "energy_mean",
    smooth_window: int = 25,
) -> dict[tuple[str, str, str, str], list[dict[str, float | str]]]:
    """Return one smoothed curve per final-training run."""

    per_run_step: dict[tuple[str, str, str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        step = _as_float(row.get("step"))
        value = _training_curve_value(row, value_mode)
        if step is None or value is None:
            continue
        architecture = _architecture_label(row)
        normalization = str(row.get("normalization", "")) or "all"
        winner = "energy" if str(row.get("winner_kind", "")).strip() == "energy" else "stability"
        run_id = str(row.get("final_run_id", "")) or f"seed-{row.get('seed_index', '')}"
        per_run_step[(architecture, normalization, winner, run_id, step)].append(value)

    raw_curves: dict[tuple[str, str, str, str], list[dict[str, float | str]]] = defaultdict(list)
    for (architecture, normalization, winner, run_id, step), values in sorted(per_run_step.items()):
        value = _mean(values)
        if value is not None:
            raw_curves[(architecture, normalization, winner, run_id)].append({"step": step, "value": value, "run_id": run_id})
    return {
        key: _smooth_training_points(points, window=smooth_window)
        for key, points in raw_curves.items()
    }


def _save_training_curve_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    winner_kind: str,
    value_mode: str,
    y_label: str,
    title: str,
    semilogy: bool = False,
    smooth_window: int = 25,
    row_axis_label: str | None = None,
    col_axis_label: str | None = None,
) -> None:
    """Save one winner family's final-training curves."""

    curves = _training_run_curves(_winner_rows(rows, winner_kind), value_mode=value_mode, smooth_window=smooth_window)
    curves = {key: points for key, points in curves.items() if key[2] == winner_kind}
    if not curves:
        plot.save_no_data(path, title)
        return

    architectures = sorted({key[0] for key in curves})
    normalizations = sorted({key[1] for key in curves})
    runs_by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    series = []
    for (architecture, normalization, _winner, run_id), points in curves.items():
        runs_by_cell[(normalization, architecture)].append(run_id)
        for point in points:
            series.append(
                {
                    "panel_key": (normalization, architecture),
                    "line_key": run_id,
                    "x": point["step"],
                    "y": point["value"],
                    "linewidth": 0.9,
                    "alpha": 0.45,
                    "marker": "",
                }
            )
    panel_notes = {
        key: f"n={len(run_ids)}"
        for key, run_ids in runs_by_cell.items()
    }
    plot.save_grouped_line_grid(
        path,
        series,
        row_keys=normalizations,
        col_keys=architectures,
        x_label="step",
        y_label=y_label,
        title=f"{title}\nEach line is one final-training run; curves use a {smooth_window}-point centered rolling mean.",
        legend_title=None,
        show_legend=False,
        sharey=semilogy,
        yscale="log" if semilogy else None,
        panel_notes=panel_notes,
        rect=(0.0, 0.0, 1.0, 0.94),
        figsize=(max(5.0, 3.1 * len(architectures)), max(3.2, 2.2 * len(normalizations))),
        row_axis_label=row_axis_label,
        col_axis_label=col_axis_label,
    )


def _save_symmetry_metric_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    row_key: str = "basis_class",
    col_key: str = "normalization",
) -> None:
    """Save one symmetry metric as architecture-by-normalization heatmaps."""

    symmetries = _unique_in_order(row.get("symmetry_task", "") for row in rows)
    if not symmetries:
        plot.save_no_data(path, title)
        return
    panel_rows = {}
    for symmetry in symmetries:
        symmetry_rows = [row for row in rows if str(row.get("symmetry_task", "")) == symmetry]
        for winner in PLOT_WINNER_KINDS:
            panel_rows[(symmetry, winner)] = _winner_rows(symmetry_rows, winner)
    plot.save_row_scoped_heatmap_grid(
        path,
        panel_rows,
        row_labels=symmetries,
        col_labels=PLOT_WINNER_KINDS,
        row_key=row_key,
        col_key=col_key,
        value_key=metric_key,
        title=title,
        panel_title=lambda symmetry, winner: f"{symmetry}\n{_winner_title(winner)}",
        transform="positive_log" if metric_key in SYMMETRY_LOG_METRICS else None,
        figsize=(max(7.0, 7.0 * len(PLOT_WINNER_KINDS)), max(3.2, 3.0 * len(symmetries))),
        subplot_adjust={"left": 0.07, "right": 0.89, "bottom": 0.08, "top": 0.90, "wspace": 0.55, "hspace": 0.65},
    )


def _save_feature_trace_metric_grid(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    row_key: str = "basis_class",
    col_key: str = "normalization",
) -> None:
    """Save one feature-trace metric as layer-by-winner heatmaps."""

    trace_rows = [
        row
        for row in rows
        if str(row.get("trace_kind", "")) == "feature_trace_stability"
        and str(row.get("layer", "")) not in FEATURE_TRACE_EXCLUDED_LAYERS
    ]
    layers = _unique_in_order(row.get("layer", "") for row in trace_rows)
    if not layers:
        plot.save_no_data(path, title)
        return

    panel_rows = {}
    for layer in layers:
        layer_rows = [row for row in trace_rows if str(row.get("layer", "")) == layer]
        for winner in PLOT_WINNER_KINDS:
            panel_rows[(layer, winner)] = _winner_rows(layer_rows, winner)
    plot.save_row_scoped_heatmap_grid(
        path,
        panel_rows,
        row_labels=layers,
        col_labels=PLOT_WINNER_KINDS,
        row_key=row_key,
        col_key=col_key,
        value_key=metric_key,
        title=title,
        panel_title=lambda layer, winner: f"{layer}\n{_winner_title(winner)}",
        figsize=(max(5.5, 5.5 * len(PLOT_WINNER_KINDS)), max(3.2, 2.55 * len(layers))),
        subplot_adjust={"left": 0.08, "right": 0.89, "bottom": 0.04, "top": 0.94, "wspace": 0.65, "hspace": 0.75},
    )


def _save_virial_residual_heatmap(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    stat: str,
    row_key: str = "basis_class",
    col_key: str = "normalization",
) -> None:
    """Save one signed-log virial-residual winner-pair heatmap."""

    _save_winner_pair_heatmap(
        path,
        rows,
        row_key=row_key,
        col_key=col_key,
        value_key=stat,
        title=f"Virial residual {stat}",
        transform="signed_log",
        width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE,
    )


def _write_figures(
    figures_dir: Path,
    tables: dict[str, list[dict[str, Any]]],
    collect_manifest: dict[str, Any],
) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    row_key, col_key = _report_axis_keys(collect_manifest, tables)
    grid_row_axis_label = col_key
    grid_col_axis_label = row_key
    architecture_rows = tables["architecture_summary.csv"]
    energy = tables["energy_by_run.csv"]
    energy_components = _combined_energy_component_rows(energy, axis_keys=(row_key, col_key))
    virial_residual = _virial_residual_rows(energy_components)
    histograms = tables["local_energy_histograms.csv"]
    stratified = tables["stratified_summary.csv"]

    def add(filename: str, writer: Any) -> None:
        writer(figures_dir / filename)
        written.append(filename)

    add(
        "1A_real_scale_energy_error_heatmap.png",
        lambda path: _save_winner_pair_heatmap(path, architecture_rows, row_key=row_key, col_key=col_key, value_key="energy_error_median", title="Median signed final energy error", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
    )
    add(
        "1A_log_scale_energy_error_heatmap.png",
        lambda path: _save_winner_pair_heatmap(path, architecture_rows, row_key=row_key, col_key=col_key, value_key="energy_error_median", title="Median signed final energy error", transform="signed_log", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
    )
    for winner in PLOT_WINNER_KINDS:
        add(
            _winner_filename("1C", winner, "local_energy_distribution_grid.png"),
            lambda path, winner=winner: _save_local_energy_distribution_grid(path, _winner_rows(histograms, winner), title=f"MCMC local-energy histograms: {_winner_title(winner)}", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )

    add("1B_energy_error_vs_local_energy_variance.png", lambda path: _save_energy_variance_scatter(path, energy, row_key=row_key, col_key=col_key, title="Absolute energy error vs local-energy variance"))
    cusp_rows = tables["cusp_profile_summary.csv"]
    for winner in PLOT_WINNER_KINDS:
        add(
            _winner_filename("2A", winner, "cusp_local_energy_grid.png"),
            lambda path, winner=winner: _save_cusp_winner_grid(path, cusp_rows, winner_kind=winner, value_key="local_energy_median", y_label="local energy", title=f"Cusp local energy profiles: {_winner_title(winner)}", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )
        add(
            _winner_filename("2B", winner, "cusp_logabs_grid.png"),
            lambda path, winner=winner: _save_cusp_winner_grid(path, cusp_rows, winner_kind=winner, value_key="logabs_median", y_label="logabs", title=f"Cusp logabs profiles: {_winner_title(winner)}", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )
        add(
            _winner_filename("2C", winner, "cusp_finite_fraction_grid.png"),
            lambda path, winner=winner: _save_cusp_winner_grid(path, cusp_rows, winner_kind=winner, value_key="finite_fraction", y_label="finite fraction", title=f"Cusp finite fraction profiles: {_winner_title(winner)}", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )
    for winner in PLOT_WINNER_KINDS:
        add(
            _winner_filename("2D", winner, "cusp_dlogabs_dr_grid.png"),
            lambda path, winner=winner: _save_cusp_derivative_winner_grid(path, cusp_rows, winner_kind=winner, title=f"Cusp derivative profiles: {_winner_title(winner)}", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )
    tail_labels = {"energy": ("3A", "3C"), "stability": ("3B", "3D")}
    for winner in PLOT_WINNER_KINDS:
        local_energy_label, _logabs_label = tail_labels[winner]
        add(
            _winner_filename(local_energy_label, winner, "tail_local_energy_lines.png"),
            lambda path, winner=winner: _save_tail_local_energy_line_grid(path, tables["tail_profile_summary.csv"], winner_kind=winner, title=f"Tail local energy: {_winner_title(winner)}", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )
    for winner in PLOT_WINNER_KINDS:
        _local_energy_label, logabs_label = tail_labels[winner]
        add(
            _winner_filename(logabs_label, winner, "tail_logabs_grid.png"),
            lambda path, winner=winner: _save_tail_logabs_line_grid(path, tables["tail_profile_summary.csv"], winner_kind=winner, title=f"Tail logabs: {_winner_title(winner)}", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )

    add(
        "3E_tail_outlier_heatmap.png",
        lambda path: _save_winner_pair_heatmap(path, architecture_rows, row_key=row_key, col_key=col_key, value_key="tail_outlier_fraction_median", title="Tail outlier fraction", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
    )

    aggregate = [row for row in stratified if row.get("stratum") == "all"]
    add(
        f"{_figure_label('4', 0)}_stratified_geometry_aggregate_heatmap.png",
        lambda path: _save_winner_pair_heatmap(path, aggregate, row_key=row_key, col_key=col_key, value_key="median_abs_energy_error", title="Stratified median absolute energy error", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
    )
    add(
        f"{_figure_label('4', 0)}_stratified_geometry_aggregate_log_heatmap.png",
        lambda path: _save_winner_pair_heatmap(path, aggregate, row_key=row_key, col_key=col_key, value_key="median_abs_energy_error", title="Stratified median absolute energy error", transform="positive_log", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
    )

    hooke_rows = tables["hooke_orbital_summary.csv"]
    for winner in PLOT_WINNER_KINDS:
        rows = _winner_rows(hooke_rows, winner)
        add(
            _winner_filename("5A", winner, "hooke_orbital_local_energy_distribution.png"),
            lambda path, rows=rows, winner=winner: _save_architecture_normalization_line_grid(path, rows, x_key="r12_center", y_key="local_energy_median", group_keys=("com_bin",), title=f"Hooke-orbital local-energy medians: {_winner_title(winner)}", legend_title="CoM bin", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )
        add(
            _winner_filename("5B", winner, "hooke_orbital_local_energy_vs_r12.png"),
            lambda path, rows=rows, winner=winner: _save_architecture_normalization_line_grid(path, rows, x_key="r12_center", y_key="local_energy_median", group_keys=("com_bin",), title=f"Hooke-orbital local energy vs r12: {_winner_title(winner)}", legend_title="CoM bin", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )
        add(
            _winner_filename("5C", winner, "hooke_orbital_local_energy_vs_radius.png"),
            lambda path, rows=rows, winner=winner: _save_architecture_normalization_line_grid(path, rows, x_key="R_norm_center", y_key="local_energy_median", group_keys=("r12_bin",), title=f"Hooke-orbital local energy vs CoM radius: {_winner_title(winner)}", legend_title="r12 bin", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )

    for metric_index, metric in enumerate(SYMMETRY_METRICS):
        add(
            f"{_figure_label('6', metric_index)}_symmetry_{metric}_heatmap_grid.png",
            lambda path, metric=metric: _save_symmetry_metric_grid(path, tables["symmetry_summary.csv"], metric_key=metric, title=f"Symmetry diagnostic: {metric}", row_key=row_key, col_key=col_key),
        )

    for metric_index, metric in enumerate(FEATURE_TRACE_METRICS):
        add(
            f"{_figure_label('7', metric_index)}_feature_trace_{metric}_heatmap_grid.png",
            lambda path, metric=metric: _save_feature_trace_metric_grid(path, tables["trace_summary.csv"], metric_key=metric, title=f"Feature-trace stability diagnostic: {metric}", row_key=row_key, col_key=col_key),
        )

    training_labels = {"energy": ("8A", "8B"), "stability": ("8C", "8D")}
    for winner in PLOT_WINNER_KINDS:
        energy_label, error_label = training_labels[winner]
        add(
            _winner_filename(energy_label, winner, "training_energy.png"),
            lambda path, winner=winner: _save_training_curve_grid(path, tables["training_curve_summary.csv"], winner_kind=winner, value_mode="energy_mean", y_label="energy mean", title=f"Final-train energy curves: {_winner_title(winner)}", row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )
        add(
            _winner_filename(error_label, winner, "abs_energy_error_semilogy.png"),
            lambda path, winner=winner: _save_training_curve_grid(path, tables["training_curve_summary.csv"], winner_kind=winner, value_mode="abs_energy_error", y_label="abs energy error |E - 2|", title=f"Final-train absolute energy error: {_winner_title(winner)}", semilogy=True, row_axis_label=grid_row_axis_label, col_axis_label=grid_col_axis_label),
        )

    for stat_index, stat in enumerate(VIRIAL_RESIDUAL_STATS):
        add(
            f"{_figure_label('9', stat_index)}_virial_residual_{stat}_log_heatmap.png",
            lambda path, stat=stat: _save_virial_residual_heatmap(path, virial_residual, stat=stat, row_key=row_key, col_key=col_key),
        )

    strata = sorted({str(row.get("stratum", "")) for row in stratified if row.get("stratum", "") not in {"", "all"}})
    for stratum_index, stratum in enumerate(strata, start=1):
        label = _figure_label("4", stratum_index)
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stratum)
        rows = [row for row in stratified if str(row.get("stratum", "")) == stratum]
        add(
            f"{label}_stratified_geometry_{safe}_heatmap.png",
            lambda path, rows=rows, stratum=stratum: _save_winner_pair_heatmap(path, rows, row_key=row_key, col_key=col_key, value_key="median_abs_energy_error", title=f"Stratified median absolute energy error: {stratum}", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
        )
        add(
            f"{label}_stratified_geometry_{safe}_log_heatmap.png",
            lambda path, rows=rows, stratum=stratum: _save_winner_pair_heatmap(path, rows, row_key=row_key, col_key=col_key, value_key="median_abs_energy_error", title=f"Stratified median absolute energy error: {stratum}", transform="positive_log", width_scale=NARROW_WINNER_HEATMAP_WIDTH_SCALE),
        )
    return written


def _copy_tables(collect_dir: Path, tables_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in tables.items():
        shutil.copyfile(collect_dir / name, tables_dir / name)
        counts[name] = len(rows)
    return counts


def _energy_component_columns(axis_keys: Sequence[str]) -> list[str]:
    """Return energy-component table columns including configured report axes."""

    columns = list(ENERGY_COMPONENT_COLUMNS)
    insert_at = columns.index("basis_class")
    for key in reversed([str(key) for key in axis_keys if str(key) and str(key) not in columns]):
        columns.insert(insert_at, key)
    return columns


def _write_energy_component_tables(tables_dir: Path, energy_rows: Sequence[dict[str, Any]], *, axis_keys: Sequence[str] = ()) -> dict[str, int]:
    """Write combined and per-winner virial tables from final MCMC energy rows."""

    by_winner = _energy_component_tables_by_winner(energy_rows, axis_keys=axis_keys)
    combined = [row for winner_id in sorted(by_winner) for row in by_winner[winner_id]]
    counts = {"energy_components_and_virial_by_winner.csv": len(combined)}
    columns = _energy_component_columns(axis_keys)
    _write_csv_columns(
        tables_dir / "energy_components_and_virial_by_winner.csv",
        combined,
        columns,
    )
    for winner_id, rows in by_winner.items():
        relative_path = f"energy_components_and_virial/{winner_id}.csv"
        _write_csv_columns(tables_dir / relative_path, rows, columns)
        counts[relative_path] = len(rows)
    return counts


def build_report(
    *,
    results_root: str | Path,
    report_attempt_id: str | None = None,
    final_collect_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Write ``09_final_report`` artifacts from compact collect outputs."""

    results_root = Path(results_root)
    final_collect_attempt_id = _resolve_collect_attempt_id(results_root, final_collect_attempt_id)
    report_attempt_id = report_attempt_id or final_collect_attempt_id or new_attempt_id()
    report_dir = stage_dir(results_root, STAGE_FINAL_REPORT) / report_attempt_id
    tables_dir = report_dir / "tables"
    figures_dir = report_dir / "figures"
    collect_dir, collect_manifest, tables = _load_collect_tables(results_root, final_collect_attempt_id)
    study = study_name_from_manifest(collect_manifest)
    table_counts = _copy_tables(collect_dir, tables_dir, tables)
    report_row_key, report_col_key = _report_axis_keys(collect_manifest, tables)
    table_counts.update(_write_energy_component_tables(tables_dir, tables["energy_by_run.csv"], axis_keys=(report_row_key, report_col_key)))
    figures = _write_figures(figures_dir, tables, collect_manifest)

    report = {
        "study": study,
        "stage": STAGE_FINAL_REPORT,
        "attempt_id": report_attempt_id,
        "final_collect_attempt_id": final_collect_attempt_id,
        "final_collect_dir": str(collect_dir),
        "report_axes": {
            "row": report_row_key,
            "column": report_col_key,
        },
        "tables": table_counts,
        "figures": figures,
        "caveats": [
            "final_report.py consumes 08_final_collect compact tables only.",
            "Raw final-eval and final-train artifacts are reduced by final_collect.py.",
            "Runtime/resource summaries are reported separately from model-quality ranking.",
        ],
    }
    write_json(report_dir / "final_report.json", report)
    (report_dir / "report.md").write_text(_report_markdown(report, tables))
    write_latest(stage_dir(results_root, STAGE_FINAL_REPORT), report_attempt_id)
    return {"attempt_dir": str(report_dir), "report": report}


def _report_markdown(report: dict[str, Any], tables: dict[str, list[dict[str, Any]]]) -> str:
    report_axes = report.get("report_axes", {}) if isinstance(report.get("report_axes"), dict) else {}
    row_key = str(report_axes.get("row", "basis_class"))
    col_key = str(report_axes.get("column", "normalization"))
    architecture = sorted(
        tables["architecture_summary.csv"],
        key=lambda row: (
            _row_axis_value(row, row_key, "basis_class"),
            _row_axis_value(row, col_key, "normalization"),
            row.get("winner_kind", ""),
        ),
    )
    report_title = str(report.get("study") or "Study").replace("_", " ").title()
    winner_scope = "energy winners" if tuple(PLOT_WINNER_KINDS) == ("energy",) else "energy and stability winners"
    winner_columns = "energy-winner columns" if tuple(PLOT_WINNER_KINDS) == ("energy",) else "energy/stability winner columns"
    winner_heatmaps = "energy winners" if tuple(PLOT_WINNER_KINDS) == ("energy",) else "energy and stability winners side by side"
    figure_1b_layout = (
        "Figure 1B uses basis panels, marker shape for `update_normalization`, and color for `feature_normalization`."
        if _figure_1b_splits_basis(row_key, col_key)
        else f"Figure 1B separates winner types into panels, uses marker shape for `{row_key}`, and color for `{col_key}`."
    )
    lines = [
        f"# {report_title} Final Report",
        "",
        "## Scope And Provenance",
        "",
        "This report consumes `08_final_collect` compact tables only. It does not parse raw model, train, or eval records.",
        f"Final collect attempt: `{report['final_collect_attempt_id']}`.",
        f"Report axes: `{row_key}` by `{col_key}`.",
        "",
        "## Final Champion Summary",
        "",
        "See `tables/run_index.csv` and `tables/energy_by_run.csv`.",
        "",
        "## Family-Level Ranking",
        "",
    ]
    if architecture:
        lines.extend([f"| {row_key} | {col_key} | winner_kind | n_success/n_expected | energy_error_median | local_energy_var_median |", "|---|---|---|---:|---:|---:|"])
        for row in architecture:
            lines.append(
                f"| {_row_axis_value(row, row_key, 'basis_class')} | {_row_axis_value(row, col_key, 'normalization')} | {row.get('winner_kind', '')} | "
                f"{row.get('n_success', '')}/{row.get('n_expected', '')} | {row.get('energy_error_median', '')} | {row.get('local_energy_var_median', '')} |"
            )
    else:
        lines.append("No architecture summary rows were found.")
    lines.extend(
        [
            "",
            "## Energy And Local-Energy Results",
            "",
            f"Energy figures use signed error relative to exact Hooke energy `E = 2`; heatmaps render {winner_heatmaps} with a shared color scale over `{row_key}` by `{col_key}`. {figure_1b_layout} Signed-log heatmap variants use real-scale cell labels.",
            "",
            "Energy component and virial tables are written to `tables/energy_components_and_virial_by_winner.csv` and to one validation-style table per winner family under `tables/energy_components_and_virial/`. The virial residual is `2 * kinetic - 2 * harmonic_trap + electron_electron`; the relative residual divides its absolute value by the absolute component scale.",
            "",
            "## Cusp Diagnostics",
            "",
            f"Cusp tables preserve center-of-mass and direction columns when present. Figures 2A/2B/2C emit {winner_scope} grids for sampled local-energy, log-amplitude, and finite-fraction profiles against `r12`; directions and seeds are pooled so each subplot has one line per CoM and variance error bars aggregate all compact direction/seed records. Figure 2D emits {winner_scope} grids for `d_logabs_dr_median` against `r12`, with `{col_key}` rows, `{row_key}` columns, solid CoM model lines, and dashed target derivative references.",
            "",
            "## Tail Diagnostics",
            "",
            f"Tail tables preserve path columns. Tail local-energy figures show line grids for {winner_scope}; lines are grouped by CoM and their error bars show q5-q85. Tail logabs figures use line grids with seed-variance error bars. Figure 3E summarizes tail outlier fraction.",
            "",
            "## Stratified Geometry Diagnostics",
            "",
            f"Stratified summaries include per-stratum rows and `stratum=all` aggregate rows. Figures 4A, 4B, ... group the aggregate and per-stratum heatmaps; each group has a real-scale and positive-log-color version with {winner_heatmaps} on a shared color scale.",
            "",
            "## Hooke-Orbital Diagnostics",
            "",
            f"Hooke-orbital summaries are binned by CoM-radius and `r12` bins. Figure 5 line plots are emitted for {winner_scope}, with `{col_key}` rows, `{row_key}` columns, and the remaining bin dimension in the external legend.",
            "",
            "## Symmetry Diagnostics",
            "",
            f"See `tables/symmetry_summary.csv` and the symmetry figures. Figures 6A, 6B, ... emit one heatmap-grid figure per scalar symmetry metric; each grid uses symmetry tasks as rows, {winner_columns}, `{row_key}` by `{col_key}` inside each subplot, and one ticked shared color scale per symmetry-task row. Error and count metric heatmaps use explicit monochrome log color.",
            "",
            "## Trace Diagnostics",
            "",
            f"See `tables/trace_summary.csv` and the trace figures. Figures 7A, 7B, and 7C focus on feature-trace stability and emit one heatmap-grid figure each for `rms_q95`, `max_abs`, and `nonfinite_count`; each grid uses layer rows, {winner_columns}, `{row_key}` by `{col_key}` inside each subplot, and one ticked colorbar with its own scale for every layer row.",
            "",
            "## Training And Resource Summary",
            "",
            f"See `tables/training_curve_summary.csv` and `tables/resource_summary.csv`. Runtime is not mixed into quality ranking. Figure 8 shows one 25-point centered-smoothed training-energy curve per final-training run for {winner_scope}, plus the corresponding semilogy absolute energy error curves with one shared vertical axis per grid.",
            "",
            "## Virial Diagnostics",
            "",
            f"See `tables/energy_components_and_virial_by_winner.csv`. Figures 9A, 9B, 9C, and 9D show virial-residual mean, median, minimum, and maximum. Heatmaps render {winner_heatmaps} with signed-log color scales; cell labels remain on the real signed scale.",
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in report["caveats"]:
        lines.append(f"- {caveat}")
    lines.extend(["", "## Next-Scan Implications", "", "Use energy, local-energy variance, pathology, cusp/tail, symmetry, and trace summaries jointly.", "", "## Tables And Figures", "", "Tables:"])
    for name, n_rows in report["tables"].items():
        lines.append(f"- `tables/{name}`: {n_rows} rows")
    lines.append("")
    lines.append("Figures:")
    for name in report["figures"]:
        lines.append(f"- `figures/{name}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-report arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--final-collect-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--smoke", action="store_true", help="Deprecated; use configs/smoke.yaml with the normal stack.")
    args = parser.parse_args(argv)
    if args.smoke:
        parser.error("use --grid experiments/hooke/pair_stability_v3/configs/smoke.yaml with the normal stage stack")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Write final report artifacts."""

    args = parse_args(argv)
    prefix = log_prefix()
    print(f"{prefix} final report results_root={args.results_root}")
    if args.final_collect_attempt_id:
        print(f"{prefix} final report using final_collect_attempt_id={args.final_collect_attempt_id}")
    else:
        print(f"{prefix} final report using latest final-collect attempt")
    result = build_report(
        results_root=args.results_root,
        report_attempt_id=args.attempt_id,
        final_collect_attempt_id=args.final_collect_attempt_id,
    )
    report = result["report"]
    prefix = log_prefix(report.get("study"))
    print(
        f"{prefix} final report consumed 08_final_collect/{report['final_collect_attempt_id']} "
        f"-> {result['attempt_dir']}"
    )
    print(f"{prefix} final report copied table rows:")
    for filename, count in report["tables"].items():
        print(f"{prefix}   {filename}: {count}")
    print(f"{prefix} final report wrote {len(report['figures'])} figures")
    figure_counts: dict[str, int] = {}
    for figure in report["figures"]:
        section = figure.split("_", 1)[0]
        figure_counts[section] = figure_counts.get(section, 0) + 1
    for section, count in sorted(figure_counts.items()):
        print(f"{prefix}   figures {section}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
