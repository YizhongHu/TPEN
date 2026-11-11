"""Reusable plotting primitives for staged reports.

The helpers in this module are intentionally study-local. Callers prepare small row dictionaries and
domain labels, while this module owns Matplotlib setup and rendering mechanics.

Reduced port of the v3 module: this study's report renders one single-panel
heatmap family and one grouped-line family, so the v3 paired heatmap, row-scoped
heatmap grid, grouped bar grid, and log-log scatter savers are not carried over.
Every domain axis still arrives as a caller-supplied key string; nothing here
knows a metric name or an experiment axis name.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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

_tpen_stats = sibling(__file__, 'stats')
_as_float = _tpen_stats.as_float
_mean = _tpen_stats.mean

POSITIVE_HEATMAP_CMAP = "Reds"
MAJOR_AXIS_LABEL_PAD_POINTS = 6.0
LEGEND_TITLE_WRAP_COLUMNS = 18


def _flatten_axes(axes: Any) -> list[Any]:
    """Return a flat list of visible Matplotlib axes."""

    if axes is None:
        return []
    if hasattr(axes, "ravel"):
        flat = list(axes.ravel())
    elif isinstance(axes, Sequence) and not hasattr(axes, "get_position"):
        flat = []
        for axis in axes:
            flat.extend(_flatten_axes(axis))
    else:
        flat = [axes]
    return [axis for axis in flat if hasattr(axis, "get_position") and axis.get_visible()]


def _points_to_figure_fraction(fig: Any, points: float, *, axis: str) -> float:
    """Convert point spacing to figure coordinates."""

    size = fig.get_size_inches()[0 if axis == "x" else 1]
    return (points / 72.0) / size


def _bbox_limits(boxes: Sequence[Any]) -> tuple[float, float, float, float]:
    """Return left, right, bottom, top limits for figure-coordinate boxes."""

    return (
        min(box.x0 for box in boxes),
        max(box.x1 for box in boxes),
        min(box.y0 for box in boxes),
        max(box.y1 for box in boxes),
    )


def _legend_title(title: str | None) -> str | None:
    """Return a compact legend title without hiding long parameter names."""

    if title is None:
        return None
    label = str(title)
    if len(label) <= LEGEND_TITLE_WRAP_COLUMNS:
        return label
    parts = label.split("_")
    if len(parts) <= 1:
        return label
    pivot = len(parts) // 2
    return f"{'_'.join(parts[:pivot])}_\n{'_'.join(parts[pivot:])}"


def add_major_axis_labels(
    fig: Any,
    axes: Any,
    *,
    row_label: str | None,
    col_label: str | None,
    col_position: str = "bottom",
    fontsize: int = 9,
    pad_points: float = MAJOR_AXIS_LABEL_PAD_POINTS,
    clamp_to_figure: bool = False,
) -> None:
    """Add figure-level labels for row/column parameter axes."""

    visible_axes = _flatten_axes(axes)
    if not visible_axes:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_boxes = [axis.get_position() for axis in visible_axes]
    tight_boxes = [
        tight_box.transformed(fig.transFigure.inverted())
        for axis in visible_axes
        if (tight_box := axis.get_tightbbox(renderer)) is not None
    ]
    left, right, bottom, top = _bbox_limits(axis_boxes)
    tight_left, _tight_right, tight_bottom, tight_top = _bbox_limits(tight_boxes or axis_boxes)
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (bottom + top)
    x_pad = _points_to_figure_fraction(fig, pad_points, axis="x")
    y_pad = _points_to_figure_fraction(fig, pad_points, axis="y")
    if row_label:
        row_x = tight_left - x_pad
        if clamp_to_figure:
            row_x = max(0.012, row_x)
        fig.text(
            row_x,
            center_y,
            str(row_label),
            rotation=90,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
        )
    if col_label:
        if col_position == "top":
            y = tight_top + y_pad
            va = "bottom"
        else:
            y = tight_bottom - y_pad
            va = "top"
        if clamp_to_figure:
            y = min(0.985, max(0.012, y))
        fig.text(
            center_x,
            y,
            str(col_label),
            ha="center",
            va=va,
            fontsize=fontsize,
            fontweight="bold",
        )


def pyplot():
    """Return Matplotlib pyplot configured for headless report rendering."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rhu/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_no_data(path: Path, title: str) -> None:
    """Save a placeholder figure for an empty report section."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def heatmap_matrix(
    rows: Sequence[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
) -> tuple[list[str], list[str], list[list[float | None]]]:
    """Return real-scale heatmap cell means for plotting and annotations."""

    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _as_float(row.get(value_key))
        if value is None:
            continue
        cells[(str(row.get(row_key, "")), str(row.get(col_key, "")))].append(value)
    if not cells:
        return [], [], []
    y_labels = sorted({key[0] for key in cells})
    x_labels = sorted({key[1] for key in cells})
    matrix = []
    for y_label in y_labels:
        row_values = []
        for x_label in x_labels:
            row_values.append(_mean(cells.get((y_label, x_label), [])))
        matrix.append(row_values)
    return y_labels, x_labels, matrix


def matrix_values(matrix: Sequence[Sequence[float | None]]) -> list[float]:
    """Return finite real-scale values from a heatmap matrix."""

    return [value for row in matrix for value in row if value is not None]


def heatmap_colorbar_label(value_key: str, transform: str | None) -> str:
    """Return a colorbar label that records any non-linear color transform."""

    if transform == "signed_log":
        return f"{value_key} (symmetric log color; labels are real scale)"
    if transform == "positive_log":
        return f"{value_key} (monochrome log color; labels are real scale)"
    if transform == "positive_linear":
        return f"{value_key} (monochrome color; labels are real scale)"
    return value_key


def resolve_heatmap_transform(values: Sequence[float], requested: str | None) -> str:
    """Choose a heatmap color scale from finite shared values."""

    if requested is not None:
        return requested
    finite = [value for value in values if math.isfinite(value)]
    if finite and min(finite) >= 0.0:
        positive = [value for value in finite if value > 0.0]
        if positive and max(positive) / min(positive) >= 10.0:
            return "positive_log"
        return "positive_linear"
    return "signed_linear"


def draw_heatmap_axis(
    fig: Any,
    ax: Any,
    *,
    y_labels: Sequence[str],
    x_labels: Sequence[str],
    matrix: Sequence[Sequence[float | None]],
    value_key: str,
    title: str,
    transform: str | None,
    scale_values: Sequence[float] | None = None,
    add_colorbar: bool = True,
) -> Any | None:
    """Draw one heatmap axis with real-scale annotations."""

    from matplotlib.colors import LogNorm, SymLogNorm

    finite_values = matrix_values(matrix)
    if not finite_values:
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(title)
        return None

    scale = [value for value in (scale_values or finite_values) if math.isfinite(value)]
    if not scale:
        scale = finite_values
    vmax = max(abs(value) for value in scale)
    vmax = vmax if vmax > 0.0 else 1.0
    resolved_transform = resolve_heatmap_transform(scale, transform)
    data = [[math.nan if value is None else value for value in row] for row in matrix]
    if resolved_transform == "signed_log":
        nonzero = [abs(value) for value in scale if value != 0.0]
        norm = SymLogNorm(linthresh=min(nonzero), vmin=-vmax, vmax=vmax, base=10) if nonzero else None
        image = ax.imshow(data, cmap="coolwarm", norm=norm, aspect="auto")
    elif resolved_transform == "positive_log":
        positive = [value for value in scale if value > 0.0]
        if positive:
            vmin = min(positive)
            positive_data = [[math.nan if value is None else max(value, vmin) for value in row] for row in matrix]
            positive_vmax = max(positive)
            if positive_vmax <= vmin:
                positive_vmax = vmin * 1.000001
            image = ax.imshow(positive_data, cmap=POSITIVE_HEATMAP_CMAP, norm=LogNorm(vmin=vmin, vmax=positive_vmax, clip=True), aspect="auto")
        else:
            image = ax.imshow(data, cmap=POSITIVE_HEATMAP_CMAP, vmin=0.0, vmax=vmax, aspect="auto")
    elif resolved_transform == "positive_linear":
        image = ax.imshow(data, cmap=POSITIVE_HEATMAP_CMAP, vmin=0.0, vmax=vmax, aspect="auto")
    else:
        image = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_title(title)
    for y_index, row in enumerate(matrix):
        for x_index, value in enumerate(row):
            if value is not None:
                ax.text(x_index, y_index, f"{value:.2g}", ha="center", va="center", fontsize=8)
    if add_colorbar:
        fig.colorbar(image, ax=ax, label=heatmap_colorbar_label(value_key, resolved_transform), fraction=0.046, pad=0.04)
    return image


def save_heatmap(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    title: str,
    transform: str | None = None,
) -> None:
    """Save one aggregated heatmap from row dictionaries."""

    path.parent.mkdir(parents=True, exist_ok=True)
    y_labels, x_labels, matrix = heatmap_matrix(rows, row_key=row_key, col_key=col_key, value_key=value_key)
    if not matrix:
        save_no_data(path, title)
        return

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(x_labels)), max(3.5, 0.8 * len(y_labels))))
    draw_heatmap_axis(
        fig,
        ax,
        y_labels=y_labels,
        x_labels=x_labels,
        matrix=matrix,
        value_key=value_key,
        title=title,
        transform=transform,
    )
    fig.tight_layout()
    add_major_axis_labels(fig, ax, row_label=row_key, col_label=col_key)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def add_shared_legend(
    fig: Any,
    handles: Sequence[Any],
    labels: Sequence[str],
    *,
    title: str | None,
    bbox_to_anchor: tuple[float, float] = (1.0, 0.5),
    fontsize: int = 6,
    title_fontsize: int = 7,
) -> None:
    """Place one shared legend beside a figure."""

    if handles:
        fig.legend(
            handles,
            labels,
            title=title,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
            loc="center left",
            bbox_to_anchor=bbox_to_anchor,
            borderaxespad=0.0,
            ncol=max(1, math.ceil(len(labels) / 28)),
        )


def _line_keys(series: Sequence[dict[str, Any]], requested: Sequence[str] | None) -> list[str]:
    if requested is not None:
        return list(requested)
    return sorted({str(row.get("line_key", "")) for row in series if str(row.get("line_key", "")) != ""})


def _series_points(rows: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
    points = []
    for row in rows:
        x = _as_float(row.get("x"))
        y = _as_float(row.get("y"))
        if x is None or y is None:
            continue
        point = {"x": x, "y": y}
        yerr = _as_float(row.get("yerr"))
        if yerr is not None:
            point["yerr"] = yerr
        yerr_low = _as_float(row.get("yerr_low"))
        yerr_high = _as_float(row.get("yerr_high"))
        if yerr_low is not None:
            point["yerr_low"] = yerr_low
        if yerr_high is not None:
            point["yerr_high"] = yerr_high
        points.append(point)
    return sorted(points, key=lambda point: point["x"])


def save_grouped_line_plot(
    path: Path,
    series: Sequence[dict[str, Any]],
    *,
    x_label: str,
    y_label: str,
    title: str,
    legend: str = "auto",
    legend_title: str | None = None,
) -> None:
    """Save a single-panel grouped line plot."""

    prepared = [{**row, "panel_key": "panel"} for row in series]
    save_grouped_line_grid(
        path,
        prepared,
        panel_keys=["panel"],
        panel_title=lambda _key: "",
        x_label=x_label,
        y_label=y_label,
        title=title,
        legend_title=legend_title,
        show_legend=legend != "none",
        legend_outside=legend == "outside",
        single_panel=True,
    )


def save_grouped_line_grid(
    path: Path,
    series: Sequence[dict[str, Any]],
    *,
    x_label: str,
    y_label: str,
    title: str,
    row_keys: Sequence[str] | None = None,
    col_keys: Sequence[str] | None = None,
    panel_keys: Sequence[Any] | None = None,
    panel_title: Callable[[Any], str] | None = None,
    line_keys: Sequence[str] | None = None,
    legend_title: str | None = None,
    show_legend: bool = True,
    legend_outside: bool = True,
    sharex: bool = True,
    sharey: bool = False,
    yscale: str | None = None,
    panel_notes: Mapping[Any, str] | None = None,
    figsize: tuple[float, float] | None = None,
    rect: tuple[float, float, float, float] | None = None,
    suptitle_y: float = 0.995,
    single_panel: bool = False,
    row_axis_label: str | None = None,
    col_axis_label: str | None = None,
    col_axis_label_position: str = "top",
) -> None:
    """Save grouped lines in either an auto grid or a row/column grid."""

    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in series:
        panel_key = row.get("panel_key")
        line_key = str(row.get("line_key", ""))
        if panel_key is None or line_key == "":
            continue
        if _as_float(row.get("x")) is None or _as_float(row.get("y")) is None:
            continue
        groups[(panel_key, line_key)].append(row)
    if not groups:
        save_no_data(path, title)
        return

    labels = _line_keys(series, line_keys)
    plt = pyplot()
    from matplotlib.lines import Line2D

    cmap = plt.get_cmap("tab20" if len(labels) > 10 else "tab10")
    colors = {label: cmap(index % cmap.N) for index, label in enumerate(labels)}

    if row_keys is not None and col_keys is not None:
        n_rows = len(row_keys)
        n_cols = len(col_keys)
        panel_grid = [[(row_key, col_key) for col_key in col_keys] for row_key in row_keys]
        default_figsize = (max(5.0, 3.1 * n_cols), max(3.2, 2.2 * n_rows))
    else:
        panel_keys = list(panel_keys or sorted({key[0] for key in groups}, key=str))
        n_cols = 1 if single_panel else min(3, max(1, math.ceil(math.sqrt(len(panel_keys)))))
        n_rows = math.ceil(len(panel_keys) / n_cols)
        padded = list(panel_keys) + [None] * (n_rows * n_cols - len(panel_keys))
        panel_grid = [padded[index * n_cols:(index + 1) * n_cols] for index in range(n_rows)]
        default_figsize = (7.0, 4.0) if single_panel else (max(5.0, 3.4 * n_cols), max(3.0, 2.6 * n_rows))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or default_figsize,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )
    for row_index, panel_row in enumerate(panel_grid):
        for col_index, key in enumerate(panel_row):
            ax = axes[row_index][col_index]
            if key is None:
                ax.axis("off")
                continue
            if yscale is not None:
                ax.set_yscale(yscale)
            plotted = False
            for label in labels:
                points = _series_points(groups.get((key, label), []))
                if yscale == "log":
                    points = [point for point in points if point["y"] > 0.0]
                if not points:
                    continue
                style_rows = groups.get((key, label), [])
                linestyle = str(style_rows[0].get("linestyle", "-")) if style_rows else "-"
                alpha = _as_float(style_rows[0].get("alpha")) if style_rows else None
                linewidth = _as_float(style_rows[0].get("linewidth")) if style_rows else None
                marker = str(style_rows[0].get("marker", "o")) if style_rows else "o"
                yerr_low = [point.get("yerr_low") for point in points]
                yerr_high = [point.get("yerr_high") for point in points]
                yerr = [point.get("yerr") for point in points]
                if any(value is not None for value in yerr_low + yerr_high):
                    yerr_arg = [
                        [0.0 if value is None else value for value in yerr_low],
                        [0.0 if value is None else value for value in yerr_high],
                    ]
                elif any(value is not None for value in yerr):
                    yerr_arg = [0.0 if value is None else value for value in yerr]
                else:
                    yerr_arg = None
                if yerr_arg is not None:
                    ax.errorbar(
                        [point["x"] for point in points],
                        [point["y"] for point in points],
                        yerr=yerr_arg,
                        marker=marker,
                        linewidth=linewidth or 1.1,
                        markersize=3.0,
                        capsize=2.0,
                        color=colors[label],
                        linestyle=linestyle,
                        alpha=alpha if alpha is not None else 1.0,
                        label=label,
                    )
                else:
                    ax.plot(
                        [point["x"] for point in points],
                        [point["y"] for point in points],
                        marker=marker,
                        linewidth=linewidth or 1.1,
                        markersize=3.0,
                        color=colors[label],
                        linestyle=linestyle,
                        alpha=alpha if alpha is not None else 1.0,
                        label=label,
                    )
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            elif panel_notes and key in panel_notes:
                ax.text(0.97, 0.94, panel_notes[key], ha="right", va="top", transform=ax.transAxes, fontsize=7)
            if row_keys is not None and col_keys is not None:
                row_label, col_label = key
                if row_index == 0:
                    ax.set_title(str(col_label), fontsize=9)
                if col_index == 0:
                    ax.set_ylabel(f"{row_label}\n{y_label}")
                if row_index == n_rows - 1:
                    ax.set_xlabel(x_label)
            else:
                ax.set_title(panel_title(key) if panel_title is not None else str(key), fontsize=9)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
            ax.grid(True, linewidth=0.35, alpha=0.35)

    handles = [
        Line2D([0], [0], marker="o", color=colors[label], linewidth=1.1, markersize=3.0, label=label)
        for label in labels
    ]
    if show_legend and handles:
        if legend_outside:
            add_shared_legend(fig, handles, labels, title=legend_title)
        elif len(handles) <= 12:
            axes[0][0].legend(handles=handles, labels=labels, fontsize=7, loc="best")
    fig.suptitle(title, y=suptitle_y)
    if rect is None:
        rect = (0.0, 0.0, 0.84 if show_legend and legend_outside and handles else 1.0, 0.94)
    fig.tight_layout(rect=rect)
    if row_keys is not None and col_keys is not None:
        add_major_axis_labels(
            fig,
            axes,
            row_label=row_axis_label,
            col_label=col_axis_label,
            col_position=col_axis_label_position,
        )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


