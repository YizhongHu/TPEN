"""Reusable plotting primitives for staged reports.

The helpers in this module are intentionally study-local. Callers prepare small row dictionaries and
domain labels, while this module owns Matplotlib setup and rendering mechanics.
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
NARROW_LOG_AXIS_MAX_INTERVALS = 4
HOLLOW_SCATTER_MARKERS = ("o", "s", "^", "v", "<", ">", "D", "d", "p", "h", "H", "8")
SCATTER_MARKER_SIZE = 58
SCATTER_MARKER_LINEWIDTH = 1.2


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


def _configure_narrow_log_ticks(ax: Any, *, axis: str) -> None:
    """Use sparse scalar ticks when a log axis spans less than one decade."""

    from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter

    lower, upper = ax.get_xlim() if axis == "x" else ax.get_ylim()
    if lower <= 0.0 or upper <= lower or upper / lower >= 10.0:
        return
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(MaxNLocator(nbins=NARROW_LOG_AXIS_MAX_INTERVALS, prune="both"))
    target.set_major_formatter(ScalarFormatter())
    target.set_minor_locator(NullLocator())


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


def save_winner_pair_heatmap(
    path: Path,
    panel_rows: Mapping[str, Sequence[dict[str, Any]]],
    *,
    row_key: str,
    col_key: str,
    value_key: str,
    title: str,
    panel_titles: Mapping[str, str] | None = None,
    transform: str | None = None,
    width_scale: float = 1.0,
) -> None:
    """Save two or more peer heatmaps side by side with one shared scale."""

    path.parent.mkdir(parents=True, exist_ok=True)
    matrices = {}
    scale_values = []
    max_x = 0
    max_y = 0
    panel_keys = list(panel_rows)
    for panel_key in panel_keys:
        y_labels, x_labels, matrix = heatmap_matrix(
            panel_rows[panel_key],
            row_key=row_key,
            col_key=col_key,
            value_key=value_key,
        )
        matrices[panel_key] = (y_labels, x_labels, matrix)
        scale_values.extend(matrix_values(matrix))
        max_x = max(max_x, len(x_labels))
        max_y = max(max_y, len(y_labels))
    if not scale_values:
        save_no_data(path, title)
        return

    plt = pyplot()
    width = max(7.0, 2.6 * max_x * len(panel_keys) * width_scale)
    fig, axes = plt.subplots(1, len(panel_keys), figsize=(width, max(3.5, 0.8 * max_y)), squeeze=False)
    images = []
    for col_index, panel_key in enumerate(panel_keys):
        y_labels, x_labels, matrix = matrices[panel_key]
        image = draw_heatmap_axis(
            fig,
            axes[0][col_index],
            y_labels=y_labels,
            x_labels=x_labels,
            matrix=matrix,
            value_key=value_key,
            title=(panel_titles or {}).get(panel_key, panel_key),
            transform=transform,
            scale_values=scale_values,
            add_colorbar=False,
        )
        if image is not None:
            images.append(image)
        if col_index > 0:
            axes[0][col_index].set_yticklabels([])
    if images:
        fig.colorbar(images[0], ax=list(axes.ravel()), label=heatmap_colorbar_label(value_key, transform), fraction=0.046, pad=0.04)
    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.16, top=0.84, wspace=0.45)
    add_major_axis_labels(fig, axes, row_label=row_key, col_label=col_key)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_row_scoped_heatmap_grid(
    path: Path,
    panel_rows: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    row_key: str,
    col_key: str,
    value_key: str,
    title: str,
    panel_title: Callable[[str, str], str] | None = None,
    transform: str | None = None,
    colorbar_ticks: str = "default",
    figsize: tuple[float, float] | None = None,
    subplot_adjust: Mapping[str, float] | None = None,
) -> None:
    """Save a heatmap grid with an independent shared color scale per row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    matrices = {}
    scale_values_by_row: dict[str, list[float]] = defaultdict(list)
    for row_label in row_labels:
        for col_label in col_labels:
            y_labels, x_labels, matrix = heatmap_matrix(
                panel_rows.get((row_label, col_label), []),
                row_key=row_key,
                col_key=col_key,
                value_key=value_key,
            )
            matrices[(row_label, col_label)] = (y_labels, x_labels, matrix)
            scale_values_by_row[row_label].extend(matrix_values(matrix))
    if not any(scale_values_by_row.values()):
        save_no_data(path, title)
        return

    plt = pyplot()
    fig, axes = plt.subplots(
        len(row_labels),
        len(col_labels),
        figsize=figsize or (max(11.0, 5.5 * len(col_labels)), max(3.2, 2.55 * len(row_labels))),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for row_index, row_label in enumerate(row_labels):
        row_scale_values = scale_values_by_row[row_label]
        row_images = []
        for col_index, col_label in enumerate(col_labels):
            ax = axes[row_index][col_index]
            y_labels, x_labels, matrix = matrices[(row_label, col_label)]
            image = draw_heatmap_axis(
                fig,
                ax,
                y_labels=y_labels,
                x_labels=x_labels,
                matrix=matrix,
                value_key=value_key,
                title=panel_title(row_label, col_label) if panel_title is not None else f"{row_label}\n{col_label}",
                transform=transform,
                scale_values=row_scale_values,
                add_colorbar=False,
            )
            if image is not None:
                row_images.append(image)
            if col_index > 0:
                ax.set_yticklabels([])
        if row_images:
            colorbar = fig.colorbar(
                row_images[0],
                ax=list(axes[row_index]),
                fraction=0.035,
                pad=0.035,
            )
            if colorbar_ticks == "none":
                colorbar.set_ticks([])
    fig.suptitle(title, y=0.995)
    if subplot_adjust is None:
        subplot_adjust = {"left": 0.08, "right": 0.89, "bottom": 0.04, "top": 0.94, "wspace": 0.65, "hspace": 0.75}
    fig.subplots_adjust(**subplot_adjust)
    add_major_axis_labels(fig, axes, row_label=row_key, col_label=col_key)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_subplot_grid(
    *,
    n_rows: int,
    n_cols: int,
    figsize: tuple[float, float],
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[Any, Any]:
    """Create a squeezed-off Matplotlib subplot grid."""

    plt = pyplot()
    return plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)


def hide_unused_axes(axes: Sequence[Any], *, start: int) -> None:
    """Hide flat subplot axes after ``start``."""

    for ax in axes[start:]:
        ax.axis("off")


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


def save_grouped_bar_grid(
    path: Path,
    bars: Sequence[dict[str, Any]],
    *,
    x_label: str,
    y_label: str,
    title: str,
    row_keys: Sequence[str],
    col_keys: Sequence[str],
    bar_keys: Sequence[str] | None = None,
    legend_title: str | None = None,
    sharex: bool = True,
    sharey: bool = False,
    figsize: tuple[float, float] | None = None,
    rect: tuple[float, float, float, float] | None = None,
    suptitle_y: float = 0.995,
    bbox_inches: str | None = "tight",
    row_axis_label: str | None = None,
    col_axis_label: str | None = None,
    col_axis_label_position: str = "top",
) -> None:
    """Save grouped bars in a row/column subplot grid."""

    path.parent.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[tuple[str, str], str], list[dict[str, Any]]] = defaultdict(list)
    for row in bars:
        panel_key = row.get("panel_key")
        if not isinstance(panel_key, tuple) or len(panel_key) != 2:
            continue
        key = str(row.get("bar_key", ""))
        if _as_float(row.get("x")) is None or _as_float(row.get("height")) is None:
            continue
        groups[(panel_key, key)].append(row)
    if not groups:
        save_no_data(path, title)
        return

    labels = list(bar_keys) if bar_keys is not None else sorted({key[1] for key in groups if key[1] != ""})
    use_legend = bool(labels)
    if not labels:
        labels = [""]
    plt = pyplot()
    from matplotlib.patches import Patch

    cmap = plt.get_cmap("tab20" if len(labels) > 10 else "tab10")
    colors = {label: cmap(index % cmap.N) for index, label in enumerate(labels)}
    fig, axes = plt.subplots(
        len(row_keys),
        len(col_keys),
        figsize=figsize or (max(5.0, 3.1 * len(col_keys)), max(3.2, 2.2 * len(row_keys))),
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
    )
    offsets = {
        label: index - (len(labels) - 1) / 2.0
        for index, label in enumerate(labels)
    }
    for row_index, row_key in enumerate(row_keys):
        for col_index, col_key in enumerate(col_keys):
            ax = axes[row_index][col_index]
            panel_key = (row_key, col_key)
            plotted = False
            for label in labels:
                rows = sorted(groups.get((panel_key, label), []), key=lambda item: float(item["x"]))
                if not rows:
                    continue
                xs = []
                heights = []
                widths = []
                yerr_low = []
                yerr_high = []
                for item in rows:
                    width = _as_float(item.get("width")) or 0.8
                    x = float(item["x"]) + offsets[label] * width
                    height = float(item["height"])
                    low = _as_float(item.get("yerr_low"))
                    high = _as_float(item.get("yerr_high"))
                    xs.append(x)
                    heights.append(height)
                    widths.append(width)
                    yerr_low.append(0.0 if low is None else low)
                    yerr_high.append(0.0 if high is None else high)
                yerr = [yerr_low, yerr_high] if any(value > 0.0 for value in yerr_low + yerr_high) else None
                color = rows[0].get("color", colors[label])
                ax.bar(
                    xs,
                    heights,
                    width=widths,
                    yerr=yerr,
                    capsize=2.0 if yerr is not None else 0.0,
                    color=color,
                    edgecolor="black",
                    linewidth=0.35,
                    alpha=0.85,
                    label=label,
                )
                plotted = True
            if not plotted:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
            if row_index == 0:
                ax.set_title(str(col_key), fontsize=9)
            if col_index == 0:
                ax.set_ylabel(f"{row_key}\n{y_label}")
            if row_index == len(row_keys) - 1:
                ax.set_xlabel(x_label)
            ax.grid(True, axis="y", linewidth=0.35, alpha=0.35)
    if use_legend:
        handles = [Patch(facecolor=colors[label], edgecolor="black", label=label) for label in labels]
        add_shared_legend(fig, handles, labels, title=legend_title, bbox_to_anchor=(0.99, 0.5), fontsize=7, title_fontsize=8)
    fig.suptitle(title, y=suptitle_y)
    if rect is None:
        rect = (0.0, 0.0, 0.88 if use_legend else 1.0, 0.94)
    fig.tight_layout(rect=rect)
    add_major_axis_labels(
        fig,
        axes,
        row_label=row_axis_label,
        col_label=col_axis_label,
        col_position=col_axis_label_position,
        clamp_to_figure=bbox_inches != "tight",
    )
    fig.savefig(path, dpi=160, bbox_inches=bbox_inches)
    plt.close(fig)


def save_loglog_scatter_grid(
    path: Path,
    points: Sequence[dict[str, Any]],
    *,
    panel_key: str,
    panel_keys: Sequence[Any],
    panel_titles: Mapping[Any, str],
    x_key: str,
    y_key: str,
    color_key: str,
    marker_key: str,
    x_label: str,
    y_label: str,
    title: str,
    color_title: str,
    marker_title: str,
) -> None:
    """Save log-log scatter panels with separate color and marker legends."""

    path.parent.mkdir(parents=True, exist_ok=True)
    clean_points = [
        point
        for point in points
        if _as_float(point.get(x_key)) is not None and _as_float(point.get(y_key)) is not None
    ]
    if not clean_points:
        save_no_data(path, title)
        return

    plt = pyplot()
    from matplotlib.lines import Line2D

    color_values = sorted({str(point.get(color_key, "")) for point in clean_points})
    marker_values = sorted({str(point.get(marker_key, "")) for point in clean_points})
    cmap = plt.get_cmap("tab20" if len(color_values) > 10 else "tab10")
    colors = {value: cmap(index % cmap.N) for index, value in enumerate(color_values)}
    marker_by_value = {
        value: HOLLOW_SCATTER_MARKERS[index % len(HOLLOW_SCATTER_MARKERS)]
        for index, value in enumerate(marker_values)
    }

    fig, axes = plt.subplots(1, len(panel_keys), figsize=(max(6.0, 3.7 * len(panel_keys)), 4.8), sharex=True, sharey=True)
    if len(panel_keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, panel_keys, strict=True):
        panel_points = [point for point in clean_points if point.get(panel_key) == key]
        for point in panel_points:
            ax.scatter(
                float(point[x_key]),
                float(point[y_key]),
                marker=marker_by_value[str(point.get(marker_key, ""))],
                s=SCATTER_MARKER_SIZE,
                facecolors="none",
                edgecolors=colors[str(point.get(color_key, ""))],
                linewidths=SCATTER_MARKER_LINEWIDTH,
                alpha=0.95,
            )
        if not panel_points:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_title(panel_titles.get(key, key))
        ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    _configure_narrow_log_ticks(axes[0], axis="x")
    axes[0].set_ylabel(y_label)

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=colors[value],
            markeredgewidth=SCATTER_MARKER_LINEWIDTH,
            markersize=7,
            label=value,
        )
        for value in color_values
    ]
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_by_value[value],
            color="black",
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=SCATTER_MARKER_LINEWIDTH,
            linestyle="none",
            markersize=7,
            label=value,
        )
        for value in marker_values
    ]
    color_legend = axes[-1].legend(handles=color_handles, title=_legend_title(color_title), fontsize=7, title_fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    axes[-1].add_artist(color_legend)
    axes[-1].legend(handles=marker_handles, title=_legend_title(marker_title), fontsize=7, title_fontsize=8, loc="lower left", bbox_to_anchor=(1.02, 0.0), borderaxespad=0.0)
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 0.90))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
