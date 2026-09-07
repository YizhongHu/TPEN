"""Select champions from a collection attempt.

Reads a ``03_collect`` summary table, aggregates seed rows into non-seed
configs, and selects configured winner kinds per configured major-grid bucket.
Local energy ranking uses seed medians, while overlap tests use the
seed-combined mean and standard error. An explicit scalar metric can still be
passed for debugging overrides.

Also writes ``task_lineage.jsonl``, a toolkit sidecar mapping each champion
row to the validation (and train) task ids of its contributing run ids,
extending the chain from ``03_collect``'s own sidecar (see
``experiments.toolkit.lineage``). This is additive: ``champions.csv`` and
``selection_report.json`` keep their stable public schema.

The generic, layout-agnostic selection engine (metric ladders, single-metric
selection, spec/reference normalization, group-by and seed aggregation) lives in
``experiments.toolkit.selection``; this study passes its own defaults
(success statuses, reference statistics, fallback wall-time metric) and its
``id_for_axes`` builder into those helpers.
"""

from __future__ import annotations

import argparse
import csv
import sys
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

_tpen_utils_ancestry = sibling(__file__, 'utils.ancestry')
source_grid_from_attempt = _tpen_utils_ancestry.source_grid_from_attempt
_tpen_utils_io = sibling(__file__, 'utils.io')
read_json = _tpen_utils_io.read_json
write_json = _tpen_utils_io.write_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_COLLECT = _tpen_utils_layout.STAGE_COLLECT
STAGE_SELECT = _tpen_utils_layout.STAGE_SELECT
latest_attempt_id = _tpen_utils_layout.latest_attempt_id
stage_dir = _tpen_utils_layout.stage_dir
write_latest = _tpen_utils_layout.write_latest
_tpen_utils_naming = sibling(__file__, 'utils.naming')
axis_id_labels_from_manifest = _tpen_utils_naming.axis_id_labels_from_manifest
champion_lineage_row_id = _tpen_utils_naming.champion_lineage_row_id
grid_axes_from_manifest = _tpen_utils_naming.grid_axes_from_manifest
id_for_axes = _tpen_utils_naming.id_for_axes
log_prefix = _tpen_utils_naming.log_prefix
study_name_from_manifest = _tpen_utils_naming.study_name_from_manifest
_tpen_utils_time = sibling(__file__, 'utils.time')
new_attempt_id = _tpen_utils_time.new_attempt_id

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.toolkit import TaskLineageRow, read_task_lineage, write_task_lineage  # noqa: E402
from experiments.toolkit.selection import (  # noqa: E402
    aggregate_candidates,
    champion_record,
    group_key as group_key_of,
    group_label_from_key,
    normalize_champion_specs,
    parse_group_by,
    reference_columns,
    reference_metrics as normalize_reference_metrics,
    select_by_spec,
)

DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
REFERENCE_STATISTICS = ("median", "mean", "stderr")
WALL_TIME_METRICS = ("train/runtime/wall_time_sec",)
SUCCESS_STATUSES = {"completed", "success"}
DEFAULT_FALLBACK_METRIC = "train/runtime/wall_time_sec"


def read_summary(collection_attempt_dir: Path) -> list[dict[str, Any]]:
    """Read the collection ``summary.csv`` rows."""

    summary = collection_attempt_dir / "summary.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"collection attempt has no summary.csv: {summary}")
    with summary.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _champion_task_lineage(
    row: dict[str, Any] | None,
    *,
    group_keys: Sequence[str],
    group_key: Sequence[str],
    winner_kind: str,
    upstream_lineage: Mapping[str, TaskLineageRow],
) -> TaskLineageRow:
    """Return one champion row's task-id lineage from its contributing run ids."""

    row_id = champion_lineage_row_id(winner_kind, group_keys, group_key)
    task_ids: dict[str, Any] = {}
    if row is not None:
        run_ids = [run_id for run_id in str(row.get("run_ids", "")).split(";") if run_id]
        for kind in ("validation", "train"):
            values = [
                upstream_lineage[run_id].task_ids[kind]
                for run_id in run_ids
                if run_id in upstream_lineage and kind in upstream_lineage[run_id].task_ids
            ]
            if values:
                task_ids[kind] = values
    return TaskLineageRow(row_id=row_id, task_ids=task_ids)


def select_champions(
    rows: Sequence[dict[str, Any]],
    *,
    config_keys: Sequence[str],
    major_axes: Sequence[str],
    minor_axes: Sequence[str],
    seed_key: str,
    axis_id_labels: dict[str, str],
    metric: str | None = None,
    mode: str = "min",
    group_by: str | Sequence[str] | None = None,
    champion_specs: Sequence[Any] | None = None,
    reference_metrics: Sequence[Any] | None = None,
    upstream_lineage: Mapping[str, TaskLineageRow] | None = None,
) -> dict[str, Any]:
    """Select configured winner kinds per major grid point."""

    if mode not in {"min", "max"}:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    group_keys = parse_group_by(group_by if group_by is not None else major_axes)
    if champion_specs is None:
        raise ValueError("champion selector specs are required")
    champion_specs = normalize_champion_specs(champion_specs)
    champion_kinds = [str(spec["name"]) for spec in champion_specs]
    reference_metric_pairs = normalize_reference_metrics(reference_metrics)
    candidates, used_fallback = aggregate_candidates(
        rows,
        config_keys=config_keys,
        major_axes=major_axes,
        minor_axes=minor_axes,
        seed_key=seed_key,
        axis_id_labels=axis_id_labels,
        success_statuses=SUCCESS_STATUSES,
        id_for_axes=id_for_axes,
    )

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault(group_key_of(row, group_keys), []).append(row)

    champions = []
    task_lineage = []
    decisions_by_group: dict[str, dict[str, list[str]]] = {}
    for group_key, group_rows in sorted(groups.items()):
        group_decisions: dict[str, list[str]] = {}
        selected_by_name: dict[str, dict[str, Any]] = {}
        for spec in champion_specs:
            kind = str(spec["name"])
            winner, decisions, selected_metric, selected_value = select_by_spec(
                group_rows,
                spec,
                selected_by_name=selected_by_name,
                default_fallback_metric=DEFAULT_FALLBACK_METRIC,
                metric_override=metric,
                mode_override=mode,
            )
            group_decisions[kind] = decisions
            if winner is not None:
                selected_by_name[kind] = winner
            champions.append(
                champion_record(
                    winner,
                    group_keys=group_keys,
                    group_key=group_key,
                    config_keys=config_keys,
                    winner_kind=kind,
                    metric=selected_metric,
                    metric_value=selected_value,
                    reference_metrics=reference_metric_pairs,
                    reference_statistics=REFERENCE_STATISTICS,
                )
            )
            if upstream_lineage is not None:
                task_lineage.append(
                    _champion_task_lineage(
                        winner,
                        group_keys=group_keys,
                        group_key=group_key,
                        winner_kind=kind,
                        upstream_lineage=upstream_lineage,
                    )
                )
        decisions_by_group[group_label_from_key(group_keys, group_key)] = group_decisions

    if candidates:
        overall, overall_decisions, overall_metric, overall_metric_value = select_by_spec(
            candidates,
            champion_specs[0],
            selected_by_name={},
            default_fallback_metric=DEFAULT_FALLBACK_METRIC,
            metric_override=metric,
            mode_override=mode,
        )
    else:
        overall = None
        overall_decisions = []
        overall_metric = ""
        overall_metric_value = ""

    overall_selected: dict[str, dict[str, Any]] = {}
    if overall is not None:
        overall_selected[str(champion_specs[0]["name"])] = overall
    secondary_spec = champion_specs[1] if len(champion_specs) > 1 else champion_specs[0]
    secondary_name = str(secondary_spec["name"])
    if candidates:
        secondary, secondary_decisions, secondary_metric, secondary_metric_value = select_by_spec(
            candidates,
            secondary_spec,
            selected_by_name=overall_selected,
            default_fallback_metric=DEFAULT_FALLBACK_METRIC,
            metric_override=None,
            mode_override=mode,
        )
    else:
        secondary = None
        secondary_decisions = []
        secondary_metric = ""
        secondary_metric_value = ""
    return {
        "champions": champions,
        "configs": candidates,
        "overall_champion": None if overall is None else overall.get("config_id", ""),
        "overall_metric": overall_metric,
        "overall_metric_value": overall_metric_value,
        "overall_decisions": overall_decisions,
        "secondary_champion_kind": secondary_name,
        "secondary_champion": None if secondary is None else secondary.get("config_id", ""),
        "secondary_metric": secondary_metric,
        "secondary_metric_value": secondary_metric_value,
        "secondary_decisions": secondary_decisions,
        "decisions_by_group": decisions_by_group,
        "used_status_fallback": used_fallback,
        "group_by": list(group_keys),
        "champion_kinds": list(champion_kinds),
        "champion_specs": champion_specs,
        "config_keys": list(config_keys),
        "reference_metrics": [
            {"label": label, "metric": source_metric}
            for label, source_metric in reference_metric_pairs
        ],
        "n_candidates": len(candidates),
        "task_lineage": task_lineage,
    }


def _resolve_collection_attempt(results_root: Path, collection_attempt_id: str | None) -> str:
    if collection_attempt_id is not None:
        return collection_attempt_id
    collect_dir = stage_dir(results_root, STAGE_COLLECT)
    attempt_id = latest_attempt_id(collect_dir)
    if attempt_id is None:
        raise FileNotFoundError(f"no collection attempts under {collect_dir}")
    return attempt_id


def _champion_specs_from_grid(
    results_root: Path,
    collection_dir: Path,
    requested: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Return champion specs from source grid manifest, optionally filtered by CLI."""

    manifest = _source_grid_manifest(results_root, collection_dir)
    configured = manifest.get("champions") if isinstance(manifest, dict) else None
    if not configured:
        raise ValueError("source grid manifest must define explicit champion selector specs")
    specs = normalize_champion_specs(configured)
    if requested is None:
        return specs
    requested_names = {str(kind) for kind in requested}
    filtered = [spec for spec in specs if str(spec.get("name")) in requested_names]
    missing = requested_names - {str(spec.get("name")) for spec in filtered}
    if missing:
        raise ValueError(f"requested champions are not defined by the source grid: {', '.join(sorted(missing))}")
    return filtered


def _reference_metrics_from_grid(
    results_root: Path,
    collection_dir: Path,
) -> list[tuple[str, str]]:
    """Return reference metrics from the source grid manifest, or defaults."""

    manifest = _source_grid_manifest(results_root, collection_dir)
    configured = manifest.get("champion_reference_metrics") if isinstance(manifest, dict) else None
    if configured:
        return list(normalize_reference_metrics(configured))
    return list(normalize_reference_metrics(None))


def _source_grid_manifest(results_root: Path, collection_dir: Path) -> dict[str, Any] | None:
    """Return the source grid manifest for a collection attempt, if available."""

    source_grid = source_grid_from_attempt(results_root, collection_dir)
    if source_grid is None or not source_grid.manifest_path.is_file():
        return None
    return source_grid.read_manifest()


def _axis_metadata_from_collection(results_root: Path, collection_dir: Path) -> dict[str, Any]:
    """Return axis metadata inherited by a collection attempt."""

    report_path = collection_dir / "collection_report.json"
    report = read_json(report_path) if report_path.is_file() else {}
    if isinstance(report, dict) and report.get("config_keys"):
        major_axes = tuple(str(axis) for axis in report.get("major_axes", ()))
        minor_axes = tuple(str(axis) for axis in report.get("minor_axes", ()))
        seed_key = str(report.get("scan_seed_axis", "seed"))
        config_keys = tuple(str(axis) for axis in report.get("config_keys", (*major_axes, *minor_axes)))
        labels = report.get("axis_id_labels") if isinstance(report.get("axis_id_labels"), dict) else {}
        return {
            "major_axes": major_axes,
            "minor_axes": minor_axes,
            "config_keys": config_keys,
            "scan_seed_axis": seed_key,
            "axis_id_labels": {axis: str(labels.get(axis, axis)) for axis in (*config_keys, seed_key)},
        }

    manifest = _source_grid_manifest(results_root, collection_dir)
    axes = grid_axes_from_manifest(manifest)
    config_keys = tuple(axes["config_axes"])
    seed_key = str(axes["scan_seed_axis"])
    labels = axis_id_labels_from_manifest(manifest, (*config_keys, seed_key))
    return {
        "major_axes": tuple(axes["major_axes"]),
        "minor_axes": tuple(axes["minor_axes"]),
        "config_keys": config_keys,
        "scan_seed_axis": seed_key,
        "axis_id_labels": labels,
    }


def _study_from_collection(results_root: Path, collection_dir: Path) -> str:
    """Return the study name inherited by a collection attempt."""

    report_path = collection_dir / "collection_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        if isinstance(report, dict) and report.get("study"):
            return study_name_from_manifest(report)
    manifest = _source_grid_manifest(results_root, collection_dir)
    if manifest is not None:
        return study_name_from_manifest(manifest)
    return study_name_from_manifest(None)


def _parse_champion_args(values: Sequence[str] | None) -> list[str] | None:
    """Parse repeated or comma-separated champion-kind CLI values."""

    if values is None:
        return None
    parsed = []
    for value in values:
        parsed.extend(part.strip() for part in str(value).split(",") if part.strip())
    return parsed


def select(
    *,
    results_root: str | Path,
    collection_attempt_id: str | None = None,
    select_attempt_id: str | None = None,
    metric: str | None = None,
    mode: str = "min",
    group_by: str | Sequence[str] | None = None,
    champion_kinds: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Select champions from a collection attempt and write a ``04_select`` attempt."""

    results_root = Path(results_root)
    collection_attempt_id = _resolve_collection_attempt(results_root, collection_attempt_id)
    select_attempt_id = select_attempt_id or new_attempt_id()
    collection_dir = stage_dir(results_root, STAGE_COLLECT) / collection_attempt_id

    rows = read_summary(collection_dir)
    study = _study_from_collection(results_root, collection_dir)
    axis_metadata = _axis_metadata_from_collection(results_root, collection_dir)
    config_keys = tuple(axis_metadata["config_keys"])
    champion_specs = _champion_specs_from_grid(results_root, collection_dir, champion_kinds)
    reference_metrics = _reference_metrics_from_grid(results_root, collection_dir)
    upstream_lineage = read_task_lineage(collection_dir)
    selection = select_champions(
        rows,
        metric=metric,
        mode=mode,
        config_keys=config_keys,
        major_axes=tuple(axis_metadata["major_axes"]),
        minor_axes=tuple(axis_metadata["minor_axes"]),
        seed_key=str(axis_metadata["scan_seed_axis"]),
        axis_id_labels=dict(axis_metadata["axis_id_labels"]),
        upstream_lineage=upstream_lineage,
        group_by=group_by,
        champion_specs=champion_specs,
        reference_metrics=reference_metrics,
    )

    attempt = stage_dir(results_root, STAGE_SELECT) / select_attempt_id
    attempt.mkdir(parents=True, exist_ok=True)

    champions = selection["champions"]
    group_keys = tuple(selection["group_by"])
    non_group_config_keys = [key for key in config_keys if key not in group_keys]
    columns = [
        *group_keys,
        "winner_kind",
        "config_id",
        "major_id",
        "minor_id",
        *non_group_config_keys,
        "seeds",
        "n_expected",
        "n_present",
        "n_success",
        "n_failed",
        "n_missing_seed",
        "metric",
        "metric_value",
        "metric_seed_mean",
        "metric_seed_stderr",
        "metric_seed_n",
        *reference_columns(reference_metrics, reference_statistics=REFERENCE_STATISTICS),
        "run_ids",
    ]
    with (attempt / "champions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for champion in champions:
            writer.writerow(champion)

    write_json(
        attempt / "source_collection_attempt.json",
        {
            "collection_attempt_id": collection_attempt_id,
            "collection_attempt_dir": str(collection_dir),
        },
    )
    write_task_lineage(attempt, selection["task_lineage"])
    report = {
        "study": study,
        "stage": STAGE_SELECT,
        "attempt_id": select_attempt_id,
        "collection_attempt_id": collection_attempt_id,
        "metric": metric,
        "mode": mode,
        "group_by": selection["group_by"],
        "major_axes": list(axis_metadata["major_axes"]),
        "minor_axes": list(axis_metadata["minor_axes"]),
        "scan_seed_axis": axis_metadata["scan_seed_axis"],
        "axis_id_labels": axis_metadata["axis_id_labels"],
        "champion_kinds": selection["champion_kinds"],
        "champion_specs": selection["champion_specs"],
        "config_keys": selection["config_keys"],
        "seed_aggregation": {
            "value": "median of successful seed rows",
            "error_bar": "sample standard error across successful seed rows",
            "mean": "arithmetic mean across successful seed rows",
        },
        "reference_metrics": selection["reference_metrics"],
        "reference_statistics": list(REFERENCE_STATISTICS),
        "wall_time_metrics": list(WALL_TIME_METRICS),
        "n_candidates": selection["n_candidates"],
        "n_configs": selection["n_candidates"],
        "n_champions": len(champions),
        "overall_champion": selection["overall_champion"],
        "overall_metric": selection["overall_metric"],
        "overall_metric_value": selection["overall_metric_value"],
        "overall_decisions": selection["overall_decisions"],
        "secondary_champion_kind": selection["secondary_champion_kind"],
        "secondary_metric": selection["secondary_metric"],
        "secondary_champion": selection["secondary_champion"],
        "secondary_metric_value": selection["secondary_metric_value"],
        "secondary_decisions": selection["secondary_decisions"],
        "decisions_by_group": selection["decisions_by_group"],
        "used_status_fallback": selection["used_status_fallback"],
        "champions": champions,
        "configs": selection["configs"],
    }
    write_json(attempt / "selection_report.json", report)
    write_latest(stage_dir(results_root, STAGE_SELECT), select_attempt_id)
    return {"attempt_dir": str(attempt), "report": report}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse select command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--collection-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None, help="Select attempt id (defaults to now).")
    parser.add_argument("--smoke", action="store_true", help="Deprecated; use configs/smoke.yaml with the normal stack.")
    parser.add_argument(
        "--metric",
        default=None,
        help="Optional scalar metric override. By default, use the ordered local-energy tie-breaker ladder.",
    )
    parser.add_argument("--mode", choices=["min", "max"], default="min")
    parser.add_argument(
        "--group-by",
        default=None,
        help="Comma-separated grouping columns for winner buckets (default: source grid major_axes).",
    )
    parser.add_argument(
        "--champions",
        nargs="+",
        default=None,
        help="Champion kinds to select, e.g. 'energy stability'. Defaults to source grid manifest.",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        parser.error("use --grid experiments/hooke/pair_stability_v3/configs/smoke.yaml with the normal stage stack")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Select champions from the command line."""

    args = parse_args(argv)
    result = select(
        results_root=args.results_root,
        collection_attempt_id=args.collection_attempt_id,
        select_attempt_id=args.attempt_id,
        metric=args.metric,
        mode=args.mode,
        group_by=args.group_by,
        champion_kinds=_parse_champion_args(args.champions),
    )
    report = result["report"]
    prefix = log_prefix(report.get("study"))
    print(
        f"{prefix} selected {report['n_champions']} champions "
        f"(overall {report['overall_champion']}, "
        f"{report['secondary_champion_kind']} {report['secondary_champion']}) -> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
