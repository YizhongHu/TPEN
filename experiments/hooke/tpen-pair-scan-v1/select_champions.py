"""Select champions from a collection attempt.

Reads a ``03_collect`` summary table, aggregates seed rows into non-seed
configs, and selects configured winner kinds per configured major-grid bucket.
Local energy ranking uses seed medians, while overlap tests use the
seed-combined mean and standard error. An explicit scalar metric can still be
passed for debugging overrides.

**Split-sample selection.** A champion spec may declare ``selection_seeds`` and
``holdout_seeds``. Selection then reads only the selection seed rows -- per
bucket AND for the cross-bucket champion -- and the champion's own metric is
re-read on the holdout rows into the ``holdout_*`` columns of ``champions.csv``.
This is not tidiness: ``min`` over many noisy candidates is a winner's curse, the
per-bucket bias scales with that bucket's run-to-run noise, and comparing buckets
by their argmins therefore favours the noisiest bucket by an artifact. Fresh final
replicates fix the champion's reported value and never its identity, because
selection already happened. A grid that declares no split keeps the previous
behaviour and says so in ``selection_report.json``.

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
    metric_distribution,
    normalize_champion_specs,
    parse_group_by,
    reference_columns,
    reference_metrics as normalize_reference_metrics,
    rows_for_seeds,
    select_by_spec,
    split_sample_seeds,
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


HOLDOUT_COLUMNS = (
    "holdout_seeds",
    "holdout_metric",
    "holdout_metric_value",
    "holdout_metric_seed_mean",
    "holdout_metric_seed_stderr",
    "holdout_metric_seed_n",
)


def _seed_count(row: dict[str, Any] | None, metric: str) -> str:
    """Return how many seed rows backed one selected metric value."""

    if row is None or not metric:
        return ""
    return str(row.get(metric.replace("_seed_median", "_seed_n"), ""))


def _holdout_columns(
    winner: dict[str, Any] | None,
    *,
    holdout_seeds: Sequence[str],
    holdout_rows_by_config: Mapping[str, dict[str, Any]],
    selected_metric: str,
) -> dict[str, str]:
    """Return the champion's HELD-OUT measurement of its own selection metric.

    The champion was chosen on the selection seed rows; these columns re-read the
    same seed-aggregated metric on the seed rows that had no vote. That value is
    unbiased with respect to the selection, which the selection-sample value is
    not: ``min`` over many noisy candidates picks favourable noise as readily as a
    genuinely better configuration.

    Empty strings when no holdout is configured, or when the champion has no row
    in the holdout sample -- a missing holdout row is a collection gap to report,
    not a reason to substitute the biased number.
    """

    columns = {column: "" for column in HOLDOUT_COLUMNS}
    columns["holdout_seeds"] = ",".join(str(seed) for seed in holdout_seeds)
    if winner is None or not holdout_seeds or not selected_metric:
        return columns
    holdout_row = holdout_rows_by_config.get(str(winner.get("config_id", "")))
    if holdout_row is None:
        return columns
    columns["holdout_metric"] = selected_metric
    columns["holdout_metric_value"] = str(holdout_row.get(selected_metric, ""))
    for statistic in ("mean", "stderr", "n"):
        source = selected_metric.replace("_seed_median", f"_seed_{statistic}")
        columns[f"holdout_metric_seed_{statistic}"] = str(holdout_row.get(source, ""))
    return columns


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

    # Seed rows are folded into per-configuration rows once per requested seed
    # sample. `None` is every row, which is what the report's `configs` table and
    # the collection-health counters describe; a split-sample spec additionally
    # asks for its selection sample and its holdout sample, and each is
    # aggregated independently so a holdout row's statistics never mix with the
    # rows that chose the champion.
    aggregated: dict[tuple[str, ...] | None, tuple[list[dict[str, Any]], bool]] = {}

    def candidates_for(seeds: tuple[str, ...] | None) -> tuple[list[dict[str, Any]], bool]:
        if seeds not in aggregated:
            sample = rows if seeds is None else rows_for_seeds(rows, seed_key=seed_key, seeds=seeds)
            aggregated[seeds] = aggregate_candidates(
                sample,
                config_keys=config_keys,
                major_axes=major_axes,
                minor_axes=minor_axes,
                seed_key=seed_key,
                axis_id_labels=axis_id_labels,
                success_statuses=SUCCESS_STATUSES,
                id_for_axes=id_for_axes,
            )
        return aggregated[seeds]

    def grouped(candidate_rows: Sequence[dict[str, Any]]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in candidate_rows:
            groups.setdefault(group_key_of(row, group_keys), []).append(row)
        return groups

    candidates, used_fallback = candidates_for(None)

    champions = []
    task_lineage = []
    decisions_by_group: dict[str, dict[str, list[str]]] = {}
    # `exclude` lets one spec avoid the config another already took, so the
    # accumulated selections are per BUCKET and outlive the spec loop.
    selected_by_group: dict[tuple[str, ...], dict[str, dict[str, Any]]] = {}
    split_sample: dict[str, dict[str, Any]] = {}
    distributions: dict[str, dict[str, dict[str, Any]]] = {}
    overall_by_kind: dict[str, dict[str, Any]] = {}

    for spec in champion_specs:
        kind = str(spec["name"])
        split = split_sample_seeds(spec)
        selection_seeds = None if split is None else split[0]
        holdout_seeds: tuple[str, ...] = () if split is None else split[1]
        spec_candidates, _ = candidates_for(selection_seeds)
        holdout_rows_by_config: dict[str, dict[str, Any]] = {}
        if holdout_seeds:
            holdout_candidates, _ = candidates_for(holdout_seeds)
            holdout_rows_by_config = {
                str(row.get("config_id", "")): row for row in holdout_candidates
            }
        best_k = int(spec.get("reference_distribution_best_k", 0) or 0)
        split_sample[kind] = {
            "selection_seeds": list(selection_seeds or ()),
            "holdout_seeds": list(holdout_seeds),
            "n_selection_configs": len(spec_candidates),
            "n_holdout_configs": len(holdout_rows_by_config),
            # Stated rather than implied: with no split, selection reads every
            # seed row and the champion's identity carries the winner's-curse bias.
            "enabled": split is not None,
        }
        distributions[kind] = {}

        for group_key, group_rows in sorted(grouped(spec_candidates).items()):
            group_label = group_label_from_key(group_keys, group_key)
            selected_by_name = selected_by_group.setdefault(group_key, {})
            winner, decisions, selected_metric, selected_value = select_by_spec(
                group_rows,
                spec,
                selected_by_name=selected_by_name,
                default_fallback_metric=DEFAULT_FALLBACK_METRIC,
                metric_override=metric,
                mode_override=mode,
            )
            decisions_by_group.setdefault(group_label, {})[kind] = decisions
            if winner is not None:
                selected_by_name[kind] = winner
            record = champion_record(
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
            record.update(
                _holdout_columns(
                    winner,
                    holdout_seeds=holdout_seeds,
                    holdout_rows_by_config=holdout_rows_by_config,
                    selected_metric=selected_metric,
                )
            )
            champions.append(record)
            if best_k:
                # The distribution is computed on the SAME rows the champion was
                # chosen from, so "the ranking flipped between champion and
                # median" compares like with like.
                distributions[kind][group_label] = {
                    label: metric_distribution(group_rows, source_metric, best_k=best_k)
                    for label, source_metric in reference_metric_pairs
                }
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

        # The cross-bucket champion is selected on the same sample as the
        # per-bucket ones. Reading the full sample here would put the holdout back
        # into a selection decision through the back door.
        if spec_candidates:
            overall_by_kind[kind] = {"candidates": spec_candidates}

    first_spec = champion_specs[0]
    first_candidates = overall_by_kind.get(str(first_spec["name"]), {}).get("candidates", [])
    if first_candidates:
        overall, overall_decisions, overall_metric, overall_metric_value = select_by_spec(
            first_candidates,
            first_spec,
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
        overall_selected[str(first_spec["name"])] = overall
    secondary_spec = champion_specs[1] if len(champion_specs) > 1 else first_spec
    secondary_name = str(secondary_spec["name"])
    secondary_candidates = overall_by_kind.get(secondary_name, {}).get("candidates", [])
    if secondary_candidates:
        secondary, secondary_decisions, secondary_metric, secondary_metric_value = select_by_spec(
            secondary_candidates,
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
        "split_sample": split_sample,
        "bucket_distributions": distributions,
        "champions": champions,
        "configs": candidates,
        "overall_champion": None if overall is None else overall.get("config_id", ""),
        "overall_metric": overall_metric,
        "overall_metric_value": overall_metric_value,
        # How many seed rows stand behind the cross-bucket headline number. Under
        # split-sample selection this must equal the SELECTION sample size, not the
        # collection's seed count: it is the one place a cross-bucket selection
        # that quietly read every seed row becomes visible, because a seed median
        # over three rows is robust to one excursion and would hide the leak.
        "overall_metric_seed_n": _seed_count(overall, overall_metric),
        "overall_decisions": overall_decisions,
        "secondary_champion_kind": secondary_name,
        "secondary_champion": None if secondary is None else secondary.get("config_id", ""),
        "secondary_metric": secondary_metric,
        "secondary_metric_value": secondary_metric_value,
        "secondary_metric_seed_n": _seed_count(secondary, secondary_metric),
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
        # The champion's own metric re-read on the seed rows that had no vote in
        # choosing it. Additive: every pre-existing column keeps its name and
        # meaning, and a grid with no split-sample block writes these empty.
        *HOLDOUT_COLUMNS,
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
        # Which seed rows were allowed to choose each champion, and which were
        # held back to measure it. Recorded per champion kind so the bias status
        # of a reported number is readable from the artifact alone.
        "split_sample": selection["split_sample"],
        # Per-bucket distribution over that bucket's configurations, for every
        # reference metric. A basis ranking that flips between champion and median
        # was a ranking of noise.
        "bucket_distributions": selection["bucket_distributions"],
        "reference_metrics": selection["reference_metrics"],
        "reference_statistics": list(REFERENCE_STATISTICS),
        "wall_time_metrics": list(WALL_TIME_METRICS),
        "n_candidates": selection["n_candidates"],
        "n_configs": selection["n_candidates"],
        "n_champions": len(champions),
        "overall_champion": selection["overall_champion"],
        "overall_metric": selection["overall_metric"],
        "overall_metric_value": selection["overall_metric_value"],
        "overall_metric_seed_n": selection["overall_metric_seed_n"],
        "secondary_metric_seed_n": selection["secondary_metric_seed_n"],
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
        parser.error("use --grid experiments/hooke/tpen-pair-scan-v1/configs/smoke.yaml with the normal stage stack")
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
