"""Collect validation attempts from one planned grid into a summary table.

Reads run ids from ``00_grid/{attempt_id}/manifest.json``, then consumes the
latest validation attempt for each planned run id,
reads each attempt's status, evaluation metrics, required source train-attempt
metrics, and recorded train-attempt provenance, and writes a ``03_collect`` attempt with
``summary.csv``, ``failures.csv``, ``collection_report.json``, and explicit
source pointers to the exact validation (and grid) attempts consumed.

Also writes ``task_lineage.jsonl``, a toolkit sidecar mapping each collected
run id to the deterministic validation (and, when resolved, train) task ids
that produced it (see ``experiments.toolkit.lineage``). This is additive:
``summary.csv``/``failures.csv``/``collection_report.json`` keep their stable
public schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

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
SourceGrid = _tpen_utils_ancestry.SourceGrid
source_grid_from_attempt = _tpen_utils_ancestry.source_grid_from_attempt
source_grid_from_id = _tpen_utils_ancestry.source_grid_from_id
_tpen_utils_io = sibling(__file__, 'utils.io')
read_json = _tpen_utils_io.read_json
write_json = _tpen_utils_io.write_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_COLLECT = _tpen_utils_layout.STAGE_COLLECT
STAGE_GRID = _tpen_utils_layout.STAGE_GRID
STAGE_TRAIN = _tpen_utils_layout.STAGE_TRAIN
STAGE_VALIDATION = _tpen_utils_layout.STAGE_VALIDATION
latest_attempt_id = _tpen_utils_layout.latest_attempt_id
stage_dir = _tpen_utils_layout.stage_dir
validation_run_dir = _tpen_utils_layout.validation_run_dir
write_latest = _tpen_utils_layout.write_latest
_tpen_utils_naming = sibling(__file__, 'utils.naming')
axis_id_labels_from_manifest = _tpen_utils_naming.axis_id_labels_from_manifest
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

from experiments.toolkit import (  # noqa: E402
    TaskLineageRow,
    stage_plan_task_ids,
    synthesized_task_id,
    write_task_lineage,
)
from experiments.toolkit.artifacts import (  # noqa: E402
    duration_from_status_file,
    load_json_dict_if_present,
    read_metrics_jsonl as read_metrics_rows,
    read_metrics_map,
    status_of,
    write_csv,
)
from experiments.toolkit.cost import (  # noqa: E402
    COST_BY_AXIS_COLUMNS,
    COST_BY_RUN_BASE_COLUMNS,
    COST_BY_TASK_COLUMNS,
    cost_by_axis_rows,
    cost_by_run_row,
    cost_by_task_rows,
)

DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"

NON_DIAGNOSTIC_DIRS = {"checkpoints", "diagnostics"}
SUCCESS_STATUSES = {"completed", "success"}

BASE_COLUMNS = (
    "run_id",
    "validation_attempt_id",
    "validation_attempt_dir",
    "status",
    "major_id",
    "minor_id",
    "config_id",
    "train_attempt_id",
    "checkpoint_path",
    "n_diagnostics",
)

TRAIN_WALL_TIME_METRIC = "train/runtime/wall_time_sec"


def read_metrics_jsonl(path: Path) -> dict[str, Any]:
    """Flatten ``metrics.jsonl`` records into ``namespace/key -> value``."""

    return read_metrics_map(path)


def _run_parameters(attempt_dir: Path) -> dict[str, Any]:
    """Recover run parameters from resolved_config.yaml when available."""

    resolved = attempt_dir / "resolved_config.yaml"
    if resolved.is_file():
        cfg = OmegaConf.load(resolved)
        params = OmegaConf.select(cfg, "run_parameters")
        if params is not None:
            return OmegaConf.to_container(params, resolve=True)
    return {}


def _count_diagnostics(attempt_dir: Path) -> int:
    """Count evaluation task outputs recorded for this attempt."""

    index = attempt_dir / "diagnostics" / "index.json"
    if index.is_file():
        data = read_json(index)
        if isinstance(data, dict) and isinstance(data.get("artifacts"), list):
            return len(data["artifacts"])
        if isinstance(data, list):
            return len(data)
    return sum(
        1
        for child in attempt_dir.iterdir()
        if child.is_dir() and child.name not in NON_DIAGNOSTIC_DIRS
    )


def _train_attempt_dir(source: dict[str, Any]) -> Path | None:
    """Return the source train attempt directory recorded by validation."""

    train_attempt_dir = source.get("train_attempt_dir")
    if train_attempt_dir:
        return Path(str(train_attempt_dir))
    checkpoint_path = source.get("checkpoint_path")
    if checkpoint_path:
        checkpoint = Path(str(checkpoint_path))
        if checkpoint.name == "checkpoints":
            return checkpoint.parent
    return None


def _metric_is_train_metric(metric: str) -> bool:
    """Return whether ``metric`` refers to the train-attempt metric namespace."""

    return metric.startswith("train/")


def _required_train_metrics(grid_manifest: dict[str, Any] | None) -> set[str]:
    """Return train metrics referenced by configured champion selection."""

    metrics: set[str] = set()
    if not isinstance(grid_manifest, dict):
        return metrics
    for spec in grid_manifest.get("champions", []) or []:
        if not isinstance(spec, dict):
            continue
        for key in ("metric", "fallback_metric"):
            metric = str(spec.get(key, "")).strip()
            if _metric_is_train_metric(metric):
                metrics.add(metric)
    for spec in grid_manifest.get("champion_reference_metrics", []) or []:
        if not isinstance(spec, dict):
            continue
        metric = str(spec.get("metric", "")).strip()
        if _metric_is_train_metric(metric):
            metrics.add(metric)
    return metrics


def _train_metrics(source: dict[str, Any], *, required_metrics: set[str]) -> dict[str, Any]:
    """Return only required metrics from the validation source train attempt."""

    train_attempt = _train_attempt_dir(source)
    if train_attempt is None or not required_metrics:
        return {}
    train_metrics: dict[str, Any] = {}
    pending = set(required_metrics)
    if TRAIN_WALL_TIME_METRIC in pending:
        wall_time = duration_from_status_file(train_attempt, clamp_negative=True)
        if wall_time is not None:
            train_metrics[TRAIN_WALL_TIME_METRIC] = wall_time
            pending.remove(TRAIN_WALL_TIME_METRIC)
    if pending:
        parsed = read_metrics_map(train_attempt / "metrics.jsonl", prefix="train")
        train_metrics.update({key: value for key, value in parsed.items() if key in pending})
    return train_metrics


def _axis_metadata(manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized axis metadata for collection."""

    axes = grid_axes_from_manifest(manifest)
    all_axes = tuple(axis for axis in axes["run_axes"] if isinstance(axis, str))
    labels = axis_id_labels_from_manifest(manifest, all_axes)
    return {
        **axes,
        "axis_id_labels": labels,
    }


def _point_from_sources(
    *,
    params: dict[str, Any],
    grid_job: dict[str, Any] | None,
    axes: Sequence[str],
) -> dict[str, Any]:
    """Return configured axis values from resolved config or source grid job."""

    choices = grid_job.get("choices", {}) if isinstance(grid_job, dict) else {}
    point = {}
    for axis in axes:
        if axis in params:
            point[axis] = params[axis]
        elif isinstance(choices, dict) and axis in choices:
            point[axis] = choices[axis]
        else:
            point[axis] = ""
    return point


def collect_validation_attempt(
    run_id: str,
    attempt_id: str,
    attempt_dir: Path,
    *,
    grid_job: dict[str, Any] | None,
    axis_metadata: dict[str, Any],
    required_train_metrics: set[str],
) -> dict[str, Any]:
    """Build one summary row from a validation attempt directory."""

    params = {} if grid_job is not None else _run_parameters(attempt_dir)
    major_axes = tuple(axis_metadata["major_axes"])
    minor_axes = tuple(axis_metadata["minor_axes"])
    config_axes = tuple(axis_metadata["config_axes"])
    run_axes = tuple(axis_metadata["run_axes"])
    labels = axis_metadata["axis_id_labels"]
    point = _point_from_sources(params=params, grid_job=grid_job, axes=run_axes)
    source_path = attempt_dir / "source_train_attempt.json"
    source = read_json(source_path) if source_path.is_file() else {}

    row: dict[str, Any] = {column: "" for column in (*BASE_COLUMNS, *run_axes)}
    row.update(
        run_id=run_id,
        validation_attempt_id=attempt_id,
        validation_attempt_dir=str(attempt_dir),
        status=status_of(attempt_dir),
        major_id=(grid_job or {}).get("major_id") or id_for_axes(point, major_axes, labels),
        minor_id=(grid_job or {}).get("minor_id") or id_for_axes(point, minor_axes, labels),
        config_id=(grid_job or {}).get("config_id") or id_for_axes(point, config_axes, labels),
        train_attempt_id=source.get("train_attempt_id", ""),
        checkpoint_path=source.get("checkpoint_path", ""),
        n_diagnostics=_count_diagnostics(attempt_dir),
    )
    row.update(point)
    row.update(_train_metrics(source, required_metrics=required_train_metrics))
    row.update(read_metrics_jsonl(attempt_dir / "metrics.jsonl"))
    return row


def _cached_stage_task_ids(
    cache: dict[str, frozenset[str] | None], stage_root: Path, attempt_id: str
) -> frozenset[str] | None:
    """Return a stage's known task ids, reading its stage plan at most once."""

    if attempt_id not in cache:
        cache[attempt_id] = stage_plan_task_ids(stage_root / "stage_plans" / attempt_id)
    return cache[attempt_id]


def _row_task_lineage(
    row: dict[str, Any],
    *,
    run_id: str,
    validation_attempt_id: str,
    results_root: Path,
    validation_known_task_ids: dict[str, frozenset[str] | None],
    train_known_task_ids: dict[str, frozenset[str] | None],
) -> TaskLineageRow:
    """Return one row's validation (and, if resolved, train) task-id lineage."""

    task_ids: dict[str, str] = {
        "validation": synthesized_task_id(
            stage=STAGE_VALIDATION,
            run_id=run_id,
            attempt_id=validation_attempt_id,
            known_task_ids=_cached_stage_task_ids(
                validation_known_task_ids, stage_dir(results_root, STAGE_VALIDATION), validation_attempt_id
            ),
        )
    }
    train_attempt_id = str(row.get("train_attempt_id") or "")
    if train_attempt_id:
        task_ids["train"] = synthesized_task_id(
            stage=STAGE_TRAIN,
            run_id=run_id,
            attempt_id=train_attempt_id,
            known_task_ids=_cached_stage_task_ids(
                train_known_task_ids, stage_dir(results_root, STAGE_TRAIN), train_attempt_id
            ),
        )
    return TaskLineageRow(row_id=run_id, task_ids=task_ids)


def _resolve_grid_source(results_root: Path, grid_attempt_id: str | None) -> SourceGrid:
    """Return the explicitly requested or latest planned source grid."""

    grid_stage = stage_dir(results_root, STAGE_GRID)
    attempt_id = grid_attempt_id or latest_attempt_id(grid_stage)
    if attempt_id is None:
        raise FileNotFoundError(f"no grid attempts under {grid_stage}")
    source_grid = source_grid_from_id(results_root, attempt_id)
    if not source_grid.manifest_path.is_file():
        raise FileNotFoundError(f"grid attempt has no manifest.json: {source_grid.manifest_path}")
    return source_grid


def _run_ids(grid_manifest: dict[str, Any]) -> list[str]:
    """Return run ids listed by one grid manifest."""

    return [str(job["run_id"]) for job in grid_manifest.get("jobs", [])]


def _run_device(attempt_dir: Path) -> str:
    """Return the recorded runtime device for one run attempt."""

    metadata = load_json_dict_if_present(attempt_dir / "metadata.json")
    runtime = metadata.get("runtime", {}) if isinstance(metadata.get("runtime"), dict) else {}
    return str(runtime.get("device", metadata.get("device", "")))


def _cost_tables(
    rows: Sequence[dict[str, Any]],
    consumed: Sequence[dict[str, Any]],
    *,
    axis_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project validation and source-train run metrics into cost rows."""

    run_axes = tuple(axis_metadata["run_axes"])
    cost_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for row, meta in zip(rows, consumed, strict=True):
        attempt_dir = Path(meta["validation_attempt_dir"])
        attempt_id = str(meta["validation_attempt_id"])
        run_id = str(row["run_id"])
        axes = {axis: row.get(axis, "") for axis in run_axes}
        validation_metrics = read_metrics_rows(attempt_dir / "metrics.jsonl")
        device = _run_device(attempt_dir)
        cost_rows.append(
            cost_by_run_row(
                validation_metrics,
                run_id=run_id,
                attempt_id=attempt_id,
                stage="validation",
                status=str(row["status"]),
                device_type=device,
                axes=axes,
            )
        )
        task_rows.extend(
            cost_by_task_rows(
                validation_metrics,
                run_id=run_id,
                attempt_id=attempt_id,
                stage="validation",
                device_type=device,
            )
        )
        source_path = attempt_dir / "source_train_attempt.json"
        source = read_json(source_path) if source_path.is_file() else {}
        train_dir = _train_attempt_dir(source)
        if train_dir is not None and train_dir.is_dir():
            cost_rows.append(
                cost_by_run_row(
                    read_metrics_rows(train_dir / "metrics.jsonl"),
                    run_id=run_id,
                    attempt_id=str(source.get("train_attempt_id", "")),
                    stage="train",
                    status=status_of(train_dir),
                    device_type=_run_device(train_dir),
                    axes=axes,
                )
            )
    return cost_rows, task_rows


def collect(
    *,
    results_root: str | Path,
    collect_attempt_id: str | None = None,
    grid_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Collect validation attempts and write a ``03_collect`` attempt."""

    results_root = Path(results_root)
    collect_attempt_id = collect_attempt_id or new_attempt_id()
    source_grid = _resolve_grid_source(results_root, grid_attempt_id)
    grid_attempt_id = source_grid.attempt_id
    grid_manifest = source_grid.read_manifest()
    study = study_name_from_manifest(grid_manifest)
    axis_metadata = _axis_metadata(grid_manifest)
    required_train_metrics = _required_train_metrics(grid_manifest)
    job_by_run = {
        str(job.get("run_id")): job
        for job in (grid_manifest or {}).get("jobs", [])
        if isinstance(job, dict) and job.get("run_id")
    }

    rows: list[dict[str, Any]] = []
    consumed: list[dict[str, Any]] = []
    lineage_rows: list[TaskLineageRow] = []
    validation_known_task_ids: dict[str, frozenset[str] | None] = {}
    train_known_task_ids: dict[str, frozenset[str] | None] = {}
    for run_id in _run_ids(grid_manifest):
        run_dir = validation_run_dir(results_root, run_id)
        attempt_id = latest_attempt_id(run_dir)
        if attempt_id is None:
            continue
        attempt_dir = run_dir / attempt_id
        attempt_source = source_grid_from_attempt(results_root, attempt_dir)
        if attempt_source is None or attempt_source.attempt_id != source_grid.attempt_id:
            continue
        row = collect_validation_attempt(
            run_id,
            attempt_id,
            attempt_dir,
            grid_job=job_by_run.get(run_id),
            axis_metadata=axis_metadata,
            required_train_metrics=required_train_metrics,
        )
        rows.append(row)
        consumed.append(
            {"run_id": run_id, "validation_attempt_id": attempt_id, "validation_attempt_dir": str(attempt_dir)}
        )
        lineage_rows.append(
            _row_task_lineage(
                row,
                run_id=run_id,
                validation_attempt_id=attempt_id,
                results_root=results_root,
                validation_known_task_ids=validation_known_task_ids,
                train_known_task_ids=train_known_task_ids,
            )
        )

    attempt = stage_dir(results_root, STAGE_COLLECT) / collect_attempt_id
    attempt.mkdir(parents=True, exist_ok=True)

    axis_columns = list(axis_metadata["run_axes"])
    base_columns = [*BASE_COLUMNS, *axis_columns]
    metric_columns = sorted({key for row in rows for key in row if key not in base_columns})
    columns = base_columns + metric_columns
    failures = [row for row in rows if str(row["status"]) not in SUCCESS_STATUSES]

    write_csv(attempt / "summary.csv", rows, columns)
    write_csv(attempt / "failures.csv", failures, columns)
    cost_rows, cost_task_rows = _cost_tables(rows, consumed, axis_metadata=axis_metadata)
    cost_axis_names = [*axis_metadata["major_axes"], *axis_metadata["minor_axes"]]
    if axis_metadata["scan_seed_axis"]:
        cost_axis_names.append(axis_metadata["scan_seed_axis"])
    write_csv(attempt / "cost_by_run.csv", cost_rows, [*COST_BY_RUN_BASE_COLUMNS, *axis_columns])
    write_csv(attempt / "cost_by_task.csv", cost_task_rows, COST_BY_TASK_COLUMNS)
    write_csv(
        attempt / "cost_by_axis.csv",
        cost_by_axis_rows(cost_rows, axis_names=cost_axis_names),
        COST_BY_AXIS_COLUMNS,
    )
    write_json(
        attempt / "source_grid_attempt.json",
        source_grid.to_record(),
    )
    write_json(attempt / "source_validation_attempts.json", consumed)
    write_task_lineage(attempt, lineage_rows)
    report = {
        "study": study,
        "stage": STAGE_COLLECT,
        "attempt_id": collect_attempt_id,
        "grid_attempt_id": grid_attempt_id,
        "major_axes": list(axis_metadata["major_axes"]),
        "minor_axes": list(axis_metadata["minor_axes"]),
        "scan_seed_axis": axis_metadata["scan_seed_axis"],
        "config_keys": list(axis_metadata["config_axes"]),
        "axis_id_labels": axis_metadata["axis_id_labels"],
        "n_collected": len(rows),
        "n_failures": len(failures),
        "metric_columns": metric_columns,
        "required_train_metrics": sorted(required_train_metrics),
    }
    write_json(attempt / "collection_report.json", report)
    write_latest(stage_dir(results_root, STAGE_COLLECT), collect_attempt_id)
    return {"attempt_dir": str(attempt), "report": report, "rows": rows}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse collect command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--grid-attempt-id",
        default=None,
        help="Override source grid attempt; defaults to 00_grid/latest.json.",
    )
    parser.add_argument("--attempt-id", default=None, help="Collect attempt id (defaults to now).")
    parser.add_argument("--smoke", action="store_true", help="Deprecated; use configs/smoke.yaml with the normal stack.")
    args = parser.parse_args(argv)
    if args.smoke:
        parser.error("use --grid experiments/hooke/pair_stability_v3/configs/smoke.yaml with the normal stage stack")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Collect validation attempts from the command line."""

    args = parse_args(argv)
    result = collect(
        results_root=args.results_root,
        collect_attempt_id=args.attempt_id,
        grid_attempt_id=args.grid_attempt_id,
    )
    report = result["report"]
    prefix = log_prefix(report.get("study"))
    print(
        f"{prefix} collected {report['n_collected']} runs "
        f"({report['n_failures']} failures) -> {result['attempt_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
