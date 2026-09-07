"""Collect compact final summaries from raw final artifacts.

``final_collect.py`` is the only final-reporting stage that reads raw training
and evaluation artifacts. It consumes ``05_final_grid``, ``06_final_train``,
and ``07_final_eval`` provenance and reduces them into compact, reusable CSV
tables under ``08_final_collect``. Plotting/reporting code should consume these
tables rather than reparsing raw task records.
"""

from __future__ import annotations

import argparse
import json
import math
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

_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_FINAL_COLLECT = _tpen_utils_layout.STAGE_FINAL_COLLECT
STAGE_FINAL_EVAL = _tpen_utils_layout.STAGE_FINAL_EVAL
STAGE_FINAL_GRID = _tpen_utils_layout.STAGE_FINAL_GRID
latest_attempt_id = _tpen_utils_layout.latest_attempt_id
stage_dir = _tpen_utils_layout.stage_dir
write_latest = _tpen_utils_layout.write_latest
_tpen_utils_naming = sibling(__file__, 'utils.naming')
log_prefix = _tpen_utils_naming.log_prefix
study_name_from_manifest = _tpen_utils_naming.study_name_from_manifest
_tpen_utils_time = sibling(__file__, 'utils.time')
new_attempt_id = _tpen_utils_time.new_attempt_id
_tpen_stats = sibling(__file__, 'stats')
_as_bool = _tpen_stats.as_bool
_as_float = _tpen_stats.as_float
_max = _tpen_stats.finite_max
_sum = _tpen_stats.finite_sum
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
    csv_value as _csv_value,
    duration_from_status as _duration_from_status,
    load_json_dict_if_present as _load_json_if_present,
    metric_map as _metric_map,
    read_csv as _read_csv,
    status_of as _status_of,
    write_csv as _write_csv,
)
from experiments.toolkit.cost import (  # noqa: E402
    COST_BY_AXIS_COLUMNS,
    COST_BY_RUN_BASE_COLUMNS,
    COST_BY_TASK_COLUMNS,
    cost_by_axis_rows,
    cost_by_run_row,
    cost_by_task_rows,
)
from experiments.toolkit.virial import derive_virial_metrics

DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
EXACT_HOOKE_ENERGY = 2.0
DEFAULT_EXPECTED_FINAL_SEEDS = 10
DEFAULT_HISTOGRAM_BINS = 32
PATHOLOGY_ABS_LOCAL_ENERGY = 10.0
AXIS_PROVENANCE_COLUMNS = ("major_id", "minor_id", "config_id")
REPORT_AXIS_COLUMN = "basis_update"
REPORT_ROW_KEY = "basis_update"
REPORT_COL_KEY = "feature_normalization"

EVAL_EXACT_METRICS = {
    "eval/energy/local_energy_mean",
    "eval/energy/local_energy_stderr",
    "eval/energy/local_energy_variance",
    "eval/energy/local_energy_n_finite",
    "eval/energy/local_energy_n_total",
    "eval/energy/local_energy_finite_fraction",
    "eval/energy/local_energy_pathology_count",
    "eval/energy/term/kinetic_mean",
    "eval/energy/term/harmonic_trap_mean",
    "eval/energy/term/electron_electron_mean",
    "eval/trace_equivariance/compared_entry_count",
    "eval/trace_equivariance/comparison_error_count",
    "eval/trace_equivariance/max_abs_error",
    "runtime/wall_time_sec",
    "runtime/peak_memory_mb",
}
TRAIN_EXACT_METRICS = {
    "runtime/wall_time_sec",
    "runtime/peak_memory_mb",
    "train/energy",
    "train/energy_stderr",
    "train/energy_variance",
    "train/grad_norm",
    "train/sampler/acceptance_rate",
}
FAILURE_METRIC_NEEDLES = (
    "failure_count",
    "nonfinite_count",
    "pathology_count",
    "outlier_count",
    "mismatch_count",
    "comparison_error_count",
    "missing_key_count",
    "extra_key_count",
    "near_zero_count",
)

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

RUN_INDEX_COLUMNS = [
    "final_run_id",
    "final_eval_attempt_id",
    "source_champion_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "model_seed",
    "sampler_seed",
    "eval_seed",
    "train_status",
    "eval_status",
    "n_eval_tasks_success",
    "n_eval_tasks_failed",
    "train_wall_time_sec",
    "eval_wall_time_sec",
]

ARCHITECTURE_SUMMARY_COLUMNS = [
    "basis_class",
    "normalization",
    "winner_kind",
    "n_success",
    "n_expected",
    "energy_error_median",
    "energy_error_q25",
    "energy_error_q75",
    "local_energy_var_median",
    "pathology_fraction_median",
    "tail_outlier_fraction_median",
    "trace_failure_count_total",
    "major_failure_mode",
]

ENERGY_BY_RUN_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "energy_mean",
    "energy_stderr",
    "energy_error",
    "kinetic_mean",
    "harmonic_trap_mean",
    "electron_electron_mean",
    "virial_residual",
    "virial_relative_residual",
    "local_energy_var",
    "finite_fraction",
    "pathology_fraction",
]

LOCAL_ENERGY_HISTOGRAM_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "bin_left",
    "bin_right",
    "bin_center",
    "count",
    "density",
]

CUSP_PROFILE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "com_id",
    "direction_id",
    "r12",
    "local_energy_median",
    "local_energy_q25",
    "local_energy_q75",
    "logabs_median",
    "logabs_q25",
    "logabs_q75",
    "d_logabs_dr_median",
    "d_logabs_dr_q25",
    "d_logabs_dr_q75",
    "target_d_logabs_dr",
    "finite_fraction",
]

TAIL_PROFILE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "com_id",
    "tail_path",
    "radius",
    "local_energy_median",
    "local_energy_q05",
    "local_energy_q25",
    "local_energy_q75",
    "local_energy_q85",
    "logabs_median",
    "logabs_q25",
    "logabs_q75",
    "exact_logabs",
    "outlier_fraction",
    "finite_fraction",
]

STRATIFIED_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "stratum",
    "local_energy_median",
    "local_energy_var",
    "abs_local_energy_q95",
    "abs_local_energy_q99",
    "pathology_fraction",
    "finite_fraction",
    "median_abs_energy_error",
]

HOOKE_ORBITAL_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "com_bin",
    "r12_bin",
    "r12_center",
    "R_norm_center",
    "local_energy_median",
    "local_energy_q25",
    "local_energy_q75",
    "abs_energy_error_median",
    "finite_fraction",
    "pathology_fraction",
]

SYMMETRY_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "symmetry_task",
    "logabs_error_max",
    "logabs_error_median",
    "sign_mismatch_count",
    "parity_mismatch_count",
    "finite_fraction",
]

TRACE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "trace_kind",
    "layer",
    "key",
    "rms_q95",
    "rms_q99",
    "max_abs",
    "nonfinite_count",
    "compared_entry_count",
    "comparison_error_count",
    "max_equivariance_error",
]

TRAINING_CURVE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "step",
    "energy_mean",
    "energy_stderr",
    "local_energy_var",
    "acceptance_rate",
    "grad_norm",
    "wall_time_sec",
]

RESOURCE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "train_wall_time_sec",
    "eval_wall_time_sec",
    "peak_memory_mb",
    "device_type",
]

FAILURE_COLUMNS = [
    "final_run_id",
    "basis_class",
    "normalization",
    "winner_kind",
    "seed_index",
    "task",
    "severity",
    "failure_mode",
    "metric_name",
    "metric_value",
    "threshold",
    "geometry_context",
]


def _resolve_final_grid_attempt_id(results_root: Path, requested: str | None) -> str:
    """Return the explicitly requested or latest planned final grid."""

    final_grid_stage = stage_dir(results_root, STAGE_FINAL_GRID)
    attempt_id = requested or latest_attempt_id(final_grid_stage)
    if attempt_id is None:
        raise FileNotFoundError(f"no final-grid attempts under {final_grid_stage}")
    final_jobs_path = final_grid_stage / attempt_id / "final_jobs.csv"
    if not final_jobs_path.is_file():
        raise FileNotFoundError(f"final-grid attempt has no final_jobs.csv: {final_jobs_path}")
    manifest_path = final_grid_stage / attempt_id / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"final-grid attempt has no manifest.json: {manifest_path}")
    return attempt_id


def _planned_final_run_ids(results_root: Path, final_grid_attempt_id: str) -> list[str]:
    """Return final-run ids listed by one final-grid plan."""

    final_jobs_path = stage_dir(results_root, STAGE_FINAL_GRID) / final_grid_attempt_id / "final_jobs.csv"
    return [str(row["final_run_id"]) for row in _read_csv(final_jobs_path)]


def _final_grid_manifest(results_root: Path, final_grid_attempt_id: str) -> dict[str, Any]:
    """Return manifest for one resolved final-grid plan."""

    manifest_path = stage_dir(results_root, STAGE_FINAL_GRID) / final_grid_attempt_id / "manifest.json"
    manifest = _load_json_if_present(manifest_path)
    if manifest.get("stage") != STAGE_FINAL_GRID:
        raise ValueError(f"manifest {manifest_path} is not a {STAGE_FINAL_GRID} manifest")
    return manifest


def _source_final_grid_attempt_id(attempt_dir: Path) -> str | None:
    """Return final-grid attempt id recorded by one final-eval attempt."""

    source = _load_json_if_present(attempt_dir / "source_final_grid_attempt.json")
    attempt_id = str(source.get("final_grid_attempt_id", "")).strip()
    if attempt_id:
        return attempt_id
    attempt_path = str(source.get("final_grid_attempt_dir", "")).strip()
    return Path(attempt_path).name if attempt_path else None


def _iter_final_eval_attempts(
    results_root: Path,
    final_eval_attempt_id: str | None,
    final_grid_attempt_id: str,
) -> list[tuple[str, str, Path]]:
    """Return final-eval attempts belonging to one planned final grid."""

    eval_stage = stage_dir(results_root, STAGE_FINAL_EVAL)
    if not eval_stage.is_dir():
        return []
    attempts = []
    for final_run_id in _planned_final_run_ids(results_root, final_grid_attempt_id):
        run_dir = eval_stage / final_run_id
        attempt_id = final_eval_attempt_id
        if attempt_id is None:
            attempt_id = latest_attempt_id(run_dir)
            if attempt_id is None:
                continue
        attempt_dir = run_dir / attempt_id
        if (
            attempt_dir.is_dir()
            and _source_final_grid_attempt_id(attempt_dir) == final_grid_attempt_id
        ):
            attempts.append((final_run_id, attempt_id, attempt_dir))
    return attempts


def _winner_kind(job: dict[str, Any]) -> str:
    raw = str(job.get("winner_kind", ""))
    return "energy" if raw == "energy" else "stability"


def _axis_names(manifest: dict[str, Any], key: str) -> tuple[str, ...]:
    raw = manifest.get(key, [])
    if not isinstance(raw, list | tuple):
        return ()
    return tuple(str(axis) for axis in raw if str(axis))


def _major_axes(manifest: dict[str, Any]) -> tuple[str, ...]:
    return _axis_names(manifest, "major_axes")


def _minor_axes(manifest: dict[str, Any]) -> tuple[str, ...]:
    return _axis_names(manifest, "minor_axes")


def _axis_columns_from_contexts(contexts: Sequence[dict[str, Any]]) -> list[str]:
    """Return stable configured-axis columns present in the final grid."""

    columns: list[str] = []
    for context in contexts:
        manifest = context.get("final_grid_manifest", {})
        for axis in (*_major_axes(manifest), *_minor_axes(manifest)):
            if axis not in columns:
                columns.append(axis)
    if columns:
        return columns
    for context in contexts:
        job = context.get("job", {})
        choices = job.get("choices", {})
        if isinstance(choices, dict):
            for axis in choices:
                axis = str(axis)
                if axis not in columns:
                    columns.append(axis)
    return columns


def _report_axes(manifest: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return the two axes used by legacy report grid aliases."""

    major_axes = _major_axes(manifest)
    major = set(major_axes)
    if {"basis", "update_normalization", "feature_normalization"}.issubset(major):
        return REPORT_ROW_KEY, REPORT_COL_KEY
    minor_axes = _minor_axes(manifest)
    row_axis = major_axes[0] if major_axes else None
    col_axis = major_axes[1] if len(major_axes) > 1 else (minor_axes[0] if minor_axes else None)
    return row_axis, col_axis


def _report_axis_values(job: dict[str, Any], manifest: dict[str, Any]) -> tuple[str, str]:
    """Return report-only two-axis labels, preserving raw axes separately."""

    major = set(_major_axes(manifest))
    if {"basis", "update_normalization", "feature_normalization"}.issubset(major):
        basis = _axis_value(job, "basis") or _basis_class(job)
        update = _axis_value(job, "update_normalization")
        feature = _axis_value(job, "feature_normalization")
        basis_update = "+".join(value for value in (basis, update) if value)
        return basis_update or _basis_class(job), feature or str(job.get("normalization", ""))

    configured_major = _major_axes(manifest)
    configured_minor = _minor_axes(manifest)
    row_axis = configured_major[0] if configured_major else None
    col_axis = configured_major[1] if len(configured_major) > 1 else (
        configured_minor[0] if configured_minor else None
    )
    return (
        _axis_value(job, row_axis) or _basis_class(job),
        _axis_value(job, col_axis) or str(job.get("normalization", "")),
    )


def _axis_value(job: dict[str, Any], axis: str | None) -> str:
    if not axis:
        return ""
    choices = job.get("choices", {})
    if isinstance(choices, dict) and axis in choices:
        return str(choices[axis])
    if axis in job:
        return str(job[axis])
    source_champion = job.get("source_champion", {})
    if isinstance(source_champion, dict) and axis in source_champion:
        return str(source_champion[axis])
    return ""


def _config_id(job: dict[str, Any]) -> str:
    source_champion = job.get("source_champion", {})
    if isinstance(source_champion, dict) and source_champion.get("config_id"):
        return str(source_champion["config_id"])
    return str(job.get("config_id", job.get("source_scan_run_id", "")))


def _axis_row(context: dict[str, Any]) -> dict[str, Any]:
    job = context["job"]
    manifest = context.get("final_grid_manifest", {})
    row: dict[str, Any] = {
        "major_id": job.get("major_id", ""),
        "minor_id": job.get("minor_id", ""),
        "config_id": _config_id(job),
    }
    for axis in (*_major_axes(manifest), *_minor_axes(manifest)):
        row[axis] = _axis_value(job, axis)
    if not any(axis in row for axis in (*_major_axes(manifest), *_minor_axes(manifest))):
        choices = job.get("choices", {})
        if isinstance(choices, dict):
            row.update({str(axis): value for axis, value in choices.items()})
    return row


def _basis_class(job: dict[str, Any]) -> str:
    return str(job.get("basis_envelope", job.get("architecture", "")))


def _seed_index(job: dict[str, Any]) -> str:
    return str(job.get("replicate_index", ""))


def _minor_hparams(job: dict[str, Any]) -> str:
    keys = ("architecture", "lr", "channels")
    return ";".join(f"{key}={job.get(key, '')}" for key in keys if job.get(key, "") != "")


def _run_context(final_run_id: str, attempt_id: str, attempt_dir: Path) -> dict[str, Any]:
    job = _load_json_if_present(attempt_dir / "source_final_job.json")
    source_final_grid = _load_json_if_present(attempt_dir / "source_final_grid_attempt.json")
    final_grid_dir = Path(str(source_final_grid.get("final_grid_attempt_dir", "")))
    final_grid_manifest = _load_json_if_present(final_grid_dir / "manifest.json")
    checkpoint = _load_json_if_present(attempt_dir / "evaluated_checkpoint.json")
    source_train = _load_json_if_present(attempt_dir / "source_final_train_attempt.json")
    eval_status_json = _load_json_if_present(attempt_dir / "status.json")
    train_attempt_dir = Path(str(source_train.get("final_train_attempt_dir", "")))
    train_status_json = _load_json_if_present(train_attempt_dir / "status.json")
    train_metadata = _load_json_if_present(train_attempt_dir / "metadata.json")
    eval_metrics = _read_metrics_jsonl_selected(attempt_dir / "metrics.jsonl", _keep_eval_metric)
    train_metrics = _read_metrics_jsonl_selected(train_attempt_dir / "metrics.jsonl", _keep_train_metric)
    return {
        "final_run_id": final_run_id,
        "attempt_id": attempt_id,
        "attempt_dir": attempt_dir,
        "job": job,
        "source_final_grid": source_final_grid,
        "final_grid_manifest": final_grid_manifest,
        "checkpoint": checkpoint,
        "source_train": source_train,
        "train_attempt_dir": train_attempt_dir,
        "train_status_json": train_status_json,
        "train_metadata": train_metadata,
        "eval_status": str(eval_status_json.get("status", _status_of(attempt_dir))),
        "eval_status_json": eval_status_json,
        "eval_metrics": eval_metrics,
        "eval_metric_map": _metric_map(eval_metrics),
        "train_metrics": train_metrics,
    }


def _metrics_items(record: dict[str, Any]) -> tuple[Any, Any, Any]:
    namespace = str(record.get("namespace", "")).strip("/")
    step = record.get("step", "")
    if "metrics" in record:
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            return step, namespace, ()
        return step, namespace, metrics.items()
    if "metric" in record and "value" in record:
        return step, namespace, ((record["metric"], record["value"]),)
    return step, namespace, ()


def _read_metrics_jsonl_selected(path: Path, keep: Any) -> list[dict[str, Any]]:
    """Read only metric streams that feed final report compact tables."""

    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step, namespace, items = _metrics_items(record)
        for key, value in items:
            metric = str(key)
            full_key = f"{namespace}/{metric}" if namespace else metric
            if keep(full_key):
                rows.append({"step": step, "namespace": namespace, "metric": metric, "value": _csv_value(value)})
    return rows


def _keep_eval_metric(key: str) -> bool:
    """Return whether one eval metric reaches compact final-report tables."""

    if key in EVAL_EXACT_METRICS:
        return True
    if key.startswith("diagnostics/") or key.startswith("eval/perf/"):
        return True
    if key.endswith("/status/task_success") or key.endswith("/status/task_failed"):
        return True
    metric = key.rsplit("/", maxsplit=1)[-1]
    return metric.endswith("finite_fraction") or any(needle in metric for needle in FAILURE_METRIC_NEEDLES)


def _keep_train_metric(key: str) -> bool:
    """Return whether one train metric reaches compact final-report tables."""

    return key in TRAIN_EXACT_METRICS or key.startswith("train/perf/")


def _base_row(context: dict[str, Any]) -> dict[str, Any]:
    job = context["job"]
    axis_row = _axis_row(context)
    basis_update, feature_normalization = _report_axis_values(job, context.get("final_grid_manifest", {}))
    return {
        "final_run_id": context["final_run_id"],
        "source_champion_id": job.get("source_champion_id", ""),
        **axis_row,
        REPORT_AXIS_COLUMN: basis_update,
        "basis_class": basis_update,
        "normalization": feature_normalization,
        "winner_kind": _winner_kind(job),
        "seed_index": _seed_index(job),
        "model_seed": job.get("final_train_model_seed", ""),
        "sampler_seed": job.get("final_train_sampler_seed", ""),
        "eval_seed": job.get("final_eval_sampler_seed", ""),
    }


def _finite_fraction(values: Sequence[Any]) -> float | None:
    if not values:
        return None
    finite = sum(1 for value in values if str(value).lower() != "false")
    return finite / len(values)


def _is_pathological(row: dict[str, Any]) -> bool:
    finite = str(row.get("finite", "True")).lower() == "true"
    local_energy = _as_float(row.get("local_energy"))
    return (not finite) or (local_energy is not None and abs(local_energy) > PATHOLOGY_ABS_LOCAL_ENERGY)


def _task_from_namespace(namespace: str) -> str:
    if namespace.startswith("eval/"):
        return namespace[len("eval/") :]
    return namespace


def _failure_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    base = _base_row(context)
    failures: list[dict[str, Any]] = []
    status = context["eval_status"]
    if status not in {"completed", "success"}:
        failures.append({**base, "task": "run", "severity": "failed", "failure_mode": "run_status", "metric_name": "status", "metric_value": status, "threshold": "completed", "geometry_context": ""})
    for metric_name, value in context["eval_metric_map"].items():
        namespace, _, key = metric_name.rpartition("/")
        task = _task_from_namespace(namespace)
        bool_value = _as_bool(value)
        numeric = _as_float(value)
        if key == "task_failed" and bool_value:
            failures.append({**base, "task": task, "severity": "failed", "failure_mode": key, "metric_name": metric_name, "metric_value": value, "threshold": "False", "geometry_context": ""})
        if numeric is None:
            continue
        failure_key = any(needle in key for needle in ("failure_count", "nonfinite_count", "pathology_count", "outlier_count", "mismatch_count", "comparison_error_count", "missing_key_count", "extra_key_count", "near_zero_count"))
        if failure_key and numeric > 0:
            failures.append({**base, "task": task, "severity": "warning", "failure_mode": key, "metric_name": metric_name, "metric_value": value, "threshold": "0", "geometry_context": ""})
        if key.endswith("finite_fraction") and numeric < 1.0:
            failures.append({**base, "task": task, "severity": "warning", "failure_mode": key, "metric_name": metric_name, "metric_value": value, "threshold": "1.0", "geometry_context": ""})
    return failures


def _major_failure_mode(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    failed = [row for row in rows if row.get("severity") == "failed"]
    row = failed[0] if failed else rows[0]
    return f"{row.get('task', '')}:{row.get('failure_mode', '')}"


def _task_status_counts(metrics: dict[str, Any]) -> tuple[int, int]:
    success = 0
    failed = 0
    for key, value in metrics.items():
        if key.endswith("/status/task_success") and _as_bool(value):
            success += 1
        if key.endswith("/status/task_failed") and _as_bool(value):
            failed += 1
    return success, failed


def _run_index_row(context: dict[str, Any]) -> dict[str, Any]:
    base = _base_row(context)
    checkpoint = context["checkpoint"]
    source_train = context["source_train"]
    train_status = "checkpoint_selected" if checkpoint.get("resolved_checkpoint_dir") else ("attempt_recorded" if source_train.get("final_train_attempt_id") else "")
    n_success, n_failed = _task_status_counts(context["eval_metric_map"])
    return {
        **base,
        "final_eval_attempt_id": context["attempt_id"],
        "train_status": train_status,
        "eval_status": context["eval_status"],
        "n_eval_tasks_success": n_success,
        "n_eval_tasks_failed": n_failed,
        "train_wall_time_sec": _format_number(_duration_from_status(context["train_status_json"])),
        "eval_wall_time_sec": _format_number(_duration_from_status(context["eval_status_json"])),
    }


def _energy_row(context: dict[str, Any]) -> dict[str, Any]:
    base = _base_row(context)
    metrics = context["eval_metric_map"]
    energy = _as_float(metrics.get("eval/energy/local_energy_mean"))
    kinetic = _as_float(metrics.get("eval/energy/term/kinetic_mean"))
    harmonic = _as_float(metrics.get("eval/energy/term/harmonic_trap_mean"))
    electron_electron = _as_float(metrics.get("eval/energy/term/electron_electron_mean"))
    residual = _as_float(metrics.get("eval/energy/virial_residual"))
    relative = _as_float(metrics.get("eval/energy/virial_relative_residual"))
    virial = (
        {"residual": residual, "relative_residual": relative}
        if residual is not None and relative is not None
        else derive_virial_metrics(kinetic, harmonic, electron_electron)
    )
    n_finite = _as_float(metrics.get("eval/energy/local_energy_n_finite"))
    n_total = _as_float(metrics.get("eval/energy/local_energy_n_total"))
    finite_fraction = _as_float(metrics.get("eval/energy/local_energy_finite_fraction"))
    if finite_fraction is None and n_finite is not None and n_total:
        finite_fraction = n_finite / n_total
    pathology_count = _as_float(metrics.get("eval/energy/local_energy_pathology_count"))
    pathology_fraction = None
    if pathology_count is not None and n_total:
        pathology_fraction = pathology_count / n_total
    return {
        **base,
        "energy_mean": _format_number(energy),
        "energy_stderr": _format_number(_as_float(metrics.get("eval/energy/local_energy_stderr"))),
        "energy_error": _format_number(None if energy is None else energy - EXACT_HOOKE_ENERGY),
        "kinetic_mean": _format_number(kinetic),
        "harmonic_trap_mean": _format_number(harmonic),
        "electron_electron_mean": _format_number(electron_electron),
        "virial_residual": _format_number(virial["residual"]),
        "virial_relative_residual": _format_number(virial["relative_residual"]),
        "local_energy_var": _format_number(_as_float(metrics.get("eval/energy/local_energy_variance"))),
        "finite_fraction": _format_number(finite_fraction),
        "pathology_fraction": _format_number(pathology_fraction),
    }


def _bin_edges(values: Sequence[float], n_bins: int = DEFAULT_HISTOGRAM_BINS) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if low == high:
        pad = max(1.0, abs(low) * 0.05)
        low -= pad
        high += pad
    width = (high - low) / n_bins
    return [low + index * width for index in range(n_bins + 1)]


def _histogram(values: Sequence[float], edges: Sequence[float]) -> list[tuple[float, float, int, float]]:
    if not values or len(edges) < 2:
        return []
    counts = [0 for _ in range(len(edges) - 1)]
    for value in values:
        if value < edges[0] or value > edges[-1]:
            continue
        index = len(edges) - 2 if value == edges[-1] else max(0, min(len(edges) - 2, int((value - edges[0]) / (edges[-1] - edges[0]) * (len(edges) - 1))))
        counts[index] += 1
    total = sum(counts)
    rows = []
    for index, count in enumerate(counts):
        left = edges[index]
        right = edges[index + 1]
        width = right - left
        density = 0.0 if total == 0 or width <= 0 else count / (total * width)
        rows.append((left, right, count, density))
    return rows


def _record_context(context: dict[str, Any], task: str, row: dict[str, Any]) -> dict[str, Any]:
    base = _base_row(context)
    out = {**base, **row, "task": task}
    if "radius" in out:
        out["R_norm"] = out.get("radius", "")
        out["R_norm_bin"] = _bin_value(out.get("radius"))
    if "r12" in out:
        out["r12_bin"] = _bin_value(out.get("r12"))
    if "center_of_mass_id" in out:
        out["com_id"] = out.get("center_of_mass_id", "")
    return out


def _bin_value(value: Any, *, width: float = 0.5) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return ""
    low = math.floor(numeric / width) * width
    high = low + width
    return f"[{low:.2g},{high:.2g})"


def _center_of_bin(label: str) -> str:
    if not label.startswith("[") or "," not in label:
        return ""
    left, right = label.strip(")").strip("[").split(",", 1)
    left_f = _as_float(left)
    right_f = _as_float(right)
    if left_f is None or right_f is None:
        return ""
    return _format_number((left_f + right_f) / 2.0)


def _task_records(context: dict[str, Any], task: str, filename: str) -> list[dict[str, Any]]:
    return [_record_context(context, task, row) for row in _read_csv(context["attempt_dir"] / task / filename)]


def _local_energy_values(contexts: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    values_by_run: dict[str, list[float]] = {}
    for context in contexts:
        rows = _read_csv(context["attempt_dir"] / "energy" / "mcmc_energy_samples.csv")
        values_by_run[context["final_run_id"]] = [value for value in (_as_float(row.get("local_energy")) for row in rows) if value is not None]
    return values_by_run


def _local_energy_histograms(contexts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    values_by_run = _local_energy_values(contexts)
    group_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    groups_by_run: dict[str, tuple[str, str, str]] = {}
    for context in contexts:
        base = _base_row(context)
        group = (str(base.get("basis_class", "")), str(base.get("normalization", "")), str(base.get("winner_kind", "")))
        groups_by_run[context["final_run_id"]] = group
        group_values[group].extend(values_by_run.get(context["final_run_id"], []))
    edges_by_group = {group: _bin_edges(values) for group, values in group_values.items()}
    rows = []
    for context in contexts:
        base = _base_row(context)
        edges = edges_by_group.get(groups_by_run.get(context["final_run_id"], ("", "", "")), [])
        for left, right, count, density in _histogram(values_by_run.get(context["final_run_id"], []), edges):
            rows.append({**base, "bin_left": _format_number(left), "bin_right": _format_number(right), "bin_center": _format_number((left + right) / 2.0), "count": count, "density": _format_number(density)})
    return rows


CUSP_DLOGABS_KEYS = ("d_logabs_dr", "dlogabs_dr", "radial_dlogabs", "radial_logabs_derivative")


def _first_float(row: dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _finite_difference(points: Sequence[tuple[float, float]], index: int) -> float | None:
    if len(points) < 2:
        return None
    if index == 0:
        left, right = points[0], points[1]
    elif index == len(points) - 1:
        left, right = points[-2], points[-1]
    else:
        left, right = points[index - 1], points[index + 1]
    dx = right[0] - left[0]
    if dx == 0.0:
        return None
    return (right[1] - left[1]) / dx


def _fill_cusp_derivative_fallback(rows: list[dict[str, Any]]) -> None:
    by_path: dict[tuple[str, str, str], list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        row.setdefault("target_d_logabs_dr", "0.5")
        if row.get("d_logabs_dr_median"):
            continue
        r12 = _as_float(row.get("r12"))
        logabs = _as_float(row.get("logabs_median"))
        if r12 is None or logabs is None:
            continue
        by_path[(str(row.get("final_run_id", "")), str(row.get("com_id", "")), str(row.get("direction_id", "")))].append((r12, row))
    for path_rows in by_path.values():
        path_rows = sorted(path_rows, key=lambda item: item[0])
        points = [(r12, _as_float(row.get("logabs_median"))) for r12, row in path_rows]
        finite_points = [(r12, logabs) for r12, logabs in points if logabs is not None]
        if len(finite_points) != len(path_rows):
            continue
        for index, (_r12, row) in enumerate(path_rows):
            derivative = _finite_difference(finite_points, index)
            if derivative is None:
                continue
            formatted = _format_number(derivative)
            row["d_logabs_dr_median"] = formatted
            row["d_logabs_dr_q25"] = formatted
            row["d_logabs_dr_q75"] = formatted


def _cusp_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _task_records(context, "cusp", "cusp_profiles.csv")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("com_id", row.get("center_of_mass_id", ""))), str(row.get("direction_id", "")), str(row.get("r12", "")))].append(row)
    out = []
    base = _base_row(context)
    for (com_id, direction_id, r12), group in sorted(groups.items()):
        energies = [_as_float(row.get("local_energy")) for row in group]
        logabs = [_as_float(row.get("logabs")) for row in group]
        derivatives = [_first_float(row, CUSP_DLOGABS_KEYS) for row in group]
        out.append({
            **base,
            "com_id": com_id,
            "direction_id": direction_id,
            "r12": r12,
            "local_energy_median": _format_number(_median(energies)),
            "local_energy_q25": _format_number(_quantile(energies, 0.25)),
            "local_energy_q75": _format_number(_quantile(energies, 0.75)),
            "logabs_median": _format_number(_median(logabs)),
            "logabs_q25": _format_number(_quantile(logabs, 0.25)),
            "logabs_q75": _format_number(_quantile(logabs, 0.75)),
            "d_logabs_dr_median": _format_number(_median(derivatives)),
            "d_logabs_dr_q25": _format_number(_quantile(derivatives, 0.25)),
            "d_logabs_dr_q75": _format_number(_quantile(derivatives, 0.75)),
            "target_d_logabs_dr": "0.5",
            "finite_fraction": _format_number(_finite_fraction([row.get("finite", "True") for row in group])),
        })
    _fill_cusp_derivative_fallback(out)
    return out


def _tail_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _task_records(context, "tail", "tail_profiles.csv")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        tail_path = str(row.get("tail_path", f"direction-{row.get('direction_id', '')}/relative-{row.get('relative_direction_id', '')}"))
        groups[(str(row.get("com_id", "")), tail_path, str(row.get("radius", "")))].append(row)
    out = []
    base = _base_row(context)
    for (com_id, tail_path, radius), group in sorted(groups.items()):
        energies = [_as_float(row.get("local_energy")) for row in group]
        logabs = [_as_float(row.get("logabs")) for row in group]
        out.append({
            **base,
            "com_id": com_id,
            "tail_path": tail_path,
            "radius": radius,
            "local_energy_median": _format_number(_median(energies)),
            "local_energy_q05": _format_number(_quantile(energies, 0.05)),
            "local_energy_q25": _format_number(_quantile(energies, 0.25)),
            "local_energy_q75": _format_number(_quantile(energies, 0.75)),
            "local_energy_q85": _format_number(_quantile(energies, 0.85)),
            "logabs_median": _format_number(_median(logabs)),
            "logabs_q25": _format_number(_quantile(logabs, 0.25)),
            "logabs_q75": _format_number(_quantile(logabs, 0.75)),
            "exact_logabs": next((row.get("exact_logabs", "") for row in group if row.get("exact_logabs", "") != ""), ""),
            "outlier_fraction": _format_number(sum(1 for row in group if _is_pathological(row)) / len(group)),
            "finite_fraction": _format_number(_finite_fraction([row.get("finite", "True") for row in group])),
        })
    return out


def _stratified_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _task_records(context, "stratified_geometry", "stratified_metrics.csv")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("stratum", ""))].append(row)
        groups["all"].append(row)
    out = []
    base = _base_row(context)
    for stratum, group in sorted(groups.items()):
        energies = [_as_float(row.get("local_energy")) for row in group]
        abs_energies = [None if value is None else abs(value) for value in energies]
        abs_errors = [None if value is None else abs(value - EXACT_HOOKE_ENERGY) for value in energies]
        out.append({
            **base,
            "stratum": stratum,
            "local_energy_median": _format_number(_median(energies)),
            "local_energy_var": _format_number(_variance(energies)),
            "abs_local_energy_q95": _format_number(_quantile(abs_energies, 0.95)),
            "abs_local_energy_q99": _format_number(_quantile(abs_energies, 0.99)),
            "pathology_fraction": _format_number(sum(1 for row in group if _is_pathological(row)) / len(group)),
            "finite_fraction": _format_number(_finite_fraction([row.get("finite", "True") for row in group])),
            "median_abs_energy_error": _format_number(_median(abs_errors)),
        })
    return out


def _hooke_orbital_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _task_records(context, "hooke_orbital", "hooke_orbital_metrics.csv")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("R_norm_bin", _bin_value(row.get("radius")))), str(row.get("r12_bin", _bin_value(row.get("r12")))))].append(row)
    out = []
    base = _base_row(context)
    for (com_bin, r12_bin), group in sorted(groups.items()):
        energies = [_as_float(row.get("local_energy")) for row in group]
        abs_errors = [None if value is None else abs(value - EXACT_HOOKE_ENERGY) for value in energies]
        out.append({
            **base,
            "com_bin": com_bin,
            "r12_bin": r12_bin,
            "r12_center": _center_of_bin(r12_bin),
            "R_norm_center": _center_of_bin(com_bin),
            "local_energy_median": _format_number(_median(energies)),
            "local_energy_q25": _format_number(_quantile(energies, 0.25)),
            "local_energy_q75": _format_number(_quantile(energies, 0.75)),
            "abs_energy_error_median": _format_number(_median(abs_errors)),
            "finite_fraction": _format_number(_finite_fraction([row.get("finite", "True") for row in group])),
            "pathology_fraction": _format_number(sum(1 for row in group if _is_pathological(row)) / len(group)),
        })
    return out


def _symmetry_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    base = _base_row(context)
    for task in ("full_model_antisymmetry", "spatial_exchange_symmetry", "rotation_consistency"):
        rows = _task_records(context, task, "transform_records.csv")
        if not rows:
            continue
        errors = [_as_float(row.get("logabs_abs_error", row.get("logabs_error", row.get("max_abs_error")))) for row in rows]
        out.append({
            **base,
            "symmetry_task": task,
            "logabs_error_max": _format_number(_max(errors)),
            "logabs_error_median": _format_number(_median(errors)),
            "sign_mismatch_count": sum(1 for row in rows if _as_bool(row.get("sign_mismatch"))),
            "parity_mismatch_count": sum(1 for row in rows if _as_bool(row.get("parity_mismatch"))),
            "finite_fraction": _format_number(_finite_fraction([row.get("finite", "True") for row in rows])),
        })
    return out


def _trace_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    base = _base_row(context)
    metrics = context["eval_metric_map"]
    for task in ("trace_equivariance", "feature_trace_stability", "readout_trace_stability"):
        rows = _task_records(context, task, "trace_records.csv")
        if not rows and task == "trace_equivariance":
            key = ""
            out.append({**base, "trace_kind": task, "layer": "", "key": key, "rms_q95": "", "rms_q99": "", "max_abs": "", "nonfinite_count": "", "compared_entry_count": metrics.get("eval/trace_equivariance/compared_entry_count", ""), "comparison_error_count": metrics.get("eval/trace_equivariance/comparison_error_count", ""), "max_equivariance_error": metrics.get("eval/trace_equivariance/max_abs_error", "")})
            continue
        for row in rows:
            key = str(row.get("entry_key", row.get("key", "")))
            layer = key.split("/", 1)[0] if key else ""
            out.append({
                **base,
                "trace_kind": task,
                "layer": layer,
                "key": key,
                "rms_q95": row.get("q95_abs", row.get("rms_q95", "")),
                "rms_q99": row.get("q99_abs", row.get("rms_q99", "")),
                "max_abs": row.get("max_abs", row.get("max_abs_error", "")),
                "nonfinite_count": row.get("nonfinite_count", row.get("readout_nonfinite_count", "")),
                "compared_entry_count": row.get("compared_entry_count", metrics.get(f"eval/{task}/compared_entry_count", "")),
                "comparison_error_count": row.get("comparison_error_count", metrics.get(f"eval/{task}/comparison_error_count", "")),
                "max_equivariance_error": row.get("max_abs_error", metrics.get(f"eval/{task}/max_abs_error", "")),
            })
    return out


def _training_curve_summary(context: dict[str, Any]) -> list[dict[str, Any]]:
    base = _base_row(context)
    by_step: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in context["train_metrics"]:
        step = str(row.get("step", ""))
        namespace = row.get("namespace", "")
        metric = row.get("metric", "")
        if namespace == "train" and metric == "energy":
            by_step[step]["energy_mean"] = row.get("value", "")
        elif namespace == "train" and metric == "energy_stderr":
            by_step[step]["energy_stderr"] = row.get("value", "")
        elif namespace == "train" and metric == "energy_variance":
            by_step[step]["local_energy_var"] = row.get("value", "")
        elif namespace == "train" and metric == "grad_norm":
            by_step[step]["grad_norm"] = row.get("value", "")
        elif namespace == "train/sampler" and metric == "acceptance_rate":
            by_step[step]["acceptance_rate"] = row.get("value", "")
    return [
        {**base, "step": step, "energy_mean": data.get("energy_mean", ""), "energy_stderr": data.get("energy_stderr", ""), "local_energy_var": data.get("local_energy_var", ""), "acceptance_rate": data.get("acceptance_rate", ""), "grad_norm": data.get("grad_norm", ""), "wall_time_sec": data.get("wall_time_sec", "")}
        for step, data in sorted(by_step.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0)
    ]


def _resource_row(context: dict[str, Any]) -> dict[str, Any]:
    base = _base_row(context)
    metadata = context["train_metadata"]
    runtime = metadata.get("runtime", {}) if isinstance(metadata.get("runtime"), dict) else {}
    train_metric_map = _metric_map(context["train_metrics"])
    peak_memory = train_metric_map.get("runtime/peak_memory_mb", metadata.get("peak_memory_mb", ""))
    return {
        **base,
        "train_wall_time_sec": _format_number(_duration_from_status(context["train_status_json"])),
        "eval_wall_time_sec": _format_number(_duration_from_status(context["eval_status_json"])),
        "peak_memory_mb": peak_memory,
        "device_type": runtime.get("device", metadata.get("device", "")),
    }


def _attempt_device(attempt_dir: Path) -> str:
    """Return the recorded runtime device for one attempt directory."""

    metadata = _load_json_if_present(Path(attempt_dir) / "metadata.json")
    runtime = metadata.get("runtime", {}) if isinstance(metadata.get("runtime"), dict) else {}
    return str(runtime.get("device", metadata.get("device", "")))


def _cost_tables_rows(contexts: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project final-train and final-eval run metrics into cost rows."""

    cost_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for context in contexts:
        axes = {key: value for key, value in _base_row(context).items() if key != "final_run_id"}
        run_id = str(context["final_run_id"])
        eval_attempt_id = str(context["attempt_id"])
        eval_device = _attempt_device(context["attempt_dir"])
        cost_rows.append(
            cost_by_run_row(
                context["train_metrics"],
                run_id=run_id,
                attempt_id=str(context["source_train"].get("final_train_attempt_id", "")),
                stage="final_train",
                status=str(context["train_status_json"].get("status", "")),
                device_type=_attempt_device(context["train_attempt_dir"]),
                axes=axes,
            )
        )
        cost_rows.append(
            cost_by_run_row(
                context["eval_metrics"],
                run_id=run_id,
                attempt_id=eval_attempt_id,
                stage="final_eval",
                status=str(context["eval_status"]),
                device_type=eval_device,
                axes=axes,
            )
        )
        task_rows.extend(
            cost_by_task_rows(
                context["eval_metrics"],
                run_id=run_id,
                attempt_id=eval_attempt_id,
                stage="final_eval",
                device_type=eval_device,
            )
        )
    return cost_rows, task_rows


def _architecture_summary(
    energy_rows: Sequence[dict[str, Any]],
    tail_rows: Sequence[dict[str, Any]],
    trace_rows: Sequence[dict[str, Any]],
    failure_rows: Sequence[dict[str, Any]],
    *,
    major_axes: Sequence[str],
    axis_columns: Sequence[str],
    expected_final_seeds: int,
) -> list[dict[str, Any]]:
    group_axes = tuple(axis for axis in major_axes if axis)
    if not group_axes:
        group_axes = ("basis_class", "normalization")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in energy_rows:
        groups[(*[str(row.get(axis, "")) for axis in group_axes], str(row["winner_kind"]))].append(row)
    tail_by_run = defaultdict(list)
    for row in tail_rows:
        tail_by_run[row["final_run_id"]].append(_as_float(row.get("outlier_fraction")))
    trace_by_run = defaultdict(list)
    for row in trace_rows:
        trace_by_run[row["final_run_id"]].append(_as_float(row.get("comparison_error_count")))
        trace_by_run[row["final_run_id"]].append(_as_float(row.get("nonfinite_count")))
    failures_by_group: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in failure_rows:
        failures_by_group[(*[str(row.get(axis, "")) for axis in group_axes], str(row["winner_kind"]))].append(row)

    out = []
    for key, rows in sorted(groups.items()):
        representative = rows[0]
        run_ids = [row["final_run_id"] for row in rows]
        energy_errors = [_as_float(row.get("energy_error")) for row in rows]
        variances = [_as_float(row.get("local_energy_var")) for row in rows]
        pathologies = [_as_float(row.get("pathology_fraction")) for row in rows]
        tail_outliers = [_median(tail_by_run[run_id]) for run_id in run_ids]
        trace_failures = [_sum(trace_by_run[run_id]) for run_id in run_ids]
        successful = [row for row in rows if _as_float(row.get("energy_mean")) is not None]
        out.append({
            **{column: representative.get(column, "") for column in (*AXIS_PROVENANCE_COLUMNS, *axis_columns)},
            REPORT_AXIS_COLUMN: representative.get(REPORT_AXIS_COLUMN, ""),
            "basis_class": representative.get("basis_class", ""),
            "normalization": representative.get("normalization", ""),
            "winner_kind": key[-1],
            "n_success": len(successful),
            "n_expected": expected_final_seeds,
            "energy_error_median": _format_number(_median(energy_errors)),
            "energy_error_q25": _format_number(_quantile(energy_errors, 0.25)),
            "energy_error_q75": _format_number(_quantile(energy_errors, 0.75)),
            "local_energy_var_median": _format_number(_median(variances)),
            "pathology_fraction_median": _format_number(_median(pathologies)),
            "tail_outlier_fraction_median": _format_number(_median(tail_outliers)),
            "trace_failure_count_total": _format_number(_sum(trace_failures)),
            "major_failure_mode": _major_failure_mode(failures_by_group.get(key, [])),
        })
    return out


def _expected_final_seeds(contexts: Sequence[dict[str, Any]]) -> int:
    for context in contexts:
        manifest = context.get("final_grid_manifest", {})
        for key in ("final_replicates", "replicates"):
            value = _as_float(manifest.get(key))
            if value is not None and value > 0:
                return int(value)
    return DEFAULT_EXPECTED_FINAL_SEEDS


def _insert_columns(columns: Sequence[str], insert_after: str, additions: Sequence[str]) -> list[str]:
    output: list[str] = []
    inserted = False
    for column in columns:
        if column not in output:
            output.append(column)
        if column == insert_after:
            for addition in additions:
                if addition not in output:
                    output.append(addition)
            inserted = True
    if not inserted:
        for addition in additions:
            if addition not in output:
                output.append(addition)
    return output


def _with_report_axis_columns(columns: Sequence[str]) -> list[str]:
    """Add report-only derived axis columns without changing raw axis columns."""

    return _insert_columns(columns, "basis_class", (REPORT_AXIS_COLUMN,))


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    _append_manifest_items(lines, payload, indent=0)
    path.write_text("\n".join(lines) + "\n")


def _append_manifest_items(lines: list[str], payload: dict[str, Any], *, indent: int) -> None:
    prefix = " " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    lines.append(f"{prefix}  {sub_key}:")
                    _append_manifest_items(lines, sub_value, indent=indent + 4)
                elif isinstance(sub_value, list):
                    lines.append(f"{prefix}  {sub_key}:")
                    for item in sub_value:
                        lines.append(f"{prefix}    - {item}")
                else:
                    lines.append(f"{prefix}  {sub_key}: {sub_value}")
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}{key}: {value}")


def collect_final_outputs(
    *,
    results_root: str | Path,
    collect_attempt_id: str | None = None,
    final_eval_attempt_id: str | None = None,
    final_grid_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Collect compact final-summary tables from final train/eval artifacts."""

    results_root = Path(results_root)
    collect_attempt_id = collect_attempt_id or new_attempt_id()
    final_grid_attempt_id = _resolve_final_grid_attempt_id(results_root, final_grid_attempt_id)
    final_grid_manifest = _final_grid_manifest(results_root, final_grid_attempt_id)
    attempt = stage_dir(results_root, STAGE_FINAL_COLLECT) / collect_attempt_id
    attempt.mkdir(parents=True, exist_ok=True)

    contexts = [
        _run_context(final_run_id, attempt_id, attempt_dir)
        for final_run_id, attempt_id, attempt_dir in _iter_final_eval_attempts(
            results_root,
            final_eval_attempt_id,
            final_grid_attempt_id,
        )
    ]
    study = study_name_from_manifest(final_grid_manifest)
    major_axes = _major_axes(final_grid_manifest)
    minor_axes = _minor_axes(final_grid_manifest)
    axis_columns = _axis_columns_from_contexts(contexts)
    axis_provenance_columns = [column for column in (*AXIS_PROVENANCE_COLUMNS, *axis_columns) if column]
    run_columns = _insert_columns(RUN_INDEX_COLUMNS, "source_champion_id", axis_provenance_columns)
    compact_columns = {
        "run_index.csv": _with_report_axis_columns(run_columns),
        "architecture_summary.csv": _with_report_axis_columns(_insert_columns(ARCHITECTURE_SUMMARY_COLUMNS, "normalization", axis_provenance_columns)),
        "energy_by_run.csv": _with_report_axis_columns(_insert_columns(ENERGY_BY_RUN_COLUMNS, "final_run_id", axis_provenance_columns)),
        "local_energy_histograms.csv": _with_report_axis_columns(_insert_columns(LOCAL_ENERGY_HISTOGRAM_COLUMNS, "final_run_id", axis_provenance_columns)),
        "cusp_profile_summary.csv": _with_report_axis_columns(_insert_columns(CUSP_PROFILE_COLUMNS, "final_run_id", axis_provenance_columns)),
        "tail_profile_summary.csv": _with_report_axis_columns(_insert_columns(TAIL_PROFILE_COLUMNS, "final_run_id", axis_provenance_columns)),
        "stratified_summary.csv": _with_report_axis_columns(_insert_columns(STRATIFIED_COLUMNS, "final_run_id", axis_provenance_columns)),
        "hooke_orbital_summary.csv": _with_report_axis_columns(_insert_columns(HOOKE_ORBITAL_COLUMNS, "final_run_id", axis_provenance_columns)),
        "symmetry_summary.csv": _with_report_axis_columns(_insert_columns(SYMMETRY_COLUMNS, "final_run_id", axis_provenance_columns)),
        "trace_summary.csv": _with_report_axis_columns(_insert_columns(TRACE_COLUMNS, "final_run_id", axis_provenance_columns)),
        "training_curve_summary.csv": _with_report_axis_columns(_insert_columns(TRAINING_CURVE_COLUMNS, "final_run_id", axis_provenance_columns)),
        "resource_summary.csv": _with_report_axis_columns(_insert_columns(RESOURCE_COLUMNS, "final_run_id", axis_provenance_columns)),
        "failure_modes.csv": _with_report_axis_columns(_insert_columns(FAILURE_COLUMNS, "final_run_id", axis_provenance_columns)),
    }
    expected_final_seeds = _expected_final_seeds(contexts)
    report_row_key, report_col_key = _report_axes(final_grid_manifest)
    resolved_eval_attempt_ids = sorted({str(context["attempt_id"]) for context in contexts})
    manifest_final_eval_attempt_id = final_eval_attempt_id
    if manifest_final_eval_attempt_id is None and len(resolved_eval_attempt_ids) == 1:
        manifest_final_eval_attempt_id = resolved_eval_attempt_ids[0]
    run_index_rows = [_run_index_row(context) for context in contexts]
    energy_rows = [_energy_row(context) for context in contexts]
    histogram_rows = _local_energy_histograms(contexts)
    cusp_rows = [row for context in contexts for row in _cusp_summary(context)]
    tail_rows = [row for context in contexts for row in _tail_summary(context)]
    stratified_rows = [row for context in contexts for row in _stratified_summary(context)]
    hooke_rows = [row for context in contexts for row in _hooke_orbital_summary(context)]
    symmetry_rows = [row for context in contexts for row in _symmetry_summary(context)]
    trace_rows = [row for context in contexts for row in _trace_summary(context)]
    training_rows = [row for context in contexts for row in _training_curve_summary(context)]
    resource_rows = [_resource_row(context) for context in contexts]
    failure_rows = [row for context in contexts for row in _failure_rows(context)]
    cost_rows, cost_task_rows = _cost_tables_rows(contexts)
    cost_axis_names = [column for column in (*major_axes, *minor_axes, REPORT_AXIS_COLUMN, "basis_class", "normalization", "winner_kind") if column]
    cost_by_run_columns = [
        *COST_BY_RUN_BASE_COLUMNS,
        "source_champion_id",
        *(column for column in axis_provenance_columns if column),
        REPORT_AXIS_COLUMN,
        "basis_class",
        "normalization",
        "winner_kind",
        "seed_index",
    ]
    architecture_rows = _architecture_summary(
        energy_rows,
        tail_rows,
        trace_rows,
        failure_rows,
        major_axes=major_axes,
        axis_columns=axis_columns,
        expected_final_seeds=expected_final_seeds,
    )

    table_specs = {
        "run_index.csv": (run_index_rows, compact_columns["run_index.csv"]),
        "architecture_summary.csv": (architecture_rows, compact_columns["architecture_summary.csv"]),
        "energy_by_run.csv": (energy_rows, compact_columns["energy_by_run.csv"]),
        "local_energy_histograms.csv": (histogram_rows, compact_columns["local_energy_histograms.csv"]),
        "cusp_profile_summary.csv": (cusp_rows, compact_columns["cusp_profile_summary.csv"]),
        "tail_profile_summary.csv": (tail_rows, compact_columns["tail_profile_summary.csv"]),
        "stratified_summary.csv": (stratified_rows, compact_columns["stratified_summary.csv"]),
        "hooke_orbital_summary.csv": (hooke_rows, compact_columns["hooke_orbital_summary.csv"]),
        "symmetry_summary.csv": (symmetry_rows, compact_columns["symmetry_summary.csv"]),
        "trace_summary.csv": (trace_rows, compact_columns["trace_summary.csv"]),
        "training_curve_summary.csv": (training_rows, compact_columns["training_curve_summary.csv"]),
        "resource_summary.csv": (resource_rows, compact_columns["resource_summary.csv"]),
        "failure_modes.csv": (failure_rows, compact_columns["failure_modes.csv"]),
        "cost_by_run.csv": (cost_rows, cost_by_run_columns),
        "cost_by_axis.csv": (cost_by_axis_rows(cost_rows, axis_names=cost_axis_names), COST_BY_AXIS_COLUMNS),
        "cost_by_task.csv": (cost_task_rows, COST_BY_TASK_COLUMNS),
    }
    for filename, (rows, columns) in table_specs.items():
        _write_csv(attempt / filename, rows, columns)

    manifest = {
        "study": study,
        "stage": STAGE_FINAL_COLLECT,
        "attempt_id": collect_attempt_id,
        "final_grid_attempt_id": final_grid_attempt_id,
        "final_eval_attempt_id": manifest_final_eval_attempt_id,
        "final_eval_attempt_ids": resolved_eval_attempt_ids,
        "final_eval_attempts": {str(context["final_run_id"]): str(context["attempt_id"]) for context in contexts},
        "n_final_eval_attempts": len(contexts),
        "major_axes": list(major_axes),
        "minor_axes": list(minor_axes),
        "axis_columns": axis_columns,
        "report_row_key": report_row_key or "basis_class",
        "report_col_key": report_col_key or "normalization",
        "expected_final_replicates": expected_final_seeds,
        "tables": {filename: len(rows) for filename, (rows, _) in table_specs.items()},
        "source_stages": {
            "final_grid": "05_final_grid",
            "final_train": "06_final_train",
            "final_eval": "07_final_eval",
        },
    }
    _write_manifest(attempt / "manifest.yaml", manifest)
    write_latest(stage_dir(results_root, STAGE_FINAL_COLLECT), collect_attempt_id)
    return {"attempt_dir": str(attempt), "manifest": manifest}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-collect arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--final-grid-attempt-id",
        default=None,
        help="Override source final grid; defaults to 05_final_grid/latest.json.",
    )
    parser.add_argument("--final-eval-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--smoke", action="store_true", help="Deprecated; use configs/smoke.yaml with the normal stack.")
    args = parser.parse_args(argv)
    if args.smoke:
        parser.error("use --grid experiments/hooke/pair_stability_v3/configs/smoke.yaml with the normal stage stack")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Collect compact final tables."""

    args = parse_args(argv)
    prefix = log_prefix()
    print(f"{prefix} final collect results_root={args.results_root}")
    if args.final_grid_attempt_id:
        print(f"{prefix} final collect using final_grid_attempt_id={args.final_grid_attempt_id}")
    else:
        print(f"{prefix} final collect using latest final-grid plan")
    if args.final_eval_attempt_id:
        print(f"{prefix} final collect using final_eval_attempt_id={args.final_eval_attempt_id}")
    else:
        print(f"{prefix} final collect using latest final-eval attempt per planned run")
    result = collect_final_outputs(
        results_root=args.results_root,
        collect_attempt_id=args.attempt_id,
        final_eval_attempt_id=args.final_eval_attempt_id,
        final_grid_attempt_id=args.final_grid_attempt_id,
    )
    manifest = result["manifest"]
    prefix = log_prefix(manifest.get("study"))
    print(
        f"{prefix} final collect consumed {manifest['n_final_eval_attempts']} "
        f"final-eval attempts -> {result['attempt_dir']}"
    )
    print(f"{prefix} final collect table rows:")
    for filename, count in manifest["tables"].items():
        print(f"{prefix}   {filename}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
