"""Plan final replicate jobs from selected champions.

Consumes a durable ``04_select`` attempt and writes a durable ``05_final_grid``
attempt. The final grid is the source of truth for final replicate indices and
the independent final train/eval seed policy.

Also writes ``task_lineage.jsonl``, a toolkit sidecar mapping each final job
to the validation/train task ids of its source champion, inherited from
``04_select``'s own sidecar (see ``experiments.toolkit.lineage``) -- the end
of Phase 5's task-id chain, from ``00_grid`` down to a final replicate. This
is additive: ``final_jobs.csv`` and ``manifest.json`` keep their stable
public schema.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

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
source_grid_from_attempt = _tpen_utils_ancestry.source_grid_from_attempt
_tpen_utils_io = sibling(__file__, 'utils.io')
read_json = _tpen_utils_io.read_json
write_json = _tpen_utils_io.write_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_FINAL_GRID = _tpen_utils_layout.STAGE_FINAL_GRID
STAGE_SELECT = _tpen_utils_layout.STAGE_SELECT
final_grid_attempt_dir = _tpen_utils_layout.final_grid_attempt_dir
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
_tpen_utils_overrides = sibling(__file__, 'utils.overrides')
AxisOverrideSpec = _tpen_utils_overrides.AxisOverrideSpec
normalize_axis_override_specs = _tpen_utils_overrides.normalize_axis_override_specs
_tpen_utils_seeds = sibling(__file__, 'utils.seeds')
final_seed_sequences = _tpen_utils_seeds.final_seed_sequences
final_seed_values = _tpen_utils_seeds.final_seed_values
seed_override_policy = _tpen_utils_seeds.seed_override_policy
seed_override_values = _tpen_utils_seeds.seed_override_values
_tpen_utils_time = sibling(__file__, 'utils.time')
STUDY_TIMEZONE = _tpen_utils_time.STUDY_TIMEZONE
new_attempt_id = _tpen_utils_time.new_attempt_id

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.toolkit import TaskLineageRow, read_task_lineage, write_task_lineage  # noqa: E402

DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
DEFAULT_GRID = STUDY_DIR / "configs" / "grid.yaml"
DEFAULT_REPLICATES = 3


def positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _resolve_selection_attempt(results_root: Path, selection_attempt_id: str | None) -> str:
    if selection_attempt_id is not None:
        return selection_attempt_id
    select_stage = stage_dir(results_root, STAGE_SELECT)
    attempt_id = latest_attempt_id(select_stage)
    if attempt_id is None:
        raise FileNotFoundError(f"no selection attempts under {select_stage}")
    return attempt_id


def read_champions(selection_dir: Path) -> list[dict[str, str]]:
    """Read selected champions from ``04_select/{attempt_id}/champions.csv``."""

    champions_path = selection_dir / "champions.csv"
    if not champions_path.is_file():
        raise FileNotFoundError(f"selection attempt has no champions.csv: {champions_path}")
    with champions_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _source_grid_manifest(results_root: Path, selection_dir: Path) -> dict[str, Any] | None:
    """Return the source ``00_grid`` manifest for this selection, if available."""

    source_grid = source_grid_from_attempt(results_root, selection_dir)
    if source_grid is None or not source_grid.manifest_path.is_file():
        return None
    return source_grid.read_manifest()


def _configured_final_replicates(source_grid_manifest: dict[str, Any] | None) -> int | None:
    """Return final_replicates from the source grid manifest, if recorded."""

    if source_grid_manifest is None:
        return None
    replicates = source_grid_manifest.get("final_replicates")
    if replicates is None:
        return None
    replicates = int(replicates)
    if replicates < 1:
        raise ValueError(
            "grid metadata final_replicates must be >= 1; "
            "pass --replicates to override an older grid manifest"
        )
    return replicates


def _default_grid_data() -> dict[str, Any]:
    """Return the default grid config, or an empty mapping when unavailable."""

    if not DEFAULT_GRID.is_file():
        return {}
    data = OmegaConf.to_container(OmegaConf.load(DEFAULT_GRID), resolve=True)
    return data if isinstance(data, dict) else {}


def _source_or_default_grid(source_grid_manifest: dict[str, Any] | None) -> dict[str, Any]:
    """Return source grid metadata, falling back to the checked-in grid config."""

    if source_grid_manifest is not None:
        return source_grid_manifest
    return _default_grid_data()


def _config_from_grid(
    source_grid_manifest: dict[str, Any] | None,
    key: str,
    requested: str | None,
) -> str:
    """Return a configured train/eval config path."""

    if requested:
        return requested
    grid_data = _source_or_default_grid(source_grid_manifest)
    value = grid_data.get(key)
    if not value:
        raise ValueError(f"grid metadata does not record {key}; pass the config explicitly")
    return str(value)


def _axis_metadata_from_grid(grid_data: dict[str, Any]) -> dict[str, Any]:
    """Return normalized final-grid axis metadata."""

    axes = grid_axes_from_manifest(grid_data)
    config_axes = tuple(axes["config_axes"])
    seed_axis = str(axes["scan_seed_axis"])
    return {
        "major_axes": tuple(axes["major_axes"]),
        "minor_axes": tuple(axes["minor_axes"]),
        "config_axes": config_axes,
        "scan_seed_axis": seed_axis,
        "axis_id_labels": axis_id_labels_from_manifest(grid_data, (*config_axes, seed_axis)),
    }


def _final_run_id(champion: dict[str, str], *, replicate_index: int) -> str:
    config_id = champion.get("config_id") or "unknown-config"
    winner_kind = champion.get("winner_kind") or "winner"
    return f"{config_id}_winner-{winner_kind}_rep-{int(replicate_index)}"


def build_final_jobs(
    champions: Sequence[dict[str, str]],
    *,
    source_selection_attempt_id: str,
    source_selection_attempt_dir: str | Path,
    major_axes: Sequence[str],
    minor_axes: Sequence[str],
    axis_id_labels: dict[str, str],
    replicates: int,
    seed_policy: dict[str, dict[str, str]] | None = None,
    seed_sequences: dict[str, dict[str, int]] | None = None,
    champion_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Expand selected champion rows into final replicate job rows."""

    selected = [
        (index, champion)
        for index, champion in enumerate(champions)
        if str(champion.get("config_id", "")).strip()
    ]
    if champion_limit is not None:
        selected = selected[: int(champion_limit)]
    jobs: list[dict[str, Any]] = []
    config_axes = (*major_axes, *minor_axes)
    for champion_index, champion in selected:
        source_champion_id = f"champion-{champion_index:04d}"
        point = {axis: champion.get(axis, "") for axis in config_axes}
        major_choices = {axis: point[axis] for axis in major_axes}
        minor_choices = {axis: point[axis] for axis in minor_axes}
        for replicate_index in range(int(replicates)):
            seeds = final_seed_values(seed_sequences, replicate_index)
            stage_seed_overrides = {
                "final_train": seed_override_values(seed_policy, "final_train", seeds),
                "final_eval": seed_override_values(seed_policy, "final_eval", seeds),
            }
            jobs.append(
                {
                    "source_selection_attempt_id": source_selection_attempt_id,
                    "source_selection_attempt_dir": str(source_selection_attempt_dir),
                    "source_champion_id": source_champion_id,
                    "source_champion_row_index": champion_index,
                    "source_scan_run_id": champion.get("config_id", ""),
                    "source_scan_run_ids": champion.get("run_ids", ""),
                    "source_scan_seeds": champion.get("seeds", ""),
                    "final_run_id": _final_run_id(champion, replicate_index=replicate_index),
                    "replicate_index": replicate_index,
                    "winner_kind": champion.get("winner_kind", ""),
                    "major_id": champion.get("major_id") or id_for_axes(point, major_axes, axis_id_labels),
                    "minor_id": champion.get("minor_id") or id_for_axes(point, minor_axes, axis_id_labels),
                    "major_choices": major_choices,
                    "minor_choices": minor_choices,
                    "choices": {**major_choices, **minor_choices},
                    **point,
                    "metric": champion.get("metric", ""),
                    "metric_value": champion.get("metric_value", ""),
                    **seeds,
                    "stage_seed_overrides": stage_seed_overrides,
                    "source_champion": dict(champion),
                }
            )
    return jobs


def _selection_group_by(selection_dir: Path, *, major_axes: Sequence[str]) -> tuple[str, ...]:
    """Return the group-by axes recorded by the source selection attempt."""

    report_path = selection_dir / "selection_report.json"
    if report_path.is_file():
        report = read_json(report_path)
        if isinstance(report, dict) and report.get("group_by"):
            return tuple(str(axis) for axis in report["group_by"])
    return tuple(major_axes)


def _final_job_task_lineage(
    jobs: Sequence[dict[str, Any]],
    *,
    champions: Sequence[dict[str, str]],
    group_keys: Sequence[str],
    upstream_lineage: Mapping[str, TaskLineageRow],
) -> list[TaskLineageRow]:
    """Return each final job's task-id lineage, inherited from its source champion."""

    lineage = []
    for job in jobs:
        champion = champions[int(job["source_champion_row_index"])]
        champion_row_id = champion_lineage_row_id(
            str(champion.get("winner_kind", "")),
            group_keys,
            tuple(str(champion.get(axis, "")) for axis in group_keys),
        )
        source = upstream_lineage.get(champion_row_id)
        task_ids = dict(source.task_ids) if source is not None else {}
        lineage.append(TaskLineageRow(row_id=str(job["final_run_id"]), task_ids=task_ids))
    return lineage


def _csv_value(value: Any) -> Any:
    if isinstance(value, dict):
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def write_final_grid_attempt(
    *,
    results_root: str | Path,
    attempt_id: str,
    created_at: str,
    source_selection_attempt_id: str,
    source_selection_attempt_dir: str | Path,
    study: str,
    train_config: str | Path,
    eval_config: str | Path,
    config_snapshots: dict[str, str],
    replicates: int,
    major_axes: Sequence[str],
    minor_axes: Sequence[str],
    axis_id_labels: dict[str, str],
    axis_overrides: dict[str, AxisOverrideSpec],
    seed_policy: dict[str, dict[str, str]],
    seed_sequences: dict[str, dict[str, int]],
    static_overrides: dict[str, Any] | None,
    champions: Sequence[dict[str, str]],
    jobs: Sequence[dict[str, Any]],
    task_lineage: Sequence[TaskLineageRow] = (),
) -> Path:
    """Write ``05_final_grid`` artifacts and return the attempt directory."""

    results_root = Path(results_root)
    attempt = final_grid_attempt_dir(results_root, attempt_id)
    (attempt / "jobs").mkdir(parents=True, exist_ok=True)

    source_selection_dir = Path(source_selection_attempt_dir)
    write_json(
        attempt / "source_selection_attempt.json",
        {
            "selection_attempt_id": source_selection_attempt_id,
            "selection_attempt_dir": str(source_selection_dir),
            "champions_path": str(source_selection_dir / "champions.csv"),
        },
    )
    champions_text = (source_selection_dir / "champions.csv").read_text()
    (attempt / "source_champions.csv").write_text(champions_text)
    write_task_lineage(attempt, list(task_lineage))

    config_axes = (*major_axes, *minor_axes)
    columns = [
        "source_selection_attempt_id",
        "source_champion_id",
        "source_champion_row_index",
        "source_scan_run_id",
        "source_scan_run_ids",
        "source_scan_seeds",
        "final_run_id",
        "replicate_index",
        "winner_kind",
        "major_id",
        "minor_id",
        *config_axes,
        "metric",
        "metric_value",
        "final_train_sampler_seed",
        "final_train_model_seed",
        "final_eval_sampler_seed",
    ]
    _write_csv(attempt / "final_jobs.csv", jobs, columns)
    for job in jobs:
        write_json(attempt / "jobs" / f"{job['final_run_id']}.json", job)

    manifest = {
        "study": study,
        "stage": STAGE_FINAL_GRID,
        "attempt_id": attempt_id,
        "created_at": created_at,
        "results_root": str(results_root),
        "source_selection_attempt_id": source_selection_attempt_id,
        "source_selection_attempt_dir": str(source_selection_dir),
        "train_config": str(train_config),
        "eval_config": str(eval_config),
        "config_snapshots": config_snapshots,
        "replicates": int(replicates),
        "final_replicates": int(replicates),
        "n_source_champions": len(champions),
        "n_jobs": len(jobs),
        "major_axes": list(major_axes),
        "minor_axes": list(minor_axes),
        "axis_id_labels": axis_id_labels,
        "axis_overrides": axis_overrides,
        "champion_kinds": sorted({str(champion.get("winner_kind", "")) for champion in champions if champion.get("winner_kind")}),
        "seed_overrides": seed_policy,
        "final_seed_sequences": seed_sequences,
        "static_overrides": static_overrides or {},
    }
    write_json(attempt / "manifest.json", manifest)
    OmegaConf.save(OmegaConf.create(manifest), attempt / "manifest.yaml")
    write_latest(stage_dir(results_root, STAGE_FINAL_GRID), attempt_id)
    return attempt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse final-grid planning arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--selection-attempt-id", default=None)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--eval-config", default=None)
    parser.add_argument("--replicates", type=positive_int, default=None)
    parser.add_argument("--limit-champions", type=positive_int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Deprecated; use configs/smoke.yaml with the normal stack.")
    args = parser.parse_args(argv)
    if args.smoke:
        parser.error("use --grid experiments/hooke/tpen-pair-scan-v1/configs/smoke.yaml with the normal stage stack")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Create a ``05_final_grid`` attempt from selected champions."""

    args = parse_args(argv)
    results_root = Path(args.results_root)
    selection_attempt_id = _resolve_selection_attempt(results_root, args.selection_attempt_id)
    selection_dir = stage_dir(results_root, STAGE_SELECT) / selection_attempt_id
    champions = read_champions(selection_dir)
    source_grid_manifest = _source_grid_manifest(results_root, selection_dir)
    source_or_default_grid = _source_or_default_grid(source_grid_manifest)
    study = study_name_from_manifest(source_or_default_grid)
    prefix = log_prefix(study)
    train_config = _config_from_grid(source_grid_manifest, "config", args.train_config)
    eval_config = _config_from_grid(source_grid_manifest, "validation_config", args.eval_config)
    config_snapshots = source_or_default_grid.get("config_snapshots", {})
    if not isinstance(config_snapshots, dict):
        raise ValueError("grid metadata config_snapshots must be a mapping")

    if args.replicates is not None:
        requested_replicates = args.replicates
    else:
        configured_replicates = _configured_final_replicates(source_or_default_grid)
        requested_replicates = (
            configured_replicates
            if configured_replicates is not None
            else DEFAULT_REPLICATES
        )
    replicates = requested_replicates
    champion_limit = args.limit_champions
    attempt_id = args.attempt_id or new_attempt_id()
    created_at = datetime.now(STUDY_TIMEZONE).isoformat(timespec="seconds")
    seed_policy = seed_override_policy(source_or_default_grid.get("seed_overrides"))
    seed_sequences = final_seed_sequences(source_or_default_grid.get("final_seed_sequences"))
    configured_static_overrides = source_or_default_grid.get("static_overrides") or {}
    if not isinstance(configured_static_overrides, dict):
        raise ValueError("grid metadata static_overrides must be a mapping")
    axis_metadata = _axis_metadata_from_grid(source_or_default_grid)
    major_axis_names = tuple(axis_metadata["major_axes"])
    minor_axis_names = tuple(axis_metadata["minor_axes"])
    config_axis_names = tuple(axis_metadata["config_axes"])
    axis_id_labels = dict(axis_metadata["axis_id_labels"])
    axis_overrides = normalize_axis_override_specs(
        source_or_default_grid.get("axis_overrides", {}),
        config_axis_names,
        context="grid metadata",
    )

    jobs = build_final_jobs(
        champions,
        source_selection_attempt_id=selection_attempt_id,
        source_selection_attempt_dir=selection_dir,
        major_axes=major_axis_names,
        minor_axes=minor_axis_names,
        axis_id_labels=axis_id_labels,
        replicates=replicates,
        seed_policy=seed_policy,
        seed_sequences=seed_sequences,
        champion_limit=champion_limit,
    )
    task_lineage = _final_job_task_lineage(
        jobs,
        champions=champions,
        group_keys=_selection_group_by(selection_dir, major_axes=major_axis_names),
        upstream_lineage=read_task_lineage(selection_dir),
    )
    attempt = write_final_grid_attempt(
        results_root=results_root,
        attempt_id=attempt_id,
        created_at=created_at,
        source_selection_attempt_id=selection_attempt_id,
        source_selection_attempt_dir=selection_dir,
        study=study,
        train_config=train_config,
        eval_config=eval_config,
        config_snapshots={str(key): str(value) for key, value in config_snapshots.items()},
        replicates=replicates,
        major_axes=major_axis_names,
        minor_axes=minor_axis_names,
        axis_id_labels=axis_id_labels,
        axis_overrides=axis_overrides,
        seed_policy=seed_policy,
        seed_sequences=seed_sequences,
        static_overrides={str(stage): dict(values or {}) for stage, values in configured_static_overrides.items()},
        champions=champions,
        jobs=jobs,
        task_lineage=task_lineage,
    )
    print(f"{prefix} wrote 05_final_grid attempt {attempt_id} with {len(jobs)} jobs -> {attempt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
