"""Plan a staged study grid.

Expands configured major/minor/scan-seed axes into scalar override lists for
the canonical ``run.py`` entrypoint and writes a durable ``00_grid`` attempt
(manifest + commands) describing the planned train jobs.

Stage layout (under ``results_root``)::

    00_grid/{attempt_id}/{manifest.json, commands.sh, grid.yaml,
                          train_config.yaml, jobs/{run_id}.json}
    01_train/{run_id}/{attempt_id}/...
    02_validation/{run_id}/{attempt_id}/...
    03_collect/{attempt_id}/...
    04_select/{attempt_id}/...
"""

from __future__ import annotations

import argparse
import itertools
import random
import shlex
from datetime import datetime
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

_tpen_utils_config = sibling(__file__, 'utils.config')
config_snapshot_names = _tpen_utils_config.config_snapshot_names
_tpen_utils_io = sibling(__file__, 'utils.io')
write_json = _tpen_utils_io.write_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_GRID = _tpen_utils_layout.STAGE_GRID
STAGE_TRAIN = _tpen_utils_layout.STAGE_TRAIN
grid_attempt_dir = _tpen_utils_layout.grid_attempt_dir
stage_dir = _tpen_utils_layout.stage_dir
train_attempt_dir = _tpen_utils_layout.train_attempt_dir
train_run_dir = _tpen_utils_layout.train_run_dir
validation_run_dir = _tpen_utils_layout.validation_run_dir
write_latest = _tpen_utils_layout.write_latest
_tpen_utils_naming = sibling(__file__, 'utils.naming')
axis_value_label = _tpen_utils_naming.axis_value_label
experiment_run_name = _tpen_utils_naming.experiment_run_name
log_prefix = _tpen_utils_naming.log_prefix
study_name = _tpen_utils_naming.study_name
_tpen_utils_overrides = sibling(__file__, 'utils.overrides')
AxisOverrideSpec = _tpen_utils_overrides.AxisOverrideSpec
axis_value_overrides = _tpen_utils_overrides.axis_value_overrides
format_override_value = _tpen_utils_overrides.format_override_value
normalize_axis_override_specs = _tpen_utils_overrides.normalize_axis_override_specs
_tpen_utils_seeds = sibling(__file__, 'utils.seeds')
final_seed_sequences = _tpen_utils_seeds.final_seed_sequences
scan_seed_values = _tpen_utils_seeds.scan_seed_values
seed_override_policy = _tpen_utils_seeds.seed_override_policy
seed_override_values = _tpen_utils_seeds.seed_override_values
_tpen_utils_time = sibling(__file__, 'utils.time')
DEFAULT_STUDY_TIMEZONE = _tpen_utils_time.DEFAULT_STUDY_TIMEZONE
new_attempt_id = _tpen_utils_time.new_attempt_id
resolve_timezone = _tpen_utils_time.resolve_timezone

STUDY_DIR = Path(__file__).resolve().parent
DEFAULT_GRID = STUDY_DIR / "configs" / "grid.yaml"


# ---------------------------------------------------------------------------
# Grid expansion and validation
# ---------------------------------------------------------------------------
def _axis_names(block: dict[str, Sequence[Any]], configured: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return deterministic axis names for one grid block."""

    if configured is not None:
        axes = tuple(str(axis) for axis in configured)
    else:
        axes = tuple(str(axis) for axis in block.keys())
    if not axes:
        raise ValueError("grid block must contain at least one axis")
    missing = [axis for axis in axes if axis not in block]
    if missing:
        raise ValueError(f"grid block is missing required axes: {', '.join(missing)}")
    return axes


def major_axes(grid_data: dict[str, Any]) -> tuple[str, ...]:
    """Return major-axis names from config, preserving YAML order by default."""

    return _axis_names(grid_data["major_grid"], grid_data.get("major_axes"))


def minor_axes(grid_data: dict[str, Any]) -> tuple[str, ...]:
    """Return minor-axis names from config, preserving YAML order by default."""

    return _axis_names(grid_data["minor_grid"], grid_data.get("minor_axes"))


def scan_seed_axis(grid_data: dict[str, Any]) -> str:
    """Return the scan seed axis name."""

    return str(grid_data.get("scan_seed_axis", "seed_index" if "scan_seed_rows" in grid_data else "seed"))


def scan_seed_rows(grid_data: dict[str, Any], seed_axis: str) -> list[dict[str, Any]]:
    """Return configured paired scan seed rows."""

    if "scan_seed_rows" in grid_data:
        configured = grid_data["scan_seed_rows"]
        if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
            raise ValueError("scan_seed_rows must be a sequence of mappings")
        rows = []
        for index, row in enumerate(configured):
            if not isinstance(row, dict):
                raise ValueError(f"scan_seed_rows[{index}] must be a mapping")
            if seed_axis not in row:
                raise ValueError(f"scan_seed_rows[{index}] is missing seed axis {seed_axis!r}")
            rows.append(dict(row))
        if not rows:
            raise ValueError("scan_seed_rows must contain at least one row")
        return rows
    return [{seed_axis: seed} for seed in grid_data.get("scan_seeds", [])]


def grid_axes(grid_data: dict[str, Any]) -> tuple[str, ...]:
    """Return the full scalar train axis order."""

    if "major_grid" in grid_data or "minor_grid" in grid_data or "scan_seeds" in grid_data or "scan_seed_rows" in grid_data:
        return (*major_axes(grid_data), *minor_axes(grid_data), scan_seed_axis(grid_data))
    return _axis_names(grid_data["grid"], grid_data.get("grid_axes"))


def _axis_points(grid: dict[str, Sequence[Any]], axes: Sequence[str]) -> list[dict[str, Any]]:
    """Expand selected axes from a grid block into dictionaries."""

    return [
        dict(zip(axes, combination, strict=True))
        for combination in itertools.product(*(list(grid[axis]) for axis in axes))
    ]


def expand_grid(grid: dict[str, Sequence[Any]], axes: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Expand a flat grid block into scalar points."""

    axes = _axis_names(grid, axes)
    return [
        dict(zip(axes, combination, strict=True))
        for combination in itertools.product(*(list(grid[axis]) for axis in axes))
    ]


def expand_split_grid(
    major_grid: dict[str, Sequence[Any]],
    minor_grid: dict[str, Sequence[Any]],
    scan_seeds: Sequence[Any],
    *,
    major_axis_names: Sequence[str],
    minor_axis_names: Sequence[str],
    seed_axis: str,
) -> list[dict[str, Any]]:
    """Expand major/minor/scan-seed blocks into scalar train points."""

    points = []
    for major in _axis_points(major_grid, major_axis_names):
        for minor in _axis_points(minor_grid, minor_axis_names):
            for seed in scan_seeds:
                seed_row = dict(seed) if isinstance(seed, dict) else {seed_axis: seed}
                points.append({**major, **minor, **seed_row})
    return points


def expand_grid_spec(grid_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand either a split grid spec or a legacy flat grid spec."""

    if "major_grid" in grid_data or "minor_grid" in grid_data or "scan_seeds" in grid_data or "scan_seed_rows" in grid_data:
        seed_axis = scan_seed_axis(grid_data)
        return expand_split_grid(
            grid_data["major_grid"],
            grid_data["minor_grid"],
            scan_seed_rows(grid_data, seed_axis),
            major_axis_names=major_axes(grid_data),
            minor_axis_names=minor_axes(grid_data),
            seed_axis=seed_axis,
        )
    return expand_grid(grid_data["grid"], grid_data.get("grid_axes"))


def champion_kinds(grid_data: dict[str, Any]) -> list[str]:
    """Return configured champion kinds."""

    kinds = [spec["name"] for spec in champion_specs(grid_data)]
    if not kinds:
        raise ValueError("champions must contain at least one winner kind")
    return kinds


def champion_specs(grid_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return configured champion selector specs.

    Each entry must be a mapping so metric semantics are visible in the grid
    snapshot and not encoded by study-local Python.
    """

    configured = grid_data.get("champions")
    if configured is None:
        raise ValueError("grid.yaml must define explicit champion selector specs")
    specs: list[dict[str, Any]] = []
    for entry in configured:
        if not isinstance(entry, dict):
            raise ValueError(f"champion entries must be mappings, got {entry!r}")
        spec = dict(entry)
        name = str(spec.get("name", "")).strip()
        selector = str(spec.get("selector", "")).strip()
        if not name:
            raise ValueError("champion specs require a non-empty name")
        if not selector:
            raise ValueError(f"champion {name!r} requires selector")
        spec["name"] = name
        spec["selector"] = selector
        specs.append(spec)
    if not specs:
        raise ValueError("champions must contain at least one selector spec")
    return specs


def champion_reference_metrics(grid_data: dict[str, Any]) -> list[dict[str, str]]:
    """Return configured extra metrics to copy into champions.csv."""

    configured = grid_data.get("champion_reference_metrics")
    if configured is None:
        return []
    metrics = []
    for entry in configured:
        if not isinstance(entry, dict):
            raise ValueError("champion_reference_metrics entries must be mappings")
        label = str(entry.get("label", "")).strip()
        metric = str(entry.get("metric", "")).strip()
        if not label or not metric:
            raise ValueError("champion_reference_metrics entries require label and metric")
        metrics.append({"label": label, "metric": metric})
    return metrics


def _config_select(config: Any, path: str, *, value: Any | None = None) -> Any:
    """Select a value from an OmegaConf config, supporting ``{value}`` templates."""

    return OmegaConf.select(config, path.format(value=value))


def choice_validation_specs(grid_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return axis validation specs from grid config."""

    configured = grid_data.get("choice_validation") or {}
    if not isinstance(configured, dict):
        raise ValueError("choice_validation must be a mapping")
    return {str(axis): dict(spec or {}) for axis, spec in configured.items()}


def choice_names(config: Any, choices_path: str) -> set[str]:
    """Return configured choice keys under ``choices_path``."""

    choices = _config_select(config, choices_path)
    if choices is None:
        raise ValueError(f"choice validation path {choices_path!r} does not exist")
    if not hasattr(choices, "keys"):
        raise ValueError(f"choice validation path {choices_path!r} is not a mapping")
    return {str(name) for name in choices.keys()}


def axis_tags(config: Any, validation_specs: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Return ``{axis: {choice: tags}}`` for axes that declare ``tags_path``."""

    tags: dict[str, dict[str, list[str]]] = {}
    for axis, spec in validation_specs.items():
        choices_path = str(spec.get("choices_path", "")).strip()
        tags_path = str(spec.get("tags_path", "")).strip()
        if not choices_path or not tags_path:
            continue
        tags[axis] = {}
        for value in choice_names(config, choices_path):
            selected = _config_select(config, tags_path, value=value) or []
            tags[axis][value] = [str(tag) for tag in selected]
    return tags


def tags_for_point(point: dict[str, Any], tags_by_axis: dict[str, dict[str, list[str]]]) -> list[str]:
    """Return tags attached to a grid point by configured tag sources."""

    tags: list[str] = []
    for axis, choices in tags_by_axis.items():
        tags.extend(choices.get(str(point.get(axis, "")), []))
    return sorted(set(tags))


def validate_grid(
    points: Sequence[dict[str, Any]],
    config: Any,
    validation_specs: dict[str, dict[str, Any]],
) -> None:
    """Fail loudly if a grid point violates configured choice validation."""

    known_by_axis = {
        axis: choice_names(config, str(spec["choices_path"]))
        for axis, spec in validation_specs.items()
        if str(spec.get("choices_path", "")).strip()
    }
    for point in points:
        for axis, known in known_by_axis.items():
            value = str(point.get(axis, ""))
            spec = validation_specs[axis]
            for suffix in spec.get("exclude_suffixes", []) or []:
                if value.endswith(str(suffix)):
                    raise ValueError(f"grid {axis} value {value!r} is excluded by suffix {suffix!r}")
            if value not in known:
                raise ValueError(f"grid {axis} value {value!r} is not under {spec['choices_path']!r}")


# ---------------------------------------------------------------------------
# Axis-wise blinding
# ---------------------------------------------------------------------------
def blinding_config(grid_data: dict[str, Any]) -> dict[str, Any]:
    """Return normalized blinding configuration."""

    configured = grid_data.get("blinding") or {}
    if not isinstance(configured, dict):
        raise ValueError("blinding must be a mapping")
    slot_prefixes = configured.get("slot_prefixes") or {}
    if not isinstance(slot_prefixes, dict):
        raise ValueError("blinding.slot_prefixes must be a mapping")
    return {
        "enabled_by_default": bool(configured.get("enabled_by_default", False)),
        "slot_prefixes": {str(axis): str(prefix) for axis, prefix in slot_prefixes.items()},
    }


def blinding_enabled(grid_data: dict[str, Any], requested: bool | None) -> bool:
    """Return whether this planning attempt should blind major axes."""

    if requested is not None:
        return bool(requested)
    return bool(blinding_config(grid_data)["enabled_by_default"])


def validate_blindable_axes(
    major_axis_names: Sequence[str],
    validation_specs: dict[str, dict[str, Any]],
) -> None:
    """Reject blinding for axes that have no choice-library indirection."""

    missing = [axis for axis in major_axis_names if axis not in validation_specs]
    if missing:
        raise ValueError(
            "cannot blind major axes without choice_validation entries: "
            + ", ".join(missing)
        )


def build_blinding_maps(
    grid_data: dict[str, Any],
    *,
    blind_seed: int,
) -> dict[str, dict[str, dict[str, str]]]:
    """Return axis-wise slot maps for configured major axes."""

    config = blinding_config(grid_data)
    prefixes = config["slot_prefixes"]
    maps: dict[str, dict[str, dict[str, str]]] = {}
    for axis in major_axes(grid_data):
        values = [str(value) for value in grid_data["major_grid"][axis]]
        slots = [f"{prefixes.get(axis, axis[:1].upper())}{index:02d}" for index in range(len(values))]
        shuffled_values = list(values)
        random.Random(f"{int(blind_seed)}:{axis}").shuffle(shuffled_values)
        slot_to_value = dict(zip(slots, shuffled_values, strict=True))
        value_to_slot = {value: slot for slot, value in slot_to_value.items()}
        maps[axis] = {
            "slot_to_value": slot_to_value,
            "value_to_slot": value_to_slot,
        }
    return maps


def apply_blinding_to_points(
    points: Sequence[dict[str, Any]],
    maps: dict[str, dict[str, dict[str, str]]],
) -> list[dict[str, Any]]:
    """Replace major-axis semantic values with blinded slot values."""

    blinded = []
    for point in points:
        row = dict(point)
        for axis, axis_maps in maps.items():
            value = str(row[axis])
            row[axis] = axis_maps["value_to_slot"][value]
        blinded.append(row)
    return blinded


def _slot_grid_data(
    grid_data: dict[str, Any],
    maps: dict[str, dict[str, dict[str, str]]],
    *,
    blind_seed: int,
    config: str | Path,
    validation_config: str | Path | None,
) -> dict[str, Any]:
    """Return manifest/grid metadata with major-axis values replaced by slots."""

    data = dict(grid_data)
    data["config"] = str(config)
    if validation_config is not None:
        data["validation_config"] = str(validation_config)
    major_grid = {axis: list(values) for axis, values in dict(grid_data["major_grid"]).items()}
    for axis, axis_maps in maps.items():
        major_grid[axis] = list(axis_maps["slot_to_value"].keys())
    data["major_grid"] = major_grid
    data["blinding"] = {
        "enabled": True,
        "blind_seed": int(blind_seed),
        "major_axes": list(maps.keys()),
        "slot_prefixes": blinding_config(grid_data)["slot_prefixes"],
    }
    return data


def _materialize_slot_config(
    config: Any,
    *,
    validation_specs: dict[str, dict[str, Any]],
    maps: dict[str, dict[str, dict[str, str]]],
    axis_override_paths: dict[str, AxisOverrideSpec],
) -> Any:
    """Return a config copy whose choice libraries are keyed by blind slots."""

    materialized = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    for axis, axis_maps in maps.items():
        spec = validation_specs.get(axis)
        if not spec:
            continue
        choices_path = str(spec.get("choices_path", "")).strip()
        if not choices_path:
            continue
        choices = {}
        for slot, value in axis_maps["slot_to_value"].items():
            selected = OmegaConf.select(config, f"{choices_path}.{value}")
            if selected is None:
                raise ValueError(f"cannot blind {axis} value {value!r}; missing {choices_path}.{value}")
            choices[slot] = selected
        OmegaConf.update(materialized, choices_path, choices, merge=False)
        first_slot = next(iter(axis_maps["slot_to_value"]))
        override_path = axis_override_paths.get(axis)
        if isinstance(override_path, str):
            OmegaConf.update(materialized, override_path, first_slot, merge=False)
    return materialized


def unblind_artifact(
    *,
    blind_seed: int,
    maps: dict[str, dict[str, dict[str, str]]],
    original_grid: str | Path,
) -> dict[str, Any]:
    """Return the dedicated semantic mapping artifact for a blinded attempt."""

    return {
        "blind_seed": int(blind_seed),
        "original_grid": str(original_grid),
        "axes": {
            axis: {
                "slot_to_value": dict(axis_maps["slot_to_value"]),
                "value_to_slot": dict(axis_maps["value_to_slot"]),
            }
            for axis, axis_maps in maps.items()
        },
    }


# ---------------------------------------------------------------------------
# Overrides and commands
# ---------------------------------------------------------------------------
def axis_id_labels(grid_data: dict[str, Any], axes: Sequence[str]) -> dict[str, str]:
    """Return axis -> id-label mapping for durable run ids."""

    configured = grid_data.get("axis_id_labels") or {}
    if not isinstance(configured, dict):
        raise ValueError("axis_id_labels must be a mapping")
    return {axis: str(configured.get(axis, axis)) for axis in axes}


def axis_override_paths(grid_data: dict[str, Any], axes: Sequence[str]) -> dict[str, AxisOverrideSpec]:
    """Return axis -> OmegaConf override path mapping."""

    configured = grid_data.get("axis_overrides") or {}
    return normalize_axis_override_specs(configured, axes, context="grid")


def static_overrides(grid_data: dict[str, Any] | None, stage: str) -> dict[str, Any]:
    """Return static override values configured for one stage."""

    configured = (grid_data or {}).get("static_overrides") or {}
    if not isinstance(configured, dict):
        raise ValueError("static_overrides must be a mapping")
    stage_overrides = configured.get(stage) or {}
    if not isinstance(stage_overrides, dict):
        raise ValueError(f"static_overrides.{stage} must be a mapping")
    return {str(path): value for path, value in stage_overrides.items()}


def _id_value(value: Any) -> str:
    """Return a compact value label for durable ids."""

    return axis_value_label(value)


def id_for(point: dict[str, Any], axes: Sequence[str], labels: dict[str, str]) -> str:
    """Return a deterministic id from selected axes."""

    return "_".join(f"{labels.get(axis, axis)}-{_id_value(point[axis])}" for axis in axes)


def train_overrides(
    point: dict[str, Any],
    *,
    study: str,
    run_id: str,
    attempt_id: str,
    results_root: str | Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, AxisOverrideSpec],
    seed_axis: str,
    seed_policy: dict[str, dict[str, str]] | None = None,
    static_stage_overrides: dict[str, Any] | None = None,
    timezone: str | None = None,
) -> list[str]:
    """Return scalar OmegaConf-style overrides for one train job."""

    seed_overrides = seed_override_values(
        seed_policy,
        "scan_train",
        scan_seed_values(point, seed_axis),
    )
    overrides = [
        *axis_value_overrides(point, axes=scalar_axes, override_specs=override_paths, stage="train"),
        *(f"{path}={value}" for path, value in seed_overrides.items()),
        *(f"{path}={format_override_value(value)}" for path, value in (static_stage_overrides or {}).items()),
        f"run.root={stage_dir(results_root, STAGE_TRAIN)}",
        "run.layout=flat",
        f"run.run_id={run_id}/{attempt_id}",
        f"study.name={study}",
        f"study.attempt_id={attempt_id}",
        f"experiment.name={study}",
        f"experiment.run_name={experiment_run_name(study, 'train')}",
    ]
    if timezone is not None:
        overrides.append(f"run.timezone={timezone}")
    return overrides


def command_for(config: str | Path, overrides: Sequence[str], *, python: str = "python") -> list[str]:
    """Return the canonical ``run.py`` command for a config and overrides."""

    return [python, "-u", "run.py", "--config", str(config), *overrides]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def build_jobs(
    points: Sequence[dict[str, Any]],
    *,
    study: str,
    attempt_id: str,
    results_root: str | Path,
    config: str | Path,
    major_axis_names: Sequence[str],
    minor_axis_names: Sequence[str],
    seed_axis: str,
    id_labels: dict[str, str],
    override_paths: dict[str, AxisOverrideSpec],
    tags_by_axis: dict[str, dict[str, list[str]]],
    seed_policy: dict[str, dict[str, str]] | None = None,
    static_stage_overrides: dict[str, Any] | None = None,
    python: str = "python",
    timezone: str | None = None,
) -> list[dict[str, Any]]:
    """Return one manifest job record per grid point."""

    jobs = []
    config_axes = (*major_axis_names, *minor_axis_names)
    run_axes = (*config_axes, seed_axis)
    for point in points:
        seed_values = scan_seed_values(point, seed_axis)
        run_id = id_for(point, run_axes, id_labels)
        major_id = id_for(point, major_axis_names, id_labels)
        minor_id = id_for(point, minor_axis_names, id_labels)
        config_id = id_for(point, config_axes, id_labels)
        overrides = train_overrides(
            point,
            study=study,
            run_id=run_id,
            attempt_id=attempt_id,
            results_root=results_root,
            scalar_axes=config_axes,
            override_paths=override_paths,
            seed_axis=seed_axis,
            seed_policy=seed_policy,
            static_stage_overrides=static_stage_overrides,
            timezone=timezone,
        )
        scan_seed_overrides = seed_override_values(
            seed_policy,
            "scan_train",
            seed_values,
        )
        jobs.append(
            {
                "run_id": run_id,
                "major_id": major_id,
                "minor_id": minor_id,
                "config_id": config_id,
                "major_choices": {axis: point[axis] for axis in major_axis_names},
                "minor_choices": {axis: point[axis] for axis in minor_axis_names},
                "scan_seed": point[seed_axis],
                "seed_values": seed_values,
                "seed_overrides": {"scan_train": scan_seed_overrides},
                "static_overrides": {"train": dict(static_stage_overrides or {})},
                "train_dir": str(train_run_dir(results_root, run_id)),
                "validation_dir": str(validation_run_dir(results_root, run_id)),
                "train_attempt_dir": str(train_attempt_dir(results_root, run_id, attempt_id)),
                "overrides": overrides,
                "command": shlex.join(command_for(config, overrides, python=python)),
                "choices": {axis: point[axis] for axis in run_axes},
                "tags": tags_for_point(point, tags_by_axis),
                "submitted": False,
                "launcher": None,
                "launcher_job_id": None,
            }
        )
    return jobs


def build_manifest(
    *,
    attempt_id: str,
    created_at: str,
    config: str | Path,
    grid: str | Path,
    results_root: str | Path,
    jobs: list[dict[str, Any]],
    grid_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``00_grid`` manifest describing planned jobs."""

    grid_data = grid_data or {}
    manifest = {
        "study": study_name(grid_data.get("study")),
        "stage": STAGE_GRID,
        "attempt_id": attempt_id,
        "created_at": created_at,
        "config": str(config),
        "grid": str(grid),
        "results_root": str(results_root),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    if "major_grid" in grid_data:
        major_axis_names = major_axes(grid_data)
        minor_axis_names = minor_axes(grid_data)
        seed_axis = scan_seed_axis(grid_data)
        all_axes = (*major_axis_names, *minor_axis_names, seed_axis)
        manifest.update(
            {
                "grid_schema": "major_minor_scan",
                "major_axes": list(major_axis_names),
                "minor_axes": list(minor_axis_names),
                "scan_seed_axis": seed_axis,
                "major_grid": grid_data.get("major_grid", {}),
                "minor_grid": grid_data.get("minor_grid", {}),
                "scan_seeds": list(grid_data.get("scan_seeds", [])),
                "scan_seed_rows": scan_seed_rows(grid_data, seed_axis),
                "axis_id_labels": axis_id_labels(grid_data, all_axes),
                "axis_overrides": axis_override_paths(grid_data, (*major_axis_names, *minor_axis_names)),
                "config_snapshots": config_snapshot_names(grid_data.get("config_snapshots")),
                "choice_validation": choice_validation_specs(grid_data),
                "seed_overrides": seed_override_policy(grid_data.get("seed_overrides")),
                "final_seed_sequences": final_seed_sequences(grid_data.get("final_seed_sequences")),
                "static_overrides": grid_data.get("static_overrides", {}),
                "champions": champion_specs(grid_data),
                "champion_kinds": champion_kinds(grid_data),
                "champion_reference_metrics": champion_reference_metrics(grid_data),
                "final_replicates": int(grid_data.get("final_replicates", 0) or 0),
            }
        )
    else:
        axes = grid_axes(grid_data)
        manifest.update(
            {
                "grid_schema": "flat",
                "grid_axes": list(axes),
                "axis_id_labels": axis_id_labels(grid_data, axes),
                "axis_overrides": axis_override_paths(grid_data, axes),
                "config_snapshots": config_snapshot_names(grid_data.get("config_snapshots")),
                "choice_validation": choice_validation_specs(grid_data),
            }
        )
    validation_config = grid_data.get("validation_config")
    if validation_config is not None:
        manifest["validation_config"] = str(validation_config)
    if "blinding" in grid_data:
        manifest["blinding"] = grid_data["blinding"]
    return manifest


def write_grid_attempt(
    *,
    results_root: str | Path,
    attempt_id: str,
    created_at: str,
    config: str | Path,
    grid: str | Path,
    grid_data: Any,
    jobs: list[dict[str, Any]],
    config_snapshot_data: dict[str, Any] | None = None,
    unblind_data: dict[str, Any] | None = None,
) -> Path:
    """Write the durable ``00_grid`` attempt and return its directory."""

    attempt = grid_attempt_dir(results_root, attempt_id)
    (attempt / "jobs").mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(
        attempt_id=attempt_id,
        created_at=created_at,
        config=config,
        grid=grid,
        results_root=results_root,
        jobs=jobs,
        grid_data=grid_data if isinstance(grid_data, dict) else None,
    )
    write_json(attempt / "manifest.json", manifest)

    # Snapshot the inputs that produced this plan.
    OmegaConf.save(OmegaConf.create(grid_data), attempt / "grid.yaml")
    snapshots = config_snapshot_names(
        grid_data.get("config_snapshots") if isinstance(grid_data, dict) else None
    )
    if config_snapshot_data is not None and "train" in config_snapshot_data:
        OmegaConf.save(config_snapshot_data["train"], attempt / snapshots["train"])
    else:
        config_text = Path(config).read_text() if Path(config).exists() else ""
        (attempt / snapshots["train"]).write_text(config_text)

    # Exact commands that train.py will read from this attempt.
    study = study_name(grid_data.get("study")) if isinstance(grid_data, dict) else study_name()
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# {study} 00_grid attempt {attempt_id}", ""]
    lines += [job["command"] for job in jobs]
    (attempt / "commands.sh").write_text("\n".join(lines) + "\n")

    validation_config = grid_data.get("validation_config") if isinstance(grid_data, dict) else None
    if config_snapshot_data is not None and "validation" in config_snapshot_data:
        OmegaConf.save(config_snapshot_data["validation"], attempt / snapshots["validation"])
    elif validation_config is not None and Path(validation_config).exists():
        (attempt / snapshots["validation"]).write_text(Path(validation_config).read_text())
    if unblind_data is not None:
        write_json(attempt / "unblind.json", unblind_data)

    # Per-job specs for downstream stages.
    for job in jobs:
        write_json(attempt / "jobs" / f"{job['run_id']}.json", job)

    write_latest(stage_dir(results_root, STAGE_GRID), attempt_id)
    return attempt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse planner command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default=str(DEFAULT_GRID), help="Grid YAML path.")
    parser.add_argument("--config", default=None, help="Train config path (defaults to grid.config).")
    parser.add_argument("--results-root", default=None, help="Results root (defaults to grid.results_root).")
    parser.add_argument(
        "--attempt-id", default=None, help="Attempt id in the study timezone (defaults to now)."
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_STUDY_TIMEZONE,
        help=(
            "IANA timezone owned by the planner: stamps attempt ids and "
            "overrides run.timezone (default America/New_York)."
        ),
    )
    parser.add_argument("--tags", nargs="*", default=None, help="Only include grid points with all configured tags.")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of planned jobs.")
    blind_group = parser.add_mutually_exclusive_group()
    blind_group.add_argument(
        "--blind",
        dest="blind",
        action="store_true",
        help="Blind configured major axes into slot identifiers.",
    )
    blind_group.add_argument(
        "--no-blind",
        dest="blind",
        action="store_false",
        help="Plan with semantic major-axis labels instead of slots.",
    )
    parser.set_defaults(blind=None)
    parser.add_argument(
        "--blind-seed",
        type=int,
        default=0,
        help="Seed for reproducible axis-wise blinding when enabled.",
    )
    parser.add_argument(
        "--python",
        default="python",
        help=(
            "Python executable name recorded in planned commands. The train "
            "launcher chooses the CPU/CUDA uv environment at launch time."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Plan the configured study grid and write a ``00_grid`` attempt."""

    args = parse_args(argv)
    grid_path = Path(args.grid)
    grid_data = OmegaConf.to_container(OmegaConf.load(grid_path), resolve=True)
    config = args.config or grid_data["config"]
    results_root = args.results_root or grid_data["results_root"]
    study = study_name(grid_data.get("study"))
    prefix = log_prefix(study)

    # The planner owns the timezone for this study: it stamps the attempt id /
    # created_at and injects run.timezone into the compiled train commands.
    tz = resolve_timezone(args.timezone)
    attempt_id = args.attempt_id or new_attempt_id(tz=tz)
    created_at = datetime.now(tz).isoformat(timespec="seconds")

    major_axis_names = major_axes(grid_data)
    minor_axis_names = minor_axes(grid_data)
    seed_axis = scan_seed_axis(grid_data)
    config_axes = (*major_axis_names, *minor_axis_names)
    all_axes = (*config_axes, seed_axis)
    true_points = expand_grid_spec(grid_data)
    config_obj = OmegaConf.load(config)
    validation_specs = choice_validation_specs(grid_data)
    seed_policy = seed_override_policy(grid_data.get("seed_overrides"))
    id_labels = axis_id_labels(grid_data, all_axes)
    override_paths = axis_override_paths(grid_data, config_axes)
    validate_grid(true_points, config_obj, validation_specs)

    points = true_points
    config_for_jobs: str | Path = config
    manifest_grid_data = dict(grid_data)
    config_snapshot_data: dict[str, Any] | None = None
    unblind_data: dict[str, Any] | None = None
    tag_config = config_obj
    if blinding_enabled(grid_data, args.blind):
        validate_blindable_axes(major_axis_names, validation_specs)
        maps = build_blinding_maps(grid_data, blind_seed=args.blind_seed)
        points = apply_blinding_to_points(true_points, maps)
        snapshots = config_snapshot_names(grid_data.get("config_snapshots"))
        attempt_dir = grid_attempt_dir(results_root, attempt_id)
        config_for_jobs = attempt_dir / snapshots["train"]
        validation_config_for_jobs: str | Path | None = None
        if grid_data.get("validation_config") is not None:
            validation_config_for_jobs = attempt_dir / snapshots["validation"]
        materialized_train = _materialize_slot_config(
            config_obj,
            validation_specs=validation_specs,
            maps=maps,
            axis_override_paths=override_paths,
        )
        config_snapshot_data = {"train": materialized_train}
        if grid_data.get("validation_config") is not None:
            validation_obj = OmegaConf.load(grid_data["validation_config"])
            config_snapshot_data["validation"] = _materialize_slot_config(
                validation_obj,
                validation_specs=validation_specs,
                maps=maps,
                axis_override_paths=override_paths,
            )
        manifest_grid_data = _slot_grid_data(
            grid_data,
            maps,
            blind_seed=args.blind_seed,
            config=config_for_jobs,
            validation_config=validation_config_for_jobs,
        )
        unblind_data = unblind_artifact(
            blind_seed=args.blind_seed,
            maps=maps,
            original_grid=grid_path,
        )
        tag_config = materialized_train
        validate_grid(points, tag_config, validation_specs)
        print(
            f"{prefix} blinded major axes {list(maps)} with seed {args.blind_seed}; "
            f"semantic map will be written to unblind.json"
        )
    tags_by_axis = axis_tags(tag_config, validation_specs)

    if args.tags:
        wanted = set(args.tags)
        kept = [p for p in points if wanted.issubset(set(tags_for_point(p, tags_by_axis)))]
        if len(kept) < len(points):
            print(f"{prefix} tag filter {sorted(wanted)}: {len(kept)}/{len(points)} jobs kept")
        points = kept
    if args.limit is not None and args.limit < len(points):
        print(f"{prefix} --limit {args.limit}: dropping {len(points) - args.limit} of {len(points)} jobs")
        points = points[: args.limit]

    jobs = build_jobs(
        points,
        study=study,
        attempt_id=attempt_id,
        results_root=results_root,
        config=config_for_jobs,
        major_axis_names=major_axis_names,
        minor_axis_names=minor_axis_names,
        seed_axis=seed_axis,
        id_labels=id_labels,
        override_paths=override_paths,
        tags_by_axis=tags_by_axis,
        seed_policy=seed_policy,
        static_stage_overrides=static_overrides(grid_data, "train"),
        python=args.python,
        timezone=args.timezone,
    )
    attempt = write_grid_attempt(
        results_root=results_root,
        attempt_id=attempt_id,
        created_at=created_at,
        config=config_for_jobs,
        grid=grid_path,
        grid_data=manifest_grid_data,
        jobs=jobs,
        config_snapshot_data=config_snapshot_data,
        unblind_data=unblind_data,
    )
    print(f"{prefix} wrote 00_grid attempt {attempt_id} with {len(jobs)} jobs -> {attempt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
