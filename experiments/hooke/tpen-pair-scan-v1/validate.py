"""Launch validation jobs into ``02_validation``.

Validation consumes an existing ``00_grid`` attempt and selected completed
``01_train`` attempts. It writes per-validation provenance and launches the
validation config recorded by the grid manifest.
"""

from __future__ import annotations

import argparse
import shlex
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

launch = sibling(__file__, 'launch')
_tpen_utils_config = sibling(__file__, 'utils.config')
config_snapshot_names = _tpen_utils_config.config_snapshot_names
_tpen_utils_io = sibling(__file__, 'utils.io')
read_json = _tpen_utils_io.read_json
write_json = _tpen_utils_io.write_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_VALIDATION = _tpen_utils_layout.STAGE_VALIDATION
grid_attempt_dir = _tpen_utils_layout.grid_attempt_dir
latest_attempt_id = _tpen_utils_layout.latest_attempt_id
stage_dir = _tpen_utils_layout.stage_dir
train_attempt_dir = _tpen_utils_layout.train_attempt_dir
train_run_dir = _tpen_utils_layout.train_run_dir
validation_attempt_dir = _tpen_utils_layout.validation_attempt_dir
write_latest = _tpen_utils_layout.write_latest
_tpen_utils_naming = sibling(__file__, 'utils.naming')
experiment_run_name = _tpen_utils_naming.experiment_run_name
log_prefix = _tpen_utils_naming.log_prefix
stage_job_name = _tpen_utils_naming.stage_job_name
study_name_from_manifest = _tpen_utils_naming.study_name_from_manifest
_tpen_utils_overrides = sibling(__file__, 'utils.overrides')
AxisOverrideSpec = _tpen_utils_overrides.AxisOverrideSpec
axis_value_overrides = _tpen_utils_overrides.axis_value_overrides
normalize_axis_override_specs = _tpen_utils_overrides.normalize_axis_override_specs
_tpen_utils_seeds = sibling(__file__, 'utils.seeds')
scan_seed_values = _tpen_utils_seeds.scan_seed_values
seed_override_values = _tpen_utils_seeds.seed_override_values

STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.toolkit import (  # noqa: E402
    ExecutorOptions,
    LocalExecutor,
    StagePlan,
    SubmissionRequest,
    SubmititExecutor,
    write_execution_records,
)
from experiments.toolkit.resources import resource_from_profile  # noqa: E402
from experiments.toolkit.specs import tasks_from_commands  # noqa: E402
from experiments.toolkit.task_state import _checkpoint_ready  # noqa: E402

DEFAULT_RESULTS_ROOT = STUDY_DIR / "results"
DEFAULT_GRID = STUDY_DIR / "configs" / "grid.yaml"

def _scalar_axes(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return non-seed axes recorded in a grid manifest."""

    if manifest.get("grid_schema") == "major_minor_scan":
        return tuple(str(axis) for axis in (*manifest.get("major_axes", []), *manifest.get("minor_axes", [])))
    seed_axis = str(manifest.get("scan_seed_axis", "seed"))
    return tuple(str(axis) for axis in manifest.get("grid_axes", []) if str(axis) != seed_axis)


def _axis_override_paths(manifest: dict[str, Any], axes: Sequence[str]) -> dict[str, AxisOverrideSpec]:
    """Return axis -> config override path from a grid manifest."""

    configured = manifest.get("axis_overrides")
    return normalize_axis_override_specs(configured, axes, context="grid manifest")


def _command_for(config: str | Path, overrides: Sequence[str], *, python: str = "python") -> list[str]:
    """Return the canonical ``run.py`` command for a validation config."""

    return [python, "-u", "run.py", "--config", str(config), *overrides]


def _job_timezone(job: dict[str, Any]) -> str | None:
    """Return the train job's planned run timezone override, if present."""

    for override in job.get("overrides", []):
        text = str(override)
        if text.startswith("run.timezone="):
            return text.split("=", 1)[1]
    return None


def validation_overrides(
    point: dict[str, Any],
    *,
    study: str,
    run_id: str,
    attempt_id: str,
    results_root: str | Path,
    checkpoint_path: str | Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, AxisOverrideSpec],
    seed_axis: str,
    seed_policy: dict[str, dict[str, str]] | None = None,
    timezone: str | None = None,
) -> list[str]:
    """Return scalar OmegaConf-style overrides for one validation job."""

    seed_overrides = seed_override_values(
        seed_policy,
        "validation",
        scan_seed_values(point, seed_axis),
    )
    overrides = [
        *axis_value_overrides(point, axes=scalar_axes, override_specs=override_paths, stage="validation"),
        *(f"{path}={value}" for path, value in seed_overrides.items()),
        f"load.path={checkpoint_path}",
        f"run.root={stage_dir(results_root, STAGE_VALIDATION)}",
        "run.layout=flat",
        f"run.run_id={run_id}/{attempt_id}",
        f"study.name={study}",
        f"study.attempt_id={attempt_id}",
        f"experiment.name={study}",
        f"experiment.run_name={experiment_run_name(study, 'validation')}",
    ]
    if timezone is not None:
        overrides.append(f"run.timezone={timezone}")
    return overrides


def latest_train_attempt_id(results_root: str | Path, run_id: str) -> str | None:
    """Return the latest eligible train attempt for ``run_id``."""

    return latest_attempt_id(train_run_dir(results_root, run_id))


def _validation_config_from_grid(
    *,
    results_root: Path,
    grid_attempt_id: str,
    requested_config: str | None,
) -> str:
    if requested_config is not None:
        return requested_config
    grid_attempt = grid_attempt_dir(results_root, grid_attempt_id)
    manifest_path = grid_attempt / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        snapshots = config_snapshot_names(manifest.get("config_snapshots"))
        validation_snapshot = grid_attempt / snapshots["validation"]
        if validation_snapshot.is_file():
            return str(validation_snapshot)
        validation_config = manifest.get("validation_config")
        if validation_config:
            return str(validation_config)
    grid_snapshot = grid_attempt_dir(results_root, grid_attempt_id) / "grid.yaml"
    if grid_snapshot.is_file():
        grid_data = OmegaConf.to_container(OmegaConf.load(grid_snapshot), resolve=True)
        if isinstance(grid_data, dict):
            snapshots = config_snapshot_names(grid_data.get("config_snapshots"))
            validation_snapshot = grid_attempt / snapshots["validation"]
            if validation_snapshot.is_file():
                return str(validation_snapshot)
            config = grid_data.get("validation_config")
        else:
            config = None
        if config:
            return str(config)
    if DEFAULT_GRID.is_file():
        grid_data = OmegaConf.to_container(OmegaConf.load(DEFAULT_GRID), resolve=True)
        config = grid_data.get("validation_config") if isinstance(grid_data, dict) else None
        if config:
            return str(config)
    raise FileNotFoundError("validation config was not requested and no grid validation_config was found")


def _validation_attempt_id(args: argparse.Namespace, grid_attempt_id: str) -> str:
    """Return the validation attempt id: explicit override, else grid-derived."""

    return args.attempt_id or grid_attempt_id


def _selected_jobs(jobs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(job) for job in jobs]


def _train_attempt_id_for_job(
    *,
    args: argparse.Namespace,
    results_root: Path,
    run_id: str,
) -> str | None:
    if args.train_attempt_id is not None:
        return args.train_attempt_id
    return latest_train_attempt_id(results_root, run_id)


def plan_validation_jobs(
    jobs: Sequence[dict[str, Any]],
    *,
    args: argparse.Namespace,
    study: str,
    results_root: Path,
    grid_attempt_id: str,
    validation_config: str | Path,
    scalar_axes: Sequence[str],
    override_paths: dict[str, AxisOverrideSpec],
    seed_axis: str,
    static_stage_overrides: dict[str, object] | None = None,
    seed_policy: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build validation launch records and write source-train provenance."""

    selected = _selected_jobs(jobs)
    validation_attempt_id = _validation_attempt_id(args, grid_attempt_id)
    planned: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for job in selected:
        run_id = str(job["run_id"])
        train_attempt_id = _train_attempt_id_for_job(
            args=args,
            results_root=results_root,
            run_id=run_id,
        )
        if train_attempt_id is None:
            skipped.append({"run_id": run_id, "reason": "no eligible train attempt"})
            continue
        train_attempt = train_attempt_dir(results_root, run_id, train_attempt_id)
        if not _checkpoint_ready(train_attempt):
            skipped.append({"run_id": run_id, "reason": f"missing checkpoint in {train_attempt}"})
            continue

        point = dict(job.get("choices", {}))
        seed_values = job.get("seed_values")
        if isinstance(seed_values, dict):
            point.update(seed_values)
        if seed_axis not in point and "scan_seed" in job:
            point[seed_axis] = job["scan_seed"]
        checkpoint_path = train_attempt / "checkpoints"
        validation_attempt = validation_attempt_dir(results_root, run_id, validation_attempt_id)
        source = {
            "run_id": run_id,
            "grid_attempt_id": grid_attempt_id,
            "train_attempt_id": train_attempt_id,
            "train_dir": str(train_run_dir(results_root, run_id)),
            "train_attempt_dir": str(train_attempt),
            "checkpoint_path": str(checkpoint_path),
        }
        write_json(validation_attempt / "source_train_attempt.json", source)
        write_json(
            validation_attempt / "source_grid_attempt.json",
            {
                "run_id": run_id,
                "grid_attempt_id": grid_attempt_id,
                "grid_attempt_dir": str(grid_attempt_dir(results_root, grid_attempt_id)),
            },
        )

        overrides = validation_overrides(
            point,
            study=study,
            run_id=run_id,
            attempt_id=validation_attempt_id,
            results_root=results_root,
            checkpoint_path=checkpoint_path,
            scalar_axes=scalar_axes,
            override_paths=override_paths,
            seed_axis=seed_axis,
            seed_policy=seed_policy,
            timezone=_job_timezone(job),
        )
        command = _command_for(validation_config, overrides)
        command = launch.with_study_timezone(command, timezone=_job_timezone(job))
        command = launch.with_overrides(command, static_stage_overrides or {})
        write_latest(validation_attempt.parent, validation_attempt_id)
        planned.append(
            {
                "run_id": run_id,
                "train_attempt_id": train_attempt_id,
                "validation_attempt_id": validation_attempt_id,
                "validation_attempt_dir": str(validation_attempt),
                "source_train_attempt": source,
                "command": shlex.join(command),
                "command_parts": command,
            }
        )
    return planned, skipped


def write_validation_submission_records(
    jobs: Sequence[dict[str, Any]],
    *,
    grid_attempt_id: str,
    backend: str,
    job_ids: Sequence[str],
    submitted_commands: Sequence[Sequence[str]],
) -> None:
    """Write validation-stage submission provenance."""

    for index, (job, job_id) in enumerate(zip(jobs, job_ids, strict=True)):
        validation_attempt = Path(str(job["validation_attempt_dir"]))
        write_json(
            validation_attempt / "submission.json",
            {
                "run_id": str(job["run_id"]),
                "grid_attempt_id": grid_attempt_id,
                "train_attempt_id": str(job["train_attempt_id"]),
                "validation_attempt_id": str(job["validation_attempt_id"]),
                "launcher": backend,
                "launcher_job_id": str(job_id),
                "command": str(job["command"]),
                "submitted_command": shlex.join([str(part) for part in submitted_commands[index]]),
            },
        )


def _stage_plan_dir(results_root: Path, attempt_id: str) -> Path:
    """Return the durable validation stage-plan directory."""

    return stage_dir(results_root, STAGE_VALIDATION) / "stage_plans" / attempt_id


def _executor(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    results_root: Path,
    study: str,
    log_attempt: str,
):
    """Return the toolkit executor for validation submissions."""

    options = ExecutorOptions(
        backend=args.backend,
        args=args,
        repo_root=repo_root,
        log_dir=stage_dir(results_root, STAGE_VALIDATION) / "slurm_logs" / log_attempt,
        job_name=stage_job_name(study, "validate"),
        smoke=False,
        chunk_size=args.chunk_size,
        allow_partial_failures=True,
        chunk_status_dir=stage_dir(results_root, STAGE_VALIDATION) / "chunk_status" / log_attempt,
    )
    executor_cls = LocalExecutor if args.backend == "local" else SubmititExecutor
    return executor_cls(
        submit_command_sets=getattr(launch, "submit_command_sets"),
        options=options,
        claim_paths_for_statuses=launch.claim_paths_for_statuses,
    )


def _resource_spec(args: argparse.Namespace) -> Any:
    """Return a backend-neutral resource request for the selected device."""

    selector = launch.selected_device(args)
    profiles = launch.device_profiles(selector)
    resolved_profiles = {}
    for profile in profiles:
        uv_environment, uv_extras, _runtime_device = launch.resolve_uv_settings_for_profile(args, profile)
        slurm = launch.slurm_parameters(args, profile=profile)
        resolved_profiles[profile] = resource_from_profile(
            profile=profile,
            partition=slurm.get("slurm_partition"),
            timeout_min=slurm.get("timeout_min"),
            mem_gb=launch.slurm_resource_mem_gb(slurm),
            cpus=slurm.get("cpus_per_task"),
            gpus=slurm.get("gpus_per_node"),
            uv_environment=uv_environment,
            uv_extras=uv_extras,
        ).to_dict()
    if len(profiles) == 1:
        return resource_from_profile(
            profile=profiles[0],
            partition=resolved_profiles[profiles[0]].get("partition"),
            timeout_min=resolved_profiles[profiles[0]].get("timeout_min"),
            mem_gb=resolved_profiles[profiles[0]].get("mem_gb"),
            cpus=resolved_profiles[profiles[0]].get("threads"),
            gpus=resolved_profiles[profiles[0]].get("gpus"),
            uv_environment=resolved_profiles[profiles[0]].get("uv_environment"),
            uv_extras=resolved_profiles[profiles[0]].get("uv_extras", ()),
        )
    return resource_from_profile(
        profile=selector,
        partition=None,
        timeout_min=None,
        mem_gb=None,
        cpus=None,
        gpus=None,
        uv_environment=None,
        uv_extras=(),
        metadata={"profiles": resolved_profiles},
    )


def build_validation_stage_plan(
    jobs: Sequence[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    results_root: Path,
    grid_attempt_id: str,
    args: argparse.Namespace,
) -> StagePlan:
    """Build a reusable toolkit stage plan for validation tasks."""

    attempt_id = _validation_attempt_id(args, grid_attempt_id)
    result_dirs = [Path(str(job["validation_attempt_dir"])) for job in jobs]
    row_status_paths = [result_dir / "launcher_status.json" for result_dir in result_dirs]
    tasks = tasks_from_commands(
        stage=STAGE_VALIDATION,
        attempt_id=attempt_id,
        jobs=jobs,
        commands=[job["command_parts"] for job in jobs],
        result_dirs=result_dirs,
        row_status_paths=row_status_paths,
        resources=_resource_spec(args),
        completion_policy="status_completed",
        source_attempts={"grid": grid_attempt_id},
    )
    return StagePlan(
        study=study_name_from_manifest(manifest),
        stage=STAGE_VALIDATION,
        attempt_id=attempt_id,
        results_root=str(results_root),
        source_attempts={"grid": grid_attempt_id},
        timezone=manifest.get("timezone"),
        smoke=False,
        metadata={
            "backend": args.backend,
            "device": launch.selected_device(args),
            "chunk_size": args.chunk_size,
            "skips_allowed": True,
        },
        tasks=tasks,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse validation command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--grid-attempt-id", default=None, help="Grid attempt to validate (defaults to latest).")
    parser.add_argument("--config", default=None, help="Validation config path (defaults to grid.validation_config).")
    parser.add_argument(
        "--train-attempt-id",
        default=None,
        help="Exact train attempt to validate.",
    )
    parser.add_argument(
        "--attempt-id",
        default=None,
        help="Validation attempt id (defaults to the grid attempt id).",
    )
    launch.add_launch_arguments(
        parser,
        smoke_help=(
            "Deprecated. Use configs/smoke.yaml with the normal stage stack."
        ),
    )
    args = parser.parse_args(argv)
    launch.reject_deprecated_smoke(parser, args)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Launch validation jobs from existing ``00_grid`` and ``01_train`` attempts."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    repo_root = Path(args.repo_root) if args.repo_root else STUDY_DIR.parents[2]
    launch.ensure_submitit_launcher_environment(
        args,
        script_path=Path(__file__).resolve(),
        argv=raw_argv,
        repo_root=repo_root,
    )
    results_root = launch.repo_path(args.results_root, repo_root)
    grid_attempt_id = launch.resolve_grid_attempt_id(results_root, args.grid_attempt_id)
    manifest = launch.load_grid_manifest(results_root, grid_attempt_id)
    study = study_name_from_manifest(manifest)
    prefix = log_prefix(study)
    if args.wait_job:
        launch.submit_dependent_launcher(
            args.wait_job,
            script_path=Path(__file__).resolve(),
            argv=raw_argv,
            repo_root=repo_root,
            log_dir=stage_dir(results_root, STAGE_VALIDATION) / "slurm_logs" / "dependent_launchers",
            job_name=stage_job_name(study, "validate-launcher"),
            partition=args.wait_launcher_partition,
            timeout_min=args.wait_launcher_timeout_min,
            study=study,
        )
        return 0
    seed_policy = manifest.get("seed_overrides")
    static_stage_overrides = launch.static_overrides_for_stage(
        "validation",
        manifest=manifest,
    )
    scalar_axes = _scalar_axes(manifest)
    override_paths = _axis_override_paths(manifest, scalar_axes)
    seed_axis = str(manifest.get("scan_seed_axis", "seed"))
    validation_config = _validation_config_from_grid(
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        requested_config=args.config,
    )
    jobs, skipped = plan_validation_jobs(
        list(manifest.get("jobs", [])),
        args=args,
        study=study,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        validation_config=validation_config,
        scalar_axes=scalar_axes,
        override_paths=override_paths,
        seed_axis=seed_axis,
        static_stage_overrides=static_stage_overrides,
        seed_policy=seed_policy,
    )
    command_sets = launch.environment_command_sets(
        [job["command_parts"] for job in jobs],
        args=args,
        repo_root=repo_root,
    )
    submitted_commands = launch.summarize_command_sets(command_sets)

    if skipped:
        print(f"{prefix} skipped {len(skipped)} validation jobs without eligible checkpoints")
    if not jobs:
        print(f"{prefix} no validation jobs ready for 00_grid/{grid_attempt_id}")
        return 1 if manifest.get("jobs") else 0

    stage_plan = build_validation_stage_plan(
        jobs,
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=grid_attempt_id,
        args=args,
    )
    log_attempt = _validation_attempt_id(args, grid_attempt_id)
    stage_plan_dir = stage_plan.write(_stage_plan_dir(results_root, log_attempt))
    execution_records = _executor(
        args=args,
        repo_root=repo_root,
        results_root=results_root,
        study=study,
        log_attempt=log_attempt,
    ).submit(
        stage_plan,
        stage_plan.tasks,
        SubmissionRequest(
            command_sets=command_sets,
            submitted_commands=submitted_commands,
        ),
    )
    job_ids = [record.launcher_job_id for record in execution_records]

    write_validation_submission_records(
        jobs,
        grid_attempt_id=grid_attempt_id,
        backend=args.backend,
        job_ids=job_ids,
        submitted_commands=submitted_commands,
    )
    write_execution_records(stage_plan_dir, execution_records)
    print(f"{prefix} launched {len(job_ids)} validation jobs from 00_grid/{grid_attempt_id} via {args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
