"""Shared launch plumbing for staged study scripts.

This module owns the execution mechanics shared by ``train.py`` and
``validate.py``: CPU/CUDA profile defaults, uv environment activation, local
execution, and Submitit submission. Stage scripts own stage-specific command
construction and provenance.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence, TypeVar

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

_tpen_utils_io = sibling(__file__, 'utils.io')
read_json = _tpen_utils_io.read_json
_tpen_utils_layout = sibling(__file__, 'utils.layout')
STAGE_GRID = _tpen_utils_layout.STAGE_GRID
attempt_ids = _tpen_utils_layout.attempt_ids
grid_attempt_dir = _tpen_utils_layout.grid_attempt_dir
latest_attempt_id = _tpen_utils_layout.latest_attempt_id
stage_dir = _tpen_utils_layout.stage_dir
_tpen_utils_naming = sibling(__file__, 'utils.naming')
log_prefix = _tpen_utils_naming.log_prefix
_tpen_utils_overrides = sibling(__file__, 'utils.overrides')
rewrite_cli_overrides = _tpen_utils_overrides.rewrite_cli_overrides
_tpen_utils_time = sibling(__file__, 'utils.time')
DEFAULT_STUDY_TIMEZONE = _tpen_utils_time.DEFAULT_STUDY_TIMEZONE

DEFAULT_CPU_UV_ENVIRONMENT = ".venv"
DEFAULT_CUDA_UV_ENVIRONMENT = ".venv-gpu"
DEFAULT_SUBMITIT_UV_ENVIRONMENT = ".venv-submitit"
DEFAULT_CPU_EXTRA = "cpu"
DEFAULT_CUDA_EXTRA = "cu126"
DEVICE_CHOICES = ("cpu", "cuda", "cpu,cuda")
DEFAULT_CPU_PARTITION = "sapphire,kozinsky,seas_compute"
DEFAULT_CUDA_PARTITION = "seas_gpu,kozinsky_gpu"
DEFAULT_TIMEOUT_MIN = 30
DEFAULT_MEM_PER_CPU_GB = 8
DEFAULT_CPU_CPUS = 4
DEFAULT_CUDA_CPUS = 4
DEFAULT_ARRAY_PARALLELISM = 16
DEFAULT_CHUNK_SIZE = 1
DEFAULT_DEPENDENT_LAUNCHER_PARTITION = "test"
DEFAULT_DEPENDENT_LAUNCHER_TIMEOUT_MIN = 30
DEFAULT_DEPENDENT_LAUNCHER_MEM_PER_CPU_GB = 8
DEFAULT_DEPENDENT_LAUNCHER_CPUS = 1
DEFAULT_LOCAL_DEADLINE_GUARD_MIN = 60
DEPRECATED_SMOKE_MESSAGE = (
    "--smoke is deprecated; use --grid "
    "experiments/hooke/tpen-pair-scan-v1/configs/smoke.yaml with the normal stage stack"
)
STUDY_DIR = Path(__file__).resolve().parent
REPO_ROOT = STUDY_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.toolkit.task_state import (  # noqa: E402
    _attempt_already_completed,
    _claim_path_for_status,
    _claim_row,
    _deadline_guard_payload,
    _deadline_guard_reached,
    _read_json_mapping,
    _terminal_row_status,
    _write_claim,
    _write_status,
    claim_paths_for_statuses,
    local_claim_deadline_unix,
    parse_deadline_unix,
)

T = TypeVar("T")


def repo_path(path: str | Path, repo_root: Path) -> Path:
    """Return ``path`` anchored at ``repo_root`` when it is relative."""

    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def resolve_grid_attempt_id(results_root: str | Path, grid_attempt_id: str | None) -> str:
    """Return the requested grid attempt id, defaulting to ``00_grid/latest``."""

    if grid_attempt_id is not None:
        return grid_attempt_id
    grid_stage = stage_dir(results_root, STAGE_GRID)
    latest = latest_attempt_id(grid_stage)
    if latest is not None:
        return latest
    ids = attempt_ids(grid_stage)
    if not ids:
        raise FileNotFoundError(f"no grid attempts under {grid_stage}")
    return ids[-1]


def strip_wait_job_args(argv: Sequence[str]) -> list[str]:
    """Return command-line args with ``--wait-job`` removed."""

    stripped: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        text = str(item)
        if text == "--wait-job":
            skip_next = True
            continue
        if text.startswith("--wait-job="):
            continue
        stripped.append(text)
    return stripped


def dependent_launcher_command(
    *,
    script_path: str | Path,
    argv: Sequence[str],
) -> list[str]:
    """Return the stage-launcher command rerun by a dependent Slurm job."""

    return [
        "uv",
        "run",
        "--extra",
        "submitit",
        "python",
        "-u",
        str(Path(script_path).resolve()),
        *strip_wait_job_args(argv),
    ]


def _slurm_time(minutes: int) -> str:
    if minutes < 1:
        raise ValueError("timeout minutes must be >= 1")
    hours, mins = divmod(int(minutes), 60)
    return f"{hours:02d}:{mins:02d}:00"


def _format_gb(value: int) -> str:
    if value < 1:
        raise ValueError("memory GB must be >= 1")
    return f"{int(value)}G"


def _parse_gb(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    for suffix in ("GB", "G"):
        if text.endswith(suffix):
            return int(text[: -len(suffix)])
    return int(text)


def slurm_resource_mem_gb(slurm: dict[str, Any]) -> int | None:
    """Return total requested memory in GB for resource metadata only."""

    if slurm.get("mem_gb") is not None:
        return int(slurm["mem_gb"])
    cpus = int(slurm.get("cpus_per_task") or 1)
    mem_per_cpu_gb = _parse_gb(slurm.get("mem_per_cpu"))
    if mem_per_cpu_gb is not None:
        return mem_per_cpu_gb * cpus
    mem_per_gpu_gb = _parse_gb(slurm.get("mem_per_gpu"))
    gpus = int(slurm.get("gpus_per_node") or slurm.get("gpus_per_task") or 1)
    if mem_per_gpu_gb is not None:
        return mem_per_gpu_gb * gpus
    return None


def submit_dependent_launcher(
    job_id: str,
    *,
    script_path: str | Path,
    argv: Sequence[str],
    repo_root: str | Path,
    log_dir: str | Path,
    job_name: str,
    partition: str = DEFAULT_DEPENDENT_LAUNCHER_PARTITION,
    timeout_min: int = DEFAULT_DEPENDENT_LAUNCHER_TIMEOUT_MIN,
    study: str | None = None,
) -> str:
    """Submit a lightweight Slurm job that reruns this stage after ``job_id``.

    The dependent job strips ``--wait-job`` from the original stage command, so
    it performs normal readiness checks and downstream submission once Slurm
    releases the dependency.
    """

    job_id = str(job_id).strip()
    if not job_id:
        return ""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    command = dependent_launcher_command(script_path=script_path, argv=argv)
    script = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(Path(repo_root).resolve()))}",
            _submitit_import_path_setup(),
            f"export UV_PROJECT_ENVIRONMENT={shlex.quote(DEFAULT_SUBMITIT_UV_ENVIRONMENT)}",
            f"exec {shlex.join(command)}",
            "",
        ]
    )
    prefix = log_prefix(study)
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--dependency=afterany:{job_id}",
            f"--job-name={job_name}",
            f"--partition={partition}",
            f"--time={_slurm_time(timeout_min)}",
            f"--mem-per-cpu={_format_gb(DEFAULT_DEPENDENT_LAUNCHER_MEM_PER_CPU_GB)}",
            f"--cpus-per-task={DEFAULT_DEPENDENT_LAUNCHER_CPUS}",
            f"--output={log_dir / '%x-%j.out'}",
        ],
        input=script,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"sbatch dependent launcher failed for {job_id}: {message}")
    submitted = (result.stdout.strip().splitlines() or [""])[-1].split(";", maxsplit=1)[0]
    if not submitted:
        raise RuntimeError("sbatch dependent launcher did not return a job id")
    print(f"{prefix} submitted dependent launcher {submitted} after Slurm job {job_id}")
    return submitted


def load_grid_manifest(results_root: str | Path, grid_attempt_id: str) -> dict[str, Any]:
    """Read the ``00_grid`` manifest for ``grid_attempt_id``."""

    manifest_path = grid_attempt_dir(results_root, grid_attempt_id) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"grid attempt has no manifest.json: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("stage") != STAGE_GRID:
        raise ValueError(f"manifest {manifest_path} is not a {STAGE_GRID} manifest")
    return manifest


def static_overrides_for_stage(
    stage: str,
    *,
    manifest: dict[str, Any],
) -> dict[str, object]:
    """Return manifest static override values for one stage."""

    configured = manifest.get("static_overrides") or {}
    if not isinstance(configured, dict):
        raise ValueError("manifest static_overrides must be a mapping")
    values = configured.get(stage) or {}
    if not isinstance(values, dict):
        raise ValueError(f"manifest static_overrides.{stage} must be a mapping")
    return {str(key): value for key, value in values.items()}


def command_for_job(job: dict[str, Any]) -> list[str]:
    """Return the exact command stored in a manifest job."""

    command = job.get("command")
    if isinstance(command, str) and command.strip():
        return shlex.split(command)
    if isinstance(command, list) and command:
        return [str(part) for part in command]
    raise ValueError(f"job {job.get('run_id', '<unknown>')!r} has no command")


def with_overrides(command: Sequence[str], overrides: dict[str, object]) -> list[str]:
    """Return ``command`` with final scalar OmegaConf overrides appended."""

    return rewrite_cli_overrides(command, overrides)


def with_runtime_device(command: Sequence[str], *, device: str) -> list[str]:
    """Return ``command`` with a final runtime.device override."""

    return with_overrides(command, {"runtime.device": device})


def with_study_timezone(command: Sequence[str], *, timezone: str | None = None) -> list[str]:
    """Return ``command`` with the study's launcher-owned timezone override."""

    return with_overrides(command, {"run.timezone": timezone or DEFAULT_STUDY_TIMEZONE})


def environment_defaults(profile: str) -> tuple[str, list[str], str]:
    """Return default uv environment, uv extras, and runtime device."""

    if profile == "cuda":
        return DEFAULT_CUDA_UV_ENVIRONMENT, [DEFAULT_CUDA_EXTRA], "cuda"
    return DEFAULT_CPU_UV_ENVIRONMENT, [DEFAULT_CPU_EXTRA], "cpu"


def normalize_device(value: str | None) -> str:
    """Return the canonical launcher device selector."""

    if value is None:
        return "cpu"
    raw = str(value).strip().lower()
    if raw in {"cpu", "cuda"}:
        return raw
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(parts) == 2 and set(parts) == {"cpu", "cuda"}:
        return "cpu,cuda"
    raise argparse.ArgumentTypeError(
        f"device must be one of {', '.join(DEVICE_CHOICES)}"
    )


def selected_device(args: argparse.Namespace) -> str:
    """Return the parsed device selector, accepting the old profile attribute."""

    return normalize_device(getattr(args, "device", getattr(args, "profile", None)))


def device_profiles(device: str) -> tuple[str, ...]:
    """Return the concrete execution profiles requested by ``device``."""

    normalized = normalize_device(device)
    if normalized == "cpu,cuda":
        return ("cpu", "cuda")
    return (normalized,)


def resolve_uv_settings_for_profile(args: argparse.Namespace, profile: str) -> tuple[str, list[str], str]:
    """Return uv environment, uv extras, and runtime device for one profile."""

    uv_environment, uv_extras, runtime_device = environment_defaults(profile)
    uv_environment = args.uv_environment or (
        args.gpu_uv_environment if profile == "cuda" else None
    ) or uv_environment
    uv_extras = args.uv_extras or (args.gpu_extras if profile == "cuda" else None) or uv_extras
    return uv_environment, list(uv_extras), runtime_device


def resolve_uv_settings(args: argparse.Namespace) -> tuple[str, list[str], str]:
    """Return uv environment, uv extras, and runtime device for parsed args."""

    profiles = device_profiles(selected_device(args))
    if len(profiles) != 1:
        raise ValueError("mixed cpu,cuda mode requires environment_command_sets")
    return resolve_uv_settings_for_profile(args, profiles[0])


def _python_in_environment(environment: str | Path, *, repo_root: str | Path) -> bool:
    """Return whether the current Python executable lives under ``environment``."""

    environment_path = Path(environment)
    if not environment_path.is_absolute():
        environment_path = Path(repo_root) / environment_path
    try:
        executable = Path(sys.executable).resolve()
        environment_path = environment_path.resolve()
    except OSError:
        return False
    return executable == environment_path or environment_path in executable.parents


def ensure_submitit_launcher_environment(
    args: argparse.Namespace,
    *,
    script_path: str | Path,
    argv: Sequence[str],
    repo_root: str | Path,
) -> None:
    """Re-exec Submitit launchers from a lightweight environment."""

    if getattr(args, "backend", None) != "submitit":
        return
    if os.environ.get("SUBMITIT_EXECUTOR") or os.environ.get("SPENN_SUBMITIT_LAUNCHER_REEXEC"):
        return
    if _python_in_environment(DEFAULT_SUBMITIT_UV_ENVIRONMENT, repo_root=repo_root):
        return
    env = dict(os.environ)
    env["UV_PROJECT_ENVIRONMENT"] = DEFAULT_SUBMITIT_UV_ENVIRONMENT
    env["SPENN_SUBMITIT_LAUNCHER_REEXEC"] = "1"
    command = [
        "uv",
        "run",
        "--extra",
        "submitit",
        "python",
        "-u",
        str(Path(script_path).resolve()),
        *[str(item) for item in argv],
    ]
    os.execvpe(command[0], command, env)


def _uses_python_executable(command_part: str) -> bool:
    """Return whether ``command_part`` names a Python executable."""

    return Path(command_part).name.startswith("python")


def _activated_python_command(command: Sequence[str]) -> list[str]:
    """Run planned Python commands through the currently active environment."""

    command = [str(part) for part in command]
    if command and _uses_python_executable(command[0]):
        return ["python", *command[1:]]
    return command


def _cpu_thread_exports(device: str) -> list[str]:
    """Return shell lines that align CPU thread pools with Slurm allocation."""

    if device != "cpu":
        return []
    thread_count = "${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}"
    return [
        f"export OMP_NUM_THREADS={thread_count}",
        f"export MKL_NUM_THREADS={thread_count}",
        f"export OPENBLAS_NUM_THREADS={thread_count}",
        f"export NUMEXPR_NUM_THREADS={thread_count}",
        f"export VECLIB_MAXIMUM_THREADS={thread_count}",
    ]


def environment_shell_command(
    command: Sequence[str],
    *,
    repo_root: Path,
    uv_environment: str,
    uv_extras: Sequence[str],
    device: str,
) -> list[str]:
    """Wrap a run command in the selected uv environment setup."""

    sync_command = ["uv", "sync"]
    for extra in uv_extras:
        sync_command.extend(["--extra", str(extra)])
    sync_lock = repo_root.parent / f".{repo_root.name}.uv-sync.lock"
    activate_path = Path(uv_environment) / "bin" / "activate"
    run_command = _activated_python_command(with_runtime_device(command, device=device))
    script = "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
            f"export UV_PROJECT_ENVIRONMENT={shlex.quote(str(uv_environment))}",
            *_cpu_thread_exports(device),
            "# Submitit chunks share this checkout and must not sync concurrently.",
            f"flock {shlex.quote(str(sync_lock))} {shlex.join(sync_command)}",
            f"source {shlex.quote(str(activate_path))}",
            f"exec {shlex.join(run_command)}",
        ]
    )
    return ["bash", "-lc", script]


def environment_command_sets(
    commands: Sequence[Sequence[str]],
    *,
    args: argparse.Namespace,
    repo_root: Path,
) -> dict[str, list[list[str]]]:
    """Return prepared run commands for each selected concrete profile."""

    command_sets: dict[str, list[list[str]]] = {}
    for profile in device_profiles(selected_device(args)):
        uv_environment, uv_extras, runtime_device = resolve_uv_settings_for_profile(args, profile)
        command_sets[profile] = [
            environment_shell_command(
                command,
                repo_root=repo_root,
                uv_environment=uv_environment,
                uv_extras=uv_extras,
                device=runtime_device,
            )
            for command in commands
        ]
    return command_sets


def summarize_command_sets(command_sets: dict[str, list[list[str]]]) -> list[list[str]]:
    """Return one provenance command per row for single or mixed submission."""

    if not command_sets:
        return []
    profiles = tuple(command_sets)
    if len(profiles) == 1:
        return list(command_sets[profiles[0]])
    n_commands = len(command_sets[profiles[0]])
    return [
        [
            "device-candidates",
            *[
                f"{profile}={shlex.join([str(part) for part in command_sets[profile][index]])}"
                for profile in profiles
            ],
        ]
        for index in range(n_commands)
    ]


def positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def nonnegative_int(value: str) -> int:
    """Parse a non-negative integer CLI value."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def balanced_chunks(items: Sequence[T], *, chunk_size: int) -> list[list[T]]:
    """Split ``items`` into evenly sized chunks no larger than ``chunk_size``."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    items = list(items)
    if not items:
        return []
    n_chunks = (len(items) + chunk_size - 1) // chunk_size
    base_size, extra = divmod(len(items), n_chunks)
    chunks: list[list[T]] = []
    start = 0
    for index in range(n_chunks):
        size = base_size + (1 if index < extra else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


def run_command_chunk(
    commands: Sequence[Sequence[str]],
    cwd: str | Path | None = None,
    allow_partial_failures: bool = False,
    row_status_paths: Sequence[str | Path | None] | None = None,
    chunk_status_path: str | Path | None = None,
    claim_paths: Sequence[str | Path | None] | None = None,
    claim_label: str | None = None,
    claim_deadline_unix: float | None = None,
    claim_deadline_guard_min: int = 0,
) -> dict[str, Any]:
    """Run prepared commands sequentially inside one local/Submitit chunk.

    With ``allow_partial_failures=False`` this preserves fail-fast scheduler
    semantics for training chunks. Evaluation launchers pass
    ``allow_partial_failures=True`` so one failed eval row records a per-row
    failure but does not abort the rest of the chunk.
    """

    failures = []
    row_results = []
    status_paths = list(row_status_paths or [None] * len(commands))
    if len(status_paths) != len(commands):
        _write_status(
            chunk_status_path,
            {
                "status": "failed",
                "n_commands": len(commands),
                "n_failed": 0,
                "error": "row_status_paths must match commands length",
            },
        )
        raise ValueError("row_status_paths must match commands length")
    claims = list(claim_paths or [None] * len(commands))
    if len(claims) != len(commands):
        _write_status(
            chunk_status_path,
            {
                "status": "failed",
                "n_commands": len(commands),
                "n_failed": 0,
                "error": "claim_paths must match commands length",
            },
        )
        raise ValueError("claim_paths must match commands length")
    deadline_guarded = False
    for index, command in enumerate(commands):
        command = [str(part) for part in command]
        command_text = shlex.join(command)
        if _attempt_already_completed(status_paths[index]):
            row_results.append(
                {
                    "status": "skipped_completed",
                    "chunk_index": index,
                    "command": command_text,
                    "claim_label": claim_label,
                }
            )
            continue
        if _deadline_guard_reached(claim_deadline_unix, claim_deadline_guard_min):
            row_results.append(
                _deadline_guard_payload(
                    index=index,
                    command=command_text,
                    claim_label=claim_label,
                    deadline_unix=claim_deadline_unix,
                    guard_min=claim_deadline_guard_min,
                )
            )
            deadline_guarded = True
            break
        claim_payload = {
            "status": "claimed",
            "chunk_index": index,
            "claim_label": claim_label,
            "command": command_text,
        }
        if not _claim_row(claims[index], claim_payload, status_paths[index]):
            row_results.append(
                {
                    "status": "skipped_claimed",
                    "chunk_index": index,
                    "command": command_text,
                    "claim_label": claim_label,
                }
            )
            continue
        _write_status(
            status_paths[index],
            {
                "status": "running",
                "chunk_index": index,
                "command": command_text,
                "claim_label": claim_label,
            },
        )
        try:
            result = subprocess.run(
                command,
                cwd=str(cwd) if cwd is not None else None,
                check=False,
            )
            returncode: int | None = int(result.returncode)
            error = ""
        except Exception as exc:
            returncode = None
            error = repr(exc)
        row_status = "success" if returncode == 0 else "failed"
        row_payload = {
            "status": row_status,
            "chunk_index": index,
            "returncode": returncode,
            "command": command_text,
        }
        if error:
            row_payload["error"] = error
        _write_status(status_paths[index], row_payload)
        row_results.append(row_payload)
        if row_status != "success":
            failures.append((index, returncode, command_text))
            if returncode is None and not allow_partial_failures:
                break
    if deadline_guarded and not failures:
        chunk_status = "deadline_guard"
    else:
        chunk_status = "success" if not failures else "partial_failed"
    chunk_payload = {
        "status": chunk_status,
        "n_commands": len(commands),
        "n_failed": len(failures),
        "rows": row_results,
    }
    if deadline_guarded:
        chunk_payload["deadline_unix"] = claim_deadline_unix
        chunk_payload["guard_min"] = claim_deadline_guard_min
    _write_status(chunk_status_path, chunk_payload)
    if failures:
        lines = [
            f"{len(failures)} of {len(commands)} command(s) failed in this chunk:",
            *[
                f"  chunk item {index}: return code {returncode}; {command}"
                for index, returncode, command in failures
            ],
        ]
        if not allow_partial_failures:
            raise RuntimeError("\n".join(lines))
        chunk_payload["message"] = "\n".join(lines)
    return chunk_payload


def _expanded_chunk_job_ids(chunk_job_ids: Sequence[str], chunks: Sequence[Sequence[Any]]) -> list[str]:
    """Return one launcher job id per original command."""

    expanded: list[str] = []
    for job_id, chunk in zip(chunk_job_ids, chunks, strict=True):
        expanded.extend([str(job_id)] * len(chunk))
    return expanded


def _print_submitted_slurm_job_ids(job_name: str, job_ids: Sequence[str]) -> None:
    """Print distinct parent Slurm job IDs after successful submission."""

    parent_ids: list[str] = []
    for job_id in job_ids:
        text = str(job_id)
        candidate = text.split("_", maxsplit=1)[0]
        parent_id = candidate if candidate.isdigit() else text
        if parent_id not in parent_ids:
            parent_ids.append(parent_id)
    if not parent_ids:
        return
    label = "job_id" if len(parent_ids) == 1 else "job_ids"
    print(f"[{job_name}] submitted Slurm {label}={','.join(parent_ids)}", flush=True)


def _submitit_import_path_setup() -> str:
    """Return a Slurm setup line that makes this script module importable.

    Submitit unpickles the mapped callable before the command payload runs.
    The stage scripts import this file as top-level ``launch``, so Slurm array
    workers must have the study directory on ``PYTHONPATH`` before Python starts.
    """

    paths = ":".join(shlex.quote(str(path)) for path in (STUDY_DIR, REPO_ROOT))
    return f"export PYTHONPATH={paths}${{PYTHONPATH:+:$PYTHONPATH}}"


def _slurm_memory_env_setup(slurm: dict[str, Any]) -> str | None:
    """Return setup that removes inherited conflicting Slurm memory exports."""

    memory_env_by_parameter = {
        "mem_gb": "SLURM_MEM_PER_NODE",
        "mem_per_cpu": "SLURM_MEM_PER_CPU",
        "mem_per_gpu": "SLURM_MEM_PER_GPU",
    }
    requested = [
        env_name
        for parameter, env_name in memory_env_by_parameter.items()
        if slurm.get(parameter) is not None
    ]
    if len(requested) != 1:
        return None
    conflicts = [
        env_name
        for env_name in memory_env_by_parameter.values()
        if env_name not in requested
    ]
    return f"unset {' '.join(conflicts)}"


def _with_submitit_import_path(slurm: dict[str, Any]) -> dict[str, Any]:
    """Return Slurm parameters with the launcher import path setup prepended."""

    existing = slurm.get("slurm_setup", [])
    if isinstance(existing, str):
        setup = [existing]
    else:
        setup = [str(line) for line in existing]
    import_path_setup = _submitit_import_path_setup()
    if import_path_setup not in setup:
        setup = [import_path_setup, *setup]
    memory_env_setup = _slurm_memory_env_setup(slurm)
    if memory_env_setup is not None and memory_env_setup not in setup:
        setup.append(memory_env_setup)
    cpu_bind_setup = "export SLURM_CPU_BIND=none"
    if cpu_bind_setup not in setup:
        setup.append(cpu_bind_setup)
    return {**slurm, "slurm_setup": setup}


def submit_local(
    commands: Sequence[Sequence[str]],
    *,
    repo_root: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    allow_partial_failures: bool = False,
    row_status_paths: Sequence[str | Path | None] | None = None,
    chunk_status_dir: str | Path | None = None,
    claim_paths: Sequence[str | Path | None] | None = None,
    claim_label: str | None = None,
    claim_deadline_unix: float | None = None,
    claim_deadline_guard_min: int = 0,
) -> list[str]:
    """Run commands sequentially in-process, grouped into balanced chunks."""

    job_ids = []
    chunks = balanced_chunks(commands, chunk_size=chunk_size)
    status_chunks = balanced_chunks(row_status_paths or [None] * len(commands), chunk_size=chunk_size)
    claim_chunks = balanced_chunks(claim_paths or [None] * len(commands), chunk_size=chunk_size)
    for index, chunk in enumerate(chunks):
        chunk_status_path = None
        if chunk_status_dir is not None:
            chunk_status_path = Path(chunk_status_dir) / f"chunk-{index:04d}.json"
        if _deadline_guard_reached(claim_deadline_unix, claim_deadline_guard_min):
            _write_status(
                chunk_status_path,
                {
                    "status": "deadline_guard",
                    "n_commands": len(chunk),
                    "n_failed": 0,
                    "rows": [],
                    "deadline_unix": claim_deadline_unix,
                    "guard_min": claim_deadline_guard_min,
                },
            )
            break
        try:
            chunk_result = run_command_chunk(
                chunk,
                cwd=repo_root,
                allow_partial_failures=allow_partial_failures,
                row_status_paths=status_chunks[index],
                chunk_status_path=chunk_status_path,
                claim_paths=claim_chunks[index],
                claim_label=claim_label,
                claim_deadline_unix=claim_deadline_unix,
                claim_deadline_guard_min=claim_deadline_guard_min,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"local chunk {index} failed") from exc
        job_ids.extend([f"local-chunk-{index}-rc0"] * len(chunk))
        if chunk_result.get("status") == "deadline_guard":
            break
    return job_ids


def submit_submitit(
    commands: Sequence[Sequence[str]],
    *,
    log_dir: Path,
    job_name: str,
    slurm: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    allow_partial_failures: bool = False,
    row_status_paths: Sequence[str | Path | None] | None = None,
    chunk_status_dir: str | Path | None = None,
    claim_paths: Sequence[str | Path | None] | None = None,
    claim_label: str | None = None,
) -> list[str]:
    """Submit prepared commands through Submitit as balanced array chunks."""

    try:
        import submitit  # lazy: optional 'submitit' extra
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "submitit backend requires the optional dependency; install with "
            "`uv sync --extra submitit`"
        ) from exc

    log_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(log_dir))
    slurm = _with_submitit_import_path(slurm)
    executor.update_parameters(name=job_name, **slurm)
    command_chunks = [
        [[str(part) for part in command] for command in chunk]
        for chunk in balanced_chunks(commands, chunk_size=chunk_size)
    ]
    if (
        allow_partial_failures
        or row_status_paths is not None
        or chunk_status_dir is not None
        or claim_paths is not None
    ):
        status_chunks = balanced_chunks(row_status_paths or [None] * len(commands), chunk_size=chunk_size)
        claim_chunks = balanced_chunks(claim_paths or [None] * len(commands), chunk_size=chunk_size)
        chunk_status_paths = [
            None if chunk_status_dir is None else str(Path(chunk_status_dir) / f"chunk-{index:04d}.json")
            for index, _chunk in enumerate(command_chunks)
        ]
        jobs = executor.map_array(
            run_command_chunk,
            command_chunks,
            [None] * len(command_chunks),
            [allow_partial_failures] * len(command_chunks),
            status_chunks,
            chunk_status_paths,
            claim_chunks,
            [claim_label] * len(command_chunks),
        )
    else:
        jobs = executor.map_array(run_command_chunk, command_chunks)
    chunk_job_ids = [str(job.job_id) for job in jobs]
    _print_submitted_slurm_job_ids(job_name, chunk_job_ids)
    return _expanded_chunk_job_ids(chunk_job_ids, command_chunks)


def _summarize_profile_job_ids(job_ids_by_profile: dict[str, list[str]]) -> list[str]:
    """Return one provenance job id string per original row."""

    if not job_ids_by_profile:
        return []
    profiles = tuple(job_ids_by_profile)
    if len(profiles) == 1:
        return list(job_ids_by_profile[profiles[0]])
    n_jobs = len(job_ids_by_profile[profiles[0]])
    return [
        ",".join(f"{profile}:{job_ids_by_profile[profile][index]}" for profile in profiles)
        for index in range(n_jobs)
    ]


def _cuda_visible_to_local_process() -> bool:
    """Return whether local mixed mode should prefer CUDA."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible.lower() not in {"", "none", "no_dev_files"}:
        return True
    return bool(os.environ.get("SLURM_JOB_GPUS"))


def _local_profile(profiles: Sequence[str]) -> str:
    """Return the concrete profile to use for a local submission."""

    if len(profiles) == 1:
        return profiles[0]
    return "cuda" if "cuda" in profiles and _cuda_visible_to_local_process() else "cpu"


def submit_command_sets(
    command_sets: dict[str, list[list[str]]],
    *,
    args: argparse.Namespace,
    backend: str,
    repo_root: Path,
    log_dir: Path,
    job_name: str,
    smoke: bool,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    allow_partial_failures: bool = False,
    row_status_paths: Sequence[str | Path | None] | None = None,
    chunk_status_dir: str | Path | None = None,
    claim_rows: bool = False,
) -> list[str]:
    """Submit prepared command sets for one or more concrete profiles."""

    profiles = tuple(command_sets)
    if not profiles:
        return []
    if backend == "local":
        profile = _local_profile(profiles)
        use_claims = len(profiles) > 1 or claim_rows
        claim_paths = claim_paths_for_statuses(row_status_paths) if use_claims else None
        claim_deadline_unix = local_claim_deadline_unix(args) if use_claims else None
        claim_deadline_guard_min = (
            getattr(args, "local_deadline_guard_min", DEFAULT_LOCAL_DEADLINE_GUARD_MIN)
            if use_claims
            else 0
        )
        local_kwargs = {
            "repo_root": repo_root,
            "chunk_size": chunk_size,
            "row_status_paths": row_status_paths,
            "chunk_status_dir": chunk_status_dir,
            "claim_paths": claim_paths,
            "claim_label": f"local-{profile}" if use_claims else None,
            "claim_deadline_unix": claim_deadline_unix,
            "claim_deadline_guard_min": claim_deadline_guard_min,
        }
        if allow_partial_failures:
            local_kwargs["allow_partial_failures"] = True
        return submit_local(command_sets[profile], **local_kwargs)
    if len(profiles) > 1 and row_status_paths is None:
        raise ValueError("mixed cpu,cuda Submitit mode requires row_status_paths")
    use_claims = len(profiles) > 1 or claim_rows
    claim_paths = claim_paths_for_statuses(row_status_paths) if use_claims else None
    job_ids_by_profile: dict[str, list[str]] = {}
    for profile in profiles:
        profile_log_dir = log_dir if len(profiles) == 1 else log_dir / profile
        profile_chunk_status_dir = (
            None
            if chunk_status_dir is None
            else (Path(chunk_status_dir) if len(profiles) == 1 else Path(chunk_status_dir) / profile)
        )
        profile_job_name = job_name if len(profiles) == 1 else f"{job_name}-{profile}"
        job_ids_by_profile[profile] = submit_submitit(
            command_sets[profile],
            log_dir=profile_log_dir,
            job_name=profile_job_name,
            slurm=slurm_parameters(args, profile=profile),
            chunk_size=chunk_size,
            allow_partial_failures=allow_partial_failures,
            row_status_paths=row_status_paths,
            chunk_status_dir=profile_chunk_status_dir,
            claim_paths=claim_paths,
            claim_label=profile if use_claims else None,
        )
    return _summarize_profile_job_ids(job_ids_by_profile)


def slurm_parameters(args: argparse.Namespace, *, profile: str) -> dict[str, Any]:
    """Return Submitit Slurm parameters for the selected profile."""

    profile_partition = (
        getattr(args, "slurm_cuda_partition", None)
        if profile == "cuda"
        else getattr(args, "slurm_cpu_partition", None)
    )
    partition = profile_partition or args.slurm_partition or (
        DEFAULT_CUDA_PARTITION if profile == "cuda" else DEFAULT_CPU_PARTITION
    )
    array_parallelism = args.slurm_array_parallelism
    if array_parallelism is None:
        array_parallelism = DEFAULT_ARRAY_PARALLELISM
    if array_parallelism < 0:
        raise ValueError("slurm_array_parallelism must be >= 0")
    cpus_per_task = args.slurm_cpus or (DEFAULT_CPU_CPUS if profile == "cpu" else DEFAULT_CUDA_CPUS)
    profile_timeout = (
        getattr(args, "slurm_cuda_timeout_min", None)
        if profile == "cuda"
        else getattr(args, "slurm_cpu_timeout_min", None)
    )
    requested_mem_per_cpu_gb = getattr(args, "slurm_mem_per_cpu_gb", None)
    legacy_mem_gb = getattr(args, "slurm_mem_gb", None)
    if requested_mem_per_cpu_gb is not None:
        mem_per_cpu_gb = requested_mem_per_cpu_gb
    elif legacy_mem_gb is not None:
        mem_per_cpu_gb = max(DEFAULT_MEM_PER_CPU_GB, (int(legacy_mem_gb) + cpus_per_task - 1) // cpus_per_task)
    else:
        mem_per_cpu_gb = DEFAULT_MEM_PER_CPU_GB
    slurm = {
        "slurm_partition": partition,
        "timeout_min": profile_timeout or args.slurm_timeout_min or DEFAULT_TIMEOUT_MIN,
        "mem_per_cpu": _format_gb(mem_per_cpu_gb),
        "cpus_per_task": cpus_per_task,
        "tasks_per_node": 1,
    }
    if array_parallelism > 0:
        slurm["slurm_array_parallelism"] = array_parallelism
    if profile == "cuda":
        slurm["gpus_per_node"] = args.slurm_gpus or 1
    return slurm


def add_launch_arguments(parser: argparse.ArgumentParser, *, smoke_help: str) -> None:
    """Add shared local/Submitit and device launch arguments."""

    parser.add_argument("--smoke", action="store_true", help=f"Deprecated. {DEPRECATED_SMOKE_MESSAGE}")
    parser.add_argument("--backend", choices=["local", "submitit"], required=True)
    parser.add_argument(
        "--device",
        type=normalize_device,
        choices=DEVICE_CHOICES,
        default="cpu",
        help=(
            "Execution device selector: cpu, cuda, or cpu,cuda. "
            "cpu,cuda submits separate CPU and CUDA candidates; the first "
            "candidate that starts claims each row. Default: cpu."
        ),
    )
    parser.add_argument(
        "--cpu",
        action="store_const",
        const="cpu",
        dest="device",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--cuda",
        action="store_const",
        const="cuda",
        dest="device",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--repo-root", default=None, help="Repo root for command working directory.")
    parser.add_argument(
        "--wait-job",
        default=None,
        help=(
            "Submit a lightweight Slurm launcher with dependency afterany:<job> "
            "and exit; the dependent launcher reruns this command without --wait-job."
        ),
    )
    parser.add_argument(
        "--wait-launcher-partition",
        default=DEFAULT_DEPENDENT_LAUNCHER_PARTITION,
        help=(
            "Slurm partition for the lightweight --wait-job launcher "
            f"(default {DEFAULT_DEPENDENT_LAUNCHER_PARTITION})."
        ),
    )
    parser.add_argument(
        "--wait-launcher-timeout-min",
        type=int,
        default=DEFAULT_DEPENDENT_LAUNCHER_TIMEOUT_MIN,
        help=(
            "Wall time in minutes for the lightweight --wait-job launcher "
            f"(default {DEFAULT_DEPENDENT_LAUNCHER_TIMEOUT_MIN})."
        ),
    )
    parser.add_argument(
        "--slurm-partition",
        default=None,
        help=(
            "Override the Slurm partition for all selected devices. Defaults "
            "to sapphire,kozinsky,seas_compute for CPU and "
            "seas_gpu,kozinsky_gpu for CUDA."
        ),
    )
    parser.add_argument("--slurm-cpu-partition", default=None, help="CPU partition override for --device cpu,cuda.")
    parser.add_argument("--slurm-cuda-partition", default=None, help="CUDA partition override for --device cpu,cuda.")
    parser.add_argument("--slurm-gpus", type=int, default=None, help="CUDA only; defaults to 1.")
    parser.add_argument("--slurm-timeout-min", type=int, default=None)
    parser.add_argument("--slurm-cpu-timeout-min", type=int, default=None)
    parser.add_argument("--slurm-cuda-timeout-min", type=int, default=None)
    parser.add_argument("--slurm-mem-per-cpu-gb", type=positive_int, default=None)
    parser.add_argument("--slurm-mem-gb", type=positive_int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--slurm-cpus", type=int, default=None)
    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Maximum desired run commands per local/Submitit chunk. Chunks are "
            f"balanced evenly; default {DEFAULT_CHUNK_SIZE} keeps one command per array task."
        ),
    )
    parser.add_argument(
        "--local-deadline",
        default=None,
        help=(
            "Local claiming deadline as UNIX seconds or an ISO timestamp. "
            "Defaults to SLURM_JOB_END_TIME when that environment variable is set."
        ),
    )
    parser.add_argument(
        "--local-deadline-guard-min",
        type=nonnegative_int,
        default=DEFAULT_LOCAL_DEADLINE_GUARD_MIN,
        help=(
            "For local claimers, stop claiming new rows this many minutes before "
            "the local deadline; set 0 to disable. "
            f"Default: {DEFAULT_LOCAL_DEADLINE_GUARD_MIN}."
        ),
    )
    parser.add_argument(
        "--slurm-array-parallelism",
        type=int,
        default=None,
        help=(
            "Maximum number of Submitit array tasks allowed to run at once "
            f"(defaults to {DEFAULT_ARRAY_PARALLELISM}; set 0 to omit the cap)."
        ),
    )
    parser.add_argument(
        "--uv-environment",
        default=None,
        help="UV project environment path to sync and activate (defaults by --device).",
    )
    parser.add_argument(
        "--uv-extra",
        action="append",
        dest="uv_extras",
        default=None,
        help="UV extra passed to uv sync; repeat for multiple extras (defaults by --device).",
    )
    # Backward-compatible aliases for the first CUDA-only train launcher.
    parser.add_argument("--gpu-uv-environment", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu-extra", action="append", dest="gpu_extras", default=None, help=argparse.SUPPRESS)


def reject_deprecated_smoke(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fail fast for retired special smoke launcher mode."""

    if getattr(args, "smoke", False):
        parser.error(DEPRECATED_SMOKE_MESSAGE)
