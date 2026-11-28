"""Render guarded Stage-1 sbatch scripts; submission is always explicit."""

from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

# Direct script execution puts only this directory on sys.path.  Add the
# checkout root before importing the source-owned experiments package.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments.atomistic.stage1.protocol import (
    _require_path_component,
    _require_sha,
    _validate_run_identity,
    identity_expectations,
)
ALLOWED_PARTITIONS = ("test", "gpu_test", "seas_gpu", "kozinsky_gpu", "gpu")
GPU_PARTITIONS = frozenset(ALLOWED_PARTITIONS[1:])
_MAX_CPUS = 256
_MAX_MEMORY_GIB = 2048
_MAX_TIME_SECONDS = 7 * 24 * 60 * 60
_DEFAULT_GPU_CPUS = 10
_DEFAULT_GPU_MEMORY = "40G"
FACILITY_RESULTS_PREFIX = "/n/netscratch/kozinsky_lab/Lab/rhu/"
_SAFE_RENDERED_VALUE = re.compile(r"[A-Za-z0-9_./:+,@=\[\]-]+\Z")


def _require_safe_value(value: object, *, field: str) -> str:
    """Reject values that could alter the generated shell program."""

    if not isinstance(value, str) or not _SAFE_RENDERED_VALUE.fullmatch(value):
        raise ValueError(f"{field} contains shell-unsafe characters")
    return value


def _binding_token(row: Mapping[str, Any]) -> str:
    """Encode the complete row binding using shell-safe alphabet only."""

    payload = json.dumps(dict(row), sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def row_overrides(row: Mapping[str, Any]) -> list[str]:
    """Return explicit Hydra overrides for every launch row kind."""

    kind = row.get("kind")
    if kind not in {"smoke", "train", "eval"}:
        raise ValueError("manifest row has an invalid kind")
    overrides = [
        f"runtime.seed={row['seed']}",
        f"run.root={row['results_root']}",
        f"experiment.name={row['experiment_name']}",
        f"experiment.sector={row['sector']}",
        f"run.run_id={row['run_id']}",
        "run.layout=nested",
    ]
    if "device" in row:
        device = row["device"]
        if device not in {"cpu", "cuda"}:
            raise ValueError("manifest row device must be 'cpu' or 'cuda'")
        overrides.append(f"runtime.device={device}")
    if kind == "eval":
        overrides.extend([
            f"evaluation_sampler.n_walkers={row['walkers']}",
            f"evaluation_tasks.mcmc_energy.generator.n_draws={row['draws']}",
            f"load.path={row['checkpoint_dir']}",
            f"trajectory_identity.checkpoint_file={row['checkpoint_file']}",
            "load.mode=model_only",
            "load.strict=true",
        ])
    else:
        overrides.extend([
            f"sampler.n_walkers={row['walkers']}",
            f"trainer.max_steps={row['updates']}",
        ])
        if "sampler_seed" in row:
            overrides.append(f"sampler.seed={row['sampler_seed']}")
        if row.get("checkpoint_steps"):
            steps = ",".join(str(step) for step in row["checkpoint_steps"])
            overrides.append(f"checkpoint_steps=[{steps}]")
    return overrides


def _partition_names(partition: str) -> list[str]:
    """Validate and split the explicit Slurm partition request."""

    if not isinstance(partition, str) or not partition:
        raise ValueError("partition must be a non-empty comma-separated allow-list")
    names = partition.split(",")
    if any(not name or name not in ALLOWED_PARTITIONS for name in names):
        raise ValueError(f"partition must be a comma-separated subset of {ALLOWED_PARTITIONS}")
    if len(set(names)) != len(names):
        raise ValueError("partition must not contain duplicates")
    return names


def _validate_gpu_resources(*, cpus_per_task: int | None, mem: str | None,
                            time: str | None) -> tuple[int, str, str]:
    """Validate GPU resources, applying the measured Stage-1 defaults.

    The GPU default is ten CPUs and 40 GiB, matching the measured Polaris
    wrapper's ten-thread CPU shape and the Stage-1 Cannon acceptance contract.
    The memory request intentionally follows that ten-thread shape with ample
    headroom; callers may still provide a larger measured shape. CPU-only
    renders do not call this helper and retain the template's existing shape.
    """

    if cpus_per_task is None:
        cpus_per_task = _DEFAULT_GPU_CPUS
    if mem is None:
        mem = _DEFAULT_GPU_MEMORY

    if not isinstance(cpus_per_task, int) or not 1 <= cpus_per_task <= _MAX_CPUS:
        raise ValueError(f"cpus_per_task must be between 1 and {_MAX_CPUS}")
    if not isinstance(mem, str) or not re.fullmatch(r"[1-9][0-9]{0,3}G", mem):
        raise ValueError("mem must be a positive GiB value such as 64G")
    if int(mem[:-1]) > _MAX_MEMORY_GIB:
        raise ValueError(f"mem must be at most {_MAX_MEMORY_GIB}G")
    match = re.fullmatch(r"([0-9]{2,3}):([0-5][0-9]):([0-5][0-9])", str(time))
    if match is None:
        raise ValueError("time must use HH:MM:SS")
    hours, minutes, seconds = (int(value) for value in match.groups())
    total = hours * 3600 + minutes * 60 + seconds
    if total < 1 or total > _MAX_TIME_SECONDS:
        raise ValueError("time must be between 00:00:01 and 168:00:00")
    return cpus_per_task, mem, str(time)


def _gpu_guard(python: str) -> str:
    """Return the in-job A100/non-MIG admission guard."""

    return '''
if ! GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)"; then
  echo "STAGE1_GUARD nvidia-smi management query failed" >&2
  exit 2
fi
printf 'STAGE1_GUARD nvidia-smi=%s\\n' "$GPU_INFO"
GPU_COUNT="$(printf '%s\\n' "$GPU_INFO" | awk 'NF { count++ } END { print count + 0 }')"
if [[ "$GPU_COUNT" != "1" ]]; then
  echo "STAGE1_GUARD expected exactly one management GPU, got $GPU_COUNT" >&2
  exit 2
fi
GPU_NAME="${GPU_INFO%%,*}"
GPU_MEMORY="${GPU_INFO##*,}"
if [[ "$GPU_NAME" != *A100* || "$GPU_NAME" == *MIG* || "$GPU_NAME" =~ [0-9]+g\\.[0-9]+gb ]]; then
  echo "STAGE1_GUARD expected a non-MIG A100, got $GPU_NAME" >&2
  exit 2
fi
GPU_MEMORY="$(printf '%s' "$GPU_MEMORY" | tr -d '[:space:]')"
if [[ "$GPU_MEMORY" != "81920" ]]; then
  echo "STAGE1_GUARD expected 81920 MiB, got $GPU_MEMORY" >&2
  exit 2
fi
if ! MIG_MODE="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader)"; then
  echo "STAGE1_GUARD nvidia-smi MIG-mode query failed" >&2
  exit 2
fi
MIG_MODE="$(printf '%s' "$MIG_MODE" | tr -d '[:space:]')"
if [[ "$MIG_MODE" != "Disabled" ]]; then
  echo "STAGE1_GUARD expected MIG mode Disabled, got $MIG_MODE" >&2
  exit 2
fi
if ! CUDA_DEVICE_COUNT="$(__GPU_PYTHON__ -c 'import torch; print(torch.cuda.device_count())')"; then
  echo "STAGE1_GUARD CUDA device-count probe failed" >&2
  exit 2
fi
if [[ ! "$CUDA_DEVICE_COUNT" =~ ^[0-9]+$ || "$CUDA_DEVICE_COUNT" != "1" ]]; then
  echo "STAGE1_GUARD expected exactly one CUDA device, got $CUDA_DEVICE_COUNT" >&2
  exit 2
fi
'''.replace("__GPU_PYTHON__", shlex.quote(python))


_DRIVER_BY_KIND = {
    "smoke": "experiments/atomistic/stage1/train.py",
    "train": "experiments/atomistic/stage1/train.py",
    "eval": "experiments/atomistic/stage1/eval.py",
}


def command_for_row(
    row: Mapping[str, Any], *, python: str, checkout_root: str | None = None,
    driver: str | None = None,
) -> str:
    """Build the single-binding driver command permitted by the launcher."""

    if row.get("kind") not in {"smoke", "train", "eval"}:
        raise ValueError("manifest row has an invalid kind")
    _require_sha(row.get("source_sha"), length=40, field="source_sha")
    _require_sha(row.get("config_sha256"), length=64, field="config_sha256")
    if checkout_root is not None:
        driver = _resolve_driver_path(_DRIVER_BY_KIND[row["kind"]], checkout_root)
    if driver is None:
        raise ValueError("checkout_root is required to derive the Stage-1 driver")
    values = {"python": python, "driver": driver, "config": row.get("config"),
              "row_id": row.get("row_id", "")}
    for field, value in values.items():
        _require_safe_value(value, field=field)
    args = [python, driver, "--binding", _binding_token(row)]
    return shlex.join(args)


def _resolve_driver_path(driver: str, checkout_root: str) -> str:
    """Resolve a repository-relative driver from the selected execution checkout."""

    path = Path(driver)
    if path.is_absolute():
        return driver
    return str((Path(checkout_root) / path).resolve())


def _train_outcome_handler(*, command: str, python: str, status_path: Path) -> str:
    """Capture the train exit code and admit only a recorded planned yield."""

    quoted_python = shlex.quote(python)
    quoted_status_path = shlex.quote(str(status_path))
    return (
        "if " + command + "; then\n"
        "  TRAIN_RC=0\n"
        "else\n"
        "  TRAIN_RC=$?\n"
        "fi\n"
        "if [[ \"$TRAIN_RC\" == \"0\" ]]; then\n"
        "  printf 'STAGE1_GUARD train_rc=%s status=not-applicable decision=continue\\n' \"$TRAIN_RC\"\n"
        "elif [[ \"$TRAIN_RC\" == \"2\" ]]; then\n"
        f"  STATUS_PATH={quoted_status_path}\n"
        "  if ! STATUS=$("
        f"{quoted_python} -c 'import json,sys; print(json.load(open(sys.argv[1], encoding=\"utf-8\")).get(\"status\", \"\"))' \"$STATUS_PATH\""
        "); then\n"
        "    echo \"STAGE1_GUARD train_rc=2 status.json unreadable path=$STATUS_PATH decision=fail\" >&2\n"
        "    exit \"$TRAIN_RC\"\n"
        "  fi\n"
        "  if [[ \"$STATUS\" != \"planned_yield\" ]]; then\n"
        "    echo \"STAGE1_GUARD train_rc=2 status=$STATUS expected=planned_yield decision=fail\" >&2\n"
        "    exit \"$TRAIN_RC\"\n"
        "  fi\n"
        "  printf 'STAGE1_GUARD train_rc=%s status=%s decision=continue\\n' \"$TRAIN_RC\" \"$STATUS\"\n"
        "else\n"
        "  echo \"STAGE1_GUARD train_rc=$TRAIN_RC decision=fail\" >&2\n"
        "  exit \"$TRAIN_RC\"\n"
        "fi\n"
    )


def render(*, row: Mapping[str, Any], checkout_root: str, python: str,
           partition: str = "test",
           scratch_base: str = "/scratch", cpus_per_task: int | None = None,
           mem: str | None = None, time: str | None = None,
           # Compatibility-only parameters for callers pinned to LR-1.  They
           # are checked against the row and never become application inputs.
           driver: str | None = None, head_sha: str | None = None,
           results_root: str | None = None) -> str:
    """Render a fail-closed script for one identity-bound row.

    ``checkout_root`` is the exact checkout staged for execution. It is
    explicit so rendering from one checkout cannot launch a different tree.
    """

    if not Path(python).is_absolute():
        raise ValueError("python must be an absolute path")
    _require_safe_value(python, field="python")
    if not Path(checkout_root).is_absolute() or checkout_root == "/":
        raise ValueError("checkout_root must be absolute and not '/'")
    _require_safe_value(checkout_root, field="checkout_root")
    _require_safe_value(scratch_base, field="scratch_base")
    if not Path(scratch_base).is_absolute() or scratch_base == "/":
        raise ValueError("scratch_base must be absolute and not '/'")
    _require_sha(row.get("source_sha"), length=40, field="row.source_sha")
    if head_sha is not None:
        _require_sha(head_sha, length=40, field="head_sha")
        if head_sha != row.get("source_sha"):
            raise ValueError("rendered head_sha does not match row source_sha")
    partitions = _partition_names(partition)
    is_gpu = bool(GPU_PARTITIONS.intersection(partitions))
    if is_gpu:
        cpus_per_task, mem, time = _validate_gpu_resources(
            cpus_per_task=cpus_per_task, mem=mem, time=time,
        )
        if row.get("device") != "cuda":
            raise ValueError("GPU partitions require a row device binding of cuda")
    elif row.get("device") not in (None, "cpu"):
        raise ValueError("CPU partitions require a row device binding of cpu")
    row_results_root = row.get("results_root")
    if not isinstance(row_results_root, str):
        raise ValueError("row results_root is required")
    if results_root is not None and results_root.rstrip("/") != row_results_root.rstrip("/"):
        raise ValueError("rendered results_root does not match row results_root")
    results_root = row_results_root
    if not results_root.startswith(FACILITY_RESULTS_PREFIX):
        raise ValueError("results_root must be below the approved Netscratch prefix")
    _require_safe_value(results_root, field="results_root")
    row_root = row.get("results_root")
    approved_root = results_root.rstrip("/")
    if (not isinstance(row_root, str)
            or (row_root != approved_root and not row_root.startswith(approved_root + "/"))):
        raise ValueError("row results_root is not bound to the rendered root")
    _require_safe_value(row_root, field="row.results_root")
    _validate_run_identity(row)
    if row.get("kind") == "eval":
        identity_expectations(row)
    for field in ("experiment_name", "sector", "run_id"):
        _require_path_component(row.get(field), field=field)
    command = command_for_row(row, python=python, checkout_root=checkout_root)
    status_path = (
        Path(results_root) / str(row["experiment_name"]) / str(row["sector"])
        / str(row["run_id"]) / "status.json"
    )
    from experiments.toolkit.job_names import stage1_row_job_name

    job_name = stage1_row_job_name(row)
    template_path = Path(__file__).with_name("templates") / "cannon.sbatch"
    template = template_path.read_text(encoding="utf-8")
    if is_gpu:
        template = template.replace(
            "#SBATCH --cpus-per-task=1\n#SBATCH --no-requeue",
            f"#SBATCH --cpus-per-task={cpus_per_task}\n"
            f"#SBATCH --mem={mem}\n#SBATCH --time={time}\n"
            "#SBATCH --gres=gpu:1\n#SBATCH --constraint=a100\n#SBATCH --no-requeue",
        )
        template = template.replace(
            "__GPU_THREAD_EXPORTS__",
            "\n".join([
                f"export OMP_NUM_THREADS={cpus_per_task}",
                f"export MKL_NUM_THREADS={cpus_per_task}",
            ]),
        )
        template = template.replace(
            'REQUESTED_PARTITION="__PARTITION__"\n'
            'if [[ "$SLURM_JOB_PARTITION" != "$REQUESTED_PARTITION" ]]; then\n'
            '  echo "STAGE1_GUARD partition delivered=$SLURM_JOB_PARTITION requested=$REQUESTED_PARTITION" >&2\n'
            '  exit 2\nfi\n',
            'REQUESTED_PARTITION="__PARTITION__"\n'
            'IFS="," read -r -a REQUESTED_PARTITIONS <<< "$REQUESTED_PARTITION"\n'
            'PARTITION_MATCH=0\n'
            'for requested in "${REQUESTED_PARTITIONS[@]}"; do\n'
            '  [[ "$SLURM_JOB_PARTITION" == "$requested" ]] && PARTITION_MATCH=1\n'
            'done\n'
            'if [[ "$PARTITION_MATCH" != "1" ]]; then\n'
            '  echo "STAGE1_GUARD partition delivered=$SLURM_JOB_PARTITION requested=$REQUESTED_PARTITION" >&2\n'
            '  exit 2\nfi\n',
        )
        template = template.replace('SCRATCH_BASE=', _gpu_guard(python) + 'SCRATCH_BASE=', 1)
    template = template.replace("__GPU_THREAD_EXPORTS__", "")
    for marker, value in {
        "__PARTITION__": partition,
        "__RESULTS_ROOT__": results_root,
        "__HEAD_SHA__": str(row["source_sha"]),
        "__JOB_NAME__": job_name,
        "__SCRATCH_BASE__": scratch_base.rstrip("/"),
        "__COMMAND__": command,
    }.items():
        template = template.replace(marker, value)
    if row.get("kind") in {"smoke", "train"}:
        return template.replace(command, _train_outcome_handler(
            command=command, python=python, status_path=status_path,
        ), 1)
    return template


def main() -> int:
    """Write a script without submitting it."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=Path, required=True)
    parser.add_argument("--checkout-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--partition", default="test")
    parser.add_argument("--scratch-base", default="/scratch")
    parser.add_argument("--cpus-per-task", type=int)
    parser.add_argument("--mem")
    parser.add_argument("--time")
    args = parser.parse_args()
    row = json.loads(args.row.read_text(encoding="utf-8"))
    args.output.write_text(render(row=row, python=args.python,
                                  checkout_root=args.checkout_root,
                                  partition=args.partition, scratch_base=args.scratch_base,
                                  cpus_per_task=args.cpus_per_task, mem=args.mem,
                                  time=args.time), encoding="utf-8")
    args.output.chmod(0o750)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
