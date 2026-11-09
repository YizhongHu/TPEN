#!/usr/bin/env python3
"""H-v2 in-job attached route (design target sections A4/B3, item 9d3aa0a3).

Invoked directly by Slurm as the launch script's job program (``launch.py``,
item 67f2eb57): ``sbatch ... run_job.py train L1`` or
``... run_job.py eval L1``. Runs entirely inside its own allocation; the
launch script only submits it.

Required environment (inherited from the submitting shell, per section B3):

``RUN``
    Durable run root; ``$RUN/checkout`` is the checkout this job runs from.
``SHA``
    Expected checkout commit, asserted against ``git -C "$RUN/checkout"
    rev-parse HEAD`` before any payload.
``SCIENCE_PYTHON``
    Absolute science environment Python (CUDA asserted against it).
``EXECUTOR_PY``
    Absolute executor environment Python (``cd executor && uv sync --locked``,
    separate from the science environment).

NOT YET COMPLETE. Section C2 schedules 27 checkpoints x 4 streams (108
evaluations) per lineage EVAL job, run one after another inside the same
allocation. This script invokes exactly one EVAL generation (stream 0
against the lineage's single recorded TRAIN checkpoint): ``attached.py``'s
``prepare_h_v2_run`` admits only one task per generation, and no
per-checkpoint/per-stream loop has been built or reviewed yet. That fan-out
is a follow-up, not invented here. EVAL also currently refuses
unconditionally before dispatch (``attached.EVAL_ROUTE_UNSUPPORTED_CODE``,
pending an operator decision on item 9d3aa0a3) -- this script reaches that
refusal rather than working around it, per the layer-9 acceptance contract.

This module imports no ``tpen``, no ``hydra``, and no ``tpen_executor``
(``experiments/README.md``); it talks to the executor only through
``attached.py``'s CLI subprocess route.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _h_v2_module(name: str):
    """Load an h-v2 sibling under its own ``h_v2_<name>`` :mod:`sys.modules` key.

    Mirrors ``attached.py``'s own private loader (see that module's
    docstring for why a bare ``import`` is unsafe here).
    """

    key = f"h_v2_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


attached = _h_v2_module("attached")


class JobError(RuntimeError):
    """The in-job route cannot proceed (environment, identity, or usage)."""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise JobError(f"required environment variable {name!r} is unset")
    return value


def check_identity(run_root: Path, expected_commit: str, science_python: str) -> None:
    """Section B3's in-job identity block: print, then assert, before any payload."""

    job_id = os.environ.get("SLURM_JOB_ID", "absent")
    partition = os.environ.get("SLURM_JOB_PARTITION", "absent")
    hostname = subprocess.run(["hostname"], check=False, capture_output=True, text=True).stdout.strip()
    print(f"[h-v2] job={job_id} partition={partition} host={hostname or 'absent'}")
    subprocess.run(["nvidia-smi", "-L"], check=False)

    checkout = run_root / "checkout"
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    if head.returncode != 0 or head.stdout.strip() != expected_commit:
        raise JobError(
            f"checkout HEAD {head.stdout.strip()!r} at {checkout} does not match expected commit "
            f"{expected_commit!r}"
        )

    torch_check = subprocess.run(
        [
            science_python,
            "-c",
            "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if torch_check.returncode != 0:
        raise JobError(f"CUDA torch check failed: {torch_check.stderr.strip()}")
    print(f"[h-v2] {torch_check.stdout.strip()}")


def write_local_allocation_and_worker(
    alloc_dir: Path, *, science_python: str, placeholder_config: str, backend_job_id: str
) -> tuple[Path, Path]:
    """Write minimal real allocation/worker fixtures for ``--dispatcher local``.

    Mirrors ``test_attached.py``'s ``_real_cli_allocation_and_worker``: one
    GPU, one task, one node needs no multi-slot binding
    (``experiments/toolkit/attached_activation.py``'s
    ``bind_allocation_context`` exists for that case, not this one).
    ``runner.config`` is a placeholder: the executor overrides it per task
    from the configured-runs sidecar ``prepare_h_v2_run`` already wrote.
    """

    alloc_dir.mkdir(parents=True, exist_ok=True)
    allocation_path = alloc_dir / "allocation.json"
    allocation_path.write_text(
        json.dumps(
            {
                "schema": "tpen-executor/configured-allocation-v1",
                "allocation_id": f"h-v2-{backend_job_id}",
                "backend_job_id": backend_job_id,
                "facts": {},
            }
        ),
        encoding="utf-8",
    )
    worker_path = alloc_dir / "worker.yaml"
    worker_path.write_text(
        "worker:\n"
        "  _target_: tpen_executor.programs.WorkerProgram\n"
        "  _recursive_: false\n"
        "  environment:\n"
        "    _target_: tpen_executor.environments.PreparedPythonEnvironment\n"
        "    _recursive_: false\n"
        f"    python: {json.dumps(science_python)}\n"
        "    env: {}\n"
        "  process:\n"
        "    _target_: tpen_executor.workers.SubprocessWorker\n"
        "    _recursive_: false\n"
        "  runner:\n"
        "    _target_: tpen_executor.runners.TPENCommand\n"
        "    _recursive_: false\n"
        f"    config: {json.dumps(placeholder_config)}\n"
        "  evidence:\n"
        "    _target_: tpen_executor.runners.TPENEvidence\n"
        "    _recursive_: false\n",
        encoding="utf-8",
    )
    return allocation_path, worker_path


def checkpoint_dir_for_eval(run_root: Path, lineage: str) -> Path:
    """Read the lineage's TRAIN checkpoint dir from its accepted receipt.

    Raises
    ------
    JobError
        If the TRAIN accepted receipt is missing or carries no segments --
        this never invents a checkpoint path.
    """

    receipt_path = run_root / "attached" / "receipts" / f"h-v2-train-{lineage}.h-v2.json"
    if not receipt_path.is_file():
        raise JobError(f"no TRAIN accepted receipt at {receipt_path}; TRAIN must complete before EVAL")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    if not segments:
        raise JobError(f"TRAIN accepted receipt {receipt_path} carries no segments")
    return Path(segments[0]["checkpoint_dir"])


def run(stage: str, lineage: str) -> None:
    """Build, prepare, and activate one H-v2 attached generation."""

    run_root = Path(_require_env("RUN")).resolve()
    expected_commit = _require_env("SHA")
    science_python = _require_env("SCIENCE_PYTHON")
    executor_python = _require_env("EXECUTOR_PY")

    check_identity(run_root, expected_commit, science_python)

    if stage == "train":
        task_id = f"h-v2-train-{lineage}"
    elif stage == "eval":
        task_id = f"h-v2-eval-{lineage}-stream0"
    else:
        raise JobError(f"stage must be 'train' or 'eval', got {stage!r}")

    handoff_dir = run_root / "attached" / "handoff"
    results_root = run_root / "attached" / "results"
    store = run_root / "attached" / "store" / task_id
    receipts_dir = run_root / "attached" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)

    if stage == "train":
        plan, overrides, config = attached.build_h_v2_train_plan(
            output_dir=handoff_dir, results_root=results_root, lineage=lineage
        )
    else:
        checkpoint_dir = checkpoint_dir_for_eval(run_root, lineage)
        plan, overrides, config = attached.build_h_v2_eval_plan(
            output_dir=handoff_dir,
            results_root=results_root,
            lineage=lineage,
            checkpoint=checkpoint_dir,
            stream=0,
        )

    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=handoff_dir)
    alloc_dir = run_root / "attached" / "alloc" / task_id
    allocation_path, worker_path = write_local_allocation_and_worker(
        alloc_dir,
        science_python=science_python,
        placeholder_config=prepared.effective_config,
        backend_job_id=os.environ.get("SLURM_JOB_ID", "unknown"),
    )
    attached.activate_h_v2_run(
        prepared,
        science_python=science_python,
        executor_python=executor_python,
        allocation=allocation_path,
        worker_config=worker_path,
        store=store,
        activation_receipt_path=receipts_dir / f"{task_id}.activation.json",
        h_v2_receipt_path=receipts_dir / f"{task_id}.h-v2.json",
        expected_source_commit=expected_commit,
        dispatcher="local",
        cwd=run_root / "checkout",
    )
    print(f"[h-v2] {task_id}: accepted")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: run_job.py {train|eval} <lineage>", file=sys.stderr)
        return 2
    stage, lineage = argv
    try:
        run(stage, lineage)
    except Exception as error:  # noqa: BLE001 - this is the job's own top-level boundary
        # afterok sees only this exit code: any refusal here must be non-zero,
        # never a bare 0 (executor CLI exit 0 alone is not science complete).
        print(f"[h-v2] job failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
