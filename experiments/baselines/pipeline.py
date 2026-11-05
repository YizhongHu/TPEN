"""Dispatch a baselines :class:`StagePlanV2` inside one allocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

from experiments.toolkit.dispatch import (
    AllocationContext,
    DispatchSpec,
    LogicalTaskSpec,
    StagePlanV2,
    admit_tasks,
    write_dispatch_records,
)
from experiments.toolkit.executor_inputs import ready_group, write_ready_group
from experiments.toolkit.parsl_attach import ParslAttachExecutor


#: Accelerators a node exposes. A SETTING: our jobs are planned against a known
#: node shape, so this is declared rather than discovered at runtime. Polaris is
#: 4 GPUs per node; override for a differently shaped facility.
ACCELERATORS_PER_NODE = 4


def accelerator_bindings(
    gpus_per_row: int, accelerators_per_node: int = ACCELERATORS_PER_NODE
) -> tuple[str, ...]:
    """Tile a node's accelerators into one visibility binding per row.

    Admissible row widths are the divisors of the node's accelerator count;
    anything else strands accelerators or straddles a node boundary. Stating that
    as divisibility keeps it correct for a node size other than four.

    ``1 -> ("0", "1", "2", "3")``, ``2 -> ("0,1", "2,3")``, ``4 -> ("0,1,2,3",)``.
    """

    if gpus_per_row < 1:
        raise ValueError(f"gpus_per_row must be positive; got {gpus_per_row}")
    if accelerators_per_node % gpus_per_row:
        raise ValueError(
            f"gpus_per_row {gpus_per_row} does not divide the node's "
            f"{accelerators_per_node} accelerators, so rows cannot tile it"
        )
    return tuple(
        ",".join(str(index) for index in range(start, start + gpus_per_row))
        for start in range(0, accelerators_per_node, gpus_per_row)
    )


def allocation_context(
    *,
    facility: str,
    gpus_per_row: int,
    run_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> AllocationContext:
    """Build an explicit accelerator context from scheduler variables."""

    environ = os.environ if environ is None else environ
    allocation_id = environ.get("PBS_JOBID") or environ.get("SLURM_JOB_ID")
    if not allocation_id:
        raise RuntimeError("a PBS_JOBID or SLURM_JOB_ID allocation is required")
    raw_nodes = str(environ.get("TPEN_NODES_PER_BLOCK", "")).strip()
    try:
        nodes = int(raw_nodes) if raw_nodes else None
    except ValueError as exc:
        raise ValueError("TPEN_NODES_PER_BLOCK must be a positive integer") from exc
    if nodes is not None and nodes <= 0:
        raise ValueError("TPEN_NODES_PER_BLOCK must be a positive integer")
    if facility == "cannon":
        raise ValueError("ParslAttachExecutor baselines dispatch is supported only on Polaris")
    values = accelerator_bindings(gpus_per_row)
    return AllocationContext(
        allocation_id=str(allocation_id),
        visibility_variable="CUDA_VISIBLE_DEVICES",
        visibility_values=values,
        run_root=str(run_root),
        environment={},
        nodes_per_block=nodes if nodes and nodes > 1 else None,
    ).validate()


def _argv(task: LogicalTaskSpec, python: str) -> tuple[str, ...]:
    """Resolve the plan's executable placeholder without shell parsing."""

    return tuple(python if token == "{python}" else token for token in task.command)


def _uniform_gpu_count(plan: StagePlanV2) -> int:
    """Return the one GPU count Parsl can apply to every row in an allocation."""

    counts = {task.resources.gpus for task in plan.tasks}
    if len(counts) != 1:
        raise ValueError(
            "ParslAttachExecutor requires uniform resources.gpus per plan; "
            f"found {sorted(count for count in counts if count is not None)}"
        )
    count = next(iter(counts))
    if count is None or count <= 0:
        raise ValueError("every baselines task must declare a positive resources.gpus")
    return count


def executor_input(plan: StagePlanV2) -> dict[str, object]:
    """Emit the executor input mapping while retaining the legacy route."""

    return ready_group(plan)


def run_pipeline(
    *,
    plan: StagePlanV2,
    facility: str,
    launch_dir: str | Path,
    admission_id: str,
    executor: ParslAttachExecutor | None = None,
    environ: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    python: str | None = None,
) -> int:
    """Read one plan, dispatch its rows concurrently, and write records."""

    launch = Path(launch_dir)
    launch.mkdir(parents=True, exist_ok=True)
    plan.validate()
    write_ready_group(launch / "executor_input.json", plan)
    gpus_per_row = _uniform_gpu_count(plan)
    context = allocation_context(
        facility=facility,
        gpus_per_row=gpus_per_row,
        run_root=launch,
        environ=environ,
    )
    executable = str(python or sys.executable)
    runtimes = {str(task.metadata["runtime"]) for task in plan.tasks}
    if len(runtimes) != 1:
        raise ValueError("one StagePlanV2 must align to one runtime")
    runtime = next(iter(runtimes))
    admitted = admit_tasks(
        plan,
        plan.tasks,
        admission_id=admission_id,
        argv_by_runtime={runtime: [_argv(task, executable) for task in plan.tasks]},
        cwd=str(cwd or Path.cwd()),
        environment={},
        admission_ref=plan.plan_id,
    )
    records = []
    verification = {"schema": "baselines-parsl-verification/v1", "complete": False, "exit_code": 1}
    active_executor = executor or ParslAttachExecutor()
    try:
        records.extend(active_executor.dispatch(admitted, context=context))
        verification.update(complete=True, exit_code=0)
        return 0
    except Exception as exc:
        verification["error"] = repr(exc)
        return 1
    finally:
        if isinstance(active_executor, ParslAttachExecutor):
            active_executor.close()
        write_dispatch_records(launch, records)
        (launch / "verification.json").write_text(
            json.dumps(verification, indent=2) + "\n", encoding="utf-8"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--facility", choices=("cannon", "polaris"), required=True)
    parser.add_argument("--launch-dir", required=True)
    parser.add_argument("--admission-id", required=True)
    args = parser.parse_args(argv)
    return run_pipeline(
        plan=StagePlanV2.read(args.plan_dir),
        facility=args.facility,
        launch_dir=args.launch_dir,
        admission_id=args.admission_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
