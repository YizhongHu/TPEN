"""Execute the He-cutover smoke inside an existing scheduler allocation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

from experiments.toolkit.dispatch import AllocationContext, DispatchSpec, StagePlanV2, write_dispatch_records
from experiments.toolkit.parsl_attach import ParslAttachExecutor

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

_tpen_admission = sibling(__file__, 'admission')
admit_plan = _tpen_admission.admit_plan
write_dispatch_specs = _tpen_admission.write_dispatch_specs
hev1 = sibling(__file__, 'hev1')
_tpen_run_train_row = sibling(__file__, 'run_train_row')
require_allocation = _tpen_run_train_row.require_allocation


def allocation_context(*, facility: str, run_root: str | Path, environ: Mapping[str, str] | None = None, deadline: str | float | None = None) -> AllocationContext:
    """Build explicit binding data from the scheduler environment."""

    environ = os.environ if environ is None else environ
    allocation_id = require_allocation(environ)
    raw_nodes = str(environ.get("TPEN_NODES_PER_BLOCK", "")).strip()
    if raw_nodes:
        try:
            requested_nodes = int(raw_nodes)
        except ValueError as exc:
            raise ValueError("TPEN_NODES_PER_BLOCK must be a positive integer") from exc
        if requested_nodes <= 0:
            raise ValueError("TPEN_NODES_PER_BLOCK must be a positive integer")
        nodes_per_block = requested_nodes if requested_nodes > 1 else None
    else:
        nodes_per_block = None
    # Empty means true inheritance: Slurm owns the MIG visibility value and
    # neither the executor nor Parsl may replace it. Polaris PBS leaves the
    # variable unset, so its four workers receive explicit accelerator ids.
    values = () if facility == "cannon" else ("0", "1", "2", "3")
    return AllocationContext(allocation_id=allocation_id, visibility_variable="CUDA_VISIBLE_DEVICES", visibility_values=values, run_root=str(run_root), deadline=deadline, environment={}, nodes_per_block=nodes_per_block).validate()


def preflight_dispatch(*, context: AllocationContext, cwd: str | Path, admission_id: str, runtime: str, python: str) -> DispatchSpec:
    """Create the worker-side environment and torch.cuda probe."""

    code = "import os,sys; print(sys.executable); print(sys.version); assert sys.version_info >= (3,12); import torch; print(os.environ.get('CUDA_VISIBLE_DEVICES')); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
    from experiments.toolkit.specs import CompletionSpec
    return DispatchSpec("preflight", admission_id, f"{admission_id}:preflight", "01_preflight", "preflight", (python, "-c", code), str(context.run_root), runtime, str(cwd), completion=CompletionSpec(policy="none"))


def run_pipeline(*, train_plan: StagePlanV2, eval_plan: StagePlanV2, facility: str, launch_dir: str | Path, admission_id: str, executor=None, environ: Mapping[str, str] | None = None) -> int:
    """Run probe, training, completion barrier, then evaluation; exit code is truth."""

    launch = Path(launch_dir)
    launch.mkdir(parents=True, exist_ok=True)
    context = allocation_context(facility=facility, run_root=launch, environ=environ)
    executor = executor or ParslAttachExecutor()
    records = []
    verification = {"schema": "he-cutover-verification/v1", "complete": False, "exit_code": 1, "stages": []}
    try:
        runtime = str(train_plan.tasks[0].metadata["runtime"])
        python = sys.executable
        records.extend(executor.dispatch((preflight_dispatch(context=context, cwd=Path.cwd(), admission_id=admission_id, runtime=runtime, python=python),), context=context))
        train = admit_plan(train_plan, admission_id=admission_id, cwd=Path.cwd(), environment={}, python=python)
        write_dispatch_specs(launch / "02_train" / "dispatch_specs.jsonl", train)
        records.extend(executor.dispatch(train, context=context))
        if not all(task.completion.is_complete() for task in train_plan.tasks):
            raise RuntimeError("training completion barrier failed")
        for task in eval_plan.tasks:
            hev1.eval_stage.require_complete_checkpoint(task.params["checkpoint_dir"])
        evaluation = admit_plan(eval_plan, admission_id=admission_id, cwd=Path.cwd(), environment={}, python=python)
        write_dispatch_specs(launch / "03_eval" / "dispatch_specs.jsonl", evaluation)
        records.extend(executor.dispatch(evaluation, context=context))
        verification.update(complete=True, exit_code=0, stages=["01_preflight", "02_train", "03_eval"])
        return 0
    except Exception as exc:
        verification["error"] = repr(exc)
        return 1
    finally:
        if isinstance(executor, ParslAttachExecutor):
            executor.close()
        write_dispatch_records(launch, records)
        (launch / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--facility", choices=("cannon", "polaris"), required=True)
    parser.add_argument("--launch-dir", required=True)
    parser.add_argument("--admission-id", required=True)
    args = parser.parse_args(argv)
    return run_pipeline(train_plan=StagePlanV2.read(Path(args.plan_dir) / "02_train"), eval_plan=StagePlanV2.read(Path(args.plan_dir) / "03_eval"), facility=args.facility, launch_dir=args.launch_dir, admission_id=args.admission_id)


if __name__ == "__main__":
    raise SystemExit(main())
