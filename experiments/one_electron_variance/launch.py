"""Run one admitted lineage, or an excluded-seed pilot, inside an allocation.

No scheduler submission, retry, cancellation, environment installation, or
cleanup occurs here. Toolkit admission owns attempt identity; each configured
run executes in a fresh child process. Exit and scientific completeness remain
separate receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from experiments.one_electron_variance.admission import validate_science_admission

from experiments.one_electron_variance.protocol import (
    DESIGN_SHA256, evaluation_config, run_directory, train_config,
)
from experiments.toolkit.dispatch import (
    DispatchRecord, LogicalTaskSpec, StagePlanV2, admit_tasks, write_dispatch_records,
)
from experiments.toolkit.resources import ResourceSpec


def _write_json(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _source_identity(expected: str) -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if head != expected or dirty:
        raise ValueError("launch requires the exact clean source recorded at admission")
    return {"source_sha": head, "tracked_tree_clean": True}


def _run_config(cfg: dict, *, directory: Path, logical_id: str, admission_id: str, source_sha: str,
                planned_task: LogicalTaskSpec | None = None, planned_plan: StagePlanV2 | None = None) -> float:
    output = run_directory(cfg)
    if output.exists():
        raise FileExistsError(f"refusing a second attempt into {output}")
    directory.mkdir(parents=True, exist_ok=False)
    if planned_task is None:
        path = directory / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        argv = (sys.executable, "-u", "run.py", "--config", str(path))
        device = cfg["runtime"]["device"]
        task = LogicalTaskSpec(logical_task_id=logical_id, stage="configured_run", run_id=cfg["run"]["run_id"],
                               command=argv, result_dir=str(output),
                               resources=ResourceSpec(profile=device, device=device,
                                                      gpus=0 if device == "cpu" else 1,
                                                      threads=4, mem_gb=32))
        plan = StagePlanV2(study="one_electron_variance", stage="configured_run", plan_id=DESIGN_SHA256,
                           results_root=cfg["run"]["root"], tasks=(task,))
    else:
        if planned_plan is None:
            raise ValueError("a planned task requires its original plan")
        task, plan = planned_task, planned_plan
        path = Path(task.inputs[0])
        argv = tuple(sys.executable if token == "{python}" else token for token in task.command)
    dispatch, = admit_tasks(plan, (task,), admission_id=admission_id,
                            argv_by_runtime={"default": (argv,)}, cwd=Path.cwd(), environment={})
    _write_json(directory / "dispatch.json", dispatch.to_dict())
    start = time.perf_counter()
    with (directory / "stdout.log").open("x") as stdout, (directory / "stderr.log").open("x") as stderr:
        # Popen's context waits even if receipt writing raises: no orphan child.
        with subprocess.Popen(dispatch.argv, cwd=dispatch.cwd, stdout=stdout, stderr=stderr,
                              env={**os.environ, **dispatch.environment}) as child:
            record = DispatchRecord.accepted(dispatch, backend="allocation_child",
                                              launcher_job_id=str(child.pid), submitted_command=dispatch.argv)
            write_dispatch_records(directory, (record,))
            code = child.wait()
    elapsed = time.perf_counter() - start
    _write_json(directory / "process_exit.json", {"exit_code": code, "elapsed_seconds": elapsed,
                                                 "source_sha": source_sha,
                                                 "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if code:
        raise RuntimeError(f"configured run failed with exit {code}; no retry authorized: {directory}")
    status = json.loads((output / "status.json").read_text())
    if status["status"] != "completed":
        raise RuntimeError("child exited without a completed run receipt")
    return elapsed


def run_pilot(root: Path, source_sha: str, admission_id: str) -> dict:
    """Measure both full-batch systems using excluded seeds and full evaluation streams."""
    rows = {}
    for index, system in enumerate(("hooke", "hydrogen")):
        cfg = train_config(system, 0, root)
        cfg["run"]["run_id"] = f"{system}-excluded-pilot"
        cfg["experiment"]["run_name"] = cfg["run"]["run_id"]
        cfg["runtime"]["seed"] = 937001 + index
        cfg["sampler"]["seed"] = 938001 + index
        cfg["trainer"]["max_steps"] = 1000
        train_seconds = _run_config(cfg, directory=root / "dispatch" / f"{system}-pilot-train",
                                    logical_id=f"pilot_train:{system}", admission_id=admission_id,
                                    source_sha=source_sha)
        evaluation = evaluation_config(system, 0, 2500, 0, root)
        evaluation["run"]["run_id"] = f"{system}-excluded-pilot-evaluate"
        evaluation["experiment"]["run_name"] = evaluation["run"]["run_id"]
        evaluation["load"]["path"] = str(run_directory(cfg) / "checkpoints" / "step_001000")
        evaluation["runtime"]["seed"] = evaluation["sampler"]["seed"] = 1937001 + index
        evaluation["evaluator"]["seed"] = 1937001 + index
        evaluation["evaluator"]["tasks"][0]["generator"]["seed"] = 1937001 + index
        eval_seconds = _run_config(evaluation, directory=root / "dispatch" / f"{system}-pilot-evaluate",
                                   logical_id=f"pilot_evaluate:{system}", admission_id=admission_id,
                                   source_sha=source_sha)
        # Deliberately conservative: charge process/setup and two checkpoint writes
        # in each 1000-update block, more frequently than science's 2500 cadence.
        seconds_per_update = train_seconds / 1000
        projection = 50000 * seconds_per_update + 18 * eval_seconds
        rows[system] = {"training_seconds": train_seconds, "seconds_per_update_upper_estimate": seconds_per_update,
                        "evaluation_stream_seconds": eval_seconds, "projected_lineage_seconds": projection,
                        "fits_six_hours_with_twenty_percent_margin": projection <= .8 * 6 * 3600}
    return {"source_sha": source_sha, "mode": "excluded_seed_full_batch_pilot", "systems": rows,
            "timing_admission": all(r["fits_six_hours_with_twenty_percent_margin"] for r in rows.values()),
            "memory_admission": "requires_review_of_resource_usage_and_delivered_device_receipts"}


def run_lineage(root: Path, source_sha: str, admission_id: str, *, system: str, lineage: int, gate: Path) -> None:
    """Execute exactly one released-source lineage after qualified timing/memory admission."""
    validate_science_admission(gate, source_sha)
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["source_sha"] != source_sha or protocol["design_sha256"] != DESIGN_SHA256:
        raise ValueError("source does not match the frozen protocol")
    for record in protocol["configs"]:
        if hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("a frozen study configuration changed")
    # Preserve the planner's opaque logical identities and prepared bytes.
    for stage in ("train", "evaluate"):
        plan = StagePlanV2.read(root / f"{stage}-plan")
        selected = [task for task in plan.tasks if task.params == {"system": system, "lineage": lineage}]
        if len(selected) != (1 if stage == "train" else 18):
            raise ValueError("prepared plan does not contain the complete lineage")
        for task in selected:
            cfg = yaml.safe_load(Path(task.inputs[0]).read_text())
            _run_config(cfg, directory=root / "dispatch" / task.run_id, logical_id=task.logical_task_id,
                        admission_id=admission_id, source_sha=source_sha, planned_task=task, planned_plan=plan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "lineage"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--admission-id", required=True)
    parser.add_argument("--system", choices=("hooke", "hydrogen"))
    parser.add_argument("--lineage", type=int, choices=range(3))
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("this launcher must execute inside a scheduler allocation")
    _source_identity(args.source_sha)
    if args.mode == "pilot":
        args.root.mkdir(parents=True, exist_ok=False)
        _write_json(args.root / "pilot.json", run_pilot(args.root, args.source_sha, args.admission_id))
    else:
        if args.system is None or args.lineage is None or args.gate is None:
            parser.error("lineage mode requires --system, --lineage and --gate")
        run_lineage(args.root, args.source_sha, args.admission_id, system=args.system,
                    lineage=args.lineage, gate=args.gate)
