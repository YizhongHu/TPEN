"""Cluster-free tests for the baselines Parsl dispatch seam."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from experiments.baselines import pipeline
from experiments.baselines.plan import build_plan
from experiments.baselines.pipeline import accelerator_bindings, allocation_context
from experiments.toolkit.parsl_attach import validate_accelerator_tiling
from experiments.toolkit.dispatch import DispatchRecord, LogicalTaskSpec, StagePlanV2
from experiments.toolkit.resources import ResourceSpec
from experiments.toolkit.specs import CompletionSpec


class RecordingExecutor:
    def __init__(self) -> None:
        self.dispatches = ()
        self.context = None

    def dispatch(self, dispatches: Any, *, context: Any) -> tuple[DispatchRecord, ...]:
        self.dispatches = tuple(dispatches)
        self.context = context
        return tuple(
            DispatchRecord.accepted(
                dispatch,
                backend="fake",
                launcher_job_id=context.allocation_id,
                submitted_command=dispatch.argv,
            )
            for dispatch in self.dispatches
        )


def _plan(tmp_path: Path, gpu_counts: tuple[int, ...]) -> StagePlanV2:
    tasks = tuple(
        LogicalTaskSpec(
            logical_task_id=f"row-{index}",
            stage="01_baseline",
            run_id=f"run-{index}",
            command=("{python}", "-c", "print('batch=4096')"),
            result_dir=str(tmp_path / f"result-{index}"),
            logs=(str(tmp_path / f"result-{index}" / "status.json"),),
            params={"total_batch_size": 4096},
            resources=ResourceSpec(profile="cuda", device="cuda", gpus=gpus),
            completion=CompletionSpec(policy="none"),
            metadata={"runtime": "ferminet"},
        )
        for index, gpus in enumerate(gpu_counts)
    )
    return StagePlanV2(
        study="baselines",
        stage="01_baseline",
        plan_id="plan-1",
        results_root=str(tmp_path / "results"),
        tasks=tasks,
    ).validate()


def test_one_plan_dispatches_uniform_one_gpu_rows_and_writes_records(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    code = pipeline.run_pipeline(
        plan=_plan(tmp_path, (1, 1)),
        facility="polaris",
        launch_dir=tmp_path / "launch",
        admission_id="admission-1",
        executor=executor,
        environ={"PBS_JOBID": "123.server"},
        cwd=tmp_path,
        python="/env/bin/python",
    )

    assert code == 0
    assert [row.environment for row in executor.dispatches] == [{}, {}]
    assert executor.context.visibility_values == ("0", "1", "2", "3")
    assert [row.argv for row in executor.dispatches] == [
        ("/env/bin/python", "-c", "print('batch=4096')")
    ] * 2
    records = (tmp_path / "launch" / "dispatch_records.jsonl").read_text().splitlines()
    assert len(records) == 2
    assert json.loads((tmp_path / "launch" / "verification.json").read_text())["complete"] is True
    assert json.loads((tmp_path / "launch" / "executor_input.json").read_text())["schema"] == "tpen-executor/ready-group-v1"


@pytest.mark.parametrize(
    ("steps", "expected_horizon", "has_unrepresentable"),
    [("100", 100, False), ("100.5", 0, True)],
)
def test_horizon_encoding_keeps_legacy_dispatch_compatible(
    tmp_path: Path, steps: str, expected_horizon: int, has_unrepresentable: bool
) -> None:
    """Wire coercion or metadata must never block the old dispatch route."""

    plan = build_plan(
        {
            "code": ("ferminet",),
            "ansatz": ("default",),
            "system": ("he",),
            "seed": (0,),
            "steps": (steps,),
            "batch_size": (32,),
        },
        command=("runner",),
        results_root=tmp_path / "results",
        plan_id="horizon-compat",
    )
    plan = replace(
        plan,
        tasks=tuple(
            replace(task, metadata={**task.metadata, "runtime": "ferminet"})
            for task in plan.tasks
        ),
    )
    executor = RecordingExecutor()
    assert pipeline.run_pipeline(
        plan=plan,
        facility="polaris",
        launch_dir=tmp_path / "launch",
        admission_id="horizon-compat",
        executor=executor,
        environ={"PBS_JOBID": "horizon.server"},
    ) == 0
    payload = json.loads((tmp_path / "launch" / "executor_input.json").read_text())
    assert payload["members"][0]["target_horizon"] == expected_horizon
    assert ("unrepresentable" in payload) is has_unrepresentable
    assert len(executor.dispatches) == 1


def test_four_gpu_rows_use_one_parsl_worker_each(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    assert pipeline.run_pipeline(
        plan=_plan(tmp_path, (4, 4, 4)),
        facility="polaris",
        launch_dir=tmp_path / "launch",
        admission_id="admission-2",
        executor=executor,
        environ={"PBS_JOBID": "456.server"},
    ) == 0
    assert executor.context.visibility_values == ("0,1,2,3",)


def test_mixed_gpu_plan_fails_before_dispatch(tmp_path: Path) -> None:
    executor = RecordingExecutor()
    bad = _plan(tmp_path, (1, 4))
    with pytest.raises(ValueError, match="uniform resources.gpus"):
        pipeline.run_pipeline(
            plan=bad,
            facility="polaris",
            launch_dir=tmp_path / "launch",
            admission_id="admission-3",
            executor=executor,
            environ={"PBS_JOBID": "789.server"},
        )
    assert executor.dispatches == ()


def test_cannon_is_not_a_parsl_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only on Polaris"):
        pipeline.allocation_context(
            facility="cannon",
            gpus_per_row=1,
            run_root=tmp_path / "launch",
            environ={"SLURM_JOB_ID": "1"},
        )


def test_accelerator_bindings_tile_the_node_for_every_admissible_width() -> None:
    """Row widths are the divisors of the node, derived rather than enumerated."""

    assert accelerator_bindings(1, 4) == ("0", "1", "2", "3")
    assert accelerator_bindings(2, 4) == ("0,1", "2,3")
    assert accelerator_bindings(4, 4) == ("0,1,2,3",)
    # The rule is divisibility, not a hardcoded (1, 2, 4), so a differently sized
    # node needs no change here.
    assert accelerator_bindings(2, 8) == ("0,1", "2,3", "4,5", "6,7")
    assert accelerator_bindings(8, 8) == ("0,1,2,3,4,5,6,7",)


@pytest.mark.parametrize(
    "gpus_per_row, fragment",
    [
        (3, "does not divide"),
        (0, "must be positive"),
        (-1, "must be positive"),
    ],
)
def test_accelerator_bindings_reject_widths_that_cannot_tile(
    gpus_per_row: int, fragment: str
) -> None:
    """A width that strands or straddles accelerators is rejected at plan time.

    This is where node saturation is enforced; the executor-side tiling check
    deliberately does not know the node's accelerator count.
    """

    with pytest.raises(ValueError, match=fragment):
        accelerator_bindings(gpus_per_row, 4)


def test_multi_node_context_admits_four_gpu_rows() -> None:
    """The previously blocked case: 4-GPU rows across more than one node.

    Before this change `_parsl_app_runner` raised
    `multi-node Parsl attach requires exactly four accelerators per node`
    for every `gpus_per_row` above 1, which confined multi-node dispatch to
    one GPU per row and blocked the production geometry outright.
    """

    for gpus_per_row in (1, 2, 4):
        context = allocation_context(
            facility="polaris",
            gpus_per_row=gpus_per_row,
            run_root="/tmp/does-not-need-to-exist",
            environ={"PBS_JOBID": "1.polaris", "TPEN_NODES_PER_BLOCK": "2"},
        )
        assert context.nodes_per_block == 2
        # The executor-side check must accept exactly what the planner produced.
        validate_accelerator_tiling(context.visibility_values)
