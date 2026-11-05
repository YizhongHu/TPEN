from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.toolkit.dispatch import LogicalTaskSpec, StagePlanV2
from experiments.toolkit.executor_inputs import (
    DIAGNOSTICS_SCHEMA,
    UnrepresentableInput,
    canonical_bytes,
    ready_group,
    write_additive_ready_group,
)


def _plan(root: Path, *, config: Path | None = None, task_id: str = "task", steps=100):
    params = {"steps": steps}
    if config is not None:
        params["config"] = str(config)
    task = LogicalTaskSpec(
        logical_task_id=task_id,
        stage="toolkit",
        run_id="run",
        command=("runner",),
        result_dir=str(root / "run"),
        params=params,
    )
    return StagePlanV2(
        study="tests",
        stage="toolkit",
        plan_id="plan",
        results_root=str(root),
        tasks=(task,),
    ).validate()


def test_declared_repo_and_results_paths_have_distinct_namespaces(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    repo_path = Path(__file__).resolve().parents[2] / "experiments/atomistic/he-cutover/smoke_grid.yaml"
    result_path = result_root / "experiments/atomistic/he-cutover/smoke_grid.yaml"
    left = ready_group(_plan(result_root, config=repo_path))
    right = ready_group(_plan(result_root, config=result_path))

    assert left["members"][0]["program"]["config_ref"]["config"]["root"] == "repo"
    assert right["members"][0]["program"]["config_ref"]["config"]["root"] == "results"
    assert canonical_bytes(left) != canonical_bytes(right)


def test_opaque_logical_id_is_not_path_projected(tmp_path: Path) -> None:
    group = ready_group(_plan(tmp_path / "results", task_id="/opaque/logical/task"))
    assert group["members"][0]["task_id"] == "/opaque/logical/task"


def test_unrepresentable_group_writes_diagnostic_and_no_ready_group(tmp_path: Path) -> None:
    path = tmp_path / "executor_input.json"
    result = write_additive_ready_group(path, _plan(tmp_path / "results", steps=Path("100.5")))

    assert result is None
    assert not path.exists()
    diagnostic = json.loads((tmp_path / "executor_input_diagnostics.json").read_text())
    assert diagnostic["schema"] == DIAGNOSTICS_SCHEMA
    assert diagnostic["unrepresentable"][0]["field"] == "steps"
    assert diagnostic["unrepresentable"][0]["task_id"] == "task"


def test_non_unrepresentable_error_still_propagates(tmp_path: Path, monkeypatch) -> None:
    import experiments.toolkit.executor_inputs as emit

    def broken(*args, **kwargs):
        raise RuntimeError("encoder bug")

    monkeypatch.setattr(emit, "ready_group", broken)
    with pytest.raises(RuntimeError, match="encoder bug"):
        write_additive_ready_group(tmp_path / "executor_input.json", _plan(tmp_path))


def test_unrepresentable_exception_is_a_value_error() -> None:
    assert issubclass(UnrepresentableInput, ValueError)
