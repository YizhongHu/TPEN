from __future__ import annotations

from dataclasses import replace
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


def test_r3_t1_literal_argv_tokens_keep_exact_plan_identity(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    first = str(root / "not-a-script.py")
    second = str(root) + "/./not-a-script.py"
    base = _plan(tmp_path / "results")
    task = base.tasks[0]
    left = replace(base, tasks=(replace(task, command=("runner", "--literal", first)),))
    right = replace(base, tasks=(replace(task, command=("runner", "--literal", second)),))

    assert left.validate() is left
    assert right.validate() is right
    assert ready_group(left)["execution_plan_digest"] != ready_group(right)["execution_plan_digest"]


def test_r3_t2_single_current_artifact_in_both_publication_orders(tmp_path: Path) -> None:
    good = _plan(tmp_path / "valid-then-refused" / "results")
    current = tmp_path / "valid-then-refused" / "executor_input.json"
    assert write_additive_ready_group(current, good) == current
    prior = current.read_bytes()
    bad = replace(good, tasks=(replace(good.tasks[0], params={"steps": Path("100.5")}),))
    assert bad.validate() is bad
    assert write_additive_ready_group(current, bad) is None
    assert not current.exists()
    assert current.with_name("executor_input_diagnostics.json").exists()
    history = sorted(current.parent.glob("executor_input.superseded-*.json"))
    assert len(history) == 1
    assert history[0].read_bytes() == prior

    refused_first = _plan(tmp_path / "refused-then-valid" / "results")
    current = tmp_path / "refused-then-valid" / "executor_input.json"
    refused = replace(
        refused_first,
        tasks=(replace(refused_first.tasks[0], params={"steps": Path("100.5")}),),
    )
    assert write_additive_ready_group(current, refused) is None
    diagnostic = current.with_name("executor_input_diagnostics.json")
    assert diagnostic.exists()
    prior_diagnostic = diagnostic.read_bytes()
    assert write_additive_ready_group(current, refused_first) == current
    assert current.exists()
    assert not diagnostic.exists()
    diagnostic_history = sorted(
        current.parent.glob("executor_input_diagnostics.superseded-*.json")
    )
    assert len(diagnostic_history) == 1
    assert diagnostic_history[0].read_bytes() == prior_diagnostic
