from __future__ import annotations

from pathlib import Path
import hashlib

import pytest
import yaml
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

cutover_plan = sibling(__file__, 'cutover_plan')
run_eval_row = sibling(__file__, 'run_eval_row')
run_train_row = sibling(__file__, 'run_train_row')


GRID = Path(__file__).with_name("smoke_grid.yaml")
PRODUCTION_GRID = Path(__file__).with_name("production_grid.yaml")
PROOF_GRID = Path(__file__).with_name("proof_grid.yaml")


def _override_value(overrides: list[str], key: str) -> str:
    prefix = f"{key}="
    values = [override.removeprefix(prefix) for override in overrides if override.startswith(prefix)]
    assert len(values) == 1
    return values[0]


def _assert_runner_matches_plan(row: dict[str, object], overrides: list[str]) -> None:
    assert _override_value(overrides, "run.layout") == "flat"
    instructed_dir = Path(_override_value(overrides, "run.root")) / _override_value(overrides, "run.run_id")
    assert str(instructed_dir) == str(row["result_dir"])


def test_strict_grid_rejects_unknown_key(tmp_path: Path) -> None:
    text = GRID.read_text(encoding="utf-8") + "unknown: true\n"
    path = tmp_path / "grid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(cutover_plan.PlanError, match="keys mismatch"):
        cutover_plan.load_grid(path)


def test_strict_grid_rejects_invalid_facility_placement(tmp_path: Path) -> None:
    payload = cutover_plan.load_grid(GRID)
    payload["facilities"]["polaris"]["partition"] = "not-debug"
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="debug or capacity on a100_40gb"):
        cutover_plan.load_grid(path)


@pytest.mark.parametrize("facility", ["cannon", "polaris"])
def test_plan_has_one_train_two_eval_and_exact_completion_specs(tmp_path: Path, facility: str) -> None:
    train, evaluation, manifest = cutover_plan.build_plans(cutover_plan.load_grid(GRID), facility=facility, results_root=tmp_path, plan_id="plan-1")
    assert [task.run_id for task in train.tasks] == ["seed-000"]
    assert [task.run_id for task in evaluation.tasks] == ["seed-000-chain-00", "seed-000-chain-01"]
    assert train.tasks[0].completion.policy == "status_completed_with_checkpoint"
    assert train.tasks[0].completion.checkpoint_path.endswith("step_000025/COMPLETE")
    assert {task.completion.policy for task in evaluation.tasks} == {"status_completed"}
    assert all(task.dependencies == (train.tasks[0].logical_task_id,) for task in evaluation.tasks)
    assert len(manifest["rows"]) == 3


def test_plan_writer_emits_rows_and_both_v2_stage_tables(tmp_path: Path) -> None:
    plans = cutover_plan.build_plans(cutover_plan.load_grid(GRID), facility="cannon", results_root=tmp_path / "results", plan_id="plan-1")
    output = cutover_plan.write_plans(tmp_path / "plan", *plans)
    assert (output / "rows.csv").read_text().splitlines()[0] == "stage,kind,row_id,facility,runtime,result_dir,checkpoint_dir"
    assert (output / "02_train/tasks.jsonl").is_file()
    assert (output / "03_eval/tasks.jsonl").is_file()


def test_train_and_eval_runner_paths_are_identical_to_the_plan(tmp_path: Path) -> None:
    train, evaluation, manifest = cutover_plan.build_plans(
        cutover_plan.load_grid(GRID), facility="cannon", results_root=tmp_path / "results", plan_id="plan-1"
    )
    train_row, eval_row = manifest["rows"][0], manifest["rows"][1]

    _assert_runner_matches_plan(train_row, run_train_row.output_overrides(train_row))
    assert train.tasks[0].completion.status_path == str(Path(str(train_row["result_dir"])) / "status.json")
    assert train.tasks[0].completion.checkpoint_path == str(Path(str(train_row["checkpoint_dir"])) / "COMPLETE")
    assert Path(str(train_row["checkpoint_dir"])).is_relative_to(Path(str(train_row["result_dir"])))

    _assert_runner_matches_plan(eval_row, run_eval_row.output_overrides(eval_row))
    assert evaluation.tasks[0].completion.status_path == str(Path(str(eval_row["result_dir"])) / "status.json")
    assert evaluation.tasks[0].completion.checkpoint_path is None
    assert eval_row["checkpoint_dir"] == train_row["checkpoint_dir"]


def test_production_plan_has_three_train_and_thirty_six_full_eval_rows(tmp_path: Path) -> None:
    train, evaluation, manifest = cutover_plan.build_plans(
        cutover_plan.load_grid(PRODUCTION_GRID),
        facility="polaris",
        results_root=tmp_path / "results",
        plan_id="full-1",
    )
    assert [task.run_id for task in train.tasks] == ["seed-000", "seed-001", "seed-002"]
    assert len(evaluation.tasks) == 36
    assert len(manifest["rows"]) == 39
    expected_tasks = {
        "mcmc_energy", "he_radial_profiles", "full_model_antisymmetry",
        "spatial_exchange_symmetry", "trace_equivariance", "he_en_numerical_atlas",
        "he_ee_ideal_vs_executed_numerical_atlas", "he_one_electron_tail_atlas",
        "he_center_of_mass_tail_atlas", "he_angular_shell_atlas",
    }
    assert all(set(task.params["task_names"]) == expected_tasks for task in evaluation.tasks)
    assert all(len(task.dependencies) == 1 for task in evaluation.tasks)
    assert {task.params["checkpoint_dir"].rsplit("/", 1)[-1] for task in evaluation.tasks} == {
        "step_100000", "step_200000", "step_300000"
    }


@pytest.mark.parametrize("facility", ["polaris", "polaris_scaling"])
def test_proof_plan_has_one_train_and_forty_eval_rows(tmp_path: Path, facility: str) -> None:
    train, evaluation, manifest = cutover_plan.build_plans(
        cutover_plan.load_grid(PROOF_GRID), facility=facility, results_root=tmp_path / "results", plan_id="proof-1"
    )
    assert [task.run_id for task in train.tasks] == ["seed-000"]
    assert len(evaluation.tasks) == 40
    assert len(manifest["rows"]) == 41
    assert all(task.params["scale"] == "smoke" for task in (*train.tasks, *evaluation.tasks))
    assert all(task.params["max_steps"] == 25 and task.params["n_walkers"] == 16 for task in train.tasks)
    assert {task.params["seed"] for task in evaluation.tasks} == set(range(1, 41))


@pytest.mark.parametrize("facility", ["polaris", "polaris_scaling"])
def test_proof_eval_rows_all_depend_on_single_train_row(tmp_path: Path, facility: str) -> None:
    train, evaluation, _ = cutover_plan.build_plans(
        cutover_plan.load_grid(PROOF_GRID), facility=facility, results_root=tmp_path / "results", plan_id="proof-1"
    )
    assert len(train.tasks) == 1
    assert all(task.dependencies == (train.tasks[0].logical_task_id,) for task in evaluation.tasks)


def test_proof_train_row_is_accepted_by_runner_configuration_path(tmp_path: Path) -> None:
    grid = cutover_plan.load_grid(PROOF_GRID)
    train_row = cutover_plan.expand_rows(grid, facility="polaris", results_root=tmp_path / "results")[0]
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "he-v1" / "configs" / "train.yaml")
    run_train_row.configure_training(cfg, train_row)
    assert cfg.trainer.max_steps == 25
    assert cfg.sampler.n_walkers == 16


@pytest.mark.parametrize("facility", ["polaris", "polaris_scaling"])
def test_proof_row_ids_are_unique_in_result_and_checkpoint_paths(tmp_path: Path, facility: str) -> None:
    rows = cutover_plan.expand_rows(
        cutover_plan.load_grid(PROOF_GRID), facility=facility, results_root=tmp_path / "results"
    )
    assert len({row["result_dir"] for row in rows}) == len(rows)
    for row in rows:
        assert Path(row["result_dir"]).parts.count(row["row_id"]) == 1
        checkpoint = Path(row["checkpoint_dir"])
        if row["kind"] == "train":
            assert checkpoint.parts.count(row["row_id"]) == 1
        else:
            assert checkpoint.parts.count("seed-000") == 1


def test_smoke_grid_bytes_and_expansion_are_unchanged(tmp_path: Path) -> None:
    smoke_bytes = GRID.read_bytes()
    assert hashlib.sha256(smoke_bytes).hexdigest() == "2b1a8a0a277a34e8c3b536cbfe27545b8e0f644ee9e1efebf15cb74f4e65d8e1"
    rows = cutover_plan.expand_rows(cutover_plan.load_grid(GRID), facility="cannon", results_root=tmp_path)
    assert [row["row_id"] for row in rows] == ["seed-000", "seed-000-chain-00", "seed-000-chain-01"]


def test_every_row_id_occurs_once_in_its_result_and_checkpoint_paths(tmp_path: Path) -> None:
    rows = cutover_plan.expand_rows(
        cutover_plan.load_grid(PRODUCTION_GRID),
        facility="polaris",
        results_root=tmp_path / "results",
    )
    assert len({row["result_dir"] for row in rows}) == len(rows)
    for row in rows:
        assert Path(row["result_dir"]).parts.count(row["row_id"]) == 1
        if row["kind"] == "train":
            assert Path(row["checkpoint_dir"]).parts.count(row["row_id"]) == 1
        else:
            seed_id = "-".join(row["row_id"].split("-")[:2])
            assert Path(row["checkpoint_dir"]).parts.count(seed_id) == 1
