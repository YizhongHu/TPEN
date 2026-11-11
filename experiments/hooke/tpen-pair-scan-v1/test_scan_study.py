"""Study-level tests for the TPEN pair-scan major/minor grid."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import pytest
from omegaconf import OmegaConf

STUDY_DIR = Path(__file__).resolve().parent
CONFIGS = STUDY_DIR / "configs"

# The study ships no checked-in grid -- a later layer owns the real one -- so the
# tests compile the scan grid themselves under ``tmp_path`` (see ``_write_grid``).
# ``config``/``validation_config`` stay repo-root-relative, so ``plan.main`` still
# has to run with the repo root as cwd, which is where pytest is invoked from.
_GRID_BODY = """\
study: tpen_pair_scan_v1
config: experiments/hooke/tpen-pair-scan-v1/configs/train.yaml
validation_config: experiments/hooke/tpen-pair-scan-v1/configs/eval.yaml

choice_libraries:
  - path: experiments/hooke/choices/basis_levels.yaml
    provides: choices.basis

config_snapshots:
  train: train_config.yaml
  validation: validation_config.yaml

major_grid:
  basis: [no-basis, hooke-total-shell]
  activation: [SiLU, Tanh]

minor_grid:
  lr: [1.0e-3, 3.0e-4]
  channels: [8, 16]

scan_seed_axis: seed_index
scan_seed_rows:
  - {seed_index: 0, training_model_seed: 0, training_sampler_seed: 10, validation_sampler_seed: 20}
  - {seed_index: 1, training_model_seed: 1, training_sampler_seed: 11, validation_sampler_seed: 21}

blinding:
  enabled_by_default: true
  slot_prefixes: {basis: B, activation: A}

axis_id_labels: {basis: b, activation: act, lr: lr, channels: ch, seed_index: seed}

axis_overrides:
  basis: run_parameters.basis_slot
  activation: run_parameters.activation_slot
  lr: run_parameters.lr
  channels: run_parameters.channels

choice_validation:
  basis: {choices_path: choices.basis, tags_path: 'choices.basis.{value}.tags'}
  activation: {choices_path: choices.activation, tags_path: 'choices.activation.{value}.tags'}

seed_overrides:
  scan_train:
    run_parameters.training_model_seed: training_model_seed
    run_parameters.training_sampler_seed: training_sampler_seed
  validation:
    run_parameters.training_model_seed: training_model_seed
    run_parameters.validation_sampler_seed: validation_sampler_seed
  final_train:
    run_parameters.training_model_seed: final_train_model_seed
    run_parameters.training_sampler_seed: final_train_sampler_seed
  final_eval:
    run_parameters.validation_sampler_seed: final_eval_sampler_seed

final_seed_sequences:
  final_train_model_seed: {start: 100, step: 1}
  final_train_sampler_seed: {start: 1000, step: 1}
  final_eval_sampler_seed: {start: 10000, step: 1}

champions:
  - name: energy
    selector: metric_ladder
    tasks: [mcmc_energy]
    metric_template: 'eval/{task}/local_energy_mean'
    mode: min
    fallback_metric: 'eval/mcmc_energy/local_energy_mean'
    fallback_mode: min

champion_reference_metrics:
  - {label: mcmc_energy, metric: 'eval/mcmc_energy/local_energy_mean'}
  - {label: stratified_variance, metric: 'eval/stratified_geometry/local_energy_variance'}

final_replicates: 9
"""

while str(STUDY_DIR) in sys.path:
    sys.path.remove(str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR))
_STUDY_TOP_LEVEL_MODULES = {
    "collect",
    "final_collect",
    "final_eval",
    "final_plan",
    "final_report",
    "final_train",
    "launch",
    "plan",
    "plot",
    "run_ids",
    "select_champions",
    "stats",
    "train",
    "utils",
    "validate",
}
for module_name in list(sys.modules):
    if module_name.split(".", maxsplit=1)[0] in _STUDY_TOP_LEVEL_MODULES:
        del sys.modules[module_name]


def _load_script(name: str) -> ModuleType:
    path = STUDY_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tpen_pair_scan_v1_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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

json_io = sibling(__file__, 'utils.io')
layout = sibling(__file__, 'utils.layout')
launch = _load_script("launch")
plan = _load_script("plan")
train = _load_script("train")
collect = _load_script("collect")
select_champions = _load_script("select_champions")
final_plan = _load_script("final_plan")
final_train = _load_script("final_train")
final_eval = _load_script("final_eval")
final_collect = _load_script("final_collect")
final_report = _load_script("final_report")
validate = _load_script("validate")
from experiments.toolkit import StagePlan, TaskLineageRow, read_task_lineage, write_task_lineage  # noqa: E402


ATTEMPT = "20260623T120000-0400"
ROOT = STUDY_DIR.parents[2]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_grid(tmp_path: Path) -> Path:
    # Deterministic: repeated calls rewrite the same bytes, so a test that needs
    # both the planned results and the grid values can call this again.
    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(f"results_root: {tmp_path / 'results'}\n{_GRID_BODY}")
    return grid_path


def _planned_results(tmp_path: Path) -> Path:
    results_root = tmp_path / "results"
    code = plan.main(
        ["--grid", str(_write_grid(tmp_path)), "--results-root", str(results_root), "--attempt-id", ATTEMPT]
    )
    assert code == 0
    return results_root


def _config_with_overrides(path: Path, overrides: Sequence[str]) -> Any:
    return OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(list(overrides)))


def _write_checkpoint_pointer(results_root: Path, run_id: str, attempt_id: str) -> Path:
    checkpoint_dir = layout.train_attempt_dir(results_root, run_id, attempt_id) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "latest.json").write_text(json.dumps({"path": "step_000000"}))
    return checkpoint_dir


def _write_final_checkpoint(results_root: Path, final_run_id: str, attempt_id: str) -> Path:
    attempt_dir = layout.final_train_attempt_dir(results_root, final_run_id, attempt_id)
    checkpoint_dir = attempt_dir / "checkpoints" / "step_000000"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "COMPLETE").write_text("")
    (checkpoint_dir / "manifest.json").write_text(json.dumps({"step": 0}) + "\n")
    latest = attempt_dir / "checkpoints" / "latest.json"
    latest.write_text(json.dumps({"checkpoint_dir": "step_000000"}) + "\n")
    (attempt_dir / "selected_checkpoint.json").write_text(
        json.dumps(
            {
                "selection_policy": "latest_checkpoint_pointer",
                "checkpoint_pointer": str(latest),
            }
        )
        + "\n"
    )
    return checkpoint_dir




def test_v3_test_partition_slurm_overrides_are_explicit() -> None:
    args = types.SimpleNamespace(
        slurm_partition="gpu_test",
        slurm_array_parallelism=None,
        slurm_timeout_min=60,
        slurm_mem_per_cpu_gb=8,
        slurm_cpus=6,
        slurm_gpus=None,
    )

    cuda = launch.slurm_parameters(args, profile="cuda")

    assert cuda["slurm_partition"] == "gpu_test"
    assert cuda["timeout_min"] == 60
    assert cuda["mem_per_cpu"] == "8G"
    assert "mem_gb" not in cuda
    assert launch.slurm_resource_mem_gb(cuda) == 48
    assert cuda["cpus_per_task"] == 6
    assert cuda["slurm_array_parallelism"] == launch.DEFAULT_ARRAY_PARALLELISM
    assert cuda["gpus_per_node"] == 1


def test_v3_submitit_mem_per_cpu_unsets_inherited_memory_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_parameters: dict[str, Any] = {}

    class FakeExecutor:
        def __init__(self, folder: str):
            self.folder = folder

        def update_parameters(self, **kwargs: Any) -> None:
            captured_parameters.update(kwargs)

        def map_array(self, fn: Any, *args: Any) -> list[types.SimpleNamespace]:
            return [types.SimpleNamespace(job_id="12345")]

    monkeypatch.setitem(sys.modules, "submitit", types.SimpleNamespace(AutoExecutor=FakeExecutor))

    launch.submit_submitit(
        [["bash", "-lc", "true"]],
        log_dir=tmp_path / "logs",
        job_name="mem-test",
        slurm={
            "slurm_partition": "gpu_test",
            "timeout_min": 10,
            "mem_per_cpu": "8G",
            "cpus_per_task": 8,
            "tasks_per_node": 1,
            "gpus_per_node": 1,
        },
    )

    setup = captured_parameters["slurm_setup"]
    assert "unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU" in setup
    assert "unset SLURM_MEM_PER_CPU" not in setup


def test_submitit_prints_parent_slurm_job_id_after_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeExecutor:
        def __init__(self, folder: str):
            self.folder = folder

        def update_parameters(self, **kwargs: Any) -> None:
            pass

        def map_array(self, fn: Any, *args: Any) -> list[types.SimpleNamespace]:
            return [
                types.SimpleNamespace(job_id="12345_0"),
                types.SimpleNamespace(job_id="12345_1"),
            ]

    monkeypatch.setitem(sys.modules, "submitit", types.SimpleNamespace(AutoExecutor=FakeExecutor))

    job_ids = launch.submit_submitit(
        [["bash", "-lc", "first"], ["bash", "-lc", "second"]],
        log_dir=tmp_path / "logs",
        job_name="pair-stability-train",
        slurm={"slurm_partition": "gpu_test"},
        chunk_size=1,
    )

    assert job_ids == ["12345_0", "12345_1"]
    assert capsys.readouterr().out == "[pair-stability-train] submitted Slurm job_id=12345\n"


def test_v2_mixed_device_prepares_cpu_and_cuda_commands() -> None:
    args = train.parse_args(
        [
            "--backend",
            "submitit",
            "--device",
            "cpu,cuda",
            "--slurm-cpu-partition",
            "test",
            "--slurm-cuda-partition",
            "gpu_test",
            "--slurm-cpu-timeout-min",
            "60",
            "--slurm-cuda-timeout-min",
            "30",
        ]
    )

    command_sets = launch.environment_command_sets(
        [["python", "-u", "run.py", "--config", "cfg.yaml", "runtime.device=cpu"]],
        args=args,
        repo_root=ROOT,
    )

    assert tuple(command_sets) == ("cpu", "cuda")
    cpu_script = command_sets["cpu"][0][-1]
    cuda_script = command_sets["cuda"][0][-1]
    assert "export UV_PROJECT_ENVIRONMENT=.venv" in cpu_script
    assert "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-1}}" in cpu_script
    assert "uv sync --extra cpu" in cpu_script
    assert f"flock {ROOT.parent / f'.{ROOT.name}.uv-sync.lock'} uv sync --extra cpu" in cpu_script
    assert "runtime.device=cpu" in cpu_script
    assert "export UV_PROJECT_ENVIRONMENT=.venv-gpu" in cuda_script
    assert "OMP_NUM_THREADS" not in cuda_script
    assert "uv sync --extra cu126" in cuda_script
    assert f"flock {ROOT.parent / f'.{ROOT.name}.uv-sync.lock'} uv sync --extra cu126" in cuda_script
    assert "runtime.device=cuda" in cuda_script

    cpu_slurm = launch.slurm_parameters(args, profile="cpu")
    assert cpu_slurm["slurm_partition"] == "test"
    assert cpu_slurm["mem_per_cpu"] == "8G"
    assert "mem_gb" not in cpu_slurm
    assert launch.slurm_resource_mem_gb(cpu_slurm) == 32
    assert cpu_slurm["cpus_per_task"] == 4
    cuda_slurm = launch.slurm_parameters(args, profile="cuda")
    assert cuda_slurm["slurm_partition"] == "gpu_test"
    assert cuda_slurm["mem_per_cpu"] == "8G"
    assert "mem_gb" not in cuda_slurm
    assert launch.slurm_resource_mem_gb(cuda_slurm) == 32
    assert cuda_slurm["cpus_per_task"] == 4
    assert cuda_slurm["gpus_per_node"] == 1


def test_v2_mixed_submitit_submits_separate_claimed_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_parameters: list[dict[str, Any]] = []
    captured_calls: list[tuple[Any, tuple[Any, ...]]] = []

    class FakeExecutor:
        def __init__(self, folder: str):
            self.folder = folder

        def update_parameters(self, **kwargs: Any) -> None:
            captured_parameters.append(kwargs)

        def map_array(self, fn: Any, *args: Any) -> list[types.SimpleNamespace]:
            captured_calls.append((fn, args))
            commands = args[0]
            return [
                types.SimpleNamespace(job_id=f"{self.folder}-job-{index}")
                for index, _command in enumerate(commands)
            ]

    monkeypatch.setitem(sys.modules, "submitit", types.SimpleNamespace(AutoExecutor=FakeExecutor))
    args = train.parse_args(
        [
            "--backend",
            "submitit",
            "--device",
            "cpu,cuda",
            "--slurm-cpu-partition",
            "test",
            "--slurm-cuda-partition",
            "gpu_test",
            "--slurm-cpu-timeout-min",
            "60",
            "--slurm-cuda-timeout-min",
            "30",
        ]
    )
    command_sets = {
        "cpu": [["bash", "-lc", "cpu"]],
        "cuda": [["bash", "-lc", "cuda"]],
    }
    row_status = tmp_path / "run" / "launcher_status.json"

    job_ids = launch.submit_command_sets(
        command_sets,
        args=args,
        backend="submitit",
        repo_root=ROOT,
        log_dir=tmp_path / "logs",
        job_name="mixed",
        smoke=False,
        row_status_paths=[row_status],
        chunk_status_dir=tmp_path / "chunks",
    )

    assert len(captured_parameters) == 2
    assert captured_parameters[0]["slurm_partition"] == "test"
    assert captured_parameters[0]["timeout_min"] == 60
    assert captured_parameters[0]["mem_per_cpu"] == "8G"
    assert "mem_gb" not in captured_parameters[0]
    assert captured_parameters[0]["cpus_per_task"] == 4
    assert captured_parameters[1]["slurm_partition"] == "gpu_test"
    assert captured_parameters[1]["timeout_min"] == 30
    assert captured_parameters[1]["mem_per_cpu"] == "8G"
    assert "mem_gb" not in captured_parameters[1]
    assert captured_parameters[1]["cpus_per_task"] == 4
    assert captured_parameters[1]["gpus_per_node"] == 1
    assert captured_calls[0][0] is launch.run_command_chunk
    assert captured_calls[0][1][5] == [[row_status.with_name("launcher_claim.json")]]
    assert captured_calls[0][1][6] == ["cpu"]
    assert captured_calls[1][1][5] == [[row_status.with_name("launcher_claim.json")]]
    assert captured_calls[1][1][6] == ["cuda"]
    assert job_ids[0].startswith("cpu:")
    assert ",cuda:" in job_ids[0]


def test_v2_local_claim_mode_uses_claim_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_submit_local(commands: Sequence[Sequence[str]], **kwargs: Any) -> list[str]:
        captured["commands"] = commands
        captured["kwargs"] = kwargs
        return ["local-0"]

    monkeypatch.setattr(launch, "submit_local", fake_submit_local)
    monkeypatch.setenv("SLURM_JOB_END_TIME", "1782579321")
    args = train.parse_args(["--backend", "local", "--device", "cpu"])
    row_status = tmp_path / "run" / "launcher_status.json"

    job_ids = launch.submit_command_sets(
        {"cpu": [["bash", "-lc", "cpu"]]},
        args=args,
        backend="local",
        repo_root=ROOT,
        log_dir=tmp_path / "logs",
        job_name="local",
        smoke=False,
        row_status_paths=[row_status],
        chunk_status_dir=tmp_path / "chunks",
        claim_rows=True,
    )

    assert job_ids == ["local-0"]
    assert captured["kwargs"]["claim_paths"] == [row_status.with_name("launcher_claim.json")]
    assert captured["kwargs"]["claim_label"] == "local-cpu"
    assert captured["kwargs"]["claim_deadline_unix"] == 1782579321.0
    assert captured["kwargs"]["claim_deadline_guard_min"] == 60


def test_v2_run_command_chunk_deadline_guard_skips_unclaimed_row(tmp_path: Path) -> None:
    row_status = tmp_path / "run" / "launcher_status.json"
    claim_path = row_status.with_name("launcher_claim.json")

    result = launch.run_command_chunk(
        [["bash", "-lc", "false"]],
        row_status_paths=[row_status],
        claim_paths=[claim_path],
        claim_label="local-cpu",
        claim_deadline_unix=0.0,
        claim_deadline_guard_min=60,
    )

    assert result["status"] == "deadline_guard"
    assert result["rows"][0]["status"] == "skipped_deadline_guard"
    assert not claim_path.exists()
    assert not row_status.exists()


def test_v2_run_command_chunk_reclaims_failed_row_claim(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "run"
    attempt_dir.mkdir()
    row_status = attempt_dir / "launcher_status.json"
    claim_path = attempt_dir / "launcher_claim.json"
    row_status.write_text(json.dumps({"status": "running"}) + "\n")
    (attempt_dir / "status.json").write_text(json.dumps({"status": "failed"}) + "\n")
    claim_path.write_text(json.dumps({"status": "claimed", "claim_label": "cuda"}) + "\n")

    result = launch.run_command_chunk(
        [["bash", "-lc", "true"]],
        row_status_paths=[row_status],
        claim_paths=[claim_path],
    )

    assert result["rows"][0]["status"] == "success"
    claim = json.loads(claim_path.read_text())
    assert claim["reclaimed"] is True
    assert claim["reclaim_reason"] == "failed"
    assert claim["previous_claim"]["claim_label"] == "cuda"


def test_v2_submitit_launcher_reexec_uses_dedicated_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_execvpe(file: str, args: list[str], env: dict[str, str]) -> None:
        captured["file"] = file
        captured["args"] = args
        captured["env"] = env
        raise RuntimeError("execvpe")

    monkeypatch.setattr(launch, "_python_in_environment", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(launch.os, "execvpe", fake_execvpe)

    args = types.SimpleNamespace(backend="submitit")
    with pytest.raises(RuntimeError, match="execvpe"):
        launch.ensure_submitit_launcher_environment(
            args,
            script_path=STUDY_DIR / "validate.py",
            argv=["--backend=submitit", "--wait-job=123"],
            repo_root=ROOT,
        )

    assert captured["file"] == "uv"
    assert captured["env"]["UV_PROJECT_ENVIRONMENT"] == ".venv-submitit"
    assert captured["env"]["SPENN_SUBMITIT_LAUNCHER_REEXEC"] == "1"
    assert captured["args"][:5] == ["uv", "run", "--extra", "submitit", "python"]
    assert "--wait-job=123" in captured["args"]


def test_v2_latest_attempt_id_prefers_pointer_with_sorted_fallback(tmp_path: Path) -> None:
    parent = tmp_path / "stage"
    (parent / "zzz").mkdir(parents=True)
    (parent / "aaa").mkdir()
    (parent / "diagnostic").mkdir()

    assert layout.latest_attempt_id(parent) == "zzz"
    layout.write_latest(parent, "aaa")

    assert layout.latest_attempt_id(parent) == "aaa"

    layout.write_latest(parent, "diagnostic")
    assert layout.latest_attempt_id(parent) == "diagnostic"
    assert layout.latest_attempt_id(parent / "missing") is None


def test_v2_final_workflow_defaults_to_single_latest_upstream_attempts(tmp_path: Path) -> None:
    results_root = tmp_path / "results"

    select_stage = layout.stage_dir(results_root, layout.STAGE_SELECT)
    (select_stage / "select-full").mkdir(parents=True)
    layout.write_latest(select_stage, "select-full")

    final_grid_stage = layout.stage_dir(results_root, layout.STAGE_FINAL_GRID)
    (final_grid_stage / "final-grid-full").mkdir(parents=True)
    layout.write_latest(final_grid_stage, "final-grid-full")

    assert final_plan._resolve_selection_attempt(results_root, None) == "select-full"
    assert final_train._resolve_final_grid_attempt_id(results_root, None) == "final-grid-full"
    assert final_eval._resolve_final_grid_attempt_id(results_root, None) == "final-grid-full"


def test_v2_train_and_validation_default_through_latest_pointers(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    job = manifest["jobs"][0]
    run_id = str(job["run_id"])

    row_status_paths = train.write_train_launch_provenance(
        [job],
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        repo_root=ROOT,
        submitted_commands=[["python", "run.py"]],
    )
    _write_checkpoint_pointer(results_root, run_id, ATTEMPT)
    _write_checkpoint_pointer(results_root, run_id, "zzz")

    assert row_status_paths == [layout.train_attempt_dir(results_root, run_id, ATTEMPT) / "launcher_status.json"]
    assert validate.latest_train_attempt_id(results_root, run_id) == ATTEMPT

    scalar_axes = validate._scalar_axes(manifest)
    args = types.SimpleNamespace(smoke=False, train_attempt_id=None, attempt_id="manual-validation")
    planned, skipped = validate.plan_validation_jobs(
        [job],
        args=args,
        study="tpen_pair_scan_v1",
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        validation_config="validation.yaml",
        scalar_axes=scalar_axes,
        override_paths=validate._axis_override_paths(manifest, scalar_axes),
        seed_axis=str(manifest["scan_seed_axis"]),
        static_stage_overrides={},
        seed_policy=manifest.get("seed_overrides"),
    )

    assert skipped == []
    assert planned[0]["train_attempt_id"] == ATTEMPT
    latest_validation = json.loads((layout.validation_run_dir(results_root, run_id) / "latest.json").read_text())
    assert latest_validation["attempt_id"] == "manual-validation"


def test_v3_train_main_writes_toolkit_stage_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_root = _planned_results(tmp_path)
    submitted_commands: list[list[str]] = []
    captured: dict[str, Any] = {}

    def fake_submit_command_sets(command_sets: dict[str, list[list[str]]], **kwargs: Any) -> list[str]:
        commands = command_sets["cpu"]
        submitted_commands.extend([list(command) for command in commands])
        captured["kwargs"] = kwargs
        assert len(kwargs["row_status_paths"]) == len(commands)
        assert kwargs["chunk_status_dir"] == results_root / "01_train" / "chunk_status" / ATTEMPT
        return [f"local-train-{index}" for index, _ in enumerate(commands)]

    monkeypatch.setattr(train.launch, "submit_command_sets", fake_submit_command_sets)

    code = train.main(
        [
            "--results-root",
            str(results_root),
            "--grid-attempt-id",
            ATTEMPT,
            "--backend",
            "local",
            "--device",
            "cpu",
        ]
    )

    assert code == 0
    assert len(submitted_commands) == 32
    assert captured["kwargs"]["backend"] == "local"
    plan_dir = results_root / "01_train" / "stage_plans" / ATTEMPT
    stage_plan = StagePlan.read(plan_dir)
    manifest = json.loads((plan_dir / "stage_manifest.json").read_text())
    tasks = [json.loads(line) for line in (plan_dir / "tasks.jsonl").read_text().splitlines()]
    executions = [json.loads(line) for line in (plan_dir / "execution_records.jsonl").read_text().splitlines()]
    assert stage_plan.n_tasks == 32
    assert stage_plan.tasks[0].stage == "01_train"
    assert manifest["study"] == "tpen_pair_scan_v1"
    assert manifest["stage"] == "01_train"
    assert manifest["n_tasks"] == 32
    assert len(tasks) == 32
    assert len(executions) == 32
    assert tasks[0]["completion"]["policy"] == "status_completed_with_checkpoint"
    assert executions[0]["launcher_job_id"] == "local-train-0"


def test_v2_validation_defaults_to_single_latest_train_attempt(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    job = manifest["jobs"][0]
    run_id = str(job["run_id"])
    diagnostic_attempt = "diagnostic-train"
    _write_checkpoint_pointer(results_root, run_id, ATTEMPT)
    _write_checkpoint_pointer(results_root, run_id, diagnostic_attempt)
    layout.write_latest(layout.train_run_dir(results_root, run_id), diagnostic_attempt)

    assert validate.latest_train_attempt_id(results_root, run_id) == diagnostic_attempt

    args = types.SimpleNamespace(smoke=False, train_attempt_id=None, attempt_id="real-validation")
    scalar_axes = validate._scalar_axes(manifest)
    planned, skipped = validate.plan_validation_jobs(
        [job],
        args=args,
        study="tpen_pair_scan_v1",
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        validation_config="validation.yaml",
        scalar_axes=scalar_axes,
        override_paths=validate._axis_override_paths(manifest, scalar_axes),
        seed_axis=str(manifest["scan_seed_axis"]),
        static_stage_overrides={},
        seed_policy=manifest.get("seed_overrides"),
    )
    assert skipped == []
    assert planned[0]["train_attempt_id"] == diagnostic_attempt

    args.train_attempt_id = ATTEMPT
    planned, skipped = validate.plan_validation_jobs(
        [job],
        args=args,
        study="tpen_pair_scan_v1",
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        validation_config="validation.yaml",
        scalar_axes=scalar_axes,
        override_paths=validate._axis_override_paths(manifest, scalar_axes),
        seed_axis=str(manifest["scan_seed_axis"]),
        static_stage_overrides={},
        seed_policy=manifest.get("seed_overrides"),
    )
    assert skipped == []
    assert planned[0]["train_attempt_id"] == ATTEMPT


def test_v3_validation_attempt_id_agrees_across_plan_and_stage_plan(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    job = manifest["jobs"][0]
    run_id = str(job["run_id"])
    _write_checkpoint_pointer(results_root, run_id, ATTEMPT)

    args = validate.parse_args(["--attempt-id", "V1", "--train-attempt-id", ATTEMPT, "--backend", "local", "--device", "cpu"])
    scalar_axes = validate._scalar_axes(manifest)
    planned, skipped = validate.plan_validation_jobs(
        [job],
        args=args,
        study="tpen_pair_scan_v1",
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        validation_config="validation.yaml",
        scalar_axes=scalar_axes,
        override_paths=validate._axis_override_paths(manifest, scalar_axes),
        seed_axis=str(manifest["scan_seed_axis"]),
        static_stage_overrides={},
        seed_policy=manifest.get("seed_overrides"),
    )
    assert skipped == []
    assert planned[0]["validation_attempt_id"] == "V1"

    stage_plan = validate.build_validation_stage_plan(
        planned,
        manifest=manifest,
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        args=args,
    )

    # An explicit --attempt-id must agree between the actual result directory
    # and the stage plan's attempt id / task ids.
    assert stage_plan.attempt_id == "V1"
    assert stage_plan.tasks[0].task_id == f"02_validation:{run_id}:V1"


def test_v2_wait_job_submits_dependent_launcher(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> types.SimpleNamespace:
        calls.append((command, kwargs))
        return types.SimpleNamespace(returncode=0, stdout="88888;cluster\n", stderr="")

    monkeypatch.setattr(launch.subprocess, "run", fake_run)

    submitted = launch.submit_dependent_launcher(
        "24211558",
        script_path=STUDY_DIR / "validate.py",
        argv=[
            "--backend=submitit",
            "--cuda",
            "--wait-job=24211558",
            "--chunk-size",
            "32",
        ],
        repo_root=ROOT,
        log_dir=tmp_path / "logs",
        job_name="pair-stability-v2-validate-launcher",
        partition="test",
        timeout_min=19,
        study="tpen_pair_scan_v1",
    )

    command, kwargs = calls[0]
    assert submitted == "88888"
    assert "--dependency=afterany:24211558" in command
    assert "--partition=test" in command
    assert "--time=00:19:00" in command
    assert "--mem-per-cpu=8G" in command
    assert not any(part.startswith("--mem=") for part in command)
    assert "--output=" + str(tmp_path / "logs" / "%x-%j.out") in command
    script = str(kwargs["input"])
    assert "UV_PROJECT_ENVIRONMENT=.venv-submitit" in script
    assert "uv run --extra submitit python -u" in script
    assert "--wait-job" not in script
    assert "--backend=submitit" in script


def test_v2_blinding_is_reproducible_by_seed(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    grid_path = _write_grid(tmp_path)
    for attempt, seed in (("SAME1", 811), ("SAME2", 811), ("DIFF", 812)):
        code = plan.main(
            [
                "--grid",
                str(grid_path),
                "--results-root",
                str(results_root),
                "--attempt-id",
                attempt,
                "--blind",
                "--blind-seed",
                str(seed),
            ]
        )
        assert code == 0

    same1 = json.loads((results_root / "00_grid" / "SAME1" / "unblind.json").read_text())
    same2 = json.loads((results_root / "00_grid" / "SAME2" / "unblind.json").read_text())
    diff = json.loads((results_root / "00_grid" / "DIFF" / "unblind.json").read_text())

    assert same1["axes"] == same2["axes"]
    assert same1["axes"] != diff["axes"]


def test_v3_plan_records_major_minor_scan_manifest(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    grid_attempt = layout.grid_attempt_dir(results_root, ATTEMPT)
    manifest = json.loads((grid_attempt / "manifest.json").read_text())

    assert manifest["study"] == "tpen_pair_scan_v1"
    assert manifest["config_snapshots"] == {
        "train": "train_config.yaml",
        "validation": "validation_config.yaml",
    }
    assert (grid_attempt / "train_config.yaml").is_file()
    assert (grid_attempt / "validation_config.yaml").is_file()
    assert not (grid_attempt / "smoke_config.yaml").exists()
    assert not (grid_attempt / "train.yaml").exists()
    assert not (grid_attempt / "eval.yaml").exists()
    assert manifest["grid_schema"] == "major_minor_scan"
    assert manifest["major_axes"] == ["basis", "activation"]
    assert manifest["minor_axes"] == ["lr", "channels"]
    assert manifest["scan_seed_axis"] == "seed_index"
    assert manifest["scan_seed_rows"] == [
        {
            "seed_index": 0,
            "training_model_seed": 0,
            "training_sampler_seed": 10,
            "validation_sampler_seed": 20,
        },
        {
            "seed_index": 1,
            "training_model_seed": 1,
            "training_sampler_seed": 11,
            "validation_sampler_seed": 21,
        },
    ]
    assert manifest["axis_id_labels"] == {
        "basis": "b",
        "activation": "act",
        "lr": "lr",
        "channels": "ch",
        "seed_index": "seed",
    }
    assert manifest["axis_overrides"] == {
        "basis": "run_parameters.basis_slot",
        "activation": "run_parameters.activation_slot",
        "lr": "run_parameters.lr",
        "channels": "run_parameters.channels",
    }
    assert manifest["choice_validation"]["basis"]["choices_path"] == "choices.basis"
    assert manifest["choice_validation"]["activation"]["choices_path"] == "choices.activation"
    assert [champion["name"] for champion in manifest["champions"]] == ["energy"]
    assert manifest["champion_kinds"] == ["energy"]
    assert manifest["champions"][0]["selector"] == "metric_ladder"
    assert manifest["seed_overrides"]["scan_train"] == {
        "run_parameters.training_model_seed": "training_model_seed",
        "run_parameters.training_sampler_seed": "training_sampler_seed",
    }
    assert manifest["seed_overrides"]["validation"] == {
        "run_parameters.training_model_seed": "training_model_seed",
        "run_parameters.validation_sampler_seed": "validation_sampler_seed",
    }
    assert manifest["seed_overrides"]["final_eval"] == {
        "run_parameters.validation_sampler_seed": "final_eval_sampler_seed",
    }
    assert manifest["final_seed_sequences"] == {
        "final_train_model_seed": {"start": 100, "step": 1},
        "final_train_sampler_seed": {"start": 1000, "step": 1},
        "final_eval_sampler_seed": {"start": 10000, "step": 1},
    }
    assert manifest["static_overrides"] == {}
    assert manifest["final_replicates"] == 9
    assert manifest["n_jobs"] == 32
    assert manifest["blinding"]["enabled"] is True
    assert manifest["blinding"]["blind_seed"] == 0

    grid = OmegaConf.load(_write_grid(tmp_path))
    unblind = json.loads((grid_attempt / "unblind.json").read_text())
    assert set(unblind["axes"]) == {"basis", "activation"}
    assert set(unblind["axes"]["basis"]["slot_to_value"].values()) == set(grid.major_grid.basis)
    assert set(unblind["axes"]["activation"]["slot_to_value"].values()) == set(grid.major_grid.activation)

    jobs = manifest["jobs"]
    assert {job["choices"]["basis"] for job in jobs} == set(unblind["axes"]["basis"]["slot_to_value"])
    assert {job["choices"]["activation"] for job in jobs} == set(
        unblind["axes"]["activation"]["slot_to_value"]
    )
    assert {float(job["choices"]["lr"]) for job in jobs} == {float(value) for value in grid.minor_grid.lr}
    assert {job["choices"]["channels"] for job in jobs} == {int(value) for value in grid.minor_grid.channels}
    assert {job["choices"]["seed_index"] for job in jobs} == {0, 1}

    job = jobs[0]
    assert job["run_id"].startswith("b-")
    assert "_act-" in job["run_id"]
    assert job["minor_id"].startswith("lr-")
    assert job["minor_choices"]["channels"] == 8
    assert job["scan_seed"] in {0, 1}
    assert {
        key: job["seed_values"][key]
        for key in ("seed_index", "training_model_seed", "training_sampler_seed", "validation_sampler_seed")
    } in manifest["scan_seed_rows"]
    assert job["seed_overrides"]["scan_train"] == {
        "run_parameters.training_model_seed": job["seed_values"]["training_model_seed"],
        "run_parameters.training_sampler_seed": job["seed_values"]["training_sampler_seed"],
    }
    assert "study.name=tpen_pair_scan_v1" in job["overrides"]
    assert "experiment.name=tpen_pair_scan_v1" in job["overrides"]
    assert "experiment.run_name=tpen_pair_scan_v1_train" in job["overrides"]
    assert (
        f"run_parameters.training_model_seed={job['seed_values']['training_model_seed']}" in job["overrides"]
    )
    assert (
        f"run_parameters.training_sampler_seed={job['seed_values']['training_sampler_seed']}"
        in job["overrides"]
    )
    assert any(str(override).startswith("run_parameters.basis_slot=B") for override in job["overrides"])
    assert any(str(override).startswith("run_parameters.activation_slot=A") for override in job["overrides"])


def test_v2_validation_config_resolves_from_manifest_snapshot(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)

    resolved = validate._validation_config_from_grid(
        results_root=results_root,
        grid_attempt_id=ATTEMPT,
        requested_config=None,
    )

    assert resolved == str(results_root / "00_grid" / ATTEMPT / "validation_config.yaml")


# The study configs own only the activation choice table now; `choices.basis` is
# merged in from the shared library the grid declares, and that composition is
# the fork-contract file's subject.
def test_v3_config_choices_cover_activation_grid_axis(tmp_path: Path) -> None:
    grid = OmegaConf.load(_write_grid(tmp_path))
    config_paths = [CONFIGS / "train.yaml", CONFIGS / "eval.yaml"]
    for config_path in config_paths:
        cfg = OmegaConf.load(config_path)
        assert set(grid.major_grid.activation) <= set(cfg.choices.activation)

        for activation in grid.major_grid.activation:
            resolved = _config_with_overrides(config_path, [f"run_parameters.activation_slot={activation}"])
            assert OmegaConf.select(resolved, "model.layers.0.mixing.activation._target_")
            assert OmegaConf.select(resolved, "model.layers.0.path_aggregation.activation._target_")


@pytest.mark.parametrize(
    ("module", "argv"),
    [
        (train, ["--smoke", "--backend", "local"]),
        (validate, ["--smoke", "--backend", "local"]),
        (final_train, ["--smoke", "--backend", "local"]),
        (final_eval, ["--smoke", "--backend", "local"]),
        (collect, ["--smoke"]),
        (select_champions, ["--smoke"]),
        (final_plan, ["--smoke"]),
        (final_collect, ["--smoke"]),
        (final_report, ["--smoke"]),
    ],
)
def test_v3_smoke_flag_is_deprecated(module: ModuleType, argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        module.parse_args(argv)
    assert exc_info.value.code == 2


def test_v2_collect_uses_status_for_required_train_wall_time(tmp_path: Path) -> None:
    train_attempt = tmp_path / "01_train" / "run-a" / "T1"
    train_attempt.mkdir(parents=True)
    (train_attempt / "status.json").write_text(
        json.dumps(
            {
                "start_time": "2026-06-24T10:00:00+00:00",
                "end_time": "2026-06-24T10:02:03+00:00",
            }
        )
        + "\n"
    )
    # This file is intentionally invalid. Wall time should come from
    # status.json without forcing collection to parse large train metrics.
    (train_attempt / "metrics.jsonl").write_text("{not-json}\n")

    metrics = collect._train_metrics(
        {"train_attempt_dir": str(train_attempt)},
        required_metrics={collect.TRAIN_WALL_TIME_METRIC},
    )

    assert metrics == {collect.TRAIN_WALL_TIME_METRIC: 123.0}


def test_v2_collect_prefers_grid_job_choices_over_resolved_config(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "02_validation" / "run-a" / "V1"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "resolved_config.yaml").write_text("run_parameters: [not-a-mapping\n")
    (attempt_dir / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")
    (attempt_dir / "metrics.jsonl").write_text("")
    axis_metadata = {
        "major_axes": ("basis", "mechanism"),
        "minor_axes": ("lr", "channels"),
        "config_axes": ("basis", "mechanism", "lr", "channels"),
        "run_axes": ("basis", "mechanism", "lr", "channels", "seed"),
        "axis_id_labels": {"basis": "b", "mechanism": "m", "lr": "lr", "channels": "ch", "seed": "seed"},
    }
    grid_job = {
        "choices": {"basis": "B00", "mechanism": "A00", "lr": 1.0e-3, "channels": 4, "seed": 0},
        "major_id": "b-B00_m-A00",
        "minor_id": "lr-1e-3_ch-4",
        "config_id": "b-B00_m-A00_lr-1e-3_ch-4",
    }

    row = collect.collect_validation_attempt(
        "run-a",
        "V1",
        attempt_dir,
        grid_job=grid_job,
        axis_metadata=axis_metadata,
        required_train_metrics=set(),
    )

    assert row["basis"] == "B00"
    assert row["mechanism"] == "A00"
    assert row["major_id"] == "b-B00_m-A00"


def test_v2_validate_main_consumes_planned_manifest_snapshot(tmp_path: Path, monkeypatch) -> None:
    results_root = _planned_results(tmp_path)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    job = manifest["jobs"][0]
    _write_checkpoint_pointer(results_root, str(job["run_id"]), ATTEMPT)
    submitted_commands: list[list[str]] = []
    captured: dict[str, Any] = {}

    def fake_submit_command_sets(command_sets: dict[str, list[list[str]]], **kwargs: Any) -> list[str]:
        commands = command_sets["cpu"]
        submitted_commands.extend([list(command) for command in commands])
        captured["kwargs"] = kwargs
        assert len(kwargs["row_status_paths"]) == len(commands)
        assert kwargs["chunk_status_dir"] == results_root / "02_validation" / "chunk_status" / "V1"
        return [f"local-validation-{index}" for index, _ in enumerate(commands)]

    # The script under test imports a direct ``launch`` module when executed as
    # a file; bind the v2 module explicitly so this test remains isolated from
    # the legacy pair_stability test module imports.
    monkeypatch.setattr(validate, "launch", launch)
    monkeypatch.setattr(validate.launch, "submit_command_sets", fake_submit_command_sets)

    code = validate.main(
        [
            "--results-root",
            str(results_root),
            "--grid-attempt-id",
            ATTEMPT,
            "--train-attempt-id",
            ATTEMPT,
            "--attempt-id",
            "V1",
            "--backend",
            "local",
        ]
    )

    assert code == 0
    assert len(submitted_commands) == 1
    assert captured["kwargs"]["backend"] == "local"
    assert captured["kwargs"]["allow_partial_failures"] is True
    script = submitted_commands[0][-1]
    assert str(results_root / "00_grid" / ATTEMPT / "validation_config.yaml") in script
    assert "run_parameters.basis_slot=" in script
    assert "run_parameters.activation_slot=" in script
    assert (
        f"run_parameters.validation_sampler_seed={job['seed_values']['validation_sampler_seed']}" in script
    )
    assert "load.path=" in script
    assert "study.name=tpen_pair_scan_v1" in script

    validation_attempt = results_root / "02_validation" / str(job["run_id"]) / "V1"
    source_train = json.loads((validation_attempt / "source_train_attempt.json").read_text())
    source_grid = json.loads((validation_attempt / "source_grid_attempt.json").read_text())
    submission = json.loads((validation_attempt / "submission.json").read_text())
    assert source_train["grid_attempt_id"] == ATTEMPT
    assert source_train["train_attempt_id"] == ATTEMPT
    assert source_grid["grid_attempt_id"] == ATTEMPT
    assert submission["launcher_job_id"] == "local-validation-0"
    assert "validation_config.yaml" in submission["submitted_command"]
    plan_dir = results_root / "02_validation" / "stage_plans" / "V1"
    stage_plan = StagePlan.read(plan_dir)
    manifest = json.loads((plan_dir / "stage_manifest.json").read_text())
    tasks = [json.loads(line) for line in (plan_dir / "tasks.jsonl").read_text().splitlines()]
    executions = [json.loads(line) for line in (plan_dir / "execution_records.jsonl").read_text().splitlines()]
    assert stage_plan.n_tasks == 1
    assert stage_plan.tasks[0].stage == "02_validation"
    assert manifest["study"] == "tpen_pair_scan_v1"
    assert manifest["stage"] == "02_validation"
    assert manifest["n_tasks"] == 1
    assert tasks[0]["completion"]["policy"] == "status_completed"
    assert executions[0]["launcher_job_id"] == "local-validation-0"


def _write_collection_summary(results_root: Path) -> None:
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    rows = []
    for job in manifest["jobs"]:
        point = dict(job["choices"])
        lr = float(point["lr"])
        seed = int(point["seed_index"])
        channel = int(point["channels"])
        # Only the minor axes move the energy: the major axes (basis, activation)
        # are grouping keys, and their blinded slot values are not comparable.
        energy = 2.0 + (0.0 if lr == 3.0e-4 else 0.2) + (0.0 if channel == 8 else 0.01)
        variance = 0.01 if lr == 1.0e-3 else 0.03
        rows.append(
            {
                "run_id": job["run_id"],
                "status": "completed",
                **{key: str(value) for key, value in point.items()},
                "major_id": job["major_id"],
                "minor_id": job["minor_id"],
                "config_id": job["config_id"],
                "eval/mcmc_energy/local_energy_mean": str(energy + 0.001 * seed),
                "eval/stratified_geometry/local_energy_variance": str(variance + 0.001 * seed),
            }
        )
    collect_dir = results_root / "03_collect" / "C1"
    _write_csv(collect_dir / "summary.csv", rows)
    (collect_dir / "source_grid_attempt.json").write_text(json.dumps({"grid_attempt_id": ATTEMPT}) + "\n")
    layout.write_latest(results_root / "03_collect", "C1")


def test_collect_defaults_to_latest_grid_plan_not_newest_validation(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    job = manifest["jobs"][0]
    validation_dir = results_root / "02_validation" / job["run_id"] / "V1"
    validation_dir.mkdir(parents=True)
    (validation_dir / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")
    (validation_dir / "source_grid_attempt.json").write_text(
        json.dumps(
            {
                "grid_attempt_id": ATTEMPT,
                "grid_attempt_dir": str(results_root / "00_grid" / ATTEMPT),
                "manifest_path": str(results_root / "00_grid" / ATTEMPT / "manifest.json"),
            }
        )
        + "\n"
    )
    (validation_dir / "source_train_attempt.json").write_text(
        json.dumps(
            {
                "run_id": job["run_id"],
                "grid_attempt_id": ATTEMPT,
                "train_attempt_id": ATTEMPT,
            }
        )
        + "\n"
    )
    (validation_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "namespace": "eval/stratified_geometry",
                "step": 0,
                "metrics": {"local_energy_mean": 2.0},
            }
        )
        + "\n"
    )

    stale_grid_id = "stale-grid"
    stale_run_id = "stale-run"
    stale_manifest = dict(manifest)
    stale_manifest["attempt_id"] = stale_grid_id
    stale_manifest["jobs"] = [{**job, "run_id": stale_run_id}]
    stale_grid_dir = results_root / "00_grid" / stale_grid_id
    _write_json(stale_grid_dir / "manifest.json", stale_manifest)
    stale_validation_dir = results_root / "02_validation" / stale_run_id / "ZZZ"
    _write_json(stale_validation_dir / "status.json", {"status": "completed"})
    _write_json(
        stale_validation_dir / "source_grid_attempt.json",
        {
            "grid_attempt_id": stale_grid_id,
            "grid_attempt_dir": str(stale_grid_dir),
            "manifest_path": str(stale_grid_dir / "manifest.json"),
        },
    )
    (stale_validation_dir / "metrics.jsonl").write_text("")

    result = collect.collect(results_root=results_root, collect_attempt_id="C0")
    report = result["report"]
    source = json.loads((results_root / "03_collect" / "C0" / "source_grid_attempt.json").read_text())
    latest = json.loads((results_root / "03_collect" / "latest.json").read_text())

    assert report["grid_attempt_id"] == ATTEMPT
    assert latest["attempt_id"] == "C0"
    assert source["grid_attempt_id"] == ATTEMPT
    assert source["manifest_path"].endswith("/00_grid/20260623T120000-0400/manifest.json")
    assert len(result["rows"]) == 1
    assert result["rows"][0]["run_id"] == job["run_id"]
    assert result["rows"][0]["basis"].startswith("B")
    assert result["rows"][0]["activation"].startswith("A")

    parallel = collect.collect(results_root=results_root, collect_attempt_id="C1")
    assert parallel["report"]["grid_attempt_id"] == ATTEMPT
    assert (results_root / "03_collect" / "C0" / "summary.csv").is_file()
    assert (results_root / "03_collect" / "C1" / "summary.csv").is_file()


def test_v3_collect_writes_task_lineage_verified_against_real_stage_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = _planned_results(tmp_path)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    job = manifest["jobs"][0]
    run_id = str(job["run_id"])
    _write_checkpoint_pointer(results_root, run_id, ATTEMPT)

    def fake_submit_command_sets(command_sets: dict[str, list[list[str]]], **kwargs: Any) -> list[str]:
        return [f"local-validation-{index}" for index, _ in enumerate(command_sets["cpu"])]

    monkeypatch.setattr(validate, "launch", launch)
    monkeypatch.setattr(validate.launch, "submit_command_sets", fake_submit_command_sets)
    code = validate.main(
        [
            "--results-root",
            str(results_root),
            "--grid-attempt-id",
            ATTEMPT,
            "--train-attempt-id",
            ATTEMPT,
            "--attempt-id",
            "V1",
            "--backend",
            "local",
        ]
    )
    assert code == 0

    validation_attempt = results_root / "02_validation" / run_id / "V1"
    (validation_attempt / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")
    (validation_attempt / "metrics.jsonl").write_text("")

    result = collect.collect(results_root=results_root, collect_attempt_id="C0")
    assert len(result["rows"]) == 1

    lineage = read_task_lineage(results_root / "03_collect" / "C0")
    assert set(lineage) == {run_id}
    assert lineage[run_id].task_ids["validation"] == f"02_validation:{run_id}:V1"
    assert lineage[run_id].task_ids["train"] == f"01_train:{run_id}:{ATTEMPT}"

    # The recorded validation task id must be a real, plan-verified task id,
    # not just a well-formed string: cross-check it against the stage plan
    # validate.py actually wrote.
    stage_plan = StagePlan.read(results_root / "02_validation" / "stage_plans" / "V1")
    assert lineage[run_id].task_ids["validation"] in {task.task_id for task in stage_plan.tasks}


def test_v3_select_chains_task_lineage_from_collection_sidecar(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    _write_collection_summary(results_root)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    write_task_lineage(
        results_root / "03_collect" / "C1",
        [
            TaskLineageRow(
                row_id=str(job["run_id"]),
                task_ids={
                    "validation": f"02_validation:{job['run_id']}:V1",
                    "train": f"01_train:{job['run_id']}:T1",
                },
            )
            for job in manifest["jobs"]
        ],
    )

    result = select_champions.select(results_root=results_root, select_attempt_id="S1")
    champions = _read_csv(results_root / "04_select" / "S1" / "champions.csv")
    lineage = read_task_lineage(results_root / "04_select" / "S1")

    assert champions
    checked = 0
    for champion in champions:
        contributing_run_ids = [run_id for run_id in champion["run_ids"].split(";") if run_id]
        if not contributing_run_ids:
            continue
        row_id = f"{champion['winner_kind']}:" + "|".join(
            f"{axis}={champion[axis]}" for axis in result["report"]["group_by"]
        )
        assert row_id in lineage
        expected_validation = {f"02_validation:{run_id}:V1" for run_id in contributing_run_ids}
        expected_train = {f"01_train:{run_id}:T1" for run_id in contributing_run_ids}
        assert set(lineage[row_id].task_ids["validation"]) == expected_validation
        assert set(lineage[row_id].task_ids["train"]) == expected_train
        checked += 1
    assert checked > 0


def test_v3_selects_energy_champions_per_major_and_plans_nine_final_seeds_by_default(
    tmp_path: Path,
) -> None:
    results_root = _planned_results(tmp_path)
    _write_collection_summary(results_root)

    result = select_champions.select(
        results_root=results_root,
        select_attempt_id="S1",
    )
    report = result["report"]
    latest = json.loads((results_root / "04_select" / "latest.json").read_text())
    assert report["champion_kinds"] == ["energy"]
    assert latest["attempt_id"] == "S1"
    assert [spec["selector"] for spec in report["champion_specs"]] == ["metric_ladder"]
    assert report["group_by"] == ["basis", "activation"]
    assert report["n_champions"] == 4

    champions = _read_csv(results_root / "04_select" / "S1" / "champions.csv")
    major_counter = Counter((row["basis"], row["activation"]) for row in champions)
    assert len(major_counter) == 4
    assert set(major_counter.values()) == {1}
    assert {row["winner_kind"] for row in champions} == {"energy"}
    assert {row["minor_id"] for row in champions} == {"lr-3e-4_ch-8"}
    true_grid = OmegaConf.load(_write_grid(tmp_path))
    assert not ({row["basis"] for row in champions} & set(true_grid.major_grid.basis))
    assert not ({row["activation"] for row in champions} & set(true_grid.major_grid.activation))
    assert {row["basis"][0] for row in champions} == {"B"}
    assert {row["activation"][0] for row in champions} == {"A"}

    code = final_plan.main(
        [
            "--results-root",
            str(results_root),
            "--attempt-id",
            "F1",
        ]
    )
    assert code == 0

    final_dir = results_root / "05_final_grid" / "F1"
    manifest = json.loads((final_dir / "manifest.json").read_text())
    jobs = [json.loads(path.read_text()) for path in sorted((final_dir / "jobs").glob("*.json"))]
    assert manifest["study"] == "tpen_pair_scan_v1"
    assert manifest["final_replicates"] == 9
    assert manifest["n_jobs"] == 36
    assert manifest["axis_overrides"] == {
        "basis": "run_parameters.basis_slot",
        "activation": "run_parameters.activation_slot",
        "lr": "run_parameters.lr",
        "channels": "run_parameters.channels",
    }
    assert len(jobs) == 36
    assert set(Counter(job["source_champion_id"] for job in jobs).values()) == {9}
    assert {int(job["replicate_index"]) for job in jobs} == set(range(9))
    assert {job["final_train_model_seed"] for job in jobs} == set(range(100, 109))
    assert {job["final_train_sampler_seed"] for job in jobs} == set(range(1000, 1009))
    assert {job["final_eval_sampler_seed"] for job in jobs} == set(range(10000, 10009))

    code = final_plan.main(
        [
            "--results-root",
            str(results_root),
            "--attempt-id",
            "F2",
            "--replicates",
            "1",
            "--limit-champions",
            "1",
        ]
    )
    assert code == 0
    final_job = json.loads(next((results_root / "05_final_grid" / "F2" / "jobs").glob("*.json")).read_text())
    assert final_job["basis"].startswith("B")
    assert final_job["activation"].startswith("A")
    assert final_job["choices"]["basis"] == final_job["basis"]
    assert final_job["choices"]["activation"] == final_job["activation"]
    assert final_job["basis"] not in set(true_grid.major_grid.basis)
    assert final_job["activation"] not in set(true_grid.major_grid.activation)


def test_v3_final_plan_chains_task_lineage_from_selection_sidecar(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    _write_collection_summary(results_root)
    manifest = json.loads((results_root / "00_grid" / ATTEMPT / "manifest.json").read_text())
    write_task_lineage(
        results_root / "03_collect" / "C1",
        [
            TaskLineageRow(
                row_id=str(job["run_id"]),
                task_ids={
                    "validation": f"02_validation:{job['run_id']}:V1",
                    "train": f"01_train:{job['run_id']}:T1",
                },
            )
            for job in manifest["jobs"]
        ],
    )
    select_champions.select(results_root=results_root, select_attempt_id="S1")

    code = final_plan.main(
        [
            "--results-root",
            str(results_root),
            "--selection-attempt-id",
            "S1",
            "--attempt-id",
            "F1",
        ]
    )
    assert code == 0

    jobs = [json.loads(path.read_text()) for path in (results_root / "05_final_grid" / "F1" / "jobs").glob("*.json")]
    lineage = read_task_lineage(results_root / "05_final_grid" / "F1")
    assert jobs
    for job in jobs:
        contributing_run_ids = [run_id for run_id in str(job["source_scan_run_ids"]).split(";") if run_id]
        assert contributing_run_ids
        expected_validation = {f"02_validation:{run_id}:V1" for run_id in contributing_run_ids}
        expected_train = {f"01_train:{run_id}:T1" for run_id in contributing_run_ids}
        row = lineage[job["final_run_id"]]
        assert set(row.task_ids["validation"]) == expected_validation
        assert set(row.task_ids["train"]) == expected_train


def test_v2_final_plan_rejects_zero_configured_replicates_without_override(tmp_path: Path) -> None:
    results_root = _planned_results(tmp_path)
    _write_collection_summary(results_root)
    select_champions.select(results_root=results_root, select_attempt_id="S1")

    manifest_path = results_root / "00_grid" / ATTEMPT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["final_replicates"] = 0
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(ValueError, match="final_replicates must be >= 1"):
        final_plan.main(
            [
                "--results-root",
                str(results_root),
                "--selection-attempt-id",
                "S1",
                "--attempt-id",
                "F0",
            ]
        )

    code = final_plan.main(
        [
            "--results-root",
            str(results_root),
            "--selection-attempt-id",
            "S1",
            "--attempt-id",
            "F1",
            "--replicates",
            "1",
            "--limit-champions",
            "1",
        ]
    )
    assert code == 0
    planned = _read_csv(results_root / "05_final_grid" / "F1" / "final_jobs.csv")
    assert len(planned) == 1


def test_v2_final_train_rejects_empty_final_grid(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    attempt = results_root / "05_final_grid" / "F0"
    attempt.mkdir(parents=True)
    (attempt / "final_jobs.csv").write_text("final_run_id\n", encoding="utf-8")
    json_io.write_json(
        attempt / "manifest.json",
        {
            "study": "tpen_pair_scan_v1",
            "stage": layout.STAGE_FINAL_GRID,
            "attempt_id": "F0",
            "train_config": str(CONFIGS / "train.yaml"),
            "major_axes": [],
            "minor_axes": [],
            "axis_overrides": {},
        },
    )

    with pytest.raises(ValueError, match="final grid attempt F0 has no jobs"):
        final_train.main(
            [
                "--results-root",
                str(results_root),
                "--final-grid-attempt-id",
                "F0",
                "--backend",
                "local",
            ]
        )


def test_v2_final_train_excludes_completed_and_resumes_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    final_grid_id = "F0"
    final_grid_dir = results_root / "05_final_grid" / final_grid_id
    final_grid_dir.mkdir(parents=True)
    _write_csv(
        final_grid_dir / "final_jobs.csv",
        [
            {
                "final_run_id": "done",
                "source_champion_id": "champion-0",
                "final_train_model_seed": 1001,
                "final_train_sampler_seed": 101,
            },
            {
                "final_run_id": "partial",
                "source_champion_id": "champion-1",
                "final_train_model_seed": 1002,
                "final_train_sampler_seed": 102,
            },
        ],
    )
    json_io.write_json(
        final_grid_dir / "manifest.json",
        {
            "study": "tpen_pair_scan_v1",
            "stage": layout.STAGE_FINAL_GRID,
            "attempt_id": final_grid_id,
            "train_config": str(CONFIGS / "train.yaml"),
            "major_axes": [],
            "minor_axes": [],
            "axis_overrides": {},
        },
    )

    done_attempt = layout.final_train_attempt_dir(results_root, "done", final_grid_id)
    _write_final_checkpoint(results_root, "done", final_grid_id)
    (done_attempt / "status.json").write_text(json.dumps({"status": "completed"}) + "\n")

    partial_attempt = layout.final_train_attempt_dir(results_root, "partial", final_grid_id)
    checkpoint = partial_attempt / "checkpoints" / "step_000003"
    checkpoint.mkdir(parents=True)
    (checkpoint / "COMPLETE").write_text("")
    (checkpoint / "manifest.json").write_text(json.dumps({"step": 3}) + "\n")

    captured: dict[str, Any] = {}

    def fake_submit_command_sets(command_sets: dict[str, list[list[str]]], **kwargs: Any) -> list[str]:
        captured["command_sets"] = command_sets
        captured["kwargs"] = kwargs
        return ["job-0"]

    monkeypatch.setattr(final_train.launch, "submit_command_sets", fake_submit_command_sets)

    code = final_train.main(
        [
            "--results-root",
            str(results_root),
            "--final-grid-attempt-id",
            final_grid_id,
            "--backend",
            "local",
            "--device",
            "cpu",
        ]
    )

    assert code == 0
    cpu_commands = captured["command_sets"]["cpu"]
    assert len(cpu_commands) == 1
    script = cpu_commands[0][-1]
    assert "run.run_id=partial/F0" in script
    assert "run.run_id=done/F0" not in script
    assert f"load.path={checkpoint}" in script
    assert "load.mode=train_resume" in script


def test_v2_final_stage_defaults_use_latest_pointers(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    final_grid_stage = results_root / "05_final_grid"
    (final_grid_stage / "zzz").mkdir(parents=True)
    (final_grid_stage / "aaa").mkdir()
    layout.write_latest(final_grid_stage, "aaa")

    assert final_train._resolve_final_grid_attempt_id(results_root, None) == "aaa"
    assert final_eval._resolve_final_grid_attempt_id(results_root, None) == "aaa"

    (final_grid_stage / "diagnostic-final-grid").mkdir()
    layout.write_latest(final_grid_stage, "diagnostic-final-grid")
    assert final_train._resolve_final_grid_attempt_id(results_root, None) == "diagnostic-final-grid"
    assert final_eval._resolve_final_grid_attempt_id(results_root, None) == "diagnostic-final-grid"

    final_run_id = "final-run-0"
    _write_final_checkpoint(results_root, final_run_id, "zzz")
    _write_final_checkpoint(results_root, final_run_id, "aaa")
    layout.write_latest(layout.final_train_run_dir(results_root, final_run_id), "aaa")

    assert final_eval.latest_final_train_attempt_id(results_root, final_run_id) == "aaa"
    assert final_eval._latest_ready_final_train_attempt_id(results_root, final_run_id) == "aaa"

    eval_run_dir = layout.final_eval_run_dir(results_root, final_run_id)
    (eval_run_dir / "zzz").mkdir(parents=True)
    (eval_run_dir / "aaa").mkdir()
    layout.write_latest(eval_run_dir, "aaa")

    _write_csv(
        final_grid_stage / "diagnostic-final-grid" / "final_jobs.csv",
        [{"final_run_id": final_run_id}],
    )
    _write_json(
        eval_run_dir / "aaa" / "source_final_grid_attempt.json",
        {
            "final_grid_attempt_id": "diagnostic-final-grid",
            "final_grid_attempt_dir": str(final_grid_stage / "diagnostic-final-grid"),
        },
    )

    assert final_collect._iter_final_eval_attempts(
        results_root,
        None,
        "diagnostic-final-grid",
    ) == [
        (final_run_id, "aaa", eval_run_dir / "aaa")
    ]


def test_final_eval_enumeration_skips_reserved_stage_dirs(tmp_path: Path) -> None:
    """Reserved stage subdirectories are never collected as final-eval runs.

    ``stage_plans`` (toolkit stage plans), ``slurm_logs`` and ``chunk_status``
    live alongside per-run directories under the final-eval stage. With an
    explicit ``final_eval_attempt_id`` the enumeration bypasses the
    latest-pointer guard, so a reserved directory that happens to hold a
    matching attempt subdirectory must still be excluded rather than surface a
    phantom run with empty metrics.
    """

    results_root = tmp_path / "results"
    attempt_id = "A0"
    eval_stage = layout.stage_dir(results_root, layout.STAGE_FINAL_EVAL)
    final_grid_id = "FG0"
    final_grid_dir = layout.final_grid_attempt_dir(results_root, final_grid_id)
    _write_csv(final_grid_dir / "final_jobs.csv", [{"final_run_id": "final-run-0"}])

    run_dir = eval_stage / "final-run-0"
    (run_dir / attempt_id).mkdir(parents=True)
    _write_json(
        run_dir / attempt_id / "source_final_grid_attempt.json",
        {
            "final_grid_attempt_id": final_grid_id,
            "final_grid_attempt_dir": str(final_grid_dir),
        },
    )
    for reserved in ("stage_plans", "slurm_logs", "chunk_status"):
        (eval_stage / reserved / attempt_id).mkdir(parents=True)

    assert final_collect._iter_final_eval_attempts(results_root, attempt_id, final_grid_id) == [
        ("final-run-0", attempt_id, run_dir / attempt_id)
    ]


def test_final_collect_defaults_to_latest_final_grid_plan(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    old_grid_id = "FG-old"
    latest_grid_id = "FG-latest"
    old_run_id = "old-final-run"
    latest_run_id = "latest-final-run"

    for grid_id, run_id in ((old_grid_id, old_run_id), (latest_grid_id, latest_run_id)):
        grid_dir = layout.final_grid_attempt_dir(results_root, grid_id)
        _write_csv(grid_dir / "final_jobs.csv", [{"final_run_id": run_id}])
        _write_json(
            grid_dir / "manifest.json",
            {
                "study": "tpen_pair_scan_v1",
                "stage": layout.STAGE_FINAL_GRID,
                "attempt_id": grid_id,
                "major_axes": [],
                "minor_axes": [],
                "final_replicates": 1,
            },
        )
        eval_dir = layout.final_eval_attempt_dir(results_root, run_id, "FE0")
        _write_json(
            eval_dir / "source_final_grid_attempt.json",
            {
                "final_grid_attempt_id": grid_id,
                "final_grid_attempt_dir": str(grid_dir),
            },
        )
        layout.write_latest(layout.final_eval_run_dir(results_root, run_id), "FE0")
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_FINAL_GRID), latest_grid_id)

    result = final_collect.collect_final_outputs(
        results_root=results_root,
        collect_attempt_id="FC0",
    )

    assert result["manifest"]["final_grid_attempt_id"] == latest_grid_id
    assert result["manifest"]["n_final_eval_attempts"] == 1
    assert _read_csv(Path(result["attempt_dir"]) / "run_index.csv")[0]["final_run_id"] == latest_run_id

    parallel = final_collect.collect_final_outputs(
        results_root=results_root,
        collect_attempt_id="FC1",
    )
    assert parallel["manifest"]["final_grid_attempt_id"] == latest_grid_id
    assert (results_root / "08_final_collect" / "FC0" / "run_index.csv").is_file()
    assert (results_root / "08_final_collect" / "FC1" / "run_index.csv").is_file()


def test_v3_final_collect_report_axes_follow_planned_major_axes() -> None:
    manifest = {
        "major_axes": ["basis", "activation"],
        "minor_axes": ["lr", "channels"],
    }
    job = {
        "choices": {
            "basis": "hooke-total-shell",
            "activation": "SiLU",
            "lr": "1.0e-3",
            "channels": 8,
        }
    }

    assert final_collect._report_axes(manifest) == ("basis", "activation")
    assert final_collect._report_axis_values(job, manifest) == ("hooke-total-shell", "SiLU")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _append_metrics(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _write_minimal_final_artifacts(
    results_root: Path,
    *,
    n_final_runs: int = 1,
    write_transform_records: bool = True,
    extra_eval_metrics: Sequence[dict[str, Any]] = (),
) -> tuple[str, list[str]]:
    """Write the smallest final-stage artifact tree ``final_collect`` accepts.

    Parameters
    ----------
    n_final_runs
        Number of planned final runs (seed replicates) to materialize. Seed
        aggregations such as ``antisymmetry_sign_mismatch_total`` are only
        meaningfully exercised with more than one row.
    write_transform_records
        When ``False``, omit every symmetry task's ``transform_records.csv``.
        This is what the scan's own `eval.yaml` produces, because it runs at
        ``artifact_level: summaries`` and per-sample transform records are only
        written at ``artifact_level: records``.
    extra_eval_metrics
        Extra ``metrics.jsonl`` records appended to every final-eval attempt,
        used to inject summary-only task metrics.
    """

    final_grid_attempt_id = "FG0"
    final_train_attempt_id = "FT0"
    final_eval_attempt_id = "FE0"
    final_run_ids = [f"final-run-{index}" for index in range(n_final_runs)]
    final_grid_dir = layout.stage_dir(results_root, layout.STAGE_FINAL_GRID) / final_grid_attempt_id

    _write_json(
        final_grid_dir / "manifest.json",
        {
            "study": "tpen_pair_scan_v1",
            "stage": layout.STAGE_FINAL_GRID,
            "attempt_id": final_grid_attempt_id,
            "major_axes": ["basis", "activation"],
            "minor_axes": ["lr", "channels"],
            "final_replicates": n_final_runs,
        },
    )
    jobs = [
        {
            "final_run_id": final_run_id,
            "source_champion_id": "champion-0",
            "winner_kind": "energy",
            "replicate_index": index,
            "major_id": "b-B00_act-A00",
            "minor_id": "lr-1e-3_ch-8",
            "config_id": "b-B00_act-A00_lr-1e-3_ch-8",
            "choices": {
                "basis": "hooke-total-shell",
                "activation": "SiLU",
                "lr": "1.0e-3",
                "channels": 8,
            },
            "final_train_model_seed": 100 + index,
            "final_train_sampler_seed": 1000 + index,
            "final_eval_sampler_seed": 10000 + index,
        }
        for index, final_run_id in enumerate(final_run_ids)
    ]
    _write_csv(final_grid_dir / "final_jobs.csv", jobs)
    layout.write_latest(layout.stage_dir(results_root, layout.STAGE_FINAL_GRID), final_grid_attempt_id)

    for job in jobs:
        final_run_id = str(job["final_run_id"])
        final_train_dir = layout.final_train_attempt_dir(results_root, final_run_id, final_train_attempt_id)
        final_eval_dir = layout.final_eval_attempt_dir(results_root, final_run_id, final_eval_attempt_id)
        _write_json(final_eval_dir / "source_final_job.json", job)
        _write_json(
            final_eval_dir / "source_final_grid_attempt.json",
            {
                "final_grid_attempt_id": final_grid_attempt_id,
                "final_grid_attempt_dir": str(final_grid_dir),
            },
        )
        _write_json(final_eval_dir / "evaluated_checkpoint.json", {"resolved_checkpoint_dir": str(final_train_dir / "checkpoints" / "step_000000")})
        _write_json(
            final_eval_dir / "source_final_train_attempt.json",
            {
                "final_train_attempt_id": final_train_attempt_id,
                "final_train_attempt_dir": str(final_train_dir),
            },
        )
        _write_json(final_eval_dir / "status.json", {"status": "completed", "start_time": "2026-07-08T00:00:00+00:00", "end_time": "2026-07-08T00:00:10+00:00"})
        _write_json(final_train_dir / "status.json", {"status": "completed", "start_time": "2026-07-08T00:00:00+00:00", "end_time": "2026-07-08T00:00:05+00:00"})
        _write_json(final_train_dir / "metadata.json", {"runtime": {"device": "cuda"}, "peak_memory_mb": 123})
        _append_metrics(
            final_train_dir / "metrics.jsonl",
            [
                {"namespace": "train", "step": 0, "metrics": {"energy": 2.1, "energy_stderr": 0.01, "energy_variance": 0.02, "grad_norm": 1.5}},
                {"namespace": "train/sampler", "step": 0, "metrics": {"acceptance_rate": 0.7}},
                {"namespace": "train/perf", "step": 0, "metrics": {"step_time_sec": 1.0, "local_energy_time_sec": 0.2, "forward_time_sec": 0.3, "backward_time_sec": 0.4}},
                {"namespace": "runtime", "step": 0, "metrics": {"wall_time_sec": 5.0, "peak_memory_mb": 123}},
                {"namespace": "debug/unrelated", "step": 0, "metrics": {"ignored": 999}},
            ],
        )
        _append_metrics(
            final_eval_dir / "metrics.jsonl",
            [
                {"namespace": "eval/mcmc_energy", "step": 0, "metrics": {"local_energy_mean": 2.01, "local_energy_stderr": 0.02, "local_energy_variance": 0.03, "local_energy_n_finite": 2, "local_energy_n_total": 2, "local_energy_finite_fraction": 1.0, "local_energy_pathology_count": 0}},
                {"namespace": "eval/mcmc_energy/term", "step": 0, "metrics": {"kinetic_mean": 1.0, "harmonic_trap_mean": 0.5, "electron_electron_mean": 1.0}},
                {"namespace": "eval/stratified_geometry/status", "step": 0, "metrics": {"task_success": True, "task_failed": False}},
                {"namespace": "diagnostics/cusp", "step": 0, "metrics": {"time_sec": 1.0}},
                {"namespace": "eval/perf/cusp", "step": 0, "metrics": {"generator_time_sec": 0.1, "calculator/local_energy_time_sec": 0.2, "summary/profile_time_sec": 0.3}},
                {"namespace": "debug/unrelated", "step": 0, "metrics": {"ignored": 999}},
                *extra_eval_metrics,
            ],
        )

        _write_csv(final_eval_dir / "mcmc_energy" / "mcmc_energy_samples.csv", [{"local_energy": 2.0}, {"local_energy": 2.1}])
        _write_csv(final_eval_dir / "cusp" / "cusp_profiles.csv", [{"center_of_mass_id": "com0", "direction_id": "dir0", "r12": 0.1, "local_energy": 2.0, "logabs": -0.2, "d_logabs_dr": 0.5, "finite": True}])
        _write_csv(final_eval_dir / "tail" / "tail_profiles.csv", [{"com_id": "com0", "tail_path": "path0", "radius": 4.0, "local_energy": 2.0, "logabs": -4.0, "exact_logabs": -4.1, "finite": True}])
        _write_csv(final_eval_dir / "stratified_geometry" / "stratified_metrics.csv", [{"stratum": "near", "local_energy": 2.0, "finite": True}])
        _write_csv(final_eval_dir / "hooke_orbital" / "hooke_orbital_metrics.csv", [{"radius": 1.0, "r12": 0.5, "local_energy": 2.0, "finite": True}])
        if write_transform_records:
            for task in final_collect.SYMMETRY_TASKS:
                _write_csv(final_eval_dir / task / "transform_records.csv", [{"logabs_abs_error": 0.0, "sign_mismatch": False, "parity_mismatch": False, "finite": True}])
        for task in final_collect.TRACE_TASKS:
            _write_csv(final_eval_dir / task / "trace_records.csv", [{"entry_key": "layers.0.mixing/value", "q95_abs": 0.01, "q99_abs": 0.02, "max_abs_error": 0.03, "nonfinite_count": 0, "compared_entry_count": 4, "comparison_error_count": 0}])
        (final_eval_dir / "unused_task").mkdir()
        (final_eval_dir / "unused_task" / "all_metrics_dump.csv").write_text("not,a,real,table\n")
        layout.write_latest(layout.final_eval_run_dir(results_root, final_run_id), final_eval_attempt_id)
    return final_eval_attempt_id, final_run_ids


def test_v3_final_collect_writes_report_contract_and_explicit_report_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_root = tmp_path / "results"
    final_eval_attempt_id, _final_run_ids = _write_minimal_final_artifacts(results_root)
    read_paths: list[Path] = []
    real_read_csv = final_collect._read_csv

    def tracked_read_csv(path: Path) -> list[dict[str, Any]]:
        read_paths.append(Path(path))
        return real_read_csv(path)

    monkeypatch.setattr(final_collect, "_read_csv", tracked_read_csv)

    result = final_collect.collect_final_outputs(
        results_root=results_root,
        collect_attempt_id="FC0",
        final_eval_attempt_id=final_eval_attempt_id,
    )
    collect_dir = Path(result["attempt_dir"])
    manifest = result["manifest"]
    run_index = _read_csv(collect_dir / "run_index.csv")
    architecture = _read_csv(collect_dir / "architecture_summary.csv")

    assert tuple(manifest["tables"]) == final_collect.COMPACT_TABLES
    assert manifest["report_row_key"] == "basis"
    assert manifest["report_col_key"] == "activation"
    assert manifest["major_axes"] == ["basis", "activation"]
    assert run_index[0]["basis"] == "hooke-total-shell"
    assert run_index[0]["activation"] == "SiLU"
    assert run_index[0][final_collect.REPORT_ROW_COLUMN] == "hooke-total-shell"
    assert run_index[0][final_collect.REPORT_COL_COLUMN] == "SiLU"
    assert architecture[0][final_collect.REPORT_ROW_COLUMN] == "hooke-total-shell"
    assert architecture[0][final_collect.REPORT_COL_COLUMN] == "SiLU"
    assert all("unused_task" not in str(path) for path in read_paths)


# ---------------------------------------------------------------------------
# Antisymmetry at `artifact_level: summaries`
#
# The scan's `configs/eval.yaml` sets `artifact_level: summaries`, so
# `full_model_antisymmetry/transform_records.csv` is NEVER written. Antisymmetry
# is a core correctness invariant of a fermionic ansatz, so a collector that
# reads only those records reports a blank `antisymmetry_logabs_error_max` and a
# `antisymmetry_sign_mismatch_total` summed over zero rows -- indistinguishable
# from "measured, no violations". These tests pin the summary-metric path.
# ---------------------------------------------------------------------------
def _antisymmetry_summary_metrics(
    *, logabs_max_abs_error: float, sign_failure_count: int
) -> list[dict[str, Any]]:
    """Return the metrics `TransformConsistencySummary` writes for the task.

    Mirrors `tpen.evaluation.summaries.TransformConsistencySummary` under the
    task namespace the scan's `configs/eval.yaml` assigns
    (`eval/full_model_antisymmetry`). These four scalars are the whole surviving
    record of the invariant at `artifact_level: summaries`.
    """

    return [
        {
            "namespace": "eval/full_model_antisymmetry",
            "step": 0,
            "metrics": {
                "logabs_max_abs_error": logabs_max_abs_error,
                "logabs_mean_abs_error": logabs_max_abs_error / 2.0,
                "sign_failure_count": sign_failure_count,
                "failure_count": sign_failure_count,
            },
        },
        {
            "namespace": "eval/full_model_antisymmetry/status",
            "step": 0,
            "metrics": {"task_success": True, "task_failed": False},
        },
    ]


def test_antisymmetry_summary_metric_survives_the_collector_metric_filter() -> None:
    # `_keep_eval_metric` drops every eval metric not explicitly retained, so
    # the fallback is only reachable if the metric is whitelisted. `sign_failure_count`
    # rides the `failure_count` needle; `logabs_max_abs_error` matches no needle
    # and must be named exactly, like its trace-equivariance counterpart.
    assert final_collect._keep_eval_metric("eval/full_model_antisymmetry/logabs_max_abs_error")
    assert final_collect._keep_eval_metric("eval/full_model_antisymmetry/sign_failure_count")


def test_final_collect_reports_antisymmetry_from_summaries_when_records_are_absent(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    final_eval_attempt_id, final_run_ids = _write_minimal_final_artifacts(
        results_root,
        n_final_runs=2,
        write_transform_records=False,
        extra_eval_metrics=_antisymmetry_summary_metrics(
            logabs_max_abs_error=0.0, sign_failure_count=0
        ),
    )
    result = final_collect.collect_final_outputs(
        results_root=results_root,
        collect_attempt_id="FC0",
        final_eval_attempt_id=final_eval_attempt_id,
    )
    symmetry = _read_csv(Path(result["attempt_dir"]) / "symmetry_summary.csv")

    # One row per final run, not zero rows: the table is the report's only input.
    assert result["manifest"]["tables"]["symmetry_summary.csv"] == len(final_run_ids)
    assert [row["final_run_id"] for row in symmetry] == final_run_ids
    assert all(row["symmetry_task"] == "full_model_antisymmetry" for row in symmetry)
    # Populated, not blank -- a blank field is what the defect produced.
    assert all(row["logabs_error_max"] != "" for row in symmetry)
    assert all(float(row["logabs_error_max"]) == 0.0 for row in symmetry)
    assert all(row["sign_mismatch_count"] == "0" for row in symmetry)


def test_a_nonzero_antisymmetry_violation_reaches_the_rendered_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # THE anti-vacuity test. Asserting that "0.0" appears would also pass while
    # the collector measures nothing, so inject a violation the report must not
    # be able to swallow, and follow it all the way to `report.md`.
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    results_root = tmp_path / "results"
    final_eval_attempt_id, _final_run_ids = _write_minimal_final_artifacts(
        results_root,
        write_transform_records=False,
        extra_eval_metrics=_antisymmetry_summary_metrics(
            logabs_max_abs_error=0.375, sign_failure_count=7
        ),
    )
    collected = final_collect.collect_final_outputs(
        results_root=results_root,
        collect_attempt_id="FC0",
        final_eval_attempt_id=final_eval_attempt_id,
    )
    symmetry = _read_csv(Path(collected["attempt_dir"]) / "symmetry_summary.csv")
    assert [float(row["logabs_error_max"]) for row in symmetry] == [0.375]
    assert [row["sign_mismatch_count"] for row in symmetry] == ["7"]

    report = final_report.build_report(results_root=results_root, attempt_id=ATTEMPT)
    report_dir = layout.stage_dir(results_root, layout.STAGE_FINAL_REPORT) / ATTEMPT
    invariants = _read_csv(report_dir / "tables" / "invariants_by_axis.csv")
    assert [row["antisymmetry_logabs_error_max"] for row in invariants] == ["0.375"]
    assert [row["antisymmetry_sign_mismatch_total"] for row in invariants] == ["7"]
    assert "0.375" in (report_dir / "report.md").read_text()
    assert report["warnings"] == []


def test_antisymmetry_sign_mismatch_total_sums_over_every_collected_seed(
    tmp_path: Path,
) -> None:
    # `antisymmetry_sign_mismatch_total` is a sum, so an empty row set and a
    # clean run both render 0. Pin the row count the sum is taken over.
    results_root = tmp_path / "results"
    final_eval_attempt_id, final_run_ids = _write_minimal_final_artifacts(
        results_root,
        n_final_runs=3,
        write_transform_records=False,
        extra_eval_metrics=_antisymmetry_summary_metrics(
            logabs_max_abs_error=0.125, sign_failure_count=2
        ),
    )
    collected = final_collect.collect_final_outputs(
        results_root=results_root,
        collect_attempt_id="FC0",
        final_eval_attempt_id=final_eval_attempt_id,
    )
    symmetry = _read_csv(Path(collected["attempt_dir"]) / "symmetry_summary.csv")
    assert len(symmetry) == len(final_run_ids) == 3

    invariants = final_report.invariants_by_axis_rows(symmetry, [])
    assert len(invariants) == 1
    assert invariants[0]["antisymmetry_sign_mismatch_total"] == 3 * 2
    # The worst case survives the reduction rather than being averaged away.
    assert invariants[0]["antisymmetry_logabs_error_max"] == "0.125"


def test_transform_records_still_win_over_summary_metrics_when_present(
    tmp_path: Path,
) -> None:
    # `artifact_level: records` remains supported: per-sample records are richer
    # (median, parity, finite fraction), so they must not be shadowed by the
    # coarser fallback.
    results_root = tmp_path / "results"
    final_eval_attempt_id, _final_run_ids = _write_minimal_final_artifacts(
        results_root,
        write_transform_records=True,
        extra_eval_metrics=_antisymmetry_summary_metrics(
            logabs_max_abs_error=0.5, sign_failure_count=9
        ),
    )
    collected = final_collect.collect_final_outputs(
        results_root=results_root,
        collect_attempt_id="FC0",
        final_eval_attempt_id=final_eval_attempt_id,
    )
    symmetry = _read_csv(Path(collected["attempt_dir"]) / "symmetry_summary.csv")
    assert len(symmetry) == 1
    assert float(symmetry[0]["logabs_error_max"]) == 0.0
    assert symmetry[0]["logabs_error_median"] == "0"
    assert symmetry[0]["finite_fraction"] == "1"


def test_v2_final_eval_defaults_to_single_latest_final_train_attempt(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    final_run_id = "final-run-0"
    final_grid_attempt_id = "FG0"
    diagnostic_attempt = "diagnostic-final-train"
    _write_final_checkpoint(results_root, final_run_id, final_grid_attempt_id)
    _write_final_checkpoint(results_root, final_run_id, diagnostic_attempt)
    layout.write_latest(layout.final_train_run_dir(results_root, final_run_id), diagnostic_attempt)

    assert final_eval.latest_final_train_attempt_id(results_root, final_run_id) == diagnostic_attempt
    assert final_eval._latest_ready_final_train_attempt_id(results_root, final_run_id) == diagnostic_attempt

    args = types.SimpleNamespace(
        smoke=False,
        final_train_attempt_id=final_grid_attempt_id,
    )
    assert (
        final_eval._final_train_attempt_id_for_job(
            args=args,
            results_root=results_root,
            final_run_id=final_run_id,
        )
        == final_grid_attempt_id
    )


def _callback_entries(config_name: str) -> list[dict[str, Any]]:
    config = OmegaConf.to_container(OmegaConf.load(CONFIGS / config_name), resolve=False)
    assert isinstance(config, dict)
    callbacks = config.get("callbacks")
    assert isinstance(callbacks, list)
    return [entry for entry in callbacks if isinstance(entry, dict)]


def _callback_targets(entries: list[dict[str, Any]]) -> set[str]:
    return {str(entry.get("_target_", "")) for entry in entries}


def test_train_config_wires_profiling_callbacks() -> None:
    entries = _callback_entries("train.yaml")
    targets = _callback_targets(entries)

    assert "tpen.callback.RunTiming" in targets
    assert "tpen.callback.TrainStepTiming" in targets
    assert "tpen.callback.TrainPhaseTiming" in targets
    phase_timing = next(entry for entry in entries if entry["_target_"] == "tpen.callback.TrainPhaseTiming")
    assert "triggers" not in phase_timing


def test_train_config_writes_only_the_terminal_checkpoint() -> None:
    entries = _callback_entries("train.yaml")
    checkpoints = [entry for entry in entries if entry["_target_"] == "tpen.callback.Checkpoint"]

    # One composed entry owns both writes. Its explicit TerminalOnly schedule
    # leaves periodic writes off and always admits the terminal boundary.
    assert len(checkpoints) == 1
    checkpoint = checkpoints[0]
    assert checkpoint["periodic"] == "${checkpoint.periodic}"
    assert checkpoint["terminal"] == "${checkpoint.terminal}"
    assert checkpoint["schedule"] == "${checkpoint.schedule}"
    assert checkpoint["payload"] == "${checkpoint.payload}"
    assert checkpoint["keep_last"] == "${checkpoint.keep_last}"
    assert "every_n_steps" not in checkpoint

    config = OmegaConf.load(CONFIGS / "train.yaml")
    assert config.checkpoint.periodic is False
    assert config.checkpoint.terminal is True
    assert config.checkpoint.schedule["_target_"] == "tpen.checkpoint.TerminalOnly"
    assert config.checkpoint.payload["_target_"] == "tpen.checkpoint.TrainResume"


def test_validation_config_wires_profiling_callbacks() -> None:
    entries = _callback_entries("eval.yaml")
    targets = _callback_targets(entries)

    assert "tpen.callback.EvaluationTiming" in targets
    assert "tpen.callback.EvaluationComponentTiming" in targets
    evaluation_timing = next(
        entry for entry in entries if entry["_target_"] == "tpen.callback.EvaluationTiming"
    )
    assert "triggers" not in evaluation_timing
    component_timing = next(
        entry for entry in entries if entry["_target_"] == "tpen.callback.EvaluationComponentTiming"
    )
    assert "triggers" not in component_timing
