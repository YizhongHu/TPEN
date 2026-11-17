"""Regression tests for cwd-independent Stage-1 gateway config resolution."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.atomistic.stage1 import eval as stage1_eval
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import render
from experiments.atomistic.stage1.protocol import (
    resolve_config_path,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = "/n/netscratch/kozinsky_lab/Lab/rhu/stage1"
BASE_ROW = {
    "config": "configs/he/train.yaml",
    "seed": 123,
    "updates": 5,
    "walkers": 2,
    "source_sha": "c" * 40,
    "config_sha256": "b" * 64,
    "results_root": RESULTS_ROOT,
    "experiment_name": "stage1",
    "sector": "he",
}


def _row(kind: str) -> dict[str, object]:
    """Return one small manifest row for a real rendered driver command."""

    row = {
        **BASE_ROW,
        "kind": kind,
        "row_id": f"he-{kind}-0",
        "run_id": f"{kind}-lineage-0",
    }
    if kind == "smoke":
        row.update(updates=3, walkers=2, checkpoint_steps=[3])
    elif kind == "eval":
        row.update(
            config="configs/he/eval.yaml",
            draws=2,
            checkpoint_dir=f"{RESULTS_ROOT}/stage1/he/train-lineage-0/checkpoints/step_000005",
            checkpoint_file=f"{RESULTS_ROOT}/stage1/he/train-lineage-0/checkpoints/step_000005/model.pt",
            checkpoint_sha256="d" * 64,
        )
    else:
        row["checkpoint_steps"] = [5]
    return row


def _rendered_argv(row: dict[str, object], driver: str, tmp_path: Path) -> list[str]:
    """Extract the command that launch.render places in the sbatch script."""

    script = render(
        row=row,
        driver=driver,
        python=sys.executable,
        head_sha=str(row["source_sha"]),
        results_root=RESULTS_ROOT,
        scratch_base=str(tmp_path / "scratch"),
    )
    return shlex.split(script.splitlines()[-1])


@pytest.mark.parametrize(
    ("kind", "driver"),
    [
        ("smoke", "experiments/atomistic/stage1/train.py"),
        ("train", "experiments/atomistic/stage1/train.py"),
        ("eval", "experiments/atomistic/stage1/eval.py"),
    ],
)
def test_rendered_driver_imports_experiments_outside_checkout(
    kind: str, driver: str, tmp_path: Path,
) -> None:
    """The real rendered command imports its source package from any cwd."""

    argv = _rendered_argv(_row(kind), driver, tmp_path)
    result = subprocess.run(
        [*argv, "--help"],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert argv[1] == str(REPO_ROOT / driver)


@pytest.mark.parametrize(
    ("kind", "driver"),
    [
        ("smoke", "experiments/atomistic/stage1/train.py"),
        ("train", "experiments/atomistic/stage1/train.py"),
        ("eval", "experiments/atomistic/stage1/eval.py"),
    ],
)
@pytest.mark.parametrize("submission_cwd", ["repo", "unrelated"])
def test_rendered_driver_resolves_config_from_any_submission_cwd(
    kind: str,
    driver: str,
    submission_cwd: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rendered command reaches load_config with its bounded row budget."""

    row = _row(kind)
    argv = _rendered_argv(row, driver, tmp_path)
    assert str(row["config"]) in argv

    module = stage1_eval if kind == "eval" else stage1_train
    validation_paths: list[str] = []
    load_paths: list[str] = []
    run_calls: list[tuple[object, dict[str, object]]] = []

    original_validate = module.validate_gateway_binding

    def spy_validate(*args: object, **kwargs: object) -> object:
        validation_paths.append(str(kwargs["config"]))
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(module, "validate_gateway_binding", spy_validate)

    import tpen.run

    original_load_config = tpen.run.load_config

    def spy_load_config(config: str, overrides: list[str] | None = None) -> object:
        load_paths.append(config)
        return original_load_config(config, overrides)

    monkeypatch.setattr(tpen.run, "load_config", spy_load_config)
    monkeypatch.setattr(
        module,
        "run_from_config",
        lambda cfg, **kwargs: (run_calls.append((cfg, kwargs)) or 0),
    )
    monkeypatch.chdir(REPO_ROOT if submission_cwd == "repo" else tmp_path)
    monkeypatch.setattr(sys, "argv", [argv[1], *argv[2:]])

    assert module.main() == 0

    expected_path = str(resolve_config_path(str(row["config"])))
    assert validation_paths == [expected_path]
    assert load_paths == [expected_path]
    assert run_calls[0][1]["config_path"] == expected_path
    cfg = run_calls[0][0]
    if kind == "eval":
        assert cfg.evaluation_sampler.n_walkers == row["walkers"]
        assert cfg.evaluation_tasks.mcmc_energy.generator.n_draws == row["draws"]
    else:
        assert cfg.sampler.n_walkers == row["walkers"]
        assert cfg.trainer.max_steps == row["updates"]
        assert list(cfg.checkpoint_steps) == list(row["checkpoint_steps"])


@pytest.mark.parametrize("submission_cwd", ["repo", "unrelated"])
def test_rendered_driver_refuses_missing_config_from_any_submission_cwd(
    submission_cwd: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing row config is rejected after the same path resolution step."""

    row = {**_row("smoke"), "config": "configs/he/does-not-exist.yaml"}
    argv = _rendered_argv(row, "experiments/atomistic/stage1/train.py", tmp_path)
    monkeypatch.chdir(REPO_ROOT if submission_cwd == "repo" else tmp_path)
    monkeypatch.setattr(sys, "argv", [argv[1], *argv[2:]])

    with pytest.raises(FileNotFoundError, match="config path does not exist"):
        stage1_train.main()
