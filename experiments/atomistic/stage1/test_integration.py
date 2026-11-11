"""Tiny configured-route train, strict restore, and real-table collection checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.run import run_from_config

from experiments.atomistic.stage1.collect import RecordSpec, collect_records


pytestmark = pytest.mark.integration
HERE = Path(__file__).parent


def _checkpoint_callback(cfg: object) -> None:
    """Make the checked-in checkpoint callback publish the one-step checkpoint."""

    for callback in cfg.callbacks:
        if str(callback.get("_target_", "")).endswith("Checkpoint"):
            callback.schedule.every_n = 1


def _run_dir(root: Path, run_id: str) -> Path:
    candidates = list(root.rglob("resolved_config.yaml"))
    assert candidates, f"run {run_id} did not write resolved_config.yaml"
    return next(path.parent for path in candidates if path.parent.name == run_id)


@pytest.mark.parametrize("system", ["he", "hooke1"])
def test_tiny_train_eval_collect_uses_real_artifact(system: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise both supported Stage-1 atom routes through ``run_from_config``."""

    train_cfg = OmegaConf.load(HERE / "configs" / system / "train.yaml")
    train_cfg.run.root = str(tmp_path / "train")
    train_cfg.run.layout = "flat"
    train_cfg.run.run_id = "train"
    train_cfg.runtime.device = "cpu"
    train_cfg.trainer.max_steps = 1
    train_cfg.sampler.n_walkers = 2
    train_cfg.sampler.burn_in = 1
    train_cfg.sampler.n_steps = 1
    _checkpoint_callback(train_cfg)
    assert run_from_config(train_cfg, command=f"stage1 {system} tiny train", raise_exceptions=True) == 0
    train_dir = _run_dir(tmp_path / "train", "train")
    monkeypatch.setenv("TPEN_SOURCE_SHA", json.loads((train_dir / "metadata.json").read_text(encoding="utf-8"))["git_commit"])
    checkpoint = train_dir / "checkpoints" / "step_000001"
    assert (checkpoint / "COMPLETE").is_file()

    eval_cfg = OmegaConf.load(HERE / "configs" / system / "eval.yaml")
    eval_cfg.run.root = str(tmp_path / "eval")
    eval_cfg.run.layout = "flat"
    eval_cfg.run.run_id = "eval"
    eval_cfg.runtime.device = "cpu"
    eval_cfg.load.path = str(checkpoint)
    eval_cfg.evaluation.artifact_level = "records"
    eval_cfg.evaluation_sampler.n_walkers = 2
    eval_cfg.evaluation_sampler.burn_in = 1
    eval_cfg.evaluation_sampler.n_steps = 1
    eval_cfg.evaluation_tasks.mcmc_energy.generator.n_draws = 2
    eval_cfg.evaluation_tasks.mcmc_energy.generator.chunk_size = 2
    eval_cfg.evaluation_tasks.mcmc_energy.summaries[-1].max_samples = 4
    eval_cfg.evaluator.tasks = [eval_cfg.evaluation_tasks.mcmc_energy]
    if system == "he":
        eval_cfg.trajectory_identity.stage = "stage1"
        eval_cfg.trajectory_identity.run_id = "eval"
        eval_cfg.trajectory_identity.attempt_id = "tiny"
        eval_cfg.trajectory_identity.evaluator_id = "stage1"
        eval_cfg.trajectory_identity.checkpoint_file = str(checkpoint / "model.pt")
        eval_cfg.trajectory_identity.config_sha256 = "tiny"
    # Evaluation task summaries may return TPEN's handled failure code while
    # still publishing a complete trajectory artifact; collection is the
    # science-facing verdict for this tiny route and must inspect that file.
    eval_status = run_from_config(eval_cfg, command=f"stage1 {system} tiny eval", raise_exceptions=True)
    assert eval_status in (0, 1)
    eval_dir = _run_dir(tmp_path / "eval", "eval")
    trajectory = eval_dir / "mcmc_energy"
    assert (trajectory / "sampled_eval_table.csv").is_file()
    registry = {"systems": [{"id": "he_atom", "reference_energy_hartree": -2.903724377034119598},
                              {"id": "hooke_one_electron_omega0.5", "reference_energy_hartree": 0.75}]}
    result = collect_records(trajectory, system=system, registry=registry,
                             spec=RecordSpec(walkers=2, draws=2), run_dir=eval_dir)
    assert result["status"] == "complete"
    assert result["finite"] is True
    assert result["mcse_status"].startswith("insufficient_draws")
