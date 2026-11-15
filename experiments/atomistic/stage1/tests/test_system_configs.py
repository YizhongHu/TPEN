"""Structural and descriptor-bound checks for the three Stage-1 configs."""

from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.run import load_config
from experiments.atomistic.stage1.protocol import physical_system_id_from_config


ROOT = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("system", ["hooke1", "he", "h2"])
def test_eval_config_selects_v2_descriptor_and_naive_evaluator(system: str) -> None:
    cfg = load_config(str(ROOT / system / "eval.yaml"))
    descriptor = instantiate(cfg.system_descriptor)
    assert len(descriptor.system_id()) == 64
    task = cfg.evaluation_tasks.mcmc_energy
    assert task.generator.system is not None
    assert task.generator.evaluator._target_ == "tpen.physics.hamiltonian.NaiveLocalEnergyEvaluator"
    assert OmegaConf.is_missing(cfg.trajectory_identity, "checkpoint_file")


@pytest.mark.parametrize("system", ["hooke1", "he", "h2"])
def test_planner_identity_is_descriptor_content_id(system: str) -> None:
    config = ROOT / system / "eval.yaml"
    assert physical_system_id_from_config(config) == instantiate(load_config(str(config)).system_descriptor).system_id()


@pytest.mark.parametrize("system", ["hooke1", "he", "h2"])
def test_train_config_declares_checkpoint_publication(system: str) -> None:
    cfg = load_config(str(ROOT / system / "train.yaml"))
    checkpoints = [entry for entry in cfg.callbacks
                   if entry.get("_target_") == "tpen.callback.Checkpoint"]
    assert len(checkpoints) == 1
    assert checkpoints[0].output_dir == "${run.dir}/checkpoints"
    assert checkpoints[0].schedule._target_ == "tpen.checkpoint.ExplicitUpdates"


@pytest.mark.parametrize("system", ["hooke1", "he"])
def test_accepted_variational_settings_match_train_and_eval(system: str) -> None:
    train = load_config(str(ROOT / system / "train.yaml"))
    eval_cfg = load_config(str(ROOT / system / "eval.yaml"))
    assert OmegaConf.to_container(train.model, resolve=True) == OmegaConf.to_container(
        eval_cfg.model, resolve=True
    )
    if system == "hooke1":
        assert train.model.factors[0]._target_ == "tpen.nn.GaussianConfinement"
        assert train.model.factors[0].coefficient == 0.25
        assert train.model.factors[0].trainable is False
    else:
        assert train.model.factors[0].trainable_range is True
        assert train.model.factors[1].law.curvature_coefficient == 0.01
        assert train.model.factors[1].law.curvature_range == 1.0
        assert train.model.factors[1].law.trainable is True
        assert train.model.readout.trainable is True
