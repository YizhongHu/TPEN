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
