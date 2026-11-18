"""Structural and descriptor-bound checks for the four Stage-1 configs."""

from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.run import load_config
from experiments.atomistic.stage1.protocol import SYSTEMS, physical_system_id_from_config
from experiments.atomistic.stage1.launch import row_overrides


ROOT = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("system", SYSTEMS)
def test_eval_config_selects_v2_descriptor_and_naive_evaluator(system: str) -> None:
    cfg = load_config(str(ROOT / system / "eval.yaml"))
    descriptor = instantiate(cfg.system_descriptor)
    assert len(descriptor.system_id()) == 64
    task = cfg.evaluation_tasks.mcmc_energy
    assert task.generator.system is not None
    assert task.generator.evaluator._target_ == "tpen.physics.hamiltonian.NaiveLocalEnergyEvaluator"
    assert OmegaConf.is_missing(cfg.trajectory_identity, "checkpoint_file")


@pytest.mark.parametrize("system", SYSTEMS)
def test_planner_identity_is_descriptor_content_id(system: str) -> None:
    config = ROOT / system / "eval.yaml"
    assert physical_system_id_from_config(config) == instantiate(load_config(str(config)).system_descriptor).system_id()


@pytest.mark.parametrize("system", SYSTEMS)
def test_train_config_declares_checkpoint_publication(system: str) -> None:
    cfg = load_config(str(ROOT / system / "train.yaml"))
    checkpoints = [entry for entry in cfg.callbacks
                   if entry.get("_target_") == "tpen.callback.Checkpoint"]
    assert len(checkpoints) == 1
    raw_callbacks = OmegaConf.to_container(cfg.callbacks, resolve=False)
    raw_checkpoint = next(entry for entry in raw_callbacks
                          if entry.get("_target_") == "tpen.callback.Checkpoint")
    assert raw_checkpoint["output_dir"] == "${run.dir}/checkpoints"
    assert checkpoints[0].schedule._target_ == "tpen.checkpoint.ExplicitUpdates"
    assert list(checkpoints[0].schedule.updates) == [1000]
    assert list(cfg.checkpoint_steps) == [1000]


@pytest.mark.parametrize("system", ["hooke1", "hooke2", "he"])
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
    elif system == "hooke2":
        assert train.model.factors[0]._target_ == "tpen.nn.GaussianConfinement"
        assert train.model.factors[0].coefficient == 0.25
        assert train.model.factors[0].trainable is False
        assert train.model.factors[1]._target_ == "tpen.nn.ElectronElectronCusp"
        assert train.model.factors[1].opposite_spin_coefficient == 0.5
        assert train.model.factors[1].trainable_range is False
        assert set(train.hamiltonian_terms) == {"kinetic", "harmonic_trap", "electron_electron"}
        assert train.hamiltonian_terms.harmonic_trap.omega == 0.5
        assert "atoms" not in train
        assert "atoms" not in eval_cfg
        assert instantiate(eval_cfg.system_descriptor).resources == {}
    else:
        assert train.model.factors[0].trainable_range is True
        assert train.model.factors[1].law.curvature_coefficient == 0.01
        assert train.model.factors[1].law.curvature_range == 1.0
        assert train.model.factors[1].law.trainable is True
        assert train.model.readout.trainable is True


def test_hooke2_is_the_nucleus_free_two_electron_pair() -> None:
    cfg = load_config(str(ROOT / "hooke2" / "eval.yaml"))
    descriptor = instantiate(cfg.system_descriptor)
    assert descriptor.particle.n_electrons == 2
    assert descriptor.particle.n_spin_up == 1
    assert descriptor.particle.n_spin_down == 1
    assert {term.slot for term in descriptor.terms} == {
        "kinetic", "harmonic_trap", "electron_electron",
    }
    assert descriptor.resources == {}


def test_train_launch_binds_declared_checkpoint_steps() -> None:
    row = {"kind": "train", "seed": 1, "results_root": "/tmp/results",
           "experiment_name": "stage1", "sector": "he", "run_id": "train-0",
           "walkers": 256, "updates": 101, "checkpoint_steps": [7, 101]}
    overrides = row_overrides(row)
    assert "checkpoint_steps=[7,101]" in overrides


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("kind", ("train", "eval"))
def test_stage1_configs_declare_jsonl_metrics_logger(system: str, kind: str) -> None:
    """Every Stage-1 run writes the canonical JSONL metrics artifact."""

    cfg = load_config(str(ROOT / system / f"{kind}.yaml"))
    loggers = OmegaConf.to_container(cfg.loggers, resolve=False)
    assert loggers == [
        {"_target_": "tpen.logging.JSONL", "path": "${run.dir}/metrics.jsonl"}
    ]
    if kind == "train":
        assert cfg.trainer.log_every_n_steps == 1
    else:
        summary_targets = {
            summary["_target_"]
            for summary in OmegaConf.to_container(
                cfg.evaluation_tasks.mcmc_energy.summaries, resolve=False
            )
        }
        assert "tpen.evaluation.summaries.LocalEnergySummary" in summary_targets
        assert "tpen.evaluation.summaries.SamplerStatsSummary" in summary_targets
