"""Structural and descriptor-bound checks for the four Stage-1 configs."""

from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from tpen.run import load_config
from experiments.atomistic.stage1.protocol import (
    SYSTEMS,
    STAGE1_PROTOCOL,
    build_manifest,
    config_identities,
    physical_system_id_from_config,
)
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
    if system in {"he", "h2"}:
        expected_steps = [0, *range(1000, 100001, 1000)]
        assert list(checkpoints[0].schedule.updates) == expected_steps
        assert list(cfg.checkpoint_steps) == expected_steps
        assert cfg.trainer.max_steps == 100000
    else:
        assert list(checkpoints[0].schedule.updates) == [1000]
        assert list(cfg.checkpoint_steps) == [1000]


def test_real_he_h2_phase1_rows_bind_protocol_horizon() -> None:
    """The production phase-1 rows must carry the configured horizon."""

    train_configs = {
        system: load_config(str(ROOT / system / "train.yaml"))
        for system in ("he", "h2")
    }
    checkpoints = list(train_configs["he"].checkpoint_steps)
    assert checkpoints == list(train_configs["h2"].checkpoint_steps)
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=ROOT, systems=("he", "h2")),
        checkpoints=checkpoints,
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/stage1",
        systems=("he", "h2"),
        phase=1,
    )
    train_rows = [row for row in rows if row["kind"] == "train"]
    assert {row["system"] for row in train_rows} == {"he", "h2"}
    assert all(row["updates"] == STAGE1_PROTOCOL.lineage_updates for row in train_rows)
    assert all(row["checkpoint_steps"] == checkpoints for row in train_rows)


@pytest.mark.parametrize("system", ["he", "h2"])
def test_real_train_config_horizon_matches_protocol(system: str) -> None:
    """The real train config and protocol must share one terminal update."""

    cfg = load_config(str(ROOT / system / "train.yaml"))
    assert cfg.trainer.max_steps == STAGE1_PROTOCOL.lineage_updates
    assert max(cfg.checkpoint_steps) == STAGE1_PROTOCOL.lineage_updates


@pytest.mark.parametrize("system", ["he", "h2"])
def test_width32_model_is_identical_between_train_and_eval(system: str) -> None:
    train = load_config(str(ROOT / system / "train.yaml"))
    eval_cfg = load_config(str(ROOT / system / "eval.yaml"))
    eval_model = eval_cfg.stage1_he_model if system == "he" else eval_cfg.stage1_h2_model
    for model in (train.model, eval_model):
        assert model.embedding.out_channels == 32
        assert model.embedding.hidden_channels == 128
        assert model.layers[0].mixing.channels == 32
        assert model.layers[0].path_aggregation.channels == 32
        assert model.readout.channels == 32


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
