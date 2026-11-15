"""Focused reviewer probes for train/eval physical-system parity."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1.protocol import (
    SYSTEMS,
    ProtocolError,
    physical_system_id_from_config,
    planned_physical_system_ids,
)


# The Cannon runner may execute this reviewer file as an external staged test;
# the exact product checkout remains the source of the configs under test.
CONFIG_ROOT = Path.cwd() / "experiments" / "atomistic" / "stage1" / "configs"


def _config(root: Path, system: str, kind: str) -> Path:
    return root / system / f"{kind}.yaml"


def _copy_train_with_original_descriptor(
    root: Path, system: str, mutate,
) -> Path:
    """Copy a train config while freezing its descriptor to its original terms."""

    source = _config(root, system, "train")
    cfg = OmegaConf.load(source)
    original_terms = OmegaConf.to_container(cfg.system_descriptor.terms, resolve=True)
    cfg.system_descriptor.terms = original_terms
    mutate(cfg)
    destination = root / system / "mutated-train.yaml"
    OmegaConf.save(cfg, destination)
    return destination


def test_all_planned_system_ids_accept_unchanged_train_eval_configs(
    tmp_path: Path,
) -> None:
    """The four checked-in train/eval pairs are valid physical plans."""

    config_root = tmp_path / "configs"
    shutil.copytree(CONFIG_ROOT, config_root)

    planned = planned_physical_system_ids(config_root)

    assert set(planned) == set(SYSTEMS)
    assert all(len(system_id) == 64 for system_id in planned.values())


@pytest.mark.parametrize("mutation", ["omega", "remove_ee"])
def test_runner_hamiltonian_drift_is_not_hidden_by_frozen_descriptor(
    mutation: str, tmp_path: Path,
) -> None:
    """Changing the train runner Hamiltonian must invalidate the plan."""

    config_root = tmp_path / "configs"
    shutil.copytree(CONFIG_ROOT, config_root)

    def change(cfg) -> None:
        if mutation == "omega":
            cfg.hamiltonian_terms.harmonic_trap.omega = 0.6
            return
        terms = OmegaConf.to_container(cfg.hamiltonian_terms, resolve=True)
        del terms["electron_electron"]
        cfg.hamiltonian_terms = terms

    train_config = _copy_train_with_original_descriptor(config_root, "hooke2", change)
    frozen = OmegaConf.load(train_config).system_descriptor.terms
    assert "electron_electron" in frozen

    with pytest.raises(ProtocolError, match="train/eval Hamiltonian mismatch for hooke2"):
        physical_system_id_from_config(
            _config(config_root, "hooke2", "eval"),
            train_config_path=train_config,
            system_name="hooke2",
        )


@pytest.mark.parametrize("mutation", ["particle_count", "spin"])
def test_runner_particle_inputs_are_bound_to_the_train_descriptor(
    mutation: str, tmp_path: Path,
) -> None:
    """Train-only particle/spin changes must not be masked by descriptor terms."""

    config_root = tmp_path / "configs"
    shutil.copytree(CONFIG_ROOT, config_root)

    def change(cfg) -> None:
        if mutation == "particle_count":
            cfg.system.n_particles = 1
            cfg.system.spin.n_up = 1
            cfg.system.spin.n_down = 0
        else:
            cfg.system.spin.n_up = 2
            cfg.system.spin.n_down = 0

    train_config = _copy_train_with_original_descriptor(config_root, "hooke2", change)

    with pytest.raises(ProtocolError, match="train/eval Hamiltonian mismatch for hooke2"):
        physical_system_id_from_config(
            _config(config_root, "hooke2", "eval"),
            train_config_path=train_config,
            system_name="hooke2",
        )


def test_runtime_seed_and_sampler_budget_do_not_change_physical_identity(
    tmp_path: Path,
) -> None:
    """Operational seed/budget changes remain outside the physical identity."""

    config_root = tmp_path / "configs"
    shutil.copytree(CONFIG_ROOT, config_root)

    def change(cfg) -> None:
        cfg.runtime.seed = 918273
        cfg.sampler.n_walkers = 17
        cfg.trainer.max_steps = 9

    train_config = _copy_train_with_original_descriptor(config_root, "hooke2", change)

    system_id = physical_system_id_from_config(
        _config(config_root, "hooke2", "eval"),
        train_config_path=train_config,
        system_name="hooke2",
    )
    assert len(system_id) == 64
