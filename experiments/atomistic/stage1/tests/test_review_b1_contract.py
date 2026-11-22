"""Reviewer probes for Stage-1 physical identity and restore compatibility."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.atomistic.stage1.protocol import HOOKE_SYSTEMS, SYSTEMS, ProtocolError, physical_system_id_from_config
from tpen.run import load_config


CONFIG_ROOT = Path(__file__).parents[1] / "configs"
def _config(system: str, kind: str = "eval") -> Path:
    return CONFIG_ROOT / system / f"{kind}.yaml"


def _copy_config(source: Path, destination: Path, mutate) -> Path:
    cfg = OmegaConf.load(source)
    mutate(cfg)
    OmegaConf.save(cfg, destination)
    return destination


@pytest.mark.parametrize(
    ("system", "mutation"),
    (
        ("hooke1", "trap"),
        ("hooke2", "trap"),
        ("hooke2", "electron_electron"),
        ("he", "nuclear_charge"),
        ("he", "descriptor_spin"),
        ("h2", "descriptor_spin"),
    ),
)
def test_physical_id_rejects_registry_physics_mismatch(
    system: str, mutation: str, tmp_path: Path
) -> None:
    """A registry label must not make a physically different descriptor valid."""

    source = _config(system)

    def change(cfg) -> None:
        if mutation == "trap":
            cfg.hamiltonian_terms.harmonic_trap.omega = 0.6
        elif mutation == "electron_electron":
            terms = OmegaConf.to_container(cfg.hamiltonian_terms, resolve=True)
            del terms["electron_electron"]
            cfg.hamiltonian_terms = terms
            cfg.system_descriptor.terms = terms
        elif mutation == "nuclear_charge":
            # Keep system.nuclei at Z=2; replace only the actual term resource.
            cfg.hamiltonian_terms.electron_nucleus.atoms = {
                "_target_": "tpen.data.atomic_configuration.AtomicConfiguration",
                "positions": {
                    "_target_": "torch.tensor",
                    "data": [[0.0, 0.0, 0.0]],
                },
                "charges": {"_target_": "torch.tensor", "data": [3.0]},
            }
        else:
            # Keep cfg.system.spin unchanged; override the descriptor particle.
            cfg.system_descriptor.particle.n_spin_up = 2
            cfg.system_descriptor.particle.n_spin_down = 0

    mutated = _copy_config(source, tmp_path / f"{system}-{mutation}.yaml", change)
    with pytest.raises(ProtocolError):
        physical_system_id_from_config(mutated)


def test_physical_id_rejects_descriptor_operator_set_mismatch(monkeypatch) -> None:
    source = _config("h2")
    cfg = OmegaConf.load(source)

    def remove_nuclear_repulsion(cfg) -> None:
        terms = OmegaConf.to_container(cfg.hamiltonian_terms, resolve=True)
        del terms["nucleus_nucleus"]
        cfg.hamiltonian_terms = terms
        cfg.system_descriptor.terms = terms

    remove_nuclear_repulsion(cfg)
    monkeypatch.setattr("tpen.run.load_config", lambda _: cfg)
    with pytest.raises(ProtocolError, match="operator set"):
        physical_system_id_from_config("in-memory-h2-missing-term.yaml")


def test_physical_id_rejects_extra_descriptor_operator(monkeypatch) -> None:
    source = _config("he")
    cfg = OmegaConf.load(source)

    def add_extra_term(cfg) -> None:
        terms = OmegaConf.to_container(cfg.hamiltonian_terms, resolve=True)
        terms["extra_trap"] = {
            "_target_": "tpen.physics.potential.HarmonicTrap",
            "omega": 0.5,
        }
        cfg.hamiltonian_terms = terms
        cfg.system_descriptor.terms = terms

    add_extra_term(cfg)
    monkeypatch.setattr("tpen.run.load_config", lambda _: cfg)
    with pytest.raises(ProtocolError, match="operator set"):
        physical_system_id_from_config("in-memory-he-extra-term.yaml")


def test_physical_id_accepts_registry_operator_set() -> None:
    assert physical_system_id_from_config(_config("h2"))


@pytest.mark.parametrize("system", SYSTEMS)
def test_plan_requires_train_eval_hamiltonian_parity(system: str) -> None:
    eval_config = _config(system)
    expected = physical_system_id_from_config(
        eval_config,
        train_config_path=_config(system, "train"),
        system_name=system,
    )
    assert expected == physical_system_id_from_config(eval_config)


@pytest.mark.parametrize(
    ("system", "mutation"),
    (("hooke1", "omega"), ("he", "charge"), ("hooke2", "electron_electron")),
)
def test_plan_rejects_train_eval_hamiltonian_drift(
    system: str, mutation: str, tmp_path: Path
) -> None:
    def change(cfg) -> None:
        if mutation == "omega":
            cfg.hamiltonian_terms.harmonic_trap.omega = 0.6
        elif mutation == "charge":
            cfg.system.nuclei.charges = [3.0]
        else:
            terms = OmegaConf.to_container(cfg.hamiltonian_terms, resolve=True)
            terms["electron_electron"] = {
                "_target_": "tpen.physics.potential.HarmonicTrap",
                "omega": 0.5,
            }
            cfg.hamiltonian_terms = terms

    train_config = _copy_config(
        _config(system, "train"), tmp_path / f"{system}-{mutation}-train.yaml", change
    )
    with pytest.raises(ProtocolError, match=rf"train/eval Hamiltonian mismatch for {system}"):
        physical_system_id_from_config(
            _config(system), train_config_path=train_config, system_name=system
        )


def test_plan_rejects_train_descriptor_terms_drift(tmp_path: Path) -> None:
    def change(cfg) -> None:
        cfg.system_descriptor.terms = {
            "kinetic": {"_target_": "tpen.physics.kinetic.KineticEnergy"},
        }

    train_config = _copy_config(
        _config("hooke2", "train"), tmp_path / "hooke2-descriptor-drift-train.yaml", change
    )
    with pytest.raises(ProtocolError, match="train/eval Hamiltonian mismatch for hooke2"):
        physical_system_id_from_config(
            _config("hooke2"), train_config_path=train_config, system_name="hooke2"
        )


@pytest.mark.parametrize("system", SYSTEMS)
def test_physical_id_ignores_run_and_evaluation_budget_changes(
    system: str, tmp_path: Path
) -> None:
    """The descriptor ID is physics-owned, not a run/sampler identity."""

    source = _config(system)
    expected = physical_system_id_from_config(source)

    def change_run_only(cfg) -> None:
        cfg.runtime.seed = 918273
        cfg.run.root = str(tmp_path / "different-results-root")
        cfg.run.dir = str(tmp_path / "different-run-dir")
        cfg.evaluation_sampler.n_walkers = 17
        cfg.evaluation_sampler.n_steps = 3
        cfg.evaluation_tasks.mcmc_energy.generator.n_draws = 9

    changed = _copy_config(source, tmp_path / f"{system}-run-only.yaml", change_run_only)
    assert physical_system_id_from_config(changed) == expected

    def change_physics(cfg) -> None:
        if system in HOOKE_SYSTEMS:
            cfg.hamiltonian_terms.harmonic_trap.omega = 0.6
        else:
            cfg.system_descriptor.particle.n_spin_up = 2
            cfg.system_descriptor.particle.n_spin_down = 0

    physics_changed = _copy_config(source, tmp_path / f"{system}-physics.yaml", change_physics)
    # Controlled negative: instantiate the descriptor directly because the
    # fixed planner is expected to reject this registry-mismatched config.
    assert instantiate(load_config(str(physics_changed)).system_descriptor).system_id() != expected


def _model_config(cfg):
    return OmegaConf.to_container(cfg.model, resolve=True)


@pytest.mark.parametrize("system", SYSTEMS)
def test_train_eval_models_are_structurally_compatible_without_forward(system: str) -> None:
    """Check structural compatibility only; this is not restore verification."""

    train_cfg = load_config(str(_config(system, "train")))
    eval_cfg = load_config(str(_config(system)))
    train_model_cfg = _model_config(train_cfg)
    eval_model_cfg = _model_config(eval_cfg)
    assert eval_model_cfg == train_model_cfg

    # Instantiate only: no samples, local-energy call, or forward pass.
    train_model = instantiate(train_cfg.model)
    eval_model = instantiate(eval_cfg.model)
    assert type(eval_model) is type(train_model)
    assert {
        key: tuple(value.shape) for key, value in eval_model.state_dict().items()
    } == {
        key: tuple(value.shape) for key, value in train_model.state_dict().items()
    }

    changed_model_cfg = deepcopy(eval_model_cfg)
    changed_model_cfg["embedding"]["hidden_channels"] += 1
    # Controlled negative: a changed architecture is discriminated structurally.
    assert changed_model_cfg != train_model_cfg
