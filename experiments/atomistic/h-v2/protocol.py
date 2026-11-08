"""Frozen H-v2 pipeline configuration helpers and identity checks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).parent
TRAINING_SEED = 910001
SAMPLER_SEED = 910002
EVALUATION_SEEDS = (910003, 910004)

#: Contract-frozen nuclear cusp law literals (``admission.pipeline_check.nuclear_law_note``).
CONTRACT_NUCLEAR_LAW = {
    "_target_": "tpen.nn.CurvatureElectronNucleusCuspLaw",
    "curvature_coefficient": 0.01,
    "curvature_range": 1.0,
    "trainable": True,
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one frozen configuration without resolving Hydra interpolations."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def training_config() -> dict[str, Any]:
    """Return a fresh copy of the frozen short-route training config."""

    return copy.deepcopy(load_yaml(ROOT / "configs" / "train.yaml"))


def evaluation_config(checkpoint: Path, *, stream: int = 0, root: Path | None = None) -> dict[str, Any]:
    """Return a fresh model-only evaluation config for one stream."""

    if stream not in (0, 1):
        raise ValueError("pipeline evaluation stream must be 0 or 1")
    config = load_yaml(ROOT / "configs" / "eval.yaml")
    config["load"]["path"] = str(checkpoint)
    config["runtime"]["seed"] = EVALUATION_SEEDS[stream]
    config["run"]["run_id"] = f"h-v2-eval-stream{stream}"
    if root is not None:
        config["run"]["root"] = str(root)
    return config


def assert_hydrogen_contract(config: dict[str, Any], *, evaluation: bool = False) -> None:
    """Refuse configs that drift from one-electron, charge-one hydrogen."""

    system = config.get("system", {})
    if system.get("n_particles") != 1 or system.get("spin") != {"n_up": 1, "n_down": 0}:
        raise ValueError("H-v2 requires exactly one spin-up electron")
    nuclei = system.get("nuclei", {})
    if nuclei.get("positions") != [[0.0, 0.0, 0.0]] or nuclei.get("charges") != [1.0]:
        raise ValueError("H-v2 requires one charge-one nucleus at the origin")
    terms = config.get("hamiltonian_terms", {})
    if set(terms) != {"kinetic", "electron_nucleus"}:
        raise ValueError("H-v2 Hamiltonian must contain only kinetic and electron-nucleus terms")
    if terms["kinetic"].get("_target_") != "tpen.physics.kinetic.KineticEnergy":
        raise ValueError("H-v2 requires the -1/2 kinetic-energy operator")
    if terms["electron_nucleus"].get("_target_") != "tpen.physics.potential.ElectronNucleusPotential":
        raise ValueError("H-v2 requires the raw electron-nucleus potential")
    if terms["electron_nucleus"].get("eps") != 0.0:
        raise ValueError("H-v2 Coulomb potential must be raw -1/r")
    sampler = config.get("sampler", {})
    if sampler.get("_target_") != "tpen.sampling.metropolis.MetropolisSampler":
        raise ValueError("H-v2 requires the Metropolis sampler")
    factors = config.get("model", {}).get("factors", [])
    targets = [factor.get("_target_") for factor in factors]
    if targets != ["tpen.nn.ElectronNucleusCusp"]:
        raise ValueError("H-v2 model must have only the nuclear cusp factor")
    if "law" not in factors[0]:
        raise ValueError("H-v2 nuclear cusp factor must declare a law")
    law = factors[0]["law"]
    if not isinstance(law, dict):
        raise ValueError("H-v2 nuclear cusp law must be a mapping")
    if law.get("_target_") != CONTRACT_NUCLEAR_LAW["_target_"]:
        raise ValueError("H-v2 nuclear cusp law must be the contract curvature law")
    if law.get("curvature_coefficient") != CONTRACT_NUCLEAR_LAW["curvature_coefficient"]:
        raise ValueError("H-v2 nuclear cusp law coefficient must be 0.01")
    if law.get("curvature_range") != CONTRACT_NUCLEAR_LAW["curvature_range"]:
        raise ValueError("H-v2 nuclear cusp law range must be 1.0")
    if law.get("trainable") is not CONTRACT_NUCLEAR_LAW["trainable"]:
        raise ValueError("H-v2 nuclear cusp law must be trainable")
    if law != CONTRACT_NUCLEAR_LAW:
        raise ValueError("H-v2 nuclear cusp law carries unexpected keys")
    if not evaluation and config.get("sampler", {}).get("seed") == config.get("runtime", {}).get("seed"):
        raise ValueError("training and sampler seeds must be distinct")
    if evaluation:
        if config.get("load", {}).get("mode") != "model_only":
            raise ValueError("fresh evaluation must request model-only restore")
        if "optimizer" in config or "trainer" in config:
            raise ValueError("fresh evaluation must not carry training state")


__all__ = [
    "CONTRACT_NUCLEAR_LAW",
    "EVALUATION_SEEDS",
    "ROOT",
    "SAMPLER_SEED",
    "TRAINING_SEED",
    "assert_hydrogen_contract",
    "evaluation_config",
    "load_yaml",
    "training_config",
]
