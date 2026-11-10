"""Frozen H-v2 pipeline configuration helpers and identity checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from experiments.toolkit.attached_activation import file_sha256

ROOT = Path(__file__).parent
TRAINING_SEED = 910001
SAMPLER_SEED = 910002
EVALUATION_SEEDS = (910003, 910004)

#: A1/TO baff945c: scale-only overrides onto the frozen pipeline-check
#: training config, applied by explicit dotted key path (never a blind
#: ``OmegaConf.merge``, which replaces ``model.layers`` wholesale instead of
#: touching one nested leaf). The overlay file's key set must equal this
#: allowlist exactly.
SCIENCE_SCALE_OVERLAY_PATH = ROOT / "configs" / "science_scale_overlay.yaml"
SCIENCE_SCALE_OVERLAY_KEYS = frozenset(
    {
        "runtime.device",
        "sampler.n_walkers",
        "trainer.max_steps",
        "model.embedding.out_channels",
        "model.embedding.hidden_channels",
        "model.layers.0.mixing.channels",
        "model.layers.0.path_aggregation.channels",
        "model.readout.channels",
    }
)

#: Source of per-lineage science seeds (A1: "no seed literal outside CI").
CONTRACT_INPUTS_PATH = ROOT / "contract_inputs.json"

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


def _set_by_dotted_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    """Set one leaf of *config* addressed by a ``.``-separated path.

    A purely-numeric segment indexes a list (``model.layers.0...``); any
    other segment indexes a mapping. This never creates a missing
    intermediate node -- the overlay only ever touches leaves that already
    exist in the frozen base config.
    """

    parts = dotted.split(".")
    node: Any = config
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value


def _science_seed_roster() -> dict[str, list[int]]:
    """Read the frozen per-lineage training seeds from ``contract_inputs.json``.

    Returns
    -------
    dict
        ``measurement.seed_roster``'s ``training_initialization_seeds`` and
        ``training_sampler_seeds`` lists -- the single frozen source; no
        seed literal is duplicated here.
    """

    payload = json.loads(CONTRACT_INPUTS_PATH.read_text(encoding="utf-8"))
    roster = payload["measurement"]["seed_roster"]
    return {
        "training_initialization_seeds": roster["training_initialization_seeds"],
        "training_sampler_seeds": roster["training_sampler_seeds"],
    }


def science_training_config(lineage: str, lineage_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one lineage's science TRAIN config, plus its config identity.

    Parameters
    ----------
    lineage : str
        Opaque lineage identity (for example ``"L1"``); recorded in the
        identity, never used here to decide which lineages are approved --
        that roster stays owned by ``launch_config.LINEAGES``, checked by
        the caller before this function is ever reached.
    lineage_index : int
        0-based index into ``contract_inputs.json``'s seed roster
        (``launch_config.LINEAGES.index(lineage)``).

    Returns
    -------
    tuple of dict, dict
        The resolved science training config (base ``train.yaml`` plus the
        scale overlay plus the lineage's frozen seeds), and a science-config
        identity record (lineage, lineage index, and the sha256 of every
        frozen input file this config was built from).

    Raises
    ------
    ValueError
        If the overlay file's key set has drifted from
        :data:`SCIENCE_SCALE_OVERLAY_KEYS`.
    """

    config = training_config()
    overlay = load_yaml(SCIENCE_SCALE_OVERLAY_PATH)
    if set(overlay) != SCIENCE_SCALE_OVERLAY_KEYS:
        raise ValueError(
            f"science scale overlay key set {set(overlay)!r} does not match the allowlist "
            f"{set(SCIENCE_SCALE_OVERLAY_KEYS)!r}"
        )
    for dotted, value in overlay.items():
        _set_by_dotted_path(config, dotted, value)
    seeds = _science_seed_roster()
    config["runtime"]["seed"] = seeds["training_initialization_seeds"][lineage_index]
    config["sampler"]["seed"] = seeds["training_sampler_seeds"][lineage_index]
    identity = {
        "lineage": lineage,
        "lineage_index": lineage_index,
        "overlay_sha256": file_sha256(SCIENCE_SCALE_OVERLAY_PATH),
        "base_train_sha256": file_sha256(ROOT / "configs" / "train.yaml"),
        "contract_inputs_sha256": file_sha256(CONTRACT_INPUTS_PATH),
    }
    return config, identity


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
    "CONTRACT_INPUTS_PATH",
    "CONTRACT_NUCLEAR_LAW",
    "EVALUATION_SEEDS",
    "ROOT",
    "SAMPLER_SEED",
    "SCIENCE_SCALE_OVERLAY_KEYS",
    "SCIENCE_SCALE_OVERLAY_PATH",
    "TRAINING_SEED",
    "assert_hydrogen_contract",
    "evaluation_config",
    "load_yaml",
    "science_training_config",
    "training_config",
]
