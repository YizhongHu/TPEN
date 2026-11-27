"""One-shot Cannon diagnostic for Stage-1 profile composition ordering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf
from tpen.run import load_config

from experiments.atomistic.stage1 import protocol, train
from experiments.atomistic.stage1.protocol import (
    HE_SIGMOID_PROFILE,
    HOOKE2_ACT50_SIGMOID_PROFILE,
    config_identities,
)
from experiments.atomistic.stage1.tests.test_he_sigmoid_profile import _rows


ROOT = Path(__file__).parents[1]


def _leaves(cfg: Any) -> dict[str, Any]:
    """Return the three profile leaves, preserving absent-path evidence."""

    model = OmegaConf.to_container(cfg.model, resolve=False)
    if not isinstance(model, dict):
        return {"model": repr(model)}
    layers = model.get("layers")
    layer = layers[0] if isinstance(layers, list) and layers else None
    return {
        "embedding": (model.get("embedding") or {}).get("activation")
        if isinstance(model.get("embedding"), dict) else None,
        "mixing": (layer.get("mixing") or {}).get("activation")
        if isinstance(layer, dict) and isinstance(layer.get("mixing"), dict) else None,
        "path_aggregation": (layer.get("path_aggregation") or {}).get("activation")
        if isinstance(layer, dict) and isinstance(layer.get("path_aggregation"), dict) else None,
    }


def _case(profile: str, system: str) -> dict[str, Any]:
    """Return one train row for the requested profile and system."""

    if system == "he":
        return next(row for row in _rows(profile) if row["kind"] == "train")
    identities = config_identities(
        config_root=ROOT / "configs", systems=(system,), science_profile=profile,
    )
    rows = protocol.build_manifest(
        protocol=protocol.STAGE1_PROTOCOL, source_sha="a" * 40,
        config_sha256=identities,
        checkpoints=list(range(0, 50_001, 1_000)),
        lineage_seeds=[11, 12, 13], checkpoint_sha256={},
        results_root="/external/he-sigmoid-diagnostic", systems=(system,), phase=1,
        device="cpu", science_profile=profile, include_smoke=False, auto_overlay=True,
    )
    return next(row for row in rows if row["kind"] == "train")


def test_report_profile_composition_steps() -> None:
    """Print He and Hooke leaves at each gateway step, then stop as diagnostic."""

    observations: dict[str, Any] = {}
    for label, profile, system in (
        ("he-sigmoid", HE_SIGMOID_PROFILE, "he"),
        ("hooke-act50-sigmoid", HOOKE2_ACT50_SIGMOID_PROFILE, "hooke2"),
    ):
        row = _case(profile, system)
        cfg = load_config(str(ROOT / row["config"]))
        steps: dict[str, Any] = {"loaded": _leaves(cfg)}
        protocol.compose_model_profile(cfg, profile)
        steps["profile_applied"] = _leaves(cfg)
        train._bind_output_identity(cfg, row)
        steps["identity_bound"] = _leaves(cfg)
        base_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        protocol.compose_train_overlay(
            cfg, overlay=str(row["overlay"]),
            expected_sha256=str(row["overlay_sha256"]),
            overlay_profile=str(row.get("overlay_profile", protocol.OVERLAY_PROFILE_V1)),
            science_profile=profile,
        )
        steps["overlay_composed"] = _leaves(cfg)
        cfg = OmegaConf.merge(cfg, OmegaConf.create(protocol.row_config_values(row)))
        steps["row_merged"] = _leaves(cfg)
        train.prepare_initial_checkpoint_config(cfg)
        steps["final_unresolved"] = _leaves(cfg)
        try:
            steps["final_resolved"] = _leaves(OmegaConf.create(
                OmegaConf.to_container(cfg, resolve=True),
            ))
        except Exception as exc:  # pragma: no cover - diagnostic output only
            steps["final_resolved_error"] = repr(exc)
        observations[label] = steps

    pytest.fail(json.dumps(observations, sort_keys=True, indent=2))
