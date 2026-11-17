"""Regression tests for the production Stage-1 planning path."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import experiments.atomistic.stage1.plan as plan_module
import experiments.atomistic.stage1.protocol as protocol_module
from experiments.atomistic.stage1.protocol import ProtocolError


CONFIG_ROOT = Path(__file__).parents[1] / "configs"


def test_planned_physical_system_ids_rejects_train_electron_electron_swap(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "configs"
    shutil.copytree(CONFIG_ROOT, config_root)
    train = config_root / "hooke2" / "train.yaml"
    cfg = OmegaConf.load(train)
    cfg.hamiltonian_terms.electron_electron = {
        "_target_": "tpen.physics.potential.HarmonicTrap",
        "omega": 0.5,
    }
    OmegaConf.save(cfg, train)

    with pytest.raises(ProtocolError, match="hooke2"):
        protocol_module.planned_physical_system_ids(config_root)


def test_plan_main_uses_protocol_planned_physical_system_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    def refuse_planning_ids() -> dict[str, str]:
        raise ProtocolError("planned IDs were not requested")

    monkeypatch.setattr(protocol_module, "planned_physical_system_ids", refuse_planning_ids)
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan.py",
            "--output", str(tmp_path / "manifest.json"),
            "--source-sha", "a" * 40,
            "--config-sha256", "b" * 64,
            "--results-root", "/external/stage1",
            "--checkpoint-bindings", str(bindings),
            "--checkpoints", "7",
            "--lineage-seeds", "11", "12", "13",
        ],
    )

    with pytest.raises(ProtocolError, match="planned IDs were not requested"):
        plan_module.main()

