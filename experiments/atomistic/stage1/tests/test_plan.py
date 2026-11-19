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
            "--checkpoints", "0", "7", "100000",
            "--lineage-seeds", "11", "12", "13",
        ],
    )

    with pytest.raises(ProtocolError, match="planned IDs were not requested"):
        plan_module.main()


def test_two_phase_plan_compares_legacy_config_digest_to_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan.py",
            "--output", str(tmp_path / "manifest.json"),
            "--source-sha", "a" * 40,
            "--config-sha256", "b" * 64,
            "--results-root", "/external/stage1",
            "--phase", "1",
            "--systems", "he",
            "--checkpoints", "0", "7", "100000",
            "--lineage-seeds", "11", "12", "13",
        ],
    )

    with pytest.raises(ValueError, match="computed two-phase identity"):
        plan_module.main()


def _checkpoint_evidence(phase1_doc: dict, *, checkpoint: int = 7, digest: str = "c" * 64) -> dict:
    train_rows = [row for row in phase1_doc["rows"] if row["kind"] == "train"]
    return {
        "schema": protocol_module.CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_doc["manifest_hash"],
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": protocol_module.manifest_hash([row])
            for row in train_rows
        },
        "checkpoints": {
            f"{row['system']}:{row['lineage']}:{checkpoint}": digest for row in train_rows
        },
    }


def test_plan_main_phase1_then_phase2_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """(a)-(d): the authentic two-phase round trip through `plan.main`."""

    phase1_path = tmp_path / "phase1.json"
    monkeypatch.setattr(
        sys, "argv",
        ["plan.py", "--output", str(phase1_path), "--source-sha", "a" * 40,
         "--results-root", "/external/stage1", "--phase", "1", "--systems", "he",
         "--checkpoints", "0", "7", "100000", "--lineage-seeds", "11", "12", "13"],
    )
    assert plan_module.main() == 0
    phase1_doc = json.loads(phase1_path.read_text(encoding="utf-8"))
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(_checkpoint_evidence(phase1_doc)), encoding="utf-8")

    # (a) the authentic round trip.
    phase2_path = tmp_path / "phase2.json"
    monkeypatch.setattr(
        sys, "argv",
        ["plan.py", "--output", str(phase2_path), "--source-sha", "a" * 40,
         "--results-root", "/external/stage1", "--phase", "2",
         "--phase1-manifest", str(phase1_path), "--checkpoint-bindings", str(evidence_path),
         "--checkpoints", "7", "--lineage-seeds", "11", "12", "13"],
    )
    assert plan_module.main() == 0
    phase2_doc = json.loads(phase2_path.read_text(encoding="utf-8"))
    assert len(phase2_doc["rows"]) == 16
    assert phase2_doc["phase"] == 2
    assert phase2_doc["config_identity_rule"] == "stage1-config-v1"
    eval_rows = [row for row in phase2_doc["rows"] if row["kind"] == "eval"]
    assert len(eval_rows) == 12
    assert all(row["phase1_manifest_hash"] == phase1_doc["manifest_hash"] for row in eval_rows)

    # (b) a matching legacy scalar passes.
    he_digest = next(row["config_sha256"] for row in phase1_doc["rows"] if row["system"] == "he")
    phase2_b_path = tmp_path / "phase2_b.json"
    monkeypatch.setattr(
        sys, "argv",
        ["plan.py", "--output", str(phase2_b_path), "--source-sha", "a" * 40,
         "--results-root", "/external/stage1", "--phase", "2",
         "--phase1-manifest", str(phase1_path), "--checkpoint-bindings", str(evidence_path),
         "--config-sha256", he_digest,
         "--checkpoints", "7", "--lineage-seeds", "11", "12", "13"],
    )
    assert plan_module.main() == 0

    # (c) a changed config between phases is refused at the planning boundary.
    phase2_c_path = tmp_path / "phase2_c.json"
    monkeypatch.setattr(
        sys, "argv",
        ["plan.py", "--output", str(phase2_c_path), "--source-sha", "a" * 40,
         "--results-root", "/external/stage1", "--phase", "2",
         "--phase1-manifest", str(phase1_path), "--checkpoint-bindings", str(evidence_path),
         "--checkpoints", "7", "--lineage-seeds", "11", "12", "13"],
    )
    with monkeypatch.context() as patched:
        patched.setattr(plan_module, "config_identities", lambda **_: {"he": "f" * 64})
        with pytest.raises(ProtocolError, match="phase-2 config_sha256 differs"):
            plan_module.main()
    assert not phase2_c_path.exists()

    # (d) checkpoint evidence from plan A does not bind to phase-1 plan B.
    phase1_b_path = tmp_path / "phase1_b.json"
    monkeypatch.setattr(
        sys, "argv",
        ["plan.py", "--output", str(phase1_b_path), "--source-sha", "a" * 40,
         "--results-root", "/external/stage1", "--phase", "1", "--systems", "he",
         "--checkpoints", "0", "7", "100000", "--lineage-seeds", "21", "22", "23"],
    )
    assert plan_module.main() == 0
    phase2_d_path = tmp_path / "phase2_d.json"
    monkeypatch.setattr(
        sys, "argv",
        ["plan.py", "--output", str(phase2_d_path), "--source-sha", "a" * 40,
         "--results-root", "/external/stage1", "--phase", "2",
         "--phase1-manifest", str(phase1_b_path), "--checkpoint-bindings", str(evidence_path),
         "--checkpoints", "7", "--lineage-seeds", "11", "12", "13"],
    )
    with pytest.raises(ProtocolError, match="checkpoint evidence does not match phase-1 manifest"):
        plan_module.main()
    assert not phase2_d_path.exists()

