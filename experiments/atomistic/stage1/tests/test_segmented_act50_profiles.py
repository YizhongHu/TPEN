"""Segmented ACT50 identities preserve the original training definitions."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import types

from omegaconf import OmegaConf
import pytest

from experiments.atomistic.stage1 import plan, protocol
from experiments.atomistic.stage1.train import build_train_config

ROOT = Path(__file__).parents[4]
STUDY = Path(__file__).parents[1]
PROFILES = ("hooke2-sigmoid50k-seg-v1", "hooke2-tanh50k-seg-v1", "hooke2-nolayer50k-seg-v1")


def _rows(owner, profile, results_root="/external/segmented-act50"):
    return owner.build_manifest(
        protocol=owner.STAGE1_PROTOCOL, source_sha="a" * 40,
        config_sha256=owner.config_identities(
            config_root=STUDY / "configs", systems=("hooke2",), science_profile=profile,
        ),
        checkpoints=list(range(0, 50_001, 1_000)), lineage_seeds=[11, 12, 13],
        checkpoint_sha256={}, results_root=results_root,
        systems=("hooke2",), phase=1, device="cuda", science_profile=profile,
        include_smoke=False, auto_overlay=True,
    )


def test_overlay_changes_only_recorder_format() -> None:
    original = (STUDY / "configs/overlays/hooke-diagnostics-v1.yaml").read_text()
    segmented = (STUDY / "configs/overlays/hooke-diagnostics-v3.yaml").read_text()
    assert segmented == original.replace(
        "  - _target_: tpen.callback.RawTrainingBatchRecorder\n",
        "  - _target_: tpen.callback.RawTrainingBatchRecorder\n    format_version: 3\n",
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_planner_renders_segmented_profile(profile, tmp_path, monkeypatch) -> None:
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(sys, "argv", [
        "plan", "--output", str(output), "--phase", "1", "--systems", "hooke2",
        "--source-sha", "a" * 40, "--results-root", str(tmp_path / "results"),
        "--lineage-seeds", "11", "12", "13", "--science-profile", profile,
    ])
    plan.main()
    rows = json.loads(output.read_text())["rows"]
    assert len(rows) == 3
    original_profile = profile.replace("-seg-v1", "-v1")
    original_rows = _rows(protocol, original_profile, str(tmp_path / "results"))
    original = build_train_config(original_rows[0])
    assert {row["config_sha256"] for row in rows}.isdisjoint(
        row["config_sha256"] for row in original_rows
    )
    # row_id is a coordinate label; the bound row/manifest carries science identity.
    assert protocol.manifest_hash(rows) != protocol.manifest_hash(original_rows)
    for row, old in zip(rows, original_rows):
        assert row["row_id"] == old["row_id"]
        assert protocol.manifest_hash([row]) != protocol.manifest_hash([old])
    for row in rows:
        assert row["science_profile"] == profile
        assert row["overlay"] == "configs/overlays/hooke-diagnostics-v3.yaml"
        assert row["overlay_profile"] == "diagnostics-v3"
        assert row["updates"] == 50_000
        cfg = build_train_config(row)
        assert OmegaConf.to_container(cfg.model) == OmegaConf.to_container(original.model)
        assert cfg.trainer.max_steps == 50_000
        targets = [cb["_target_"] for cb in cfg.callbacks]
        recorder = next(cb for cb in cfg.callbacks if cb["_target_"] == "tpen.callback.RawTrainingBatchRecorder")
        assert recorder.format_version == 3
        # Retain the original composition; runtime PlannedYield is appended later.
        assert targets.index("tpen.callback.Checkpoint") < targets.index("tpen.callback.RawTrainingBatchRecorder")
        assert recorder.every_n_steps == 50
        assert recorder.partition_size_steps == 1000
        smoke = protocol.derive_smoke_request(row, results_root=str(tmp_path / "smoke"), stop_at=1000)
        assert smoke["execution"] == {"stop_at": 1000}
        assert smoke["updates"] == 50_000


def test_existing_profiles_rows_and_hashes_unchanged(monkeypatch) -> None:
    """Compare complete rows with the frozen pre-change protocol, not two new calls."""
    revision = (Path(__file__).parent / "fixtures/segmented-act50-base.txt").read_text().strip()
    path = "experiments/atomistic/stage1/protocol.py"
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
    reference = types.ModuleType("_segmented_act50_original_protocol")
    reference.__file__ = str(STUDY / "protocol.py")
    monkeypatch.setitem(sys.modules, reference.__name__, reference)
    exec(compile(source, reference.__file__, "exec"), reference.__dict__)
    for relative in ("configs/overlays/hooke-diagnostics-v1.yaml", "configs/overlays/hooke-diagnostics-v2.yaml",
                     "configs/hooke2/train.yaml", "configs/hooke2/eval.yaml"):
        before = subprocess.check_output(["git", "show", f"{revision}:experiments/atomistic/stage1/{relative}"], cwd=ROOT)
        assert (STUDY / relative).read_bytes() == before
    for profile in ("hooke2-sigmoid50k-v1", "hooke2-tanh50k-v1", "hooke2-nolayer50k-v1"):
        assert json.dumps(_rows(protocol, profile), sort_keys=True) == json.dumps(_rows(reference, profile), sort_keys=True)
