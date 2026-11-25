"""Focused T2 admission tests imported from the recorder-order review."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from experiments.atomistic.stage1 import plan as stage1_plan
from experiments.atomistic.stage1 import protocol as protocol_module
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.protocol import (
    Protocol,
    ProtocolError,
    build_manifest,
    config_identities,
    manifest_hash,
    overlay_sha256,
)
from tpen.run import load_config


ROOT = Path(__file__).parents[1]
CONFIG_ROOT = ROOT / "configs"


def _protocol() -> Protocol:
    """Use a small manifest shape while retaining the Stage-1 row contract."""

    return Protocol(1, 2, 1, 2, 3, 4, 2, 2)


def _v2_train_row() -> tuple[dict[str, object], object]:
    """Build one real v2 row and resolved producer config."""

    overlay = ROOT / protocol_module.OVERLAY_V2_RELATIVE_PATH
    rows = build_manifest(
        protocol=_protocol(),
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=("he",)),
        checkpoints=[0, 1],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/n/netscratch/kozinsky_lab/Lab/r1-review",
        systems=("he",),
        phase=1,
        device="cpu",
        overlay=str(overlay),
        overlay_sha256=overlay_sha256(overlay),
        overlay_profile=protocol_module.OVERLAY_PROFILE_V2,
    )
    row = next(row for row in rows if row["kind"] == "train")
    return row, stage1_train.build_train_config(row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("every_n_steps", 7),
        ("path", "${run.dir}/tampered-history"),
        ("partition_size_steps", 17),
        ("phase", "train"),
        ("start_step", 0),
    ],
)
def test_v2_resolved_validation_rejects_recorder_payload_mutation(
    field: str, value: object,
) -> None:
    """Recorder ordering must not admit a changed capture contract."""

    row, cfg = _v2_train_row()
    recorder = next(
        entry for entry in cfg.callbacks
        if entry.get("_target_") == "tpen.callback.RawTrainingBatchRecorder"
    )
    recorder[field] = value
    with pytest.raises(ProtocolError, match="callback profile drifted"):
        protocol_module.validate_resolved_train_config(cfg, row)


@pytest.mark.parametrize("system", protocol_module.SYSTEMS)
def test_v2_canonical_single_checkpoint_passes(system: str) -> None:
    """Every canonical Stage-1 base composes one recorder before its publisher."""

    overlay = ROOT / protocol_module.overlay_path_for_system(
        system, profile=protocol_module.OVERLAY_PROFILE_V2,
    )
    cfg = load_config(str(CONFIG_ROOT / system / "train.yaml"))
    protocol_module.compose_train_overlay(
        cfg,
        overlay=str(overlay),
        expected_sha256=overlay_sha256(overlay),
        overlay_profile=protocol_module.OVERLAY_PROFILE_V2,
    )
    targets = [entry["_target_"] for entry in cfg.callbacks]
    checkpoints = [i for i, target in enumerate(targets) if target == "tpen.callback.Checkpoint"]
    recorders = [
        i for i, target in enumerate(targets)
        if target == "tpen.callback.RawTrainingBatchRecorder"
    ]
    assert len(checkpoints) == 1
    assert recorders == [checkpoints[0] - 1]


def test_v2_composition_refuses_multiple_checkpoint_publishers() -> None:
    """The v2 one-writer contract rejects unsupported publisher multiplicity."""

    system = "he"
    overlay = ROOT / protocol_module.overlay_path_for_system(
        system, profile=protocol_module.OVERLAY_PROFILE_V2,
    )
    cfg = load_config(str(CONFIG_ROOT / system / "train.yaml"))
    checkpoint = next(
        entry for entry in cfg.callbacks
        if entry.get("_target_") == "tpen.callback.Checkpoint"
    )
    cfg.callbacks.append(copy.deepcopy(checkpoint))
    with pytest.raises(ProtocolError, match="multiple base Checkpoint publishers"):
        protocol_module.compose_train_overlay(
            cfg,
            overlay=str(overlay),
            expected_sha256=overlay_sha256(overlay),
            overlay_profile=protocol_module.OVERLAY_PROFILE_V2,
        )


def _phase1_with_overlay(tmp_path: Path, profile: str) -> tuple[Path, Path, Path]:
    """Write a phase-1 manifest and matching evidence envelope for CLI tests."""

    overlay = ROOT / protocol_module.overlay_path_for_system("he", profile=profile)
    rows = build_manifest(
        protocol=_protocol(),
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=("he",)),
        checkpoints=[0, 1],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path / "results"),
        systems=("he",),
        phase=1,
        device="cpu",
        overlay=str(overlay),
        overlay_sha256=overlay_sha256(overlay),
        overlay_profile=profile,
    )
    phase1_hash = manifest_hash(rows)
    phase1_path = tmp_path / "phase1.json"
    bindings_path = tmp_path / "bindings.json"
    phase1_path.write_text(
        json.dumps({"phase": 1, "manifest_hash": phase1_hash, "rows": rows}),
        encoding="utf-8",
    )
    bindings_path.write_text(
        json.dumps({
            "schema": protocol_module.CHECKPOINT_EVIDENCE_SCHEMA,
            "phase1_manifest_hash": phase1_hash,
            "phase1_train_row_hashes": {
                f"{row['system']}:{row['lineage']}": manifest_hash([row])
                for row in rows if row["kind"] == "train"
            },
            "checkpoints": {
                f"{row['system']}:{row['lineage']}:1": "c" * 64
                for row in rows if row["kind"] == "train"
            },
        }),
        encoding="utf-8",
    )
    return phase1_path, bindings_path, overlay


def _run_phase2(
    monkeypatch: pytest.MonkeyPatch,
    *,
    phase1_path: Path,
    bindings_path: Path,
    output: Path,
    overlay: Path | None = None,
    profile: str | None = None,
) -> int:
    """Invoke the real phase-2 CLI with explicit optional identity arguments."""

    argv = [
        "stage1-plan", "--output", str(output), "--source-sha", "a" * 40,
        "--results-root", str(output.parent / "results"), "--phase", "2",
        "--phase1-manifest", str(phase1_path), "--checkpoint-bindings", str(bindings_path),
        "--checkpoints", "1", "--lineage-seeds", "11", "12", "13", "--systems", "he",
    ]
    if overlay is not None:
        argv.extend(["--overlay", str(overlay), "--overlay-sha256", overlay_sha256(overlay)])
    if profile is not None:
        argv.extend(["--overlay-profile", profile])
    monkeypatch.setattr(sys, "argv", argv)
    return stage1_plan.main()


def test_phase2_cli_refuses_conflicting_explicit_profile_and_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v1 phase-1 document cannot be silently requested as diagnostics-v2."""

    phase1, bindings, v2_overlay = _phase1_with_overlay(
        tmp_path, protocol_module.OVERLAY_PROFILE_V1,
    )
    output = tmp_path / "conflict.json"
    with pytest.raises((ProtocolError, ValueError), match="profile|overlay|phase-1"):
        _run_phase2(
            monkeypatch,
            phase1_path=phase1,
            bindings_path=bindings,
            output=output,
            overlay=v2_overlay,
            profile=protocol_module.OVERLAY_PROFILE_V2,
        )
    assert not output.exists()


@pytest.mark.parametrize("profile", protocol_module.OVERLAY_PROFILES)
def test_phase2_cli_accepts_matching_explicit_profile_and_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str,
) -> None:
    """An explicit phase-2 identity must match its phase-1 document."""

    phase1, bindings, overlay = _phase1_with_overlay(tmp_path, profile)
    output = tmp_path / f"{profile}.json"
    assert _run_phase2(
        monkeypatch,
        phase1_path=phase1,
        bindings_path=bindings,
        output=output,
        overlay=overlay,
        profile=profile,
    ) == 0
    assert output.exists()


def test_phase2_cli_inherits_omitted_profile_from_phase1_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting the profile preserves the phase-1-backed inheritance path."""

    phase1, bindings, overlay = _phase1_with_overlay(
        tmp_path, protocol_module.OVERLAY_PROFILE_V2,
    )
    output = tmp_path / "inherited.json"
    assert _run_phase2(
        monkeypatch,
        phase1_path=phase1,
        bindings_path=bindings,
        output=output,
        overlay=overlay,
    ) == 0
    assert output.exists()
