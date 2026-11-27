"""ACT50 r1 consumer-boundary probes and green controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra.utils import instantiate
import torch

from experiments.atomistic.stage1 import collect as stage1_collect
from experiments.atomistic.stage1 import eval as stage1_eval
from experiments.atomistic.stage1 import plan as stage1_plan
from experiments.atomistic.stage1 import protocol as protocol_module
from experiments.atomistic.stage1.launch import _binding_token
from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA,
    HOOKE2_ACT50_A27,
    HOOKE2_ACT50_ALL_PROFILES,
    HOOKE2_ACT50_SEGMENTED_PROFILES,
    HOOKE2_ACT50_PROFILES,
    HOOKE2_ACT50_SIGMOID_PROFILE,
    HOOKE2_ACT50_TANH_PROFILE,
    STAGE1_PROTOCOL,
    ProtocolError,
    bind_evaluation_manifest,
    build_manifest,
    config_identities,
    manifest_hash,
)
from experiments.atomistic.stage1.train import build_train_config


ROOT = Path(__file__).parents[1]
CONFIG_ROOT = ROOT / "configs"


def _act50_phase1(profile: str) -> list[dict[str, object]]:
    """Build the production-shaped three-lineage phase-1 train manifest."""

    return build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=config_identities(
            config_root=CONFIG_ROOT, systems=("hooke2",), science_profile=profile,
        ),
        checkpoints=list(range(0, 50_001, 1_000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/act50-review-r1",
        systems=("hooke2",),
        phase=1,
        device="cuda",
        science_profile=profile,
        include_smoke=False,
        auto_overlay=True,
    )


def _write_act50_phase1(tmp_path: Path, profile: str) -> tuple[Path, Path]:
    rows = _act50_phase1(profile)
    phase1_hash = manifest_hash(rows)
    phase1_path = tmp_path / "phase1.json"
    phase1_path.write_text(
        json.dumps({"phase": 1, "manifest_hash": phase1_hash, "rows": rows}),
        encoding="utf-8",
    )
    bindings = {
        "schema": protocol_module.CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in rows
            if row["kind"] == "train"
        },
        "checkpoints": {
            f"{row['system']}:{row['lineage']}:{checkpoint}": "c" * 64
            for row in rows
            if row["kind"] == "train"
            for checkpoint in protocol_module.HOOKE2_ACT50_A27
        },
    }
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(json.dumps(bindings), encoding="utf-8")
    return phase1_path, bindings_path


def _run_phase2_without_optional_identity(
    monkeypatch: pytest.MonkeyPatch, phase1: Path, bindings: Path, output: Path,
) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1-plan",
            "--output",
            str(output),
            "--source-sha",
            "a" * 40,
            "--results-root",
            str(output.parent / "results"),
            "--phase",
            "2",
            "--phase1-manifest",
            str(phase1),
            "--checkpoint-bindings",
            str(bindings),
        ],
    )
    return stage1_plan.main()


def test_phase2_act50_infers_a27_from_phase1_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted phase-2 profile/checkpoints inherit ACT50 and emit 324 eval rows."""

    phase1, bindings = _write_act50_phase1(tmp_path, HOOKE2_ACT50_SIGMOID_PROFILE)
    output = tmp_path / "phase2.json"
    assert _run_phase2_without_optional_identity(monkeypatch, phase1, bindings, output) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert len(document["rows"]) == 3 + 3 * 27 * STAGE1_PROTOCOL.eval_streams
    eval_rows = [row for row in document["rows"] if row["kind"] == "eval"]
    assert len(eval_rows) == 324
    assert {row["science_profile"] for row in eval_rows} == {HOOKE2_ACT50_SIGMOID_PROFILE}
    assert {row["evaluation_selection"] for row in eval_rows} == {"A27"}


@pytest.mark.parametrize("profile", HOOKE2_ACT50_SEGMENTED_PROFILES)
def test_phase2_segmented_manifest_preserves_a27_and_bound_identity(
    profile: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit segmented CLI profile preserves its phase-1 authority."""

    phase1, bindings = _write_act50_phase1(tmp_path, profile)
    phase1_document = json.loads(phase1.read_text(encoding="utf-8"))
    output = tmp_path / "phase2.json"
    monkeypatch.setattr(sys, "argv", [
        "stage1-plan", "--output", str(output), "--source-sha", "a" * 40,
        "--results-root", str(tmp_path / "results"), "--phase", "2",
        "--phase1-manifest", str(phase1), "--checkpoint-bindings", str(bindings),
        "--science-profile", profile,
    ])
    assert stage1_plan.main() == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    train_rows = [row for row in document["rows"] if row["kind"] == "train"]
    assert train_rows == phase1_document["rows"]
    assert document["manifest_hash"] == manifest_hash(document["rows"])
    eval_rows = [row for row in document["rows"] if row["kind"] == "eval"]
    expected_roster = {
        (lineage, checkpoint, stream)
        for lineage in range(STAGE1_PROTOCOL.lineages)
        for checkpoint in HOOKE2_ACT50_A27
        for stream in range(STAGE1_PROTOCOL.eval_streams)
    }
    assert len(eval_rows) == len(expected_roster) == 324
    assert {(row["lineage"], row["checkpoint"], row["stream"])
            for row in eval_rows} == expected_roster
    for row in eval_rows:
        parent = next(train for train in train_rows if train["lineage"] == row["lineage"])
        assert row["science_profile"] == profile
        assert row["evaluation_selection"] == "A27"
        assert row["config_sha256"] == parent["config_sha256"]
        assert row["phase1_manifest_hash"] == phase1_document["manifest_hash"]
        assert row["phase1_train_row_hash"] == manifest_hash([parent])
        assert row["depends_on"] == parent["row_id"]


def test_phase2_act50_refuses_conflicting_explicit_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit ACT50 profile cannot override the phase-1 science authority."""

    phase1, bindings = _write_act50_phase1(tmp_path, HOOKE2_ACT50_TANH_PROFILE)
    output = tmp_path / "conflict.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage1-plan",
            "--output",
            str(output),
            "--source-sha",
            "a" * 40,
            "--results-root",
            str(tmp_path / "results"),
            "--phase",
            "2",
            "--phase1-manifest",
            str(phase1),
            "--checkpoint-bindings",
            str(bindings),
            "--science-profile",
            HOOKE2_ACT50_SIGMOID_PROFILE,
        ],
    )
    with pytest.raises((ProtocolError, ValueError), match="profile|phase-1|conflict"):
        stage1_plan.main()
    assert not output.exists()


@pytest.mark.parametrize("profile", HOOKE2_ACT50_PROFILES)
def test_gateway_rejects_tampered_act50_variant(profile: str) -> None:
    """The row variant is part of the producer admission contract, not a label."""

    row = dict(_act50_phase1(profile)[0])
    row["variant"] = "tanh" if row["variant"] != "tanh" else "sigmoid"
    with pytest.raises(ProtocolError, match="variant|profile"):
        protocol_module.validate_gateway_binding(
            binding=_binding_token(row), expected_kinds={"train"},
        )


def test_same_act50_profile_checkpoint_is_accepted(tmp_path: Path) -> None:
    """A checkpoint saved and evaluated under one ACT50 profile is a green control."""

    row = _act50_phase1(HOOKE2_ACT50_SIGMOID_PROFILE)[0]
    train_cfg = build_train_config(row)
    model = instantiate(train_cfg.model).to(dtype=torch.float64)
    context = SimpleNamespace(
        cfg=train_cfg,
        run_dir=tmp_path / "train-run",
        metadata=SimpleNamespace(device="cpu", dtype="float64"),
    )
    from tpen.checkpoint import save_checkpoint

    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=1,
        completed_updates=1,
        model=model,
        context=context,
        save_optimizer=False,
        save_trainer=False,
        save_sampler=False,
        save_rng=False,
    )
    eval_row = dict(row)
    eval_row.update(
        {
            "kind": "eval",
            "config": "configs/hooke2/eval.yaml",
            "checkpoint": 0,
            "stream": 0,
            "draws": 1,
            "run_id": "same-profile-eval",
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_file": str(checkpoint_dir / "model.pt"),
        }
    )
    eval_cfg = stage1_eval.build_eval_config(eval_row)
    stage1_eval._refuse_cross_variant_checkpoint(eval_cfg, eval_row)


@pytest.mark.parametrize("profile", HOOKE2_ACT50_ALL_PROFILES)
def test_act50_synthetic_producer_to_collector_requires_complete_a27(
    profile: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Exercise all six finite profiles through a bounded synthetic collector."""

    phase1_rows = _act50_phase1(profile)
    phase1_hash = manifest_hash(phase1_rows)
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows
        },
        "checkpoints": {
            f"hooke2:{lineage}:{checkpoint}": "c" * 64
            for lineage in range(STAGE1_PROTOCOL.lineages)
            for checkpoint in HOOKE2_ACT50_A27
        },
    }
    eval_rows = bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=list(HOOKE2_ACT50_A27),
        checkpoint_evidence=evidence,
        source_sha="a" * 40,
        config_sha256=config_identities(
            config_root=CONFIG_ROOT, systems=("hooke2",), science_profile=profile,
        ),
        protocol=STAGE1_PROTOCOL,
        phase1_manifest_hash=phase1_hash,
    )
    rows = phase1_rows + eval_rows
    document = {
        "schema": "stage1/v4",
        "manifest_hash": manifest_hash(rows),
        "rows": rows,
    }
    identity = {"schema": "tpen-config-identity-v1", "sha256": "b" * 64}
    physical_id = str(eval_rows[0]["expected_physical_system_id"])

    monkeypatch.setattr(stage1_collect, "inspect_eval_run", lambda path: {
        "verdict": "completed",
        "resolved_config_identity": identity,
        "source_git_sha": "a" * 40,
    })
    monkeypatch.setattr(stage1_collect, "_record_path", lambda path: path / "m.json")
    monkeypatch.setattr(
        stage1_collect,
        "inspect_trajectory_record",
        lambda path, **kwargs: SimpleNamespace(
            record_schema="trajectory_records/v2",
            draw_count=256,
            walker_count=4096,
            row_count=256 * 4096,
            finite_count=256 * 4096,
            mean=1.0,
            variance=0.0,
            csv_sha256="d" * 64,
            system_id=physical_id,
            execution_method_id="synthetic-act50",
            resolved_config_identity=identity,
            model_provenance="checkpoint",
            checkpoint_sha256=kwargs["expected_checkpoint_sha256"],
            record_content_id="e" * 64,
        ),
    )
    monkeypatch.setattr(stage1_collect, "read_energy_grid", lambda record: [[1.0]])
    monkeypatch.setattr(
        stage1_collect, "_reference", lambda system_id, registry_path: (-2.9, "synthetic")
    )

    result = stage1_collect.collect(document)
    assert len(result["records"]) == 3 * len(HOOKE2_ACT50_A27) * 4
    assert len(result["aggregates"]) == 3 * len(HOOKE2_ACT50_A27)

    missing_rows = list(rows)
    missing_rows.pop()
    missing = {
        "schema": "stage1/v4",
        "manifest_hash": manifest_hash(missing_rows),
        "rows": missing_rows,
    }
    with pytest.raises(stage1_collect.CollectionError, match="grid mismatch"):
        stage1_collect.collect(missing)
