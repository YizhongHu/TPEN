"""Checkpoint evidence and row selection through production entry points."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.atomistic.stage1.evidence import build_evidence
from experiments.atomistic.stage1.select_eval import select_rows
from experiments.atomistic.stage1.launch import _binding_token, command_for_row
from experiments.atomistic.stage1.protocol import (
    HE_H2_EVAL_CHECKPOINTS, HOOKE2_ACT50_SIGMOID_PROFILE, HOOKE2_ACT50_A27, MANIFEST_SCHEMA,
    STAGE1_PROTOCOL, build_manifest, config_identities, derive_smoke_request,
    manifest_hash, validate_gateway_binding,
)
from experiments.toolkit.run_paths import resolve_run_dir

ROOT = Path(__file__).parents[1]


def _document(rows, phase=1):
    return {"schema": MANIFEST_SCHEMA, "phase": phase,
            "manifest_hash": manifest_hash(rows), "rows": rows}


@pytest.fixture(params=[False, True], ids=["science", "smoke"])
def phase1(request, tmp_path):
    """Build real profile rows with tiny complete checkpoint artifacts."""
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL, source_sha=source,
        config_sha256=config_identities(config_root=ROOT / "configs", systems=("hooke2",),
                                       science_profile=HOOKE2_ACT50_SIGMOID_PROFILE),
        checkpoints=list(range(0, 50001, 1000)), lineage_seeds=[11, 12, 13],
        checkpoint_sha256={}, results_root=str(tmp_path / "science"), systems=("hooke2",),
        phase=1, science_profile=HOOKE2_ACT50_SIGMOID_PROFILE,
        evaluation_selection="A27", include_smoke=False, auto_overlay=True,
    )
    if request.param:
        rows = [derive_smoke_request(row, results_root=str(tmp_path / "smoke"), stop_at=1000)
                for row in rows]
    _write_checkpoints(rows, (0, 1000) if request.param else HOOKE2_ACT50_A27,
                       smoke=request.param)
    return _document(rows)


def _write_checkpoints(rows, roster, *, smoke=False):
    """Write small model-only artifacts shared by the profile fixtures."""
    from experiments.atomistic.stage1.train import build_train_config
    from tpen.checkpoint.hashing import checkpoint_hashes
    from tpen.checkpoint.manifest import CHECKPOINT_KIND, CHECKPOINT_SCHEMA_VERSION, CheckpointManifest
    from tpen.checkpoint.payload import MODEL_ONLY_PAYLOAD

    for row in rows:
        run = resolve_run_dir(row)
        run.mkdir(parents=True)
        status = "planned_yield" if smoke else "completed"
        (run / "status.json").write_text(json.dumps({"status": status}))
        (run / "run-outcome.json").write_text(json.dumps({
            "status": status, "completed_updates": 1000, "next_iteration": 1000,
        }))
        cfg = build_train_config(row)
        for step in roster:
            directory = run / "checkpoints" / f"step_{step:06d}"
            directory.mkdir(parents=True)
            payload = f"model-{row['lineage']}-{step}".encode()
            (directory / "model.pt").write_bytes(payload)
            hashes = checkpoint_hashes(cfg)
            hashes["model"] = hashlib.sha256(payload).hexdigest()
            CheckpointManifest(
                schema_version=CHECKPOINT_SCHEMA_VERSION, kind=CHECKPOINT_KIND,
                next_iteration=step, completed_updates=step, created_at_unix=0.0,
                files={"model": "model.pt"}, hashes=hashes, runtime={}, provenance={},
                payload=MODEL_ONLY_PAYLOAD.to_manifest(),
            ).write(directory / "manifest.json")
            (directory / "COMPLETE").write_text("fixture\n")


def _cli(module, *args):
    result = subprocess.run([sys.executable, "-m", f"experiments.atomistic.stage1.{module}",
                             *map(str, args)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_production_cli_roundtrip(phase1, tmp_path):
    """Evidence, planner and selector preserve bindings admitted by the gateway."""
    source, evidence, planned, selected = [tmp_path / name for name in
                                          ("phase1.json", "evidence.json", "phase2.json", "rows.jsonl")]
    source.write_text(json.dumps(phase1))
    _cli("evidence", "--phase1-manifest", source, "--output", evidence)
    before = evidence.read_bytes()
    _cli("plan", "--phase", 2, "--phase1-manifest", source,
         "--checkpoint-bindings", evidence, "--output", planned)
    assert evidence.read_bytes() == before
    _cli("select_eval", "--manifest", planned, "--all", "--output", selected)
    rows = [json.loads(line) for line in selected.read_text().splitlines()]
    expected = [row for row in json.loads(planned.read_text())["rows"] if row["kind"] == "eval"]
    assert rows == expected
    roster = {0, 1000} if phase1["rows"][0].get("evaluation_roster") else set(HOOKE2_ACT50_A27)
    assert {row["checkpoint"] for row in rows} == roster
    for row in rows:
        admitted = validate_gateway_binding(binding=_binding_token(row), expected_kinds={"eval"})
        assert admitted == row
        assert "--binding" in command_for_row(row, python=sys.executable,
                                               driver="experiments/atomistic/stage1/eval.py")
    first = rows[0]
    _cli("select_eval", "--manifest", planned, "--system", first["system"],
         "--lineage", first["lineage"], "--checkpoint", first["checkpoint"],
         "--stream", first["stream"], "--output", selected)
    assert json.loads(selected.read_text()) == first


@pytest.mark.parametrize("wrong_kind", [False, True], ids=["failed", "other-row-kind"])
def test_status_refusal(phase1, wrong_kind):
    assert build_evidence(phase1)["checkpoints"]
    run = resolve_run_dir(phase1["rows"][0])
    status = ("completed" if phase1["rows"][0].get("evaluation_roster") else "planned_yield") if wrong_kind else "failed"
    (run / "status.json").write_text(json.dumps({"status": status}))
    with pytest.raises(ValueError, match="expected status"):
        build_evidence(phase1)


@pytest.mark.parametrize("missing", ["COMPLETE", "model.pt", "manifest.json"])
def test_checkpoint_refusal(phase1, missing):
    evidence = build_evidence(phase1)
    row = phase1["rows"][0]
    step = min(int(key.split(":")[-1]) for key in evidence["checkpoints"])
    (resolve_run_dir(row) / "checkpoints" / f"step_{step:06d}" / missing).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        build_evidence(phase1)



def test_missing_checkpoint_directory(phase1):
    evidence = build_evidence(phase1)
    row = phase1["rows"][0]
    step = min(int(key.split(":")[-1]) for key in evidence["checkpoints"])
    directory = resolve_run_dir(row) / "checkpoints" / f"step_{step:06d}"
    directory.rename(directory.with_name(directory.name + "-unavailable"))
    with pytest.raises(FileNotFoundError, match="checkpoint directory"):
        build_evidence(phase1)


def test_manifest_refusal(phase1):
    assert build_evidence(phase1)["checkpoints"]
    phase1["manifest_hash"] = "f" * 64
    with pytest.raises(ValueError, match="manifest_hash"):
        build_evidence(phase1)


def test_roster_marker_refusal(phase1):
    assert build_evidence(phase1)["checkpoints"]
    phase1["rows"][0]["evaluation_roster"] = "unknown"
    with pytest.raises(ValueError, match="roster"):
        build_evidence(_document(phase1["rows"]))


def test_lineage_refusal(phase1):
    assert build_evidence(phase1)["checkpoints"]
    rows = copy.deepcopy(phase1["rows"])
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        build_evidence(_document(rows))
    if rows[0].get("evaluation_roster") is None:
        with pytest.raises(ValueError, match="incomplete|scope"):
            build_evidence(_document(phase1["rows"][:-1]))


def test_smoke_boundary_refusal(phase1):
    if not phase1["rows"][0].get("evaluation_roster"):
        return
    assert build_evidence(phase1)["checkpoints"]
    run = resolve_run_dir(phase1["rows"][0])
    (run / "run-outcome.json").write_text(json.dumps({
        "status": "planned_yield", "completed_updates": 999, "next_iteration": 999,
    }))
    with pytest.raises(ValueError, match="stop_at"):
        build_evidence(phase1)


@pytest.mark.parametrize("field,value", [("system", "he"), ("lineage", 1),
                                         ("checkpoint", 1000), ("stream", 1)])
def test_selector_filters(field, value):
    rows = [{"kind": "eval", "system": "hooke2", "lineage": 0, "checkpoint": 0, "stream": 0},
            {"kind": "eval", "system": "hooke2", "lineage": 0, "checkpoint": 0, "stream": 0}]
    rows[1][field] = value
    document = _document(rows, phase=2)
    assert select_rows(document, **{field: value}) == [rows[1]]
    with pytest.raises(ValueError, match="no matching"):
        select_rows(document, system="absent")
    document["manifest_hash"] = "f" * 64
    with pytest.raises(ValueError, match="manifest_hash"):
        select_rows(document)


def test_he_roster_b_roundtrip(tmp_path):
    """Select the He terminal roster-B row; refuse the old 50000 example."""
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL, source_sha=source_sha,
        config_sha256=config_identities(config_root=ROOT / "configs", systems=("he",)),
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1, 1000)),
        lineage_seeds=[11, 12, 13], checkpoint_sha256={},
        results_root=str(tmp_path / "he"), systems=("he",), phase=1,
        include_smoke=False, auto_overlay=True,
    )
    _write_checkpoints(rows, HE_H2_EVAL_CHECKPOINTS)
    source, evidence, planned, selected = [tmp_path / name for name in
                                          ("phase1.json", "evidence.json", "phase2.json", "row.json")]
    source.write_text(json.dumps(_document(rows)))
    _cli("evidence", "--phase1-manifest", source, "--output", evidence)
    _cli("plan", "--phase", 2, "--phase1-manifest", source,
         "--checkpoint-bindings", evidence, "--output", planned)
    arguments = ["--manifest", str(planned), "--system", "he", "--lineage", "0",
                 "--stream", "0", "--output", str(selected)]
    _cli("select_eval", *arguments, "--checkpoint", 100000)
    row = json.loads(selected.read_text())
    assert row["checkpoint"] == 100000
    assert validate_gateway_binding(binding=_binding_token(row), expected_kinds={"eval"}) == row
    before = selected.read_bytes()
    refused = subprocess.run(
        [sys.executable, "-m", "experiments.atomistic.stage1.select_eval",
         *arguments, "--checkpoint", "50000"], capture_output=True, text=True,
    )
    assert refused.returncode != 0
    assert "no matching eval rows" in refused.stderr
    assert selected.read_bytes() == before
