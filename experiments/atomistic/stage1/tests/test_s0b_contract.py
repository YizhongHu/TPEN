"""Focused He/H2 roster-B and evaluation-contract tests."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from experiments.atomistic.stage1 import collect as collect_module
from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA,
    HE_H2_EVAL_CHECKPOINTS,
    HE_H2_EVAL_SELECTION,
    HE_H2_EVAL_SELECTION_ID,
    ProtocolError,
    STAGE1_PROTOCOL,
    bind_evaluation_manifest,
    config_identities,
    build_manifest,
    manifest_hash,
    validate_he_h2_eval_grid,
)
from tpen.run import load_config


ROOT = Path(__file__).parents[1]
CONFIG_ROOT = ROOT / "configs"
SOURCE_SHA = "a" * 40
PHYSICAL_ID = "c" * 64


def _phase2_document(tmp_path: Path, system: str = "he") -> dict[str, object]:
    """Build one production-shaped phase-1 plus roster-B document."""

    config_sha256 = config_identities(config_root=CONFIG_ROOT, systems=(system,))
    phase1_rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha=SOURCE_SHA,
        config_sha256=config_sha256,
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1,
                              STAGE1_PROTOCOL.retention_cadence)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path),
        systems=(system,),
        phase=1,
    )
    phase1_hash = manifest_hash(phase1_rows)
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{system}:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"{system}:{lineage}:{checkpoint}": PHYSICAL_ID
            for lineage in range(STAGE1_PROTOCOL.lineages)
            for checkpoint in HE_H2_EVAL_CHECKPOINTS
        },
    }
    eval_rows = bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=list(HE_H2_EVAL_CHECKPOINTS),
        checkpoint_evidence=evidence,
        expected_physical_system_id={system: PHYSICAL_ID},
        source_sha=SOURCE_SHA,
        config_sha256=config_sha256,
        protocol=STAGE1_PROTOCOL,
        phase1_manifest_hash=phase1_hash,
    )
    rows = list(phase1_rows) + eval_rows
    return {"schema": "stage1/v3", "manifest_hash": manifest_hash(rows), "rows": rows}


def test_he_h2_eval_configs_bind_burn_in_and_stride() -> None:
    for system in ("he", "h2"):
        cfg = load_config(str(CONFIG_ROOT / system / "eval.yaml"))
        assert cfg.evaluation_sampler.n_walkers == 4096
        assert cfg.evaluation_sampler.burn_in == 1000
        assert cfg.evaluation_sampler.n_steps == 20
        assert cfg.evaluation_tasks.mcmc_energy.generator.n_draws == 256


def test_production_phase2_emits_exact_roster_b_grid(tmp_path: Path) -> None:
    document = _phase2_document(tmp_path)
    eval_rows = [row for row in document["rows"] if row["kind"] == "eval"]
    assert len(eval_rows) == 3 * len(HE_H2_EVAL_CHECKPOINTS) * 4
    assert {row["checkpoint"] for row in eval_rows} == set(HE_H2_EVAL_CHECKPOINTS)
    assert {row["evaluation_selection"] for row in eval_rows} == {HE_H2_EVAL_SELECTION}
    assert {row["evaluation_selection_id"] for row in eval_rows} == {HE_H2_EVAL_SELECTION_ID}
    assert all(row["phase1_manifest_hash"] for row in eval_rows)
    assert all(row["phase1_train_row_hash"] for row in eval_rows)


@pytest.mark.parametrize(
    "checkpoints",
    [
        [50_000],
        list(HE_H2_EVAL_CHECKPOINTS[:-1]),
        list(HE_H2_EVAL_CHECKPOINTS) + [50_000],
    ],
    ids=["terminal-only", "missing-terminal", "out-of-roster"],
)
def test_production_phase2_rejects_non_b_checkpoint_selection(
    tmp_path: Path, checkpoints: list[int],
) -> None:
    document = _phase2_document(tmp_path)
    phase1_rows = [row for row in document["rows"] if row["kind"] != "eval"]
    phase1_hash = manifest_hash(phase1_rows)
    system = "he"
    config_sha256 = config_identities(config_root=CONFIG_ROOT, systems=(system,))
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{system}:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"{system}:{lineage}:{checkpoint}": PHYSICAL_ID
            for lineage in range(STAGE1_PROTOCOL.lineages)
            for checkpoint in checkpoints
        },
    }
    with pytest.raises(ProtocolError, match="roster-B"):
        bind_evaluation_manifest(
            phase1_rows=phase1_rows,
            checkpoints=checkpoints,
            checkpoint_evidence=evidence,
            expected_physical_system_id={system: PHYSICAL_ID},
            source_sha=SOURCE_SHA,
            config_sha256=config_sha256,
            protocol=STAGE1_PROTOCOL,
            phase1_manifest_hash=phase1_hash,
        )


@pytest.mark.parametrize("drop_count", [1, 4])
def test_roster_b_grid_rejects_131_or_128_rows(tmp_path: Path, drop_count: int) -> None:
    document = _phase2_document(tmp_path)
    eval_rows = [row for row in document["rows"] if row["kind"] == "eval"]
    validate_he_h2_eval_grid(eval_rows, expected_systems=("he",), expected_profile="B")
    missing = [
        (row["system"], row["lineage"], row["checkpoint"], row["stream"])
        for row in eval_rows[:drop_count]
    ]
    with pytest.raises(ProtocolError, match=r"^He/H2 roster-B grid mismatch: ") as excinfo:
        validate_he_h2_eval_grid(
            eval_rows[drop_count:], expected_systems=("he",), expected_profile="B"
        )
    message = str(excinfo.value)
    assert f"expected 132 rows, got {132 - drop_count}" in message
    assert f"missing={missing!r}" in message
    assert "extra=[]" in message


@pytest.mark.parametrize("checkpoint", [0, 100_000])
def test_roster_b_grid_requires_zero_and_terminal_coordinates(
    tmp_path: Path, checkpoint: int,
) -> None:
    document = _phase2_document(tmp_path)
    eval_rows = [row for row in document["rows"] if row["kind"] == "eval"]
    tampered = [row for row in eval_rows if row["checkpoint"] != checkpoint]
    with pytest.raises(ProtocolError, match="grid mismatch"):
        validate_he_h2_eval_grid(
            tampered, expected_systems=("he",), expected_profile="B"
        )


def test_collection_rejects_missing_checkpoint_group_before_run_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    document = _phase2_document(tmp_path)
    document["rows"] = [
        row for row in document["rows"]
        if row.get("kind") != "eval" or row.get("checkpoint") != 0
    ]
    document["manifest_hash"] = manifest_hash(document["rows"])
    monkeypatch.setattr(
        collect_module, "inspect_eval_run", lambda *_: pytest.fail("run I/O before grid check")
    )
    with pytest.raises(
        collect_module.CollectionError,
        match=r"^Hooke evaluation grid is incomplete: He/H2 roster-B grid mismatch: ",
    ) as excinfo:
        collect_module.collect(document)
    message = str(excinfo.value)
    assert "expected 132 rows, got 120" in message
    assert "missing=[('he', 0, 0, 0), ('he', 0, 0, 1), ('he', 0, 0, 2), ('he', 0, 0, 3)" in message
    assert "extra=[]" in message


@pytest.mark.parametrize(
    "path",
    [
        ROOT / "README.md",
        ROOT / "runbooks" / "helium.md",
        ROOT / "runbooks" / "h2-r1.4.md",
    ],
)
def test_semantic_docs_pin_roster_b_contract(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    semantic = text.split("## Identities and paths", 1)[0]
    assert re.search(r"(?<!\d)1,?000(?!\d)", semantic)
    assert re.search(r"(?<!\d)20(?!\d)", semantic)
    assert re.search(r"(?<!\d)132(?!\d)", semantic)
    if path.name == "README.md":
        he_h2_sampler_rows = "\n".join(
            line for line in semantic.splitlines()
            if "He eval sampler" in line or "H2 eval sampler" in line
        )
        assert "1000" in he_h2_sampler_rows
        assert "steps 20" in he_h2_sampler_rows
        assert "burn-in 100, steps 10" not in he_h2_sampler_rows
    else:
        assert "burn-in=1000" in semantic
        assert "stride=20" in semantic
        assert not re.search(r"burn-in[ =`]+100(?:\D|$)", semantic)
        assert not re.search(r"stride[ =`]+10(?:\D|$)", semantic)
