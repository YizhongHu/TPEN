"""Bounded delta probes for collection's phase-1 profile authority."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest

import experiments.atomistic.stage1.collect as collect_module
from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA,
    HOOKE_A37,
    HOOKE_EVAL_SELECTION,
    HOOKE_SCIENCE_PROFILE,
    STAGE1_PROTOCOL,
    bind_evaluation_manifest,
    build_manifest,
    config_identities,
    manifest_hash,
)


ROOT = Path(__file__).parents[1]


def _two_hooke_document(tmp_path: Path) -> dict[str, object]:
    protocol = replace(STAGE1_PROTOCOL, eval_walkers=2, eval_draws=2)
    systems = ("hooke1", "hooke2")
    config_sha256 = config_identities(config_root=ROOT / "configs", systems=systems)
    physical_ids = {"hooke1": "a" * 64, "hooke2": "b" * 64}
    phase1_rows = build_manifest(
        protocol=protocol,
        source_sha="a" * 40,
        config_sha256=config_sha256,
        checkpoints=list(range(0, 100001, 1000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path),
        systems=systems,
        phase=1,
        science_profile=HOOKE_SCIENCE_PROFILE,
        evaluation_selection=HOOKE_EVAL_SELECTION,
        expected_physical_system_id=physical_ids,
    )
    phase1_manifest_hash = manifest_hash(phase1_rows)
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_manifest_hash,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"{system}:{lineage}:{checkpoint}": "c" * 64
            for system in systems for lineage in range(protocol.lineages)
            for checkpoint in HOOKE_A37
        },
    }
    eval_rows = bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=list(HOOKE_A37),
        checkpoint_evidence=evidence,
        expected_physical_system_id=physical_ids,
        source_sha="a" * 40,
        config_sha256=config_sha256,
        protocol=protocol,
        phase1_manifest_hash=phase1_manifest_hash,
    )
    rows = list(phase1_rows) + eval_rows
    return {"schema": "stage1/v3", "manifest_hash": manifest_hash(rows), "rows": rows}


def _install_collection_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = {"schema": "tpen-config-identity-v1", "sha256": "d" * 64}
    monkeypatch.setattr(collect_module, "inspect_eval_run", lambda path: {
        "verdict": "completed",
        "resolved_config_identity": identity,
        "source_git_sha": "a" * 40,
    })
    monkeypatch.setattr(collect_module, "_record_path", lambda path: path / "metadata.json")
    monkeypatch.setattr(
        collect_module,
        "inspect_trajectory_record",
        lambda path, **kwargs: SimpleNamespace(
            record_schema="trajectory_records/v2",
            draw_count=2,
            walker_count=2,
            row_count=4,
            finite_count=4,
            mean=2.5,
            variance=1.25,
            csv_sha256="e" * 64,
            system_id="b" * 64 if "hooke2" in str(path) else "a" * 64,
            execution_method_id="method",
            resolved_config_identity=identity,
            model_provenance="checkpoint",
            checkpoint_sha256=kwargs["expected_checkpoint_sha256"],
            record_content_id="f" * 64,
        ),
    )
    monkeypatch.setattr(collect_module, "read_energy_grid", lambda record: [[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(collect_module, "_reference", lambda system_id, registry_path: (-2.9, "transcribed"))


def test_complete_two_hooke_profile_control_collects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    document = _two_hooke_document(tmp_path)
    _install_collection_fakes(monkeypatch)
    result = collect_module.collect(document)
    assert len(result["records"]) == 2 * 3 * len(HOOKE_A37) * 4


@pytest.mark.parametrize("mutation", ["absent", "none"])
def test_mixed_hooke_train_profiles_reject_before_inspection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str,
) -> None:
    document = _two_hooke_document(tmp_path)
    rows = [dict(row) for row in document["rows"]]
    phase1_rows = [row for row in rows if row["kind"] != "eval"]
    target = next(row for row in phase1_rows if row["kind"] == "train" and row["system"] == "hooke1")
    if mutation == "absent":
        target.pop("science_profile")
    else:
        target["science_profile"] = None

    phase1_manifest_hash = manifest_hash(phase1_rows)
    train_by_id = {row["row_id"]: row for row in phase1_rows if row["kind"] == "train"}
    for row in rows:
        if row["kind"] == "eval":
            row["phase1_manifest_hash"] = phase1_manifest_hash
            row["phase1_train_row_hash"] = manifest_hash([train_by_id[row["depends_on"]]])
    tampered = {"schema": "stage1/v3", "manifest_hash": manifest_hash(rows), "rows": rows}
    monkeypatch.setattr(
        collect_module,
        "inspect_eval_run",
        lambda path: pytest.fail("mixed profile manifest reached inspect_eval_run"),
    )
    with pytest.raises(collect_module.CollectionError, match="mix science and legacy profiles"):
        collect_module.collect(tampered)
