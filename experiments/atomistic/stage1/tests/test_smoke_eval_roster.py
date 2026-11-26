"""Smoke-prefix evaluation-roster admission and collection contracts."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.atomistic.stage1 import collect as collect_module
from experiments.atomistic.stage1 import train as train_module
from experiments.atomistic.stage1.launch import _binding_token
from experiments.atomistic.stage1.protocol import (
    CHECKPOINT_EVIDENCE_SCHEMA,
    HOOKE2_ACT50_A27,
    HOOKE2_ACT50_SIGMOID_PROFILE,
    HOOKE2_ACT50_NOLAYER_PROFILE,
    HOOKE_SCIENCE_PROFILE,
    SMOKE_EVAL_ROSTER_MARKER,
    STAGE1_PROTOCOL,
    ProtocolError,
    bind_evaluation_manifest,
    build_manifest,
    config_identities,
    derive_smoke_request,
    manifest_hash,
    smoke_eval_roster,
    stable_row_id,
    validate_gateway_binding,
)


ROOT = Path(__file__).parents[1]
SOURCE_SHA = "a" * 40
CONFIG_ROOT = ROOT / "configs"


def _science_rows(tmp_path: Path, *, profile: str) -> list[dict[str, object]]:
    checkpoints = list(range(0, 50_001 if profile.startswith("hooke2-") else 100_001, 1000))
    return build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha=SOURCE_SHA,
        config_sha256=config_identities(
            config_root=CONFIG_ROOT, systems=("hooke2",),
            science_profile=profile if profile.startswith("hooke2-") else None,
        ),
        checkpoints=checkpoints,
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path / "science"),
        systems=("hooke2",),
        phase=1,
        science_profile=profile,
        evaluation_selection="A27" if profile.startswith("hooke2-") else "A37",
        include_smoke=False,
        auto_overlay=True,
    )


def _smoke_phase1(tmp_path: Path, *, profile: str, lineages: int = 3) -> list[dict[str, object]]:
    source_rows = _science_rows(tmp_path, profile=profile)
    return [
        derive_smoke_request(
            row,
            results_root=str(tmp_path / "smoke"),
            stop_at=1_000,
        )
        for row in source_rows
        if row["kind"] == "train" and int(row["lineage"]) < lineages
    ]


def _evidence(phase1_rows: list[dict[str, object]], checkpoints: tuple[int, ...]) -> dict[str, object]:
    return {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": manifest_hash(phase1_rows),
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows
        },
        "checkpoints": {
            f"{row['system']}:{row['lineage']}:{checkpoint}": "c" * 64
            for row in phase1_rows
            for checkpoint in smoke_eval_roster(row)
        },
    }


@pytest.mark.parametrize("profile", (HOOKE_SCIENCE_PROFILE, HOOKE2_ACT50_SIGMOID_PROFILE))
def test_derived_roster_is_exact_prefix_and_marker_is_identity_bearing(
    tmp_path: Path, profile: str,
) -> None:
    phase1 = _smoke_phase1(tmp_path, profile=profile)
    assert all(row["evaluation_roster"] == SMOKE_EVAL_ROSTER_MARKER for row in phase1)
    assert all(smoke_eval_roster(row) == (0, 1_000) for row in phase1)
    assert all("evaluation_roster" not in row for row in _science_rows(tmp_path, profile=profile))
    markerless = copy.deepcopy(phase1)
    for row in markerless:
        row.pop("evaluation_roster")
    assert manifest_hash(phase1) != manifest_hash(markerless)


def test_derive_refuses_empty_roster_and_unknown_marker(tmp_path: Path) -> None:
    source = _science_rows(tmp_path, profile=HOOKE2_ACT50_SIGMOID_PROFILE)[0]
    source["checkpoint_steps"] = [1, 50_000]
    with pytest.raises(ProtocolError, match="roster is empty"):
        derive_smoke_request(source, results_root=str(tmp_path / "smoke"), stop_at=1)

    source = _science_rows(tmp_path, profile=HOOKE2_ACT50_SIGMOID_PROFILE)[0]
    source["evaluation_roster"] = "smoke-prefix-v2"
    with pytest.raises(ProtocolError, match="unsupported evaluation_roster"):
        derive_smoke_request(source, results_root=str(tmp_path / "smoke"), stop_at=1_000)


def test_smoke_binding_uses_only_phase1_lineages_and_planned_streams(tmp_path: Path) -> None:
    phase1 = _smoke_phase1(tmp_path, profile=HOOKE2_ACT50_SIGMOID_PROFILE, lineages=1)
    evidence = _evidence(phase1, HOOKE2_ACT50_A27)
    bound = bind_evaluation_manifest(
        phase1_rows=phase1,
        checkpoints=list(HOOKE2_ACT50_A27),
        checkpoint_evidence=evidence,
        expected_physical_system_id={"hooke2": "d" * 64},
        source_sha=SOURCE_SHA,
        config_sha256=config_identities(
            config_root=CONFIG_ROOT, systems=("hooke2",),
            science_profile=HOOKE2_ACT50_SIGMOID_PROFILE,
        ),
        protocol=replace(STAGE1_PROTOCOL, eval_walkers=2, eval_draws=3),
        phase1_manifest_hash=manifest_hash(phase1),
    )
    assert len(bound) == 1 * 2 * 4
    assert {row["lineage"] for row in bound} == {0}
    assert {row["stream"] for row in bound} == {0, 1, 2, 3}
    assert all(row["evaluation_roster"] == SMOKE_EVAL_ROSTER_MARKER for row in bound)


def test_science_row_cannot_claim_smoke_roster(tmp_path: Path) -> None:
    phase1 = _science_rows(tmp_path, profile=HOOKE2_ACT50_SIGMOID_PROFILE)
    for row in phase1:
        row["evaluation_roster"] = SMOKE_EVAL_ROSTER_MARKER
    phase1_hash = manifest_hash(phase1)
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in phase1
        },
        "checkpoints": {
            f"hooke2:{row['lineage']}:{checkpoint}": "c" * 64
            for row in phase1 for checkpoint in (0, 1_000)
        },
    }
    with pytest.raises(ProtocolError, match="requires execution.stop_at"):
        bind_evaluation_manifest(
            phase1_rows=phase1,
            checkpoints=list(HOOKE2_ACT50_A27),
            checkpoint_evidence=evidence,
            expected_physical_system_id={"hooke2": "d" * 64},
            config_sha256=config_identities(
                config_root=CONFIG_ROOT, systems=("hooke2",),
                science_profile=HOOKE2_ACT50_SIGMOID_PROFILE,
            ),
        )


@pytest.mark.parametrize("marker", (SMOKE_EVAL_ROSTER_MARKER, "smoke-prefix-v2"))
def test_train_gateway_rejects_marker_before_config_or_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, marker: str,
) -> None:
    """Only a derived train request may reach the runner with a roster marker."""

    science = next(
        row for row in _science_rows(tmp_path, profile=HOOKE2_ACT50_SIGMOID_PROFILE)
        if row["kind"] == "train"
    )
    forged = copy.deepcopy(science)
    forged["evaluation_roster"] = marker
    if marker == SMOKE_EVAL_ROSTER_MARKER:
        forged.pop("execution", None)
    monkeypatch.setattr(
        train_module, "build_train_config",
        lambda *_args, **_kwargs: pytest.fail("marker refusal must precede config composition"),
    )
    with monkeypatch.context() as isolated:
        isolated.setattr(sys, "argv", ["stage1-train", "--binding", _binding_token(forged)])
        with pytest.raises(ProtocolError, match="evaluation_roster|execution.stop_at"):
            train_module.main()

    derived = derive_smoke_request(
        science, results_root=str(tmp_path / "smoke"), stop_at=1_000,
    )
    assert validate_gateway_binding(
        binding=_binding_token(derived), expected_kinds={"train"},
    ) == derived


def _independent_evidence(
    phase1_rows: list[dict[str, object]], checkpoints: tuple[int, ...],
) -> dict[str, object]:
    """Build checkpoint evidence without consulting the roster under test."""

    return {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": manifest_hash(phase1_rows),
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows
        },
        "checkpoints": {
            f"{row['system']}:{row['lineage']}:{checkpoint}": "c" * 64
            for row in phase1_rows
            for checkpoint in checkpoints
        },
    }


@pytest.mark.parametrize("stop_at", (0, 50_000))
def test_saved_smoke_admission_rejects_nonpositive_or_terminal_stop(
    tmp_path: Path, stop_at: int,
) -> None:
    """Binder, planner, and collector all reject forged saved boundaries."""

    phase1 = _smoke_phase1(tmp_path, profile=HOOKE2_ACT50_SIGMOID_PROFILE)
    for row in phase1:
        row["execution"] = {"stop_at": stop_at}
    evidence = _independent_evidence(phase1, tuple(
        checkpoint for checkpoint in HOOKE2_ACT50_A27 if checkpoint <= stop_at
    ))
    with pytest.raises(ProtocolError, match="stop_at"):
        bind_evaluation_manifest(
            phase1_rows=phase1,
            checkpoints=list(HOOKE2_ACT50_A27),
            checkpoint_evidence=evidence,
            expected_physical_system_id={"hooke2": "d" * 64},
            source_sha=SOURCE_SHA,
            config_sha256=config_identities(
                config_root=CONFIG_ROOT, systems=("hooke2",),
                science_profile=HOOKE2_ACT50_SIGMOID_PROFILE,
            ),
            protocol=replace(STAGE1_PROTOCOL, eval_walkers=2, eval_draws=3),
            phase1_manifest_hash=manifest_hash(phase1),
        )

    phase1_path = tmp_path / f"phase1-{stop_at}.json"
    phase1_path.write_text(json.dumps({
        "schema": "stage1/v4", "phase": 1,
        "manifest_hash": manifest_hash(phase1), "rows": phase1,
    }) + "\n", encoding="utf-8")
    evidence_path = tmp_path / f"evidence-{stop_at}.json"
    evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    output = tmp_path / f"phase2-{stop_at}.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "plan.py"), "--output", str(output),
         "--phase1-manifest", str(phase1_path), "--checkpoint-bindings", str(evidence_path)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "stop_at" in result.stderr
    assert not output.exists()

    document = {
        "schema": "stage1/v4", "manifest_hash": manifest_hash(phase1), "rows": phase1,
    }
    with pytest.raises(collect_module.CollectionError, match="stop_at"):
        collect_module.collect(document)


def test_saved_smoke_admission_accepts_positive_nonterminal_control(tmp_path: Path) -> None:
    """A valid 1000-step derived boundary remains admitted everywhere."""

    phase1 = _smoke_phase1(tmp_path, profile=HOOKE2_ACT50_SIGMOID_PROFILE, lineages=1)
    evidence = _evidence(phase1, HOOKE2_ACT50_A27)
    bound = bind_evaluation_manifest(
        phase1_rows=phase1,
        checkpoints=list(HOOKE2_ACT50_A27),
        checkpoint_evidence=evidence,
        expected_physical_system_id={"hooke2": "d" * 64},
        source_sha=SOURCE_SHA,
        config_sha256=config_identities(
            config_root=CONFIG_ROOT, systems=("hooke2",),
            science_profile=HOOKE2_ACT50_SIGMOID_PROFILE,
        ),
        protocol=replace(STAGE1_PROTOCOL, eval_walkers=2, eval_draws=3),
        phase1_manifest_hash=manifest_hash(phase1),
    )
    assert {row["checkpoint"] for row in bound} == {0, 1_000}


def test_collect_accepts_smoke_roster_and_refuses_science_roster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    phase1 = _smoke_phase1(tmp_path, profile=HOOKE2_ACT50_NOLAYER_PROFILE)
    evidence = _evidence(phase1, HOOKE2_ACT50_A27)
    small_protocol = replace(STAGE1_PROTOCOL, eval_walkers=2, eval_draws=3)
    eval_rows = bind_evaluation_manifest(
        phase1_rows=phase1,
        checkpoints=list(HOOKE2_ACT50_A27),
        checkpoint_evidence=evidence,
        expected_physical_system_id={"hooke2": "d" * 64},
        config_sha256=config_identities(
            config_root=CONFIG_ROOT, systems=("hooke2",),
            science_profile=HOOKE2_ACT50_NOLAYER_PROFILE,
        ),
        protocol=small_protocol,
        phase1_manifest_hash=manifest_hash(phase1),
    )
    rows = phase1 + eval_rows
    identity = {"schema": "tpen-config-identity-v1", "sha256": "e" * 64}
    monkeypatch.setattr(collect_module, "inspect_eval_run", lambda _path: {
        "verdict": "completed", "resolved_config_identity": identity, "source_git_sha": SOURCE_SHA,
    })
    monkeypatch.setattr(collect_module, "_record_path", lambda path: path / "metadata.json")
    monkeypatch.setattr(
        collect_module, "inspect_trajectory_record",
        lambda _path, **kwargs: SimpleNamespace(
            record_schema="trajectory_records/v2", draw_count=3, walker_count=2,
            row_count=6, finite_count=6, mean=1.0, variance=0.0,
            csv_sha256="f" * 64, system_id="d" * 64,
            execution_method_id="fixture", resolved_config_identity=identity,
            model_provenance="checkpoint", checkpoint_sha256=kwargs["expected_checkpoint_sha256"],
            record_content_id="a" * 64,
        ),
    )
    monkeypatch.setattr(collect_module, "read_energy_grid", lambda _record: [[1.0, 1.0]] * 3)
    monkeypatch.setattr(collect_module, "_reference", lambda _system_id, _registry_path: (-2.9, "fixture"))
    document = {
        "schema": "stage1/v4", "manifest_hash": manifest_hash(rows), "rows": rows,
    }
    collected = collect_module.collect(document)
    assert len(collected["records"]) == 3 * 2 * 4

    extra = copy.deepcopy(eval_rows[0])
    extra["checkpoint"] = 25_000
    extra["row_id"] = stable_row_id(system="hooke2", kind="eval", lineage=0,
                                     checkpoint=25_000, stream=0)
    tampered = phase1 + eval_rows + [extra]
    tampered_document = {
        "schema": "stage1/v4", "manifest_hash": manifest_hash(tampered), "rows": tampered,
    }
    with pytest.raises(collect_module.CollectionError, match="smoke-prefix-v1"):
        collect_module.collect(tampered_document)
