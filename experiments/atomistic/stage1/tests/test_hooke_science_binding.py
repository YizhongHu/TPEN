"""Focused contract tests for the Hooke H-v2-matched science profile."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import experiments.atomistic.stage1.collect as collect_module
import experiments.atomistic.stage1.protocol as protocol_module
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import _binding_token, command_for_row, row_overrides
from experiments.atomistic.stage1.protocol import (
    HOOKE_A37,
    HOOKE_EVAL_SELECTION,
    HOOKE_EVAL_SELECTION_ID,
    HOOKE_SCIENCE_PROFILE,
    HOOKE_TRAINING_SEED_RULE,
    ProtocolError,
    STAGE1_PROTOCOL,
    build_manifest,
    config_identities,
    sampler_seed,
    training_seed,
    validate_gateway_binding,
    validate_hooke_eval_grid,
    overlay_sha256,
)


ROOT = Path(__file__).parents[1]
HOOKE_OVERLAY = ROOT / "configs" / "overlays" / "hooke-diagnostics-v1.yaml"


@pytest.mark.parametrize("system", ["hooke1", "hooke2", "qho2e"])
def test_hooke_train_eval_shape_and_physics_are_explicit(system: str) -> None:
    train = OmegaConf.load(ROOT / "configs" / system / "train.yaml")
    evaluation = OmegaConf.load(ROOT / "configs" / system / "eval.yaml")
    for cfg in (train, evaluation):
        assert cfg.model.embedding.out_channels == 32
        assert cfg.model.embedding.hidden_channels == 128
        assert cfg.model.layers[0].mixing.channels == 32
        assert cfg.model.layers[0].path_aggregation.channels == 32
        assert cfg.model.readout.channels == 32
        assert cfg.model.readout.trainable is True
        assert cfg.model.factors[0].coefficient == 0.25
        assert cfg.model.factors[0].trainable is False
    if system == "hooke2":
        factor = train.model.factors[1]
        assert factor.same_spin_coefficient == 0.25
        assert factor.opposite_spin_coefficient == 0.5
        assert factor.range_parameter == 1.0
        assert factor.trainable_range is False
    else:
        assert len(train.model.factors) == 1
        assert train.model.factors[0]._target_ == "tpen.nn.GaussianConfinement"
    assert evaluation.evaluation_sampler.burn_in == 1000
    assert evaluation.evaluation_sampler.n_steps == 20


def test_sampler_seed_uses_the_versioned_domain_payload() -> None:
    payload = json.dumps(
        ["stage1/train-sampler-v1", "hooke1", 2, 13], separators=(",", ":")
    ).encode()
    expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)
    assert sampler_seed(system="hooke1", lineage=2, lineage_seed=13) == expected
    assert sampler_seed(system="hooke1", lineage=2, lineage_seed=13) != training_seed(
        system="hooke1", lineage=2, lineage_seed=13
    )


def _hooke_phase1() -> list[dict[str, object]]:
    return build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=ROOT / "configs", systems=("hooke1",)),
        checkpoints=list(range(0, 100001, 1000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/stage1",
        systems=("hooke1",),
        phase=1,
        science_profile=HOOKE_SCIENCE_PROFILE,
        evaluation_selection=HOOKE_EVAL_SELECTION,
        overlay=str(HOOKE_OVERLAY),
        overlay_sha256=overlay_sha256(HOOKE_OVERLAY),
    )


def test_hooke_phase1_rows_bind_distinct_sampler_seeds_and_gateway() -> None:
    rows = _hooke_phase1()
    train = next(row for row in rows if row["kind"] == "train")
    assert train["training_seed_rule"] == HOOKE_TRAINING_SEED_RULE
    assert train["evaluation_selection_id"] == HOOKE_EVAL_SELECTION_ID
    assert train["overlay"] == "configs/overlays/hooke-diagnostics-v1.yaml"
    assert train["seed"] != train["sampler_seed"]
    overrides = row_overrides(train)
    assert f"runtime.seed={train['seed']}" in overrides
    assert f"sampler.seed={train['sampler_seed']}" in overrides
    assert validate_gateway_binding(
        binding=_binding_token(train), config=str(train["config"]),
        overrides=overrides, kind="train", overlay=str(train["overlay"]),
        overlay_sha256=str(train["overlay_sha256"]),
    ) == train
    with pytest.raises(ProtocolError, match="sampler.seed"):
        validate_gateway_binding(
            binding=_binding_token(train), config=str(train["config"]),
            overrides=[item for item in overrides if not item.startswith("sampler.seed=")],
            kind="train", overlay=str(train["overlay"]),
            overlay_sha256=str(train["overlay_sha256"]),
        )
    tampered = dict(train)
    tampered["sampler_seed"] = tampered["seed"]
    with pytest.raises(ProtocolError, match="sampler seed does not match"):
        validate_gateway_binding(
            binding=_binding_token(tampered), config=str(tampered["config"]),
            overrides=row_overrides(tampered), kind="train",
            overlay=str(tampered["overlay"]),
            overlay_sha256=str(tampered["overlay_sha256"]),
        )

    tampered = dict(train)
    tampered["seed"] = int(tampered["seed"]) + 1
    with pytest.raises(ProtocolError, match="training seed does not match"):
        protocol_module._validate_hooke_science_train_row(tampered)
    with pytest.raises(ProtocolError, match="training seed does not match"):
        validate_gateway_binding(
            binding=_binding_token(tampered), config=str(tampered["config"]),
            overrides=row_overrides(tampered), kind="train",
            overlay=str(tampered["overlay"]),
            overlay_sha256=str(tampered["overlay_sha256"]),
        )


def test_hooke_train_gateway_rechecks_resolved_seed_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rendered Hooke gateway carries one self-contained binding."""

    train = next(row for row in _hooke_phase1() if row["kind"] == "train")
    command = command_for_row(train, python="/usr/bin/python3", driver="train.py")
    assert "--binding" in command
    assert all(flag not in command for flag in ("--config", "--kind", "--overlay"))


def test_hooke_train_gateway_rechecks_resolved_model_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public gateway has no repeated model override channel."""

    train = next(row for row in _hooke_phase1() if row["kind"] == "train")
    with pytest.raises(SystemExit):
        monkeypatch.setattr(
            sys, "argv", ["stage1-train", "--binding", _binding_token(train),
                           "model.readout.trainable=false"],
        )
        stage1_train.main()


def test_legacy_he_h2_rows_do_not_inherit_hooke_profile_fields() -> None:
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=config_identities(
            config_root=ROOT / "configs", systems=("he", "h2"),
        ),
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1, 1000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/stage1",
        systems=("he", "h2"),
        phase=1,
    )
    profile_fields = {
        "science_profile", "training_seed_rule", "lineage_seed", "sampler_seed",
        "evaluation_selection", "evaluation_selection_id", "selection_id",
    }
    assert all(profile_fields.isdisjoint(row) for row in rows)


def test_hooke_grid_rejects_count_preserving_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_a37 = list(range(0, 20001, 1000)) + list(range(25000, 100001, 5000))
    assert HOOKE_EVAL_SELECTION == "A37"
    rows = [
        {"kind": "eval", "system": "hooke1", "lineage": lineage,
         "checkpoint": checkpoint, "stream": stream,
         "science_profile": HOOKE_SCIENCE_PROFILE,
         "evaluation_selection": HOOKE_EVAL_SELECTION}
        for lineage in range(3) for checkpoint in expected_a37 for stream in range(4)
    ]
    validate_hooke_eval_grid(rows)
    rows[-1]["checkpoint"] = expected_a37[0]
    with pytest.raises(ProtocolError, match="A37 grid mismatch"):
        validate_hooke_eval_grid(rows)

    original_validator = protocol_module.validate_hooke_eval_grid

    def tamper_phase2_rows(rows: list[dict[str, object]]) -> None:
        tampered = [dict(row) for row in rows]
        tampered[-1]["checkpoint"] = HOOKE_A37[0]
        original_validator(tampered)

    monkeypatch.setattr(protocol_module, "validate_hooke_eval_grid", tamper_phase2_rows)
    checkpoint_sha256 = {
        ("hooke1", lineage, checkpoint): "b" * 64
        for lineage in range(3)
        for checkpoint in HOOKE_A37
    }
    with pytest.raises(ProtocolError, match="exact production protocol dimensions"):
        build_manifest(
            # A37 is a phase-2 selection, not the phase-1 retention cadence.
            protocol=replace(STAGE1_PROTOCOL, retention_cadence=None),
            source_sha="a" * 40,
            config_sha256={"hooke1": "c" * 64},
            checkpoints=list(HOOKE_A37),
            lineage_seeds=[11, 12, 13],
            checkpoint_sha256=checkpoint_sha256,
            results_root="/external/stage1",
            systems=("hooke1",),
            phase=2,
            science_profile=HOOKE_SCIENCE_PROFILE,
            evaluation_selection=HOOKE_EVAL_SELECTION,
            overlay=str(HOOKE_OVERLAY),
            overlay_sha256=overlay_sha256(HOOKE_OVERLAY),
            expected_physical_system_id={"hooke1": "d" * 64},
        )


def _two_hooke_phase2_document(tmp_path: Path) -> tuple[dict[str, object], object]:
    """Build a complete, planner-shaped two-Hooke A37 document for collection tests."""

    protocol = STAGE1_PROTOCOL
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
        overlay=str(HOOKE_OVERLAY),
        overlay_sha256=overlay_sha256(HOOKE_OVERLAY),
        expected_physical_system_id=physical_ids,
    )
    phase1_hash = protocol_module.manifest_hash(phase1_rows)
    evidence = {
        "schema": protocol_module.CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": protocol_module.manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"{system}:{lineage}:{checkpoint}": "c" * 64
            for system in systems for lineage in range(protocol.lineages)
            for checkpoint in HOOKE_A37
        },
    }
    eval_rows = protocol_module.bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=list(HOOKE_A37),
        checkpoint_evidence=evidence,
        expected_physical_system_id=physical_ids,
        source_sha="a" * 40,
        config_sha256=config_sha256,
        protocol=protocol,
        phase1_manifest_hash=phase1_hash,
    )
    rows = list(phase1_rows) + eval_rows
    return {
        "schema": "stage1/v3",
        "manifest_hash": protocol_module.manifest_hash(rows),
        "rows": rows,
    }, protocol


def _install_two_hooke_collection_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = {"schema": "tpen-config-identity-v1", "sha256": "d" * 64}
    monkeypatch.setattr(collect_module, "inspect_eval_run", lambda path: {
        "verdict": "completed",
        "resolved_config_identity": identity,
        "source_git_sha": "a" * 40,
    })
    monkeypatch.setattr(collect_module, "_record_path", lambda path: path / "m.json")

    def fake_record(path, **kwargs):
        physical_id = "b" * 64 if "hooke2" in str(path) else "a" * 64
        return type("Record", (), {
            "record_schema": "trajectory_records/v2",
            "draw_count": 256,
            "walker_count": 4096,
            "row_count": 256 * 4096,
            "finite_count": 256 * 4096,
            "mean": 2.5,
            "variance": 1.25,
            "csv_sha256": "e" * 64,
            "system_id": physical_id,
            "execution_method_id": "method",
            "resolved_config_identity": identity,
            "model_provenance": "checkpoint",
            "checkpoint_sha256": kwargs["expected_checkpoint_sha256"],
            "record_content_id": "f" * 64,
        })()

    monkeypatch.setattr(collect_module, "inspect_trajectory_record", fake_record)
    monkeypatch.setattr(collect_module, "read_energy_grid", lambda record: [[1.0, 2.0], [3.0, 4.0]])
    monkeypatch.setattr(collect_module, "_reference", lambda system_id, registry_path: (-2.9, "transcribed"))


@pytest.mark.parametrize("remove_all_eval_rows", [False, True])
def test_collection_requires_all_planned_hooke_systems_before_inspection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    remove_all_eval_rows: bool,
) -> None:
    """A complete control passes; missing one/all eval populations fail pre-I/O."""

    document, _ = _two_hooke_phase2_document(tmp_path)
    _install_two_hooke_collection_fakes(monkeypatch)
    control = collect_module.collect(document)
    assert len(control["records"]) == 2 * 3 * len(HOOKE_A37) * 4

    predicate = (
        (lambda row: row["kind"] != "eval")
        if remove_all_eval_rows
        else (lambda row: row["system"] != "hooke2" or row["kind"] != "eval")
    )
    tampered = dict(document)
    tampered["rows"] = [row for row in document["rows"] if predicate(row)]
    tampered["manifest_hash"] = protocol_module.manifest_hash(tampered["rows"])
    monkeypatch.setattr(
        collect_module,
        "inspect_eval_run",
        lambda path: pytest.fail("collection inspected a run before grid admission"),
    )
    with pytest.raises(collect_module.CollectionError, match="Hooke evaluation grid is incomplete"):
        collect_module.collect(tampered)


def test_phase2_rejects_evaluation_seed_collision_with_train_sampler_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Evaluation seeds must be disjoint from both train seed namespaces."""

    document, protocol = _two_hooke_phase2_document(tmp_path)
    phase1_rows = [row for row in document["rows"] if row["kind"] != "eval"]
    evidence = {
        "schema": protocol_module.CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": protocol_module.manifest_hash(phase1_rows),
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": protocol_module.manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"{system}:{lineage}:{checkpoint}": "c" * 64
            for system in ("hooke1", "hooke2") for lineage in range(protocol.lineages)
            for checkpoint in HOOKE_A37
        },
    }
    train_sampler_seed = next(row["sampler_seed"] for row in phase1_rows if row["kind"] == "train")
    original_evaluation_seed = protocol_module.evaluation_seed
    monkeypatch.setattr(
        protocol_module,
        "evaluation_seed",
        lambda *, system, lineage, checkpoint, stream: (
            train_sampler_seed
            if (system, lineage, checkpoint, stream) == ("hooke1", 0, HOOKE_A37[0], 0)
            else original_evaluation_seed(
                system=system, lineage=lineage, checkpoint=checkpoint, stream=stream
            )
        ),
    )
    with pytest.raises(ProtocolError, match="jointly unique"):
        protocol_module.bind_evaluation_manifest(
            phase1_rows=phase1_rows,
            checkpoints=list(HOOKE_A37),
            checkpoint_evidence=evidence,
            expected_physical_system_id={"hooke1": "a" * 64, "hooke2": "b" * 64},
            source_sha="a" * 40,
            config_sha256=config_identities(config_root=ROOT / "configs", systems=("hooke1", "hooke2")),
            protocol=protocol,
            phase1_manifest_hash=evidence["phase1_manifest_hash"],
        )


def test_phase2_rejects_full_row_tamper_even_with_recomputed_phase1_hash(
    tmp_path: Path,
) -> None:
    """Recomputing a phase-1 hash cannot rewrite the authenticated train row."""

    document, protocol = _two_hooke_phase2_document(tmp_path)
    phase1_rows = [dict(row) for row in document["rows"] if row["kind"] != "eval"]
    original_hash = protocol_module.manifest_hash(phase1_rows)
    evidence = {
        "schema": protocol_module.CHECKPOINT_EVIDENCE_SCHEMA,
        # The tampered caller may recompute this outer hash, but not the
        # independently recorded train-row identities.
        "phase1_manifest_hash": "d" * 64,
        "phase1_train_row_hashes": {
            f"{row['system']}:{row['lineage']}": protocol_module.manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"{system}:{lineage}:{checkpoint}": "c" * 64
            for system in ("hooke1", "hooke2")
            for lineage in range(protocol.lineages)
            for checkpoint in HOOKE_A37
        },
    }
    tampered = [dict(row) for row in phase1_rows]
    tampered_train = next(row for row in tampered if row["kind"] == "train")
    tampered_train["overlay_sha256"] = "0" * 64
    tampered_hash = protocol_module.manifest_hash(tampered)
    evidence["phase1_manifest_hash"] = tampered_hash
    with pytest.raises(ProtocolError, match="train identity"):
        protocol_module.bind_evaluation_manifest(
            phase1_rows=tampered,
            checkpoints=list(HOOKE_A37),
            checkpoint_evidence=evidence,
            expected_physical_system_id={"hooke1": "a" * 64, "hooke2": "b" * 64},
            config_sha256=config_identities(
                config_root=ROOT / "configs", systems=("hooke1", "hooke2")
            ),
            protocol=protocol,
            phase1_manifest_hash=tampered_hash,
        )
    assert original_hash != tampered_hash
