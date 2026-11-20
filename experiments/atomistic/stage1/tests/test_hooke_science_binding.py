"""Focused contract tests for the Hooke H-v2-matched science profile."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import experiments.atomistic.stage1.protocol as protocol_module
from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import _binding_token, row_overrides
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
)


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("system", ["hooke1", "hooke2"])
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
    )


def test_hooke_phase1_rows_bind_distinct_sampler_seeds_and_gateway() -> None:
    rows = _hooke_phase1()
    train = next(row for row in rows if row["kind"] == "train")
    assert train["training_seed_rule"] == HOOKE_TRAINING_SEED_RULE
    assert train["evaluation_selection_id"] == HOOKE_EVAL_SELECTION_ID
    assert train["seed"] != train["sampler_seed"]
    overrides = row_overrides(train)
    assert f"runtime.seed={train['seed']}" in overrides
    assert f"sampler.seed={train['sampler_seed']}" in overrides
    assert validate_gateway_binding(
        binding=_binding_token(train), config=str(train["config"]),
        overrides=overrides, kind="train",
    ) == train
    with pytest.raises(ProtocolError, match="sampler.seed"):
        validate_gateway_binding(
            binding=_binding_token(train), config=str(train["config"]),
            overrides=[item for item in overrides if not item.startswith("sampler.seed=")],
            kind="train",
        )
    tampered = dict(train)
    tampered["sampler_seed"] = tampered["seed"]
    with pytest.raises(ProtocolError, match="sampler seed does not match"):
        validate_gateway_binding(
            binding=_binding_token(tampered), config=str(tampered["config"]),
            overrides=row_overrides(tampered), kind="train",
        )

    tampered = dict(train)
    tampered["seed"] = int(tampered["seed"]) + 1
    with pytest.raises(ProtocolError, match="training seed does not match"):
        protocol_module._validate_hooke_science_train_row(tampered)
    with pytest.raises(ProtocolError, match="training seed does not match"):
        validate_gateway_binding(
            binding=_binding_token(tampered), config=str(tampered["config"]),
            overrides=row_overrides(tampered), kind="train",
        )


def test_hooke_train_gateway_rechecks_resolved_seed_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The producer boundary refuses a post-composition seed tamper."""

    train = next(row for row in _hooke_phase1() if row["kind"] == "train")
    overrides = row_overrides(train)
    tampered_overrides = [
        f"runtime.seed={int(train['seed']) + 1}"
        if override.startswith("runtime.seed=") else override
        for override in overrides
    ]
    with monkeypatch.context() as isolated:
        # Keep the gateway admission test focused: this isolates the second,
        # post-OmegaConf composition check owned by train.py.
        isolated.setattr(stage1_train, "validate_gateway_binding", lambda **_: train)
        isolated.setattr(
            stage1_train,
            "run_from_config",
            lambda **_: pytest.fail("seed-tampered config reached execution"),
        )
        isolated.setattr(
            sys,
            "argv",
            [
                "stage1-train", "--config", str(train["config"]), "--kind", "train",
                *tampered_overrides, "--binding", _binding_token(train),
            ],
        )
        with pytest.raises(ValueError, match="runtime and sampler seeds are not bound"):
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
    with pytest.raises(ProtocolError, match="A37 grid mismatch"):
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
            expected_physical_system_id={"hooke1": "d" * 64},
        )
