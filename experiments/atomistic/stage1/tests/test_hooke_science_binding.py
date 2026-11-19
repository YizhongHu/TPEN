"""Focused contract tests for the Hooke H-v2-matched science profile."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

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


def test_hooke_grid_rejects_count_preserving_replacement() -> None:
    rows = [
        {"kind": "eval", "system": "hooke1", "lineage": lineage,
         "checkpoint": checkpoint, "stream": stream,
         "science_profile": HOOKE_SCIENCE_PROFILE,
         "evaluation_selection": HOOKE_EVAL_SELECTION}
        for lineage in range(3) for checkpoint in HOOKE_A37 for stream in range(4)
    ]
    validate_hooke_eval_grid(rows)
    rows[-1]["checkpoint"] = HOOKE_A37[0]
    with pytest.raises(ProtocolError, match="A37 grid mismatch"):
        validate_hooke_eval_grid(rows)
