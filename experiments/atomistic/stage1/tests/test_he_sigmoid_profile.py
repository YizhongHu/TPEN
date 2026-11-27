"""Contracts for the versioned He sigmoid model profile."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tpen.run import load_config

from experiments.atomistic.stage1 import eval as stage1_eval
from experiments.atomistic.stage1.protocol import (
    HE_H2_EVAL_CHECKPOINTS,
    HE_SIGMOID_PROFILE,
    ProtocolError,
    STAGE1_PROTOCOL,
    build_manifest,
    config_identities,
)
from experiments.atomistic.stage1.train import build_train_config


CONFIG_ROOT = Path(__file__).parents[1] / "configs"


def _rows(profile: str | None, *, phase: int = 1) -> list[dict[str, Any]]:
    """Build native or sigmoid He rows with the production horizon."""

    checkpoints = (
        list(range(0, STAGE1_PROTOCOL.lineage_updates + 1,
                   STAGE1_PROTOCOL.retention_cadence))
        if phase == 1 else list(HE_H2_EVAL_CHECKPOINTS)
    )
    identities = config_identities(
        config_root=CONFIG_ROOT, systems=("he",), science_profile=profile,
    )
    checkpoint_sha256 = {
        ("he", lineage, checkpoint): "b" * 64
        for lineage in range(STAGE1_PROTOCOL.lineages)
        for checkpoint in checkpoints
    }
    return build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=identities,
        checkpoints=checkpoints,
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256=checkpoint_sha256,
        results_root="/external/he-sigmoid-test",
        systems=("he",),
        phase=phase,
        device="cpu",
        science_profile=profile,
        include_smoke=False,
        auto_overlay=True,
    )


def _diff_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    """Return paths whose resolved values differ between two mappings."""

    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in set(left) | set(right):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths |= _diff_paths(left[key], right[key], path)
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths: set[str] = set()
        for index in range(max(len(left), len(right))):
            path = f"{prefix}.{index}"
            if index >= len(left) or index >= len(right):
                paths.add(path)
            else:
                paths |= _diff_paths(left[index], right[index], path)
        return paths
    return set() if left == right else {prefix}


def _model(cfg: Any) -> dict[str, Any]:
    """Resolve only the model subtree used for profile comparison."""

    value = OmegaConf.to_container(cfg.model, resolve=True)
    assert isinstance(value, dict)
    return value


def test_he_sigmoid_rendered_train_and_eval_diff_only_activation_leaves() -> None:
    native_train = build_train_config(_rows(None)[0])
    sigmoid_train = build_train_config(_rows(HE_SIGMOID_PROFILE)[0])
    native_eval_row = next(row for row in _rows(None, phase=2) if row["kind"] == "eval")
    sigmoid_eval_row = next(row for row in _rows(HE_SIGMOID_PROFILE, phase=2)
                            if row["kind"] == "eval")
    native_eval = stage1_eval.build_eval_config(native_eval_row)
    sigmoid_eval = stage1_eval.build_eval_config(sigmoid_eval_row)

    expected = {
        "embedding.activation",
        "layers.0.mixing.activation._target_",
        "layers.0.path_aggregation.activation._target_",
    }
    assert _diff_paths(_model(native_train), _model(sigmoid_train)) == expected
    assert _diff_paths(_model(native_eval), _model(sigmoid_eval)) == expected
    assert _model(native_train) == _model(native_eval)
    assert _model(native_train) == _model(load_config(str(CONFIG_ROOT / "he" / "train.yaml")))
    assert _model(native_eval) == _model(load_config(str(CONFIG_ROOT / "he" / "eval.yaml")))
    assert OmegaConf.to_yaml(native_train.model, resolve=True) == OmegaConf.to_yaml(
        native_eval.model, resolve=True,
    )
    assert _model(native_train)["layers"][0]["mixing"]["activation"]["_target_"] == "torch.nn.SiLU"
    assert _model(native_train)["layers"][0]["path_aggregation"]["activation"]["_target_"] == "torch.nn.SiLU"
    assert _model(sigmoid_train)["embedding"]["activation"]["_target_"] == "torch.nn.Sigmoid"


def test_he_sigmoid_manifest_identity_and_admission_are_he_only() -> None:
    rows = _rows(HE_SIGMOID_PROFILE)
    train_rows = [row for row in rows if row["kind"] == "train"]
    assert {row["science_profile"] for row in train_rows} == {HE_SIGMOID_PROFILE}
    assert {row["evaluation_selection"] for row in train_rows} == {"B"}
    assert {row["evaluation_selection_id"] for row in train_rows} == {"he-h2-b-v1"}
    assert {row["seed"] for row in train_rows} == {row["seed"] for row in _rows(None)
                                                    if row["kind"] == "train"}
    assert config_identities(config_root=CONFIG_ROOT, systems=("he",))["he"] != rows[0]["config_sha256"]

    with pytest.raises(ProtocolError, match="He sigmoid profile"):
        build_manifest(
            protocol=STAGE1_PROTOCOL, source_sha="a" * 40,
            config_sha256=config_identities(config_root=CONFIG_ROOT, systems=("h2",)),
            checkpoints=list(range(0, 100001, 1000)), lineage_seeds=[11, 12, 13],
            checkpoint_sha256={}, results_root="/external/he-sigmoid-test",
            systems=("h2",), phase=1, science_profile=HE_SIGMOID_PROFILE,
            include_smoke=False, auto_overlay=True,
        )
    with pytest.raises(ProtocolError, match="unknown science profile"):
        build_manifest(
            protocol=STAGE1_PROTOCOL, source_sha="a" * 40,
            config_sha256=config_identities(config_root=CONFIG_ROOT, systems=("he",)),
            checkpoints=list(range(0, 100001, 1000)), lineage_seeds=[11, 12, 13],
            checkpoint_sha256={}, results_root="/external/he-sigmoid-test",
            systems=("he",), phase=1, science_profile="he-sigmoid-typo",
            include_smoke=False, auto_overlay=True,
        )


@pytest.mark.parametrize(
    ("source_profile", "target_profile"),
    ((None, HE_SIGMOID_PROFILE), (HE_SIGMOID_PROFILE, None)),
)
def test_he_eval_refuses_cross_profile_checkpoint_before_writes(
    source_profile: str | None, target_profile: str | None,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """He eval rejects both directions of native/sigmoid checkpoint mixing."""

    from tpen.checkpoint import save_checkpoint

    source_row = next(row for row in _rows(source_profile) if row["kind"] == "train")
    source_cfg = build_train_config(source_row)
    source_model = instantiate(source_cfg.model).to(dtype=torch.float64)
    source_context = SimpleNamespace(
        cfg=source_cfg, run_dir=tmp_path / "source-run",
        metadata=SimpleNamespace(device="cpu", dtype="float64"),
    )
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "source-checkpoints", next_iteration=1,
        completed_updates=1, model=source_model, context=source_context,
        save_optimizer=False, save_trainer=False, save_sampler=False, save_rng=False,
    )
    row = next(row for row in _rows(target_profile, phase=2) if row["kind"] == "eval")
    row = dict(row)
    row.update({
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": str(checkpoint_dir / "model.pt"),
        "results_root": str(tmp_path / "eval-results"),
    })
    monkeypatch.setattr(stage1_eval, "validate_gateway_binding", lambda **_: row)
    monkeypatch.setattr(stage1_eval, "run_from_config", lambda *_a, **_k: pytest.fail("eval ran"))
    monkeypatch.setattr(sys, "argv", ["eval.py", "--binding", "unused"])
    with pytest.raises(stage1_eval.EvalRunRefusal, match="different model profile"):
        stage1_eval.main()


@pytest.mark.parametrize(
    ("source_profile", "target_profile"),
    ((None, HE_SIGMOID_PROFILE), (HE_SIGMOID_PROFILE, None)),
)
def test_he_collect_refuses_cross_profile_checkpoint(
    source_profile: str | None, target_profile: str | None, tmp_path: Path,
) -> None:
    """Collection applies the same checkpoint/profile boundary as eval."""

    from experiments.atomistic.stage1 import collect as stage1_collect
    from tpen.checkpoint import save_checkpoint

    source_row = next(row for row in _rows(source_profile) if row["kind"] == "train")
    source_cfg = build_train_config(source_row)
    source_context = SimpleNamespace(
        cfg=source_cfg, run_dir=tmp_path / "source-run",
        metadata=SimpleNamespace(device="cpu", dtype="float64"),
    )
    source_model = instantiate(source_cfg.model).to(dtype=torch.float64)
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "source-checkpoints", next_iteration=1,
        completed_updates=1, model=source_model, context=source_context,
        save_optimizer=False, save_trainer=False, save_sampler=False, save_rng=False,
    )
    row = dict(next(row for row in _rows(target_profile, phase=2) if row["kind"] == "eval"))
    row.update({
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": str(checkpoint_dir / "model.pt"),
    })
    with pytest.raises(stage1_collect.CollectionError, match="does not match binding"):
        stage1_collect._validate_he_profile_checkpoint(row)
