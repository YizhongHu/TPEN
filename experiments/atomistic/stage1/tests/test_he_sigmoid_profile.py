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
    CHECKPOINT_EVIDENCE_SCHEMA,
    HE_H2_EVAL_CHECKPOINTS,
    HE_SIGMOID_PROFILE,
    ProtocolError,
    STAGE1_PROTOCOL,
    _select_resolved_path,
    bind_evaluation_manifest,
    build_manifest,
    config_identities,
    manifest_hash,
    planned_physical_system_ids,
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
    phase1_rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=identities,
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1,
                               STAGE1_PROTOCOL.retention_cadence)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root="/external/he-sigmoid-test",
        systems=("he",),
        phase=1,
        device="cpu",
        science_profile=profile,
        include_smoke=False,
        auto_overlay=True,
    )
    if phase == 1:
        return phase1_rows
    phase1_hash = manifest_hash(phase1_rows)
    evidence = {
        "schema": CHECKPOINT_EVIDENCE_SCHEMA,
        "phase1_manifest_hash": phase1_hash,
        "phase1_train_row_hashes": {
            f"he:{row['lineage']}": manifest_hash([row])
            for row in phase1_rows if row["kind"] == "train"
        },
        "checkpoints": {
            f"he:{lineage}:{checkpoint}": "b" * 64
            for lineage in range(STAGE1_PROTOCOL.lineages)
            for checkpoint in checkpoints
        },
    }
    return bind_evaluation_manifest(
        phase1_rows=phase1_rows,
        checkpoints=checkpoints,
        checkpoint_evidence=evidence,
        expected_physical_system_id=planned_physical_system_ids(),
        source_sha="a" * 40,
        config_sha256=identities,
        protocol=STAGE1_PROTOCOL,
        phase1_manifest_hash=phase1_hash,
        device="cpu",
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


def _full_snapshot(cfg: Any) -> dict[str, Any]:
    """Return the complete unresolved config, not only the model subtree."""

    value = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(value, dict)
    return value


def _drop_snapshot_paths(snapshot: dict[str, Any], paths: set[str]) -> None:
    """Remove only explicitly row-owned paths before full snapshot comparison."""

    for path in paths:
        cursor: Any = snapshot
        parts = path.split(".")
        for part in parts[:-1]:
            assert isinstance(cursor, dict), path
            cursor = cursor[part]
        assert isinstance(cursor, dict), path
        cursor.pop(parts[-1])


def test_resolved_path_selector_supports_lists_and_rejects_missing_paths() -> None:
    """The resolved-config guard must understand model layer list indices."""

    resolved = {
        "model": {"layers": [{"mixing": {"activation": {"_target_": "torch.nn.Sigmoid"}}}]},
    }
    assert _select_resolved_path(
        resolved, "model.layers.0.mixing.activation._target_"
    ) == "torch.nn.Sigmoid"
    with pytest.raises(ProtocolError, match="resolved train config is missing"):
        _select_resolved_path(resolved, "model.layers.0.path_aggregation.activation._target_")
    with pytest.raises(ProtocolError, match="resolved train config is missing"):
        _select_resolved_path(resolved, "model.layers.1.mixing.activation._target_")


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


def test_native_he_build_preserves_full_config_outside_row_owned_setup() -> None:
    """Native composition is a full-config no-op beyond gateway-owned setup."""

    train_row = dict(next(row for row in _rows(None) if row["kind"] == "train"))
    # The manifest helper opts into diagnostics by default.  Remove that
    # independent row-owned overlay so this test isolates native model/config
    # composition from the overlay's expected callback and logger additions.
    for key in ("overlay", "overlay_sha256", "overlay_profile"):
        train_row.pop(key, None)
    native_train = build_train_config(train_row)
    raw_train = load_config(str(CONFIG_ROOT / "he" / "train.yaml"))
    raw_train_snapshot = _full_snapshot(raw_train)
    built_train_snapshot = _full_snapshot(native_train)

    # The builder intentionally owns these fields.  Check the one structural
    # train setup change explicitly, then remove only those paths from the
    # otherwise complete snapshots.
    assert built_train_snapshot["checkpoint_steps"] == raw_train_snapshot["checkpoint_steps"][1:]
    assert built_train_snapshot["callbacks"] == raw_train_snapshot["callbacks"] + [
        {
            "_target_": "tpen.callback.InitialCheckpoint",
            "output_dir": "${run.dir}/checkpoints",
        }
    ]
    train_owned_paths = {
        "runtime.seed", "runtime.device", "run.root", "run.run_id", "run.layout",
        "experiment.name", "experiment.sector", "sampler.n_walkers",
        "sampler.seed", "trainer.max_steps", "checkpoint_steps", "callbacks",
    }
    _drop_snapshot_paths(raw_train_snapshot, train_owned_paths)
    _drop_snapshot_paths(built_train_snapshot, train_owned_paths)
    assert OmegaConf.to_yaml(OmegaConf.create(raw_train_snapshot), resolve=False) == OmegaConf.to_yaml(
        OmegaConf.create(built_train_snapshot), resolve=False,
    )

    eval_row = next(row for row in _rows(None, phase=2) if row["kind"] == "eval")
    native_eval = stage1_eval.build_eval_config(eval_row)
    raw_eval = load_config(str(CONFIG_ROOT / "he" / "eval.yaml"))
    raw_eval_snapshot = _full_snapshot(raw_eval)
    built_eval_snapshot = _full_snapshot(native_eval)
    eval_owned_paths = {
        "runtime.seed", "runtime.device", "run.root", "run.run_id", "run.layout",
        "experiment.name", "experiment.sector", "evaluation_sampler.n_walkers",
        "evaluation_tasks.mcmc_energy.generator.n_draws", "load.path",
        "trajectory_identity.checkpoint_file",
    }
    _drop_snapshot_paths(raw_eval_snapshot, eval_owned_paths)
    _drop_snapshot_paths(built_eval_snapshot, eval_owned_paths)
    assert OmegaConf.to_yaml(OmegaConf.create(raw_eval_snapshot), resolve=False) == OmegaConf.to_yaml(
        OmegaConf.create(built_eval_snapshot), resolve=False,
    )


def test_he_sigmoid_manifest_identity_and_admission_are_he_only() -> None:
    rows = _rows(HE_SIGMOID_PROFILE)
    train_rows = [row for row in rows if row["kind"] == "train"]
    native_eval_rows = [row for row in _rows(None, phase=2) if row["kind"] == "eval"]
    sigmoid_eval_rows = [row for row in _rows(HE_SIGMOID_PROFILE, phase=2)
                         if row["kind"] == "eval"]
    assert {row["science_profile"] for row in train_rows} == {HE_SIGMOID_PROFILE}
    assert {row["evaluation_selection"] for row in train_rows} == {"B"}
    assert {row["evaluation_selection_id"] for row in train_rows} == {"he-h2-b-v1"}
    assert {row["evaluation_selection"] for row in native_eval_rows} == {
        row["evaluation_selection"] for row in sigmoid_eval_rows
    } == {"B"}
    assert {row["seed"] for row in train_rows} == {row["seed"] for row in _rows(None)
                                                    if row["kind"] == "train"}
    assert config_identities(config_root=CONFIG_ROOT, systems=("he",))["he"] != rows[0]["config_sha256"]

    with pytest.raises(ProtocolError, match="He sigmoid profile only admits the he system"):
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
