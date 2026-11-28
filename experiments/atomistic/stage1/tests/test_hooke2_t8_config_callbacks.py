"""Acceptance tests for the Hooke-2e T8 arm callback contract."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import train as stage1_train
from experiments.atomistic.stage1.launch import _binding_token
from experiments.atomistic.stage1.protocol import (
    Protocol,
    build_manifest,
    derive_smoke_request,
    resolve_config_path,
)
from experiments.toolkit.run_paths import resolve_run_dir


CONFIG_ROOT = Path(__file__).parents[1] / "configs" / "hooke2"
_EXCLUDED_HOOKE2_CONFIGS = {
    "train.yaml": "canonical baseline used to define the parity contract",
    "eval.yaml": "evaluation-only config, not a train arm",
}


def _discover_runnable_hooke2_configs(config_root: Path = CONFIG_ROOT) -> tuple[Path, ...]:
    """Discover train-arm configs while documenting non-arm YAMLs explicitly."""

    configs = tuple(sorted(config_root.glob("*.yaml")))
    assert configs, f"no Hooke-2 Stage-1 configs found under {config_root}"
    return tuple(
        path for path in configs if path.name not in _EXCLUDED_HOOKE2_CONFIGS
    )


def _assert_arm_checkpoint_contract(config_root: Path) -> None:
    """Require every discovered arm to match the canonical train callbacks."""

    standard = OmegaConf.load(config_root / "train.yaml")
    expected_callbacks = OmegaConf.to_container(standard.callbacks, resolve=False)
    for path in _discover_runnable_hooke2_configs(config_root):
        cfg = OmegaConf.load(path)
        steps = cfg.get("checkpoint_steps")
        callbacks = cfg.get("callbacks")
        assert steps is not None, f"{path.name} is missing checkpoint_steps"
        assert callbacks is not None, f"{path.name} is missing callbacks"
        assert list(steps) == list(standard.checkpoint_steps), path.name
        assert OmegaConf.to_container(callbacks, resolve=False) == expected_callbacks, path.name

        checkpoint = next(
            callback for callback in callbacks
            if callback._target_ == "tpen.callback.Checkpoint"
        )
        assert checkpoint.schedule._target_ == "tpen.checkpoint.ExplicitUpdates"
        assert list(checkpoint.schedule.updates) == list(standard.checkpoint_steps)
        assert checkpoint.payload._target_ == "tpen.checkpoint.TrainResume"


def _planned_train_row(tmp_path: Path, arm: str) -> dict[str, object]:
    """Build one production-shaped train row and derive its bounded smoke."""

    planned = build_manifest(
        protocol=Protocol(
            smoke_updates=1,
            smoke_walkers=2,
            lineage_updates=3,
            lineage_walkers=2,
            lineages=3,
            eval_streams=4,
            eval_walkers=2,
            eval_draws=2,
        ),
        source_sha="a" * 40,
        # This callback contract test intentionally uses the legacy digest
        # shape: the opt-in arm is not the canonical Stage-1 train.yaml.
        config_sha256="b" * 64,
        checkpoints=[0, 1, 2, 3],
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path / "planned"),
        systems=("hooke2",),
        phase=1,
        include_smoke=False,
    )
    source = next(row for row in planned if row["kind"] == "train")
    source["config"] = f"configs/hooke2/{arm}.yaml"
    return derive_smoke_request(
        source,
        results_root=str(tmp_path / f"smoke-{arm}"),
        stop_at=1,
    )


def test_t8_arms_match_train_checkpoint_contract() -> None:
    """Every discovered arm retains train.yaml's callback contract."""

    _assert_arm_checkpoint_contract(CONFIG_ROOT)


def test_every_t8_arm_declares_a_train_resume_checkpoint() -> None:
    """The parity guard checks every discovered arm for a TrainResume callback."""

    train = OmegaConf.load(CONFIG_ROOT / "train.yaml")
    for path in _discover_runnable_hooke2_configs():
        cfg = OmegaConf.load(path)
        callbacks = cfg.get("callbacks")
        assert callbacks is not None, f"{path.name} is missing callbacks"
        assert list(cfg.checkpoint_steps) == list(train.checkpoint_steps)
        checkpoints = [
            callback for callback in callbacks
            if callback._target_ == "tpen.callback.Checkpoint"
        ]
        assert len(checkpoints) == 1, path.name
        assert checkpoints[0].payload._target_ == "tpen.checkpoint.TrainResume"
        assert list(checkpoints[0].schedule.updates) == list(cfg.checkpoint_steps)


def test_new_runnable_arm_without_callbacks_fails_contract(tmp_path: Path) -> None:
    """An unlisted arm without callbacks is rejected by the inventory guard."""

    config_root = tmp_path / "hooke2"
    config_root.mkdir()
    for name in ("train.yaml", "eval.yaml"):
        shutil.copyfile(CONFIG_ROOT / name, config_root / name)
    (config_root / "new_arm.yaml").write_text(
        "experiment: {name: new_arm}\ncheckpoint_steps: [1000]\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="new_arm.yaml is missing callbacks"):
        _assert_arm_checkpoint_contract(config_root)


def test_experiment_test_respects_core_import_boundary() -> None:
    """This experiments test uses only the documented public execution boundary."""

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    forbidden = {"tpen.checkpoint", "tpen.run"}
    direct_core_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in forbidden
    }
    assert direct_core_imports == set()


@pytest.mark.parametrize(
    "arm", [path.stem for path in _discover_runnable_hooke2_configs()],
)
def test_native_train_planned_yield_publishes_train_resume_checkpoint(
    arm: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native train gateway must yield only after a full checkpoint is published."""

    row = _planned_train_row(tmp_path, arm)
    assert resolve_config_path(row["config"]).is_file()

    monkeypatch.setattr(
        stage1_train.sys,
        "argv",
        ["stage1-train", "--binding", _binding_token(row)],
    )
    exit_code = stage1_train.main()
    run_dir = resolve_run_dir(row)
    outcome = json.loads((run_dir / "run-outcome.json").read_text(encoding="utf-8"))
    # At the unfixed head, the same correctly bounded row reaches PlannedYield
    # without a publisher and must expose this exact refusal to the control.
    if exit_code != 2:
        assert outcome["error"] == (
            "planned yield boundary lacks a published full-state checkpoint"
        )
    assert exit_code == 2, outcome  # documented planned_yield process exit code
    assert outcome["status"] == "planned_yield"
    stop_at = int(row["execution"]["stop_at"])
    assert outcome["next_iteration"] == stop_at
    assert outcome["completed_updates"] == stop_at
    checkpoint_ref = outcome["checkpoint_ref"]
    assert checkpoint_ref["profile"] == "train_resume"
    checkpoint = run_dir / "checkpoints" / f"step_{stop_at:06d}"
    assert checkpoint_ref["next_iteration"] == stop_at
    assert checkpoint_ref["completed_updates"] == stop_at
    assert Path(checkpoint_ref["checkpoint_dir"]) == checkpoint
    assert checkpoint.is_dir()
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["payload"]["profile"] == "train_resume"
    assert (checkpoint / "COMPLETE").is_file()
