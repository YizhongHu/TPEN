"""Contracts for the configured Stage-1 train-resume gateway."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1 import restart, train as stage1_train
from experiments.atomistic.stage1.launch import row_overrides
from experiments.atomistic.stage1.protocol import (
    HOOKE_SCIENCE_PROFILE,
    STAGE1_PROTOCOL,
    Protocol,
    build_manifest,
    config_identities,
    derive_smoke_request,
    overlay_sha256,
    overlay_path_for_system,
)
from experiments.atomistic.stage1.train import build_train_config
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.inspection import inspect_run
from tpen.run import run_from_config


CONFIG_ROOT = Path(__file__).parents[1] / "configs"
OVERLAY = CONFIG_ROOT / "overlays" / "diagnostics-v1.yaml"
HOOKE_OVERLAY = CONFIG_ROOT / "overlays" / "hooke-diagnostics-v1.yaml"
REPO_ROOT = Path(__file__).resolve().parents[4]


def _cfg(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "experiment": {"name": "stage1_he", "sector": "atomistic"},
            "run": {"root": str(tmp_path), "run_id": "run-0", "layout": "nested"},
            "system": {"id": "he_atom"},
            "trainer": {"max_steps": 100000},
            "runtime": {"seed": 11},
            "overlay_marker": "preserve-me",
            "runner": {"_target_": "tpen.runner.Train"},
            "callbacks": [
                {"_target_": "tpen.callback.InitialCheckpoint", "output_dir": "old"},
                {"_target_": "tpen.callback.ParameterActivity", "output_path": "old-activity"},
            ],
        }
    )


def _row(tmp_path: Path) -> dict[str, object]:
    return {
        "row_id": "he-lineage-0",
        "system": "he",
        "system_id": "he_atom",
        "lineage": 0,
        "seed": 11,
        "results_root": str(tmp_path),
        "experiment_name": "stage1_he",
        "sector": "atomistic",
        "run_id": "run-0",
        "updates": 100000,
        "config_sha256": "a" * 64,
    }


def _evidence(run_dir: Path, checkpoint_dir: Path, *, step: int = 0) -> object:
    return SimpleNamespace(
        verdict="incomplete",
        refusal=None,
        checkpoint={
            "checkpoint_dir": str(checkpoint_dir),
            "next_iteration": step,
            "completed_updates": step,
        },
        checkpoint_ref=None,
        run_dir=str(run_dir),
    )


def test_train_cli_rejects_removed_restart_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """LR-3 keeps no second CLI path for selecting a restart."""

    monkeypatch.setattr(sys, "argv", ["stage1-train", "--binding", "ignored", "--restart"])
    with pytest.raises(SystemExit):
        stage1_train.main()


def test_missing_resume_refuses_before_runner_call(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    calls: list[object] = []
    run_dir = Path(tmp_path) / "stage1_he" / "atomistic" / "run-0"
    checkpoint = run_dir / "checkpoints" / "step_000000"
    checkpoint.mkdir(parents=True)
    refusal = SimpleNamespace(
        verdict="refused",
        refusal={"code": "missing-checkpoint", "reason": "checkpoint is absent"},
        checkpoint=None,
        checkpoint_ref=None,
    )
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "inspect_run", lambda _path: _evidence(run_dir, checkpoint))
    monkeypatch.setattr(restart, "run_from_config", lambda *_a, **_k: calls.append(True) or 0)

    with pytest.raises(restart.RestartRefusal, match="checkpoint is absent"):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: refusal,
        )
    assert calls == []


def test_row_update_horizon_mismatch_refuses_before_runner_call(
    tmp_path: Path, monkeypatch,
) -> None:
    """A row/config H disagreement is rejected before inspection or launch."""

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    row["updates"] = 99_999
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "run_from_config", lambda *_a, **_k: calls.append(True) or 0)

    with pytest.raises(restart.RestartRefusal, match="horizon") as error:
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: pytest.fail("inspection must not run"),
        )
    assert error.value.cause is restart.RestartCause.HORIZON_MISMATCH
    assert calls == []


def test_fresh_train_stop_uses_runtime_planned_yield_without_changing_cfg(
    tmp_path: Path, monkeypatch,
) -> None:
    """A qualification stop is an operational callback, not a config edit."""

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    row["checkpoint_steps"] = [0, 1_000, 100_000]
    row["execution"] = {"stop_at": 1_000}
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    captured: dict[str, object] = {}

    def fake_run(run_cfg, **kwargs):
        captured["cfg"] = OmegaConf.to_container(run_cfg, resolve=True)
        captured.update(kwargs)
        return 2

    assert restart.execute_train(
        cfg=cfg, row=row, config_path="train.yaml", run=fake_run,
    ) == 2
    callbacks = captured["runtime_callbacks"]
    assert isinstance(callbacks, tuple)
    callback = callbacks[0]
    assert type(callback).__name__ == "PlannedYield"
    assert callback.target_iteration == 1_000
    assert callback.target_iterations == 100_000
    resolved = captured["cfg"]
    assert isinstance(resolved, dict)
    assert resolved["trainer"]["max_steps"] == 100_000


def test_fresh_recorder_overlay_reaches_runner_before_continuation_guard(
    tmp_path: Path, monkeypatch,
) -> None:
    """A fresh recorder-enabled row is not mistaken for a continuation."""

    row = _real_restart_row(tmp_path / "fresh", run_id="fresh-recorder")
    config_path = CONFIG_ROOT / "he" / "train.yaml"
    cfg = build_train_config(
        row,
        config_path=config_path,
        overrides=row_overrides(row),
        overlay=str(OVERLAY),
        overlay_sha256=overlay_sha256(OVERLAY),
    )
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    def fake_run(*_args: object, **_kwargs: object) -> int:
        calls.append(True)
        return 0

    assert restart.execute_train(
        cfg=cfg, row=row, config_path=config_path, run=fake_run,
    ) == 0
    assert calls == [True]


def test_reached_prefix_stop_refuses_before_runner_call(tmp_path: Path) -> None:
    """A published prefix cannot be asked to stop at its already-reached boundary."""

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    row["checkpoint_steps"] = [0, 1_000, 100_000]
    row["execution"] = {"stop_at": 1_000}
    run_dir = Path(tmp_path) / "stage1_he" / "atomistic" / "run-0"
    checkpoint = run_dir / "checkpoints" / "step_001000"
    checkpoint.mkdir(parents=True)
    evidence = _evidence(run_dir, checkpoint, step=1_000)

    with pytest.raises(restart.RestartRefusal, match="stale or already reached"):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: evidence,
            run=lambda *_a, **_k: pytest.fail("stale prefix launched"),
        )


def test_existing_recorder_overlay_refuses_before_inspection_or_runner(
    tmp_path: Path, monkeypatch,
) -> None:
    """A recorder-enabled continuation remains blocked at the Stage-1 boundary."""

    row = _real_restart_row(tmp_path / "continuation", run_id="recorder-continuation")
    config_path = CONFIG_ROOT / "he" / "train.yaml"
    cfg = build_train_config(
        row,
        config_path=config_path,
        overrides=row_overrides(row),
        overlay=str(OVERLAY),
        overlay_sha256=overlay_sha256(OVERLAY),
    )
    resolve_run_dir(row).mkdir(parents=True)
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    with pytest.raises(restart.RestartRefusal, match="7eaf9e0c.*diagnostics-v1"):
        restart.execute_train(
            cfg=cfg,
            row=row,
            config_path=config_path,
            inspector=lambda _path: pytest.fail("recorder refusal must precede inspection"),
            run=lambda *_a, **_k: calls.append(True) or 0,
        )
    assert calls == []


def test_recorder_configured_refuses_before_runner_call(tmp_path: Path, monkeypatch) -> None:
    """Deferred recorder resume is loud and never reaches inspection or launch."""

    cfg = _cfg(tmp_path)
    cfg.stage1_overlay = {"path": "configs/overlays/diagnostics-v1.yaml"}
    cfg.callbacks.append({
        "_target_": "tpen.callback.RawTrainingBatchRecorder",
        "path": "old-training-history",
    })
    row = _row(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "run_from_config", lambda *_a, **_k: calls.append(True) or 0)

    with pytest.raises(restart.RestartRefusal, match="7eaf9e0c.*diagnostics-v1"):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: pytest.fail("recorder refusal must precede inspection"),
        )
    assert calls == []


def test_exact_runner_load_and_overlay_survive_configured_restore(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = Path(tmp_path) / "stage1_he" / "atomistic" / "run-0"
    checkpoint = run_dir / "checkpoints" / "step_000000"
    checkpoint.mkdir(parents=True)
    observed: dict[str, object] = {}
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    def fake_run(run_cfg, **kwargs):
        observed["load"] = OmegaConf.to_container(run_cfg.runner.load, resolve=True)
        observed["overlay_marker"] = run_cfg.overlay_marker
        observed["callback_targets"] = [item["_target_"] for item in run_cfg.callbacks]
        observed["kwargs"] = kwargs
        return 0

    result = restart.execute_restart(
        cfg=cfg,
        row=row,
        config_path="train.yaml",
        inspector=lambda path: _evidence(run_dir, checkpoint),
        run=fake_run,
    )

    assert observed["load"] == {"mode": "train_resume", "path": str(checkpoint.resolve())}
    assert observed["overlay_marker"] == "preserve-me"
    assert observed["callback_targets"] == ["tpen.callback.ParameterActivity"]
    assert observed["kwargs"] == {"config_path": "train.yaml", "command": "stage1 restart"}
    assert result.runner_load == observed["load"]
    assert result.exit_code == 0


def test_row_and_config_run_identity_are_preserved(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    cfg.run.run_id = "different-run"
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    with pytest.raises(restart.RestartRefusal, match="run directory"):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: pytest.fail("inspection must not run"),
            run=lambda *_a, **_k: calls.append(True) or 0,
        )
    assert calls == []


def test_manifest_system_identity_refuses_before_runner_call(tmp_path: Path, monkeypatch) -> None:
    """A row cannot resume a checkpoint under another Hamiltonian identity."""

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    row["system_id"] = "h2_molecule"
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    with pytest.raises(restart.RestartRefusal, match="system identity") as error:
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: pytest.fail("identity refusal must precede inspection"),
            run=lambda *_a, **_k: calls.append(True) or 0,
        )
    assert error.value.cause is restart.RestartCause.INVALID_RESUME
    assert calls == []


def test_valid_but_different_system_identity_refuses_before_runner_call(
    tmp_path: Path, monkeypatch,
) -> None:
    """Two valid Stage-1 systems cannot be mixed during a requested resume."""

    cfg = _cfg(tmp_path)
    cfg.system.id = "h2_molecule"
    row = _row(tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    with pytest.raises(restart.RestartRefusal, match=r"he \(he_atom\).*h2 \(h2_molecule\)") as error:
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: pytest.fail("identity refusal must precede inspection"),
            run=lambda *_a, **_k: calls.append(True) or 0,
        )
    assert error.value.cause is restart.RestartCause.INVALID_RESUME
    assert calls == []


def test_requested_resume_never_falls_back_to_fresh(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    refusal = SimpleNamespace(
        verdict="refused",
        refusal={"code": "missing-run-dir", "reason": "run directory not found"},
        checkpoint=None,
        checkpoint_ref=None,
    )
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    with pytest.raises(restart.RestartRefusal) as error:
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: refusal,
            run=lambda *_a, **_k: pytest.fail("fresh fallback"),
        )
    assert error.value.cause is restart.RestartCause.MISSING_RESUME


def test_incompatible_checkpoint_counters_refuse_before_runner(tmp_path: Path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = Path(tmp_path) / "stage1_he" / "atomistic" / "run-0"
    checkpoint = run_dir / "checkpoints" / "step_100001"
    checkpoint.mkdir(parents=True)
    evidence = _evidence(run_dir, checkpoint, step=100001)
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)

    with pytest.raises(restart.RestartRefusal, match="incompatible"):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: evidence,
            run=lambda *_a, **_k: pytest.fail("incompatible resume launched"),
        )


def test_full_horizon_checkpoint_is_incompatible_resume(tmp_path: Path, monkeypatch) -> None:
    """R2 keeps a generic incompatible-input refusal, not terminal lineage classification."""

    cfg = _cfg(tmp_path)
    row = _row(tmp_path)
    run_dir = Path(tmp_path) / "stage1_he" / "atomistic" / "run-0"
    checkpoint = run_dir / "checkpoints" / "step_100000"
    checkpoint.mkdir(parents=True)
    evidence = _evidence(run_dir, checkpoint, step=100000)
    monkeypatch.setattr(restart, "validate_resolved_train_config", lambda *_a, **_k: None)
    monkeypatch.setattr(restart, "run_from_config", lambda *_a, **_k: pytest.fail("full horizon launched"))

    with pytest.raises(restart.RestartRefusal, match="full Stage-1 horizon") as error:
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path="train.yaml",
            inspector=lambda _path: evidence,
        )
    assert error.value.cause is restart.RestartCause.INVALID_RESUME


def _real_restart_row(
    tmp_path: Path, *, run_id: str, lineage_updates: int = 2,
) -> dict[str, object]:
    """Build a bounded He row retaining the committed diagnostics overlay."""

    protocol = Protocol(
        smoke_updates=1,
        smoke_walkers=2,
        lineage_updates=lineage_updates,
        lineage_walkers=2,
        lineages=3,
        eval_streams=4,
        eval_walkers=2,
        eval_draws=1,
        retention_cadence=1,
    )
    rows = build_manifest(
        protocol=protocol,
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=("he",)),
        checkpoints=list(range(lineage_updates + 1)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path),
        systems=("he",),
        phase=1,
        device="cpu",
        overlay=OVERLAY,
        overlay_sha256=overlay_sha256(OVERLAY),
    )
    row = next(item for item in rows if item["kind"] == "train")
    row["run_id"] = run_id
    return row


def _real_hooke_restart_row(tmp_path: Path, *, run_id: str) -> dict[str, object]:
    """Build one production-shaped Hooke row for overlay-composition refusal."""

    rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=CONFIG_ROOT, systems=("hooke1",)),
        checkpoints=list(range(0, STAGE1_PROTOCOL.lineage_updates + 1, 1000)),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path),
        systems=("hooke1",),
        phase=1,
        device="cpu",
        overlay=HOOKE_OVERLAY,
        overlay_sha256=overlay_sha256(HOOKE_OVERLAY),
        science_profile=HOOKE_SCIENCE_PROFILE,
        evaluation_selection="A37",
    )
    row = next(item for item in rows if item["kind"] == "train")
    row["run_id"] = run_id
    return row


@pytest.mark.parametrize(
    ("label", "system", "config_name", "overlay", "expected_overlay_fragment"),
    (
        ("protocol-default", "he", "he", overlay_path_for_system("he"), "diagnostics-v1"),
        ("diagnostics-v1", "he", "he", str(OVERLAY), "diagnostics-v1"),
        ("hooke-diagnostics-v1", "hooke1", "hooke1", str(HOOKE_OVERLAY), "hooke-diagnostics-v1"),
    ),
)
def test_real_recorder_compositions_refuse_before_runner_call(
    tmp_path: Path,
    monkeypatch,
    label: str,
    system: str,
    config_name: str,
    overlay: str,
    expected_overlay_fragment: str,
) -> None:
    """Every committed recorder-enabled train composition is refused loudly."""

    if system == "he":
        row = _real_restart_row(tmp_path / label, run_id=f"{label}-run")
    else:
        row = _real_hooke_restart_row(tmp_path / label, run_id=f"{label}-run")
    config_path = CONFIG_ROOT / config_name / "train.yaml"
    cfg = build_train_config(
        row,
        config_path=config_path,
        overrides=row_overrides(row),
        overlay=overlay,
        overlay_sha256=overlay_sha256(overlay),
    )
    assert any(
        callback.get("_target_") == "tpen.callback.RawTrainingBatchRecorder"
        for callback in cfg.callbacks
    )
    calls: list[object] = []
    monkeypatch.setattr(restart, "run_from_config", lambda *_a, **_k: calls.append(True) or 0)

    with pytest.raises(
        restart.RestartRefusal,
        match=rf"7eaf9e0c.*{expected_overlay_fragment}",
    ):
        restart.execute_restart(
            cfg=cfg,
            row=row,
            config_path=config_path,
            overlay=overlay,
            overlay_sha256=overlay_sha256(overlay),
            inspector=lambda _path: pytest.fail("recorder refusal must precede inspection"),
        )
    assert calls == []


def _assert_nested_state_equal(actual: object, expected: object, *, path: str = "$") -> None:
    """Compare nested checkpoint state with a path to the first difference."""

    import numpy as np
    import torch

    def mismatch(message: str) -> None:
        raise AssertionError(f"{path}: {message}")

    if torch.is_tensor(actual) or torch.is_tensor(expected):
        if not torch.is_tensor(actual) or not torch.is_tensor(expected):
            mismatch("one value is a tensor and the other is not")
        if actual.dtype != expected.dtype:
            mismatch(f"dtype differs ({actual.dtype!s} != {expected.dtype!s})")
        if actual.device != expected.device:
            mismatch(f"device differs ({actual.device!s} != {expected.device!s})")
        if actual.shape != expected.shape:
            mismatch(f"shape differs ({tuple(actual.shape)!r} != {tuple(expected.shape)!r})")
        if not torch.equal(actual, expected):
            mismatch("tensor values differ")
        return

    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if not isinstance(actual, np.ndarray) or not isinstance(expected, np.ndarray):
            mismatch("one value is a NumPy array and the other is not")
        if actual.dtype != expected.dtype:
            mismatch(f"dtype differs ({actual.dtype!s} != {expected.dtype!s})")
        if actual.shape != expected.shape:
            mismatch(f"shape differs ({actual.shape!r} != {expected.shape!r})")
        if not np.array_equal(actual, expected):
            mismatch("NumPy array values differ")
        return

    if dataclasses.is_dataclass(actual) or dataclasses.is_dataclass(expected):
        if not dataclasses.is_dataclass(actual) or not dataclasses.is_dataclass(expected):
            mismatch("one value is a dataclass and the other is not")
        if type(actual) is not type(expected):
            mismatch(f"types differ ({type(actual)!r} != {type(expected)!r})")
        for field in dataclasses.fields(actual):
            _assert_nested_state_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
                path=f"{path}.{field.name}",
            )
        return

    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
            mismatch("one value is a mapping and the other is not")
        if actual.keys() != expected.keys():
            mismatch(f"keys differ ({list(actual)!r} != {list(expected)!r})")
        for key in actual:
            _assert_nested_state_equal(
                actual[key], expected[key], path=f"{path}[{key!r}]"
            )
        return

    if isinstance(actual, bytes) or isinstance(expected, bytes):
        if type(actual) is not type(expected) or actual != expected:
            mismatch("bytes differ")
        return

    if (
        (isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)))
        or (isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)))
    ):
        if not isinstance(actual, Sequence) or not isinstance(expected, Sequence):
            mismatch("one value is a sequence and the other is not")
        if type(actual) is not type(expected):
            mismatch(f"types differ ({type(actual)!r} != {type(expected)!r})")
        if len(actual) != len(expected):
            mismatch(f"lengths differ ({len(actual)} != {len(expected)})")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_state_equal(
                actual_item, expected_item, path=f"{path}[{index}]"
            )
        return

    if type(actual) is not type(expected):
        mismatch(f"types differ ({type(actual)!r} != {type(expected)!r})")
    try:
        equal = actual == expected
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        mismatch(f"comparison raised {type(exc).__name__}: {exc}")
    if not isinstance(equal, (bool, np.bool_)) or not bool(equal):
        mismatch(f"values differ ({actual!r} != {expected!r})")


def _assert_checkpoint_payload_equal(left: Path, right: Path) -> frozenset[object]:
    """Compare serialized tensor/state payloads, not archive container bytes."""

    import torch

    left_payload = torch.load(left, map_location="cpu", weights_only=False)
    right_payload = torch.load(right, map_location="cpu", weights_only=False)
    if not isinstance(left_payload, Mapping) or not isinstance(right_payload, Mapping):
        raise AssertionError(f"{left.name}: checkpoint payloads must be mappings")
    left_keys = frozenset(left_payload)
    right_keys = frozenset(right_payload)
    if left_keys != right_keys:
        raise AssertionError(f"{left.name}: top-level keys differ ({left_keys!r} != {right_keys!r})")
    _assert_nested_state_equal(left_payload, right_payload, path=left.name)
    return left_keys


def test_nested_state_equal_handles_checkpoint_like_payload() -> None:
    """The parity helper handles every container kind used by checkpoint state."""

    import numpy as np
    import torch

    @dataclasses.dataclass(frozen=True)
    class SamplerState:
        positions: object

    expected = {
        "numpy": ("MT19937", np.array([1, 2, 3], dtype=np.uint32)),
        "torch": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "dataclass": SamplerState(positions=(b"state", None)),
    }
    actual = {
        "numpy": ("MT19937", np.array([1, 2, 3], dtype=np.uint32)),
        "torch": torch.tensor([1.0, 2.0], dtype=torch.float64),
        "dataclass": SamplerState(positions=(b"state", None)),
    }

    _assert_nested_state_equal(actual, expected, path="payload")

    actual["numpy"] = ("MT19937", np.array([1, 9, 3], dtype=np.uint32))
    with pytest.raises(AssertionError, match=r"payload\['numpy'\]\[1\]"):
        _assert_nested_state_equal(actual, expected, path="payload")


def _real_restart_child() -> None:
    """Run one derived-prefix, resumed, or uninterrupted phase in a new interpreter."""

    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    row = json.loads(Path(payload["row_path"]).read_text(encoding="utf-8"))
    recorder_free_row = {**row, "overlay": None, "overlay_sha256": None}
    config_path = Path(str(payload["config_path"]))
    phase = str(payload["phase"])
    effective_row = (
        recorder_free_row
        if phase == "fresh"
        else {key: value for key, value in recorder_free_row.items() if key != "execution"}
    )
    cfg = build_train_config(
        effective_row,
        config_path=config_path,
        overrides=row_overrides(effective_row),
        overlay=str(OVERLAY),
        overlay_sha256=overlay_sha256(OVERLAY),
    )
    # The diagnostics overlay remains bound and visible, but this parity arm
    # deliberately exercises the recorder-free Stage-1 configuration while
    # core item 7eaf9e0c owns recorder resume semantics.
    cfg.callbacks = [
        callback for callback in cfg.callbacks
        if callback.get("_target_") != "tpen.callback.RawTrainingBatchRecorder"
    ]
    if phase == "fresh":
        code = restart.execute_train(
            cfg=cfg,
            row=effective_row,
            config_path=config_path,
            command="stage1 real restart fresh",
            run=run_from_config,
        )
        evidence = inspect_run(resolve_run_dir(effective_row))
    elif phase in {"resumed", "horizon-mismatch"}:
        code = restart.execute_train(
            cfg=cfg,
            row=effective_row,
            config_path=config_path,
            command="stage1 real restart resumed",
            run=run_from_config,
        )
        evidence = inspect_run(resolve_run_dir(effective_row))
    elif phase == "reference":
        code = restart.execute_train(
            cfg=cfg,
            row=effective_row,
            config_path=config_path,
            command="stage1 real restart reference",
            run=run_from_config,
        )
        evidence = inspect_run(resolve_run_dir(effective_row))
    else:
        raise ValueError(f"unknown phase: {phase}")
    run_dir = resolve_run_dir(effective_row)
    outcome_doc = json.loads((run_dir / "run-outcome.json").read_text(encoding="utf-8"))
    result: dict[str, object] = {
        "outcome": evidence.outcome,
        "exit_code": code,
        "inspection": evidence.to_dict(),
        "target_iterations": outcome_doc["target_iterations"],
        "error": outcome_doc.get("error"),
        "exception_type": outcome_doc.get("exception_type"),
    }
    if phase == "resumed":
        result["runner_load"] = {
            "mode": "train_resume",
            "path": str(evidence.checkpoint["checkpoint_dir"]),
        }
    Path(str(payload["diagnostics_path"])).write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


def _run_real_restart_child(payload: dict[str, object]) -> dict[str, object]:
    """Launch one real phase and return its final JSON result."""

    row_path = Path(str(payload["row_path"]))
    phase = str(payload["phase"])
    argument_path = row_path.with_name(f"{row_path.stem}.{phase}.args.json")
    stdout_path = row_path.with_name(f"{row_path.stem}.{phase}.stdout")
    stderr_path = row_path.with_name(f"{row_path.stem}.{phase}.stderr")
    diagnostics_path = row_path.with_name(f"{row_path.stem}.{phase}.diagnostics.json")
    child_payload = {**payload, "diagnostics_path": str(diagnostics_path)}
    argument_path.write_text(json.dumps(child_payload, sort_keys=True) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from experiments.atomistic.stage1.tests.test_stage1_restart "
                "import _real_restart_child; _real_restart_child()"
            ),
            str(argument_path),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if diagnostics_path.is_file():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        run_dir = Path(str(diagnostics["inspection"]["run_dir"]))
        evidence_root = os.environ.get("VERIFY_EVIDENCE_DIR")
        if evidence_root is None and os.environ.get("RUN_ROOT"):
            evidence_root = str(Path(os.environ["RUN_ROOT"]) / "evidence")
        if evidence_root:
            preserved = Path(evidence_root) / "stage1-restart" / phase
            preserved.mkdir(parents=True, exist_ok=True)
            for name in ("run-outcome.json", "status.json", "resolved_config.yaml", "error.json"):
                source = run_dir / name
                if source.is_file():
                    shutil.copyfile(source, preserved / name)
            shutil.copyfile(diagnostics_path, preserved / "diagnostics.json")
            shutil.copyfile(stdout_path, preserved / "child.stdout")
            shutil.copyfile(stderr_path, preserved / "child.stderr")
    assert completed.returncode == 0, (
        f"{phase} child exited {completed.returncode}; "
        f"stdout={stdout_path}\n{completed.stdout}\n"
        f"stderr={stderr_path}\n{completed.stderr}"
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.integration
def test_real_fresh_process_restore_parity_with_checkpoint_zero_and_overlay(
    tmp_path: Path,
) -> None:
    """A fresh-process configured restore matches an uninterrupted two-step run.

    This deliberately exercises the real TPEN He model on CPU for only two
    updates; the bounded integration cost is justified by the contract's
    checkpoint-zero, overlay, and process-boundary parity requirement.
    """

    config_path = CONFIG_ROOT / "he" / "train.yaml"
    source_row = _real_restart_row(tmp_path / "science", run_id="train-lineage-0")
    resumed_row = derive_smoke_request(
        source_row, results_root=str(tmp_path / "resumed"), stop_at=1,
    )
    resumed_row_path = tmp_path / "resumed-row.json"
    resumed_row_path.parent.mkdir(parents=True, exist_ok=True)
    resumed_row_path.write_text(json.dumps(resumed_row, sort_keys=True) + "\n", encoding="utf-8")
    fresh_payload = {
        "phase": "fresh", "row_path": str(resumed_row_path), "config_path": str(config_path),
    }
    fresh_result = _run_real_restart_child(fresh_payload)
    assert fresh_result["outcome"] == "planned_yield"
    assert fresh_result["exit_code"] == 2
    assert fresh_result["target_iterations"] == resumed_row["updates"]
    fresh_inspection = fresh_result["inspection"]
    assert fresh_inspection["outcome"] == "planned_yield"
    assert fresh_inspection["checkpoint"]["next_iteration"] == 1
    assert fresh_inspection["checkpoint"]["completed_updates"] == 1
    fresh_ref = fresh_inspection["checkpoint_ref"]
    assert fresh_ref["profile"] == "train_resume"
    assert fresh_ref["next_iteration"] == 1
    assert fresh_ref["completed_updates"] == 1
    assert isinstance(fresh_ref["content_id"], str)

    catalog = resolve_run_dir(resumed_row) / "checkpoints" / "publications.jsonl"
    catalog_bytes = catalog.read_bytes()
    catalog.unlink()
    missing_publication = inspect_run(resolve_run_dir(resumed_row))
    assert missing_publication.verdict == "refused"
    assert missing_publication.refusal is not None
    # The core evidence contract maps a missing publication to this refusal.
    assert missing_publication.refusal["code"] == "unpublished-checkpoint-ref"
    catalog.write_bytes(catalog_bytes)

    resumed_result = _run_real_restart_child({**fresh_payload, "phase": "resumed"})
    if resumed_result["outcome"] != "completed" or resumed_result["exit_code"] != 0:
        resumed_stderr = resumed_row_path.with_name("resumed-row.resumed.stderr")
        resumed_stdout = resumed_row_path.with_name("resumed-row.resumed.stdout")
        resumed_diagnostics = resumed_row_path.with_name("resumed-row.resumed.diagnostics.json")
        diagnostics = json.loads(resumed_diagnostics.read_text(encoding="utf-8"))
        run_dir = Path(str(diagnostics["inspection"]["run_dir"]))
        def read_artifact(name: str) -> str:
            path = run_dir / name
            return path.read_text(encoding="utf-8") if path.is_file() else "<missing>"

        pytest.fail(
            "resumed child did not complete:\n"
            f"result={resumed_result!r}\n"
            f"runner_load={diagnostics.get('runner_load')!r}\n"
            f"checkpoint={diagnostics['inspection'].get('checkpoint')!r}\n"
            f"run-outcome.json={read_artifact('run-outcome.json')}\n"
            f"status.json={read_artifact('status.json')}\n"
            f"stdout_path={resumed_stdout}\n{resumed_stdout.read_text(encoding='utf-8')}\n"
            f"stderr_path={resumed_stderr}\n"
            f"{chr(10).join(resumed_stderr.read_text(encoding='utf-8').splitlines()[-80:])}"
        )

    reference_row = _real_restart_row(tmp_path / "reference", run_id="train-lineage-reference")
    reference_row_path = tmp_path / "reference-row.json"
    reference_row_path.write_text(json.dumps(reference_row, sort_keys=True) + "\n", encoding="utf-8")
    reference_result = _run_real_restart_child({
        "phase": "reference", "row_path": str(reference_row_path), "config_path": str(config_path),
    })
    assert reference_result["outcome"] == "completed"
    assert reference_result["exit_code"] == 0

    resumed_dir = resolve_run_dir(resumed_row)
    reference_dir = resolve_run_dir(reference_row)
    for run_dir in (resumed_dir, reference_dir):
        assert (run_dir / "checkpoints" / "step_000000" / "COMPLETE").is_file()
        resolved = OmegaConf.load(run_dir / "resolved_config.yaml")
        assert resolved.stage1_overlay.path == "configs/overlays/diagnostics-v1.yaml"
        assert resolved.stage1_overlay.sha256 == overlay_sha256(OVERLAY)
        assert not any(
            callback.get("_target_") == "tpen.callback.RawTrainingBatchRecorder"
            for callback in resolved.callbacks
        )
        assert any(
            callback.get("_target_") == "tpen.callback.DataIntegrity"
            for callback in resolved.callbacks
        )

    resumed_checkpoint = resumed_dir / "checkpoints" / "step_000002"
    reference_checkpoint = reference_dir / "checkpoints" / "step_000002"
    payload_files = ("model.pt", "optimizer.pt", "sampler.pt", "rng.pt")
    covered_payload_keys = {
        name: _assert_checkpoint_payload_equal(resumed_checkpoint / name, reference_checkpoint / name)
        for name in payload_files
    }
    import torch

    expected_payload_keys = {
        name: frozenset(torch.load(reference_checkpoint / name, map_location="cpu", weights_only=False))
        for name in payload_files
    }
    assert covered_payload_keys == expected_payload_keys
    assert set(covered_payload_keys) == set(payload_files)
    assert all(covered_payload_keys.values())
    resumed_trainer = json.loads((resumed_checkpoint / "trainer.json").read_text(encoding="utf-8"))
    reference_trainer = json.loads((reference_checkpoint / "trainer.json").read_text(encoding="utf-8"))
    assert resumed_trainer == reference_trainer
    assert {"next_iteration", "completed_updates"} <= set(resumed_trainer)

    # Keep the H=2 green route above as the ordinary prefix/continuation
    # control.  This isolated arm reuses a separately published H=2 prefix,
    # then rebuilds both row and config for H=3 before asking a fresh process
    # to continue it.  Core's trainer_config compatibility refusal must stop
    # the restore before any update or checkpoint publication.
    h_sensitive_source = _real_restart_row(
        tmp_path / "h-sensitive-source", run_id="h-sensitive-prefix", lineage_updates=2,
    )
    h_sensitive_row = derive_smoke_request(
        h_sensitive_source, results_root=str(tmp_path / "h-sensitive"), stop_at=1,
    )
    h_sensitive_row_path = tmp_path / "h-sensitive-row.json"
    h_sensitive_row_path.write_text(
        json.dumps(h_sensitive_row, sort_keys=True) + "\n", encoding="utf-8",
    )
    h_sensitive_fresh = _run_real_restart_child({
        "phase": "fresh",
        "row_path": str(h_sensitive_row_path),
        "config_path": str(config_path),
    })
    assert h_sensitive_fresh["outcome"] == "planned_yield"
    assert h_sensitive_fresh["exit_code"] == 2

    h3_row = _real_restart_row(
        tmp_path / "h-sensitive", run_id="h-sensitive-prefix", lineage_updates=3,
    )
    h3_row_path = tmp_path / "h3-row.json"
    h3_row_path.write_text(json.dumps(h3_row, sort_keys=True) + "\n", encoding="utf-8")
    prefix_dir = resolve_run_dir(h_sensitive_row)
    prefix_checkpoint = prefix_dir / "checkpoints" / "step_000001"
    catalog = prefix_dir / "checkpoints" / "publications.jsonl"
    before_catalog = catalog.read_bytes()
    before_payloads = {
        name: (prefix_checkpoint / name).read_bytes()
        for name in ("model.pt", "optimizer.pt", "sampler.pt", "rng.pt", "trainer.json")
    }

    h3_result = _run_real_restart_child({
        "phase": "horizon-mismatch",
        "row_path": str(h3_row_path),
        "config_path": str(config_path),
    })
    assert h3_result["outcome"] == "failed"
    assert h3_result["exit_code"] == 1
    assert h3_result["exception_type"] == "ValueError"
    assert "trainer_config mismatch" in h3_result["error"]
    assert catalog.read_bytes() == before_catalog
    assert not (prefix_dir / "checkpoints" / "step_000002").exists()
    assert {
        name: (prefix_checkpoint / name).read_bytes()
        for name in before_payloads
    } == before_payloads
