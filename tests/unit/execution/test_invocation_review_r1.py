"""Review r2 (S2, item 4023abbb, finding F5): ``run_configured_once`` restore wiring.

Historical note: at an earlier E3 head, ``run_configured_once`` rewrote
``runner.load``/``load`` into a copied config before calling
``run_from_config``. S2 replaces that config-patch path: a selected E1
decision is now passed through as a typed, immutable
``OperationalRestoreRequest`` via ``run_from_config(..., restore_request=...)``,
and the config itself is never patched with restore fields.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tpen.checkpoint import TrainResume, save_checkpoint
from tpen.checkpoint.reference import CheckpointRef
from tpen.execution import RecoveryDecision, RecoveryOutcome, RecoveryPhase
from tpen.execution.invocation import run_configured_once
from tpen.execution.recovery import OperationalRestoreRequest, train_resume_load_overlay


def _save_checkpoint(tmp_path: Path, *, step: int = 4, target_iterations: int = 8) -> CheckpointRef:
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = SimpleNamespace(state_dict=lambda: {"next_iteration": step, "completed_updates": step})
    sampler = SimpleNamespace(mcmc_state_dict=lambda: {"walker": step})
    context = SimpleNamespace(
        cfg=OmegaConf.create({"study": {"name": "r1"}, "trainer": {"max_steps": target_iterations}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id=f"r1-{step}", extra={}),
        run_dir=tmp_path,
    )
    checkpoint_dir = save_checkpoint(
        output_dir=tmp_path / "checkpoints",
        next_iteration=step,
        completed_updates=step,
        model=model,
        optimizer=optimizer,
        trainer=trainer,
        sampler=sampler,
        payload=TrainResume(),
        context=context,
    )
    return CheckpointRef.from_directory(checkpoint_dir)


def _decision(ref: CheckpointRef, *, target_iterations: int = 8) -> RecoveryDecision:
    return RecoveryDecision(
        phase=RecoveryPhase.RECOVERY,
        outcome=RecoveryOutcome.RECOVERY,
        target_iterations=target_iterations,
        next_iteration=ref.next_iteration,
        completed_updates=ref.completed_updates,
        checkpoint=ref,
        load=train_resume_load_overlay(ref),
    )


def _call(monkeypatch, cfg, decision, tmp_path: Path):
    from tpen import run as run_module

    captured = []

    def fake_run(config, **kwargs):
        captured.append((config, kwargs))
        return 0

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    run_configured_once(cfg, decision=decision, output_path=tmp_path / "fresh")
    assert len(captured) == 1
    return captured[0]


def test_restore_request_is_passed_through_without_patching_cfg_load(monkeypatch, tmp_path: Path) -> None:
    ref = _save_checkpoint(tmp_path)
    decision = _decision(ref)
    cfg = OmegaConf.create(
        {
            "run": {"root": "ignored"},
            "load": {"mode": "none"},
            "runner": {"_target_": "x", "load": {"mode": "none"}},
        }
    )
    captured_cfg, captured_kwargs = _call(monkeypatch, cfg, decision, tmp_path)

    # The config-patch path is gone: neither runner.load nor the top-level
    # load mirror is rewritten by this adapter.
    assert OmegaConf.to_container(captured_cfg.runner.load, resolve=True) == {"mode": "none"}
    assert OmegaConf.to_container(captured_cfg.load, resolve=True) == {"mode": "none"}
    restore_request = captured_kwargs["restore_request"]
    assert isinstance(restore_request, OperationalRestoreRequest)
    assert restore_request.load == decision.load


def test_no_load_decision_passes_no_restore_request(monkeypatch, tmp_path: Path) -> None:
    decision = RecoveryDecision(
        phase=RecoveryPhase.INITIAL,
        outcome=RecoveryOutcome.INITIAL,
        target_iterations=8,
    )
    cfg = OmegaConf.create({"run": {"root": "ignored"}, "runner": {"_target_": "x"}})
    captured_cfg, captured_kwargs = _call(monkeypatch, cfg, decision, tmp_path)

    assert "load" not in captured_cfg.runner
    assert captured_kwargs["restore_request"] is None


def test_model_only_decision_reaches_runner_via_legacy_overlay_with_no_restore_request(
    monkeypatch, tmp_path: Path
) -> None:
    """S2 non-goal: a ``model_only`` load decision is out of scope for
    ``OperationalRestoreRequest`` (which hard-requires ``train_resume``).
    ``run_configured_once`` must route it through dev's unchanged pre-S2
    config-overlay path instead of refusing it."""

    decision = RecoveryDecision(
        phase=RecoveryPhase.RECOVERY,
        outcome=RecoveryOutcome.RECOVERY,
        target_iterations=8,
        load={"mode": "model_only", "path": "/somewhere"},
    )
    cfg = OmegaConf.create(
        {
            "run": {"root": "ignored"},
            "load": {"mode": "none"},
            "runner": {"_target_": "x", "load": {"mode": "none"}},
        }
    )
    captured_cfg, captured_kwargs = _call(monkeypatch, cfg, decision, tmp_path)

    assert OmegaConf.to_container(captured_cfg.runner.load, resolve=True) == {
        "mode": "model_only",
        "path": "/somewhere",
    }
    assert OmegaConf.to_container(captured_cfg.load, resolve=True) == {
        "mode": "model_only",
        "path": "/somewhere",
    }
    assert "restore_request" not in captured_kwargs or captured_kwargs["restore_request"] is None


def test_conflicting_concrete_configured_load_is_refused_through_the_real_boundary(
    monkeypatch, tmp_path: Path
) -> None:
    """Consumer-side: a configured runner.load that disagrees with the
    selected decision is refused by run_from_config's real
    ``_apply_operational_restore`` conflict check, not silently shadowed by
    this adapter (F5). Hydra instantiation is bypassed via a monkeypatched
    ``_instantiate_runner`` so this test exercises real wiring from
    ``run_configured_once`` through to the real restore boundary without the
    heavier full-``Train`` fixture used by the public-entry test."""

    from tpen import run as run_module

    # R2.2 requires ref.next_iteration <= target_iterations, so this fixture
    # uses an admissible cursor (step == target_iterations) rather than the
    # step-4/H-2 cursor that would now be refused by OperationalRestoreRequest
    # construction before this test's conflict check is ever reached.
    ref = _save_checkpoint(tmp_path, step=2, target_iterations=2)
    decision = _decision(ref, target_iterations=2)
    fixture = Path("tests/integration/artifacts/training/vmc_smoke.yaml")
    cfg = OmegaConf.load(fixture)
    cfg.run.root = str(tmp_path / "configured-run-source")
    cfg.trainer.max_steps = 2
    cfg.runner.load = {"mode": "train_resume", "path": "/elsewhere"}
    conflicting_runner = SimpleNamespace(
        load={"mode": "train_resume", "path": "/elsewhere"},
        trainer=SimpleNamespace(max_steps=decision.target_iterations),
    )
    monkeypatch.setattr(run_module, "_instantiate_runner", lambda context: conflicting_runner)

    with pytest.raises(ValueError, match="conflicts"):
        run_configured_once(cfg, decision=decision, output_path=tmp_path / "fresh", raise_exceptions=True)
