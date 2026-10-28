"""Review r1: E3 run_configured_once load-overlay placement.

Target layer: E3 @ 7d86fe49f4f6b66344c5b6bcce2f2f01f605ccf4 (also run at CR head).

Polarity at the E3 head:
- test_overlay_reaches_concrete_runner_load: GREEN. The configured runner
  receives the selected checkpoint overlay even when a concrete top-level
  ``load`` key and a concrete ``runner.load`` value were both supplied.
- test_overlay_flows_through_interpolated_runner_load: EXPECTED GREEN (pin).
- test_overlay_force_added_when_no_load_keys: EXPECTED GREEN (pin).
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from tpen.execution import RecoveryDecision, RecoveryOutcome, RecoveryPhase
from tpen.execution.invocation import run_configured_once


def _decision(tmp_path: Path) -> RecoveryDecision:
    return RecoveryDecision(
        phase=RecoveryPhase.RECOVERY,
        outcome=RecoveryOutcome.RECOVERY,
        target_iterations=8,
        next_iteration=4,
        completed_updates=3,
        load={"mode": "train_resume", "path": str(tmp_path / "step_000004")},
    )


def _captured_runner_load(cfg):
    node = OmegaConf.select(cfg, "runner.load")
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return node


def _call(monkeypatch, cfg, decision, tmp_path: Path):
    from tpen import run as run_module

    captured = []

    def fake_run(config, **kwargs):
        captured.append(config)
        return 0

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    run_configured_once(cfg, decision=decision, output_path=tmp_path / "fresh")
    assert len(captured) == 1
    return captured[0]


def test_overlay_reaches_concrete_runner_load(monkeypatch, tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    cfg = OmegaConf.create(
        {
            "run": {"root": "ignored"},
            "load": {"mode": "none"},
            "runner": {"_target_": "x", "load": {"mode": "none"}},
        }
    )
    captured = _call(monkeypatch, cfg, decision, tmp_path)
    assert _captured_runner_load(captured) == decision.load


def test_overlay_flows_through_interpolated_runner_load(monkeypatch, tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    cfg = OmegaConf.create(
        {
            "run": {"root": "ignored"},
            "load": {"mode": "none"},
            "runner": {"_target_": "x", "load": "${load}"},
        }
    )
    captured = _call(monkeypatch, cfg, decision, tmp_path)
    assert _captured_runner_load(captured) == decision.load


def test_overlay_force_added_when_no_load_keys(monkeypatch, tmp_path: Path) -> None:
    decision = _decision(tmp_path)
    cfg = OmegaConf.create({"run": {"root": "ignored"}, "runner": {"_target_": "x"}})
    captured = _call(monkeypatch, cfg, decision, tmp_path)
    assert _captured_runner_load(captured) == decision.load
