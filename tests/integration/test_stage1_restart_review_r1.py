"""Reviewer tests for Stage-1 restart lease and publication boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from experiments.atomistic.stage1.restart import (
    RestartApproval,
    RestartRefusal,
    execute_restart,
)
from experiments.atomistic.stage1.protocol import STAGE1_PROTOCOL
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.checkpoint import TrainResume, save_checkpoint

from tests.integration.test_stage1_restartable import (
    CONFIG_ROOT,
    _production_cfg,
    _row,
)


def _publish(run_dir: Path, *, step: int = 1000, run_id: str = "stage1-chain") -> None:
    """Publish a real full-state checkpoint with controlled provenance."""

    import torch

    model = torch.nn.Linear(1, 1, bias=False)
    context = SimpleNamespace(
        cfg=OmegaConf.create({"trainer": {"max_steps": STAGE1_PROTOCOL.lineage_updates}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id=run_id, extra={}),
        run_dir=run_dir,
    )
    save_checkpoint(
        output_dir=run_dir / "checkpoints",
        next_iteration=step,
        completed_updates=step,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        trainer=SimpleNamespace(
            state_dict=lambda: {"next_iteration": step, "completed_updates": step}
        ),
        sampler=SimpleNamespace(mcmc_state_dict=lambda: {"walker": step}),
        payload=TrainResume(),
        context=context,
    )


def _attempt_cfg(row: dict[str, object], system: str):
    return _production_cfg(row, CONFIG_ROOT / system / "train.yaml")


@pytest.mark.integration
def test_inflight_initial_lease_blocks_second_driver_after_publication(tmp_path: Path) -> None:
    """A publication during an admitted initial attempt does not admit a peer."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    first_cfg = _attempt_cfg(row, system)
    second_cfg = _attempt_cfg(row, system)
    published = Event()
    release = Event()
    second_calls: list[object] = []
    first_errors: list[BaseException] = []

    def first_driver(_cfg, **_kwargs):
        _publish(run_dir)
        published.set()
        assert release.wait(timeout=10)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "planned_yield"}) + "\n", encoding="utf-8"
        )
        return 7

    def first_attempt() -> None:
        try:
            execute_restart(
                cfg=first_cfg, row=row, config_path=config_path,
                run=first_driver, run_dir=run_dir,
            )
        except BaseException as exc:  # pragma: no cover - diagnostic propagation
            first_errors.append(exc)

    thread = Thread(target=first_attempt)
    thread.start()
    assert published.wait(timeout=10)

    def second_driver(_cfg, **_kwargs):
        second_calls.append(True)
        return 0

    try:
        with pytest.raises(RestartRefusal, match="occupied predecessor lease"):
            execute_restart(
                cfg=second_cfg, row=row, config_path=config_path,
                run=second_driver, run_dir=run_dir,
            )
        assert second_calls == []
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert first_errors == []


@pytest.mark.integration
def test_invalid_initial_takeover_approval_refuses_before_takeover_lease(tmp_path: Path) -> None:
    """An occupied fresh lease still requires a complete takeover approval."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    attempts = run_dir / "attempts"
    attempts.mkdir(parents=True)
    (attempts / "initial.json").write_text("{}\n", encoding="utf-8")
    cfg = _attempt_cfg(row, system)
    calls: list[object] = []
    approval = RestartApproval(
        predecessor_content_id="",
        terminal_pbs_proof="",
        quiescent=False,
        adopted_budget="",
        lineage_id="wrong-lineage",
        config_identity="wrong-config",
        overlay_identity="wrong-overlay",
        target_iterations=-1,
    )

    with pytest.raises(RestartRefusal):
        execute_restart(
            cfg=cfg, row=row, config_path=config_path,
            run=lambda *_args, **_kwargs: calls.append(True) or 0,
            run_dir=run_dir, takeover=approval,
        )
    assert calls == []
    assert not (attempts / "initial.takeover.json").exists()


@pytest.mark.integration
def test_fresh_attempt_does_not_forward_raw_runner_load(tmp_path: Path) -> None:
    """Fresh admission must not let configured load state bypass typed restore."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    cfg = _attempt_cfg(row, system)
    OmegaConf.update(
        cfg, "runner.load", {"mode": "train_resume", "path": "/foreign/checkpoint"},
        merge=False, force_add=True,
    )
    calls: list[object] = []

    try:
        result = execute_restart(
            cfg=cfg, row=row, config_path=config_path,
            run=lambda run_cfg, **_kwargs: calls.append(OmegaConf.select(run_cfg, "runner.load")) or 0,
            run_dir=run_dir,
        )
    except RestartRefusal as exc:
        assert "runner.load" in str(exc)
        result = None
    assert calls == []
    assert result is None or result.outcome == "parked"
    assert not (run_dir / "attempts").exists()


@pytest.mark.integration
def test_failed_terminal_status_cannot_be_reported_completed(tmp_path: Path) -> None:
    """A terminal checkpoint does not erase a durable failed status."""

    system = "he"
    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    _publish(run_dir, step=STAGE1_PROTOCOL.lineage_updates)
    (run_dir / "status.json").write_text(
        json.dumps({"status": "failed"}) + "\n", encoding="utf-8"
    )
    cfg = _attempt_cfg(row, system)
    calls: list[object] = []

    result = execute_restart(
        cfg=cfg, row=row, config_path=config_path,
        run=lambda *_args, **_kwargs: calls.append(True) or 0,
        run_dir=run_dir,
    )
    assert calls == []
    assert result.outcome != "completed"
