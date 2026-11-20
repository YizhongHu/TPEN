"""Cannon-facing restart-chain acceptance for the Stage-1 protocol."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from experiments.atomistic.stage1.protocol import (
    STAGE1_PROTOCOL,
    build_manifest,
    config_identities,
)
from experiments.atomistic.stage1.restart import execute_restart
from experiments.atomistic.stage1.train import prepare_initial_checkpoint_config
from tpen.checkpoint import TrainResume, save_checkpoint
from tpen.run import load_config


CONFIG_ROOT = Path(__file__).parents[2] / "experiments" / "atomistic" / "stage1" / "configs"


def _row(system: str, tmp_path: Path) -> dict[str, object]:
    config_root = CONFIG_ROOT
    config_path = config_root / system / "train.yaml"
    rows = build_manifest(
        protocol=STAGE1_PROTOCOL,
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=config_root, systems=(system,)),
        checkpoints=list(load_config(str(config_path)).checkpoint_steps),
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path),
        systems=(system,),
        phase=1,
        device="cpu",
    )
    return next(row for row in rows if row["kind"] == "train" and row["lineage"] == 0)


def _publish(run_dir: Path, step: int) -> None:
    """Publish a real full-state checkpoint with the core checkpoint writer."""

    model = torch.nn.Linear(1, 1, bias=False)
    context = SimpleNamespace(
        cfg=OmegaConf.create({"trainer": {"max_steps": STAGE1_PROTOCOL.lineage_updates}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id="stage1-chain", extra={}),
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


@pytest.mark.integration
@pytest.mark.stage1_real_restart
@pytest.mark.parametrize("system", ["he", "h2"])
def test_real_cannon_cpu_restartable_multi_attempt_chain(
    system: str, tmp_path: Path,
) -> None:
    """Exercise real He/H2 config admission and a two-attempt published chain.

    The checkpoint artifacts are produced by the real core writer so this
    checks publication, typed restore admission, leases, and outcome receipts
    on Cannon CPU without spending a science budget inside the test harness.
    The production 100,000-update driver remains the separately authorized
    runbook action; this test never changes that immutable horizon.
    """

    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = tmp_path / system / "lineage-0"
    first_cfg = load_config(str(config_path))
    prepare_initial_checkpoint_config(first_cfg)
    first_calls: list[dict[str, object]] = []

    def first_driver(_cfg, **kwargs):
        first_calls.append(kwargs)
        _publish(run_dir, 1000)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "planned_yield"}) + "\n", encoding="utf-8"
        )
        return 7

    first = execute_restart(
        cfg=first_cfg,
        row=row,
        config_path=config_path,
        run=first_driver,
        run_dir=run_dir,
    )
    assert first.outcome == "planned_yield"
    assert first.exit_code == 7
    assert first_calls == [{"config_path": str(config_path), "command": "stage1 restart"}]

    second_cfg = load_config(str(config_path))
    prepare_initial_checkpoint_config(second_cfg)
    second_calls: list[dict[str, object]] = []

    def second_driver(_cfg, **kwargs):
        second_calls.append(kwargs)
        assert kwargs["restore_request"].__class__.__name__ == "OperationalRestoreRequest"
        _publish(run_dir, STAGE1_PROTOCOL.lineage_updates)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "completed"}) + "\n", encoding="utf-8"
        )
        return 99

    second = execute_restart(
        cfg=second_cfg,
        row=row,
        config_path=config_path,
        run=second_driver,
        run_dir=run_dir,
    )
    assert second.outcome == "completed"
    assert second.exit_code == 99
    assert len(second_calls) == 1
    assert second_calls[0]["restore_request"].load["mode"] == "train_resume"
    assert first.lease_path != second.lease_path
    assert first.outcome_path != second.outcome_path
    assert first.lease_path.is_file() and second.lease_path.is_file()
    assert first.outcome_path.is_file() and second.outcome_path.is_file()
