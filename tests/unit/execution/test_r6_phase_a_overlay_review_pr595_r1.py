"""PR #595 sharper fail-closed gateway overlay pin."""

from __future__ import annotations

from copy import deepcopy

import pytest


def test_r6_resolved_descendant_supply_names_runner_trainer(tmp_path):
    from omegaconf import OmegaConf
    from tpen.execution import ReceiptBindingError, RecoveryPhase, capture_science_intent, select_recovery
    from tpen.execution.invocation import _materialize_configured_worker_config

    source = {
        "system": {"spatial_dim": 3},
        "runtime": {"seed": 17},
        "model": {"spatial_dim": "${system.spatial_dim}", "seed": "${runtime.seed}"},
        "sampler": {"spatial_dim": "${system.spatial_dim}", "seed": "${runtime.seed}"},
        "trainer": {"_target_": "tpen.training.trainer.VMCTrainer", "max_steps": 20},
        "runner": {"model": "${model}", "sampler": "${sampler}", "trainer": "${trainer}"},
        "run": {"root": "${run.base}", "base": str(tmp_path / "old")},
    }
    intent = capture_science_intent({"system": "probe"}, target_iterations=20, source_config=source)
    supplied = deepcopy(source)
    supplied["runner"]["trainer"] = {"max_steps": 20}
    with pytest.raises(ReceiptBindingError, match=r"omits captured source leaves: runner\.trainer"):
        _materialize_configured_worker_config(
            OmegaConf.create(supplied),
            intent=intent,
            decision=select_recovery(intent, phase=RecoveryPhase.INITIAL),
            output_path=tmp_path / "new",
            declared_overlays={"trainer.max_steps": 7},
        )
