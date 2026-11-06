"""Round-6 Phase-A A1: E4 must carry configured overlays into E3."""

import json
from pathlib import Path

import pytest


def _batch():
    from tpen.execution import ExecutionConfig, ReadyBatch, ReadyItem
    from tpen.execution import capture_science_intent, payload_digest, source_config_digest

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}, "trainer": {"max_steps": 2}}
    science = {"system": "hooke", "electrons": 2}
    intent = capture_science_intent(science, target_iterations=2, source_config=source)
    payload = {"science": science, "target_iterations": 2}
    source_digest = source_config_digest(source)
    execution = ExecutionConfig(
        lineage_id="r6-overlay-lineage",
        backend="balsam",
        recovery="checkpoint",
        source_config_digest=source_digest,
        payload_digest=payload_digest(payload),
        backend_options={"site": "r6-test"},
    )
    batch = ReadyBatch(
        lineage_id=execution.lineage_id,
        batch_id="r6-overlay-batch",
        source_config_digest=source_digest,
        payload_digest=execution.payload_digest,
        science_input_digest=intent.science_input_digest,
        target_iterations=2,
        execution=execution,
        items=(ReadyItem("r6-invocation", "r6-task", "r6-output", ("worker",), "results/r6"),),
        payload=payload,
    )
    return batch, intent, source


class _Client:
    def __init__(self):
        self.register_calls = 0
        self.observe_calls = 0

    def register_job(self, *, lineage_id, batch, backend_options):
        self.register_calls += 1
        return {"job_id": "r6-job"}

    def observe_job(self, job_id):
        self.observe_calls += 1
        raise AssertionError("worker start must not observe the backend")


def test_r6_e4_configured_overlay_reaches_core_and_start_receipt(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    from tpen import run as run_module
    from tpen.execution import BalsamAdapter, BalsamAdapterStatus, RecoveryPhase, WorkerStartGateway

    batch, intent, source = _batch()
    client = _Client()
    adapter = BalsamAdapter(
        batch.execution,
        client,
        WorkerStartGateway(tmp_path / "outputs"),
        tmp_path / "receipts",
    )
    registration = adapter.register_lineage(batch)
    assert registration is not None
    seen = []
    observed_start = []

    def core(cfg, **kwargs):
        assert kwargs["raise_exceptions"] is True
        seen.append(cfg.trainer.max_steps)
        observed_start.append(json.loads((Path(cfg.run.root).parent / "start.json").read_text(encoding="utf-8")))
        return 0

    monkeypatch.setattr(run_module, "run_from_config", core)
    result = adapter.start_configured(
        cfg=OmegaConf.create({**source, "run": {"root": "unused"}}),
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        job_id=registration.job_id,
        declared_overlays={"trainer.max_steps": 1},
    )
    assert result.status is BalsamAdapterStatus.STARTED
    assert seen == [1]
    assert observed_start[0]["declared_overlays"] == {"trainer.max_steps": 1}
    start = result.invocation.start
    assert start.declared_overlays == {"trainer.max_steps": 1}
    assert client.register_calls == 1
    assert client.observe_calls == 0


def test_r6_e4_no_overlay_is_a_distinct_green_control(monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    from tpen import run as run_module
    from tpen.execution import BalsamAdapter, BalsamAdapterStatus, RecoveryPhase, WorkerStartGateway

    batch, intent, source = _batch()
    client = _Client()
    adapter = BalsamAdapter(batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts")
    registration = adapter.register_lineage(batch)
    seen = []
    observed_start = []

    def core(cfg, **kwargs):
        seen.append(cfg.trainer.max_steps)
        observed_start.append(json.loads((Path(cfg.run.root).parent / "start.json").read_text(encoding="utf-8")))
        return 0

    monkeypatch.setattr(run_module, "run_from_config", core)
    result = adapter.start_configured(
        cfg=OmegaConf.create({**source, "run": {"root": "unused"}}),
        batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, job_id=registration.job_id,
    )
    assert result.status is BalsamAdapterStatus.STARTED
    assert seen == [2]
    assert observed_start[0]["declared_overlays"] == {}
    assert result.invocation.start.declared_overlays == {}
    assert client.observe_calls == 0


def test_r6_e3_direct_configured_overlay_is_a_green_control(monkeypatch, tmp_path):
    import json
    from omegaconf import OmegaConf
    from tpen import run as run_module
    from tpen.execution import RecoveryPhase, WorkerStartGateway

    batch, intent, source = _batch()
    seen = []
    observed_start = []

    def core(cfg, **kwargs):
        seen.append(cfg.trainer.max_steps)
        observed_start.append(json.loads((Path(cfg.run.root).parent / "start.json").read_text(encoding="utf-8")))
        return 0

    monkeypatch.setattr(run_module, "run_from_config", core)
    result = WorkerStartGateway(tmp_path / "outputs").start_configured(
        cfg=OmegaConf.create({**source, "run": {"root": "unused"}}),
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        declared_overlays={"trainer.max_steps": 1},
    )
    assert seen == [1]
    assert observed_start[0]["declared_overlays"] == {"trainer.max_steps": 1}


def test_r6_unresolved_overlay_constructs_consumers_and_reads_back(tmp_path):
    """Unresolved source shape carries overlays into constructed consumers."""

    from omegaconf import OmegaConf
    from tpen.execution import capture_science_intent
    from tpen.execution.invocation import (
        _materialize_configured_worker_config,
    )
    from tpen.execution import RecoveryPhase, select_recovery

    source = {
        "system": {"spatial_dim": 3},
        "runtime": {"seed": 17},
        "model": {"spatial_dim": "${system.spatial_dim}", "seed": "${runtime.seed}"},
        "sampler": {"spatial_dim": "${system.spatial_dim}", "seed": "${runtime.seed}"},
        "trainer": {
            "_target_": "tpen.training.trainer.VMCTrainer",
            "max_steps": 20,
        },
        "runner": {
            "model": "${model}",
            "sampler": "${sampler}",
            "trainer": "${trainer}",
        },
        "run": {"root": "${run.base}", "base": str(tmp_path / "old")},
        "checkpoint": {"output_dir": "${run.root}/checkpoints"},
        "metrics": {"path": "${run.root}/metrics.jsonl"},
        "loggers": [{"_target_": "tpen.logging.JSONL", "path": "${run.root}/metrics.jsonl"}],
    }
    intent = capture_science_intent(
        {"system": "probe"}, target_iterations=20, source_config=source
    )
    decision = select_recovery(intent, phase=RecoveryPhase.INITIAL)
    output_root = tmp_path / "new-root"

    worker = _materialize_configured_worker_config(
        OmegaConf.create(source),
        intent=intent,
        decision=decision,
        output_path=output_root,
        declared_overlays={"trainer.max_steps": 7, "run.root": str(output_root)},
    )

    _assert_constructed_overlay_readback(worker, output_root)


def _assert_constructed_overlay_readback(worker, output_root):
    from hydra.utils import instantiate

    trainer = instantiate(worker.trainer)
    logger = instantiate(worker.loggers[0])
    assert trainer.max_steps == 7
    assert worker.runner.trainer.max_steps == trainer.max_steps
    assert worker.runner.model.spatial_dim == worker.model.spatial_dim == 3
    assert worker.runner.sampler.seed == worker.sampler.seed == 17
    assert worker.checkpoint.output_dir == str(output_root / "checkpoints")
    assert worker.metrics.path == str(output_root / "metrics.jsonl")
    assert logger.path == output_root / "metrics.jsonl"


def test_r6_resolved_descendant_supply_is_fail_closed(tmp_path):
    """A resolved descendant cannot masquerade as the captured section."""

    from omegaconf import OmegaConf
    from tpen.execution import ReceiptBindingError, RecoveryPhase, capture_science_intent, select_recovery
    from tpen.execution.invocation import _materialize_configured_worker_config

    source = {
        "trainer": {"max_steps": 20},
        "runner": {"trainer": "${trainer}"},
        "run": {"root": "${run.base}", "base": str(tmp_path / "old")},
    }
    intent = capture_science_intent({"system": "probe"}, target_iterations=20, source_config=source)
    supplied = OmegaConf.create({
        "trainer": {"max_steps": 20},
        "runner": {"trainer": {"max_steps": 20}},
        "run": {"root": str(tmp_path / "old")},
    })
    with pytest.raises(ReceiptBindingError, match="omits captured source leaves"):
        _materialize_configured_worker_config(
            supplied,
            intent=intent,
            decision=select_recovery(intent, phase=RecoveryPhase.INITIAL),
            output_path=tmp_path / "new",
            declared_overlays={"trainer.max_steps": 7},
        )


def test_r6_overlay_application_mutant_is_red(tmp_path):
    """The readback pin fails for the mutant that drops declared overlays."""

    from omegaconf import OmegaConf
    from tpen.execution import RecoveryPhase, capture_science_intent, select_recovery
    from tpen.execution.invocation import _materialize_configured_worker_config

    source = {"trainer": {"max_steps": 20}, "run": {"root": "${run.base}", "base": str(tmp_path / "old")}}
    intent = capture_science_intent({"system": "probe"}, target_iterations=20, source_config=source)
    worker = _materialize_configured_worker_config(
        OmegaConf.create(source),
        intent=intent,
        decision=select_recovery(intent, phase=RecoveryPhase.INITIAL),
        output_path=tmp_path / "new",
        declared_overlays={},
    )
    with pytest.raises(AssertionError):
        assert worker.trainer.max_steps == 7
