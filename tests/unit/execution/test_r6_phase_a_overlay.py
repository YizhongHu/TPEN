"""Round-6 Phase-A A1: E4 must carry configured overlays into E3."""

import json
from pathlib import Path


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
