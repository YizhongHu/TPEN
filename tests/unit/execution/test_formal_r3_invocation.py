"""Formal review round 3: E3 source-witness completeness and overlays."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from tpen.execution import (
    ExecutionConfig,
    InvocationOutcome,
    ReadyBatch,
    ReadyItem,
    ReceiptBindingError,
    RecoveryPhase,
    WorkerStartGateway,
    capture_science_intent,
    source_config_digest,
)
from tpen.execution.config import payload_digest


def _batch(source: dict[str, object]) -> tuple[ReadyBatch, object]:
    science = {"system": "hooke", "electrons": 2}
    payload = {"science": science, "target_iterations": 2}
    intent = capture_science_intent(science, target_iterations=2, source_config=source)
    execution = ExecutionConfig(
        lineage_id="r3-e3-lineage",
        backend="direct",
        recovery="none",
        source_config_digest=source_config_digest(source),
        payload_digest=payload_digest(payload),
    )
    return (
        ReadyBatch(
            lineage_id=execution.lineage_id,
            batch_id="r3-e3-batch",
            source_config_digest=execution.source_config_digest,
            payload_digest=execution.payload_digest,
            science_input_digest=intent.science_input_digest,
            target_iterations=2,
            execution=execution,
            items=(ReadyItem("r3-e3-inv", "r3-e3-task", "r3-e3-out", ("worker",), "results/e3"),),
            payload=payload,
        ),
        intent,
    )


def _run_spy(monkeypatch):
    from tpen import run as run_module

    calls = []

    def fake_run(config, **kwargs):
        calls.append((config, kwargs))
        return 0

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    return calls


def test_missing_captured_leaf_is_refused_before_science(monkeypatch, tmp_path: Path) -> None:
    source = {
        "runner": {"_target_": "tests.helpers.NeverConstructed"},
        "optimizer": {"lr": 0.01},
    }
    batch, intent = _batch(source)
    cfg = OmegaConf.create({"runner": source["runner"]})
    calls = _run_spy(monkeypatch)

    try:
        invocation = WorkerStartGateway(tmp_path / "outputs").start_configured(
            cfg=cfg, batch=batch, intent=intent, phase=RecoveryPhase.INITIAL
        )
    except (ReceiptBindingError,):
        pass
    else:
        assert invocation.outcome.outcome is InvocationOutcome.TERMINAL
        assert invocation.outcome.reason.strip()
    assert calls == []


def test_null_subtree_replacing_captured_leaves_is_refused(monkeypatch, tmp_path: Path) -> None:
    source = {
        "runner": {"_target_": "tests.helpers.NeverConstructed"},
        "trainer": {"max_steps": 2},
    }
    batch, intent = _batch(source)
    cfg = OmegaConf.create({"runner": source["runner"], "trainer": None})
    calls = _run_spy(monkeypatch)

    try:
        invocation = WorkerStartGateway(tmp_path / "outputs").start_configured(
            cfg=cfg, batch=batch, intent=intent, phase=RecoveryPhase.INITIAL
        )
    except (ReceiptBindingError,):
        pass
    else:
        assert invocation.outcome.outcome is InvocationOutcome.TERMINAL
        assert invocation.outcome.reason.strip()
    assert calls == []


def test_supplied_extra_opaque_leaf_still_accepted(monkeypatch, tmp_path: Path) -> None:
    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    cfg = OmegaConf.create({**source, "legacy": {"flag": True}})
    calls = _run_spy(monkeypatch)

    WorkerStartGateway(tmp_path / "outputs").start_configured(
        cfg=cfg, batch=batch, intent=intent, phase=RecoveryPhase.INITIAL
    )
    assert len(calls) == 1


def test_start_receipt_records_declared_overlays(monkeypatch, tmp_path: Path) -> None:
    source = {
        "runner": {"_target_": "tests.helpers.NeverConstructed"},
        "trainer": {"max_steps": 2},
    }
    batch, intent = _batch(source)
    calls = _run_spy(monkeypatch)

    invocation = WorkerStartGateway(tmp_path / "outputs").start_configured(
        cfg=OmegaConf.create(source),
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        declared_overlays={"trainer.max_steps": 1},
    )
    assert len(calls) == 1
    start_on_disk = json.loads(
        (invocation.output_path.parent / "start.json").read_text(encoding="utf-8")
    )
    assert start_on_disk["declared_overlays"] == {"trainer.max_steps": 1}
