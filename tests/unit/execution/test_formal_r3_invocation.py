"""Formal review round 3: E3 source-witness completeness and overlay provenance.

Target: E3 @ 651f43f28ffb16672046ca625eb3c58d16249ae2 and the CR head.

Polarity at those heads (preregistered):
- test_missing_captured_leaf_is_refused_before_science: EXPECTED RED.  The
  witness check iterates only supplied leaves and the write-back loop copies a
  captured source leaf into the worker config only when that leaf is already
  present in the supplied config, so a supplied config that simply omits a
  captured science leaf reaches the runner without it while receipts bind the
  captured source_config_digest.
- test_null_subtree_replacing_captured_leaves_is_refused: EXPECTED RED.  A
  supplied ``null`` subtree collapses the captured leaves below it, which the
  leaf-wise comparison never visits.
- test_supplied_extra_opaque_leaf_still_accepted: EXPECTED GREEN (pin).  Extra
  supplied schema leaves outside the captured snapshot are a documented
  allowance; the missing-leaf fix must not remove it silently.
- test_start_receipt_records_declared_overlays: EXPECTED RED.  Declared
  operational overlays (for example ``trainer.max_steps``) change what the
  worker actually runs but leave no durable trace in the worker-start receipt.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from tpen.config_schema import ClosedSchemaError
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
    """RED at E3/CR: omitting a captured science leaf must not reach the runner."""

    source = {
        "runner": {"_target_": "tests.helpers.NeverConstructed"},
        "optimizer": {"lr": 0.01},
    }
    batch, intent = _batch(source)
    # The supplied worker config omits the captured optimizer section entirely.
    cfg = OmegaConf.create({"runner": source["runner"]})
    calls = _run_spy(monkeypatch)

    try:
        invocation = WorkerStartGateway(tmp_path / "outputs").start_configured(
            cfg=cfg, batch=batch, intent=intent, phase=RecoveryPhase.INITIAL
        )
    except (ReceiptBindingError, ClosedSchemaError):
        pass
    else:
        assert invocation.outcome.outcome is InvocationOutcome.TERMINAL, (
            "a captured science leaf absent from the supplied config must be "
            "exception or durable TERMINAL, got "
            f"{invocation.outcome.outcome!r}; captured runner calls={calls!r}"
        )
        assert invocation.outcome.reason.strip()
    assert calls == [], (
        "a supplied config missing the captured optimizer.lr leaf reached "
        "tpen.run.run_from_config while receipts bind the captured "
        "source_config_digest"
    )


def test_null_subtree_replacing_captured_leaves_is_refused(monkeypatch, tmp_path: Path) -> None:
    """RED at E3/CR: a null subtree silently deletes every captured leaf below it."""

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
    except (ReceiptBindingError, ClosedSchemaError):
        pass
    else:
        assert invocation.outcome.outcome is InvocationOutcome.TERMINAL, (
            "a null subtree over captured leaves must be exception or durable "
            f"TERMINAL, got {invocation.outcome.outcome!r}; calls={calls!r}"
        )
        assert invocation.outcome.reason.strip()
    assert calls == [], (
        "a supplied null trainer subtree erased captured trainer.max_steps and "
        "still reached the runner"
    )


def test_supplied_extra_opaque_leaf_still_accepted(monkeypatch, tmp_path: Path) -> None:
    """GREEN pin: extra supplied schema leaves outside the capture stay allowed."""

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    cfg = OmegaConf.create({**source, "legacy": {"flag": True}})
    calls = _run_spy(monkeypatch)

    WorkerStartGateway(tmp_path / "outputs").start_configured(
        cfg=cfg, batch=batch, intent=intent, phase=RecoveryPhase.INITIAL
    )
    assert len(calls) == 1


def test_start_receipt_records_declared_overlays(monkeypatch, tmp_path: Path) -> None:
    """RED at E3/CR: declared operational overlays must be durable in the receipt."""

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
    declared = start_on_disk.get("declared_overlays")
    assert declared == {"trainer.max_steps": 1}, (
        "the worker ran with a declared operational overlay that left no "
        f"durable trace in the start receipt; receipt keys={sorted(start_on_disk)}"
    )
