"""Formal review round 5 reviewer tests (E3): issues 5 (control), 6, 8, 9 and NEW-B1."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.execution import WorkerStartGateway


def _r5_batch(source, *, backend="direct"):
    from tpen.execution import ExecutionConfig, ReadyBatch, ReadyItem, capture_science_intent, payload_digest, source_config_digest

    payload = {"science": {"system": "hooke", "electrons": 2}, "target_iterations": 2}
    intent = capture_science_intent(payload["science"], target_iterations=2, source_config=source)
    execution = ExecutionConfig("r5-lineage", backend, "checkpoint", source_config_digest(source), payload_digest(payload), backend_options={"site": "site-a"})
    batch = ReadyBatch("r5-lineage", "r5-batch", execution.source_config_digest, execution.payload_digest, intent.science_input_digest, 2, execution, (ReadyItem("r5-inv", "r5-task", "r5-output", ("worker",), "results/r5"),), payload)
    return batch, intent


def _completed(*_args, **_kwargs):
    from tpen.execution import CoreOutcomeFacts

    return CoreOutcomeFacts.completed(target_iterations=2, next_iteration=2, completed_updates=2)


def _gateway(tmp_path):
    return WorkerStartGateway(tmp_path / "outputs")


@pytest.mark.parametrize("changed", [False, True])
def test_r5_bound_configured_adapter_entry_uses_intent_materialization(monkeypatch, tmp_path, changed):
    from tpen import run as run_module
    from tpen.execution import InvocationOutcome, RecoveryPhase, run_bound_configured_once

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}, "optimizer": {"lr": 0.01}}
    cfg = OmegaConf.create({**source, "optimizer": {"lr": 0.02 if changed else 0.01}})
    batch, intent = _r5_batch(source)
    calls = []

    def core(worker_cfg, **kwargs):
        assert (Path(worker_cfg.run.root).parent / "start.json").is_file()
        assert kwargs["raise_exceptions"] is True
        calls.append(worker_cfg.optimizer.lr)
        return 0

    monkeypatch.setattr(run_module, "run_from_config", core)
    result = _gateway(tmp_path).start(
        batch=batch, intent=intent, phase=RecoveryPhase.INITIAL,
        run_once=lambda decision, output_path: run_bound_configured_once(
            cfg, intent=intent, decision=decision, output_path=output_path,
        ),
    )
    if changed:
        assert calls == []
        assert result.outcome.outcome is InvocationOutcome.TERMINAL
        assert result.outcome.exception_type == "ReceiptBindingError"
    else:
        assert calls == [0.01]
        assert result.outcome.outcome is InvocationOutcome.PARKED


@pytest.mark.parametrize("which", ["start", "outcome"])
def test_r5_worker_receipt_decoders_own_non_mapping_refusal(tmp_path, which):
    from tpen.execution import RecoveryPhase

    batch, intent = _r5_batch({"runner": {"_target_": "tests.helpers.NeverConstructed"}})
    result = _gateway(tmp_path).start(
        batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=lambda *_: _completed()
    )
    cls = type(result.start if which == "start" else result.outcome)
    with pytest.raises(ValueError):
        cls.from_mapping(None)


# PREDICTED: RED at E3, E4, E5, CR (issue 6)
def test_r5_reconcile_during_live_start_does_not_get_silently_replaced(tmp_path):
    from tpen.execution import ReceiptBindingError, RecoveryPhase

    batch, intent = _r5_batch({"runner": {"_target_": "tests.helpers.NeverConstructed"}})
    gateway = _gateway(tmp_path)
    stored = []
    namespaces = []

    def run_once(_decision, output_path):
        # A reconciler that runs while this worker is still alive (the lost-ack case).
        namespaces.append(Path(output_path).parent)
        stored.append(WorkerStartGateway.reconcile(Path(output_path).parent))
        return _completed()

    raised = None
    try:
        gateway.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=run_once)
    except (ReceiptBindingError, FileExistsError) as exc:
        raised = type(exc).__name__
    assert stored and stored[0] is not None, f"observed reconcile receipt={stored!r}"
    assert stored[0].outcome.value == "parked", f"observed reconcile outcome={stored[0].outcome!r}"
    durable = json.loads((namespaces[0] / "outcome.json").read_text(encoding="utf-8"))
    assert durable == stored[0].to_dict(), (
        "durable outcome.json was replaced after a reconciler published it: "
        f"durable outcome={durable.get('outcome')!r}, published outcome={stored[0].outcome.value!r}, "
        f"start raised={raised!r}"
    )


# PREDICTED: GREEN at E3, E4, E5, CR (issue 6 control)
def test_r5_outcome_receipt_without_concurrent_reconcile_is_stable(tmp_path):
    from tpen.execution import RecoveryPhase

    batch, intent = _r5_batch({"runner": {"_target_": "tests.helpers.NeverConstructed"}})
    gateway = _gateway(tmp_path)
    result = gateway.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=lambda *_: _completed())
    namespace = Path(result.start.output_path).parent
    durable = json.loads((namespace / "outcome.json").read_text(encoding="utf-8"))
    assert durable == result.outcome.to_dict(), f"durable outcome.json observed {durable!r}"
    later = WorkerStartGateway.reconcile(namespace)
    assert later == result.outcome, f"later reconcile observed {later!r}, expected {result.outcome!r}"
    assert later.outcome.value == "completed", f"later reconcile outcome observed {later.outcome!r}"


@pytest.mark.parametrize("which", ["start", "outcome"])
# PREDICTED: RED at E3, E4, E5, CR (issue 9)
def test_r5_worker_receipt_decoders_refuse_unknown_keys(tmp_path, which):
    from tpen.execution import RecoveryPhase, WorkerOutcomeReceipt, WorkerStartReceipt

    batch, intent = _r5_batch({"runner": {"_target_": "tests.helpers.NeverConstructed"}})
    result = _gateway(tmp_path).start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=lambda *_: _completed())
    receipt = result.start if which == "start" else result.outcome
    cls = WorkerStartReceipt if which == "start" else WorkerOutcomeReceipt
    data = receipt.to_dict(); data["unexpected"] = "x"
    with pytest.raises((ValueError, TypeError)):
        cls.from_mapping(data)


# PREDICTED: GREEN at E3, E4, E5, CR (issue 9 control)
def test_r5_worker_receipt_decoders_round_trip(tmp_path):
    from tpen.execution import RecoveryPhase, WorkerOutcomeReceipt, WorkerStartReceipt

    batch, intent = _r5_batch({"runner": {"_target_": "tests.helpers.NeverConstructed"}})
    result = _gateway(tmp_path).start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=lambda *_: _completed())
    assert WorkerStartReceipt.from_mapping(result.start.to_dict()) == result.start, "start round trip differed"
    assert WorkerOutcomeReceipt.from_mapping(result.outcome.to_dict()) == result.outcome, "outcome round trip differed"


@pytest.mark.parametrize("config", [
    {"load": {"path": None, "mode": "none", "strict": True}},
    {"load": {"mode": "none"}, "runner": {"load": "${load}"}},
    {"load": None, "runner": {"load": "${load}"}},
], ids=["explicit-mode-none", "mode-none-interpolated", "null-interpolated"])
# PREDICTED: RED at E3, E4, E5, CR (NEW-B1)
def test_r5_initial_canonical_noload_config_invokes_runner_once(monkeypatch, tmp_path, config):
    from tpen import run as run_module
    from tpen.execution import RecoveryPhase

    config = {**config, "run": {"root": "outputs"}, "trainer": {"max_steps": 2}, "runner": {"_target_": "tests.helpers.NeverConstructed", **config.get("runner", {})}}
    calls = []
    monkeypatch.setattr(run_module, "run_from_config", lambda cfg, **kwargs: calls.append(cfg) or 0)
    batch, intent = _r5_batch(config)
    result = _gateway(tmp_path).start_configured(cfg=OmegaConf.create(config), batch=batch, intent=intent, phase=RecoveryPhase.INITIAL)
    assert len(calls) == 1, f"calls={len(calls)} outcome={result.outcome.outcome!r} reason={result.outcome.reason!r}"


# PREDICTED: GREEN at E3, E4, E5, CR (NEW-B1 control)
def test_r5_initial_plain_null_load_invokes_runner_once(monkeypatch, tmp_path):
    from tpen import run as run_module
    from tpen.execution import RecoveryPhase

    config = {"load": None, "run": {"root": "outputs"}, "trainer": {"max_steps": 2}, "runner": {"_target_": "tests.helpers.NeverConstructed"}}
    calls = []
    monkeypatch.setattr(run_module, "run_from_config", lambda cfg, **kwargs: calls.append(cfg) or 0)
    batch, intent = _r5_batch(config)
    result = _gateway(tmp_path).start_configured(cfg=OmegaConf.create(config), batch=batch, intent=intent, phase=RecoveryPhase.INITIAL)
    assert len(calls) == 1, f"calls={len(calls)} outcome={result.outcome.outcome!r} reason={result.outcome.reason!r}"


# PREDICTED: GREEN at E3, E4, E5, CR (issue 5 control)
def test_r5_start_configured_refuses_cfg_not_bound_to_intent(monkeypatch, tmp_path):
    from tpen import run as run_module
    from tpen.execution import InvocationOutcome, RecoveryPhase

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}, "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.01}, "run": {"root": "outputs"}}
    changed = {**source, "optimizer": {**source["optimizer"], "lr": 0.02}}
    calls = []
    monkeypatch.setattr(run_module, "run_from_config", lambda cfg, **kwargs: calls.append(cfg) or 0)
    batch, intent = _r5_batch(source)
    result = _gateway(tmp_path).start_configured(cfg=OmegaConf.create(changed), batch=batch, intent=intent, phase=RecoveryPhase.INITIAL)
    assert calls == [], f"runner calls observed: {calls!r}"
    assert result.outcome.outcome is InvocationOutcome.TERMINAL, f"observed outcome={result.outcome.outcome!r}"
