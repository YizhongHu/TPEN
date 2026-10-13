"""Source tests for the E3 worker-start and outcome gateway."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.execution import (
    CoreOutcomeFacts,
    CoreStatus,
    ExecutionConfig,
    InvocationOutcome,
    ReadyBatch,
    ReadyItem,
    RecoveryPhase,
    RecoveryDecision,
    ReceiptBindingError,
    ScienceIntent,
    WorkerStartGateway,
    capture_science_intent,
    payload_digest,
    run_configured_once,
    validate_start_receipt,
)


def _batch() -> tuple[ReadyBatch, ScienceIntent]:
    science = {"system": "hooke", "electrons": 2}
    intent = capture_science_intent(
        science,
        target_iterations=3,
        source_config={"model": {"_target_": "tests.helpers.NeverConstructed"}},
    )
    payload = {"science": science, "target_iterations": 3}
    source_digest = "a" * 64
    payload_digest_value = payload_digest(payload)
    config = ExecutionConfig(
        lineage_id="lineage-e3",
        backend="direct",
        recovery="checkpoint",
        source_config_digest=source_digest,
        payload_digest=payload_digest_value,
    )
    return (
        ReadyBatch(
            lineage_id="lineage-e3",
            batch_id="batch-e3",
            source_config_digest=source_digest,
            payload_digest=payload_digest_value,
            science_input_digest=intent.science_input_digest,
            target_iterations=3,
            execution=config,
            items=(
                ReadyItem(
                    invocation_id="logical-invocation-e3",
                    logical_task_id="task-e3",
                    fresh_output_id="planned-output-e3",
                    argv=("tpen-worker", "--config", "science.yaml"),
                    result_dir="results/e3",
                    payload={"seed": 7},
                ),
            ),
            payload=payload,
        ),
        intent,
    )


def _completed() -> CoreOutcomeFacts:
    return CoreOutcomeFacts.completed(target_iterations=3, next_iteration=3, completed_updates=3)


def test_start_receipt_is_durable_before_construction_and_restart_gets_fresh_path(tmp_path: Path) -> None:
    batch, intent = _batch()
    gateway = WorkerStartGateway(tmp_path)
    observed: list[Path] = []

    def construct_after_receipt(decision, output_path):
        observed.append(output_path)
        assert (output_path.parent / "start.json").is_file()
        return _completed()

    first = gateway.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=construct_after_receipt,
    )
    second = gateway.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=construct_after_receipt,
    )

    assert first.outcome.outcome is InvocationOutcome.COMPLETED
    assert second.outcome.outcome is InvocationOutcome.COMPLETED
    assert first.output_path != second.output_path
    assert len(observed) == 2
    assert first.output_path.is_dir() and second.output_path.is_dir()
    assert first.output_path.parent.joinpath("outcome.json").is_file()


def test_no_checkpoint_recovery_is_durable_park_and_does_not_call_core(tmp_path: Path) -> None:
    batch, intent = _batch()
    called = []

    result = WorkerStartGateway(tmp_path).start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.RECOVERY,
        run_once=lambda *_: called.append(True),
    )

    assert called == []
    assert result.outcome.outcome is InvocationOutcome.PARKED
    assert result.outcome.status is CoreStatus.UNKNOWN
    assert "no committed checkpoint" in result.outcome.reason
    assert json.loads((result.output_path.parent / "outcome.json").read_text())["outcome"] == "parked"


def test_crash_between_namespace_and_receipt_leaves_reconcilable_gap(tmp_path: Path) -> None:
    batch, intent = _batch()

    def crash(stage: str) -> None:
        if stage == "before_receipt_publish":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        WorkerStartGateway(tmp_path).start(
            batch=batch,
            intent=intent,
            phase=RecoveryPhase.INITIAL,
            run_once=lambda *_: _completed(),
            attempt_id="crashed-start",
            fault_injector=crash,
        )
    namespace = tmp_path / batch.lineage_id / batch.batch_id / "crashed-start"
    assert namespace.is_dir()
    assert not (namespace / "start.json").exists()
    assert WorkerStartGateway.reconcile(namespace) is None


def test_missing_outcome_after_durable_start_is_parked_without_resubmit(tmp_path: Path) -> None:
    batch, intent = _batch()

    def crash_after_receipt(stage: str) -> None:
        if stage == "receipt_published":
            raise RuntimeError("lost acknowledgement")

    with pytest.raises(RuntimeError, match="lost acknowledgement"):
        WorkerStartGateway(tmp_path).start(
            batch=batch,
            intent=intent,
            phase=RecoveryPhase.INITIAL,
            run_once=lambda *_: _completed(),
            attempt_id="lost-ack",
            fault_injector=crash_after_receipt,
        )
    namespace = tmp_path / batch.lineage_id / batch.batch_id / "lost-ack"
    parked = WorkerStartGateway.reconcile(namespace)
    assert parked is not None
    assert parked.outcome is InvocationOutcome.PARKED
    assert "no durable core outcome" in parked.reason


def test_reused_start_binding_is_refused(tmp_path: Path) -> None:
    batch, intent = _batch()
    result = WorkerStartGateway(tmp_path).start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
        attempt_id="binding-test",
    )
    try:
        validate_start_receipt(result.start, replace(batch, batch_id="changed"), batch.items[0])
    except ReceiptBindingError as exc:
        assert "different lineage or batch" in str(exc)
    else:  # pragma: no cover - assertion keeps the mutant polarity explicit
        raise AssertionError("changed receipt binding was accepted")


def test_reused_attempt_namespace_is_refused_without_a_second_core_call(tmp_path: Path) -> None:
    batch, intent = _batch()
    calls = []
    gateway = WorkerStartGateway(tmp_path)
    gateway.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: calls.append(True) or _completed(),
        attempt_id="same-attempt",
    )
    with pytest.raises(ReceiptBindingError, match="namespace already exists"):
        gateway.start(
            batch=batch,
            intent=intent,
            phase=RecoveryPhase.INITIAL,
            run_once=lambda *_: calls.append(True) or _completed(),
            attempt_id="same-attempt",
        )
    assert calls == [True]


def test_terminal_failure_is_not_a_retry_and_scheduler_zero_is_not_completion(tmp_path: Path) -> None:
    batch, intent = _batch()
    terminal = WorkerStartGateway(tmp_path).start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: CoreOutcomeFacts(
            status=CoreStatus.TERMINAL,
            target_iterations=3,
            next_iteration=0,
            completed_updates=0,
            reason="incompatible restore",
        ),
    )
    scheduler_zero = WorkerStartGateway(tmp_path).start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: 0,
    )
    assert terminal.outcome.outcome is InvocationOutcome.TERMINAL
    assert terminal.outcome.reason == "incompatible restore"
    assert scheduler_zero.outcome.outcome is InvocationOutcome.PARKED
    assert "completion facts" in scheduler_zero.outcome.reason


def test_completed_requires_both_distinct_core_counters_to_reach_target(tmp_path: Path) -> None:
    batch, intent = _batch()
    result = WorkerStartGateway(tmp_path).start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: CoreOutcomeFacts.completed(
            target_iterations=3, next_iteration=3, completed_updates=2
        ),
    )
    assert result.outcome.outcome is InvocationOutcome.TERMINAL
    assert "fixed target horizon" in result.outcome.reason
    assert result.outcome.next_iteration == 3
    assert result.outcome.completed_updates == 2


def test_run_configured_once_keeps_legacy_return_and_applies_only_worker_overlay(monkeypatch, tmp_path: Path) -> None:
    from tpen import run as run_module

    calls = []

    def fake_run(cfg, **kwargs):
        calls.append((cfg, kwargs))
        return 17

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    cfg = OmegaConf.create({"run": {"root": "original"}, "runner": {"load": None}})
    decision = RecoveryDecision(
        phase=RecoveryPhase.INITIAL,
        outcome="initial",
        target_iterations=3,
    )

    assert run_configured_once(cfg, decision=decision, output_path=tmp_path / "fresh", command="worker") == 17
    assert len(calls) == 1
    assert calls[0][0].run.root == str(tmp_path / "fresh")
    assert calls[0][0].runner.load is None
    assert calls[0][1] == {"command": "worker"}
