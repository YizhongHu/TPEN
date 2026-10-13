"""Source tests for the E3 worker-start and outcome gateway."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.checkpoint import CheckpointRef, resolve_checkpoint_dir
from tpen.checkpoint.payload import TrainResume
from tpen.config_schema import ClosedSchemaError
from tpen.execution import (
    CheckpointEvidence,
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


def _write_core_completion_artifacts(output_path: Path, *, next_iteration: int = 3, completed_updates: int = 3) -> None:
    """Write the core-owned status and checkpoint shape used by gateway tests."""

    run_dir = output_path / "experiment" / "sector" / "run-id"
    checkpoint_dir = run_dir / "checkpoints" / f"step_{next_iteration:06d}"
    checkpoint_dir.mkdir(parents=True)
    files = {
        "model": "model.pt",
        "optimizer": "optimizer.pt",
        "trainer": "trainer.json",
        "sampler": "sampler.pt",
        "rng": "rng.pt",
    }
    for name in files.values():
        (checkpoint_dir / name).write_bytes(b"core-artifact")
    (checkpoint_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "tpen.checkpoint",
                "next_iteration": next_iteration,
                "completed_updates": completed_updates,
                "created_at_unix": 1.0,
                "files": files,
                "hashes": {},
                "runtime": {"device": "cpu", "dtype": "float64"},
                "provenance": {"run_id": "run-id"},
                "payload": TrainResume().to_manifest(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (checkpoint_dir.parent / "latest.json").write_text(
        json.dumps({"checkpoint_dir": checkpoint_dir.name, "step": next_iteration}),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"status": "completed", "current_event": "run_end"}),
        encoding="utf-8",
    )


def test_configured_gateway_materializes_completion_from_core_artifacts(monkeypatch, tmp_path: Path) -> None:
    """A successful configured call completes only from status plus checkpoint facts."""

    from tpen import run as run_module

    batch, intent = _batch()

    def fake_run(cfg, **kwargs):
        assert kwargs["raise_exceptions"] is True
        _write_core_completion_artifacts(Path(cfg.run.root))
        return 0

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    result = WorkerStartGateway(tmp_path).start_configured(
        cfg=OmegaConf.create({"run": {"root": "ignored"}}),
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
    )

    assert result.run_result.status is CoreStatus.COMPLETED
    assert result.outcome.outcome is InvocationOutcome.COMPLETED
    assert result.outcome.next_iteration == 3
    assert result.outcome.completed_updates == 3


def test_configured_zero_return_without_final_checkpoint_is_parked(monkeypatch, tmp_path: Path) -> None:
    """An integer zero cannot replace a committed final-counter artifact."""

    from tpen import run as run_module

    batch, intent = _batch()

    def fake_run(cfg, **kwargs):
        assert kwargs["raise_exceptions"] is True
        run_dir = Path(cfg.run.root) / "experiment" / "sector" / "run-id"
        run_dir.mkdir(parents=True)
        (run_dir / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        return 0

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    result = WorkerStartGateway(tmp_path).start_configured(
        cfg=OmegaConf.create({"run": {"root": "ignored"}}),
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
    )

    assert result.outcome.outcome is InvocationOutcome.PARKED
    assert "no committed final checkpoint" in result.outcome.reason


def test_configured_gateway_forces_raise_exceptions_for_named_config_refusal(monkeypatch, tmp_path: Path) -> None:
    """The gateway preserves a deterministic core refusal instead of parking return 1."""

    from tpen import run as run_module

    batch, intent = _batch()

    def fake_run(cfg, **kwargs):
        assert kwargs["raise_exceptions"] is True
        raise ClosedSchemaError([])

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    result = WorkerStartGateway(tmp_path).start_configured(
        cfg=OmegaConf.create({"run": {"root": "ignored"}}),
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
    )

    assert result.outcome.outcome is InvocationOutcome.TERMINAL
    assert result.outcome.exception_type == "ClosedSchemaError"


def test_configured_gateway_does_not_classify_training_value_error_as_terminal(monkeypatch, tmp_path: Path) -> None:
    """A numeric/application ValueError outside a named core gate parks."""

    from tpen import run as run_module

    batch, intent = _batch()

    def fake_run(cfg, **kwargs):
        assert kwargs["raise_exceptions"] is True
        raise ValueError("numeric failure in training")

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    result = WorkerStartGateway(tmp_path).start_configured(
        cfg=OmegaConf.create({"run": {"root": "ignored"}}),
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
    )

    assert result.outcome.outcome is InvocationOutcome.PARKED
    assert result.outcome.status is CoreStatus.UNKNOWN


def test_configured_gateway_runs_real_tiny_train_and_restarts_in_distinct_namespace(tmp_path: Path) -> None:
    """The actual configured entry point supplies completion facts to the gateway."""

    pytest.importorskip("torch")
    config_path = Path(__file__).parents[2] / "integration" / "artifacts" / "training" / "vmc_smoke.yaml"
    cfg = OmegaConf.load(config_path)
    batch, intent = _batch()
    gateway = WorkerStartGateway(tmp_path)

    first = gateway.start_configured(
        cfg=cfg,
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
    )
    second = gateway.start_configured(
        cfg=cfg,
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
    )

    assert first.outcome.outcome is InvocationOutcome.COMPLETED
    assert first.outcome.next_iteration == 3
    assert first.outcome.completed_updates == 3
    assert second.outcome.outcome is InvocationOutcome.COMPLETED
    assert first.output_path != second.output_path
    assert len(tuple(first.output_path.rglob("status.json"))) == 1
    assert len(tuple(second.output_path.rglob("status.json"))) == 1


def test_real_bad_config_is_terminal(tmp_path: Path) -> None:
    """The configured preconstruction schema refusal is terminal, not a retry."""

    from tpen.hi_schema import HI_EXPERIMENT_NAME

    config_path = Path(__file__).parents[2] / "integration" / "artifacts" / "training" / "vmc_smoke.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.experiment.name = HI_EXPERIMENT_NAME
    batch, intent = _batch()

    result = WorkerStartGateway(tmp_path).start_configured(
        cfg=cfg,
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
    )

    assert result.outcome.outcome is InvocationOutcome.TERMINAL
    assert result.outcome.exception_type == "ClosedSchemaError"


def test_real_incompatible_train_resume_restore_is_terminal(tmp_path: Path) -> None:
    """A core restore/schema refusal is terminal and never becomes a retry."""

    pytest.importorskip("torch")
    config_path = Path(__file__).parents[2] / "integration" / "artifacts" / "training" / "vmc_smoke.yaml"
    cfg = OmegaConf.load(config_path)
    batch, intent = _batch()
    gateway = WorkerStartGateway(tmp_path)
    first = gateway.start_configured(
        cfg=cfg,
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
    )
    assert first.outcome.outcome is InvocationOutcome.COMPLETED
    checkpoint_dir = resolve_checkpoint_dir(next(first.output_path.rglob("checkpoints")))
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence = CheckpointEvidence.from_ref(
        ref,
        science_input=intent.science_input,
        target_iterations=intent.target_iterations,
    )

    incompatible = OmegaConf.load(config_path)
    incompatible.system.omega = 0.75
    result = gateway.start_configured(
        cfg=incompatible,
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.RECOVERY,
        evidence=evidence,
    )

    assert result.outcome.outcome is InvocationOutcome.TERMINAL
    assert result.outcome.exception_type == "ValueError"
