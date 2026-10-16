"""Formal review R4 controls for the E4 Balsam adapter."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.execution import (
    BalsamAdapter,
    BalsamAdapterStatus,
    BalsamAcknowledgementUnknown,
    BalsamBindingError,
    CoreOutcomeFacts,
    ExecutionConfig,
    InvocationOutcome,
    ReadyBatch,
    ReadyItem,
    ReceiptBindingError,
    RecoveryPhase,
    WorkerStartGateway,
    capture_science_intent,
    payload_digest,
    source_config_digest,
)


class _Client:
    def __init__(self) -> None:
        self.register_calls = 0
        self.observe_calls = 0

    def register_job(self, *, lineage_id, batch, backend_options):
        del lineage_id, batch, backend_options
        self.register_calls += 1
        return {"id": 42}

    def observe_job(self, job_id):
        del job_id
        self.observe_calls += 1
        return {"job_id": 42, "state": "active"}


def _batch(source: dict[str, object]) -> tuple[ReadyBatch, object]:
    science = {"system": "hooke", "electrons": 2}
    payload = {"science": science, "target_iterations": 2}
    intent = capture_science_intent(science, target_iterations=2, source_config=source)
    execution = ExecutionConfig(
        lineage_id="r4-lineage",
        backend="balsam",
        recovery="checkpoint",
        source_config_digest=source_config_digest(source),
        payload_digest=payload_digest(payload),
        backend_options={"site": "site-a"},
    )
    batch = ReadyBatch(
        lineage_id=execution.lineage_id,
        batch_id="r4-batch",
        source_config_digest=execution.source_config_digest,
        payload_digest=execution.payload_digest,
        science_input_digest=intent.science_input_digest,
        target_iterations=2,
        execution=execution,
        items=(ReadyItem("r4-inv", "r4-task", "r4-output", ("worker",), "results/r4"),),
        payload=payload,
    )
    return batch, intent


def _completed(*_args, **_kwargs) -> CoreOutcomeFacts:
    return CoreOutcomeFacts.completed(target_iterations=2, next_iteration=2, completed_updates=2)


def test_a1_source_digest_binding_refuses_changed_intent_and_cfg_before_science(
    monkeypatch, tmp_path: Path
) -> None:
    """A changed intent source cannot run against an immutable old batch."""

    old_source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}, "optimizer": {"lr": 0.01}}
    new_source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}, "optimizer": {"lr": 0.02}}
    batch, _old_intent = _batch(old_source)
    new_intent = capture_science_intent(
        {"system": "hooke", "electrons": 2}, target_iterations=2, source_config=new_source
    )
    calls: list[object] = []

    def fake_run(config, **kwargs):
        calls.append((config, kwargs))
        return 0

    from tpen import run as run_module

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    try:
        invocation = WorkerStartGateway(tmp_path / "outputs").start_configured(
            cfg=OmegaConf.create(new_source),
            batch=batch,
            intent=new_intent,
            phase=RecoveryPhase.INITIAL,
        )
    except ReceiptBindingError:
        pass
    else:
        assert invocation.outcome.outcome in {InvocationOutcome.TERMINAL, InvocationOutcome.PARKED}
        assert invocation.outcome.outcome is not InvocationOutcome.COMPLETED
        assert invocation.outcome.reason.strip()
    assert calls == []


def test_a1_matching_source_digest_control_invokes_once(monkeypatch, tmp_path: Path) -> None:
    """The matching-path control also pins receipt-before-call order."""

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    calls: list[object] = []
    receipt_already_durable: list[bool] = []
    from tpen import run as run_module

    def fake_run(config, **kwargs):
        calls.append(config)
        run_root = Path(config.run.root)
        receipt_already_durable.append((run_root.parent / "start.json").is_file())
        return 0

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    WorkerStartGateway(tmp_path / "outputs").start_configured(
        cfg=OmegaConf.create(source), batch=batch, intent=intent, phase=RecoveryPhase.INITIAL
    )
    assert len(calls) == 1
    assert receipt_already_durable == [True]


@pytest.mark.parametrize(
    "cfg",
    (
        {"runner": {"_target_": "tests.helpers.NeverConstructed", "load": None}},
        {"runner": {"_target_": "tests.helpers.NeverConstructed"}, "load": None},
    ),
    ids=("runner-load-null", "top-level-load-null"),
)
def test_a2_initial_null_load_is_not_over_refused(monkeypatch, tmp_path: Path, cfg: dict[str, object]) -> None:
    """INITIAL null load declarations are no-load and invoke once."""

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    calls: list[object] = []
    from tpen import run as run_module

    monkeypatch.setattr(run_module, "run_from_config", lambda config, **kwargs: calls.append(config) or 0)
    WorkerStartGateway(tmp_path / "outputs").start_configured(
        cfg=OmegaConf.create(cfg), batch=batch, intent=intent, phase=RecoveryPhase.INITIAL
    )
    assert len(calls) == 1


def test_a3_orphaned_outcome_is_not_reconciled_as_completed(tmp_path: Path) -> None:
    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    calls: list[object] = []

    def run_once(*args, **kwargs):
        calls.append((args, kwargs))
        return _completed(*args, **kwargs)

    invocation_a = WorkerStartGateway(tmp_path / "outputs-a").start(
        batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=run_once, attempt_id="a"
    )
    invocation_b = WorkerStartGateway(tmp_path / "outputs-b").start(
        batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=run_once, attempt_id="b"
    )
    shutil.copyfile(
        invocation_a.output_path.parent / "outcome.json",
        invocation_b.output_path.parent / "outcome.json",
    )
    try:
        reconciled = WorkerStartGateway.reconcile(invocation_b.output_path.parent)
    except ReceiptBindingError:
        pass
    else:
        assert reconciled is not None
        assert reconciled.outcome in {InvocationOutcome.TERMINAL, InvocationOutcome.PARKED}
        assert reconciled.outcome is not InvocationOutcome.COMPLETED
        assert reconciled.reason.strip()
    assert len(calls) == 2


def test_a3_matching_outcome_control_reconciles(tmp_path: Path) -> None:
    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    gateway = WorkerStartGateway(tmp_path / "outputs")
    invocation = gateway.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=_completed)
    reconciled = WorkerStartGateway.reconcile(invocation.output_path.parent)
    assert reconciled is not None
    assert reconciled.start_receipt_digest == invocation.start.receipt_id


def test_a3_missing_outcome_receipt_parks_durably_on_reconcile(tmp_path: Path) -> None:
    """A crash after the start receipt but before outcome writing parks durably."""

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    gateway = WorkerStartGateway(tmp_path / "outputs")

    def crash_after_publish(stage: str) -> None:
        if stage == "receipt_published":
            raise RuntimeError("simulated crash after start receipt, before outcome")

    with pytest.raises(RuntimeError):
        gateway.start(
            batch=batch,
            intent=intent,
            phase=RecoveryPhase.INITIAL,
            run_once=_completed,
            fault_injector=crash_after_publish,
        )
    namespaces = list((tmp_path / "outputs").glob("*/*/*"))
    assert len(namespaces) == 1
    namespace = namespaces[0]
    assert (namespace / "start.json").is_file()
    assert not (namespace / "outcome.json").exists()
    reconciled = WorkerStartGateway.reconcile(namespace)
    assert reconciled is not None
    assert reconciled.outcome is InvocationOutcome.PARKED
    assert (namespace / "outcome.json").is_file()


def test_a4_backend_options_are_bound_before_registration(tmp_path: Path) -> None:
    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, _intent = _batch(source)
    changed = ExecutionConfig(
        lineage_id=batch.execution.lineage_id,
        backend="balsam",
        recovery=batch.execution.recovery,
        source_config_digest=batch.execution.source_config_digest,
        payload_digest=batch.execution.payload_digest,
        backend_options={"site": "site-b"},
    )
    client = _Client()
    adapter = BalsamAdapter(changed, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts")
    with pytest.raises(BalsamBindingError):
        adapter.register_lineage(batch)
    assert client.register_calls == 0


def test_a4_matching_backend_options_control_registers_once(tmp_path: Path) -> None:
    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, _intent = _batch(source)
    client = _Client()
    adapter = BalsamAdapter(batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts")
    assert adapter.register_lineage(batch) is not None
    assert client.register_calls == 1


def test_a5_concurrent_worker_parks_preserve_both_reasons(monkeypatch, tmp_path: Path) -> None:
    from tpen.execution import balsam as balsam_module

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    client = _Client()
    adapter = BalsamAdapter(batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts")
    assert adapter.register_lineage(batch) is not None
    barrier = threading.Barrier(2)
    original_publish = balsam_module._publish_distinct_json

    def synchronized_publish(path: Path, payload: dict[str, object]) -> None:
        if path.name.startswith("worker-park"):
            barrier.wait(timeout=10)
        original_publish(path, payload)

    monkeypatch.setattr(balsam_module, "_publish_distinct_json", synchronized_publish)
    results: list[object] = []
    errors: list[BaseException] = []

    def launch(job_id: str) -> None:
        try:
            results.append(
                adapter.start(
                    batch=batch,
                    intent=intent,
                    phase=RecoveryPhase.INITIAL,
                    run_once=_completed,
                    job_id=job_id,
                )
            )
        except BaseException as exc:  # preserve thread failure as evidence
            errors.append(exc)

    threads = [threading.Thread(target=launch, args=("",)), threading.Thread(target=launch, args=("wrong-b",))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(results) == 2
    assert all(result.status is BalsamAdapterStatus.PARKED for result in results)
    park_files = sorted((tmp_path / "receipts" / batch.lineage_id).glob("*worker-park*.json"))
    reasons = {json.loads(path.read_text(encoding="utf-8"))["reason"] for path in park_files}
    assert len(reasons) == 2
    assert client.register_calls == 1
    assert client.observe_calls == 0


def test_a5_matching_sequential_worker_starts_remain_started(tmp_path: Path) -> None:
    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    client = _Client()
    adapter = BalsamAdapter(batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts")
    assert adapter.register_lineage(batch) is not None
    first = adapter.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=_completed, job_id="42")
    second = adapter.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=_completed, job_id="42")
    assert first.status is BalsamAdapterStatus.STARTED
    assert second.status is BalsamAdapterStatus.STARTED
    assert client.register_calls == 1
    assert client.observe_calls == 0


def test_a5_repeated_identical_worker_park_is_idempotent(tmp_path: Path) -> None:
    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    client = _Client()
    adapter = BalsamAdapter(batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts")
    assert adapter.register_lineage(batch) is not None
    adapter.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=_completed, job_id="")
    adapter.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=_completed, job_id="")
    parks = sorted((tmp_path / "receipts" / batch.lineage_id).glob("*worker-park*.json"))
    assert len(parks) == 1
    assert "missing or invalid" in json.loads(parks[0].read_text(encoding="utf-8"))["reason"]
    assert client.register_calls == 1
    assert client.observe_calls == 0


def test_a6_observation_parks_preserve_distinct_backend_errors(tmp_path: Path) -> None:
    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, _intent = _batch(source)
    client = _Client()
    adapter = BalsamAdapter(batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts")
    assert adapter.register_lineage(batch) is not None
    reasons = iter(("socket timeout", "gateway reset"))

    def observe(_job_id: str) -> object:
        raise OSError(next(reasons))

    client.observe_job = observe  # type: ignore[method-assign]
    assert adapter.observe_lineage(batch) is None
    assert adapter.observe_lineage(batch) is None
    parks = sorted((tmp_path / "receipts" / batch.lineage_id).glob("observation-park*.json"))
    assert {json.loads(path.read_text(encoding="utf-8"))["reason"] for path in parks} == {
        "backend Job observation is unknown; park for reconciliation: socket timeout",
        "backend Job observation is unknown; park for reconciliation: gateway reset",
    }
    assert len(parks) == 2
    assert not (tmp_path / "receipts" / batch.lineage_id / "observation.json").exists()


def test_a6_identical_observation_park_is_idempotent_without_submission(tmp_path: Path) -> None:
    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, _intent = _batch(source)
    client = _Client()
    adapter = BalsamAdapter(batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts")
    assert adapter.register_lineage(batch) is not None
    submission_calls = client.register_calls

    def observe(_job_id: str) -> object:
        raise OSError("same backend error")

    client.observe_job = observe  # type: ignore[method-assign]
    assert adapter.observe_lineage(batch) is None
    assert adapter.observe_lineage(batch) is None
    parks = sorted((tmp_path / "receipts" / batch.lineage_id).glob("observation-park*.json"))
    assert len(parks) == 1
    assert "same backend error" in json.loads(parks[0].read_text(encoding="utf-8"))["reason"]
    assert client.register_calls == submission_calls


def test_a6_reentrant_registration_parks_both_unresolved_ack_reasons(tmp_path: Path) -> None:
    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, _intent = _batch(source)
    client = _Client()
    adapter = BalsamAdapter(batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts")
    entered = False

    def register(**_kwargs) -> object:
        nonlocal entered
        if not entered:
            entered = True
            assert adapter.register_lineage(batch) is None
        raise BalsamAcknowledgementUnknown("lost acknowledgement")

    client.register_job = register  # type: ignore[method-assign]
    assert adapter.register_lineage(batch) is None
    parks = sorted((tmp_path / "receipts" / batch.lineage_id).glob("registration-park*.json"))
    reasons = {json.loads(path.read_text(encoding="utf-8"))["reason"] for path in parks}
    assert reasons == {
        "registration acknowledgement is unresolved; park without resubmitting",
        "registration acknowledgement was lost; park without resubmitting",
    }
    assert len(parks) == 2
