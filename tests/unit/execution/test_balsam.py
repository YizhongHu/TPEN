"""Source contract tests for the optional Balsam lineage adapter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tpen.execution import (
    BalsamAcknowledgementUnknown,
    BalsamAdapter,
    BalsamAdapterStatus,
    BalsamBindingError,
    BalsamJobObservation,
    BalsamJobState,
    BalsamPermanentRefusal,
    CoreOutcomeFacts,
    ExecutionConfig,
    ReadyBatch,
    ReadyItem,
    RecoveryPhase,
    ScienceIntent,
    WorkerStartGateway,
    capture_science_intent,
    payload_digest,
)


class FakeBalsamClient:
    """Test seam for registration and observation, with no optional import."""

    def __init__(self, *, state: BalsamJobState = BalsamJobState.ACTIVE) -> None:
        self.state = state
        self.register_calls = 0
        self.observe_calls = 0
        self.register_exception: Exception | None = None
        self.observe_exception: Exception | None = None
        self.ack: object = {"job_id": "balsam-job-opaque"}

    def register_job(self, *, lineage_id, batch, backend_options):
        self.register_calls += 1
        assert lineage_id == batch.lineage_id
        assert batch.execution.backend == "balsam"
        assert backend_options["site"] == "test-site"
        if self.register_exception is not None:
            raise self.register_exception
        return self.ack

    def observe_job(self, job_id):
        self.observe_calls += 1
        if self.observe_exception is not None:
            raise self.observe_exception
        return BalsamJobObservation(job_id=job_id, state=self.state)


def _batch(*, payload: dict[str, object] | None = None, backend_options=None):
    science = {"system": "hooke", "electrons": 2}
    intent = capture_science_intent(
        science,
        target_iterations=2,
        source_config={"model": {"_target_": "tests.helpers.NeverConstructed"}},
    )
    payload = payload or {"science": science, "target_iterations": 2}
    source_digest = "a" * 64
    payload_digest_value = payload_digest(payload)
    config = ExecutionConfig(
        lineage_id="lineage-balsam",
        backend="balsam",
        recovery="checkpoint",
        source_config_digest=source_digest,
        payload_digest=payload_digest_value,
        backend_options=backend_options or {"site": "test-site"},
    )
    return (
        ReadyBatch(
            lineage_id="lineage-balsam",
            batch_id="batch-balsam",
            source_config_digest=source_digest,
            payload_digest=payload_digest_value,
            science_input_digest=intent.science_input_digest,
            target_iterations=2,
            execution=config,
            items=(
                ReadyItem(
                    invocation_id="logical-invocation-balsam",
                    logical_task_id="task-balsam",
                    fresh_output_id="planned-output-balsam",
                    argv=("tpen-worker", "--config", "science.yaml"),
                    result_dir="results/balsam",
                    payload={"seed": 7},
                ),
            ),
            payload=payload,
        ),
        intent,
    )


def _completed():
    return CoreOutcomeFacts.completed(target_iterations=2, next_iteration=2, completed_updates=1)


def _adapter(tmp_path: Path, client: FakeBalsamClient, batch: ReadyBatch):
    return BalsamAdapter(
        batch.execution,
        client,
        WorkerStartGateway(tmp_path / "outputs"),
        tmp_path / "receipts",
    )


def _register(adapter: BalsamAdapter, batch: ReadyBatch):
    receipt = adapter.register_lineage(batch)
    assert receipt is not None
    return receipt


def test_registration_is_one_persistent_job_and_each_start_is_a_fresh_e3_invocation(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    adapter = _adapter(tmp_path, client, batch)
    registration = _register(adapter, batch)
    output_paths: list[Path] = []

    def run_once(_decision, output_path):
        output_paths.append(output_path)
        assert (output_path.parent / "start.json").is_file()
        return _completed()

    first = adapter.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=run_once, job_id=registration.job_id)
    second = adapter.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=run_once, job_id=registration.job_id)

    assert first.status is BalsamAdapterStatus.STARTED
    assert second.status is BalsamAdapterStatus.STARTED
    assert first.registration is not None
    assert first.registration.job_id == second.registration.job_id == "balsam-job-opaque"
    assert client.register_calls == 1
    assert client.observe_calls == 0
    assert output_paths[0] != output_paths[1]
    assert (tmp_path / "receipts" / "lineage-balsam" / "registration-intent.json").is_file()
    assert (tmp_path / "receipts" / "lineage-balsam" / "registration.json").is_file()


def test_changed_intent_refuses_reuse_before_client_call(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    adapter = _adapter(tmp_path, client, batch)
    _register(adapter, batch)

    changed_batch, changed_intent = _batch(payload={"science": {"system": "different"}, "target_iterations": 2})
    changed_adapter = _adapter(tmp_path, client, changed_batch)
    with pytest.raises(BalsamBindingError, match="different Balsam intent"):
        changed_adapter.register_lineage(changed_batch)
    assert client.register_calls == 1


def test_lost_registration_ack_is_durable_park_and_never_resubmitted(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    client.register_exception = BalsamAcknowledgementUnknown("client call lost")
    adapter = _adapter(tmp_path, client, batch)

    first = adapter.register_lineage(batch)
    client.register_exception = None
    second = adapter.register_lineage(batch)

    assert first is None
    assert second is None
    assert client.register_calls == 1
    assert (tmp_path / "receipts" / "lineage-balsam" / "registration-park.json").is_file()


def test_client_exception_after_intent_is_parked_without_blind_resubmit(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    client.register_exception = OSError("connection reset after request")
    adapter = _adapter(tmp_path, client, batch)

    result = adapter.register_lineage(batch)

    assert result is None
    assert client.register_calls == 1


def test_allocation_loss_is_backend_lifecycle_and_e3_remains_outcome_authority(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient(state=BalsamJobState.ALLOCATION_LOST)
    adapter = _adapter(tmp_path, client, batch)
    registration = _register(adapter, batch)
    observed = adapter.observe_lineage(batch)
    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
        job_id=registration.job_id,
    )

    assert observed is not None
    assert client.observe_calls == 1
    assert result.status is BalsamAdapterStatus.STARTED
    assert result.observation is None
    assert result.invocation is not None
    assert result.invocation.outcome.outcome.value == "completed"
    assert (tmp_path / "receipts" / "lineage-balsam" / "observation.json").is_file()


def test_explicit_backend_refusal_is_typed_and_does_not_register(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    client.register_exception = BalsamPermanentRefusal("invalid Balsam application")
    with pytest.raises(BalsamPermanentRefusal, match="invalid Balsam application"):
        _adapter(tmp_path, client, batch).register_lineage(batch)
    assert client.register_calls == 1
    assert (tmp_path / "receipts" / "lineage-balsam" / "registration-intent.json").is_file()
    assert not (tmp_path / "receipts" / "lineage-balsam" / "registration.json").exists()
    assert (tmp_path / "receipts" / "lineage-balsam" / "registration-refusal.json").is_file()


def test_importing_core_and_adapter_does_not_import_optional_balsam() -> None:
    code = """
import sys
import tpen.execution
import tpen.execution.balsam
assert 'balsam' not in sys.modules
assert not any(name == 'balsam.api' or name.startswith('balsam.') for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_registration_receipt_binding_mutation_is_refused(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    adapter = _adapter(tmp_path, client, batch)
    registration = _register(adapter, batch)
    path = tmp_path / "receipts" / "lineage-balsam" / "registration.json"
    data = path.read_text()
    path.write_text(data.replace(batch.payload_digest, "c" * 64))
    with pytest.raises(BalsamBindingError, match="receipt payload_digest"):
        adapter.start(
            batch=batch,
            intent=intent,
            phase=RecoveryPhase.INITIAL,
            run_once=lambda *_: _completed(),
            job_id=registration.job_id,
        )


def test_worker_without_registration_never_registers_and_durably_parks(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    adapter = _adapter(tmp_path, client, batch)
    called = False

    def run_once(*_args):
        nonlocal called
        called = True
        return _completed()

    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=run_once,
        job_id="balsam-job-opaque",
    )

    assert result.status is BalsamAdapterStatus.PARKED
    assert not called
    assert client.register_calls == 0
    assert client.observe_calls == 0
    assert (tmp_path / "receipts" / "lineage-balsam" / "worker-park.json").is_file()


def test_worker_with_registration_park_never_registers_again(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    client.register_exception = BalsamAcknowledgementUnknown("lost ack")
    adapter = _adapter(tmp_path, client, batch)
    assert adapter.register_lineage(batch) is None

    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
        job_id="balsam-job-opaque",
    )

    assert result.status is BalsamAdapterStatus.PARKED
    assert client.register_calls == 1
    assert client.observe_calls == 0


def test_worker_mismatched_job_id_parks_without_backend_calls(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    adapter = _adapter(tmp_path, client, batch)
    _register(adapter, batch)

    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
        job_id="another-job",
    )

    assert result.status is BalsamAdapterStatus.PARKED
    assert "differs" in result.reason
    assert client.register_calls == 1
    assert client.observe_calls == 0


def test_int_job_ack_is_canonicalized_and_reused_by_worker(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    client.ack = {"job_id": 12345}
    adapter = _adapter(tmp_path, client, batch)
    registration = _register(adapter, batch)

    assert registration.job_id == "12345"
    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
        job_id=12345,
    )
    assert result.status is BalsamAdapterStatus.STARTED
    assert client.register_calls == 1
    assert client.observe_calls == 0


def test_observation_failure_is_submitter_park_and_worker_still_starts(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    adapter = _adapter(tmp_path, client, batch)
    registration = _register(adapter, batch)
    client.observe_exception = OSError("proxy unavailable")

    assert adapter.observe_lineage(batch) is None
    assert client.observe_calls == 1
    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
        job_id=registration.job_id,
    )
    assert result.status is BalsamAdapterStatus.STARTED
    assert result.invocation is not None


def test_permanent_observation_refusal_is_typed(tmp_path: Path) -> None:
    batch, _intent = _batch()
    client = FakeBalsamClient(state=BalsamJobState.PERMANENT_FAILURE)
    adapter = _adapter(tmp_path, client, batch)
    _register(adapter, batch)

    with pytest.raises(BalsamPermanentRefusal, match="Balsam reported permanent"):
        adapter.observe_lineage(batch)
    assert (tmp_path / "receipts" / "lineage-balsam" / "observation.json").is_file()
