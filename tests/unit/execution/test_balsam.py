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

    def register_job(self, *, lineage_id, batch, backend_options):
        self.register_calls += 1
        assert lineage_id == batch.lineage_id
        assert batch.execution.backend == "balsam"
        assert backend_options["site"] == "test-site"
        if self.register_exception is not None:
            raise self.register_exception
        return {"job_id": "balsam-job-opaque"}

    def observe_job(self, job_id):
        self.observe_calls += 1
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


def test_registration_is_one_persistent_job_and_each_start_is_a_fresh_e3_invocation(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    adapter = _adapter(tmp_path, client, batch)
    output_paths: list[Path] = []

    def run_once(_decision, output_path):
        output_paths.append(output_path)
        assert (output_path.parent / "start.json").is_file()
        return _completed()

    first = adapter.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=run_once)
    second = adapter.start(batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=run_once)

    assert first.status is BalsamAdapterStatus.STARTED
    assert second.status is BalsamAdapterStatus.STARTED
    assert first.registration is not None
    assert first.registration.job_id == second.registration.job_id == "balsam-job-opaque"
    assert client.register_calls == 1
    assert client.observe_calls == 2
    assert output_paths[0] != output_paths[1]
    assert (tmp_path / "receipts" / "lineage-balsam" / "registration-intent.json").is_file()
    assert (tmp_path / "receipts" / "lineage-balsam" / "registration.json").is_file()


def test_changed_intent_refuses_reuse_before_client_call(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    _adapter(tmp_path, client, batch).start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
    )

    changed_batch, changed_intent = _batch(payload={"science": {"system": "different"}, "target_iterations": 2})
    changed_adapter = _adapter(tmp_path, client, changed_batch)
    with pytest.raises(BalsamBindingError, match="different Balsam intent"):
        changed_adapter.start(
            batch=changed_batch,
            intent=changed_intent,
            phase=RecoveryPhase.INITIAL,
            run_once=lambda *_: _completed(),
        )
    assert client.register_calls == 1


def test_lost_registration_ack_is_durable_park_and_never_resubmitted(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    client.register_exception = BalsamAcknowledgementUnknown("client call lost")
    adapter = _adapter(tmp_path, client, batch)

    first = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
    )
    client.register_exception = None
    second = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
    )

    assert first.status is BalsamAdapterStatus.PARKED
    assert second.status is BalsamAdapterStatus.PARKED
    assert "without resubmitting" in second.reason
    assert client.register_calls == 1
    assert (tmp_path / "receipts" / "lineage-balsam" / "registration-park.json").is_file()


def test_client_exception_after_intent_is_parked_without_blind_resubmit(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    client.register_exception = OSError("connection reset after request")
    adapter = _adapter(tmp_path, client, batch)

    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
    )

    assert result.status is BalsamAdapterStatus.PARKED
    assert client.register_calls == 1


def test_allocation_loss_is_backend_lifecycle_and_e3_remains_outcome_authority(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient(state=BalsamJobState.ALLOCATION_LOST)
    adapter = _adapter(tmp_path, client, batch)
    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
    )

    assert result.status is BalsamAdapterStatus.STARTED
    assert result.observation is not None
    assert result.observation.state is BalsamJobState.ALLOCATION_LOST
    assert result.invocation is not None
    assert result.invocation.outcome.outcome.value == "completed"


def test_explicit_backend_refusal_is_typed_and_does_not_register(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeBalsamClient()
    client.register_exception = BalsamPermanentRefusal("invalid Balsam application")
    with pytest.raises(BalsamPermanentRefusal, match="invalid Balsam application"):
        _adapter(tmp_path, client, batch).start(
            batch=batch,
            intent=intent,
            phase=RecoveryPhase.INITIAL,
            run_once=lambda *_: _completed(),
        )
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
    adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
    )
    path = tmp_path / "receipts" / "lineage-balsam" / "registration.json"
    data = path.read_text()
    path.write_text(data.replace(batch.payload_digest, "c" * 64))
    with pytest.raises(BalsamBindingError, match="receipt payload_digest"):
        adapter.start(
            batch=batch,
            intent=intent,
            phase=RecoveryPhase.INITIAL,
            run_once=lambda *_: _completed(),
        )
