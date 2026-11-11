"""Review r1: E4 registration park/receipt precedence.

Target layer: E4 @ cfaf419b7926ca40bced7d4cf223c4bdb3753d65 (also run at CR head).

Scenario: submitter A wins the registration-marker race and completes an
acknowledged registration; racer B had already durably parked ("another
registration owner is unresolved"). State on disk: valid intent + lock +
receipt + park. ``_ensure_registration`` checks receipt before park, so the
submitter sees success; ``_load_registration`` checks park before receipt, so
every worker start parks forever on a validly acknowledged Job.

Polarity at the E4 head:
- test_worker_starts_on_valid_ack_despite_stale_race_park: EXPECTED RED.
- test_submitter_still_returns_receipt_with_stale_race_park: EXPECTED GREEN
  (pin; with the red test it demonstrates the submitter/worker disagreement).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen.execution import (
    BalsamAdapter,
    BalsamAdapterStatus,
    CoreOutcomeFacts,
    ExecutionConfig,
    ReadyBatch,
    ReadyItem,
    RecoveryPhase,
    WorkerStartGateway,
    capture_science_intent,
    payload_digest,
    source_config_digest,
)
from tpen.execution.balsam import (
    BALSAM_PARK_SCHEMA,
    BALSAM_REFUSAL_SCHEMA,
    REGISTRATION_PARK_NAME,
    REGISTRATION_REFUSAL_NAME,
)
from tpen.execution.balsam import BalsamPermanentRefusal


class FakeClient:
    def __init__(self) -> None:
        self.register_calls = 0
        self.ack = {"job_id": "job-review-r1"}

    def register_job(self, *, lineage_id, batch, backend_options):
        self.register_calls += 1
        return self.ack

    def observe_job(self, job_id):
        raise AssertionError("observe_job must not be called in this test")


def _batch():
    science = {"system": "hooke", "electrons": 2}
    intent = capture_science_intent(science, target_iterations=2)
    payload = {"science": science, "target_iterations": 2}
    source_digest = source_config_digest({})
    digest = payload_digest(payload)
    config = ExecutionConfig(
        lineage_id="lineage-review-park",
        backend="balsam",
        recovery="checkpoint",
        source_config_digest=source_digest,
        payload_digest=digest,
        backend_options={"site": "test-site"},
    )
    batch = ReadyBatch(
        lineage_id="lineage-review-park",
        batch_id="batch-review-park",
        source_config_digest=source_digest,
        payload_digest=digest,
        science_input_digest=intent.science_input_digest,
        target_iterations=2,
        execution=config,
        items=(
            ReadyItem(
                invocation_id="logical-review-park",
                logical_task_id="task-review-park",
                fresh_output_id="output-review-park",
                argv=("worker",),
                result_dir="results/review-park",
            ),
        ),
        payload=payload,
    )
    return batch, intent


def _write_race_loser_park(receipts: Path, batch: ReadyBatch) -> None:
    path = receipts / batch.lineage_id / REGISTRATION_PARK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": BALSAM_PARK_SCHEMA,
                "lineage_id": batch.lineage_id,
                "batch_id": batch.batch_id,
                "ready_batch_digest": batch.identity,
                "reason": "another registration owner is unresolved; park for reconciliation",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _completed():
    return CoreOutcomeFacts.completed(target_iterations=2, next_iteration=2, completed_updates=2)


def test_worker_starts_on_valid_ack_despite_stale_race_park(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeClient()
    adapter = BalsamAdapter(
        batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts"
    )
    receipt = adapter.register_lineage(batch)
    assert receipt is not None
    _write_race_loser_park(tmp_path / "receipts", batch)

    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: _completed(),
        job_id=receipt.job_id,
    )

    assert client.register_calls == 1
    assert result.status is BalsamAdapterStatus.STARTED


def test_submitter_still_returns_receipt_with_stale_race_park(tmp_path: Path) -> None:
    batch, intent = _batch()
    client = FakeClient()
    adapter = BalsamAdapter(
        batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts"
    )
    receipt = adapter.register_lineage(batch)
    assert receipt is not None
    _write_race_loser_park(tmp_path / "receipts", batch)

    again = adapter.register_lineage(batch)

    assert client.register_calls == 1
    assert again is not None and again.job_id == receipt.job_id


def test_refusal_precedence_is_shared_by_submitter_and_worker(tmp_path: Path) -> None:
    """A durable typed refusal outranks a valid acknowledgement everywhere."""

    batch, intent = _batch()
    client = FakeClient()
    adapter = BalsamAdapter(
        batch.execution,
        client,
        WorkerStartGateway(tmp_path / "outputs"),
        tmp_path / "receipts",
    )
    receipt = adapter.register_lineage(batch)
    assert receipt is not None
    refusal_path = tmp_path / "receipts" / batch.lineage_id / REGISTRATION_REFUSAL_NAME
    refusal_path.write_text(
        json.dumps(
            {
                "schema": BALSAM_REFUSAL_SCHEMA,
                "lineage_id": batch.lineage_id,
                "batch_id": batch.batch_id,
                "ready_batch_digest": batch.identity,
                "reason": "operator denied this registration",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(BalsamPermanentRefusal, match="operator denied") as submitter:
        adapter.register_lineage(batch)
    with pytest.raises(BalsamPermanentRefusal, match="operator denied") as worker:
        adapter.start(
            batch=batch,
            intent=intent,
            phase=RecoveryPhase.INITIAL,
            run_once=lambda *_: _completed(),
            job_id=receipt.job_id,
        )
    assert type(submitter.value) is type(worker.value)
