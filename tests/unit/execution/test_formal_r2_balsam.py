"""Formal review round 2: E4 foreign backend observation identity."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen.execution import (
    BalsamAdapter,
    BalsamBindingError,
    BalsamJobObservation,
    BalsamJobState,
    CoreOutcomeFacts,
    ExecutionConfig,
    ReadyBatch,
    ReadyItem,
    RecoveryPhase,
    WorkerStartGateway,
    capture_science_intent,
)
from tpen.execution.config import payload_digest


class FakeClient:
    def __init__(self, observation) -> None:
        self.observation = observation
        self.register_calls = 0
        self.observe_calls = 0

    def register_job(self, *, lineage_id, batch, backend_options):
        self.register_calls += 1
        return {"id": 42}

    def observe_job(self, job_id):
        self.observe_calls += 1
        return self.observation


def _batch() -> tuple[ReadyBatch, object]:
    payload = {"science": {"system": "hooke"}, "target_iterations": 2}
    intent = capture_science_intent(payload["science"], target_iterations=2)
    execution = ExecutionConfig(
        lineage_id="r2-e4-lineage",
        backend="balsam",
        recovery="checkpoint",
        source_config_digest="a" * 64,
        payload_digest=payload_digest(payload),
        backend_options={"site": "review"},
    )
    return (
        ReadyBatch(
            lineage_id=execution.lineage_id,
            batch_id="r2-e4-batch",
            source_config_digest=execution.source_config_digest,
            payload_digest=execution.payload_digest,
            science_input_digest=intent.science_input_digest,
            target_iterations=2,
            execution=execution,
            items=(ReadyItem("r2-e4-inv", "r2-e4-task", "r2-e4-out", ("worker",), "results/e4"),),
            payload=payload,
        ),
        intent,
    )


def _registered_adapter(tmp_path: Path, observation):
    batch, intent = _batch()
    client = FakeClient(observation)
    adapter = BalsamAdapter(
        batch.execution,
        client,
        WorkerStartGateway(tmp_path / "outputs"),
        tmp_path / "receipts",
    )
    assert adapter.register_lineage(batch) is not None
    return adapter, batch, intent, client


@pytest.mark.parametrize(
    "state",
    (BalsamJobState.ALLOCATION_LOST, BalsamJobState.COMPLETED, BalsamJobState.PERMANENT_FAILURE),
)
def test_foreign_mapping_observation_is_parked_not_attributed(
    tmp_path: Path, state: BalsamJobState
) -> None:
    """RED at E4: mapping observations must bind to registered Job 42."""

    adapter, batch, _intent, _client = _registered_adapter(
        tmp_path,
        {"job_id": 99, "state": state.value, "reason": "foreign backend row"},
    )
    try:
        observed = adapter.observe_lineage(batch)
    except BalsamBindingError:
        observed = None
    assert observed is None
    lineage_dir = tmp_path / "receipts" / batch.lineage_id
    assert not (lineage_dir / "observation.json").exists()
    park = lineage_dir / "observation-park.json"
    assert park.is_file(), "foreign mapping must leave only a parked observation"


@pytest.mark.parametrize("observed_id", (42, "42"), ids=("integer-id", "string-id"))
def test_same_job_mapping_observation_is_accepted(tmp_path: Path, observed_id) -> None:
    """GREEN controls: integer and canonical-string 42 are the same Job."""

    adapter, batch, _intent, _client = _registered_adapter(
        tmp_path, {"job_id": observed_id, "state": "completed", "reason": ""}
    )
    observed = adapter.observe_lineage(batch)
    assert observed is not None
    assert observed.job_id == "42"
    assert (tmp_path / "receipts" / batch.lineage_id / "observation.json").is_file()


def test_typed_foreign_job_observation_is_refused(tmp_path: Path) -> None:
    """GREEN control: typed foreign observations are parked and return None."""

    adapter, batch, _intent, _client = _registered_adapter(
        tmp_path, BalsamJobObservation(99, BalsamJobState.COMPLETED)
    )
    observed = adapter.observe_lineage(batch)
    assert observed is None
    lineage_dir = tmp_path / "receipts" / batch.lineage_id
    assert not (lineage_dir / "observation.json").exists()
    assert (lineage_dir / "observation-park.json").is_file()


def test_worker_start_never_registers_or_observes_backend(tmp_path: Path) -> None:
    """GREEN control: an acknowledged worker start uses no backend calls."""

    adapter, batch, intent, client = _registered_adapter(tmp_path, {"job_id": 42, "state": "completed"})
    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: CoreOutcomeFacts.completed(
            target_iterations=2, next_iteration=2, completed_updates=2
        ),
        job_id=42,
    )
    assert result.status.value == "started"
    assert client.register_calls == 1
    assert client.observe_calls == 0
