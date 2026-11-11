"""Formal review round 3: E4 observation identity and park durability."""

from __future__ import annotations

import json
from pathlib import Path

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
)
from tpen.execution.balsam import WORKER_PARK_NAME
from tpen.execution.config import payload_digest


class FakeClient:
    def __init__(self, observation=None) -> None:
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
        lineage_id="r3-e4-lineage",
        backend="balsam",
        recovery="checkpoint",
        source_config_digest="a" * 64,
        payload_digest=payload_digest(payload),
        backend_options={"site": "review"},
    )
    return (
        ReadyBatch(
            lineage_id=execution.lineage_id,
            batch_id="r3-e4-batch",
            source_config_digest=execution.source_config_digest,
            payload_digest=execution.payload_digest,
            science_input_digest=intent.science_input_digest,
            target_iterations=2,
            execution=execution,
            items=(ReadyItem("r3-e4-inv", "r3-e4-task", "r3-e4-out", ("worker",), "results/e4"),),
            payload=payload,
        ),
        intent,
    )


def _registered_adapter(tmp_path: Path, observation):
    batch, intent = _batch()
    client = FakeClient(observation)
    adapter = BalsamAdapter(
        batch.execution, client, WorkerStartGateway(tmp_path / "outputs"), tmp_path / "receipts"
    )
    assert adapter.register_lineage(batch) is not None
    return adapter, batch, intent, client


def test_mapping_observation_without_job_id_is_parked(tmp_path: Path) -> None:
    adapter, batch, _intent, _client = _registered_adapter(
        tmp_path, {"state": "completed", "reason": "foreign backend row with no id"}
    )
    observed = adapter.observe_lineage(batch)
    lineage_dir = tmp_path / "receipts" / batch.lineage_id
    assert observed is None
    assert not (lineage_dir / "observation.json").exists()
    assert (lineage_dir / "observation-park.json").is_file()


def test_mapping_observation_with_explicit_matching_id_is_accepted(tmp_path: Path) -> None:
    adapter, batch, _intent, _client = _registered_adapter(
        tmp_path, {"job_id": 42, "state": "completed", "reason": ""}
    )
    observed = adapter.observe_lineage(batch)
    assert observed is not None
    assert observed.job_id == "42"
    assert (tmp_path / "receipts" / batch.lineage_id / "observation.json").is_file()


def test_second_worker_park_preserves_first_reason(tmp_path: Path) -> None:
    adapter, batch, intent, _client = _registered_adapter(
        tmp_path, {"job_id": 42, "state": "active", "reason": ""}
    )

    def run_once(*_args):
        return CoreOutcomeFacts.completed(
            target_iterations=2, next_iteration=2, completed_updates=2
        )

    first = adapter.start(
        batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=run_once, job_id=""
    )
    second = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=run_once,
        job_id="not-the-registered-job",
    )
    assert first.status is BalsamAdapterStatus.PARKED
    assert second.status is BalsamAdapterStatus.PARKED
    assert second.reason != first.reason
    lineage_dir = tmp_path / "receipts" / batch.lineage_id
    park_bodies = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(lineage_dir.glob(f"*{WORKER_PARK_NAME}"))
    ] + [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(lineage_dir.glob("worker-park-*.json"))
    ]
    reasons = {body.get("reason") for body in park_bodies}
    assert first.reason in reasons
