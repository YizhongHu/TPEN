"""Formal review round 3: E4 observation identity floor and park-evidence durability.

Target: E4 @ 7b6b0aeeaa5a8952cbb13b8665de54209f3901a0 and the CR head.

Polarity at those heads (preregistered):
- test_mapping_observation_without_job_id_is_parked: EXPECTED RED.  A mapping
  observation lacking the ``job_id`` key inherits the expected registration id
  (``value.get("job_id", expected_job_id)``), so an id-less foreign backend row
  is accepted and written as the authoritative observation.
- test_mapping_observation_with_explicit_matching_id_is_accepted: EXPECTED
  GREEN (pin; the identity floor must not break the matching case).
- test_second_worker_park_preserves_first_reason: EXPECTED RED.  Worker-park
  records are published with an in-place atomic replace, so a second park with
  a different reason destroys the first park's failure evidence.
"""

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
        batch.execution,
        client,
        WorkerStartGateway(tmp_path / "outputs"),
        tmp_path / "receipts",
    )
    assert adapter.register_lineage(batch) is not None
    return adapter, batch, intent, client


def test_mapping_observation_without_job_id_is_parked(tmp_path: Path) -> None:
    """RED at E4/CR: an id-less mapping row must park, never inherit identity."""

    adapter, batch, _intent, _client = _registered_adapter(
        tmp_path, {"state": "completed", "reason": "foreign backend row with no id"}
    )
    observed = adapter.observe_lineage(batch)
    lineage_dir = tmp_path / "receipts" / batch.lineage_id
    assert observed is None, (
        "an observation mapping with no job_id inherited the registered Job "
        f"identity and was accepted: {observed!r}"
    )
    assert not (lineage_dir / "observation.json").exists(), (
        "an id-less mapping row was written as the authoritative observation"
    )
    assert (lineage_dir / "observation-park.json").is_file()


def test_mapping_observation_with_explicit_matching_id_is_accepted(tmp_path: Path) -> None:
    """GREEN pin: a mapping row that names the registered Job stays accepted."""

    adapter, batch, _intent, _client = _registered_adapter(
        tmp_path, {"job_id": 42, "state": "completed", "reason": ""}
    )
    observed = adapter.observe_lineage(batch)
    assert observed is not None
    assert observed.job_id == "42"
    assert (tmp_path / "receipts" / batch.lineage_id / "observation.json").is_file()


def test_second_worker_park_preserves_first_reason(tmp_path: Path) -> None:
    """RED at E4/CR: a later worker park must not destroy earlier park evidence."""

    adapter, batch, intent, _client = _registered_adapter(
        tmp_path, {"job_id": 42, "state": "active", "reason": ""}
    )

    def run_once(*_args):
        return CoreOutcomeFacts.completed(
            target_iterations=2, next_iteration=2, completed_updates=2
        )

    # First park: invalid launched Job identity.
    first = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=run_once,
        job_id="",
    )
    assert first.status is BalsamAdapterStatus.PARKED
    first_reason = first.reason
    # Second park on the same lineage: mismatched launched Job identity.
    second = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=run_once,
        job_id="not-the-registered-job",
    )
    assert second.status is BalsamAdapterStatus.PARKED
    assert second.reason != first_reason

    lineage_dir = tmp_path / "receipts" / batch.lineage_id
    park_bodies = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(lineage_dir.glob(f"*{WORKER_PARK_NAME}"))
    ] + [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(lineage_dir.glob("worker-park*.json"))
        if path.name != WORKER_PARK_NAME
    ]
    reasons = {body.get("reason") for body in park_bodies}
    assert first_reason in reasons, (
        "the first worker-park reason (failure evidence) was destroyed by the "
        f"second park; surviving reasons={sorted(filter(None, reasons))}"
    )
