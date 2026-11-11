"""Round-7 review P7-1: concurrent first Balsam registration of one lineage.

Scenario
--------
Submitter A wins the registration marker in ``BalsamAdapter._ensure_registration``
(``tpen/execution/balsam.py:769-785``). A concurrent submitter B for the same
lineage arrives before A reaches its park check, writes ``registration-park.json``
and returns ``None``. A then sees that park and also returns ``None`` without
calling ``register_job``: no Job is registered, no acknowledgement was lost, and
every later registration attempt returns ``None``.

Owning layer and SHA
--------------------
E4 ``58d02137050b22e5213ffcc0fecaf09ac484f3b2``; also run at CR
``c368d7aeb8c6e566407f2a07df344a850218973b``.

Expected polarity
-----------------
The two race timings and the permanence test are expected RED at E4 and CR.
The single-owner and late-racer controls are expected GREEN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tpen.execution.balsam as balsam_module
from tpen.execution import (
    BalsamAdapter,
    ExecutionConfig,
    ReadyBatch,
    ReadyItem,
    WorkerStartGateway,
    capture_science_intent,
    payload_digest,
    source_config_digest,
)
from tpen.execution.balsam import REGISTRATION_INTENT_NAME, REGISTRATION_PARK_NAME


class _Client:
    """Counting client; registration acknowledges one opaque Job id."""

    def __init__(self) -> None:
        self.register_calls = 0

    def register_job(self, *, lineage_id, batch, backend_options):
        self.register_calls += 1
        return {"id": 101}

    def observe_job(self, job_id):
        raise AssertionError("observe_job must not be called in this test")


def _batch() -> ReadyBatch:
    science = {"system": "hooke", "electrons": 2}
    intent = capture_science_intent(science, target_iterations=2)
    payload = {"science": science, "target_iterations": 2}
    digest = payload_digest(payload)
    source = source_config_digest({})
    config = ExecutionConfig(
        lineage_id="r7-registration-race",
        backend="balsam",
        recovery="checkpoint",
        source_config_digest=source,
        payload_digest=digest,
        backend_options={"site": "test-site"},
    )
    return ReadyBatch(
        lineage_id=config.lineage_id,
        batch_id="r7-batch",
        source_config_digest=source,
        payload_digest=digest,
        science_input_digest=intent.science_input_digest,
        target_iterations=2,
        execution=config,
        items=(
            ReadyItem(
                invocation_id="r7-invocation",
                logical_task_id="r7-task",
                fresh_output_id="r7-output",
                argv=("worker",),
                result_dir="results/r7",
            ),
        ),
        payload=payload,
    )


def _adapter(root: Path, batch: ReadyBatch, client: _Client) -> BalsamAdapter:
    # Every submitter shares the same receipt root, as on a shared filesystem.
    return BalsamAdapter(
        batch.execution, client, WorkerStartGateway(root / "outputs"), root / "receipts"
    )


def _files(root: Path, batch: ReadyBatch) -> list[str]:
    lineage_dir = root / "receipts" / batch.lineage_id
    return sorted(path.name for path in lineage_dir.iterdir()) if lineage_dir.exists() else []


def _race_on_intent_write(monkeypatch, racer, *, before: bool) -> dict:
    """Run ``racer`` once around the owner's durable intent publication."""

    original = balsam_module._atomic_write_json
    observed: dict = {}

    def wrapped(path: Path, data) -> None:
        if "racer" not in observed and path.name == REGISTRATION_INTENT_NAME:
            if before:
                observed["racer"] = racer()
                original(path, data)
            else:
                original(path, data)
                observed["racer"] = racer()
            return
        original(path, data)

    monkeypatch.setattr(balsam_module, "_atomic_write_json", wrapped)
    return observed


@pytest.mark.parametrize("racer_timing", ["before-intent", "after-intent"])
def test_r7_owner_registers_despite_racing_loser_park(
    tmp_path: Path, monkeypatch, racer_timing: str
) -> None:
    batch = _batch()
    client = _Client()
    owner = _adapter(tmp_path, batch, client)
    racer = _adapter(tmp_path, batch, client)
    observed = _race_on_intent_write(
        monkeypatch, lambda: racer.register_lineage(batch), before=racer_timing == "before-intent"
    )

    result = owner.register_lineage(batch)
    files = _files(tmp_path, batch)

    # The racer losing the marker and parking is the documented loser behaviour.
    assert "racer" in observed, f"racer did not run; files={files!r}"
    assert observed["racer"] is None, f"racer={observed['racer']!r}; files={files!r}"
    # The marker owner must still register exactly one Job.
    assert result is not None and result.job_id == "101", (
        f"owner={result!r}; register_calls={client.register_calls}; files={files!r}"
    )
    assert client.register_calls == 1, (
        f"owner={result!r}; register_calls={client.register_calls}; files={files!r}"
    )


def test_r7_race_park_is_permanent_documentation(tmp_path: Path, monkeypatch) -> None:
    batch = _batch()
    client = _Client()
    owner = _adapter(tmp_path, batch, client)
    racer = _adapter(tmp_path, batch, client)
    _race_on_intent_write(monkeypatch, lambda: racer.register_lineage(batch), before=True)
    first = owner.register_lineage(batch)
    monkeypatch.undo()

    second = owner.register_lineage(batch)
    third = _adapter(tmp_path, batch, client).register_lineage(batch)
    files = _files(tmp_path, batch)

    # After a benign race, a later submitter must be able to obtain the receipt.
    assert second is not None and third is not None, (
        f"first={first!r}; second={second!r}; third={third!r}; "
        f"register_calls={client.register_calls}; files={files!r}"
    )


def test_r7_single_owner_registers_once(tmp_path: Path) -> None:
    batch = _batch()
    client = _Client()
    result = _adapter(tmp_path, batch, client).register_lineage(batch)

    assert result is not None and result.job_id == "101"
    assert client.register_calls == 1
    assert REGISTRATION_PARK_NAME not in _files(tmp_path, batch)


def test_r7_late_racer_after_ack_reuses_receipt(tmp_path: Path) -> None:
    batch = _batch()
    client = _Client()
    first = _adapter(tmp_path, batch, client).register_lineage(batch)
    second = _adapter(tmp_path, batch, client).register_lineage(batch)
    files = _files(tmp_path, batch)

    assert first is not None and second is not None and second.job_id == first.job_id
    assert client.register_calls == 1
    assert REGISTRATION_PARK_NAME not in files, json.dumps(files)
