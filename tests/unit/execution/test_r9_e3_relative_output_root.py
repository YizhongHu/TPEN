"""R9-2: E3 must resolve a relative worker output root consistently."""

from __future__ import annotations

import json
from pathlib import Path

from tpen.execution import (
    ExecutionConfig,
    ReadyBatch,
    ReadyItem,
    capture_science_intent,
    payload_digest,
    source_config_digest,
)
from tpen.artifacts import ArtifactManager
from tpen.execution import InvocationOutcome, RecoveryPhase, WorkerStartGateway


def test_r9_relative_output_root_namespace_matches_tpen_run_root(tmp_path: Path, monkeypatch) -> None:
    # PREDICTED: RED at CR
    batch, intent = _batch()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    captured: list[Path] = []

    def capture(_decision, output_path):
        captured.append(output_path)
        return None

    result = WorkerStartGateway("rel-outputs").start(
        batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=capture
    )
    # A legacy ``None`` run result is parked; normalize in case the receipt
    # stores the enum value as a string.
    assert InvocationOutcome(result.outcome.outcome) is InvocationOutcome.PARKED
    assert captured
    assert ArtifactManager(str(captured[0]), "exp", "sector", "rid").root == captured[0].absolute()


def _batch():
    science = {"system": "hooke", "electrons": 2}
    source = {"model": {"_target_": "tests.helpers.NeverConstructed"}}
    intent = capture_science_intent(science, target_iterations=3, source_config=source)
    payload = {"science": science, "target_iterations": 3}
    source_digest = source_config_digest(source)
    payload_digest_value = payload_digest(payload)
    config = ExecutionConfig(
        lineage_id="lineage-e3", backend="direct", recovery="checkpoint",
        source_config_digest=source_digest, payload_digest=payload_digest_value,
    )
    return ReadyBatch(
        lineage_id="lineage-e3", batch_id="batch-e3", source_config_digest=source_digest,
        payload_digest=payload_digest_value, science_input_digest=intent.science_input_digest,
        target_iterations=3, execution=config,
        items=(ReadyItem(invocation_id="logical-invocation-e3", logical_task_id="task-e3",
                         fresh_output_id="planned-output-e3", argv=("tpen-worker",), result_dir="results/e3"),),
        payload=payload,
    ), intent


def test_r9_relative_output_root_start_receipt_is_absolute(tmp_path: Path, monkeypatch) -> None:
    # PREDICTED: RED at CR
    batch, intent = _batch()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    result = WorkerStartGateway("rel-outputs").start(
        batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=lambda *_: None
    )
    receipt = json.loads((result.output_path.parent / "start.json").read_text(encoding="utf-8"))
    assert Path(receipt["output_path"]).is_absolute()


def test_r9_absolute_output_root_namespace_matches_tpen_run_root(tmp_path: Path) -> None:
    # PREDICTED: GREEN (control)
    batch, intent = _batch()
    root = tmp_path / "abs"
    captured: list[Path] = []

    def capture(_decision, output_path):
        captured.append(output_path)
        return None

    WorkerStartGateway(root).start(
        batch=batch, intent=intent, phase=RecoveryPhase.INITIAL, run_once=capture
    )
    assert ArtifactManager(str(captured[0]), "exp", "sector", "rid").root == captured[0].absolute()
