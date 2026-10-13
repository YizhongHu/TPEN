"""Review r1: classification of deterministic selection-time refusals.

Target layer: E3 @ 7fe26b40 (also run at CR head 63548ca8).

Lane contract: "Incompatible restore or deterministic config/application
refusal: terminal failure for this submitted intent; never convert it to
unlimited retry"; E3 acceptance: "deterministic config/incompatible restore
failures are terminal rather than retries". At the head,
``classify_core_outcome`` maps EVERY E1 selection refusal (including
deterministic INCOMPATIBLE/UNSUPPORTED/MODEL_ONLY) to PARKED, while the same
incompatibility raised inside tpen.checkpoint at restore time is TERMINAL.

All tests are EXPECTED GREEN at E3 and CR; the matrix and receipt test pin
the typed terminal-versus-park policy and durable refusal_code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen.execution import (
    CheckpointEvidence,
    ExecutionConfig,
    InvocationOutcome,
    RecoveryDecision,
    RecoveryOutcome,
    ReadyBatch,
    ReadyItem,
    RecoveryPhase,
    RecoveryRefusal,
    RefusalCode,
    WorkerOutcomeReceipt,
    WorkerStartGateway,
    capture_science_intent,
    classify_core_outcome,
    payload_digest,
)


def _batch():
    science = {"system": "hooke", "electrons": 2}
    intent = capture_science_intent(science, target_iterations=3)
    payload = {"science": science, "target_iterations": 3}
    source_digest = "a" * 64
    digest = payload_digest(payload)
    config = ExecutionConfig(
        lineage_id="lineage-review-refusal",
        backend="direct",
        recovery="checkpoint",
        source_config_digest=source_digest,
        payload_digest=digest,
    )
    batch = ReadyBatch(
        lineage_id="lineage-review-refusal",
        batch_id="batch-review-refusal",
        source_config_digest=source_digest,
        payload_digest=digest,
        science_input_digest=intent.science_input_digest,
        target_iterations=3,
        execution=config,
        items=(
            ReadyItem(
                invocation_id="logical-review-refusal",
                logical_task_id="task-review-refusal",
                fresh_output_id="output-review-refusal",
                argv=("worker",),
                result_dir="results/review-refusal",
            ),
        ),
        payload=payload,
    )
    return batch, intent


def test_selection_time_incompatible_binding_is_terminal(tmp_path: Path) -> None:
    batch, intent = _batch()
    evidence = CheckpointEvidence(
        ref=None, science_input_digest="c" * 64, target_iterations=3
    )
    called = []

    result = WorkerStartGateway(tmp_path).start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.RECOVERY,
        run_once=lambda *_: called.append(True),
        evidence=evidence,
    )

    assert called == []
    assert result.outcome.outcome is InvocationOutcome.TERMINAL


def test_selection_refusal_reason_is_durable(tmp_path: Path) -> None:
    batch, intent = _batch()
    evidence = CheckpointEvidence(
        ref=None, science_input_digest="c" * 64, target_iterations=3
    )

    result = WorkerStartGateway(tmp_path).start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.RECOVERY,
        run_once=lambda *_: None,
        evidence=evidence,
    )

    assert result.outcome.outcome in (InvocationOutcome.PARKED, InvocationOutcome.TERMINAL)
    assert "different immutable science input" in result.outcome.reason
    on_disk = json.loads((result.output_path.parent / "outcome.json").read_text())
    assert "different immutable science input" in on_disk["reason"]
    start_on_disk = json.loads((result.output_path.parent / "start.json").read_text())
    assert start_on_disk["recovery_outcome"] == "refused"


@pytest.mark.parametrize(
    ("code", "expected"),
    (
        (RefusalCode.INCOMPATIBLE, InvocationOutcome.TERMINAL),
        (RefusalCode.UNSUPPORTED, InvocationOutcome.TERMINAL),
        (RefusalCode.MODEL_ONLY, InvocationOutcome.TERMINAL),
        (RefusalCode.INCOMPLETE, InvocationOutcome.PARKED),
        (RefusalCode.NO_CHECKPOINT, InvocationOutcome.PARKED),
    ),
    ids=(
        "incompatible-terminal",
        "unsupported-terminal",
        "model-only-terminal",
        "incomplete-parked",
        "no-checkpoint-parked",
    ),
)
def test_all_refusal_codes_have_explicit_outcomes(
    code: RefusalCode, expected: InvocationOutcome
) -> None:
    decision = RecoveryDecision(
        phase=RecoveryPhase.RECOVERY,
        outcome=RecoveryOutcome.REFUSED,
        target_iterations=3,
        refusal=RecoveryRefusal(code, f"{code.value} refusal"),
    )
    outcome, _reason = classify_core_outcome(None, decision=decision)
    assert outcome is expected


def test_incompatible_refusal_code_survives_receipt_and_disk(tmp_path: Path) -> None:
    batch, intent = _batch()
    evidence = CheckpointEvidence(
        ref=None, science_input_digest="c" * 64, target_iterations=3
    )
    result = WorkerStartGateway(tmp_path).start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.RECOVERY,
        run_once=lambda *_: None,
        evidence=evidence,
    )
    assert result.outcome.refusal_code == RefusalCode.INCOMPATIBLE.value
    round_tripped = WorkerOutcomeReceipt.from_mapping(result.outcome.to_dict())
    assert round_tripped.refusal_code == RefusalCode.INCOMPATIBLE.value
    on_disk = json.loads((result.output_path.parent / "outcome.json").read_text())
    assert on_disk["refusal_code"] == RefusalCode.INCOMPATIBLE.value
