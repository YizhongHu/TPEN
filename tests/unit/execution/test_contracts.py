"""Source tests for E2 execution configuration, batches, and receipts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tpen.execution import (
    BackendJobReceipt,
    ExecutionConfig,
    ReadyBatch,
    ReadyItem,
    decode_ready_batch,
    encode_ready_batch,
    payload_digest,
    recovery_selection_digest,
    source_config_digest,
    require_lineage_payload,
    validate_receipt_binding,
)


def _batch() -> ReadyBatch:
    payload = {"science": {"system": "hooke", "electrons": 2}, "target_iterations": 10}
    source_digest = "a" * 64
    config = ExecutionConfig(
        lineage_id="lineage-opaque",
        backend="parsl-attach",
        recovery="checkpoint",
        source_config_digest=source_digest,
        payload_digest=payload_digest(payload),
        backend_options={"partition": "test"},
        recovery_options={"mode": "train_resume"},
    )
    return ReadyBatch(
        lineage_id="lineage-opaque",
        batch_id="batch-opaque",
        source_config_digest=source_digest,
        payload_digest=payload_digest(payload),
        science_input_digest="b" * 64,
        target_iterations=10,
        execution=config,
        items=(
            ReadyItem(
                invocation_id="invocation-opaque",
                logical_task_id="logical-opaque",
                fresh_output_id="fresh-output-opaque",
                argv=("launcher", "--run", "opaque"),
                result_dir="results/run-a",
                payload={"seed": 7},
            ),
        ),
        payload=payload,
    )


def test_ready_batch_codec_is_deterministic_and_preserves_opaque_intent() -> None:
    batch = _batch()
    encoded = encode_ready_batch(batch)

    assert encoded == encode_ready_batch(batch)
    restored = decode_ready_batch(encoded)
    assert restored == batch
    assert restored.lineage_id == "lineage-opaque"
    assert restored.items[0].invocation_id == "invocation-opaque"
    assert restored.execution.backend_options["partition"] == "test"
    with pytest.raises(TypeError):
        restored.items[0].payload["seed"] = 8  # type: ignore[index]


def test_reused_lineage_with_different_payload_is_refused() -> None:
    """Mutation control: deleting the digest comparison must make this red."""
    batch = _batch()
    bindings = {batch.lineage_id: batch.payload_digest}

    assert require_lineage_payload(bindings, lineage_id=batch.lineage_id, payload=batch.payload) == batch.payload_digest
    with pytest.raises(ValueError, match="different payload"):
        require_lineage_payload(bindings, lineage_id=batch.lineage_id, payload={"changed": True})


def test_duplicate_identity_is_refused() -> None:
    """Mutation control: deleting duplicate admission must make this red."""
    batch = _batch()
    with pytest.raises(ValueError, match="duplicate invocation_id"):
        replace(
            batch,
            items=(batch.items[0], replace(batch.items[0], logical_task_id="other")),
        )


def test_receipt_binding_refuses_a_mutated_batch_identity() -> None:
    """Mutation control: deleting receipt binding must make this red."""
    batch = _batch()
    receipt = BackendJobReceipt(
        lineage_id=batch.lineage_id,
        batch_id=batch.batch_id,
        invocation_id=batch.items[0].invocation_id,
        backend="parsl-attach",
        job_id="job-opaque",
        ready_batch_digest=batch.identity,
        source_config_digest=batch.source_config_digest,
        payload_digest=batch.payload_digest,
    )
    validate_receipt_binding(receipt, batch)

    mutated = replace(batch, batch_id="different-batch")
    with pytest.raises(ValueError, match="different lineage or batch"):
        validate_receipt_binding(receipt, mutated)


def test_execution_config_round_trip_keeps_recovery_selection_separate() -> None:
    from tpen.execution import RecoveryOutcome, RecoveryPhase, RecoveryDecision

    selection = RecoveryDecision(
        phase=RecoveryPhase.INITIAL,
        outcome=RecoveryOutcome.INITIAL,
        target_iterations=10,
    )
    config = ExecutionConfig(
        lineage_id="lineage-opaque",
        backend="direct",
        recovery="checkpoint",
        source_config_digest="a" * 64,
        payload_digest="b" * 64,
        recovery_selection_digest=recovery_selection_digest(selection),
    )

    assert ExecutionConfig.decode(config.encode()) == config
    assert config.to_dict()["backend"] == "direct"
    assert source_config_digest({"science": {"system": "hooke"}}) != config.payload_digest

