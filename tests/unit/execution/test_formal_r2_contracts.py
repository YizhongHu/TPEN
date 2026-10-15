"""Formal review round 2: E2 receipt identity and backend bindings.

Target: PR530 ``06c5e538`` and the CR head.  The two-row fixture makes the
cross-pair mutation definitive: invocation A paired with output B is not a
valid binding, independently of any filesystem-path convention.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tpen.execution import (
    BackendJobReceipt,
    ExecutionConfig,
    FreshOutputReceipt,
    InvocationReceipt,
    ReadyBatch,
    ReadyItem,
    ResultReceipt,
    validate_receipt_binding,
)
from tpen.execution.config import payload_digest


def _batch() -> ReadyBatch:
    payload = {"science": {"system": "hooke"}, "target_iterations": 2}
    execution = ExecutionConfig(
        lineage_id="r2-e2-lineage",
        backend="direct",
        recovery="none",
        source_config_digest="a" * 64,
        payload_digest=payload_digest(payload),
    )
    return ReadyBatch(
        lineage_id=execution.lineage_id,
        batch_id="r2-e2-batch",
        source_config_digest=execution.source_config_digest,
        payload_digest=execution.payload_digest,
        science_input_digest="b" * 64,
        target_iterations=2,
        execution=execution,
        items=(
            ReadyItem("invocation-a", "task-a", "output-a", ("worker",), "results/a"),
            ReadyItem("invocation-b", "task-b", "output-b", ("worker",), "results/b"),
        ),
        payload=payload,
    )


def _receipt(kind: str, batch: ReadyBatch):
    common = dict(
        lineage_id=batch.lineage_id,
        batch_id=batch.batch_id,
        invocation_id="invocation-a",
    )
    if kind == "invocation":
        return InvocationReceipt(
            **common,
            job_id="job-a",
            fresh_output_id="output-a",
            ready_batch_digest=batch.identity,
            status="started",
        )
    if kind == "fresh-output":
        return FreshOutputReceipt(
            **common,
            fresh_output_id="output-a",
            output_path="results/a",
            payload_digest=batch.payload_digest,
        )
    if kind == "result":
        return ResultReceipt(
            **common,
            fresh_output_id="output-a",
            result_receipt_id="result-a",
            payload_digest=batch.payload_digest,
            result_digest="c" * 64,
            status="completed",
        )
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ("invocation", "fresh-output", "result"))
def test_crossed_invocation_and_output_pair_is_refused(kind: str) -> None:
    """RED at E2: A's invocation cannot be paired with B's output identity."""

    batch = _batch()
    receipt = replace(_receipt(kind, batch), fresh_output_id="output-b")
    if kind == "fresh-output":
        receipt = replace(receipt, output_path="results/b")
    with pytest.raises(ValueError):
        validate_receipt_binding(receipt, batch)


@pytest.mark.parametrize("kind", ("invocation", "fresh-output", "result"))
def test_matching_invocation_and_output_pair_is_accepted(kind: str) -> None:
    """GREEN control: each exact A/A receipt binding is accepted."""

    batch = _batch()
    validate_receipt_binding(_receipt(kind, batch), batch)


def test_backend_job_receipt_with_different_backend_is_refused() -> None:
    """RED at E2: a backend receipt cannot claim a different backend."""

    batch = _batch()
    receipt = BackendJobReceipt(
        lineage_id=batch.lineage_id,
        batch_id=batch.batch_id,
        invocation_id="invocation-a",
        backend="balsam",
        job_id="job-a",
        ready_batch_digest=batch.identity,
        source_config_digest=batch.source_config_digest,
        payload_digest=batch.payload_digest,
    )
    with pytest.raises(ValueError):
        validate_receipt_binding(receipt, batch)


def test_backend_job_receipt_with_matching_backend_is_accepted() -> None:
    """GREEN control: a backend receipt naming the selected backend is valid."""

    batch = _batch()
    receipt = BackendJobReceipt(
        lineage_id=batch.lineage_id,
        batch_id=batch.batch_id,
        invocation_id="invocation-a",
        backend="direct",
        job_id="job-a",
        ready_batch_digest=batch.identity,
        source_config_digest=batch.source_config_digest,
        payload_digest=batch.payload_digest,
    )
    validate_receipt_binding(receipt, batch)
