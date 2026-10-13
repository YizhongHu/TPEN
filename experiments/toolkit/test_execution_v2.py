"""Torch-free tests for the experiment-side v2 data adapter."""

from __future__ import annotations

import inspect

import pytest

from experiments.toolkit import execution_v2


def _batch() -> dict[str, object]:
    return execution_v2.ready_batch_v2(
        lineage_id="lineage-opaque",
        batch_id="batch-opaque",
        source_config_digest="a" * 64,
        payload={"science": {"system": "hooke"}},
        rows=(
            {"invocation_id": "invocation-opaque", "logical_task_id": "logical-opaque"},
        ),
    )


def test_v2_adapter_round_trip_preserves_data() -> None:
    batch = _batch()
    assert execution_v2.decode_ready_batch_v2(execution_v2.encode_ready_batch_v2(batch)) == batch


def test_v2_adapter_refuses_altered_payload_digest() -> None:
    batch = _batch()
    batch["payload"] = {"science": {"system": "mutated"}}
    with pytest.raises(ValueError, match="payload digest"):
        execution_v2.encode_ready_batch_v2(batch)


def test_v2_adapter_refuses_duplicate_invocation_identity() -> None:
    with pytest.raises(ValueError, match="duplicate invocation_id"):
        execution_v2.ready_batch_v2(
            lineage_id="lineage-opaque",
            batch_id="batch-opaque",
            source_config_digest="a" * 64,
            payload={},
            rows=({"invocation_id": "same"}, {"invocation_id": "same"}),
        )


def test_experiment_adapter_has_no_core_import_boundary_crossing() -> None:
    source = inspect.getsource(execution_v2)
    assert "import tpen" not in source
    assert "from tpen" not in source

