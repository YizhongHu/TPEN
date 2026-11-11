"""R9-4: adapter/core empty-row validation must agree at the boundary."""

from __future__ import annotations

import pytest

from experiments.toolkit.execution_v2 import encode_ready_batch_v2, ready_batch_v2
from tpen.execution import decode_ready_batch


def _kwargs(rows):
    return dict(
        lineage_id="r9-lineage", batch_id="r9-batch", source_config_digest="a" * 64,
        science_input_digest="b" * 64, target_iterations=2, backend="direct",
        recovery="checkpoint", payload={"x": 1}, rows=rows,
    )


def test_r9_adapter_refuses_empty_rows() -> None:
    # GREEN after the E2 adapter fix.
    with pytest.raises(ValueError):
        ready_batch_v2(**_kwargs([]))


def test_r9_one_row_batch_round_trips_adapter_and_core() -> None:
    # GREEN control.
    row = {"invocation_id": "i", "logical_task_id": "t", "fresh_output_id": "o", "argv": ["worker"], "result_dir": "r"}
    decoded = decode_ready_batch(encode_ready_batch_v2(ready_batch_v2(**_kwargs([row]))))
    assert len(decoded.items) == 1
