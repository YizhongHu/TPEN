"""Ported R4 controls for E2 wire-field typing."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("field", ("logical_task_id", "fresh_output_id", "result_dir"))
def test_v2_row_fields_are_individually_strings(field: str) -> None:
    from experiments.toolkit.execution_v2 import ready_batch_v2

    row = {
        "invocation_id": "invocation",
        "logical_task_id": "logical",
        "fresh_output_id": "output",
        "argv": ("worker",),
        "result_dir": "results",
    }
    row[field] = ["wrong-type"]
    with pytest.raises(ValueError, match=field):
        ready_batch_v2(
            lineage_id="lineage",
            batch_id="batch",
            source_config_digest="0" * 64,
            science_input_digest="1" * 64,
            target_iterations=2,
            backend="direct",
            recovery="none",
            payload={"science": {}},
            rows=[row],
        )


def test_v2_row_with_all_string_fields_is_accepted() -> None:
    from experiments.toolkit.execution_v2 import ready_batch_v2

    batch = ready_batch_v2(
        lineage_id="lineage",
        batch_id="batch",
        source_config_digest="0" * 64,
        science_input_digest="1" * 64,
        target_iterations=2,
        backend="direct",
        recovery="none",
        payload={"science": {}},
        rows=[
            {
                "invocation_id": "invocation",
                "logical_task_id": "logical",
                "fresh_output_id": "output",
                "argv": ("worker",),
                "result_dir": "results",
            }
        ],
    )
    assert batch["items"][0]["logical_task_id"] == "logical"
