"""Round-7 review P7-3: experiment adapter and core decoder validation parity.

Scenario
--------
``experiments/toolkit/execution_v2.py:100-121`` checks only duplicate
``invocation_id`` values and non-empty strings. The core decoder refuses a
duplicate ``fresh_output_id`` (``tpen/execution/batch.py:137-139``), a digest
that is not 64 lowercase hex characters, and a whitespace-only identity. An
experiment can therefore build, encode and decode a v2 batch without error that
the worker's ``decode_ready_batch`` later refuses inside the allocation.

Owning layer and SHA
--------------------
E2 ``a70d324546931ae61234517a73e04096290d4ba6``; also run at CR
``c368d7aeb8c6e566407f2a07df344a850218973b``.

Expected polarity
-----------------
The three ``test_r7_adapter_refuses_what_core_refuses`` parameters are expected
RED at E2 and CR. The core-refusal controls and the valid round trip are
expected GREEN.
"""

from __future__ import annotations

import copy
import json

import pytest

from experiments.toolkit.execution_v2 import encode_ready_batch_v2, ready_batch_v2
from tpen.execution import decode_ready_batch

VARIANTS = ["dup-fresh-output", "non-hex-digest", "whitespace-id"]


def _rows() -> list[dict]:
    return [
        {"invocation_id": "i-1", "logical_task_id": "task-1", "fresh_output_id": "out-1",
         "argv": ["worker"], "result_dir": "r1"},
        {"invocation_id": "i-2", "logical_task_id": "task-2", "fresh_output_id": "out-2",
         "argv": ["worker"], "result_dir": "r2"},
    ]


def _kwargs() -> dict:
    return dict(
        lineage_id="r7-lineage",
        batch_id="r7-batch",
        source_config_digest="a" * 64,
        science_input_digest="b" * 64,
        target_iterations=2,
        backend="direct",
        recovery="checkpoint",
        payload={"x": 1},
        rows=_rows(),
    )


def _malformed_kwargs(variant: str) -> dict:
    kwargs = _kwargs()
    if variant == "dup-fresh-output":
        kwargs["rows"][1]["fresh_output_id"] = kwargs["rows"][0]["fresh_output_id"]
    elif variant == "non-hex-digest":
        kwargs["source_config_digest"] = "z" * 64
    elif variant == "whitespace-id":
        kwargs["lineage_id"] = "   "
    else:  # pragma: no cover - guards the parametrization itself
        raise AssertionError(variant)
    return kwargs


@pytest.mark.parametrize("variant", VARIANTS)
def test_r7_adapter_refuses_what_core_refuses(variant: str) -> None:
    try:
        batch = ready_batch_v2(**_malformed_kwargs(variant))
    except ValueError:
        return
    # The adapter accepted it; record what the core does with the same bytes.
    try:
        decode_ready_batch(encode_ready_batch_v2(batch))
    except ValueError as exc:
        core = f"core refuses: {exc}"
    else:
        core = "core also accepts"
    raise AssertionError(f"adapter accepted {variant}; {core}")


@pytest.mark.parametrize("variant", VARIANTS)
def test_r7_core_refuses_each_malformed_variant(variant: str) -> None:
    # Edit a valid adapter mapping directly so this control does not depend on
    # the adapter's own validation.
    malformed = copy.deepcopy(ready_batch_v2(**_kwargs()))
    if variant == "dup-fresh-output":
        malformed["items"][1]["fresh_output_id"] = malformed["items"][0]["fresh_output_id"]
    elif variant == "non-hex-digest":
        malformed["source_config_digest"] = "z" * 64
        malformed["execution"]["source_config_digest"] = "z" * 64
    else:
        malformed["lineage_id"] = "   "
        malformed["execution"]["lineage_id"] = "   "
    with pytest.raises(ValueError):
        decode_ready_batch(json.dumps(malformed, sort_keys=True).encode("utf-8"))


def test_r7_valid_adapter_batch_decodes_in_core() -> None:
    mapping = ready_batch_v2(**_kwargs())
    decoded = decode_ready_batch(encode_ready_batch_v2(mapping))

    assert decoded.lineage_id == mapping["lineage_id"]
    assert decoded.batch_id == mapping["batch_id"]
    assert tuple(item.invocation_id for item in decoded.items) == ("i-1", "i-2")
    assert decoded.payload_digest == mapping["payload_digest"]
