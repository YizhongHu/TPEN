"""Review r1: E2 wire decode strictness and identity stability.

Target layer: E2 @ 2c5241d4 (also run at CR head).

Polarity at the E2 head:
- test_wire_with_unknown_field_is_refused: EXPECTED RED (low severity:
  lenient decode silently drops unknown keys, so two different wire byte
  streams map to one batch identity).
- test_roundtrip_identity_is_stable_under_key_order: EXPECTED GREEN (pin).
"""

from __future__ import annotations

import json

import pytest

from experiments.toolkit.execution_v2 import encode_ready_batch_v2, ready_batch_v2
from tpen.execution import decode_ready_batch_wire


def _adapter_bytes() -> bytes:
    return encode_ready_batch_v2(
        ready_batch_v2(
            lineage_id="lineage-review-wire",
            batch_id="batch-review-wire",
            source_config_digest="a" * 64,
            science_input_digest="b" * 64,
            target_iterations=10,
            backend="direct",
            recovery="none",
            payload={"science": {"system": "hooke"}},
            rows=(
                {
                    "invocation_id": "invocation-review-wire",
                    "logical_task_id": "logical-review-wire",
                    "fresh_output_id": "fresh-review-wire",
                    "argv": ("launcher",),
                    "result_dir": "results/review-wire",
                },
            ),
        )
    )


def test_wire_with_unknown_field_is_refused() -> None:
    data = json.loads(_adapter_bytes())
    data["unexpected_review_field"] = 1
    tampered = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises((ValueError, TypeError)):
        decode_ready_batch_wire(tampered)


def test_roundtrip_identity_is_stable_under_key_order() -> None:
    wire = _adapter_bytes()
    batch = decode_ready_batch_wire(wire)
    reordered = dict(reversed(list(json.loads(wire).items())))
    batch_again = decode_ready_batch_wire(json.dumps(reordered).encode())
    assert batch_again.identity == batch.identity
    assert batch_again == batch
