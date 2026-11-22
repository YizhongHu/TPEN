"""T2 proposals: reject admissions that violate the v2 writer contract.

Not executed by the reviewer. On the reviewed head, the unchanged control is
expected to pass and every damaged-ledger refusal is expected to fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpen_executor._codec import canonical_bytes
from tpen_executor.results import SuccessorAdmissionRecord
from tpen_executor.storage import DirectoryStore

from test_b1a_b_publication import _fixture, _publish


@pytest.mark.parametrize(
    "damage", ["none", "group_digest", "group_id", "lineage_id", "v1-downgrade"]
)
def test_review_s6aa_manifest_checks_admission_context(tmp_path: Path, damage: str) -> None:
    """Reopening must enforce the same binding context as admission writing."""

    fixture = _fixture(tmp_path, ("successor",), remote=False)
    store, group = fixture[:2]
    _publish(fixture)
    ledger_paths = list(Path(store.root).rglob("successor-admission.json"))
    assert len(ledger_paths) == 1
    ledger = ledger_paths[0]
    payload = json.loads(ledger.read_bytes())
    if damage == "none":
        admission = DirectoryStore(store.root).manifest(group).attempts[0].successor_admission
        assert admission is not None
        assert admission.to_dict() == payload
        return

    binding = payload["successor_binding"]
    if damage == "group_digest":
        binding["group_digest"] = "f" * 64
        assert binding["group_digest"] != group.identity
    elif damage == "group_id":
        binding["allocation_attempt"]["group_id"]["value"] = "foreign-group"
        assert binding["allocation_attempt"]["group_id"]["value"] != group.group_id.value
    elif damage == "lineage_id":
        binding["lineage_id"] = "foreign-lineage"
        assert binding["lineage_id"] != group.members[0].lineage_id
    else:
        payload["schema"] = "tpen-executor/successor-admission-v1"
        payload.pop("successor_binding")

    # The fault is contextual, not malformed JSON, schema, or canonical bytes.
    encoded = canonical_bytes(payload)
    assert SuccessorAdmissionRecord.decode(encoded).encode() == encoded
    ledger.write_bytes(encoded)
    with pytest.raises(ValueError):
        DirectoryStore(store.root).manifest(group)
