"""Storage-layer contract checks and mutation anchors for B1b2-L."""

from pathlib import Path

import pytest

from tpen_executor.allocations import ControllerGenerationV1, LaunchIntentV1
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, GroupId, InvocationId


MUTANTS = (
    {"name": "monotonic-check-off", "anchor": "record.generation != expected", "replacement": "False"},
    {"name": "predecessor-check-off", "anchor": "record.predecessor_record_digest != predecessor", "replacement": "False"},
    {"name": "nonce-check-off", "anchor": "records[-1].claimant_nonce != nonce", "replacement": "False"},
    {"name": "replay-byte-compare-off", "anchor": "record.encode() == current.encode()", "replacement": "True"},
    {"name": "intent-exclusivity-off", "anchor": "            if os.path.lexists(path):\n                raise LaunchIntentRejected", "replacement": "            if False:\n                raise LaunchIntentRejected"},
    {"name": "intent-stale-check-off", "anchor": "self.require_current_controller(controller_record.group_id, controller_record.generation, controller_record.claimant_nonce)", "replacement": "None"},
)


def test_mutant_anchors_are_unique_and_compile_when_applied():
    source = Path(__file__).parents[1].joinpath("src/tpen_executor/storage.py").read_text()
    anchors = [row["anchor"] for row in MUTANTS]
    assert len(anchors) == len(set(anchors)) >= 6
    for row in MUTANTS:
        assert source.count(row["anchor"]) == 1
        compile(source.replace(row["anchor"], row["replacement"], 1), "storage-mutant.py", "exec")


def test_controller_generation_wire_is_canonical_and_strict():
    record = ControllerGenerationV1(
        GroupId("group"), 1, AllocationId("allocation"), BackendJobId("job"),
        "0" * 64, "nonce", None,
    )
    assert ControllerGenerationV1.decode(record.encode()) == record
    with pytest.raises(ValueError):
        ControllerGenerationV1.decode(record.to_dict() | {"unexpected": True})


def test_launch_intent_wire_is_canonical_and_strict():
    record = LaunchIntentV1(
        GroupId("group"), "task", AttemptId("attempt"),
        InvocationId("invocation"), 1, "nonce",
        AllocationId("allocation"), BackendJobId("job"),
    )
    assert LaunchIntentV1.decode(record.encode()) == record
    with pytest.raises(ValueError):
        LaunchIntentV1.decode(record.to_dict() | {"unexpected": True})
