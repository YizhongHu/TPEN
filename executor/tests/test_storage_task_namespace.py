"""Q2 task-directory namespace acceptance probes."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.storage import DirectoryStore


def _group(task_id: str) -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    member = TaskSpec(task_id, task_id, program, digest(task_id), 1, "steps")
    return ReadyGroup("q2", (member,), digest("q2-plan"))


def _allocation() -> ConfiguredAllocation:
    return ConfiguredAllocation(AllocationId("q2-allocation"), BackendJobId("q2-job"), {})


def test_colon_task_id_is_preserved_as_one_namespace_component(tmp_path: Path) -> None:
    task_id = "he-cutover:02_train:seed-000"
    group = _group(task_id)
    fence = DirectoryStore(tmp_path / "store").begin_attempt(group, task_id, _allocation())
    assert Path(fence.output_dir).parent.name == task_id
    assert fence.task_id == task_id


@pytest.mark.parametrize("task_id", [".", "..", "../escape", "/absolute", "a/b", "a\\b", "a\x00b"])
def test_task_namespace_rejects_traversal_and_separator_forms(tmp_path: Path, task_id: str) -> None:
    group = _group(task_id)
    with pytest.raises((TypeError, ValueError)):
        DirectoryStore(tmp_path / "store").begin_attempt(group, task_id, _allocation())
    assert not (tmp_path / "store").exists()
