"""P10-H attempt-fence checks for canonical run-directory identity.

| test (file:line) | mutant | why |
| --- | --- | --- |
| T1 (`executor/tests/test_review_p10h_fence_realpath.py:58`) | M1 | A symlinked store root must compare by resolved identity, not spelling. |
| T2 (`executor/tests/test_review_p10h_fence_realpath.py:75`) | M2 | An evidence symlink escaping the attempt must not pass loose containment. |
| T3 (`executor/tests/test_review_p10h_fence_realpath.py:86`) | M3 | A sibling attempt's core directory must fail exact equality. |
| T4 (`executor/tests/test_review_p10h_fence_realpath.py:95`) | M4 | A missing evidence directory must fail strict resolution. |
"""

from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.runners import CoreRunEvidence
from tpen_executor.storage import AttemptFence, DirectoryStore, FenceRejected


def _group() -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    member = TaskSpec("task-a", "task-a", program, digest("task-a"), 1, "steps")
    return ReadyGroup("group-1", (member,), digest("group-1-plan"))


def _allocation() -> ConfiguredAllocation:
    return ConfiguredAllocation(AllocationId("allocation-1"), BackendJobId("job-1"), {})


def _evidence(run_dir: str) -> CoreRunEvidence:
    return CoreRunEvidence.from_mapping(
        {
            "schema": "tpen.run-evidence/v1",
            "run_dir": run_dir,
            "verdict": "completed",
            "run_status": "completed",
            "checkpoint": {
                "checkpoint_dir": f"{run_dir}/checkpoint-1",
                "schema_version": 1,
                "next_iteration": 1,
                "completed_updates": 1,
            },
            "refusal": None,
        }
    )


def _finished_attempt(tmp_path: Path) -> tuple[DirectoryStore, AttemptFence]:
    store = DirectoryStore(tmp_path / "store")
    fence = store.begin_attempt(_group(), "task-a", _allocation(), "start")
    (Path(fence.output_dir) / "core").mkdir()
    store.finish_attempt(fence, "completed", "finish")
    return store, fence


def test_t1_symlinked_store_root_is_accepted_and_kills_m1(tmp_path: Path) -> None:
    real_root = tmp_path / "real-eagle"
    real_root.mkdir()
    aliased_root = tmp_path / "eagle"
    aliased_root.symlink_to(real_root, target_is_directory=True)

    store = DirectoryStore(aliased_root / "store")
    fence = store.begin_attempt(_group(), "task-a", _allocation(), "start")
    (Path(fence.output_dir) / "core").mkdir()
    store.finish_attempt(fence, "completed", "finish")
    reported_run_dir = Path(fence.output_dir).resolve(strict=True) / "core"

    segment = store.accept_segment(fence, _evidence(str(reported_run_dir)))

    assert segment.task_id == "task-a"


def test_t2_escaping_evidence_symlink_is_rejected_and_kills_m2(tmp_path: Path) -> None:
    store, fence = _finished_attempt(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped = Path(fence.output_dir) / "core-escape"
    escaped.symlink_to(outside, target_is_directory=True)

    with pytest.raises(FenceRejected, match="core evidence run_dir crosses the attempt fence"):
        store.accept_segment(fence, _evidence(str(escaped)))


def test_t3_sibling_attempt_core_is_rejected_and_kills_m3(tmp_path: Path) -> None:
    store, fence = _finished_attempt(tmp_path)
    sibling_core = Path(fence.output_dir).parent / "sibling-attempt" / "core"
    sibling_core.mkdir(parents=True)

    with pytest.raises(FenceRejected, match="core evidence run_dir crosses the attempt fence"):
        store.accept_segment(fence, _evidence(str(sibling_core)))


def test_t4_missing_evidence_run_dir_is_rejected_and_kills_m4(tmp_path: Path) -> None:
    store, fence = _finished_attempt(tmp_path)
    missing = Path(fence.output_dir) / "missing-core"

    with pytest.raises(FenceRejected, match="core evidence run_dir crosses the attempt fence"):
        store.accept_segment(fence, _evidence(str(missing)))
