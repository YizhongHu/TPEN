"""P10-H attempt-fence checks for canonical run-directory identity.

| test (file:line) | mutant | why |
| --- | --- | --- |
| T1 (`executor/tests/test_review_p10h_fence_realpath.py:58`) | M1 | A symlinked store root must compare by resolved identity, not spelling. |
| T2 (`executor/tests/test_review_p10h_fence_realpath.py:75`) | M2 | An evidence symlink escaping the attempt must not pass loose containment. |
| T3 (`executor/tests/test_review_p10h_fence_realpath.py:86`) | M3 | A sibling attempt's core directory must fail exact equality. |
| T4 (`executor/tests/test_review_p10h_fence_realpath.py:95`) | M4 | A missing evidence directory must use strict resolution, not only a later directory check. |
| T5 (`executor/tests/test_review_p10h_fence_realpath.py:109`) | M5 | A store-side core symlink must be rejected before canonical comparison. |
"""

from pathlib import Path

import pytest
import tpen_executor.storage as storage

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.runners import CoreRunEvidence
from tpen_executor.storage import AttemptFence, DirectoryStore, FenceRejected

CONFIG_IDENTITY = {"schema": "tpen-config-identity-v1", "sha256": "a" * 64}


def _group() -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    member = TaskSpec("task-a", "task-a", program, digest("task-a"), 1, "steps")
    return ReadyGroup("group-1", (member,), digest("group-1-plan"))


def _allocation() -> ConfiguredAllocation:
    return ConfiguredAllocation(AllocationId("allocation-1"), BackendJobId("job-1"), {})


def _evidence(run_dir: str, *, identity: dict[str, str] | None = None) -> CoreRunEvidence:
    data = {
        "schema": "tpen.run-evidence/v2" if identity is not None else "tpen.run-evidence/v1",
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
    if identity is not None:
        data["resolved_config_identity"] = identity
    return CoreRunEvidence.from_mapping(data)


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

    segment = store.accept_segment(fence, _evidence(str(reported_run_dir), identity=CONFIG_IDENTITY))

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


def test_t4_missing_evidence_run_dir_is_rejected_and_kills_m4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, fence = _finished_attempt(tmp_path)
    missing = Path(fence.output_dir) / "missing-core"
    calls: list[bool | None] = []
    original_realpath = storage.os.path.realpath

    def strict_realpath(path, *args, **kwargs):
        calls.append(kwargs.get("strict"))
        return original_realpath(path, *args, **kwargs)

    monkeypatch.setattr(storage.os.path, "realpath", strict_realpath)

    with pytest.raises(FenceRejected, match="core evidence run_dir crosses the attempt fence"):
        store.accept_segment(fence, _evidence(str(missing)))
    assert calls and all(strict is True for strict in calls)


@pytest.mark.parametrize("target_kind", ["outside", "sibling"], ids=["outside", "another-attempt"])
def test_t5_store_side_core_symlink_is_rejected(tmp_path: Path, target_kind: str) -> None:
    store, fence = _finished_attempt(tmp_path)
    if target_kind == "outside":
        target = tmp_path / "outside" / "core"
    else:
        target = Path(fence.output_dir).parent / "another-attempt" / "core"
    target.mkdir(parents=True)
    store_core = Path(fence.output_dir) / "core"
    store_core.rmdir()
    store_core.symlink_to(target, target_is_directory=True)

    with pytest.raises(FenceRejected, match="core evidence run_dir crosses the attempt fence"):
        store.accept_segment(fence, _evidence(str(target)))
