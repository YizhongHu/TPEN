"""P8 accepted-segment and cross-project manifest contract tests."""

from __future__ import annotations

from pathlib import Path
import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.results import AcceptedSegment, ManifestAttempt, OutcomeReceipt, RunManifest, StartReceipt
from tpen_executor.runners import CoreRunEvidence
from tpen_executor.storage import DirectoryStore, FenceRejected
from helpers.p8_manifest_fixture import cross_project_manifest


GROUP_DIGEST = "76a21ab64e2533a35421b67a8c16c832967fcc6cadf730c9fdfd505840389ebf"


def _member(task_id: str, attempt_id: str, status: str = "completed") -> ManifestAttempt:
    attempt = AttemptId(attempt_id)
    output_dir = f"/runs/group-1/{task_id}/{attempt_id}"
    start = StartReceipt(task_id, attempt, AllocationId("allocation-1"), BackendJobId("job-1"), GROUP_DIGEST, "2026-11-05T00:00:00Z", output_dir)
    outcome = OutcomeReceipt(task_id, attempt, GROUP_DIGEST, "2026-11-05T00:01:00Z", status, output_dir)
    return ManifestAttempt(task_id, attempt, start, outcome, output_dir)


def _segment(task_id: str, attempt_id: str, predecessor: str | None, step: int = 1) -> AcceptedSegment:
    return AcceptedSegment(task_id, AttemptId(attempt_id), predecessor, f"/runs/group-1/{task_id}/{attempt_id}/checkpoint-{step}", step, step)


def test_p8_manifest_v2_round_trips_and_v1_decodes_read_only() -> None:
    manifest = cross_project_manifest()
    assert RunManifest.decode(manifest.encode()) == manifest
    old = dict(manifest.to_dict())
    old["schema"] = RunManifest.V1_SCHEMA
    old.pop("segments")
    decoded = RunManifest.decode(old)
    assert decoded.segments == ()
    assert decoded.encode() == RunManifest("group-1", GROUP_DIGEST, manifest.attempts, ()).encode()


def test_p8_records_are_frozen_hashable_and_reject_scalar_sequences() -> None:
    segment = _segment("task-a", "attempt-1", None)
    assert hash(segment) == hash(_segment("task-a", "attempt-1", None))
    with pytest.raises(AttributeError):
        segment.checkpoint_dir = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        RunManifest("group-1", GROUP_DIGEST, "not-a-sequence")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "segments, message",
    [
        ((_segment("task-a", "attempt-1", "wrong"),), "first"),
        ((_segment("task-a", "attempt-1", None), _segment("task-a", "attempt-2", "/wrong", 2)), "predecessor"),
        ((_segment("task-a", "attempt-1", None, 1), _segment("task-a", "attempt-2", "/runs/group-1/task-a/attempt-1/checkpoint-1", 1)), "increasing"),
        ((_segment("task-a", "attempt-1", None), _segment("task-a", "attempt-1", None, 2)), "duplicate"),
    ],
    ids=["wrong-first-predecessor", "wrong-predecessor", "non-increasing-coordinate", "duplicate-attempt"],
)
def test_p8_chain_rejections_are_construction_time(segments: tuple[AcceptedSegment, ...], message: str) -> None:
    attempts = tuple(_member("task-a", f"attempt-{index}") for index in (1, 2))
    with pytest.raises((ValueError, TypeError), match=message):
        RunManifest("group-1", GROUP_DIGEST, attempts, segments)


def test_p8_segment_on_failed_attempt_is_rejected() -> None:
    with pytest.raises(ValueError, match="completed"):
        RunManifest("group-1", GROUP_DIGEST, (_member("task-a", "attempt-1", "failed"),), (_segment("task-a", "attempt-1", None),))


def _group() -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    return ReadyGroup("group-1", (TaskSpec("task-a", "task-a", program, digest("task-a"), 1, "steps"),), digest({"group": 1}))


def _evidence(run_dir: str, *, verdict: str = "completed", checkpoint_dir: str | None = None, step: int = 1) -> CoreRunEvidence:
    return CoreRunEvidence.from_mapping(
        {
            "schema": "tpen.run-evidence/v1",
            "run_dir": run_dir,
            "verdict": verdict,
            "run_status": verdict,
            "checkpoint": {
                "checkpoint_dir": checkpoint_dir or f"{run_dir}/checkpoint-{step}",
                "schema_version": "v1",
                "next_iteration": step,
                "completed_updates": step,
            },
            "refusal": None,
        }
    )


def test_p8_store_accept_segment_is_durable_fenced_and_one_shot(tmp_path: Path) -> None:
    group = _group()
    store = DirectoryStore(tmp_path / "store")
    allocation = ConfiguredAllocation(AllocationId("allocation-1"), BackendJobId("job-1"), {})
    fence = store.begin_attempt(group, "task-a", allocation, "2026-11-05T00:00:00Z")
    store.finish_attempt(fence, "completed", "2026-11-05T00:01:00Z")
    evidence = _evidence(str(Path(fence.output_dir) / "core"), checkpoint_dir=str(Path(fence.output_dir) / "core" / "checkpoint-1"))
    accepted = store.accept_segment(fence, evidence)
    rebuilt = DirectoryStore(tmp_path / "store").manifest(group)
    assert rebuilt.segments == (accepted,)
    with pytest.raises(FenceRejected, match="already has"):
        store.accept_segment(fence, evidence)
    foreign = DirectoryStore(tmp_path / "foreign")
    with pytest.raises(FenceRejected):
        foreign.accept_segment(fence, evidence)


def test_p8_fixture_is_generated_by_executor_helper() -> None:
    assert cross_project_manifest().encode().rstrip(b"\n") == (Path(__file__).parent / "fixtures" / "run-manifest-v2.json").read_bytes().rstrip(b"\n")
