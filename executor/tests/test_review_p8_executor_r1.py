"""Independent P8 r1 record, chain, and durable store probes."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import tpen_executor.storage as storage
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.results import AcceptedSegment, RunManifest
from tpen_executor.runners import CoreRunEvidence
from tpen_executor.storage import DirectoryStore, FenceRejected


def setup_store(tmp_path):
    program = ProgramSpecRef({}, digest({}))
    group = ReadyGroup("g", (TaskSpec("a", "a", program, digest("a"), 1, "steps"),), digest("group"))
    return DirectoryStore(tmp_path / "store"), group, ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})


def finish(store, group, allocation, step):
    fence = store.begin_attempt(group, "a", allocation, "2026-11-05T00:00:00Z")
    store.finish_attempt(fence, "completed", "2026-11-05T00:01:00Z")
    core = str(Path(fence.output_dir) / "core")
    evidence = CoreRunEvidence.from_mapping(dict(schema="tpen.run-evidence/v1", run_dir=core, verdict="completed", run_status="completed", refusal=None, checkpoint=dict(checkpoint_dir=core + f"/checkpoints/step_{step:06d}", schema_version="2", next_iteration=step, completed_updates=step)))
    return fence, evidence


@pytest.mark.parametrize("fork", [False, True], ids=["valid-third", "refuse-fork"])
def test_t2_store_uses_chain_tip_not_uuid_sort(tmp_path, monkeypatch, fork):
    ids = iter(["f" * 32, "1" * 32, "8" * 32])
    monkeypatch.setattr(storage, "uuid4", lambda: SimpleNamespace(hex=next(ids)))
    store, group, allocation = setup_store(tmp_path)
    fence1, evidence1 = finish(store, group, allocation, 1)
    segment1 = store.accept_segment(fence1, evidence1)
    fence2, evidence2 = finish(store, group, allocation, 2)
    segment2 = store.accept_segment(fence2, evidence2, segment1.checkpoint_dir)
    assert store.manifest(group).segments == (segment1, segment2)
    fence3, evidence3 = finish(store, group, allocation, 3)
    if fork:
        with pytest.raises(FenceRejected, match="predecessor"):
            store.accept_segment(fence3, evidence3, segment1.checkpoint_dir)
        assert not (Path(fence3.output_dir) / "segment.json").exists()
    else:
        segment3 = store.accept_segment(fence3, evidence3, segment2.checkpoint_dir)
        assert DirectoryStore(tmp_path / "store").manifest(group).segments == (segment1, segment2, segment3)


def test_t3_store_rejects_foreign_completed_outcome_before_publish(tmp_path):
    store, group, allocation = setup_store(tmp_path)
    fence, evidence = finish(store, group, allocation, 1)
    outcome_path = Path(fence.output_dir) / "outcome.json"
    payload = json.loads(outcome_path.read_text())
    payload["attempt_id"] = "foreign-attempt"
    outcome_path.write_text(json.dumps(payload))
    with pytest.raises(FenceRejected):
        store.accept_segment(fence, evidence)
    assert not (Path(fence.output_dir) / "segment.json").exists()


@pytest.mark.parametrize("mode", ["decode", "construct"])
@pytest.mark.parametrize("bad", ["gap", "duplicate", "backward-step", "backward-updates", "equal", "failed", "unknown", "foreign-task"])
def test_t4_chain_refusals_in_both_entrypoints(tmp_path, mode, bad):
    store, group, allocation = setup_store(tmp_path)
    f1, e1 = finish(store, group, allocation, 2)
    s1 = store.accept_segment(f1, e1)
    f2, e2 = finish(store, group, allocation, 4)
    s2 = store.accept_segment(f2, e2, s1.checkpoint_dir)
    good = store.manifest(group)
    attempts, segments = list(good.attempts), list(good.segments)
    if bad == "gap": segments[1] = replace(s2, predecessor_checkpoint="missing")
    elif bad == "duplicate": segments[1] = replace(s2, attempt_id=s1.attempt_id)
    elif bad == "backward-step": segments[1] = replace(s2, next_iteration=1)
    elif bad == "backward-updates": segments[1] = replace(s2, completed_updates=1)
    elif bad == "equal": segments[1] = replace(s2, next_iteration=2, completed_updates=2)
    elif bad == "failed": attempts[1] = replace(attempts[1], outcome_receipt=replace(attempts[1].outcome_receipt, status="failed"))
    elif bad == "unknown": segments[1] = replace(s2, attempt_id=type(s2.attempt_id)("absent"))
    else: segments[1] = replace(s2, task_id="foreign")
    # Locate the actual second attempt independently of random UUID sort order.
    if bad == "failed":
        attempts = [replace(a, outcome_receipt=replace(a.outcome_receipt, status="failed")) if a.attempt_id == s2.attempt_id else a for a in good.attempts]
    with pytest.raises((TypeError, ValueError)):
        if mode == "construct": RunManifest(good.group_id, good.group_digest, attempts, segments)
        else:
            data = good.to_dict()
            data["attempts"] = [a.to_dict() for a in attempts]
            data["segments"] = [s.to_dict() for s in segments]
            RunManifest.decode(json.dumps(data))


def test_t5_store_fences_and_atomic_publish(tmp_path, monkeypatch):
    store, group, allocation = setup_store(tmp_path)
    fence, evidence = finish(store, group, allocation, 1)
    with pytest.raises(FenceRejected, match="crosses"):
        store.accept_segment(fence, replace(evidence, run_dir=str(tmp_path / "foreign")))
    links, synced = [], []
    original_link, original_sync = storage.os.link, storage._fsync_directory
    def link(source, destination):
        assert not Path(destination).exists()
        links.append((Path(source).read_bytes(), destination))
        return original_link(source, destination)
    def sync(path):
        synced.append(path)
        return original_sync(path)
    monkeypatch.setattr(storage.os, "link", link)
    monkeypatch.setattr(storage, "_fsync_directory", sync)
    segment = store.accept_segment(fence, evidence)
    assert links == [(segment.encode(), str(Path(fence.output_dir) / "segment.json"))]
    assert fence.output_dir in synced
    assert DirectoryStore(tmp_path / "store").manifest(group).segments == (segment,)
    before = (Path(fence.output_dir) / "segment.json").read_bytes()
    with pytest.raises(FenceRejected, match="already has"):
        store.accept_segment(fence, evidence)
    assert (Path(fence.output_dir) / "segment.json").read_bytes() == before
    finish(store, group, allocation, 2)
    with pytest.raises(FenceRejected, match="stale"):
        store.accept_segment(fence, evidence)
