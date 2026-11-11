"""P5c R2 delta probes; execute only in a cluster allocation."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.storage import DirectoryStore, FenceRejected
import tpen_executor.storage as storage


def _group(*ids):
    program = ProgramSpecRef({}, digest({}))
    return ReadyGroup("r2", tuple(TaskSpec(i, i, program, digest(i), 1, "steps") for i in ids), digest({}))


def _allocation():
    return ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})


@pytest.mark.parametrize("shared", ["root", "task"], ids=["shared-root", "shared-task"])
def test_r2_t1_concurrent_first_begins_tolerate_a_directory_creation_winner(tmp_path, shared):
    """Both valid starts must survive a shared directory's first creation."""
    ctx = mp.get_context("fork")
    gate, results = ctx.Barrier(2), ctx.Queue()
    root = tmp_path / "store"
    group = _group("a", "b") if shared == "root" else _group("a")
    if shared == "root":
        target = root
        task_ids = ("a", "b")
    else:
        parent = root / "groups" / group.identity
        parent.mkdir(parents=True)
        target = parent / "a"
        task_ids = ("a", "a")

    def beginner(task_id):
        real_mkdir = storage.os.mkdir
        reached = False

        def synchronized_mkdir(path, *args, **kwargs):
            nonlocal reached
            if os.path.abspath(os.fspath(path)) == str(target):
                reached = True
                gate.wait(10)
            return real_mkdir(path, *args, **kwargs)

        storage.os.mkdir = synchronized_mkdir
        try:
            fence = DirectoryStore(root).begin_attempt(group, task_id, _allocation(), "start")
            results.put(("accepted", reached, fence.attempt_id.value))
        except Exception as error:
            results.put((type(error).__name__, reached, str(error)))

    children = [ctx.Process(target=beginner, args=(task_id,)) for task_id in task_ids]
    for child in children:
        child.start()
    for child in children:
        child.join(20)
        assert not child.is_alive(), "bounded begin child did not finish"
        assert child.exitcode == 0
    observed = [results.get(timeout=5) for _ in children]
    assert all(row[1] for row in observed), f"creation rendezvous not reached: {observed!r}"
    assert [row[0] for row in observed] == ["accepted", "accepted"], f"a valid concurrent begin lost the shared-directory creation race: {observed!r}"
    assert len({row[2] for row in observed}) == 2
    assert len(DirectoryStore(root).manifest(group).attempts) == 2


def _snapshot(path):
    if path.is_file():
        return {"file": path.read_bytes()}
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("component", ["root", "groups", "group", "task", "attempt", "start", "outcome", "pointer", "lock"])
def test_r2_t2_finish_and_manifest_refuse_each_preexisting_symlink(tmp_path, component):
    """Every F5 store-owned component rejects a pre-existing link unchanged."""
    root = tmp_path / "store"
    store, group = DirectoryStore(root), _group("a")
    fence = store.begin_attempt(group, "a", _allocation(), "start")
    output = Path(fence.output_dir)
    if component == "outcome":
        store.finish_attempt(fence, "completed", "finish")
    paths = {
        "root": root,
        "groups": root / "groups",
        "group": output.parent.parent,
        "task": output.parent,
        "attempt": output,
        "start": output / "start.json",
        "outcome": output / "outcome.json",
        "pointer": output.parent / "current-attempt.json",
        "lock": output.parent / ".task.lock",
    }
    path = paths[component]
    saved = path.with_name(path.name + ".preserved")
    directory = path.is_dir()
    path.rename(saved)
    path.symlink_to(saved, target_is_directory=directory)
    before = _snapshot(saved)
    with pytest.raises(FenceRejected, match="symlink"):
        store.finish_attempt(fence, "failed", "after-link")
    with pytest.raises(ValueError, match="symlink"):
        store.manifest(group)
    assert path.is_symlink()
    assert _snapshot(saved) == before
