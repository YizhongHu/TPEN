"""P5c R3 competing-creator probes; run only in a cluster allocation."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.storage import DirectoryStore, FenceRejected
import tpen_executor.storage as storage


@pytest.mark.parametrize("winner", ["directory", "symlink", "file", "vanished"])
def test_r3_t1_competing_creator_is_revalidated_before_begin(tmp_path, monkeypatch, winner):
    """Accept and persist a real winner; refuse other race outcomes unchanged."""
    program = ProgramSpecRef({}, digest({}))
    group = ReadyGroup("r3", (TaskSpec("a", "a", program, digest("a"), 1, "steps"),), digest({}))
    allocation = ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})
    root = tmp_path / "store"
    parent = root / "groups" / group.identity
    parent.mkdir(parents=True)
    target = parent / "a"
    outside = tmp_path / "outside"
    outside.mkdir()
    real_mkdir, real_fsync = storage.os.mkdir, storage._fsync_directory
    hooks, synced = [], []

    def competing_mkdir(path, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) == str(target):
            hooks.append(winner)
            if winner == "directory":
                real_mkdir(path, *args, **kwargs)
            elif winner == "symlink":
                target.symlink_to(outside, target_is_directory=True)
            elif winner == "file":
                target.write_bytes(b"competing creator's bytes")
            # A vanished winner models removal before the post-EEXIST lstat.
            raise FileExistsError(errno.EEXIST, "simulated competing creator", path)
        return real_mkdir(path, *args, **kwargs)

    def observed_fsync(path):
        synced.append(os.path.abspath(os.fspath(path)))
        return real_fsync(path)

    monkeypatch.setattr(storage.os, "mkdir", competing_mkdir)
    monkeypatch.setattr(storage, "_fsync_directory", observed_fsync)
    store = DirectoryStore(root)
    if winner == "directory":
        fence = store.begin_attempt(group, "a", allocation, "start")
        assert (Path(fence.output_dir) / "start.json").is_file()
        assert len(store.manifest(group).attempts) == 1
        assert synced.count(str(parent)) == 1, "accepted winner's parent link was not fsynced"
    else:
        message = "missing" if winner == "vanished" else "not a directory"
        with pytest.raises(FenceRejected, match=message):
            store.begin_attempt(group, "a", allocation, "start")
        assert synced == []
        assert list(parent.iterdir()) == ([] if winner == "vanished" else [target])
        if winner == "symlink":
            assert target.is_symlink()
            assert os.readlink(target) == str(outside)
        elif winner == "file":
            assert target.read_bytes() == b"competing creator's bytes"
    assert list(outside.iterdir()) == []
    assert hooks == [winner], "the intended competing-creator branch was not exercised once"
