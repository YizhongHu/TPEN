"""Unrun delta-review proposals: historical reads and intent durability."""

from pathlib import Path

import pytest

import tpen_executor.storage as storage
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import InvocationId
from tpen_executor.storage import DirectoryStore

from test_review_b1b2_l import _record, _started


class _Interrupted(RuntimeError):
    """Simulate interruption after a record link but before directory fsync."""


@pytest.mark.parametrize("advance_attempt", [False, True])
def test_historical_intent_survives_controller_and_attempt_supersession(tmp_path, advance_attempt):
    group, store, fence, binding = _started(tmp_path)
    first = store.acquire_controller_generation(group, _record(group))
    intent = store.create_launch_intent(fence, first, binding)
    store.acquire_controller_generation(group, _record(group, 2, "next", first.record_digest))
    if advance_attempt:
        # The public fresh-start seam advances the same pointer checked by
        # begin_admitted_successor. This isolates the historical read contract;
        # it does not authorize a controller to bypass successor admission.
        current = store.begin_attempt(
            group, fence.task_id,
            ConfiguredAllocation(first.allocation_id, first.backend_job_id, {}),
            allocation_attempt=binding.allocation_attempt,
            invocation_id=InvocationId("new-current-invocation"),
        )
        assert current.attempt_id != fence.attempt_id
    path = Path(fence.output_dir) / "launch-intent.json"
    original = path.read_bytes()
    recovered = DirectoryStore(store.root).read_launch_intent(fence)
    assert recovered == intent
    assert path.read_bytes() == original


def test_intent_gate_syncs_recovered_controller_before_publication(tmp_path, monkeypatch):
    group, store, fence, binding = _started(tmp_path)
    controller = _record(group)
    original_sync = storage._fsync_directory

    def interrupt_after_link(path):
        if (Path(path) / "generation-1.json").exists():
            raise _Interrupted()
        original_sync(path)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_fsync_directory", interrupt_after_link)
        with pytest.raises(_Interrupted):
            store.acquire_controller_generation(group, controller)

    generation_path, = tmp_path.rglob("generation-1.json")
    synced = []
    original_write = storage._write_new

    def record_sync(path):
        original_sync(path)
        synced.append(Path(path))

    def require_authority_barrier(path, payload, **kwargs):
        if Path(path).name == "launch-intent.json":
            assert generation_path.parent in synced, "intent published before recovered authority directory sync"
        return original_write(path, payload, **kwargs)

    # Exercise this gate directly, without a separate require/replay that would
    # mask its missing barrier. This is a storage-order oracle, not permission
    # for a production controller to continue after an uncertain acquire.
    with monkeypatch.context() as patch:
        patch.setattr(storage, "_fsync_directory", record_sync)
        patch.setattr(storage, "_write_new", require_authority_barrier)
        intent = DirectoryStore(store.root).create_launch_intent(fence, controller, binding)
    assert intent.controller_generation == controller.generation
