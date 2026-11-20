"""Unrun T2 proposals for authority recovery and intent provenance."""

from dataclasses import replace
from pathlib import Path

import pytest

import tpen_executor.storage as storage
from tpen_executor.contracts import GroupId, InvocationId
from tpen_executor.storage import ControllerGenerationRejected, DirectoryStore, FenceRejected

from test_review_b1b2_l import _group, _record, _started


class _Interrupted(RuntimeError):
    """Simulated interruption before a durability barrier."""


@pytest.mark.parametrize("operation", ["replay", "require"])
def test_visible_generation_recovery_requires_directory_barrier(tmp_path, monkeypatch, operation):
    group = _group()
    store = DirectoryStore(tmp_path)
    record = _record(group)
    original_sync = storage._fsync_directory

    def interrupt_after_link(path):
        if (Path(path) / "generation-1.json").exists():
            raise _Interrupted()
        original_sync(path)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_fsync_directory", interrupt_after_link)
        with pytest.raises(_Interrupted):
            store.acquire_controller_generation(group, record)

    generation_path, = tmp_path.rglob("generation-1.json")
    synced = []

    def record_sync(path):
        original_sync(path)
        synced.append(Path(path))

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_fsync_directory", record_sync)
        reopened = DirectoryStore(tmp_path)
        if operation == "replay":
            recovered = reopened.acquire_controller_generation(group, record)
        else:
            recovered = reopened.require_current_controller(group, 1, record.claimant_nonce)
    assert recovered == record
    # An ordering oracle, not a simulated power loss or facility qualification.
    assert generation_path.parent in synced, "visible generation returned without a directory barrier"


def test_recovery_persists_ledger_directory_parent(tmp_path, monkeypatch):
    group = _group()
    store = DirectoryStore(tmp_path)
    original_sync = storage._fsync_directory
    ledger_parents = []

    def interrupt_after_mkdir(path):
        if (Path(path) / "controller-generations").is_dir():
            ledger_parents.append(Path(path))
            raise _Interrupted()
        original_sync(path)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_fsync_directory", interrupt_after_mkdir)
        with pytest.raises(_Interrupted):
            store.acquire_controller_generation(group, _record(group))
    ledger_parent, = ledger_parents
    synced = []

    def record_sync(path):
        original_sync(path)
        synced.append(Path(path))

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_fsync_directory", record_sync)
        DirectoryStore(tmp_path).acquire_controller_generation(group, _record(group))
    assert ledger_parent in synced, "existing ledger directory link was never persisted"


@pytest.mark.parametrize("crossing", ["invocation", "task", "controller", "output", "stale"])
def test_intent_rechecks_durable_fence_and_binding(tmp_path, crossing):
    group, store, fence, binding = _started(tmp_path / "store")
    controller = store.acquire_controller_generation(group, _record(group))
    original_output = Path(fence.output_dir)
    foreign_output = tmp_path / "foreign"
    foreign_output.mkdir()
    if crossing == "invocation":
        binding = replace(binding, invocation_id=InvocationId("different-invocation"))
    elif crossing == "task":
        binding = replace(binding, task_id="different-task")
    elif crossing == "controller":
        # Keep the real current generation+nonce while changing its identity.
        controller = replace(controller, registration_receipt_digest="c" * 64)
    elif crossing == "output":
        fence = replace(fence, output_dir=str(foreign_output))
    else:
        from tpen_executor.allocations import ConfiguredAllocation

        store.begin_attempt(
            group, fence.task_id,
            ConfiguredAllocation(controller.allocation_id, controller.backend_job_id, {}),
            allocation_attempt=binding.allocation_attempt,
            invocation_id=InvocationId("new-current-invocation"),
        )
    with pytest.raises(FenceRejected):
        store.create_launch_intent(fence, controller, binding)
    assert not (original_output / "launch-intent.json").exists()
    assert not (foreign_output / "launch-intent.json").exists()


@pytest.mark.parametrize("damage", ["output", "group", "noncanonical"])
def test_intent_recovery_rejects_crossed_or_noncanonical_record(tmp_path, damage):
    group, store, fence, binding = _started(tmp_path / "store")
    controller = store.acquire_controller_generation(group, _record(group))
    intent = store.create_launch_intent(fence, controller, binding)
    path = Path(fence.output_dir) / "launch-intent.json"
    if damage == "output":
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        (foreign / "launch-intent.json").write_bytes(intent.encode())
        fence = replace(fence, output_dir=str(foreign))
    elif damage == "group":
        path.write_bytes(replace(intent, group_id=GroupId("other-group")).encode())
    else:
        path.write_bytes(intent.encode() + b"\n")
    with pytest.raises(FenceRejected):
        DirectoryStore(store.root).read_launch_intent(fence)


@pytest.mark.parametrize("damage", ["group", "predecessor", "gap", "duplicate", "filename", "noncanonical"])
@pytest.mark.parametrize("operation", ["require", "acquire"])
def test_invalid_ledger_never_grants_authority(tmp_path, damage, operation):
    group = _group()
    store = DirectoryStore(tmp_path)
    first = store.acquire_controller_generation(group, _record(group))
    second = store.acquire_controller_generation(group, _record(group, 2, "next", first.record_digest))
    path, = tmp_path.rglob("generation-2.json")
    if damage == "group":
        second = replace(second, group_id=GroupId("other-group"))
        path.write_bytes(second.encode())
    elif damage == "predecessor":
        second = replace(second, predecessor_record_digest="c" * 64)
        path.write_bytes(second.encode())
    elif damage == "gap":
        # Move only a fixture-owned record, preserving the evidence bytes.
        (path.parent / "generation-1.json").rename(tmp_path / "saved-generation-1")
    elif damage == "duplicate":
        (path.parent / "generation-copy.json").write_bytes(second.encode())
    elif damage == "filename":
        path.rename(path.parent / "generation-3.json")
    else:
        path.write_bytes(second.encode() + b"\n")
    with pytest.raises(ControllerGenerationRejected):
        if operation == "require":
            store.require_current_controller(group, 2, "next")
        else:
            store.acquire_controller_generation(group, _record(group, 3, "third", second.record_digest))


def test_same_group_id_shares_authority_across_plan_identity(tmp_path):
    group = _group()
    changed = replace(group, members=(replace(group.members[0], lineage_id="other-lineage"),))
    assert changed.identity != group.identity
    store = DirectoryStore(tmp_path)
    first = store.acquire_controller_generation(group, _record(group))
    second = store.acquire_controller_generation(changed, _record(changed, 2, "next", first.record_digest))
    assert store.require_current_controller(group, 2, "next") == second
    with pytest.raises(ControllerGenerationRejected):
        store.require_current_controller(changed, 1, first.claimant_nonce)


def test_partial_temporary_record_is_not_authority(tmp_path):
    group = _group()
    store = DirectoryStore(tmp_path)
    first = store.acquire_controller_generation(group, _record(group))
    path, = tmp_path.rglob("generation-1.json")
    (path.parent / ".tmp-interrupted").write_bytes(b"incomplete JSON")
    assert DirectoryStore(tmp_path).require_current_controller(group, 1, first.claimant_nonce) == first
