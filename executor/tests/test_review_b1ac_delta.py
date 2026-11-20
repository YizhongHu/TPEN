"""Unrun delta-review proposals for interrupted admitted-start recovery."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

import tpen_executor.storage as storage
from tpen_executor._codec import canonical_bytes
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AttemptId, InvocationId
from tpen_executor.results import StartReceipt
from tpen_executor.storage import DirectoryStore, FenceRejected

from test_b1a_b_publication import _fixture


class _Interrupted(RuntimeError):
    """Simulated interruption before publication finishes."""


def _admitted_start(tmp_path):
    store, group, lease, _, bindings, successors, fences = _fixture(tmp_path)
    request, acceptance, successor = successors["successor"]
    predecessor = fences["successor"]
    admission = store.record_successor_admission(
        predecessor, request, acceptance,
        predecessor_binding=bindings["successor"], successor_binding=successor,
    )
    allocation = ConfiguredAllocation(
        successor.allocation_attempt.allocation_id,
        successor.allocation_attempt.backend_job_id, {},
    )
    output = Path(predecessor.output_dir).parent / successor.attempt_id.value

    def start():
        return DirectoryStore(store.root).begin_admitted_successor(
            group, "successor", allocation, "2026-11-20T00:00:00Z", "remote",
            allocation_attempt=successor.allocation_attempt, admission=admission,
        )

    return store, group, lease, predecessor, output, start


@pytest.mark.parametrize("damage", ["crossed-start", "pointer-schema", "predecessor-symlink", "unrelated-current"])
def test_partial_start_rechecks_predecessor_before_writes(tmp_path, monkeypatch, damage):
    store, group, lease, predecessor, output, start = _admitted_start(tmp_path)
    original_write = storage._write_new

    def interrupt_invocation(path, payload, **kwargs):
        if Path(path) == output / "invocation.json":
            raise _Interrupted()
        original_write(path, payload, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_write_new", interrupt_invocation)
        with pytest.raises(_Interrupted):
            start()

    pointer = output.parent / "current-attempt.json"
    predecessor_dir = Path(predecessor.output_dir)
    if damage == "crossed-start":
        path = predecessor_dir / "start.json"
        record = StartReceipt.decode(path.read_bytes())
        path.write_bytes(replace(record, attempt_id=AttemptId("foreign-predecessor")).encode())
    elif damage == "pointer-schema":
        record = json.loads(pointer.read_bytes())
        record["schema"] = "foreign-pointer-schema"
        pointer.write_bytes(canonical_bytes(record))
    elif damage == "predecessor-symlink":
        # This pre-existing symlink is installed between calls, not swapped
        # concurrently while storage is validating paths.
        moved = predecessor_dir.with_name("relocated-predecessor")
        predecessor_dir.rename(moved)
        predecessor_dir.symlink_to(moved, target_is_directory=True)
    else:
        store.begin_attempt(
            group, "successor", ConfiguredAllocation(lease.allocation_id, lease.backend_job_id, {}),
            allocation_attempt=lease, invocation_id=InvocationId("other-current"),
        )

    before_pointer = pointer.read_bytes()
    before_start = (output / "start.json").read_bytes()
    with pytest.raises(FenceRejected):
        start()
    assert pointer.read_bytes() == before_pointer
    assert (output / "start.json").read_bytes() == before_start
    assert not (output / "invocation.json").exists()


def test_recovery_syncs_visible_receipts_before_pointer_commit(tmp_path, monkeypatch):
    _, _, _, _, output, start = _admitted_start(tmp_path)
    original_sync = storage._fsync_directory
    original_pointer = storage._write_pointer

    def interrupt_after_invocation_link(path):
        if Path(path) == output and (output / "invocation.json").exists():
            # _write_new has fsynced the payload and linked it, but has not
            # synced the directory entry. Both receipts are visible now.
            raise _Interrupted()
        original_sync(path)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_fsync_directory", interrupt_after_invocation_link)
        with pytest.raises(_Interrupted):
            start()

    assert (output / "start.json").is_file()
    assert (output / "invocation.json").is_file()
    synced_receipts = False

    def record_sync(path):
        nonlocal synced_receipts
        original_sync(path)
        if Path(path) == output:
            synced_receipts = True

    def require_receipts_durable(path, attempt_id):
        # This is a durability-order oracle, not a simulated physical reboot:
        # publishing the current pointer must follow the receipt-dir barrier.
        assert synced_receipts, "pointer committed before recovered receipt directory was synced"
        original_pointer(path, attempt_id)

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_fsync_directory", record_sync)
        patch.setattr(storage, "_write_pointer", require_receipts_durable)
        recovered = start()
    assert Path(recovered.output_dir) == output
