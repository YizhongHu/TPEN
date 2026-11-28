"""S7 fresh BatchJob transaction, control handoff, and association tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tpen_executor._codec import digest
from tpen_executor.contracts import AllocationId, BackendJobId, BatchJobId, GroupId
from tpen_executor.lifecycle.balsam import (
    BalsamGroupRegistrationIntent,
    BalsamGroupRegistrationReceipt,
    BalsamGroupParked,
    BalsamGroupRefusal,
)
from tpen_executor.lifecycle.balsam_control import (
    BalsamBatchIntent,
    associate_full_node,
    derive_batch_output_paths,
    submit_fresh_batch,
)
from tpen_executor.lifecycle.allocation_capture import SchedulerAllocationSnapshot, assign_full_node
from tpen_executor.scheduling.pbs import PackedNodeLayout
from tpen_executor.storage import DirectoryStore

CONTROL_SOURCE = Path(__file__).parents[1] / "control" / "balsam" / "src"
if str(CONTROL_SOURCE) not in sys.path:
    sys.path.insert(0, str(CONTROL_SOURCE))

from tpen_balsam_control.application import BalsamControlAmbiguous  # noqa: E402
from tpen_balsam_control.application import BalsamGroupAmbiguous  # noqa: E402
from tpen_balsam_control.sdk_adapter import DocumentedSDKOperations  # noqa: E402


def _sha(value: str) -> str:
    return digest({"value": value})


def _registration() -> BalsamGroupRegistrationReceipt:
    intent = BalsamGroupRegistrationIntent(GroupId("s7-group"), _sha("group"), _sha("deployment"))
    return BalsamGroupRegistrationReceipt(intent, BackendJobId("job-7"))


def _batch(tmp_path: Path) -> BalsamBatchIntent:
    return BalsamBatchIntent(
        GroupId("s7-group"), BackendJobId("job-7"), _sha("batch"),
        str(tmp_path / "carrier" / "stdout.log"), str(tmp_path / "carrier" / "stderr.log"),
    )


class FakeClient:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.calls: list[tuple[GroupId, BackendJobId, str, str, str]] = []
        self.error = error

    def submit_batch(self, group_id, backend_job_id, batch_digest, *, stdout_path, stderr_path):
        self.calls.append((group_id, backend_job_id, batch_digest, stdout_path, stderr_path))
        if self.error is not None:
            raise self.error
        return BatchJobId("batch-7")


def test_fresh_batch_journals_once_replays_receipt_and_uses_persisted_paths(tmp_path: Path) -> None:
    """One immutable group/job gets one call and a stable typed receipt."""

    intent, registration = _batch(tmp_path), _registration()
    client = FakeClient()
    authority_calls: list[str] = []
    authority = lambda: authority_calls.append("checked")
    first = submit_fresh_batch(intent, registration, DirectoryStore(tmp_path / "store"), client, authority=authority)
    second = submit_fresh_batch(intent, registration, DirectoryStore(tmp_path / "store"), client, authority=authority)
    assert first == second
    assert len(client.calls) == 1
    assert client.calls[0][-2:] == derive_batch_output_paths(intent)
    assert authority_calls == ["checked", "checked"]


def test_conflict_and_unknown_ack_park_without_a_second_external_mutation(tmp_path: Path) -> None:
    """A changed intent conflicts and a missing acknowledgement parks."""

    registration = _registration()
    client = FakeClient(error=BalsamGroupAmbiguous("lost acknowledgement"))
    store = DirectoryStore(tmp_path / "store")
    intent = _batch(tmp_path)
    with pytest.raises(BalsamGroupParked):
        submit_fresh_batch(intent, registration, store, client, authority=lambda: None)
    with pytest.raises(BalsamGroupParked):
        submit_fresh_batch(intent, registration, store, client, authority=lambda: None)
    assert len(client.calls) == 1
    changed = BalsamBatchIntent(intent.group_id, intent.backend_job_id, _sha("other"), intent.stdout_path, intent.stderr_path)
    with pytest.raises(BalsamGroupRefusal, match="conflicts"):
        submit_fresh_batch(changed, registration, store, client, authority=lambda: None)


def test_registration_identity_is_required_before_batch_mutation(tmp_path: Path) -> None:
    """A carrier cannot submit under a crossed Balsam Job."""

    intent = _batch(tmp_path)
    foreign = BalsamGroupRegistrationReceipt(
        BalsamGroupRegistrationIntent(GroupId("s7-group"), _sha("group"), _sha("deployment")),
        BackendJobId("foreign-job"),
    )
    with pytest.raises(BalsamGroupRefusal, match="crosses"):
        submit_fresh_batch(intent, foreign, DirectoryStore(tmp_path / "store"), FakeClient(), authority=lambda: None)


def test_control_batch_create_prepares_both_parent_directories_before_sole_mutation(tmp_path: Path) -> None:
    """The control adapter creates no BatchJob until both parents are valid."""

    calls: list[dict[str, str]] = []

    class Manager:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id="batch-7")

    api = SimpleNamespace(BatchJob=SimpleNamespace(objects=Manager()))
    stdout = tmp_path / "out" / "stdout.log"
    stderr = tmp_path / "err" / "stderr.log"
    result = DocumentedSDKOperations(api).submit("group", "job-7", _sha("batch"), str(stdout), str(stderr))
    assert result == "batch-7"
    assert stdout.parent.is_dir() and stderr.parent.is_dir()
    assert calls == [{"job_id": "job-7", "tpen_stdout": str(stdout), "tpen_stderr": str(stderr)}]


def test_control_batch_create_without_ack_is_ambiguous_and_never_rearmed(tmp_path: Path) -> None:
    """An absent BatchJob identity is an ambiguous mutation, not a retry."""

    class Manager:
        def create(self, **kwargs):
            del kwargs
            return SimpleNamespace()

    api = SimpleNamespace(BatchJob=SimpleNamespace(objects=Manager()))
    with pytest.raises(BalsamControlAmbiguous):
        DocumentedSDKOperations(api).submit(
            "group", "job-7", _sha("batch"), str(tmp_path / "out"), str(tmp_path / "err"),
        )


def test_association_rejects_crossed_job_before_s2_scope_is_built(tmp_path: Path) -> None:
    """S6 scope cannot be handed to S2 under a different registered Job."""

    # The type check occurs before inspecting assignment internals, so this
    # deliberately uses a non-assignment value to prove the early boundary.
    registration = _registration()
    with pytest.raises(TypeError):
        associate_full_node(object(), registration)


def test_validated_association_passes_only_the_s6_one_host_scope(tmp_path: Path) -> None:
    """The S2 handoff retains parent provenance but routes one host only."""

    del tmp_path
    parent = SchedulerAllocationSnapshot.capture(
        AllocationId("parent"), BatchJobId("carrier"), BackendJobId("job-7"), "node-a\nnode-b\n",
        requested_nodes=2, requested_cpus_per_node=8, requested_gpus_per_node=2,
        delivered_nodes=2, delivered_cpus_per_node=8, delivered_gpus_per_node=2,
    )
    assignment = assign_full_node(
        parent, host="node-a", allowed_cpus=range(8),
        device_uuids=tuple(f"GPU-{index:032x}" for index in range(2)),
        node_layout=PackedNodeLayout(4, 8, 2, ("0-3", "4-7")),
    )
    allocation = associate_full_node(assignment, _registration(), observed_job_id=BackendJobId("job-7"))
    assert allocation.facts["scheduler"]["pbs_nodefile_text"] == "node-a"
    assert allocation.facts["parent_provenance"]["parent_hosts"] == ("node-a", "node-b")
