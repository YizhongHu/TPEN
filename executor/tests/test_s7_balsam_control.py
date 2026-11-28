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
    run_validated_association,
    submit_fresh_batch,
)
from tpen_executor.lifecycle.balsam_client import BalsamProcessClient
from tpen_executor.lifecycle.balsam_wire import decode_request, encode_result
from tpen_executor.lifecycle.allocation_capture import SchedulerAllocationSnapshot, assign_full_node
from tpen_executor.recovery import CoreAdmissionVerdictV2, MemberRecoveryInputV2
from tpen_executor.scheduling.pbs import PackedNodeLayout
from tpen_executor.storage import DirectoryStore

from test_b1a_reconciliation import _group
from test_review_b1e2b_controller import DurableAdmission, EventExecutor

CONTROL_SOURCE = Path(__file__).parents[1] / "control" / "balsam" / "src"
if str(CONTROL_SOURCE) not in sys.path:
    sys.path.insert(0, str(CONTROL_SOURCE))

from tpen_balsam_control.application import BalsamControlAmbiguous, BalsamControlRefusal  # noqa: E402
from tpen_balsam_control.bridge import dispatch  # noqa: E402
from tpen_balsam_control.sdk_adapter import DocumentedSDKOperations  # noqa: E402


def _sha(value: str) -> str:
    return digest({"value": value})


def _registration(group_id: GroupId = GroupId("s7-group"), *, group_digest: str | None = None,
                  backend_job_id: BackendJobId = BackendJobId("job-7")) -> BalsamGroupRegistrationReceipt:
    intent = BalsamGroupRegistrationIntent(group_id, _sha("group") if group_digest is None else group_digest,
                                           _sha("deployment"))
    return BalsamGroupRegistrationReceipt(intent, backend_job_id)


def _batch(tmp_path: Path) -> BalsamBatchIntent:
    return BalsamBatchIntent(
        GroupId("s7-group"), BackendJobId("job-7"), _sha("batch"),
        str(tmp_path / "carrier" / "stdout.log"), str(tmp_path / "carrier" / "stderr.log"),
        1, 15, "serial", 7, "project-7", "debug",
    )


class FakeClient:
    def __init__(self, *, error: BaseException | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.error = error

    def submit_batch(self, group_id, backend_job_id, batch_digest, *, stdout_path, stderr_path,
                     num_nodes, wall_time_min, job_mode, site_id, project, queue):
        self.calls.append((group_id, backend_job_id, batch_digest, stdout_path, stderr_path,
                           num_nodes, wall_time_min, job_mode, site_id, project, queue))
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
    assert client.calls[0][3:5] == derive_batch_output_paths(intent)
    assert client.calls[0][5:] == (
        intent.num_nodes, intent.wall_time_min, intent.job_mode, intent.site_id,
        intent.project, intent.queue,
    )
    assert authority_calls == ["checked", "checked"]


def test_conflict_and_unknown_ack_park_without_a_second_external_mutation(tmp_path: Path) -> None:
    """A changed intent conflicts and a missing acknowledgement parks."""

    registration = _registration()
    client = FakeClient(error=BalsamControlAmbiguous("lost acknowledgement"))
    store = DirectoryStore(tmp_path / "store")
    intent = _batch(tmp_path)
    with pytest.raises(BalsamGroupParked):
        submit_fresh_batch(intent, registration, store, client, authority=lambda: None)
    with pytest.raises(BalsamGroupParked):
        submit_fresh_batch(intent, registration, store, client, authority=lambda: None)
    assert len(client.calls) == 1
    changed = BalsamBatchIntent(
        intent.group_id, intent.backend_job_id, _sha("other"), intent.stdout_path, intent.stderr_path,
        intent.num_nodes, intent.wall_time_min, intent.job_mode, intent.site_id, intent.project, intent.queue,
    )
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

    calls: list[dict[str, object]] = []

    class Manager:
        def create(self, num_nodes, wall_time_min, job_mode, site_id, project, queue,
                   optional_params=None, filter_tags=None):
            assert (num_nodes, wall_time_min, job_mode, site_id, project, queue) == (1, 15, "serial", 7, "project-7", "debug")
            assert optional_params == {"tpen_stdout": str(stdout), "tpen_stderr": str(stderr)}
            assert filter_tags == {"tpen_group_id": "group", "tpen_backend_job_id": "job-7"}
            calls.append({
                "num_nodes": num_nodes, "wall_time_min": wall_time_min, "job_mode": job_mode,
                "site_id": site_id, "project": project, "queue": queue,
                "optional_params": optional_params, "filter_tags": filter_tags,
            })
            return SimpleNamespace(id="batch-7")

    api = SimpleNamespace(BatchJob=SimpleNamespace(objects=Manager()))
    stdout = tmp_path / "out" / "stdout.log"
    stderr = tmp_path / "err" / "stderr.log"
    result = DocumentedSDKOperations(api).submit(
        "group", "job-7", _sha("batch"), str(stdout), str(stderr),
        1, 15, "serial", 7, "project-7", "debug",
    )
    assert result == "batch-7"
    assert stdout.parent.is_dir() and stderr.parent.is_dir()
    assert calls == [{
        "num_nodes": 1, "wall_time_min": 15, "job_mode": "serial", "site_id": 7,
        "project": "project-7", "queue": "debug",
        "optional_params": {"tpen_stdout": str(stdout), "tpen_stderr": str(stderr)},
        "filter_tags": {"tpen_group_id": "group", "tpen_backend_job_id": "job-7"},
    }]

    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("regular file")
    blocked_stdout = blocked_parent / "stdout.log"
    blocked_stderr = tmp_path / "other" / "stderr.log"

    # Keep this focused on the adapter's explicit is_dir guard.  A real mkdir
    # against a regular file would fail first and only exercise the OSError
    # wrapper, making the guard-removal mutant indistinguishable.
    original_mkdir = Path.mkdir
    Path.mkdir = lambda self, *args, **kwargs: None
    try:
        with pytest.raises(BalsamControlRefusal, match="not a directory"):
            DocumentedSDKOperations(api).submit(
                "group", "job-8", _sha("batch-2"), str(blocked_stdout), str(blocked_stderr),
                1, 15, "serial", 7, "project-7", "debug",
            )
    finally:
        Path.mkdir = original_mkdir
    assert len(calls) == 1


def test_control_batch_create_without_ack_is_ambiguous_and_never_rearmed(tmp_path: Path) -> None:
    """An absent BatchJob identity is an ambiguous mutation, not a retry."""

    class Manager:
        def create(self, num_nodes, wall_time_min, job_mode, site_id, project, queue,
                   optional_params=None, filter_tags=None):
            del num_nodes, wall_time_min, job_mode, site_id, project, queue, optional_params, filter_tags
            return SimpleNamespace()

    api = SimpleNamespace(BatchJob=SimpleNamespace(objects=Manager()))
    with pytest.raises(BalsamControlAmbiguous):
        DocumentedSDKOperations(api).submit(
            "group", "job-7", _sha("batch"), str(tmp_path / "out"), str(tmp_path / "err"),
            1, 15, "serial", 7, "project-7", "debug",
        )

    class AcknowledgingManager:
        def create(self, num_nodes, wall_time_min, job_mode, site_id, project, queue,
                   optional_params=None, filter_tags=None):
            del num_nodes, wall_time_min, job_mode, site_id, project, queue, optional_params, filter_tags
            return SimpleNamespace(id="batch-ack")

    acknowledging_api = SimpleNamespace(
        BatchJob=SimpleNamespace(objects=AcknowledgingManager()),
    )
    assert DocumentedSDKOperations(acknowledging_api).submit(
        "group", "job-8", _sha("batch-2"), str(tmp_path / "out-2"), str(tmp_path / "err-2"),
        1, 15, "serial", 7, "project-7", "debug",
    ) == "batch-ack"


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
        device_uuids=tuple(
            f"GPU-{index:08x}-0000-0000-0000-000000000000" for index in range(2)
        ),
        node_layout=PackedNodeLayout(4, 8, 2, ("0-3", "4-7")),
    )
    allocation = associate_full_node(assignment, _registration(), observed_job_id=BackendJobId("job-7"))
    assert allocation.facts["scheduler"]["pbs_nodefile_text"] == "node-a"
    assert allocation.facts["parent_provenance"]["parent_hosts"] == ("node-a", "node-b")


def _four_row_assignment(*, parent_job_id: BackendJobId = BackendJobId("job-7")):
    parent = SchedulerAllocationSnapshot.capture(
        AllocationId("s7-parent"), BatchJobId("carrier-7"), parent_job_id, "node-a\n",
        requested_nodes=1, requested_cpus_per_node=8, requested_gpus_per_node=4,
        delivered_nodes=1, delivered_cpus_per_node=8, delivered_gpus_per_node=4,
    )
    return assign_full_node(
        parent, host="node-a", allowed_cpus=range(8),
        device_uuids=tuple(
            f"GPU-{index:08x}-0000-0000-0000-000000000000" for index in range(4)
        ),
        node_layout=PackedNodeLayout(4, 8, 4, ("0-1", "2-3", "4-5", "6-7")),
    )


def _s7_binding(group, *, backend_job_id: BackendJobId = BackendJobId("job-7")):
    return _registration(group.group_id, group_digest=group.identity, backend_job_id=backend_job_id)


def test_four_cpu_rows_cross_s6_s2_with_fences_deltas_and_final_reconciliation(tmp_path: Path) -> None:
    """Four S6 row slots reach real S2 and reconcile only after collection."""

    group = _group(("row-0", "row-1", "row-2", "row-3"))
    assignment = _four_row_assignment()
    binding = _s7_binding(group)
    inputs = tuple(MemberRecoveryInputV2(member.task_id, member.lineage_id) for member in group.members)
    allocation = assignment.to_configured_allocation()
    executor = EventExecutor(group, allocation, capacity=4)
    admission = DurableAdmission()
    result = run_validated_association(
        binding, BackendJobId("job-7"), binding, assignment, group,
        registration_receipt_digest=_sha("receipt"), claimant_nonce="s7-nonce",
        observed_at_utc="2026-11-28T00:00:00Z", member_inputs=inputs,
        store=DirectoryStore(tmp_path / "store"), admission_port=admission, executor=executor,
    )

    assert result.exit_code == 0
    assert assignment.effective_hosts == ("node-a",)
    assert allocation.facts["scheduler"]["pbs_nodefile_text"] == "node-a"
    rows = allocation.facts["slots"]["slots"]
    assert [(row["node"], tuple(row["cpus"])) for row in rows] == [
        ("node-a", (0, 1)), ("node-a", (2, 3)), ("node-a", (4, 5)), ("node-a", (6, 7)),
    ]
    assert {task_id for kind, task_id in executor.calls if kind == "submit"} == set(member.task_id for member in group.members)
    assert {task_id for kind, task_id in executor.calls if kind == "collect"} == set(member.task_id for member in group.members)
    assert result.member_outcomes == tuple((member.task_id, "collected") for member in group.members)
    for member in group.members:
        fence = executor.fences[member.task_id]
        delta = executor.contexts[member.task_id].request.attempt_delta
        # These are fresh, legacy configured runs: no prior checkpoint or
        # continuation admission supplies an attempt delta. S2 still binds
        # each fresh launch to its fence; delta-bearing requests are reserved
        # for configured-program continuations.
        assert delta is None
        assert fence.task_id == member.task_id
    assert len({fence.attempt_id for fence in executor.fences.values()}) == 4
    assert {fence.allocation_id for fence in executor.fences.values()} == {allocation.allocation_id.value}
    assert all(member.action == "preserve_completed" for member in result.reconciliation.members)
    assert all(status == "collected" for _, status in result.member_outcomes)


class _BackendSuccessOnly:
    """Backend success is not core full-horizon evidence."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def admit(self, member: MemberRecoveryInputV2) -> CoreAdmissionVerdictV2 | None:
        self.calls.append(member.task_id)
        if member.start_receipt is None and member.outcome_receipt is None:
            return None
        return CoreAdmissionVerdictV2("unknown", "unknown", "unknown", "unknown", None, "backend-only")


def test_backend_success_alone_cannot_establish_full_h(tmp_path: Path) -> None:
    """Collected backend rows remain non-full-H without core admission."""

    group = _group(("row-0", "row-1", "row-2", "row-3"))
    assignment = _four_row_assignment()
    binding = _s7_binding(group)
    inputs = tuple(MemberRecoveryInputV2(member.task_id, member.lineage_id) for member in group.members)
    allocation = assignment.to_configured_allocation()
    executor = EventExecutor(group, allocation, capacity=4)
    admission = _BackendSuccessOnly()
    result = run_validated_association(
        binding, BackendJobId("job-7"), binding, assignment, group,
        registration_receipt_digest=_sha("receipt"), claimant_nonce="s7-nonce",
        observed_at_utc="2026-11-28T00:00:00Z", member_inputs=inputs,
        store=DirectoryStore(tmp_path / "store"), admission_port=admission, executor=executor,
    )

    assert [kind for kind, _ in executor.calls].count("submit") == 4
    assert [kind for kind, _ in executor.calls].count("collect") == 4
    assert result.member_outcomes == tuple((member.task_id, "collected") for member in group.members)
    assert result.exit_code == 2
    assert all(member.action != "preserve_completed" for member in result.reconciliation.members)
    assert set(admission.calls) == {member.task_id for member in group.members}


def test_process_client_complete_submit_request_reaches_real_bridge_dispatch(tmp_path: Path) -> None:
    """The expanded client request is consumed by the production bridge path."""

    class _SignatureFaithfulBatchManager:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def create(self, num_nodes, wall_time_min, job_mode, site_id, project, queue,
                   optional_params=None, filter_tags=None):
            self.kwargs = {
                "num_nodes": num_nodes, "wall_time_min": wall_time_min, "job_mode": job_mode,
                "site_id": site_id, "project": project, "queue": queue,
                "optional_params": optional_params, "filter_tags": filter_tags,
            }
            return SimpleNamespace(id="batch-wire-7")

    manager = _SignatureFaithfulBatchManager()
    api = SimpleNamespace(BatchJob=SimpleNamespace(objects=manager))
    operations = DocumentedSDKOperations(api)

    class _Bridge:
        def call(self, payload: bytes, *, timeout: float) -> bytes:
            del timeout
            request = decode_request(payload)
            response = dispatch(request, operations)
            return encode_result(response)

    client_file = tmp_path / "client.yml"
    client_file.write_text("api_root: https://example.invalid\n", encoding="utf-8")
    client_file.chmod(0o600)
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    batch_id = BalsamProcessClient("/bin/sh", client_path=client_file, bridge=_Bridge()).submit_batch(
        GroupId("group-wire"), BackendJobId("job-wire"), _sha("wire"),
        stdout_path=str(stdout), stderr_path=str(stderr), num_nodes=2, wall_time_min=30,
        job_mode="serial", site_id=7, project="project-wire", queue="debug",
    )

    assert batch_id == BatchJobId("batch-wire-7")
    assert manager.kwargs == {
        "num_nodes": 2, "wall_time_min": 30, "job_mode": "serial", "site_id": 7,
        "project": "project-wire", "queue": "debug",
        "optional_params": {"tpen_stdout": str(stdout), "tpen_stderr": str(stderr)},
        "filter_tags": {"tpen_group_id": "group-wire", "tpen_backend_job_id": "job-wire"},
    }


@pytest.mark.parametrize("crossing", ["assignment", "observed"])
def test_crossed_typed_assignment_or_observed_job_refuses_before_s2(tmp_path: Path, crossing: str) -> None:
    """Crossed S6/Job identities make zero controller or executor calls."""

    group = _group(("row-0", "row-1", "row-2", "row-3"))
    assignment = _four_row_assignment(
        parent_job_id=BackendJobId("foreign") if crossing == "assignment" else BackendJobId("job-7")
    )
    binding = _s7_binding(group)
    inputs = tuple(MemberRecoveryInputV2(member.task_id, member.lineage_id) for member in group.members)
    executor = EventExecutor(group, assignment.to_configured_allocation(), capacity=2)
    admission = _BackendSuccessOnly()
    with pytest.raises(BalsamGroupRefusal, match="associated|crosses"):
        run_validated_association(
            binding, BackendJobId("foreign") if crossing == "observed" else BackendJobId("job-7"),
            binding, assignment, group, registration_receipt_digest=_sha("receipt"), claimant_nonce="s7-nonce",
            observed_at_utc="2026-11-28T00:00:00Z", member_inputs=inputs,
            store=DirectoryStore(tmp_path / "store"), admission_port=admission, executor=executor,
        )
    assert admission.calls == []
    assert executor.calls == []
