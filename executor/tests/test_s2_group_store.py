"""S2-L2 canonical group-store creation and reopen contract."""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ControllerGenerationV1
from tpen_executor.contracts import ExecutionPlanV2, GroupId, ProgramSpecRef, ReadyGroupV2, TaskSpec, AllocationId, BackendJobId
from tpen_executor.lifecycle.group_store import (
    CanonicalGroupBinding, GroupStoreRefused, InitialGroupStoreRequest, ReentryGroupStoreRequest, initial, reentry,
)


MUTANTS = [
    {"id": "root-exists", "target": "executor/src/tpen_executor/lifecycle/group_store.py", "old": "os.mkdir(root_path)", "new": "pass", "killer": "test_initial_existing_root_refuses"},
    {"id": "manifest-strict", "target": "executor/src/tpen_executor/lifecycle/group_store.py", "old": "store.manifest(request.group)", "new": "None", "killer": "test_manifest_reconstruction_refuses_foreign_component"},
    {"id": "binding-read-from-store", "target": "executor/src/tpen_executor/lifecycle/group_store.py", "old": "stored = store.read_canonical(_BINDING_PATH, CanonicalGroupBinding.decode)", "new": "stored = _binding(request, anchor=store.controller_generations(request.group)[0].record_digest)", "killer": "test_missing_binding_refuses"},
    {"id": "binding-regular", "target": "executor/src/tpen_executor/storage.py", "old": "_require_regular(path, \"canonical receipt\")", "new": "pass", "killer": "test_symlink_binding_refuses"},
    {"id": "binding-canonical", "target": "executor/src/tpen_executor/storage.py", "old": "if payload != record.encode():", "new": "if False:", "killer": "test_noncanonical_binding_refuses"},
    {"id": "binding-exact", "target": "executor/src/tpen_executor/lifecycle/group_store.py", "old": "stored != expected", "new": "False", "killer": "test_foreign_binding_refuses"},
    {"id": "binding-ledger-anchor", "target": "executor/src/tpen_executor/lifecycle/group_store.py", "old": "anchor=generations[0].record_digest", "new": "anchor=stored.controller_lineage", "killer": "test_tampered_anchor_refuses_without_caller_anchor"},
    {"id": "authority-fence", "target": "executor/src/tpen_executor/lifecycle/group_store.py", "old": "acquired = store.acquire_controller_generation(request.group, request.controller)", "new": "acquired = request.controller", "killer": "test_successor_generation_reentry_and_retry"},
    {"id": "fresh-invocation", "target": "executor/src/tpen_executor/lifecycle/group_store.py", "old": "uuid4().hex", "new": "'fixed'", "killer": "test_reentry_gets_fresh_invocation_location"},
]


def _fixture():
    config = {"program": "fixture"}
    program = ProgramSpecRef(config, digest(config))
    task = TaskSpec("task", "lineage", program, "a" * 64, 1, "steps")
    plan = ExecutionPlanV2(config, digest(config), ("task",), "continue", "preserve")
    group = ReadyGroupV2(GroupId("group"), (task,), plan)
    controller = ControllerGenerationV1(group.group_id, 1, AllocationId("allocation"), BackendJobId("job"), "b" * 64, "nonce", None)
    return group, controller


def _request(group, controller, **changes):
    values = {"group": group, "execution_bundle_digest": "c" * 64, "deployment_lineage": "deployment-lineage", "controller": controller}
    values.update(changes)
    return ReentryGroupStoreRequest(**values)


def _tree(root):
    return sorted((str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()) for path in root.rglob("*") if path.is_file())


def test_initial_create_and_identical_reentry(tmp_path):
    group, controller = _fixture()
    root = tmp_path / "store"
    created = initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    reopened = reentry(_request(group, controller), root)
    assert reopened.binding == created.binding
    assert reopened.invocation_result_path != reentry(_request(group, controller), root).invocation_result_path


def test_successor_generation_reentry_and_retry(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    successor = ControllerGenerationV1(group.group_id, 2, AllocationId("allocation-2"), BackendJobId("job-2"), "b" * 64, "nonce-2", controller.record_digest)
    first = reentry(_request(group, successor), root)
    retry = reentry(_request(group, successor), root)
    assert first.binding == retry.binding
    assert first.invocation_result_path != retry.invocation_result_path


def test_reentry_gets_fresh_invocation_location(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    first = reentry(_request(group, controller), root)
    second = reentry(_request(group, controller), root)
    assert first.invocation_result_path != second.invocation_result_path


def test_initial_existing_root_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"; root.mkdir()
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused):
        initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    assert _tree(tmp_path) == before


def test_manifest_reconstruction_refuses_foreign_component(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    (root / "groups" / group.identity / "foreign").mkdir(parents=True)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_initial_partial_failure_is_left_for_later_refusal(tmp_path, monkeypatch):
    group, controller = _fixture(); root = tmp_path / "store"
    from tpen_executor.storage import DirectoryStore
    monkeypatch.setattr(DirectoryStore, "acquire_controller_generation", lambda *args: (_ for _ in ()).throw(OSError("injected")))
    with pytest.raises(GroupStoreRefused):
        initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused):
        initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    with pytest.raises(GroupStoreRefused):
        reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_identical_reentry_separate_process(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    code = "from tpen_executor.lifecycle.group_store import reentry,ReentryGroupStoreRequest; from tpen_executor.contracts import *; from tpen_executor.allocations import *; from tpen_executor._codec import digest; import sys; c={'program':'fixture'}; p=ProgramSpecRef(c,digest(c)); t=TaskSpec('task','lineage',p,'a'*64,1,'steps'); g=ReadyGroupV2(GroupId('group'),(t,),ExecutionPlanV2(c,digest(c),('task',),'continue','preserve')); k=ControllerGenerationV1(g.group_id,1,AllocationId('allocation'),BackendJobId('job'),'b'*64,'nonce',None); r=reentry(ReentryGroupStoreRequest(g,'c'*64,'deployment-lineage',k),sys.argv[1]); print(r.binding.group_identity)"
    result = subprocess.run([sys.executable, "-c", code, str(root)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == group.identity


@pytest.mark.parametrize("name,change", [
    ("missing", "unlink"), ("truncated", "truncate"), ("symlink", "symlink"),
])
def test_binding_faults_do_not_change_bytes(tmp_path, name, change):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    binding = root / "group-binding.json"
    if change == "unlink": binding.unlink()
    elif change == "truncate": binding.write_bytes(binding.read_bytes()[:4])
    else: binding.unlink(); binding.symlink_to(root / "elsewhere")
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_missing_binding_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    (root / "group-binding.json").unlink(); before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_symlink_binding_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    binding = root / "group-binding.json"
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_bytes(binding.read_bytes())
    binding.unlink(); binding.symlink_to(elsewhere); before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_noncanonical_binding_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    path = root / "group-binding.json"; path.write_bytes(path.read_bytes() + b"\n"); before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_foreign_binding_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller, execution_bundle_digest="d" * 64), root)
    assert _tree(tmp_path) == before


def test_stale_controller_generation_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    candidate = ControllerGenerationV1(group.group_id, 1, AllocationId("allocation-stale"), BackendJobId("job-2"), "b" * 64, "nonce-stale", None)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, candidate), root)
    assert _tree(tmp_path) == before


def test_skipped_controller_generation_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    candidate = ControllerGenerationV1(group.group_id, 3, AllocationId("allocation-skipped"), BackendJobId("job-2"), "b" * 64, "nonce-skipped", controller.record_digest)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, candidate), root)
    assert _tree(tmp_path) == before


def test_forked_controller_generation_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    candidate = ControllerGenerationV1(group.group_id, 2, AllocationId("allocation-forked"), BackendJobId("job-2"), "b" * 64, "nonce-forked", "e" * 64)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, candidate), root)
    assert _tree(tmp_path) == before


def test_group_binding_mismatch_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    other, _ = _fixture(); other = ReadyGroupV2(GroupId("foreign"), other.members, other.execution_plan)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(other, controller), root)
    assert _tree(tmp_path) == before


def test_deployment_and_controller_lineage_mismatch_refuse(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller, deployment_lineage="foreign"), root)
    assert _tree(tmp_path) == before


def test_foreign_caller_controller_anchor_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused):
        reentry(_request(group, controller, controller_lineage_anchor="d" * 64), root)
    assert _tree(tmp_path) == before


def test_tampered_anchor_refuses_without_caller_anchor(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    created = initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    path = root / "group-binding.json"
    path.write_bytes(CanonicalGroupBinding(created.binding.group_identity, created.binding.execution_bundle_digest,
                                           created.binding.deployment_lineage,
                                           "d" * 64).encode())
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_stale_controller_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "deployment-lineage", controller), root)
    newer = ControllerGenerationV1(group.group_id, 2, controller.allocation_id, controller.backend_job_id, controller.registration_receipt_digest, "new", controller.record_digest)
    from tpen_executor.storage import DirectoryStore
    DirectoryStore(root).acquire_controller_generation(group, newer)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_mutants_are_unique_compilable_and_have_killers():
    names = {name for name, value in globals().items() if callable(value)}
    for row in MUTANTS:
        source = (Path(__file__).parents[2] / row["target"]).read_text()
        assert source.count(row["old"]) == 1
        assert row["killer"] in names
        assert compile(source.replace(row["old"], row["new"], 1), row["target"], "exec")

