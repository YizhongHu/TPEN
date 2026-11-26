"""S2-L2 canonical group-store creation and reopen contract."""

import hashlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ControllerGenerationV1
from tpen_executor.contracts import ExecutionPlanV2, GroupId, ProgramSpecRef, ReadyGroupV2, TaskSpec, AllocationId, BackendJobId
from tpen_executor.lifecycle.group_store import (
    GroupStoreRefused, InitialGroupStoreRequest, ReentryGroupStoreRequest, initial, reentry,
)


MUTANTS = [
    {"id": "root-exists", "target": "group_store.py", "old": "os.mkdir(root_path)", "new": "pass", "killer": "test_initial_existing_root_refuses"},
    {"id": "binding-canonical", "target": "group_store.py", "old": "stored.encode() != Path(os.path.join(store.root, _BINDING_PATH)).read_bytes()", "new": "False", "killer": "test_noncanonical_binding_refuses"},
    {"id": "binding-exact", "target": "group_store.py", "old": "stored != expected", "new": "False", "killer": "test_foreign_binding_refuses"},
    {"id": "authority-current", "target": "group_store.py", "old": "store.require_current_controller(request.group, expected.controller_generation, expected.controller_nonce)", "new": "pass", "killer": "test_stale_controller_refuses"},
    {"id": "fresh-invocation", "target": "group_store.py", "old": "uuid4().hex", "new": "'fixed'", "killer": "test_reentry_gets_fresh_invocation_location"},
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
    values = {"group": group, "execution_bundle_digest": "c" * 64, "allocation_lineage": "allocation-lineage", "deployment_lineage": "deployment-lineage", "controller": controller}
    values.update(changes)
    return ReentryGroupStoreRequest(**values)


def _tree(root):
    return sorted((str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()) for path in root.rglob("*") if path.is_file())


def test_initial_create_and_identical_reentry(tmp_path):
    group, controller = _fixture()
    root = tmp_path / "store"
    created = initial(InitialGroupStoreRequest(group, "c" * 64, "allocation-lineage", "deployment-lineage", controller), root)
    reopened = reentry(_request(group, controller), root)
    assert reopened.binding == created.binding
    assert reopened.invocation_result_path != reentry(_request(group, controller), root).invocation_result_path


def test_reentry_gets_fresh_invocation_location(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "allocation-lineage", "deployment-lineage", controller), root)
    first = reentry(_request(group, controller), root)
    second = reentry(_request(group, controller), root)
    assert first.invocation_result_path != second.invocation_result_path


def test_initial_existing_root_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"; root.mkdir()
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused):
        initial(InitialGroupStoreRequest(group, "c" * 64, "allocation-lineage", "deployment-lineage", controller), root)
    assert _tree(tmp_path) == before


def test_identical_reentry_separate_process(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "allocation-lineage", "deployment-lineage", controller), root)
    code = "from tpen_executor.lifecycle.group_store import reentry,ReentryGroupStoreRequest; from tpen_executor.contracts import *; from tpen_executor.allocations import *; from tpen_executor._codec import digest; import sys; c={'program':'fixture'}; p=ProgramSpecRef(c,digest(c)); t=TaskSpec('task','lineage',p,'a'*64,1,'steps'); g=ReadyGroupV2(GroupId('group'),(t,),ExecutionPlanV2(c,digest(c),('task',),'continue','preserve')); k=ControllerGenerationV1(g.group_id,1,AllocationId('allocation'),BackendJobId('job'),'b'*64,'nonce',None); r=reentry(ReentryGroupStoreRequest(g,'c'*64,'allocation-lineage','deployment-lineage',k),sys.argv[1]); print(r.binding.group_identity)"
    result = subprocess.run([sys.executable, "-c", code, str(root)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == group.identity


@pytest.mark.parametrize("name,change", [
    ("missing", "unlink"), ("truncated", "truncate"), ("symlink", "symlink"),
])
def test_binding_faults_do_not_change_bytes(tmp_path, name, change):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "allocation-lineage", "deployment-lineage", controller), root)
    binding = root / "group-binding.json"
    if change == "unlink": binding.unlink()
    elif change == "truncate": binding.write_bytes(binding.read_bytes()[:4])
    else: binding.unlink(); binding.symlink_to(root / "elsewhere")
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_noncanonical_binding_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "allocation-lineage", "deployment-lineage", controller), root)
    path = root / "group-binding.json"; path.write_bytes(path.read_bytes() + b"\n"); before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_foreign_binding_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "allocation-lineage", "deployment-lineage", controller), root)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller, execution_bundle_digest="d" * 64), root)
    assert _tree(tmp_path) == before


def test_stale_controller_refuses(tmp_path):
    group, controller = _fixture(); root = tmp_path / "store"
    initial(InitialGroupStoreRequest(group, "c" * 64, "allocation-lineage", "deployment-lineage", controller), root)
    newer = ControllerGenerationV1(group.group_id, 2, controller.allocation_id, controller.backend_job_id, controller.registration_receipt_digest, "new", controller.record_digest)
    from tpen_executor.storage import DirectoryStore
    DirectoryStore(root).acquire_controller_generation(group, newer)
    before = _tree(tmp_path)
    with pytest.raises(GroupStoreRefused): reentry(_request(group, controller), root)
    assert _tree(tmp_path) == before


def test_mutants_are_unique_compilable_and_have_killers():
    source = Path(inspect.getfile(initial)).read_text()
    names = {name for name, value in globals().items() if callable(value)}
    for row in MUTANTS:
        assert source.count(row["old"]) == 1
        assert row["killer"] in names
        assert compile(source.replace(row["old"], row["new"], 1), row["target"], "exec")

