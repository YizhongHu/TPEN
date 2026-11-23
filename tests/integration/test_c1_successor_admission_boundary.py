"""Root producer/consumer boundary for the public C1 reader."""

from __future__ import annotations

import json
from pathlib import Path

from tpen.execution.recovery import (
    ScienceIntent,
    operational_restore_request_from_admission,
    select_successor_admission,
)

from tests.helpers.core_admission import checkpoint, evidence
from tests.helpers.executor_boundary import executor_call


_EXECUTOR_CHILD = r'''
import builtins
import json
import sys
from pathlib import Path

_real_import = builtins.__import__
_tpen_imports = []

def _block_tpen_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tpen" or name.startswith("tpen."):
        _tpen_imports.append(name)
        raise ModuleNotFoundError("executor child must not import tpen")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = _block_tpen_import

def _emit(value):
    modules = sorted(name for name in sys.modules if name == "tpen" or name.startswith("tpen."))
    value["_executor_boundary"] = {"tpen_imports": list(_tpen_imports), "tpen_modules": modules}
    print(json.dumps(value, sort_keys=True))

from types import SimpleNamespace
from tpen_executor import read_successor_admission
import tpen_executor.storage as storage
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.runners import CoreRunEvidence
from tpen_executor.storage import AttemptFence, DirectoryStore

if any(name == "tpen" or name.startswith("tpen.") for name in sys.modules):
    raise RuntimeError("executor child started with tpen already imported")

def _group():
    program = ProgramSpecRef({}, digest({}))
    member = TaskSpec("c1", "c1", program, digest("c1"), 1, "steps")
    return ReadyGroup("c1", (member,), digest("c1-group"))

def _allocation():
    return ConfiguredAllocation(AllocationId("c1-allocation"), BackendJobId("c1-job"), {})

def _fence(payload):
    return AttemptFence(str(Path(payload["store_root"]).resolve()), payload["group_digest"], payload["task_id"], payload["allocation_id"], AttemptId(payload["attempt_id"]), payload["output_dir"])

payload = json.loads(sys.argv[2])
store = DirectoryStore(payload["store_root"])
group = _group()
allocation = _allocation()

if sys.argv[1] == "init":
    fence = store.begin_attempt(group, "c1", allocation, "2026-11-22T00:00:00Z")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    _emit({"store_root": store.root, "group_digest": fence.group_digest, "task_id": fence.task_id, "allocation_id": fence.allocation_id, "attempt_id": fence.attempt_id.value, "output_dir": fence.output_dir, "core_dir": str(core_dir)})
elif sys.argv[1] == "record-core":
    fence = _fence(payload)
    evidence = CoreRunEvidence.from_mapping(json.loads(Path(payload["evidence_path"]).read_text()))
    store.finish_attempt(fence, "failed", "2026-11-22T00:01:00Z")
    _emit({"record": store.record_core_outcome(fence, evidence).to_dict()})
elif sys.argv[1] == "build-request":
    fence = _fence(payload)
    _emit(store.build_successor_admission_request(fence, AttemptId(payload["successor_attempt_id"]), payload["science_input_digest"], payload["target_iterations"]))
elif sys.argv[1] == "record-read":
    fence = _fence(payload)
    store.record_successor_admission(fence, payload["request"], payload["core_acceptance_digest"])
    storage.uuid4 = lambda: SimpleNamespace(hex=payload["successor_attempt_id"])
    successor = store.begin_attempt(group, "c1", allocation, "2026-11-22T00:02:00Z")
    result = read_successor_admission(successor.output_dir, task_id=successor.task_id, attempt_id=successor.attempt_id.value, group_digest=successor.group_digest, science_input_digest=payload["science_input_digest"], target_iterations=payload["target_iterations"])
    _emit({"result": result, "successor": {"attempt_id": successor.attempt_id.value, "output_dir": successor.output_dir}})
else:
    raise SystemExit("unknown action")
'''


def test_c1_reader_preserves_real_core_restore_boundary(tmp_path: Path) -> None:
    """Core admits the child-produced request; executor returns its closed request."""
    store_root = tmp_path / "executor-store"
    initialized = executor_call(_EXECUTOR_CHILD, tmp_path, "init", {"store_root": str(store_root)})
    predecessor_core = Path(initialized["core_dir"])
    ref = checkpoint(predecessor_core)
    evidence_path = predecessor_core / "root-evidence.json"
    evidence_path.write_text(json.dumps(evidence(ref).to_dict(), sort_keys=True), encoding="utf-8")
    executor_call(_EXECUTOR_CHILD, tmp_path, "record-core", {**initialized, "evidence_path": str(evidence_path)})

    intent = ScienceIntent({"fixture": "c1-boundary"}, target_iterations=8)
    request = executor_call(_EXECUTOR_CHILD, tmp_path, "build-request", {**initialized, "successor_attempt_id": "c1-successor", "science_input_digest": intent.science_input_digest, "target_iterations": intent.target_iterations})
    accepted = select_successor_admission(intent, request)
    assert accepted.accepted
    result = executor_call(_EXECUTOR_CHILD, tmp_path, "record-read", {**initialized, "request": request, "core_acceptance_digest": "b" * 64, "successor_attempt_id": "c1-successor", "science_input_digest": intent.science_input_digest, "target_iterations": intent.target_iterations})
    restore = operational_restore_request_from_admission(intent, result["result"]["request"])
    assert restore.ref.content_id == ref.content_id
    assert restore.next_iteration == ref.next_iteration
    assert restore.completed_updates == ref.completed_updates
    assert restore.target_iterations == intent.target_iterations
    assert restore.load == {"mode": "train_resume", "path": str(ref.checkpoint_dir)}
