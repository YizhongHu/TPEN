"""Independent P8 wire parity and real public-core inspection probes."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import pytest
from experiments.toolkit.run_results import ManifestRefusal, read, RUN_MANIFEST_V1, RUN_MANIFEST_V2
from tpen.checkpoint.manifest import CheckpointManifest
from tpen.inspection import inspect_run

ROOT = Path(__file__).resolve().parents[2]


def payload_fixture():
    return json.loads((ROOT / "experiments/toolkit/fixtures/run-manifest-v2.json").read_text())


def core_checkpoint(core, step=1):
    core.mkdir(parents=True, exist_ok=True)
    (core / "status.json").write_text(json.dumps({"status": "completed"}))
    cp = core / "checkpoints" / f"step_{step:06d}"
    cp.mkdir(parents=True)
    model = b"review model"
    (cp / "model.pt").write_bytes(model)
    CheckpointManifest(schema_version=2, kind="tpen.checkpoint", next_iteration=step, completed_updates=step, created_at_unix=1.0, files={"model": "model.pt"}, hashes={"model_sha256": hashlib.sha256(model).hexdigest()}, runtime={}, provenance={}).write(cp / "manifest.json")
    (cp / "COMPLETE").write_text("complete\n")
    return cp


def materialize(tmp_path):
    p = payload_fixture()
    out = tmp_path / "attempt"
    cp = core_checkpoint(out / "core")
    a = p["attempts"][0]
    for item in [a, a["start_receipt"], a["outcome_receipt"]]: item["output_dir"] = str(out)
    p["segments"][0]["checkpoint_dir"] = str(cp)
    return p, out


def write(tmp_path, p):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(p))
    return path


def executor(code, *args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "executor/src"))
    return subprocess.run([os.environ["P8_EXECUTOR_PYTHON"], "-c", code, *map(str, args)], cwd=ROOT / "executor", env=env, text=True, capture_output=True, timeout=60)


@pytest.mark.parametrize("bad", ["foreign-group", "receipt-group-mismatch", "short-digest", "bad-result-digest", "unknown-outcome"])
def test_t1_toolkit_must_match_executor_strict_wire_refusals(tmp_path, bad):
    p, _ = materialize(tmp_path)
    a = p["attempts"][0]
    if bad == "foreign-group": p["group_digest"] = "b" * 64
    elif bad == "receipt-group-mismatch": a["outcome_receipt"]["group_digest"] = "b" * 64
    elif bad == "short-digest":
        for item in [p, a["start_receipt"], a["outcome_receipt"]]: item["group_digest"] = "short"
    elif bad == "bad-result-digest": a["outcome_receipt"]["result_digest"] = "short"
    else:
        a["outcome_receipt"]["status"] = "made-up"
        p["segments"] = []  # Do not let the completed-segment guard mask status validation.
    path = write(tmp_path, p)
    check = executor("import sys; from tpen_executor.results import RunManifest\ntry: RunManifest.decode(open(sys.argv[1], 'rb').read())\nexcept (ValueError, TypeError): print('REFUSED')\nelse: raise AssertionError('executor unexpectedly accepted invalid wire')", path)
    assert check.returncode == 0 and check.stdout.strip() == "REFUSED", (check.stdout, check.stderr)
    with pytest.raises(ManifestRefusal): read(path)


def test_t6_default_reader_uses_real_core_and_refuses_changed_bytes(tmp_path):
    p, out = materialize(tmp_path)
    evidence = inspect_run(out / "core")
    assert evidence.verdict == "completed", evidence.to_dict()
    assert evidence.checkpoint["schema_version"] == 2
    path = write(tmp_path, p)
    result = read(path)
    assert len(result.chains) == 1
    assert len(result.chains[0].segments) == 1
    assert result.chains[0].segments[0].checkpoint_dir == evidence.checkpoint["checkpoint_dir"]
    (Path(evidence.checkpoint["checkpoint_dir"]) / "model.pt").write_bytes(b"changed")
    with pytest.raises(ManifestRefusal, match="core inspection"):
        read(path)


def test_t7_real_core_evidence_can_cross_store_to_default_toolkit(tmp_path):
    code = '''import json, sys
from pathlib import Path
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.runners import CoreRunEvidence
from tpen_executor.storage import DirectoryStore
store = DirectoryStore(Path(sys.argv[1]) / 'store')
program = ProgramSpecRef({}, digest({}))
group = ReadyGroup('g', (TaskSpec('a', 'a', program, digest('a'), 1, 'steps'),), digest('group'))
if sys.argv[2] == 'begin':
    f = store.begin_attempt(group, 'a', ConfiguredAllocation(AllocationId('alloc'), BackendJobId('job'), {}), '2026-11-05T00:00:00Z')
    Path(sys.argv[1], 'fence.json').write_text(json.dumps(dict(root=f.root, group_digest=f.group_digest, task_id=f.task_id, allocation_id=f.allocation_id, attempt_id=f.attempt_id.value, output_dir=f.output_dir)))
    print(f.output_dir)
else:
    from tpen_executor.storage import AttemptFence
    from tpen_executor.contracts import AttemptId
    d = json.loads(Path(sys.argv[1], 'fence.json').read_text()); d['attempt_id'] = AttemptId(d['attempt_id'])
    f = AttemptFence(**d)
    store.finish_attempt(f, 'completed', '2026-11-05T00:01:00Z')
    evidence = CoreRunEvidence.from_mapping(json.loads(Path(sys.argv[1], 'evidence.json').read_text()))
    store.accept_segment(f, evidence)
    Path(sys.argv[1], 'manifest.json').write_bytes(store.manifest(group).encode())
'''
    start = executor(code, tmp_path, "begin")
    assert start.returncode == 0, start.stderr
    core = Path(start.stdout.strip()) / "core"
    core_checkpoint(core)
    evidence = inspect_run(core)
    assert evidence.verdict == "completed", evidence.to_dict()
    (tmp_path / "evidence.json").write_text(json.dumps(evidence.to_dict()))
    accept = executor(code, tmp_path, "accept")
    assert accept.returncode == 0, accept.stderr
    result = read(tmp_path / "manifest.json")
    assert len(result.chains[0].segments) == 1


def test_t10_real_executor_store_manifest_with_placement_receipt_reads_in_toolkit(tmp_path):
    """A real executor store manifest with S2b-1 placement crosses the boundary."""

    code = '''import json, sys
from pathlib import Path
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.results import SciencePlacementReceipt
from tpen_executor.runners import CoreRunEvidence
from tpen_executor.storage import AttemptFence, DirectoryStore
store = DirectoryStore(Path(sys.argv[1]) / 'store')
program = ProgramSpecRef({}, digest({}))
group = ReadyGroup('g', (TaskSpec('a', 'a', program, digest('a'), 1, 'steps'),), digest('group'))
if sys.argv[2] == 'begin':
    fence = store.begin_attempt(group, 'a', ConfiguredAllocation(AllocationId('alloc'), BackendJobId('job'), {}), '2026-11-05T00:00:00Z')
    Path(sys.argv[1], 'fence.json').write_text(json.dumps(dict(root=fence.root, group_digest=fence.group_digest, task_id=fence.task_id, allocation_id=fence.allocation_id, attempt_id=fence.attempt_id.value, output_dir=fence.output_dir)))
    print(fence.output_dir)
else:
    d = json.loads(Path(sys.argv[1], 'fence.json').read_text())
    d['attempt_id'] = AttemptId(d['attempt_id'])
    fence = AttemptFence(**d)
    store.finish_attempt(fence, 'completed', '2026-11-05T00:01:00Z')
    placement = SciencePlacementReceipt(
        'a', fence.attempt_id, fence.group_digest, 'alloc', 'job', 'slot-a', digest({'payload': 'a'}), fence.output_dir,
        {'host': 'node-a', 'pid': 1234, 'proc_start_ticks': 55, 'affinity': [0], 'thread_settings': {'OMP_NUM_THREADS': '1'}, 'cuda_visible_devices': None, 'observed_at_utc': '2026-11-05T00:00:01Z'},
        {'host': 'node-a', 'cpus': [0], 'devices': []},
        {'status': 'unavailable', 'reason': 'CPU-only test', 'method': 's2b-1-child-generation-no-sampler', 'started_at_utc': '2026-11-05T00:00:01Z', 'finished_at_utc': '2026-11-05T00:01:00Z', 'samples': [], 'active_uuids': [], 'active_pci': []},
        {'host': 'matched', 'affinity': 'matched', 'device': 'unavailable'},
    )
    store.publish_placement(fence, placement)
    evidence = CoreRunEvidence.from_mapping(json.loads(Path(sys.argv[1], 'evidence.json').read_text()))
    store.accept_segment(fence, evidence)
    Path(sys.argv[1], 'manifest.json').write_bytes(store.manifest(group).encode())
'''
    start = executor(code, tmp_path, "begin")
    assert start.returncode == 0, start.stderr
    core = Path(start.stdout.strip()) / "core"
    checkpoint = core_checkpoint(core)
    evidence = inspect_run(core)
    assert evidence.verdict == "completed", evidence.to_dict()
    (tmp_path / "evidence.json").write_text(json.dumps(evidence.to_dict()))
    finish = executor(code, tmp_path, "accept")
    assert finish.returncode == 0, finish.stderr

    result = read(tmp_path / "manifest.json")
    placement = result.chains[0].segments[0].placement_receipt
    assert placement is not None
    assert placement["schema"] == "tpen-executor/science-placement-receipt-v1"
    assert placement["slot_id"] == "slot-a"
    assert placement["child"]["proc_start_ticks"] == 55
    assert placement["gpu"]["status"] == "unavailable"
    assert result.chains[0].segments[0].checkpoint_dir == str(checkpoint)


def test_t8_wire_fixtures_and_experiment_boundary():
    for name in ["run-manifest-v1.json", "run-manifest-v2.json"]:
        assert (ROOT / "executor/tests/fixtures" / name).read_bytes() == (ROOT / "experiments/toolkit/fixtures" / name).read_bytes()
    for path in (ROOT / "experiments").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import): assert all(n.name.split('.')[0] != "tpen_executor" for n in node.names), path
            if isinstance(node, ast.ImportFrom): assert (node.module or '').split('.')[0] != "tpen_executor", path
    assert RUN_MANIFEST_V1 == "tpen-executor/run-manifest-v1"
    assert RUN_MANIFEST_V2 == "tpen-executor/run-manifest-v2"


@pytest.mark.parametrize("bad", ["duplicate-root-key", "duplicate-nested-key", "null-outcome", "reversed-chain"])
def test_t9_reader_fail_closed_without_other_defenses_masking(tmp_path, bad):
    p, out = materialize(tmp_path)
    if bad == "null-outcome": p["attempts"][0]["outcome_receipt"] = None
    elif bad == "reversed-chain":
        second = copy.deepcopy(p["attempts"][0])
        second["attempt_id"] = "second"
        for receipt in [second["start_receipt"], second["outcome_receipt"]]: receipt["attempt_id"] = "second"
        p["attempts"].append(second)
        s = dict(p["segments"][0], attempt_id="second", predecessor_checkpoint=p["segments"][0]["checkpoint_dir"], checkpoint_dir="later", next_iteration=2, completed_updates=2)
        p["segments"] = [s, p["segments"][0]]
    path = write(tmp_path, p)
    if bad.startswith("duplicate"):
        text = path.read_text()
        if bad == "duplicate-root-key": text = text.replace('{', '{"schema":"wrong",', 1)
        else: text = text.replace('"task_id":', '"task_id":"wrong","task_id":', 1)
        path.write_text(text)
    with pytest.raises(ManifestRefusal): read(path, lambda _: pytest.fail("must refuse before inspection"))
