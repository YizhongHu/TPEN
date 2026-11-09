"""Independent P8 wire parity and real public-core inspection probes."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from omegaconf import OmegaConf
from experiments.toolkit.run_results import ManifestRefusal, read, RUN_MANIFEST_V1, RUN_MANIFEST_V2
from experiments.toolkit.config_identity import expected_resolved_config_identity
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


def executor(code, *args, python=None):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "executor/src"))
    executable = python or os.environ["P8_EXECUTOR_PYTHON"]
    return subprocess.run([executable, "-c", code, *map(str, args)], cwd=ROOT / "executor", env=env, text=True, capture_output=True, timeout=60)


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


def test_t7_real_core_executor_and_study_identity_agree(tmp_path):
    config = tmp_path / "native_recovery.yaml"
    source = ROOT / "tests/integration/artifacts/execution/native_recovery.yaml"
    config.write_text(
        source.read_text()
        + "\n# EVAL-style run-derived operational path retained in the intended composition.\n"
        + "evaluator:\n"
        + "  _target_: tpen.evaluation.Evaluator\n"
        + "  tasks:\n"
        + "    - name: trajectory\n"
        + "      output_dir: ${run.dir}/trajectory\n",
        encoding="utf-8",
    )
    overrides = ("trainer.max_steps=1", "sampler.n_walkers=2", "sampler.burn_in=0", "system.omega=0.75")
    manifest = tmp_path / "manifest.json"
    code = '''import os, sys
from pathlib import Path
from types import SimpleNamespace
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.dispatch import LocalDispatcher
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.runners import TPENCommand, TPENEvidence
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.workers import SubprocessWorker
root = Path(sys.argv[3]).resolve()
config = Path(sys.argv[1]).resolve()
env = PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": os.pathsep.join((str(root), str(root / "executor" / "src")))})
worker = SimpleNamespace(materialize=lambda _: {"environment": env, "process": SubprocessWorker(), "runner": TPENCommand(str(config)), "evidence": TPENEvidence()})
program = ProgramSpecRef({}, digest({}))
group = ReadyGroup("g", (TaskSpec("run", "run", program, digest("run"), 1, "steps"),), digest("group"))
allocation = ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})
store = DirectoryStore(Path(sys.argv[2]).with_suffix(".store"))
configured = ConfiguredRun("run", str(config), tuple(sys.argv[4:]))
result = LocalDispatcher(worker).run(group, {"run": configured}, AttachedAllocation(allocation, store), timeout=120)
Path(sys.argv[2]).write_bytes(result.encode())
'''
    # The executor process itself is the real dispatcher path, while the
    # launched core child must use the Torch-enabled root environment.  The
    # arm-specific P8 interpreter is intentionally dependency-light and cannot
    # stand in for the core runtime here.
    run = executor(code, config, manifest, ROOT, *overrides, python=sys.executable)
    assert run.returncode == 0, (run.stdout, run.stderr)
    result = read(manifest)
    assert len(result.chains) == 1 and len(result.chains[0].segments) == 1
    actual_identity = result.chains[0].segments[0].resolved_config_identity
    intended = OmegaConf.to_container(
        OmegaConf.merge(OmegaConf.load(config), OmegaConf.from_dotlist(list(overrides))),
        resolve=False,
    )
    expected = expected_resolved_config_identity(intended)
    assert actual_identity == expected
    mismatched = copy.deepcopy(intended)
    mismatched["system"]["omega"] = 0.5
    assert expected_resolved_config_identity(mismatched) != actual_identity


def test_t7_reader_refuses_explicit_null_segment_identity(tmp_path):
    payload, _ = materialize(tmp_path)
    payload["segments"][0]["resolved_config_identity"] = None
    path = write(tmp_path, payload)
    with pytest.raises(ManifestRefusal, match="must not be null"):
        read(path, lambda _: pytest.fail("must refuse before inspection"))


def test_t10_reader_refuses_a_segment_identity_mismatch(tmp_path):
    p, out = materialize(tmp_path)
    identity = expected_resolved_config_identity({"science": {"charge": 1}})
    (out / "core" / "run_start.json").write_text(json.dumps({"resolved_config_identity": identity}))
    p["segments"][0]["resolved_config_identity"] = identity
    p["segments"][0]["resolved_config_identity"]["sha256"] = "b" * 64
    path = write(tmp_path, p)
    with pytest.raises(ManifestRefusal, match="identity"):
        read(path)


def test_t10_real_executor_store_manifest_with_placement_and_identity_reads_in_toolkit(tmp_path):
    """One real P8 manifest carries both S2b-1 placement and core identity."""

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
    identity = expected_resolved_config_identity({"science": {"charge": 1}})
    (core / "run_start.json").write_text(json.dumps({"resolved_config_identity": identity}))
    evidence = inspect_run(core)
    assert evidence.verdict == "completed", evidence.to_dict()
    assert evidence.resolved_config_identity == identity
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
    assert result.chains[0].segments[0].resolved_config_identity == identity


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
