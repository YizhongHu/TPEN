"""A-8 root/executor subprocess boundary probe."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
EXECUTOR_SRC = ROOT / "executor" / "src"


PRODUCER = r'''
import hashlib
import json
import os
from pathlib import Path
import sys
import importlib.util

root = Path.cwd()
out = Path(sys.argv[1])
module_path = root / "experiments/atomistic/he-cutover/cutover_plan.py"
spec = importlib.util.spec_from_file_location("he_cutover_cutover_plan", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
grid = module.load_grid(root / "experiments/atomistic/he-cutover/smoke_grid.yaml")
train, evaluation, manifest = module.build_plans(grid, facility="cannon", results_root=out / "results", plan_id="a8-fixture")
module.write_plans(out / "plan", train, evaluation, manifest)
input_path = out / "plan/03_eval/executor_input.json"
group = json.loads(input_path.read_text())
canonical = json.dumps(group, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
group_digest = hashlib.sha256(canonical).hexdigest()
(out / "run-ok.yaml").write_text("ok")
(out / "run-fail.yaml").write_text("fail")
runs = []
for index, member in enumerate(group["members"]):
    config = out / ("run-ok.yaml" if index == 0 else "run-fail.yaml")
    runs.append({"id": member["task_id"], "config": str(config.resolve()), "overrides": ["payload.from_root=true"]})
(out / "runs.json").write_text(json.dumps({"schema": "tpen-executor/configured-runs-v1", "group_digest": group_digest, "runs": runs}, sort_keys=True, separators=(",", ":")))
(out / "allocation.json").write_text(json.dumps({"schema": "tpen-executor/configured-allocation-v1", "allocation_id": "a8-allocation", "backend_job_id": "a8-job", "facts": {}}, sort_keys=True, separators=(",", ":")))
stub = out / "stub/tpen"
stub.mkdir(parents=True)
(stub / "__init__.py").write_text("")
(stub / "run.py").write_text("""
import json
import sys
from pathlib import Path
import tpen.run as tpen_run
def main(argv):
    config = Path(argv[argv.index('--config') + 1])
    values = dict(item.split('=', 1) for item in argv if '=' in item)
    run_dir = Path(values['run.root']) / values['run.run_id']
    if config.read_text() == 'fail':
        return 9
    run_dir.mkdir(parents=True, exist_ok=True)
    stub_root = Path(__file__).resolve().parent
    selected_module = Path(tpen_run.__file__).resolve()
    assert selected_module.is_relative_to(stub_root)
    assert selected_module != (stub_root / '__init__.py').resolve()
    (run_dir / 'selected-tpen-module.txt').write_text(str(selected_module))
    (run_dir / 'evidence.json').write_text(json.dumps({'schema':'tpen.run-evidence/v2','run_dir':str(run_dir),'verdict':'completed','run_status':'completed','checkpoint':{'checkpoint_dir':str(run_dir / 'checkpoint'),'schema_version':1,'next_iteration':1,'completed_updates':1},'refusal':None,'resolved_config_identity':{'schema':'tpen-config-identity-v1','sha256':'a'*64}}))
    return 0
""")
(stub / "inspection.py").write_text("import sys\nfrom pathlib import Path\nprint(Path(sys.argv[1], 'evidence.json').read_text())\n")
worker = f"""worker:
  _target_: tpen_executor.programs.WorkerProgram
  _recursive_: false
  environment:
    _target_: tpen_executor.environments.PreparedPythonEnvironment
    _recursive_: false
    python: {json.dumps(os.environ["P9C_ROOT_PYTHON"])}
    env:
      PYTHONPATH: {json.dumps(str(stub.parent.resolve()))}
  process:
    _target_: tpen_executor.workers.SubprocessWorker
    _recursive_: false
  runner:
    _target_: tpen_executor.runners.TPENCommand
    _recursive_: false
    config: {json.dumps(str((out / "run-ok.yaml").resolve()))}
  evidence:
    _target_: tpen_executor.runners.TPENEvidence
    _recursive_: false
"""
(out / "worker.yaml").write_text(worker)
if "tpen_executor" in sys.modules:
    raise SystemExit("root producer imported tpen_executor")
'''


READER = r'''
import json
from pathlib import Path
import sys
from experiments.toolkit.run_results import ManifestRefusal, read

manifest_path = Path(sys.argv[1])
def inspect_run(path):
    return json.loads(Path(path, "evidence.json").read_text())
result = read(manifest_path, inspect_run=inspect_run)
expected_stub_module = manifest_path.parent.parent / "stub/tpen/run.py"
selected_module = Path(result.chains[0].segments[0].checkpoint_dir).parent / "selected-tpen-module.txt"
stub_root = (manifest_path.parent.parent / "stub/tpen").resolve()
selected_module_path = Path(selected_module.read_text()).resolve()
assert selected_module_path.is_relative_to(stub_root)
assert selected_module_path != (stub_root / "__init__.py").resolve()
payload = {
    "chains": len(result.chains),
    "coordinates": [[segment.next_iteration, segment.completed_updates] for segment in result.chains[0].segments],
    "preserved_attempts": [
        {"task_id": attempt.task_id, "outcome": attempt.outcome, "reason": attempt.reason}
        for attempt in result.preserved_attempts
    ],
    "stub_module_path": str(selected_module_path),
    "expected_stub_module_path": str(expected_stub_module.resolve()),
}
tampered = manifest_path.with_name("tampered-manifest.json")
document = json.loads(manifest_path.read_text())
document["segments"][0]["checkpoint_dir"] = "tampered-checkpoint"
tampered.write_text(json.dumps(document))
try:
    read(tampered, inspect_run=inspect_run)
except ManifestRefusal as error:
    payload["tamper_refused"] = error.code
else:
    raise SystemExit("tampered evidence was accepted")
print(json.dumps(payload, sort_keys=True))
'''


def test_a8_root_producer_executor_cli_and_root_reader_are_separate(tmp_path: Path) -> None:
    executor_python = os.environ.get("P9C_EXECUTOR_PYTHON")
    root_python = os.environ.get("P9C_ROOT_PYTHON")
    if not executor_python:
        pytest.skip("named skip: P9C_EXECUTOR_PYTHON is unset")
    if not root_python:
        pytest.skip("named skip: P9C_ROOT_PYTHON is unset")

    root_env = os.environ.copy()
    root_env["PYTHONPATH"] = str(ROOT)
    root_env["P9C_ROOT_PYTHON"] = root_python
    producer = subprocess.run([root_python, "-c", PRODUCER, str(tmp_path)], cwd=ROOT, env=root_env, capture_output=True, text=True)
    assert producer.returncode == 0, producer.stderr

    executor_env = os.environ.copy()
    executor_env["PYTHONPATH"] = str(EXECUTOR_SRC)
    cli = subprocess.run(
        [executor_python, "-m", "tpen_executor", "dispatch", "--input", str(tmp_path / "plan/03_eval/executor_input.json"), "--store", str(tmp_path / "store"), "--allocation", str(tmp_path / "allocation.json"), "--worker-config", str(tmp_path / "worker.yaml"), "--runs", str(tmp_path / "runs.json")],
        cwd=tmp_path,
        env=executor_env,
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 3
    assert json.loads(cli.stdout)["schema"] == "tpen-executor/run-manifest-v2"

    reader_env = os.environ.copy()
    reader_env["PYTHONPATH"] = str(ROOT)
    reader = subprocess.run([root_python, "-c", READER, str(tmp_path / "store/run-manifest.json")], cwd=ROOT, env=reader_env, capture_output=True, text=True)
    assert reader.returncode == 0, reader.stderr
    observed = json.loads(reader.stdout)
    assert observed["chains"] == 1
    assert observed["coordinates"] == [[1, 1]]
    assert len(observed["preserved_attempts"]) == 1
    assert observed["preserved_attempts"][0]["outcome"] == "failed"
    assert observed["preserved_attempts"][0]["reason"] == "failed-or-incomplete"
    assert observed["stub_module_path"] == observed["expected_stub_module_path"]
    assert observed["tamper_refused"] == "core-evidence-mismatch"
