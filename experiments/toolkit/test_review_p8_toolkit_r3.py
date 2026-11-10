"""L2 consumer proof: real executor placement output crosses into P8.

Mutant controls
---------------
M7
    Old: accept only ``science-placement-receipt-v1`` and discard v2 fields.
    New: decode the producer's closed v2 receipt and manifest-v3 verdict.
    Killer: ``test_l2_real_executor_producer_to_p8_reader_contract`` in the
    root suite, using the real executor dispatch/store producer.
M8
    Old: freeze placement evidence after core inspection.
    New: validate finite JSON evidence while reading the manifest.
    Killer: ``test_p8_reader_refuses_v2_wire_mutations_before_inspection``.
M9
    Old: trust a root placement verdict without reconciling its entries.
    New: require the manifest-level UNQUALIFIED entries to equal v2 receipts.
    Killer: the real unavailable producer path plus the manifest reconciliation
    check in ``run_results._manifest``.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import textwrap

import pytest

from experiments.toolkit.run_results import ManifestRefusal, read


ROOT = Path(__file__).resolve().parents[2]


def _executor_python() -> str:
    value = os.environ.get("P8_EXECUTOR_PYTHON") or os.environ.get("P9C_EXECUTOR_PYTHON")
    if not value:
        pytest.skip("named skip: P8_EXECUTOR_PYTHON/P9C_EXECUTOR_PYTHON is unset")
    return value


def _producer_code() -> str:
    source = textwrap.dedent(
        """
        import json, os, socket, subprocess, sys
        from pathlib import Path
        from tpen_executor._codec import digest
        from tpen_executor.allocations import ConfiguredAllocation
        from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
        from tpen_executor.placement import AllocationSlot
        from tpen_executor.tasks import ConfiguredRun, ConfiguredRunTable

        root = Path(sys.argv[1])
        mode = sys.argv[2]
        fake_bin = Path(sys.argv[3])
        stub = root / "stub" / "tpen"
        stub.mkdir(parents=True)
        (stub / "__init__.py").write_text("", encoding="utf-8")
        (stub / "run.py").write_text('''
import json, sys
from pathlib import Path
def main(argv):
    values = dict(item.split("=", 1) for item in argv if "=" in item)
    run_dir = Path(values["run.root"]) / values.get("run.run_id", "core")
    checkpoint = run_dir / "checkpoint" / "step_000001"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (run_dir / "evidence.json").write_text(json.dumps({
        "schema": "tpen.run-evidence/v1", "run_dir": str(run_dir),
        "verdict": "completed", "run_status": "completed",
        "checkpoint": {"checkpoint_dir": str(checkpoint), "schema_version": 1,
                        "next_iteration": 1, "completed_updates": 1},
        "refusal": None,
    }), encoding="utf-8")
    return 0
if __name__ == "__main__": raise SystemExit(main(sys.argv[1:]))
''', encoding="utf-8")
        (stub / "inspection.py").write_text('''
import json, sys
from pathlib import Path
print(Path(sys.argv[1], "evidence.json").read_text(encoding="utf-8"))
''', encoding="utf-8")

        config = root / "config.yaml"
        config.write_text("mode: consumer-test\\n", encoding="utf-8")
        group = ReadyGroup("l2-consumer", (TaskSpec("task-a", "task-a", ProgramSpecRef({}, digest({})), digest("task-a"), 1, "steps"),), digest("l2-consumer"))
        cpu = min(getattr(os, "sched_getaffinity", lambda _pid: {0})(0))
        slot = AllocationSlot("logical-slot", "logical-label", ("GPU-ABC",), (cpu,), (), ()).to_dict()
        allocation = ConfiguredAllocation(
            AllocationId("allocation-a"), BackendJobId("job-a"),
            {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot]},
             "scheduler": {"PBS_NODEFILE": socket.gethostname() + "\\n"}},
        )
        paths = {name: root / name for name in ("input.json", "allocation.json", "runs.json", "worker.yaml", "store")}
        paths["input.json"].write_bytes(group.encode())
        paths["allocation.json"].write_bytes(allocation.encode())
        paths["runs.json"].write_bytes(ConfiguredRunTable(group.identity, (ConfiguredRun("task-a", str(config)),)).encode())
        child_env = {"PYTHONPATH": str(stub), "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", "")}
        if mode == "mismatch": child_env["L2_GPU_MISMATCH"] = "1"
        if mode == "refused":
            child_env["TPEN_PLACEMENT_REFUSAL"] = json.dumps({"code": "child-placement-refused", "reason": "consumer refusal control"})
        worker = {
            "worker": {"_target_": "tpen_executor.programs.WorkerProgram", "_recursive_": False,
                "environment": {"_target_": "tpen_executor.environments.PreparedPythonEnvironment", "_recursive_": False,
                    "python": sys.executable, "env": child_env},
                "process": {"_target_": "tpen_executor.workers.SubprocessWorker", "_recursive_": False},
                "runner": {"_target_": "tpen_executor.runners.TPENCommand", "_recursive_": False, "config": str(config)},
                "evidence": {"_target_": "tpen_executor.runners.TPENEvidence", "_recursive_": False}}}
        # This is intentionally a tiny YAML emitter for scalar/map values used by
        # the executor CLI; it keeps the producer entirely inside the executor env.
        lines = ["worker:", "  _target_: tpen_executor.programs.WorkerProgram", "  _recursive_: false", "  environment:", "    _target_: tpen_executor.environments.PreparedPythonEnvironment", "    _recursive_: false", "    python: " + json.dumps(sys.executable), "    env:"]
        for key, value in child_env.items(): lines.append("      " + key + ": " + json.dumps(value))
        lines += ["  process:", "    _target_: tpen_executor.workers.SubprocessWorker", "    _recursive_: false", "  runner:", "    _target_: tpen_executor.runners.TPENCommand", "    _recursive_: false", "    config: " + json.dumps(str(config)), "  evidence:", "    _target_: tpen_executor.runners.TPENEvidence", "    _recursive_: false"]
        paths["worker.yaml"].write_text("\\n".join(lines) + "\\n", encoding="utf-8")
        command = [sys.executable, "-m", "tpen_executor", "dispatch", "--input", str(paths["input.json"]), "--store", str(paths["store"]), "--allocation", str(paths["allocation.json"]), "--worker-config", str(paths["worker.yaml"]), "--runs", str(paths["runs.json"])]
        result = subprocess.run(command, cwd=root, env=dict(os.environ, PYTHONPATH=os.environ["PYTHONPATH"]), capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout, file=sys.stderr); print(result.stderr, file=sys.stderr); raise SystemExit(result.returncode)
        print(paths["store"] / "run-manifest.json")
        """
    )
    return "\n".join(line[8:] if line.startswith("        ") else line for line in source.splitlines()) + "\n"


def _run_real_producer(tmp_path: Path, mode: str) -> Path:
    executor_python = _executor_python()
    fake_bin = tmp_path / mode / "bin"
    fake_bin.mkdir(parents=True)
    fake = fake_bin / "nvidia-smi"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "if '--version' in sys.argv: print('NVIDIA-SMI consumer-test')\n"
        "elif '--query-gpu=uuid,pci.bus_id' in sys.argv:\n"
        "    print('GPU-ABC, 0000:00:01.0')\n"
        "    if os.environ.get('L2_GPU_MISMATCH') == '1': print('GPU-OUT, 0000:00:02.0')\n"
        "elif '--query-compute-apps=pid,gpu_uuid' in sys.argv:\n"
        "    print(f\"{os.getppid()},GPU-OUT\" if os.environ.get('L2_GPU_MISMATCH') == '1' else f\"{os.getppid()},GPU-ABC\")\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    root = tmp_path / mode
    env = dict(os.environ, PYTHONPATH=str(ROOT / "executor" / "src"))
    if mode == "mismatch":
        env["L2_GPU_MISMATCH"] = "1"
    if mode == "unavailable":
        fake.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        fake.chmod(0o755)
    result = subprocess.run([executor_python, "-c", _producer_code(), str(root), mode, str(fake_bin)], cwd=ROOT / "executor", env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return Path(result.stdout.strip())


def _reader_inspector(manifest: dict[str, object]):
    checkpoint = manifest["segments"][0]["checkpoint_dir"]
    return lambda _: {"verdict": "completed", "checkpoint": {"checkpoint_dir": checkpoint, "schema_version": 1, "next_iteration": 1, "completed_updates": 1}}


def test_l2_real_executor_producer_to_p8_reader_contract(tmp_path: Path) -> None:
    """M7-M9: real dispatch output crosses P8 for all placement verdict classes."""

    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("named skip: real placement producer requires Linux CPU-affinity support")
    matched_path = _run_real_producer(tmp_path, "matched")
    matched_manifest = json.loads(matched_path.read_text(encoding="utf-8"))
    matched = read(matched_path, _reader_inspector(matched_manifest))
    assert matched.chains[0].segments[0].placement_receipt["verdict"]["qualification"] == "qualified"
    assert matched.chains[0].segments[0].placement_receipt["verdict"]["reason_class"] == "qualified"
    assert not matched.placement_unqualified

    for mode, reason_class in (("mismatch", "mismatch"), ("unavailable", "telemetry-unavailable"), ("refused", "refused")):
        path = _run_real_producer(tmp_path, mode)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        result = read(path, _reader_inspector(manifest) if manifest["segments"] else None)
        attempts = result.chains[0].segments if result.chains else result.preserved_attempts
        placement = attempts[0].placement_receipt
        assert placement["verdict"]["reason_class"] == reason_class
        if mode == "unavailable":
            assert result.placement_unqualified
            assert "GPU PLACEMENT UNQUALIFIED" in result.placement_verdict["message"]

    foreign = copy.deepcopy(matched_manifest)
    foreign["attempts"][0]["placement_receipt"]["attempt_id"] = "foreign-attempt"
    foreign_path = matched_path.with_name("foreign.json")
    foreign_path.write_text(json.dumps(foreign), encoding="utf-8")
    with pytest.raises(ManifestRefusal, match="attempt disagrees"):
        read(foreign_path, lambda _: pytest.fail("cross-attempt receipt must refuse before inspection"))

    v1 = ROOT / "experiments" / "toolkit" / "fixtures" / "run-manifest-v1.json"
    backward = read(v1, lambda _: pytest.fail("v1 backward read must not inspect an empty chain"))
    assert backward.chains == ()
