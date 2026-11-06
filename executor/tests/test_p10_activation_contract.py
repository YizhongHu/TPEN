"""P10 A1-A5 contract tests with a fake core process and no science run."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tpen_executor import composition
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.placement import binding_from_allocation
from tpen_executor.tasks import ConfiguredRun, ConfiguredRunTable


SRC = Path(__file__).parents[1] / "src"


def _group() -> ReadyGroup:
    program = ProgramSpecRef({"route": "he-train-attached"}, digest({"route": "he-train-attached"}))
    return ReadyGroup("p10", (TaskSpec("train:seed-0000", "train:seed-0000", program, digest("science"), 1, "steps"),), digest("plan"))


def _inputs(tmp_path: Path, *, evidence: bool = True) -> dict[str, Path]:
    group = _group()
    config = tmp_path / "config.yaml"
    config.write_text("ok", encoding="utf-8")
    package = tmp_path / "fake_core" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        textwrap.dedent(
            f"""
            import json, os, sys
            from pathlib import Path
            def main(argv):
                values = dict(item.split('=', 1) for item in argv if '=' in item)
                run_dir = Path(values['run.root']) / values['run.run_id']
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / 'process-identity.json').write_text(json.dumps({'pid': os.getpid(), 'start_token': __import__('time').time_ns(), 'nonce': os.urandom(8).hex(), 'hostname': __import__('socket').gethostname()}))
                {"(run_dir / 'evidence.json').write_text(json.dumps({'schema':'tpen.run-evidence/v1','run_dir':str(run_dir),'verdict':'completed','run_status':'completed','checkpoint':{'checkpoint_dir':str(run_dir / 'checkpoint'),'schema_version':1,'next_iteration':1,'completed_updates':1},'refusal':None}))" if evidence else "pass"}
                return 0
            """
        ),
        encoding="utf-8",
    )
    (package / "inspection.py").write_text(
        "import sys; from pathlib import Path; print(Path(sys.argv[1], 'evidence.json').read_text())\n",
        encoding="utf-8",
    )
    allocation = ConfiguredAllocation(
        AllocationId("alloc"),
        BackendJobId("job"),
        {"binding": {"environment": {"P10_NODE": "node-0"}, "worker_id": "worker-0", "node": "node-0"}},
    )
    worker = tmp_path / "worker.yaml"
    worker.write_text(
        textwrap.dedent(
            f"""
            worker:
              _target_: tpen_executor.programs.WorkerProgram
              _recursive_: false
              environment:
                _target_: tpen_executor.environments.PreparedPythonEnvironment
                _recursive_: false
                python: {json.dumps(sys.executable)}
                env:
                  PYTHONPATH: {json.dumps(str(package.parent))}
              process:
                _target_: tpen_executor.workers.SubprocessWorker
                _recursive_: false
              runner:
                _target_: tpen_executor.runners.TPENCommand
                _recursive_: false
                config: {json.dumps(str(config))}
              evidence:
                _target_: tpen_executor.runners.TPENEvidence
                _recursive_: false
            """
        ),
        encoding="utf-8",
    )
    paths = {
        "input": tmp_path / "executor_input.json",
        "runs": tmp_path / "configured-runs.json",
        "allocation": tmp_path / "allocation.json",
        "worker": worker,
    }
    paths["input"].write_bytes(group.encode())
    paths["runs"].write_bytes(ConfiguredRunTable(group.identity, (ConfiguredRun("train:seed-0000", str(config)),)).encode())
    paths["allocation"].write_bytes(allocation.encode())
    return paths


def _cli(paths: dict[str, Path], store: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = dict(PYTHONPATH=str(SRC), **{key: value for key, value in {}.items()})
    return subprocess.run(
        [sys.executable, "-m", "tpen_executor", "dispatch", "--input", str(paths["input"]), "--runs", str(paths["runs"]), "--allocation", str(paths["allocation"]), "--worker-config", str(paths["worker"]), "--store", str(store), *extra],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a1_fake_core_attached_cli_publishes_manifest(tmp_path: Path) -> None:
    pytest.importorskip("parsl")
    paths = _inputs(tmp_path)
    result = _cli(paths, tmp_path / "store", "--dispatcher", "parsl")
    assert result.returncode == 0, result.stderr
    manifest = json.loads(result.stdout)
    assert manifest["schema"] == "tpen-executor/run-manifest-v2"
    assert manifest["segments"][0]["task_id"] == "train:seed-0000"


@pytest.mark.parametrize("mutation", ["diagnostic", "group", "order"])
def test_a2_named_admission_controls_create_no_store(tmp_path: Path, mutation: str) -> None:
    paths = _inputs(tmp_path)
    if mutation == "diagnostic":
        paths["input"].with_name("executor_input_diagnostics.json").write_text("{}", encoding="utf-8")
    elif mutation == "group":
        raw = json.loads(paths["runs"].read_text())
        raw["group_digest"] = "0" * 64
        paths["runs"].write_text(json.dumps(raw), encoding="utf-8")
    else:
        raw = json.loads(paths["runs"].read_text())
        raw["runs"][0]["id"] = "other"
        paths["runs"].write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(composition.CLIRefusal):
        composition.admit(paths["input"], tmp_path / "store", paths["allocation"], paths["worker"], paths["runs"])
    assert not (tmp_path / "store").exists()


def test_a3_reader_rejects_empty_committed_chain_after_zero_exit(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, evidence=False)
    result = _cli(paths, tmp_path / "store")
    assert result.returncode == 0
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from experiments.toolkit.attached_activation import write_activation_receipt
    from experiments.toolkit.run_results import ManifestRefusal
    with pytest.raises(ManifestRefusal, match="no committed|reader chains"):
        write_activation_receipt(
            tmp_path / "store" / "run-manifest.json",
            tmp_path / "receipt.json",
            expected_task_ids=("train:seed-0000",),
        )


def test_a4_allocation_context_preserves_named_binding_facts() -> None:
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from experiments.toolkit.attached_activation import allocation_context_payload

    context = SimpleNamespace(
        allocation_id="alloc",
        visibility_variable="CUDA_VISIBLE_DEVICES",
        visibility_values=("0",),
        run_root="/runs",
        deadline=123.5,
        deadline_env_var="END",
        deadline_guard_min=1,
        environment={"OMP_NUM_THREADS": "4"},
        nodes_per_block=None,
    )
    snapshot = allocation_context_payload(
        context,
        allocation_id="alloc",
        backend_job_id="job",
        observed_environment={"HOST": "n0"},
        observed_hostname="n0",
    )
    assert snapshot["context"]["deadline"] == 123.5
    assert snapshot["context"]["environment"] == {"OMP_NUM_THREADS": "4"}
    assert snapshot["observed"]["hostname"] == "n0"

    allocation = ConfiguredAllocation(
        AllocationId("alloc"),
        BackendJobId("job"),
        {"binding": {"environment": {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4"}, "node": "n0"}, "deadline": "d"},
    )
    binding = binding_from_allocation(allocation)
    assert dict(binding.env) == {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "4"}
    assert binding.node == "n0"
    assert allocation.facts["deadline"] == "d"


def test_a5_two_fresh_stores_have_disjoint_attempt_namespaces(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    first = _cli(paths, tmp_path / "store-1")
    second = _cli(paths, tmp_path / "store-2")
    assert first.returncode == second.returncode == 0
    left = json.loads(first.stdout)["attempts"][0]["attempt_id"]
    right = json.loads(second.stdout)["attempts"][0]["attempt_id"]
    assert left != right
    assert (tmp_path / "store-1" / "run-manifest.json").is_file()
    assert (tmp_path / "store-2" / "run-manifest.json").is_file()
    first_output = Path(json.loads(first.stdout)["attempts"][0]["output_dir"])
    second_output = Path(json.loads(second.stdout)["attempts"][0]["output_dir"])
    first_identity = json.loads((first_output / "core" / "process-identity.json").read_text())
    second_identity = json.loads((second_output / "core" / "process-identity.json").read_text())
    assert first_identity["nonce"] != second_identity["nonce"]
    assert first_output != second_output
