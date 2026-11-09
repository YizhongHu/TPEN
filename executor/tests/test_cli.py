"""P9c public CLI acceptance probes A-1 through A-7."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from tpen_executor import composition
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.tasks import ConfiguredRun, ConfiguredRunTable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _group(task_ids: tuple[str, ...]) -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    return ReadyGroup(
        "cli-group",
        tuple(TaskSpec(task_id, task_id, program, digest(task_id), 1, "steps") for task_id in task_ids),
        digest("cli-plan"),
    )


def _stub_package(path: Path) -> Path:
    package = path / "stub" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path

            def main(argv):
                config = Path(argv[argv.index("--config") + 1])
                mode = config.read_text(encoding="utf-8")
                values = dict(item.split("=", 1) for item in argv if "=" in item)
                run_dir = Path(values["run.root"]) / values["run.run_id"]
                if mode == "fail":
                    return 7
                run_dir.mkdir(parents=True, exist_ok=True)
                print(json.dumps({"argv": argv}))
                (run_dir / "evidence.json").write_text(json.dumps({
                    "schema": "tpen.run-evidence/v1",
                    "run_dir": str(run_dir),
                    "verdict": "completed",
                    "run_status": "completed",
                    "checkpoint": {"checkpoint_dir": str(run_dir / "checkpoint"), "schema_version": 1, "next_iteration": 1, "completed_updates": 1},
                    "refusal": None,
                }), encoding="utf-8")
                return 0
            """
        ),
        encoding="utf-8",
    )
    (package / "inspection.py").write_text(
        "import json, sys\nfrom pathlib import Path\nprint(Path(sys.argv[1], 'evidence.json').read_text())\n",
        encoding="utf-8",
    )
    return package.parent


def _inputs(tmp_path: Path, task_ids: tuple[str, ...] = ("task:one",), modes: dict[str, str] | None = None) -> dict[str, Path | ReadyGroup]:
    modes = {} if modes is None else modes
    group = _group(task_ids)
    stub = _stub_package(tmp_path)
    config_paths = {}
    for task_id in task_ids:
        config = tmp_path / f"{task_id.replace(':', '_')}.yaml"
        config.write_text(modes.get(task_id, "ok"), encoding="utf-8")
        config_paths[task_id] = config
    allocation = ConfiguredAllocation(AllocationId("cli-allocation"), BackendJobId("cli-job"), {})
    worker = textwrap.dedent(
        f"""
        worker:
          _target_: tpen_executor.programs.WorkerProgram
          _recursive_: false
          environment:
            _target_: tpen_executor.environments.PreparedPythonEnvironment
            _recursive_: false
            python: {json.dumps(sys.executable)}
            env:
              PYTHONPATH: {json.dumps(str(stub))}
          process:
            _target_: tpen_executor.workers.SubprocessWorker
            _recursive_: false
          runner:
            _target_: tpen_executor.runners.TPENCommand
            _recursive_: false
            config: {json.dumps(str(config_paths[task_ids[0]]))}
          evidence:
            _target_: tpen_executor.runners.TPENEvidence
            _recursive_: false
        """
    )
    paths = {
        "input": tmp_path / "executor_input.json",
        "allocation": tmp_path / "allocation.json",
        "runs": tmp_path / "runs.json",
        "worker": tmp_path / "worker.yaml",
        "store": tmp_path / "store",
    }
    paths["input"].write_bytes(group.encode())  # type: ignore[union-attr]
    paths["allocation"].write_bytes(allocation.encode())
    paths["runs"].write_bytes(ConfiguredRunTable(group.identity, tuple(ConfiguredRun(task_id, str(config_paths[task_id])) for task_id in task_ids)).encode())  # type: ignore[union-attr]
    paths["worker"].write_text(worker, encoding="utf-8")
    return {**paths, "group": group}


def _run_cli(paths: dict[str, Path | ReadyGroup], *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    command = [sys.executable, "-m", "tpen_executor", "dispatch", "--input", str(paths["input"]), "--store", str(paths["store"]), "--allocation", str(paths["allocation"]), "--worker-config", str(paths["worker"]), "--runs", str(paths["runs"]), *extra]
    return subprocess.run(command, env=env, capture_output=True, text=True)


def test_a1_help_and_typed_usage_refusal(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    help_result = subprocess.run([sys.executable, "-m", "tpen_executor", "--help"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "dispatch" in help_result.stdout
    refusal = subprocess.run([sys.executable, "-m", "tpen_executor", "dispatch", "--bad"], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert refusal.returncode == 2
    assert json.loads(refusal.stderr)["schema"] == "tpen-executor/cli-error-v1"
    assert refusal.stdout == ""


def test_a2_local_dispatch_publishes_v2_manifest_and_preserves_colon_id(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, ("he-cutover:02_train:seed-000", "he-cutover:02_train:seed-001"))
    result = _run_cli(paths)
    assert result.returncode == 0
    manifest = json.loads(result.stdout)
    assert manifest["schema"] == "tpen-executor/run-manifest-v2"
    assert [item["task_id"] for item in manifest["attempts"]] == ["he-cutover:02_train:seed-000", "he-cutover:02_train:seed-001"]
    assert (Path(paths["store"]) / "run-manifest.json").exists()


@pytest.mark.parametrize("mutation", ["diagnostic", "duplicate-json"])
def test_a3_named_admission_refusals_do_not_create_store(tmp_path: Path, mutation: str) -> None:
    paths = _inputs(tmp_path)
    if mutation == "diagnostic":
        Path(paths["input"]).with_name("executor_input_diagnostics.json").write_text("{}", encoding="utf-8")
    else:
        Path(paths["input"]).write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    result = _run_cli(paths)
    assert result.returncode == 2
    assert result.stdout == ""
    error = json.loads(result.stderr)
    assert error["schema"] == "tpen-executor/cli-error-v1"
    if mutation == "diagnostic":
        assert error["code"] == "diagnostic-present"
    assert not Path(paths["store"]).exists()


def test_a3_empty_runs_refuses_before_store(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    runs = json.loads(Path(paths["runs"]).read_text(encoding="utf-8"))
    runs["runs"] = []
    Path(paths["runs"]).write_text(json.dumps(runs), encoding="utf-8")
    result = _run_cli(paths)
    assert result.returncode == 2
    assert result.stdout == ""
    error = json.loads(result.stderr)
    assert error["schema"] == "tpen-executor/cli-error-v1"
    assert error["code"] == "invalid-runs"
    assert not Path(paths["store"]).exists()


def test_a3_wrong_group_digest_refuses_before_store(tmp_path: Path) -> None:
    paths = _inputs(tmp_path, ("task:first", "task:last"))
    runs = json.loads(Path(paths["runs"]).read_text(encoding="utf-8"))
    wrong_digest = "0" * 64
    if runs["group_digest"] == wrong_digest:
        wrong_digest = "1" * 64
    runs["group_digest"] = wrong_digest
    Path(paths["runs"]).write_text(json.dumps(runs), encoding="utf-8")
    result = _run_cli(paths)
    assert result.returncode == 2
    assert result.stdout == ""
    error = json.loads(result.stderr)
    assert error["schema"] == "tpen-executor/cli-error-v1"
    assert error["code"] == "group-digest-mismatch"
    assert not Path(paths["store"]).exists()


def test_a4_local_imports_no_parsl_module(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run([sys.executable, "-c", "import sys; import tpen_executor.cli; print('parsl' in sys.modules)"], cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_a5_executor_sources_have_no_root_imports_or_module_run_policy_violation() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in (SRC / "tpen_executor").rglob("*.py"))
    assert "import experiments" not in source
    assert "import tpen" not in source
    assert "python -m tpen.run" not in source


def test_a6_failed_child_is_operational_failure_with_preserved_manifest(tmp_path: Path) -> None:
    task_id = "failed:task"
    paths = _inputs(tmp_path, (task_id,), {task_id: "fail"})
    result = _run_cli(paths)
    assert result.returncode == 3
    manifest = json.loads(result.stdout)
    assert manifest["attempts"][0]["outcome_receipt"]["status"] == "failed"
    assert (Path(paths["store"]) / "run-manifest.json").exists()


def test_a7_existing_and_symlink_stores_refuse_without_reuse(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    first = _run_cli(paths)
    assert first.returncode == 0
    second = _run_cli(paths)
    assert second.returncode == 2
    target = tmp_path / "target-store"
    target.mkdir()
    symlink = tmp_path / "symlink-store"
    symlink.symlink_to(target, target_is_directory=True)
    paths["store"] = symlink
    third = _run_cli(paths)
    assert third.returncode == 2


def test_f1_wrong_worker_environment_refuses_before_store_reservation(tmp_path: Path) -> None:
    """A malformed prepared-environment slot is an admission refusal."""
    paths = _inputs(tmp_path)
    worker = Path(paths["worker"])
    worker.write_text(
        worker.read_text(encoding="utf-8").replace(
            "tpen_executor.environments.PreparedPythonEnvironment",
            "tpen_executor.programs.DummyWorker",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(composition.CLIRefusal, match="PreparedPythonEnvironment"):
        composition.admit(paths["input"], paths["store"], paths["allocation"], worker, paths["runs"])
    assert not Path(paths["store"]).exists()


def test_f1_interpolated_store_refuses_before_store_reservation(tmp_path: Path) -> None:
    """A literal unresolved store interpolation is rejected before mkdir."""
    paths = _inputs(tmp_path)
    store = tmp_path / "${unresolved}"
    with pytest.raises(composition.CLIRefusal, match="interpolation"):
        composition.admit(paths["input"], store, paths["allocation"], paths["worker"], paths["runs"])
    assert not store.exists()


def test_f2_failed_manifest_write_never_exposes_final_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A write fault leaves no partial bytes at the published manifest name."""
    target = tmp_path / "run-manifest.json"
    real_fdopen = os.fdopen

    class FaultWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, payload):
            self.stream.write(payload[:1])
            self.stream.flush()
            raise OSError("committed control write failure")

    monkeypatch.setattr(os, "fdopen", lambda fd, mode: FaultWriter(real_fdopen(fd, mode)))
    with pytest.raises(OSError, match="committed control"):
        composition._publish_new(target, b'{"complete":true}\n')
    assert not target.exists()


def test_f8_deep_known_program_field_is_typed_admission_refusal(tmp_path: Path) -> None:
    """Known program payload depth cannot escape record admission as recursion."""
    paths = _inputs(tmp_path)
    raw = Path(paths["input"]).read_text(encoding="utf-8")
    nested = "[" * 1500 + "0" + "]" * 1500
    old_digest = hashlib.sha256(b"{}").hexdigest()
    new_digest = hashlib.sha256(nested.encode()).hexdigest()
    raw = raw.replace('"config_ref":{}', '"config_ref":' + nested, 1)
    raw = raw.replace('"content_digest":"' + old_digest + '"', '"content_digest":"' + new_digest + '"', 1)
    Path(paths["input"]).write_text(raw, encoding="utf-8")
    with pytest.raises(composition.CLIRefusal) as refused:
        composition.admit(paths["input"], paths["store"], paths["allocation"], paths["worker"], paths["runs"])
    assert refused.value.code in {"invalid-json", "invalid-ready-group", "admission-depth"}
    assert not Path(paths["store"]).exists()
