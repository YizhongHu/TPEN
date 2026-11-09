"""P5b subprocess and public evidence acceptance tests."""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import tpen_executor.workers as workers_module
from tpen_executor.configuration import StagedHydra
from tpen_executor.dispatch.contract import _child_context_keys
from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.runners import (
    CHILD_CONTEXT_KEYS_ENV,
    CONTEXT_ENV_KEY_PATTERN,
    CoreRunEvidence,
    EvidenceUnavailable,
    RUN_SHIM,
    TPENCommand,
    TPENEvidence,
)
from tpen_executor.workers import ProcessOutcome, SubprocessWorker


def _science(tmp_path: Path) -> tuple[PreparedPythonEnvironment, Path]:
    package = tmp_path / "tpen"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        "import os,sys,time\n"
        "def main(argv):\n"
        "    if os.environ.get('SLEEP'): time.sleep(float(os.environ['SLEEP']))\n"
        "    print('ARGS=' + repr(list(argv)))\n"
        "    print('MODE=' + os.environ.get('MODE',''))\n"
        "    print('SETUP=' + os.environ.get('SETUP_WAS_SOURCED',''))\n"
        "    return int(os.environ.get('EXIT_CODE','0'))\n",
        encoding="utf-8",
    )
    (package / "inspection.py").write_text(
        "import json,os,sys\n"
        "value=os.environ.get('EVIDENCE')\n"
        "print(value if value is not None else json.dumps({'schema':'tpen.run-evidence/v1','run_dir':sys.argv[1],'verdict':'refused','run_status':None,'checkpoint':None,'refusal':{'code':'missing-run-dir','reason':'stub'}}))\n",
        encoding="utf-8",
    )
    setup = tmp_path / "setup.sh"
    setup.write_text("export SETUP_WAS_SOURCED=yes\n", encoding="utf-8")
    return PreparedPythonEnvironment(
        sys.executable,
        SourceScript(str(setup)),
        {"PYTHONPATH": str(tmp_path), "MODE": "overlay", "PYTHONDONTWRITEBYTECODE": "1"},
    ), package


def test_subprocess_worker_preserves_setup_env_and_argument_boundaries(tmp_path: Path) -> None:
    environment, _ = _science(tmp_path)
    out, err = tmp_path / "out", tmp_path / "err"
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, "--value", "space ' quote $ value"),
        cwd=tmp_path,
        stdout_path=out,
        stderr_path=err,
    )
    assert outcome.argv == (sys.executable, "-c", RUN_SHIM, "--value", "space ' quote $ value")
    assert outcome.returncode == 0 and not outcome.timed_out
    assert "ARGS=['--value', \"space ' quote $ value\"]" in out.read_text()
    assert "MODE=overlay" in out.read_text()
    assert "SETUP=yes" in out.read_text()
    assert set(tmp_path.iterdir()) >= {out, err}


def test_subprocess_worker_real_child_preserves_no_setup_argument_boundaries(tmp_path: Path) -> None:
    child = tmp_path / "child.py"
    child.write_text("import json,sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8")
    environment = PreparedPythonEnvironment(sys.executable, None, {"PYTHONDONTWRITEBYTECODE": "1"})
    args = (str(child), "space value", 'quote"value', "$HOME")
    out, err = tmp_path / "args.out", tmp_path / "args.err"

    outcome = SubprocessWorker().run(environment, args, cwd=tmp_path, stdout_path=out, stderr_path=err)

    assert outcome.returncode == 0
    assert json.loads(out.read_text(encoding="utf-8")) == list(args[1:])


def test_tpen_command_has_exact_public_run_argv() -> None:
    command = TPENCommand("/tmp/config.yaml", ("run.run_id=quoted value",))
    assert command.args == ("-c", command.args[1], "--config", "/tmp/config.yaml", "run.run_id=quoted value")


def test_subprocess_worker_real_timeout_records_returncode_and_logs(tmp_path: Path) -> None:
    environment, _ = _science(tmp_path)
    environment = PreparedPythonEnvironment(environment.python, environment.setup, {**environment.env, "SLEEP": "5"})
    out, err = tmp_path / "timeout.out", tmp_path / "timeout.err"
    outcome = SubprocessWorker().run(environment, ("-c", RUN_SHIM), cwd=tmp_path, stdout_path=out, stderr_path=err, timeout=0.5)
    assert outcome.timed_out is True
    assert outcome.returncode is not None
    assert out.is_file() and err.is_file()


def test_subprocess_worker_real_nonzero_returncode_is_recorded(tmp_path: Path) -> None:
    environment, _ = _science(tmp_path)
    environment = PreparedPythonEnvironment(environment.python, environment.setup, {**environment.env, "EXIT_CODE": "3"})
    out, err = tmp_path / "nonzero.out", tmp_path / "nonzero.err"
    outcome = SubprocessWorker().run(environment, ("-c", RUN_SHIM), cwd=tmp_path, stdout_path=out, stderr_path=err)
    assert outcome.timed_out is False and outcome.returncode == 3
    assert out.is_file() and err.is_file()


def test_subprocess_worker_only_writes_named_logs(tmp_path: Path) -> None:
    environment, _ = _science(tmp_path)
    out, err = tmp_path / "named.out", tmp_path / "named.err"
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    SubprocessWorker().run(environment, ("-c", RUN_SHIM), cwd=tmp_path, stdout_path=out, stderr_path=err)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert after - before == {Path("named.out"), Path("named.err")}


def test_run_shim_observation_excludes_unallowlisted_environment(tmp_path: Path) -> None:
    environment, _ = _science(tmp_path)
    environment = PreparedPythonEnvironment(
        environment.python,
        environment.setup,
        {
            **environment.env,
            "CUDA_VISIBLE_DEVICES": "7",
            "OMP_NUM_THREADS": "4",
            "TPEN_CTX_CONTEXT_VALUE": "declared",
            "P10_TEST_SECRET": "xyz",
            CHILD_CONTEXT_KEYS_ENV: json.dumps(["TPEN_CTX_CONTEXT_VALUE", "P10_TEST_SECRET"]),
        },
    )
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, f"run.root={tmp_path}"),
        cwd=tmp_path,
        stdout_path=tmp_path / "observation.out",
        stderr_path=tmp_path / "observation.err",
    )
    assert outcome.returncode == 0
    observation = json.loads((tmp_path / "executor-child-observation.json").read_text(encoding="utf-8"))
    assert observation["environment"] == {
        "CUDA_VISIBLE_DEVICES": "7",
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "OMP_NUM_THREADS": "4",
        "TPEN_CTX_CONTEXT_VALUE": "declared",
    }
    assert "P10_TEST_SECRET" not in observation["environment"]


def test_run_shim_observation_records_spawned_child_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment, _ = _science(tmp_path)
    spawned_pids: list[int] = []
    real_popen = workers_module.subprocess.Popen

    def recording_popen(*args: object, **kwargs: object) -> object:
        process = real_popen(*args, **kwargs)
        spawned_pids.append(process.pid)
        return process

    monkeypatch.setattr(workers_module.subprocess, "Popen", recording_popen)
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, f"run.root={tmp_path}"),
        cwd=tmp_path,
        stdout_path=tmp_path / "observation.out",
        stderr_path=tmp_path / "observation.err",
    )

    assert outcome.returncode == 0
    assert len(spawned_pids) == 1
    observation = json.loads((tmp_path / "executor-child-observation.json").read_text(encoding="utf-8"))
    assert observation["pid"] == spawned_pids[0]


def test_context_key_pattern_is_the_executor_shim_pattern() -> None:
    """The source constant is available for the cross-project contract test."""

    assert CONTEXT_ENV_KEY_PATTERN == r"^(CUDA_VISIBLE_DEVICES|TPEN_CTX_[A-Z0-9_]{1,64})$"


def test_child_context_keys_preserve_default_and_declared_context_names() -> None:
    allocation = SimpleNamespace(
        facts={
            "attached_context": {
                "context": {
                    "environment": {"TPEN_CTX_THREAD_SETTING": "4"},
                    "visibility_variable": "CUDA_VISIBLE_DEVICES",
                }
            }
        }
    )
    assert _child_context_keys(allocation) == (
        "CUDA_VISIBLE_DEVICES",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "TPEN_CTX_THREAD_SETTING",
    )


def test_tpen_command_launches_stub_with_exact_args(tmp_path: Path) -> None:
    environment, _ = _science(tmp_path)
    (tmp_path / "tpen" / "run.py").write_text(
        "import json\n"
        "def main(argv):\n"
        "    print(json.dumps(list(argv)))\n"
        "    return 0\n",
        encoding="utf-8",
    )
    config = str(tmp_path / "config.yaml")
    command = TPENCommand(config, ("a=1", "b=x y"))
    out, err = tmp_path / "command.out", tmp_path / "command.err"
    outcome = SubprocessWorker().run(environment, command.args, cwd=tmp_path, stdout_path=out, stderr_path=err)
    assert outcome.returncode == 0
    assert json.loads(out.read_text(encoding="utf-8")) == ["--config", config, "a=1", "b=x y"]


def test_packaged_worker_materializes_real_p5b_classes() -> None:
    staged = StagedHydra().compose_config()
    env = {"paths": {"control_python": "/prepared/control/python", "control_setup": "/prepared/control/setup.sh", "science_python": "/prepared/science/python", "science_setup": "/prepared/science/setup.sh"}}
    worker = staged.programs["program.dispatcher.worker"].materialize(env)
    assert isinstance(worker.process, SubprocessWorker)
    assert isinstance(worker.runner, TPENCommand)
    assert isinstance(worker.evidence, TPENEvidence)


def test_tpen_evidence_decodes_all_verdicts_and_never_reads_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment, _ = _science(tmp_path)
    run_dir = tmp_path / "run dir"
    run_dir.mkdir()
    payload = {
        "schema": "tpen.run-evidence/v1", "run_dir": str(run_dir), "verdict": "completed",
        "run_status": "completed", "checkpoint": {"checkpoint_dir": "/c", "schema_version": 2, "next_iteration": 2, "completed_updates": 1}, "refusal": None,
    }
    environment = PreparedPythonEnvironment(environment.python, environment.setup, {**environment.env, "EVIDENCE": json.dumps(payload)})
    out, err = tmp_path / "evidence.out", tmp_path / "evidence.err"
    hits: list[str] = []
    # Compute this before patching: abspath/normpath are lexical and do not
    # consult the filesystem, unlike Path.resolve().
    root = os.path.normpath(os.path.abspath(str(run_dir)))
    reentrant = False

    def under_run_dir(value: object) -> bool:
        if isinstance(value, int):
            # os.stat accepts file descriptors; they are not path evidence.
            return False
        try:
            candidate_value = os.fspath(value)
            if isinstance(candidate_value, bytes):
                candidate_value = os.fsdecode(candidate_value)
            candidate = os.path.normpath(os.path.abspath(candidate_value))
        except (AttributeError, TypeError, ValueError):
            return False
        return candidate == root or candidate.startswith(root + os.sep)

    def guarded(name: str, original):
        def wrapper(*args, **kwargs):
            nonlocal reentrant
            if reentrant:
                return original(*args, **kwargs)
            reentrant = True
            try:
                values = args[:1]
                if any(under_run_dir(value) for value in values):
                    hits.append(name)
                # Keep the original call inside the guard: e.g. os.stat can
                # itself call another patched filesystem wrapper.
                return original(*args, **kwargs)
            finally:
                reentrant = False

        return wrapper

    def guarded_path(name: str, original):
        def wrapper(self: Path, *args, **kwargs):
            nonlocal reentrant
            if reentrant:
                return original(self, *args, **kwargs)
            reentrant = True
            try:
                if under_run_dir(self):
                    hits.append(name)
                return original(self, *args, **kwargs)
            finally:
                reentrant = False

        return wrapper

    with monkeypatch.context() as patch:
        for module, name in (
            (builtins, "open"), (io, "open"), (os, "open"), (os, "stat"), (os, "lstat"),
            (os, "scandir"), (os, "listdir"), (os.path, "exists"), (os.path, "isfile"),
            (os.path, "isdir"),
        ):
            patch.setattr(module, name, guarded(f"{module.__name__}.{name}", getattr(module, name)))
        for name in ("stat", "exists", "is_file", "is_dir", "iterdir", "open", "read_text", "read_bytes"):
            patch.setattr(Path, name, guarded_path(f"Path.{name}", getattr(Path, name)))
        evidence = TPENEvidence().read(
            SubprocessWorker(), environment, run_dir, cwd=tmp_path, stdout_path=out, stderr_path=err
        )
    assert isinstance(evidence, CoreRunEvidence) and evidence.verdict == "completed"
    assert hits == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: {**x, "schema": "bad"},
        lambda x: {**x, "extra": 1},
        lambda x: {key: value for key, value in x.items() if key != "verdict"},
        lambda x: {**x, "run_status": 1},
        lambda x: {**x, "refusal": {"code": "unknown", "reason": "x"}, "verdict": "refused", "checkpoint": None},
    ],
    ids=["bad-schema", "extra-key", "missing-key", "bad-type", "unknown-refusal-code"],
)
def test_core_evidence_rejects_schema_key_type_and_refusal_mutants(mutate) -> None:
    base = {"schema": "tpen.run-evidence/v1", "run_dir": "/r", "verdict": "incomplete", "run_status": None, "checkpoint": None, "refusal": None}
    with pytest.raises(EvidenceUnavailable):
        CoreRunEvidence.from_mapping(mutate(base))


class _StubWorker:
    """Return a caller-controlled process outcome without launching a child."""

    def __init__(self, outcome: ProcessOutcome):
        self.outcome = outcome

    def run(self, *args, **kwargs) -> ProcessOutcome:
        return self.outcome


@pytest.mark.parametrize("content", ["", "not-json", "{} {}"], ids=["empty", "non-json", "multiple-objects"])
def test_evidence_rejects_empty_non_json_and_multiple_objects(tmp_path: Path, content: str) -> None:
    stdout, stderr = tmp_path / "stdout", tmp_path / "stderr"
    stdout.write_text(content, encoding="utf-8")
    outcome = ProcessOutcome((sys.executable,), 0, str(stdout), str(stderr), False)
    with pytest.raises(EvidenceUnavailable):
        TPENEvidence().read(_StubWorker(outcome), PreparedPythonEnvironment(sys.executable), "/run", stdout_path=stdout, stderr_path=stderr)


@pytest.mark.parametrize("outcome", [
    ProcessOutcome((sys.executable,), 1, "unused", "unused", False),
    ProcessOutcome((sys.executable,), -9, "unused", "unused", True),
], ids=["nonzero", "timeout"])
def test_evidence_rejects_nonzero_and_timeout(tmp_path: Path, outcome: ProcessOutcome) -> None:
    with pytest.raises(EvidenceUnavailable):
        TPENEvidence().read(_StubWorker(outcome), PreparedPythonEnvironment(sys.executable), "/run", stdout_path=tmp_path / "out", stderr_path=tmp_path / "err")


def test_executor_sources_have_no_forbidden_imports() -> None:
    root = Path(__file__).parents[1] / "src" / "tpen_executor"
    for path in (root / "workers.py", root / "runners.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert all(not (name == "tpen" or name.startswith("tpen.") or name == "experiments" or name.startswith("experiments.")) for name in names)
