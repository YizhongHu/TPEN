"""P5b subprocess and public evidence acceptance tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

from tpen_executor.configuration import StagedHydra
from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.runners import CoreRunEvidence, EvidenceUnavailable, TPENCommand, TPENEvidence
from tpen_executor.workers import ProcessOutcome, SubprocessWorker


def _science(tmp_path: Path) -> tuple[PreparedPythonEnvironment, Path]:
    package = tmp_path / "tpen"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        "import os,sys,time\n"
        "if os.environ.get('SLEEP'): time.sleep(float(os.environ['SLEEP']))\n"
        "print('ARGS=' + repr(sys.argv[1:]))\n"
        "print('MODE=' + os.environ.get('MODE',''))\n"
        "print('SETUP=' + os.environ.get('SETUP_WAS_SOURCED',''))\n"
        "sys.exit(int(os.environ.get('EXIT_CODE','0')))\n",
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
        ("-m", "tpen.run", "--value", "space ' quote $ value"),
        cwd=tmp_path,
        stdout_path=out,
        stderr_path=err,
    )
    assert outcome.argv == (sys.executable, "-m", "tpen.run", "--value", "space ' quote $ value")
    assert outcome.returncode == 0 and not outcome.timed_out
    assert "ARGS=['--value', \"space ' quote $ value\"]" in out.read_text()
    assert "MODE=overlay" in out.read_text()
    assert "SETUP=yes" in out.read_text()
    assert set(tmp_path.iterdir()) >= {out, err}


def test_tpen_command_has_exact_public_run_argv() -> None:
    command = TPENCommand("/tmp/config.yaml", ("run.run_id=quoted value",))
    assert command.args == ("-m", "tpen.run", "--config", "/tmp/config.yaml", "run.run_id=quoted value")


def test_subprocess_worker_real_timeout_records_returncode_and_logs(tmp_path: Path) -> None:
    environment, _ = _science(tmp_path)
    environment = PreparedPythonEnvironment(environment.python, environment.setup, {**environment.env, "SLEEP": "5"})
    out, err = tmp_path / "timeout.out", tmp_path / "timeout.err"
    outcome = SubprocessWorker().run(environment, ("-m", "tpen.run"), cwd=tmp_path, stdout_path=out, stderr_path=err, timeout=0.5)
    assert outcome.timed_out is True
    assert outcome.returncode is not None
    assert out.is_file() and err.is_file()


def test_subprocess_worker_real_nonzero_returncode_is_recorded(tmp_path: Path) -> None:
    environment, _ = _science(tmp_path)
    environment = PreparedPythonEnvironment(environment.python, environment.setup, {**environment.env, "EXIT_CODE": "3"})
    out, err = tmp_path / "nonzero.out", tmp_path / "nonzero.err"
    outcome = SubprocessWorker().run(environment, ("-m", "tpen.run"), cwd=tmp_path, stdout_path=out, stderr_path=err)
    assert outcome.timed_out is False and outcome.returncode == 3
    assert out.is_file() and err.is_file()


def test_subprocess_worker_only_writes_named_logs(tmp_path: Path) -> None:
    environment, _ = _science(tmp_path)
    out, err = tmp_path / "named.out", tmp_path / "named.err"
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    SubprocessWorker().run(environment, ("-m", "tpen.run"), cwd=tmp_path, stdout_path=out, stderr_path=err)
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert after - before == {Path("named.out"), Path("named.err")}


def test_tpen_command_launches_stub_with_exact_args(tmp_path: Path) -> None:
    environment, _ = _science(tmp_path)
    (tmp_path / "tpen" / "run.py").write_text(
        "import json,sys\nprint(json.dumps(sys.argv[1:]))\n", encoding="utf-8"
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
        "run_status": "completed", "checkpoint": {"checkpoint_dir": "/c", "schema_version": "v2", "next_iteration": 2, "completed_updates": 1}, "refusal": None,
    }
    environment = PreparedPythonEnvironment(environment.python, environment.setup, {**environment.env, "EVIDENCE": json.dumps(payload)})
    out, err = tmp_path / "evidence.out", tmp_path / "evidence.err"
    original_open = Path.open

    def guarded_open(self: Path, *args, **kwargs):
        if run_dir in self.parents or self == run_dir:
            raise AssertionError("evidence client opened run_dir directly")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    evidence = TPENEvidence().read(SubprocessWorker(), environment, run_dir, cwd=tmp_path, stdout_path=out, stderr_path=err)
    assert isinstance(evidence, CoreRunEvidence) and evidence.verdict == "completed"


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
