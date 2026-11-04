"""Independent P5b review probes; execute only in a cluster allocation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import sys
import time

import pytest

from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.runners import CoreRunEvidence, EvidenceUnavailable, TPENCommand, TPENEvidence
from tpen_executor.workers import SubprocessWorker


def _science_environment():
    python = os.environ.get("P5B_REVIEW_SCIENCE_PYTHON")
    if python is None:
        pytest.skip("independent root --extra cpu environment required")
    return PreparedPythonEnvironment(python)


def test_r1_t0_real_core_module_help_prints_usage(tmp_path):
    out = tmp_path / "out"
    command = TPENCommand(str(tmp_path / "config.yaml"))
    outcome = SubprocessWorker().run(
        _science_environment(), (*command.args, "--help"), cwd=tmp_path,
        stdout_path=out, stderr_path=tmp_path / "err",
    )
    assert outcome.returncode == 0
    assert "--config" in out.read_text(), "rc=0 alone does not prove the CLI ran"


def test_r1_t0_real_core_missing_config_is_not_success(tmp_path):
    command = TPENCommand(str(tmp_path / "missing-config.yaml"))
    outcome = SubprocessWorker().run(
        _science_environment(), command.args, cwd=tmp_path,
        stdout_path=tmp_path / "out", stderr_path=tmp_path / "err",
    )
    assert outcome.returncode != 0, "missing config must not be reported as a successful process"


def _payload(verdict="completed"):
    return {
        "schema": "tpen.run-evidence/v1", "run_dir": "/run", "verdict": verdict,
        "run_status": verdict,
        "checkpoint": None if verdict == "refused" else {
            "checkpoint_dir": "/run/checkpoints/c", "schema_version": "v2",
            "next_iteration": 2, "completed_updates": 1,
        },
        "refusal": {"code": "missing-run-dir", "reason": "absent"} if verdict == "refused" else None,
    }


def _read(tmp_path, text):
    package = tmp_path / "tpen"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "inspection.py").write_text(
        "import os\nprint(os.environ['REVIEW_EVIDENCE'])\n", encoding="utf-8"
    )
    environment = PreparedPythonEnvironment(sys.executable, env={
        "PYTHONPATH": str(tmp_path), "REVIEW_EVIDENCE": text,
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return TPENEvidence().read(
        SubprocessWorker(), environment, "/run", cwd=tmp_path,
        stdout_path=tmp_path / "out", stderr_path=tmp_path / "err",
    )


@pytest.mark.parametrize("field,verdict,key,value", [
    ("checkpoint", "completed", "next_iteration", 999),
    ("refusal", "refused", "code", "unknown"),
])
def test_r1_t1_decoded_nested_evidence_is_immutable(field, verdict, key, value):
    evidence = CoreRunEvidence.from_mapping(_payload(verdict))
    with pytest.raises((TypeError, AttributeError)):
        getattr(evidence, field)[key] = value


@pytest.mark.parametrize("verdict", ["completed", "refused"])
def test_r1_t1_evidence_hash_matches_mapping_equality(verdict):
    first = CoreRunEvidence.from_mapping(_payload(verdict))
    value = _payload(verdict)
    for key in ("checkpoint", "refusal"):
        if value[key] is not None:
            value[key] = dict(reversed(tuple(value[key].items())))
    second = CoreRunEvidence.from_mapping(value)
    assert first == second
    assert hash(first) == hash(second)


@pytest.mark.parametrize("nested", [False, True], ids=["duplicate-verdict", "duplicate-refusal-code"])
def test_r1_t2_duplicate_json_members_fail_closed(tmp_path, nested):
    if nested:
        text = json.dumps(_payload("refused")).replace(
            '"code": "missing-run-dir"', '"code": "unknown", "code": "missing-run-dir"'
        )
    else:
        text = json.dumps(_payload()).replace(
            '"verdict": "completed"', '"verdict": "failed", "verdict": "completed"'
        )
    with pytest.raises(EvidenceUnavailable):
        _read(tmp_path, text)


def test_r1_t3_excessive_json_nesting_is_typed_unavailable(tmp_path):
    with pytest.raises(EvidenceUnavailable):
        _read(tmp_path, "[" * 1500 + "0" + "]" * 1500)


@pytest.mark.parametrize("failure", ["interpreter", "cwd", "log-parent"])
def test_r1_t3_inspection_transport_oserrors_are_typed_unavailable(tmp_path, failure):
    python = str(tmp_path / "missing-python") if failure == "interpreter" else sys.executable
    cwd = tmp_path / "missing-cwd" if failure == "cwd" else tmp_path
    stdout = tmp_path / "missing-parent" / "out" if failure == "log-parent" else tmp_path / "out"
    with pytest.raises(EvidenceUnavailable):
        TPENEvidence().read(
            SubprocessWorker(), PreparedPythonEnvironment(python), "/run", cwd=cwd,
            stdout_path=stdout, stderr_path=tmp_path / "err",
        )


@pytest.mark.parametrize("during_setup", [False, True], ids=["science-grandchild", "setup-child"])
def test_r1_t4_timeout_stops_descendants_before_return(tmp_path, during_setup):
    marker = tmp_path / "late-write"
    ready = tmp_path / "ready"
    child = tmp_path / "descendant.py"
    child.write_text(
        "import pathlib,sys,time\ntime.sleep(2.5)\npathlib.Path(sys.argv[1]).write_text('still running')\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import pathlib,subprocess,sys,time\n"
        "subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]])\n"
        "pathlib.Path(sys.argv[3]).write_text('spawned')\n"
        "time.sleep(4.5)\n",
        encoding="utf-8",
    )
    args = (str(parent), str(child), str(marker), str(ready))
    if during_setup:
        setup = tmp_path / "setup.sh"
        setup.write_text(shlex.join((sys.executable, *args)) + "\n", encoding="utf-8")
        environment = PreparedPythonEnvironment(sys.executable, SourceScript(str(setup)))
        args = ("-c", "pass")
    else:
        environment = PreparedPythonEnvironment(sys.executable)
    outcome = SubprocessWorker().run(
        environment, args, cwd=tmp_path, stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err", timeout=1.5,
    )
    assert outcome.timed_out
    assert ready.is_file(), "probe must have spawned the descendant before timeout"
    # Both bounded probe processes have naturally exited by this deadline even
    # on the defective implementation; no indefinite process is left behind.
    time.sleep(3.5)
    assert not marker.exists(), "a descendant wrote after the worker returned timed_out=True"


def test_r1_t5_command_rejects_scalar_override_string():
    with pytest.raises(TypeError):
        TPENCommand("/config.yaml", "run.n_steps=20")


@pytest.mark.parametrize("verdict", ["completed", "incomplete", "failed", "refused"])
def test_r1_t6_real_inspection_child_decodes_every_verdict(tmp_path, verdict):
    evidence = _read(tmp_path, json.dumps(_payload(verdict)))
    assert evidence.verdict == verdict
    assert evidence.run_status == verdict


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("checkpoint_dir", None),
    ("next_iteration", True), ("completed_updates", -1),
])
def test_r1_t6_invalid_nested_checkpoint_fields_are_rejected(field, value):
    payload = _payload()
    payload["checkpoint"][field] = value
    with pytest.raises(EvidenceUnavailable):
        CoreRunEvidence.from_mapping(payload)


def test_r1_t6_environment_overlay_wins_without_mutating_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("P5B_REVIEW_OVERLAY", "owner")
    environment = PreparedPythonEnvironment(sys.executable, env={"P5B_REVIEW_OVERLAY": "child"})
    out = tmp_path / "out"
    outcome = SubprocessWorker().run(
        environment, ("-c", "import os;print(os.environ['P5B_REVIEW_OVERLAY'])"),
        cwd=tmp_path, stdout_path=out, stderr_path=tmp_path / "err",
    )
    assert outcome.returncode == 0
    assert out.read_text().strip() == "child"
    assert os.environ["P5B_REVIEW_OVERLAY"] == "owner"
