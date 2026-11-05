"""P5b round-2 probes for accepted immutability and documentation fixes."""

from __future__ import annotations

import json
import sys

import pytest

from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.runners import CoreRunEvidence
from tpen_executor.workers import SubprocessWorker


def _fields(verdict):
    return {
        "schema": "tpen.run-evidence/v1", "run_dir": "/run", "verdict": verdict,
        "run_status": "completed" if verdict == "completed" else None,
        "checkpoint": {
            "checkpoint_dir": "/run/checkpoints/c", "schema_version": "v2",
            "next_iteration": 2, "completed_updates": 1,
        } if verdict == "completed" else None,
        "refusal": {"code": "missing-run-dir", "reason": "absent"} if verdict == "refused" else None,
    }


@pytest.mark.parametrize("verdict,field,key,value", [
    ("completed", "checkpoint", "next_iteration", 999),
    ("refused", "refusal", "code", "unknown"),
])
def test_r2_t1_direct_evidence_construction_freezes_nested_values(verdict, field, key, value):
    evidence = CoreRunEvidence(**_fields(verdict))
    with pytest.raises((TypeError, AttributeError)):
        getattr(evidence, field)[key] = value


@pytest.mark.parametrize("verdict,field,key,value", [
    ("completed", "checkpoint", "next_iteration", 999),
    ("refused", "refusal", "reason", "changed"),
])
def test_r2_t2_caller_mapping_mutation_cannot_change_evidence_hash(verdict, field, key, value):
    fields = _fields(verdict)
    evidence = CoreRunEvidence(**fields)
    original_hash = hash(evidence)
    original_value = getattr(evidence, field)[key]
    fields[field][key] = value
    assert hash(evidence) == original_hash, "caller mutation changed a frozen record's hash"
    assert getattr(evidence, field)[key] == original_value


def test_r2_t3_setup_is_last_writer_after_owner_and_descriptor_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("P5B_R2_PRECEDENCE", "owner")
    setup = tmp_path / "setup.sh"
    setup.write_text("export P5B_R2_PRECEDENCE=setup\n", encoding="utf-8")
    environment = PreparedPythonEnvironment(
        sys.executable, SourceScript(str(setup)), {"P5B_R2_PRECEDENCE": "descriptor"}
    )
    stdout = tmp_path / "stdout"
    outcome = SubprocessWorker().run(
        environment,
        ("-c", "import json,os; print(json.dumps(os.environ['P5B_R2_PRECEDENCE']))"),
        cwd=tmp_path, stdout_path=stdout, stderr_path=tmp_path / "stderr",
    )
    assert outcome.returncode == 0
    assert json.loads(stdout.read_text()) == "setup"
