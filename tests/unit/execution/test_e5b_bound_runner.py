"""Real in-process two-start evidence for the E5b configured worker."""

import json
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.unit.execution.test_formal_r2_e5 import FIXTURE, PrefixWalltimeTrap, _job_data, _prepare


def test_real_e5b_two_start_restore_keeps_component_hashes(monkeypatch, tmp_path):
    from tools import e5b_polaris_runner as runner
    from tpen.checkpoint.hashing import checkpoint_hashes
    from tpen.checkpoint.schema import read_manifest
    from tpen.execution import CheckpointEvidence

    root = tmp_path / "two-start"
    batch, _intent = _prepare(root)
    config_before = FIXTURE.read_bytes()
    original_sleep = time.sleep

    def walltime_wait(_seconds):
        raise PrefixWalltimeTrap()

    monkeypatch.setenv("PBS_JOBID", "hash-prefix")
    monkeypatch.setattr(runner.time, "sleep", walltime_wait)
    with pytest.raises(PrefixWalltimeTrap):
        runner.run_attempt(job_id="Job42", job_data=_job_data(root))
    evidence = json.loads((root / "checkpoint-evidence.json").read_text())
    checkpoint = CheckpointEvidence.from_mapping(evidence).ref
    assert checkpoint is not None
    assert checkpoint.next_iteration == 4 < batch.target_iterations
    manifest = read_manifest(checkpoint.checkpoint_dir, mode="train_resume")
    expected = checkpoint_hashes(OmegaConf.load(FIXTURE))
    for component in ("model_config", "optimizer_config", "trainer_config", "sampler_config", "hamiltonian_config"):
        assert manifest.hashes[component] == expected[component], component

    monkeypatch.setattr(runner.time, "sleep", original_sleep)
    monkeypatch.setenv("PBS_JOBID", "hash-successor")
    assert runner.run_attempt(job_id="Job42", job_data=_job_data(root)) == 0
    application = json.loads((root / "successor-application.json").read_text())
    assert application["successor_outcome"] == "completed"
    assert application["reference_outcome"] == "completed"
    assert application["successor_output"] != application["reference_output"]
    namespace = Path(application["successor_output"]).parent
    start = json.loads((namespace / "start.json").read_text())
    outcome = json.loads((namespace / "outcome.json").read_text())
    assert start["recovery_outcome"] == "recovery"
    assert start["next_iteration"] == checkpoint.next_iteration
    assert outcome["next_iteration"] == batch.target_iterations
    assert FIXTURE.read_bytes() == config_before


def test_e5b_unbound_config_is_terminal_before_real_runner(monkeypatch, tmp_path):
    from tools import e5b_polaris_runner as runner
    from tpen import run as core

    root = tmp_path / "unbound"
    _prepare(root)
    cfg = OmegaConf.load(FIXTURE)
    cfg.optimizer.lr = 0.02
    path = tmp_path / "changed.yaml"
    OmegaConf.save(cfg, path)
    data = _job_data(root)
    data["config_path"] = str(path)

    def forbidden(*args, **kwargs):
        pytest.fail("unbound E5b config reached core construction")

    monkeypatch.setattr(core, "run_from_config", forbidden)
    monkeypatch.setenv("PBS_JOBID", "unbound")
    assert runner.run_attempt(job_id="Job42", job_data=data) == 3
    outcomes = list((root / "outputs").rglob("outcome.json"))
    assert len(outcomes) == 1
    outcome = json.loads(outcomes[0].read_text())
    assert outcome["outcome"] == "terminal"
    assert outcome["exception_type"] == "ReceiptBindingError"
