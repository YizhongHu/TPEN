"""Formal review round 5 reviewer tests (E5 tools): issues 1, 5 and 10.

Issue 1: the E5b prefix shortens ``trainer.max_steps``, so the successor's
``train_resume`` restore compares a different ``trainer_config`` hash.
Issue 5: the Balsam worker path never applies E3's configured-run binding.
Issue 10: the Balsam state vocabulary used by the E5b client.
"""

import json
import time

import pytest
from omegaconf import OmegaConf

from tests.unit.execution.test_formal_r2_e5 import (
    FIXTURE,
    PrefixWalltimeTrap,
    _job_data,
    _prepare,
    _record_worker_backend_calls,
    _runner_module,
)


_REAL_SLEEP = time.sleep


def _trap(*_args, **_kwargs):
    raise PrefixWalltimeTrap()


def _run_pair(monkeypatch, tmp_path, *, equal_trainer: bool):
    """Run the real E5b prefix start, then the real successor start."""

    from tpen.execution import RecoveryPhase

    root = tmp_path / ("r5-equal-trainer" if equal_trainer else "r5-two-start")
    _prepare(root)
    runner = _runner_module()
    calls: list[tuple[str, object]] = []
    _record_worker_backend_calls(monkeypatch, runner, calls)

    # Start 1: bounded prefix; the walltime wait is trapped after evidence is published.
    monkeypatch.setenv("PBS_JOBID", "r5-alloc-1")
    monkeypatch.setattr(runner.time, "sleep", _trap)
    with pytest.raises(PrefixWalltimeTrap):
        runner.run_attempt(job_id="Job42", job_data=_job_data(root))
    assert (root / "checkpoint-evidence.json").is_file(), "prefix checkpoint evidence was not observed"

    # Start 2: successor of the same Job.  The successor path never waits, so restore sleep.
    monkeypatch.setattr(runner.time, "sleep", _REAL_SLEEP)
    monkeypatch.setenv("PBS_JOBID", "r5-alloc-2")
    if equal_trainer:
        original = runner._run_once_factory

        def wrapped(config, *, config_path, phase, prefix_steps=None):
            if phase is RecoveryPhase.RECOVERY:
                from tpen.execution import CheckpointEvidence

                prefix_evidence = CheckpointEvidence.from_mapping(json.loads((root / "checkpoint-evidence.json").read_text()))
                prefix_config = OmegaConf.load(prefix_evidence.ref.checkpoint_dir.parent.parent / "config.yaml")
                copy = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
                # R5 issue 1: 4 was the defective prefix value. Match the
                # trainer section actually captured by the fixed prefix.
                OmegaConf.update(copy, "trainer", OmegaConf.to_container(prefix_config.trainer, resolve=False), merge=False)
                return original(copy, config_path=config_path, phase=phase, prefix_steps=prefix_steps)
            return original(config, config_path=config_path, phase=phase, prefix_steps=prefix_steps)

        monkeypatch.setattr(runner, "_run_once_factory", wrapped)
    rc = runner.run_attempt(job_id="Job42", job_data=_job_data(root))
    namespaces = sorted((root / "outputs").glob("*/*/polaris-successor-r5-alloc-2"))
    assert len(namespaces) == 1, f"successor namespaces observed: {namespaces!r}"
    outcome = json.loads((namespaces[0] / "outcome.json").read_text(encoding="utf-8"))
    print(f"SUCCESSOR_OUTCOME={outcome.get('outcome')!r} REASON={outcome.get('reason')!r} RC={rc!r}")
    application_path = root / "successor-application.json"
    application = json.loads(application_path.read_text(encoding="utf-8")) if application_path.is_file() else {}
    print(f"SUCCESSOR_APPLICATION={application!r}")
    return calls, rc, outcome, application


# PREDICTED: RED at E5 and CR (issue 1)
def test_r5_e5b_prefix_then_successor_recovers_to_target(monkeypatch, tmp_path):
    calls, rc, outcome, application = _run_pair(monkeypatch, tmp_path, equal_trainer=False)
    assert calls == [], f"backend calls observed: {calls!r}"
    assert rc == 0 and application.get("successor_outcome") == "completed", (
        f"run_attempt rc={rc!r}; successor outcome={outcome.get('outcome')!r}; "
        f"reason={outcome.get('reason')!r}"
    )


# PREDICTED: GREEN at E5 and CR (issue 1 single-variable control)
def test_r5_e5b_successor_with_prefix_trainer_section_passes_restore(monkeypatch, tmp_path):
    calls, _rc, outcome, _application = _run_pair(monkeypatch, tmp_path, equal_trainer=True)
    assert calls == [], f"backend calls observed: {calls!r}"
    assert outcome.get("outcome") != "terminal", f"observed outcome={outcome!r}"
    assert "trainer_config" not in str(outcome.get("reason", "")), f"observed reason={outcome.get('reason')!r}"


class R5StopBeforeScience(RuntimeError):
    """Raised by the spy so that no science runs."""


# PREDICTED: RED at E5 and CR (issue 5)
def test_r5_e5b_worker_refuses_config_not_bound_to_intent(monkeypatch, tmp_path):
    root = tmp_path / "r5-unbound"
    _prepare(root)  # batch and intent are built from FIXTURE (optimizer.lr 0.01)
    runner = _runner_module()
    changed = OmegaConf.load(FIXTURE)
    changed.optimizer.lr = 0.02
    changed_path = tmp_path / "changed.yaml"
    OmegaConf.save(changed, changed_path)
    job_data = _job_data(root)
    job_data["config_path"] = str(changed_path)
    seen: list[float] = []

    def spy(cfg, *args, **kwargs):
        seen.append(float(OmegaConf.select(cfg, "optimizer.lr")))
        raise R5StopBeforeScience("stop before science")

    monkeypatch.setattr(runner, "run_configured_once", spy)
    monkeypatch.setattr(runner.time, "sleep", _trap)
    monkeypatch.setenv("PBS_JOBID", "r5-unbound")
    try:
        rc = runner.run_attempt(job_id="Job42", job_data=job_data)
    except PrefixWalltimeTrap:
        rc = None
    assert seen == [], (
        "the worker passed a config whose optimizer.lr is not the one bound in the "
        f"immutable intent to the runner: observed lr values {seen!r}, run_attempt rc={rc!r}"
    )


# Upstream Balsam 0.7 JobState names; not verified against the installed Polaris package.
@pytest.mark.parametrize(
    "raw,expected",
    [("RUN_TIMEOUT", "ALLOCATION_LOST"), ("STAGED_IN", "ACTIVE"), ("RUN_DONE", "ACTIVE"), ("STAGED_OUT", "ACTIVE")],
)
# PREDICTED: RED at E5 and CR (issue 10)
def test_r5_balsam_state_vocabulary_maps_lifecycle_names(raw, expected):
    from tools.e5b_polaris_client import BalsamJobState, _map_state

    observed = _map_state(raw)
    assert observed is getattr(BalsamJobState, expected), f"{raw} mapped to {observed!r}"


@pytest.mark.parametrize(
    "raw,expected",
    [("RESTART_READY", "ALLOCATION_LOST"), ("JOB_FINISHED", "COMPLETED"), ("FAILED", "PERMANENT_FAILURE"), ("RUNNING", "ACTIVE")],
)
# PREDICTED: GREEN at E5 and CR (issue 10 control)
def test_r5_balsam_state_vocabulary_controls(raw, expected):
    from tools.e5b_polaris_client import BalsamJobState, _map_state

    observed = _map_state(raw)
    assert observed is getattr(BalsamJobState, expected), f"{raw} mapped to {observed!r}"
