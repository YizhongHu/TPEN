"""Formal review round 2: E5 native prefix and same-Job start policy.

The native arms use the repository's real ``native_recovery.yaml`` and the
real configured runner.  They use only the backend-neutral E4 durable receipt
seam; no Balsam package, API, network, or facility allocation is involved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tpen.execution import (
    BalsamAdapter,
    CoreOutcomeFacts,
    ExecutionConfig,
    ReadyBatch,
    ReadyItem,
    RecoveryPhase,
    WorkerStartGateway,
    capture_science_intent,
    payload_digest,
    source_config_digest,
)
from tpen.execution.invocation import _committed_checkpoint


FIXTURE = Path(__file__).parents[2] / "integration" / "artifacts" / "execution" / "native_recovery.yaml"


class PrefixWalltimeTrap(BaseException):
    """Bound the pilot's intentional walltime wait without accepting exit 3."""


class ScienceCallbackTrap(RuntimeError):
    pass


class RegisterOnlyClient:
    def __init__(self) -> None:
        self.register_calls = 0

    def register_job(self, *, lineage_id, batch, backend_options):
        self.register_calls += 1
        return {"id": "Job42"}

    def observe_job(self, job_id):  # pragma: no cover - worker tripwire
        raise AssertionError("E5 unit arm must not observe the backend")


def _prepare(root: Path) -> tuple[ReadyBatch, object]:
    config = OmegaConf.load(str(FIXTURE))
    source = OmegaConf.to_container(config, resolve=False)
    target = int(config.trainer.max_steps)
    science_input = {
        "fixture": "native-recovery-v1",
        "system": source["system"],
        "target_iterations": target,
    }
    intent = capture_science_intent(
        science_input, target_iterations=target, source_config=source
    )
    payload = {
        "fixture": "native-recovery-v1",
        "science_input": science_input,
        "target_iterations": target,
    }
    digest = payload_digest(payload)
    execution = ExecutionConfig(
        lineage_id="r2-e5-lineage",
        backend="balsam",
        recovery="checkpoint",
        source_config_digest=source_config_digest(source),
        payload_digest=digest,
        backend_options={"fixture": str(FIXTURE)},
    )
    batch = ReadyBatch(
        lineage_id=execution.lineage_id,
        batch_id="r2-e5-batch",
        source_config_digest=execution.source_config_digest,
        payload_digest=digest,
        science_input_digest=intent.science_input_digest,
        target_iterations=target,
        execution=execution,
        items=(
            ReadyItem(
                "e5-invocation",
                "e5-task",
                "e5-output",
                ("tpen", "native-recovery.yaml"),
                "native-recovery",
            ),
        ),
        payload=payload,
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "intent.json").write_text(
        json.dumps(intent.to_dict(), sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "batch.json").write_bytes(batch.encode())
    client = RegisterOnlyClient()
    adapter = BalsamAdapter(
        batch.execution,
        client,
        WorkerStartGateway(root / "unused-worker-output"),
        root / "receipts",
    )
    assert adapter.register_lineage(batch) is not None
    return batch, intent


def _job_data(root: Path) -> dict[str, object]:
    return {
        "pilot_root": str(root),
        "config_path": str(FIXTURE),
        "prefix_steps": 4,
    }


def _runner_module():
    from tools import e5b_polaris_runner

    return e5b_polaris_runner


def test_native_prefix_publishes_evidence_before_bounded_walltime_trap(
    monkeypatch, tmp_path: Path
) -> None:
    """RED at E5: native run dir is nested, but runner reads only its prefix."""

    root = tmp_path / "native-prefix"
    _prepare(root)
    runner = _runner_module()
    monkeypatch.setenv("PBS_JOBID", "fixture-allocation-1")

    def trap(_seconds: float) -> None:
        raise PrefixWalltimeTrap

    monkeypatch.setattr(runner.time, "sleep", trap)
    with pytest.raises(PrefixWalltimeTrap):
        runner.run_attempt(job_id="Job42", job_data=_job_data(root))

    assert (root / "checkpoint-evidence.json").is_file()
    assert (root / "prefix-ready.json").is_file()


def test_native_prefix_nested_run_dir_resolves_full_checkpoint_control(
    monkeypatch, tmp_path: Path
) -> None:
    """GREEN control: core lookup succeeds at the actual nested run directory."""

    root = tmp_path / "native-prefix-control"
    _prepare(root)
    runner = _runner_module()
    monkeypatch.setenv("PBS_JOBID", "fixture-allocation-control")
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: (_ for _ in ()).throw(PrefixWalltimeTrap()))
    with pytest.raises(PrefixWalltimeTrap):
        runner.run_attempt(job_id="Job42", job_data=_job_data(root))

    status_paths = list((root / "outputs").rglob("status.json"))
    assert status_paths, "native runner must emit a run-owned status artifact"
    reference = _committed_checkpoint(status_paths[0].parent)
    assert reference is not None
    assert reference.checkpoint_dir.is_dir()
    assert (reference.checkpoint_dir / "COMPLETE").is_file()


def test_prior_initial_start_without_evidence_parks_before_science(
    monkeypatch, tmp_path: Path
) -> None:
    """RED at E5: a same-Job successor cannot infer INITIAL from missing evidence."""

    root = tmp_path / "prior-initial"
    batch, intent = _prepare(root)
    prior = BalsamAdapter(
        batch.execution,
        RegisterOnlyClient(),
        WorkerStartGateway(root / "prior-output"),
        root / "receipts",
    )
    prior.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
        run_once=lambda *_: CoreOutcomeFacts.completed(
            target_iterations=batch.target_iterations,
            next_iteration=batch.target_iterations,
            completed_updates=batch.target_iterations,
        ),
        job_id="Job42",
        attempt_id="prior-initial-attempt",
    )
    assert not (root / "checkpoint-evidence.json").exists()

    runner = _runner_module()
    calls = []

    def spy(*_args, **_kwargs):
        calls.append(True)
        raise ScienceCallbackTrap("configured science callback reached")

    monkeypatch.setattr(runner, "run_configured_once", spy)
    monkeypatch.setenv("PBS_JOBID", "fixture-allocation-successor")
    runner.run_attempt(job_id="Job42", job_data=_job_data(root))
    assert calls == [], "prior same-Job INITIAL receipt must park before science"


def test_first_ever_start_without_evidence_invokes_initial_once(
    monkeypatch, tmp_path: Path
) -> None:
    """GREEN control: no prior start receipt permits one initial callback."""

    root = tmp_path / "first-ever"
    _prepare(root)
    runner = _runner_module()
    calls = []

    def spy(*_args, **_kwargs):
        calls.append(True)
        return None

    monkeypatch.setattr(runner, "run_configured_once", spy)
    monkeypatch.setenv("PBS_JOBID", "fixture-allocation-first")
    runner.run_attempt(job_id="Job42", job_data=_job_data(root))
    assert len(calls) == 1
