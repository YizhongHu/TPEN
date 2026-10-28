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


class AllocationDeath(BaseException):
    """Simulate allocation loss immediately after the durable start receipt."""


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


def _record_worker_backend_calls(monkeypatch, runner, calls: list[tuple[str, object]]) -> None:
    """Instrument the real worker client without replacing its product type."""

    def register(self, **kwargs):
        calls.append(("register_job", kwargs))
        raise AssertionError("worker attempted Balsam registration")

    def observe(self, job_id):
        calls.append(("observe_job", job_id))
        raise AssertionError("worker attempted Balsam observation")

    monkeypatch.setattr(runner.NoLiveBackendClient, "register_job", register)
    monkeypatch.setattr(runner.NoLiveBackendClient, "observe_job", observe)


def test_native_prefix_publishes_evidence_before_bounded_walltime_trap(
    monkeypatch, tmp_path: Path
) -> None:
    """RED at E5: native run dir is nested, but runner reads only its prefix."""

    root = tmp_path / "native-prefix"
    _prepare(root)
    runner = _runner_module()
    client_calls: list[tuple[str, object]] = []
    _record_worker_backend_calls(monkeypatch, runner, client_calls)
    monkeypatch.setenv("PBS_JOBID", "fixture-allocation-1")

    def trap(_seconds: float) -> None:
        raise PrefixWalltimeTrap

    monkeypatch.setattr(runner.time, "sleep", trap)
    with pytest.raises(PrefixWalltimeTrap):
        runner.run_attempt(job_id="Job42", job_data=_job_data(root))

    assert (root / "checkpoint-evidence.json").is_file()
    assert (root / "prefix-ready.json").is_file()
    assert client_calls == []


def test_native_prefix_nested_run_dir_resolves_full_checkpoint_control(
    monkeypatch, tmp_path: Path
) -> None:
    """GREEN control: the actual prefix emits a valid nested checkpoint ref.

    The current runner may return 3 after its prefix lookup misses the nested
    run directory; that return is allowed only for this control. The control
    independently resolves the one actual nested status parent with the core
    checkpoint reader.
    """

    root = tmp_path / "native-prefix-control"
    _prepare(root)
    runner = _runner_module()
    client_calls: list[tuple[str, object]] = []
    _record_worker_backend_calls(monkeypatch, runner, client_calls)
    monkeypatch.setenv("PBS_JOBID", "fixture-allocation-control")
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: (_ for _ in ()).throw(PrefixWalltimeTrap()))
    try:
        return_code = runner.run_attempt(job_id="Job42", job_data=_job_data(root))
    except PrefixWalltimeTrap:
        return_code = None
    assert return_code in (None, 3)

    status_paths = list((root / "outputs").rglob("status.json"))
    assert len(status_paths) == 1, "native runner must emit one run-owned status artifact"
    reference = _committed_checkpoint(status_paths[0].parent)
    assert reference is not None
    assert reference.checkpoint_dir.is_dir()
    assert (reference.checkpoint_dir / "COMPLETE").is_file()
    assert client_calls == []


def test_prior_initial_start_without_evidence_parks_before_science(
    monkeypatch, tmp_path: Path
) -> None:
    """RED at E5: a dead actual INITIAL start must park its same-Job successor."""

    root = tmp_path / "prior-initial"
    batch, intent = _prepare(root)
    runner = _runner_module()
    client_calls: list[tuple[str, object]] = []
    _record_worker_backend_calls(monkeypatch, runner, client_calls)
    prior = BalsamAdapter(
        batch.execution,
        RegisterOnlyClient(),
        WorkerStartGateway(root / "outputs"),
        root / "receipts",
    )
    def allocation_dies(event: str) -> None:
        if event == "receipt_published":
            raise AllocationDeath("fixture allocation died after start receipt")

    with pytest.raises(AllocationDeath, match="after start receipt"):
        prior.start(
            batch=batch,
            intent=intent,
            phase=RecoveryPhase.INITIAL,
            run_once=lambda *_: (_ for _ in ()).throw(
                ScienceCallbackTrap("prior science callback must not run")
            ),
            job_id="Job42",
            attempt_id="prior-initial-attempt",
            fault_injector=allocation_dies,
        )
    assert not (root / "checkpoint-evidence.json").exists()
    assert not list((root / "outputs").rglob("outcome.json"))

    calls = []

    def spy(*_args, **_kwargs):
        calls.append(True)
        raise ScienceCallbackTrap("configured science callback reached")

    monkeypatch.setattr(runner, "run_configured_once", spy)
    monkeypatch.setenv("PBS_JOBID", "fixture-allocation-successor")
    runner.run_attempt(job_id="Job42", job_data=_job_data(root))
    assert calls == [], "prior same-Job INITIAL receipt must park before science"
    assert client_calls == []
    assert not (root / "checkpoint-evidence.json").exists()

    # A correct implementation may now publish a durable E3 parked outcome
    # after reconciling the prior start.  Require typed park evidence with a
    # reason, while reading only this fixture root and preserving the stronger
    # pre-successor assertions above (no science and no outcome immediately
    # after the injected allocation death).
    retained_park_records: list[tuple[Path, dict]] = []
    for candidate in root.rglob("*.json"):
        if not (
            candidate.name in {"outcome.json", "worker-park.json", "registration-park.json"}
            or candidate.name.endswith("-application.json")
        ):
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        typed_status = str(payload.get("status", payload.get("outcome", ""))).lower()
        typed_outcome = str(payload.get("outcome", payload.get("successor_outcome", ""))).lower()
        reason = payload.get("reason")
        if typed_status == "parked" or typed_outcome == "parked":
            assert isinstance(reason, str) and reason.strip(), (
                f"durable park evidence must carry a nonempty reason: {candidate}"
            )
            retained_park_records.append((candidate, payload))
    assert retained_park_records, "same-Job successor must retain typed parked E3/application evidence"


def test_first_ever_start_without_evidence_invokes_initial_once(
    monkeypatch, tmp_path: Path
) -> None:
    """GREEN control: no prior start receipt permits one initial callback."""

    root = tmp_path / "first-ever"
    _prepare(root)
    runner = _runner_module()
    client_calls: list[tuple[str, object]] = []
    _record_worker_backend_calls(monkeypatch, runner, client_calls)
    calls = []

    def spy(*_args, **_kwargs):
        calls.append(True)
        return None

    monkeypatch.setattr(runner, "run_configured_once", spy)
    monkeypatch.setenv("PBS_JOBID", "fixture-allocation-first")
    runner.run_attempt(job_id="Job42", job_data=_job_data(root))
    assert len(calls) == 1
    assert client_calls == []


def test_unmodified_worker_client_is_accepted_by_adapter(tmp_path: Path) -> None:
    """RED at E5: the shipped worker client must satisfy the adapter protocol."""

    root = tmp_path / "unmodified-worker-client"
    batch, _intent = _prepare(root)
    runner = _runner_module()

    # Deliberately use the owner source's unmodified client.  This regression
    # must expose the product protocol mismatch directly; it must not install
    # the compatible tripwire used by the four scenario-isolation leaves.
    BalsamAdapter(
        batch.execution,
        runner.NoLiveBackendClient(),
        WorkerStartGateway(root / "worker-output"),
        root / "unmodified-receipts",
    )

