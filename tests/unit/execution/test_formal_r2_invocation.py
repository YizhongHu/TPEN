"""Formal review round 2: E3 immutable-config and initial-load bindings."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from tpen.checkpoint.payload import TrainResume
from tpen.execution import (
    ExecutionConfig,
    InvocationOutcome,
    ReadyBatch,
    ReadyItem,
    RecoveryPhase,
    ReceiptBindingError,
    capture_science_intent,
    source_config_digest,
)
from tpen.execution.config import payload_digest
from tpen.config_schema import ClosedSchemaError


def _batch(source: dict[str, object]) -> tuple[ReadyBatch, object]:
    science = {"system": "hooke", "electrons": 2}
    payload = {"science": science, "target_iterations": 2}
    intent = capture_science_intent(
        science, target_iterations=2, source_config=source
    )
    execution = ExecutionConfig(
        lineage_id="r2-e3-lineage",
        backend="direct",
        recovery="none",
        source_config_digest=source_config_digest(source),
        payload_digest=payload_digest(payload),
    )
    return (
        ReadyBatch(
            lineage_id=execution.lineage_id,
            batch_id="r2-e3-batch",
            source_config_digest=execution.source_config_digest,
            payload_digest=execution.payload_digest,
            science_input_digest=intent.science_input_digest,
            target_iterations=2,
            execution=execution,
            items=(ReadyItem("r2-e3-inv", "r2-e3-task", "r2-e3-out", ("worker",), "results/e3"),),
            payload=payload,
        ),
        intent,
    )


def _full_checkpoint(root: Path) -> Path:
    """Create a committed manifest/ref fixture for load-binding validation.

    This is not a claim that the byte payloads are a native scientific
    train-resume checkpoint; the policy under test only needs a structurally
    complete committed reference accepted by the checkpoint reader.
    """

    path = root / "step_000001"
    path.mkdir(parents=True)
    files = {
        "model": "model.pt",
        "optimizer": "optimizer.pt",
        "trainer": "trainer.json",
        "sampler": "sampler.pt",
        "rng": "rng.pt",
    }
    for name in files.values():
        (path / name).write_bytes(b"committed-manifest-ref-fixture")
    (path / "trainer.json").write_text(
        json.dumps({"next_iteration": 1, "completed_updates": 1}), encoding="utf-8"
    )
    manifest = {
        "schema_version": 2,
        "kind": "tpen.checkpoint",
        "next_iteration": 1,
        "completed_updates": 1,
        "created_at_unix": 1.0,
        "files": files,
        "hashes": {},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "r2-e3", "git_sha": "review"},
        "payload": TrainResume().to_manifest(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return path


def _run_spy(monkeypatch):
    from tpen import run as run_module

    calls = []

    def fake_run(config, **kwargs):
        calls.append((config, kwargs))
        return 0

    monkeypatch.setattr(run_module, "run_from_config", fake_run)
    return calls


def test_changed_optimizer_config_is_refused_before_science(monkeypatch, tmp_path: Path) -> None:
    """RED at E3: changed optimizer syntax cannot reuse the immutable intent."""

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}, "optimizer": {"lr": 0.01}}
    batch, intent = _batch(source)
    cfg = OmegaConf.create({"runner": source["runner"], "optimizer": {"lr": 0.02}})
    calls = _run_spy(monkeypatch)

    try:
        from tpen.execution import WorkerStartGateway

        invocation = WorkerStartGateway(tmp_path / "outputs").start_configured(
            cfg=cfg, batch=batch, intent=intent, phase=RecoveryPhase.INITIAL
        )
    except (ReceiptBindingError, ClosedSchemaError):
        pass
    else:
        assert invocation.outcome.outcome is InvocationOutcome.TERMINAL, (
            "source-config refusal must be exception or durable TERMINAL, "
            f"got {invocation.outcome.outcome!r}; captured runner calls={calls!r}; "
            f"cfg={OmegaConf.to_container(cfg, resolve=False)!r}"
        )
        assert invocation.outcome.reason.strip(), "durable terminal refusal needs a nonempty reason"
    assert calls == [], "source-config refusal must precede tpen.run.run_from_config"


def test_matching_source_config_calls_runner_once_after_start_receipt(monkeypatch, tmp_path: Path) -> None:
    """GREEN control: matching captured syntax reaches the runner once."""

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}, "optimizer": {"lr": 0.01}}
    batch, intent = _batch(source)
    calls = _run_spy(monkeypatch)

    from tpen.execution import WorkerStartGateway

    invocation = WorkerStartGateway(tmp_path / "outputs").start_configured(
        cfg=OmegaConf.create(source),
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
    )
    assert len(calls) == 1
    assert (invocation.output_path.parent / "start.json").is_file()


def test_initial_concrete_train_resume_load_is_refused_without_science(
    monkeypatch, tmp_path: Path
) -> None:
    """RED at E3: INITIAL cannot silently honor an undeclared checkpoint load."""

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    checkpoint = _full_checkpoint(tmp_path / "checkpoint")
    cfg = OmegaConf.create(
        {
            **source,
            "runner": {
                "_target_": "tests.helpers.NeverConstructed",
                "load": {"mode": "train_resume", "path": str(checkpoint)},
            },
        }
    )
    calls = _run_spy(monkeypatch)

    from tpen.execution import WorkerStartGateway

    try:
        invocation = WorkerStartGateway(tmp_path / "outputs").start_configured(
            cfg=cfg, batch=batch, intent=intent, phase=RecoveryPhase.INITIAL
        )
    except (ReceiptBindingError, ClosedSchemaError):
        pass
    else:
        assert invocation.outcome.outcome is InvocationOutcome.TERMINAL, (
            "undeclared initial load must be exception or durable TERMINAL, "
            f"got {invocation.outcome.outcome!r}; captured runner calls={calls!r}"
        )
        assert invocation.outcome.reason.strip(), "durable terminal refusal needs a nonempty reason"
    assert calls == [], "undeclared initial load must not invoke science"


def test_matching_no_load_initial_config_is_accepted(monkeypatch, tmp_path: Path) -> None:
    """GREEN control: first-ever INITIAL start with no load invokes once."""

    source = {"runner": {"_target_": "tests.helpers.NeverConstructed"}}
    batch, intent = _batch(source)
    calls = _run_spy(monkeypatch)

    from tpen.execution import WorkerStartGateway

    WorkerStartGateway(tmp_path / "outputs").start_configured(
        cfg=OmegaConf.create(source),
        batch=batch,
        intent=intent,
        phase=RecoveryPhase.INITIAL,
    )
    assert len(calls) == 1
