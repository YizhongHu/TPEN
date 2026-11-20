"""Cannon-facing restart-chain acceptance for the Stage-1 protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from experiments.atomistic.stage1.protocol import (
    Protocol,
    STAGE1_PROTOCOL,
    build_manifest,
    config_identities,
)
from experiments.atomistic.stage1.restart import execute_restart
from experiments.atomistic.stage1.train import prepare_initial_checkpoint_config
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.checkpoint import TrainResume, save_checkpoint
from tpen.execution.chain_resume import (
    ChannelEvent,
    ProcessEvidence,
    assert_fresh_processes,
    compare_channel_parity,
    e5_channel_map,
)
from tpen.run import load_config


CONFIG_ROOT = Path(__file__).parents[2] / "experiments" / "atomistic" / "stage1" / "configs"


def _row(
    system: str, tmp_path: Path, *, protocol: Protocol = STAGE1_PROTOCOL,
) -> dict[str, object]:
    config_root = CONFIG_ROOT
    config_path = config_root / system / "train.yaml"
    checkpoints = (
        [0, 2, 4, 6, 8]
        if protocol.lineage_updates == 8
        else list(load_config(str(config_path)).checkpoint_steps)
    )
    rows = build_manifest(
        protocol=protocol,
        source_sha="a" * 40,
        config_sha256=config_identities(config_root=config_root, systems=(system,)),
        checkpoints=checkpoints,
        lineage_seeds=[11, 12, 13],
        checkpoint_sha256={},
        results_root=str(tmp_path),
        systems=(system,),
        phase=1,
        device="cpu",
    )
    return next(row for row in rows if row["kind"] == "train" and row["lineage"] == 0)


def _bounded_protocol() -> Protocol:
    """Return the test-only bounded lineage protocol used by real workers."""

    return Protocol(
        smoke_updates=1,
        smoke_walkers=2,
        lineage_updates=8,
        lineage_walkers=2,
        lineages=3,
        eval_streams=4,
        eval_walkers=2,
        eval_draws=2,
        retention_cadence=2,
    )


class _Stage1Trace:
    """Collect the five native post-cursor channels in a real worker."""

    def __init__(self, *, phase: str, seed: int) -> None:
        self.phase = phase
        self.pid = os.getpid()
        self.seed = seed
        self.seed_source = f"runtime.seed={seed}"
        self.command: tuple[str, ...] = ()
        self.current_step = -1
        self.events: list[ChannelEvent] = []
        self._seen: set[tuple[str, int]] = set()
        self.trainer: Any | None = None

    def begin_step(self, trainer: Any) -> None:
        """Bind observations to the trainer's durable resume cursor."""

        self.trainer = trainer
        self.current_step = int(trainer.next_iteration)
        self.record("trainer_cursor", "first_resumed_step", self.current_step)
        self.record(
            "process_random", "first_process_random",
            float(torch.rand((), device="cpu").item()),
        )

    def record(self, channel: str, event_kind: str, value: Any) -> None:
        """Retain the first random-dependent event for one channel and step."""

        key = (channel, self.current_step)
        if key in self._seen:
            return
        self._seen.add(key)
        if hasattr(value, "item"):
            value = value.item()
        self.events.append(
            ChannelEvent(
                channel=channel,
                step=self.current_step,
                event_kind=event_kind,
                value=float(value) if isinstance(value, (int, float)) else value,
            )
        )

    def write(self, path: Path) -> None:
        """Write JSON evidence that the parent converts to ProcessEvidence."""

        path.write_text(
            json.dumps({
                "phase": self.phase,
                "pid": self.pid,
                "seed": self.seed,
                "seed_source": self.seed_source,
                "command": list(self.command),
                "events": [
                    {
                        "channel": event.channel,
                        "step": event.step,
                        "event_kind": event.event_kind,
                        "value": event.value,
                        "random_dependent": event.random_dependent,
                    }
                    for event in self.events
                ],
            }, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _install_stage1_trace(trace: _Stage1Trace) -> Any:
    """Install test-only probes around the real Stage-1 runtime classes."""

    from contextlib import ExitStack

    from tpen.nn.tpen_wave_function import TPENWaveFunction
    from tpen.sampling.metropolis import MetropolisSampler
    from tpen.training.trainer import VMCTrainer
    import tpen.checkpoint.restore as restore

    stack = ExitStack()
    original_collect = MetropolisSampler.collect_samples
    original_propose = MetropolisSampler._propose
    original_forward = TPENWaveFunction.forward
    original_step = torch.optim.Adam.step
    original_fit = VMCTrainer.fit
    original_apply_rng = restore.apply_rng_state

    def collect_samples(self, *args, **kwargs):
        trace.begin_step(trace.trainer or self)
        return original_collect(self, *args, **kwargs)

    def propose(self, model, walkers):
        proposals, log_q_ratio = original_propose(self, model, walkers)
        trace.record(
            "sampler_proposal", "first_sampler_proposal",
            float(proposals.detach().mean().item()),
        )
        return proposals, log_q_ratio

    def forward(self, batch, *args, **kwargs):
        output = original_forward(self, batch, *args, **kwargs)
        trace.record(
            "model_logabs", "first_model_evaluation",
            float(output.logabs.detach().mean().item()),
        )
        return output

    def optimizer_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        parameters = [
            parameter.detach()
            for group in self.param_groups
            for parameter in group["params"]
        ]
        norm = float(torch.cat([parameter.reshape(-1) for parameter in parameters]).norm().item())
        trace.record("optimizer_update", "first_optimizer_update", norm)
        return result

    def fit(self, *args, **kwargs):
        trace.trainer = self
        return original_fit(self, *args, **kwargs)

    def apply_rng(state, device):
        original_apply_rng(state, device)
        trace.seed_source = "checkpoint:train_resume:rng_state"

    MetropolisSampler.collect_samples = collect_samples
    MetropolisSampler._propose = propose
    TPENWaveFunction.forward = forward
    torch.optim.Adam.step = optimizer_step
    VMCTrainer.fit = fit
    restore.apply_rng_state = apply_rng

    def undo() -> None:
        MetropolisSampler.collect_samples = original_collect
        MetropolisSampler._propose = original_propose
        TPENWaveFunction.forward = original_forward
        torch.optim.Adam.step = original_step
        VMCTrainer.fit = original_fit
        restore.apply_rng_state = original_apply_rng

    stack.callback(undo)
    return stack


def _real_chain_child() -> None:
    """Run one real chain phase in a distinct Python process."""

    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    system = str(payload["system"])
    phase = str(payload["phase"])
    row = json.loads(Path(payload["row_path"]).read_text(encoding="utf-8"))
    config_path = Path(str(payload["config_path"]))
    trace = _Stage1Trace(phase=phase, seed=int(row["seed"]))
    trace_path = Path(str(payload["trace_path"]))
    protocol = _bounded_protocol()
    import experiments.atomistic.stage1.restart as restart_module

    # This replacement is process-local test configuration only; no committed
    # constant or config file is changed.
    original_protocol = restart_module.STAGE1_PROTOCOL
    restart_module.STAGE1_PROTOCOL = protocol
    try:
        cfg = load_config(str(config_path))
        cfg.run.root = str(row["results_root"])
        cfg.run.run_id = str(row["run_id"])
        cfg.sampler.n_walkers = protocol.lineage_walkers
        cfg.sampler.burn_in = 1
        cfg.sampler.n_steps = 1
        cfg.trainer.max_steps = protocol.lineage_updates
        cfg.checkpoint_steps = [0, 2, 4, 6, 8]
        prepare_initial_checkpoint_config(cfg)
        with _install_stage1_trace(trace):
            if phase in {"fresh", "resumed", "control"}:
                planned = phase == "fresh"

                def real_run(run_cfg, *, config_path, command, restore_request=None):
                    callbacks = (
                        __import__("tpen.callback.planned_yield", fromlist=["PlannedYield"])
                        .PlannedYield(2, protocol.lineage_updates),
                    ) if planned else ()
                    if phase == "control":
                        restore_request = None
                    return __import__("tpen.run", fromlist=["run_from_config"]).run_from_config(
                        run_cfg,
                        config_path=str(config_path),
                        command=command,
                        restore_request=restore_request,
                        runtime_callbacks=callbacks,
                        raise_exceptions=True,
                    )

                result = restart_module.execute_restart(
                    cfg=cfg,
                    row=row,
                    config_path=config_path,
                    command=f"stage1 real restart {system} {phase}",
                    run=real_run,
                    run_dir=Path(str(payload["run_dir"])),
                )
                result_payload = {
                    "outcome": result.outcome,
                    "exit_code": result.exit_code,
                    "receipt": dict(result.receipt),
                }
            else:
                code = __import__("tpen.run", fromlist=["run_from_config"]).run_from_config(
                    cfg,
                    config_path=str(config_path),
                    command=f"stage1 real reference {system}",
                )
                result_payload = {"outcome": "completed", "exit_code": code}
    finally:
        restart_module.STAGE1_PROTOCOL = original_protocol
    trace.command = tuple(sys.argv)
    trace.write(trace_path)
    print(json.dumps(result_payload, sort_keys=True))


def _run_real_child(payload: dict[str, object]) -> tuple[dict[str, object], ProcessEvidence]:
    """Launch and decode one real worker process."""

    argument_path = Path(str(payload["trace_path"])).with_suffix(".args.json")
    argument_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, "-c",
            "from tests.integration.test_stage1_restartable import _real_chain_child; _real_chain_child()",
            str(argument_path),
        ],
        cwd=str(Path(__file__).parents[2]),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    evidence_payload = json.loads(Path(str(payload["trace_path"])).read_text(encoding="utf-8"))
    evidence = ProcessEvidence(
        phase=str(evidence_payload["phase"]),
        pid=int(evidence_payload["pid"]),
        seed=int(evidence_payload["seed"]),
        seed_source=str(evidence_payload["seed_source"]),
        command=tuple(str(item) for item in evidence_payload["command"]),
        returncode=completed.returncode,
        events=tuple(
            ChannelEvent(
                channel=str(event["channel"]),
                step=int(event["step"]),
                event_kind=str(event["event_kind"]),
                value=event["value"],
                random_dependent=bool(event["random_dependent"]),
            )
            for event in evidence_payload["events"]
        ),
    )
    return result, evidence


def _publish(run_dir: Path, step: int) -> None:
    """Publish a real full-state checkpoint with the core checkpoint writer."""

    model = torch.nn.Linear(1, 1, bias=False)
    context = SimpleNamespace(
        cfg=OmegaConf.create({"trainer": {"max_steps": STAGE1_PROTOCOL.lineage_updates}}),
        metadata=SimpleNamespace(device="cpu", dtype="float64", run_id="stage1-chain", extra={}),
        run_dir=run_dir,
    )
    save_checkpoint(
        output_dir=run_dir / "checkpoints",
        next_iteration=step,
        completed_updates=step,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        trainer=SimpleNamespace(
            state_dict=lambda: {"next_iteration": step, "completed_updates": step}
        ),
        sampler=SimpleNamespace(mcmc_state_dict=lambda: {"walker": step}),
        payload=TrainResume(),
        context=context,
    )


@pytest.mark.integration
@pytest.mark.stage1_real_restart
@pytest.mark.parametrize("system", ["he", "h2"])
def test_restart_admission_lease_receipt_chain_with_stub_driver(
    system: str, tmp_path: Path,
) -> None:
    """Exercise real He/H2 config admission and a two-attempt published chain.

    The checkpoint artifacts are produced by the real core writer so this
    checks publication, typed restore admission, leases, and outcome receipts
    on Cannon CPU without spending a science budget inside the test harness.
    The production 100,000-update driver remains the separately authorized
    runbook action; this test never changes that immutable horizon.
    """

    row = _row(system, tmp_path)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = tmp_path / system / "lineage-0"
    first_cfg = load_config(str(config_path))
    prepare_initial_checkpoint_config(first_cfg)
    first_calls: list[dict[str, object]] = []

    def first_driver(_cfg, **kwargs):
        first_calls.append(kwargs)
        _publish(run_dir, 1000)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "planned_yield"}) + "\n", encoding="utf-8"
        )
        return 7

    first = execute_restart(
        cfg=first_cfg,
        row=row,
        config_path=config_path,
        run=first_driver,
        run_dir=run_dir,
    )
    assert first.outcome == "planned_yield"
    assert first.exit_code == 7
    assert first_calls == [{"config_path": str(config_path), "command": "stage1 restart"}]

    second_cfg = load_config(str(config_path))
    prepare_initial_checkpoint_config(second_cfg)
    second_calls: list[dict[str, object]] = []

    def second_driver(_cfg, **kwargs):
        second_calls.append(kwargs)
        assert kwargs["restore_request"].__class__.__name__ == "OperationalRestoreRequest"
        _publish(run_dir, STAGE1_PROTOCOL.lineage_updates)
        (run_dir / "status.json").write_text(
            json.dumps({"status": "completed"}) + "\n", encoding="utf-8"
        )
        return 99

    second = execute_restart(
        cfg=second_cfg,
        row=row,
        config_path=config_path,
        run=second_driver,
        run_dir=run_dir,
    )
    assert second.outcome == "completed"
    assert second.exit_code == 99
    assert len(second_calls) == 1
    assert second_calls[0]["restore_request"].load["mode"] == "train_resume"
    assert first.lease_path != second.lease_path
    assert first.outcome_path != second.outcome_path
    assert first.lease_path.is_file() and second.lease_path.is_file()
    assert first.outcome_path.is_file() and second.outcome_path.is_file()


@pytest.mark.integration
@pytest.mark.stage1_real_restart
@pytest.mark.parametrize("system", ["he", "h2"])
def test_real_cannon_cpu_restartable_multi_attempt_chain(
    system: str, tmp_path: Path,
) -> None:
    """Run real He/H2 fresh -> published -> distinct-process restore chains.

    The test-only bounded protocol uses eight lineage updates, two walkers, and
    checkpoint boundaries at 0, 2, 4, 6, and 8. Both attempts execute the real
    configured TPEN model, Metropolis sampler, optimizer, checkpoint publisher,
    and public run_from_config gateway. The resumed process is compared with a
    separate uninterrupted process through the five supported chain_resume
    channels: model_logabs, optimizer_update, process_random, sampler_proposal,
    and trainer_cursor. A fresh control starts at cursor zero in a new run dir;
    its cursor therefore diverges from the restored cursor at the first event.
    """

    protocol = _bounded_protocol()
    row = _row(system, tmp_path / "resumed", protocol=protocol)
    config_path = CONFIG_ROOT / system / "train.yaml"
    run_dir = resolve_run_dir(row)
    row_path = tmp_path / f"{system}-resumed-row.json"
    row_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    first_payload = {
        "system": system,
        "phase": "fresh",
        "row_path": str(row_path),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "trace_path": str(tmp_path / f"{system}-fresh-trace.json"),
    }
    first_result, _first_evidence = _run_real_child(first_payload)
    assert first_result["outcome"] == "planned_yield", json.dumps(first_result, sort_keys=True)
    assert first_result["exit_code"] == 2
    first_receipt = first_result["receipt"]
    assert isinstance(first_receipt, dict)
    assert first_receipt["next_iteration"] == 2
    resume_cursor = int(first_receipt["next_iteration"]) - 1

    resumed_payload = {**first_payload, "phase": "resumed", "trace_path": str(tmp_path / f"{system}-resumed-trace.json")}
    resumed_result, resumed_evidence = _run_real_child(resumed_payload)
    assert resumed_result["outcome"] == "completed"
    assert resumed_result["exit_code"] == 0

    reference_row = _row(system, tmp_path / "reference", protocol=protocol)
    reference_row_path = tmp_path / f"{system}-reference-row.json"
    reference_row_path.write_text(json.dumps(reference_row, sort_keys=True) + "\n", encoding="utf-8")
    reference_payload = {
        "system": system,
        "phase": "reference",
        "row_path": str(reference_row_path),
        "config_path": str(config_path),
        "run_dir": str(tmp_path / "reference-run"),
        "trace_path": str(tmp_path / f"{system}-reference-trace.json"),
    }
    reference_result, reference_evidence = _run_real_child(reference_payload)
    assert reference_result["outcome"] == "completed"
    assert reference_result["exit_code"] == 0

    assert_fresh_processes(reference_evidence, resumed_evidence)
    channel_map = e5_channel_map()
    parity = compare_channel_parity(
        reference_evidence,
        resumed_evidence,
        resume_step=resume_cursor,
        channel_map=channel_map,
        expected_mismatches={spec.limb: frozenset() for spec in channel_map},
    )
    assert all(parity.values())

    control_row = _row(system, tmp_path / "control", protocol=protocol)
    control_row_path = tmp_path / f"{system}-control-row.json"
    control_row_path.write_text(json.dumps(control_row, sort_keys=True) + "\n", encoding="utf-8")
    control_payload = {
        "system": system,
        "phase": "control-fresh",
        "row_path": str(control_row_path),
        "config_path": str(config_path),
        "run_dir": str(tmp_path / "control-run"),
        "trace_path": str(tmp_path / f"{system}-control-trace.json"),
    }
    control_result, control_evidence = _run_real_child(control_payload)
    assert control_result["outcome"] == "completed"
    assert control_result["exit_code"] == 0
    resumed_cursor = next(
        event.step for event in resumed_evidence.events if event.channel == "trainer_cursor"
    )
    control_cursor = next(
        event.step for event in control_evidence.events if event.channel == "trainer_cursor"
    )
    assert control_cursor == 0
    assert resumed_cursor == resume_cursor + 1
    assert control_cursor != resumed_cursor
