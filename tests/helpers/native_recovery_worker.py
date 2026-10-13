"""Subprocess worker for the native E5 qualification.

The parent process owns fake registration and allocation-loss observation.  A
worker receives only the durable batch, intent, Job identity, and (for a
successor) serialized checkpoint evidence.  It therefore exercises the same
E4 worker seam and E3 configured-run call that an actual launcher would use.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from tpen.execution import (
    BalsamAdapter,
    ReadyBatch,
    RecoveryPhase,
    ScienceIntent,
    WorkerStartGateway,
    decode_ready_batch,
    validate_native_recovery_profile,
)
from tpen.checkpoint.restore import restore_checkpoint
from tpen.execution.invocation import _core_outcome_from_artifacts, run_configured_once
from tpen.runner import Train
from tpen.runner.base import _assert_eager_initialized, _is_torch_module, _place_module_for_runtime
from tpen.training.optim import make_optimizer
from tests.helpers.native_chain_resume_trace import (
    NativeTrace,
    install_native_trace,
    skip_restore_owners,
)


class NoLiveBackendClient:
    """Client-shaped object proving worker start does not need a live API."""

    def register_job(self, **_kwargs):  # pragma: no cover - a worker must not call this
        raise AssertionError("worker attempted backend registration")

    def observe_job(self, _job_id):  # pragma: no cover - a worker must not call this
        raise AssertionError("worker attempted backend observation")


class PartialRestoreTrain(Train):
    """Test-only runner for the single-variable partial-restore control.

    The native E1 selector still supplies a full ``train_resume`` decision.
    This runner deliberately replaces only that load operation with the
    ``model_only`` checkpoint primitive, then restores the checkpoint cursor
    without restoring optimizer, sampler, or process-RNG state.  It bypasses
    the production ``Train`` model-only refusal only inside this qualification
    worker; no production configured path accepts this runner.
    """

    def run(self, context):
        """Load model weights and cursor, then train with fresh mutable state."""

        if _is_torch_module(self.model):
            _place_module_for_runtime(self.model, context)
            _assert_eager_initialized(self.model)
            self.model.train()

        mode = str(self.load.get("mode", "none")) if self.load is not None else "none"
        optimizer = make_optimizer(self.optimizer, self.model.parameters())
        if mode == "model_only":
            report = restore_checkpoint(
                load=self.load,
                model=self.model,
                context=context,
            )
            # Keep the recovery cursor identical while intentionally leaving
            # optimizer, sampler-MCMC, and process-RNG state fresh.
            self.trainer.next_iteration = int(report.next_iteration or 0)
            self.trainer.completed_updates = int(report.completed_updates or 0)
            self.load = None
        elif mode == "train_resume":
            # This branch exists only so the Cannon mutation pair can replace
            # the partial overlay with the real E1 overlay and prove that the
            # divergence assertion then goes red.
            resolve_update_state = getattr(self.trainer, "resolve_update_state", None)
            if callable(resolve_update_state):
                resolve_update_state(model=self.model, optimizer=optimizer)
            report = restore_checkpoint(
                load=self.load,
                model=self.model,
                optimizer=optimizer,
                trainer=self.trainer,
                sampler=self.sampler,
                context=context,
            )
            rebuild_update_state = getattr(self.trainer, "rebuild_update_state", None)
            if callable(rebuild_update_state):
                rebuild_update_state(model=self.model)
            del report
            self.load = None

        # Reuse the just-restored optimizer for the full-restore mutant; the
        # model-only control keeps this newly constructed optimizer fresh.
        self.optimizer = optimizer
        return super().run(context)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--phase", choices=("initial", "recovery"), required=True)
    parser.add_argument("--label", default="worker")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--partial-restore", action="store_true")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument(
        "--disable-limb",
        action="append",
        choices=("model", "optimizer", "trainer_cursor", "sampler_mcmc", "process_rng"),
    )
    parser.add_argument("--suppress-channel")
    return parser


def main() -> int:
    """Run one actual worker process and print its typed adapter result."""

    args = _parser().parse_args()
    batch = decode_ready_batch(args.batch.read_bytes())
    intent_data = json.loads(args.intent.read_text(encoding="utf-8"))
    intent = ScienceIntent(
        science_input=intent_data["science_input"],
        target_iterations=intent_data["target_iterations"],
        source_config=intent_data["source_config"],
    )
    cfg = OmegaConf.load(args.config)
    validate_native_recovery_profile(cfg)
    evidence = None
    if args.evidence is not None:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))

    adapter = BalsamAdapter(
        batch.execution,
        NoLiveBackendClient(),
        WorkerStartGateway(args.outputs),
        args.receipts,
    )

    run_cfg = cfg
    selected_decision = None
    trace = (
        NativeTrace(
            phase=args.phase,
            seed=int(cfg.runtime.seed),
            resume_step=args.resume_step,
            suppress_channels=()
            if args.suppress_channel is None
            else (args.suppress_channel,),
        )
        if args.trace is not None
        else None
    )

    def run_once(decision, output_path):
        nonlocal run_cfg, selected_decision
        selected_decision = decision
        if args.partial_restore:
            if decision.checkpoint is None:
                raise AssertionError("partial restore requires a selected checkpoint")
            run_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            OmegaConf.update(
                run_cfg,
                "runner._target_",
                "tests.helpers.native_recovery_worker.PartialRestoreTrain",
                merge=False,
            )
            partial_load = {
                "mode": "model_only",
                "path": str(decision.checkpoint.checkpoint_dir),
            }
            selected_decision = replace(decision, load=partial_load)
        trace_stack = install_native_trace(trace) if trace is not None else None
        try:
            with skip_restore_owners(args.disable_limb or ()):
                run_configured_once(
                    run_cfg,
                    decision=selected_decision or decision,
                    output_path=output_path,
                    config_path=str(args.config),
                    command="e5-native-worker",
                    raise_exceptions=True,
                )
        finally:
            if trace_stack is not None:
                trace_stack.close()
            if trace is not None:
                trace.write(args.trace)
        return _core_outcome_from_artifacts(output_path, decision=decision)

    result = adapter.start(
        batch=batch,
        intent=intent,
        phase=RecoveryPhase(args.phase),
        run_once=run_once,
        job_id=args.job_id,
        evidence=evidence,
    )
    invocation = result.invocation
    output = {
        "status": result.status.value,
        "reason": result.reason,
        "job_id": None if result.registration is None else result.registration.job_id,
        "invocation": None
        if invocation is None
        else {
            "output_path": invocation.start.output_path,
            "outcome": invocation.outcome.outcome.value,
            "status": invocation.outcome.status.value,
            "reason": invocation.outcome.reason,
            "refusal_code": invocation.outcome.refusal_code,
            "next_iteration": invocation.outcome.next_iteration,
            "completed_updates": invocation.outcome.completed_updates,
        },
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if result.status.value == "started" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
