"""Subprocess worker for the native E5 qualification.

The parent process owns fake registration and allocation-loss observation.  A
worker receives only the durable batch, intent, Job identity, and (for a
successor) serialized checkpoint evidence.  It therefore exercises the same
E4 worker seam and E3 configured-run call that an actual launcher would use.
"""

from __future__ import annotations

import argparse
import json
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
from tpen.execution.invocation import _core_outcome_from_artifacts, run_configured_once


class NoLiveBackendClient:
    """Client-shaped object proving worker start does not need a live API."""

    def register_job(self, **_kwargs):  # pragma: no cover - a worker must not call this
        raise AssertionError("worker attempted backend registration")

    def observe_job(self, _job_id):  # pragma: no cover - a worker must not call this
        raise AssertionError("worker attempted backend observation")


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

    def run_once(decision, output_path):
        run_configured_once(
            cfg,
            decision=decision,
            output_path=output_path,
            config_path=str(args.config),
            command="e5-native-worker",
            raise_exceptions=True,
        )
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
            "next_iteration": invocation.outcome.next_iteration,
            "completed_updates": invocation.outcome.completed_updates,
        },
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if result.status.value == "started" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
