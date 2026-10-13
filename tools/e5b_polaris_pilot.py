"""Submitter/reconciler commands for the bounded E5b Polaris pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tpen.execution import (
    BalsamAdapter,
    ExecutionConfig,
    ReadyBatch,
    ReadyItem,
    capture_science_intent,
    payload_digest,
    source_config_digest,
)

from tools.e5b_polaris_client import PolarisBalsamClient


FIXTURE_RELATIVE = "tests/integration/artifacts/execution/native_recovery.yaml"


def _batch(
    root: Path,
    config_path: Path,
    site_root: Path,
    checkout_root: Path,
    app_root: Path,
    lineage_id: str,
    batch_id: str,
) -> ReadyBatch:
    """Materialize the E5a immutable fixture as one E2-ready batch."""

    config = OmegaConf.load(str(config_path))
    source = OmegaConf.to_container(config, resolve=False)
    target = int(config.trainer.max_steps)
    science_input = {
        "fixture": "native-recovery-v1",
        "system": source["system"],
        "target_iterations": target,
    }
    intent = capture_science_intent(science_input, target_iterations=target, source_config=source)
    payload = {"fixture": "native-recovery-v1", "science_input": science_input, "target_iterations": target}
    digest = payload_digest(payload)
    execution = ExecutionConfig(
        lineage_id=lineage_id,
        backend="balsam",
        recovery="checkpoint",
        source_config_digest=source_config_digest(source),
        payload_digest=digest,
        backend_options={
            "site_root": str(site_root),
            "checkout_root": str(checkout_root),
            "app_root": str(app_root),
            "pilot_root": str(root),
            "site_packages": "/home/rhu/.local/lib/python3.13/site-packages",
            "app_module": "tools.e5b_polaris_app",
            "app_class": "PolarisE5bApp",
            "python_exe": "/soft/applications/conda/2025-09-25/mconda3/bin/python",
            "ld_library_prefix": [
                "/home/rhu/tpen-libshim-20260823",
                "/soft/compilers/cudatoolkit/cuda-13.0.1/lib64",
                "/soft/compilers/cudatoolkit/cuda-12.9.1/lib64",
            ],
            "wall_time_min": 5,
            "workdir": batch_id,
            "job_data": {
                "pilot_root": str(root),
                "config_path": str(config_path),
                "prefix_steps": max(1, target // 2),
                "fixture": FIXTURE_RELATIVE,
            },
        },
    )
    batch = ReadyBatch(
        lineage_id=execution.lineage_id,
        batch_id=batch_id,
        source_config_digest=execution.source_config_digest,
        payload_digest=digest,
        science_input_digest=intent.science_input_digest,
        target_iterations=target,
        execution=execution,
        items=(ReadyItem(
            invocation_id="e5b-polaris-invocation",
            logical_task_id="e5b-polaris-task",
            fresh_output_id="e5b-polaris-planned-output",
            argv=("tpen", "native-recovery.yaml"),
            result_dir="native-recovery",
            payload={"seed": int(config.runtime.seed)},
        ),),
        payload=payload,
    )
    (root / "intent.json").write_text(json.dumps(intent.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
    (root / "batch.json").write_bytes(batch.encode())
    return batch


def _adapter(args: argparse.Namespace, batch: ReadyBatch) -> BalsamAdapter:
    """Build a submitter-side E4 adapter around the lazy real client."""

    options = dict(batch.execution.backend_options)
    client = PolarisBalsamClient(args.site, options)
    # The gateway is not used by submitter commands; worker starts use the
    # backend-free gateway constructed inside the Balsam app process.
    from tpen.execution import WorkerStartGateway

    return BalsamAdapter(batch.execution, client, WorkerStartGateway(args.root / "submitter-unused"), args.root / "receipts")


def prepare(args: argparse.Namespace) -> None:
    """Write serialized inputs and register exactly one persistent Job."""

    args.root.mkdir(parents=True, exist_ok=True)
    batch = _batch(
        args.root,
        args.config,
        args.site,
        args.checkout,
        args.app_root,
        args.lineage_id,
        args.batch_id,
    )
    adapter = _adapter(args, batch)
    registration = adapter.register_lineage(batch)
    if registration is None:
        raise RuntimeError("Balsam registration did not acknowledge a persistent Job")
    receipt = {
        "schema": "tpen.execution/e5b-prepare-v1",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "batch_id": batch.batch_id,
        "lineage_id": batch.lineage_id,
        "registration": registration.to_dict(),
        "site_root": str(args.site),
        "pilot_root": str(args.root),
    }
    (args.root / "prepare.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))


def observe(args: argparse.Namespace) -> None:
    """Record one backend observation after the allocation loss."""

    prepare_data = json.loads((args.root / "prepare.json").read_text(encoding="utf-8"))
    batch = ReadyBatch.decode((args.root / "batch.json").read_bytes())
    observed = _adapter(args, batch).observe_lineage(batch)
    record = {
        "schema": "tpen.execution/e5b-observe-v1",
        "source_sha": prepare_data["source_sha"],
        "registration": prepare_data["registration"],
        "observation": None if observed is None else observed.to_dict(),
    }
    (args.root / "observation.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))


def main() -> int:
    """Dispatch a submitter/reconciler command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "observe"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--app-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lineage-id", default="e5b-polaris-lineage-a2")
    parser.add_argument("--batch-id", default="e5b-polaris-batch-a2")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        observe(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
