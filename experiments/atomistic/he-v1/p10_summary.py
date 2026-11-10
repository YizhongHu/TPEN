"""Build the P10 terminal summary from product-shaped run artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml


def summarize(manifest_path: Path, summary_path: Path, child_observation_filename: str) -> dict:
    """Read the product artifacts and write the terminal summary."""

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "tpen-executor/run-manifest-v2", manifest
    assert len(manifest["attempts"]) == 1, manifest
    assert len(manifest["segments"]) == 1, manifest
    attempt = manifest["attempts"][0]
    receipt = attempt["placement_receipt"]
    attempt_root = Path(attempt["output_dir"])
    observation = json.loads((attempt_root / child_observation_filename).read_text())
    resolved = yaml.safe_load((attempt_root / "core" / "resolved_config.yaml").read_text())
    segment = manifest["segments"][0]
    checkpoint = Path(segment["checkpoint_dir"])
    assert (checkpoint / "COMPLETE").is_file(), checkpoint
    assert resolved["trainer"]["max_steps"] == 25, resolved
    assert resolved["sampler"]["n_walkers"] == 16, resolved
    assert observation["cuda_visible_devices"] == "0", observation
    assert observation["environment"]["OMP_NUM_THREADS"] == "4", observation
    eagle_checkpoint = str(checkpoint)
    lus_checkpoint = eagle_checkpoint.replace("/eagle/", "/lus/eagle/", 1)
    fence_pass = os.path.realpath(eagle_checkpoint) == os.path.realpath(lus_checkpoint)
    assert fence_pass, (eagle_checkpoint, lus_checkpoint, os.path.realpath(eagle_checkpoint), os.path.realpath(lus_checkpoint))
    summary = {
        "manifest": str(manifest_path),
        "manifest_schema": manifest["schema"],
        "accepted_segment": {"task_id": segment["task_id"], "completed_updates": segment["completed_updates"], "next_iteration": segment["next_iteration"], "checkpoint_dir": segment["checkpoint_dir"]},
        "attempt_status": attempt["outcome_receipt"]["status"],
        "checkpoint_complete": str(checkpoint / "COMPLETE"),
        "child_observation": str(attempt_root / child_observation_filename),
        "child_hostname": observation["hostname"],
        "child_cuda_visible_devices": observation["cuda_visible_devices"],
        "child_omp_num_threads": observation["environment"]["OMP_NUM_THREADS"],
        "resolved_trainer_max_steps": resolved["trainer"]["max_steps"],
        "resolved_sampler_n_walkers": resolved["sampler"]["n_walkers"],
        "placement_receipt": {"comparison": receipt["comparison"], "declared": receipt["declared"], "observed": {"host": receipt["child"]["host"], "cuda_visible_devices": receipt["child"]["cuda_visible_devices"], "affinity": receipt["child"]["affinity"]}, "refusal": receipt.get("refusal"), "gpu_status": receipt["gpu"]["status"]},
        "fence": {"outcome": "PASS", "eagle_path": eagle_checkpoint, "lus_path": lus_checkpoint, "eagle_realpath": os.path.realpath(eagle_checkpoint), "lus_realpath": os.path.realpath(lus_checkpoint)},
        "home": os.environ["HOME"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    result = summarize(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
    print(json.dumps(result, sort_keys=True))
