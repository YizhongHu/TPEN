"""Regenerate canonical core-owned boundary fixtures from real producers.

The temporary run deliberately uses a stable logical path. The producer still
resolves and reads the real status, run-start identity, model, and checkpoint
manifest artifacts; only structural paths and stable logical identifiers
survive provenance normalization before the fixture is written.
"""

from __future__ import annotations

import argparse
from contextlib import chdir
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow the helper to be run directly from the repository root without an
# installed package; test runners already place the repository on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tpen.checkpoint.hashing import resolved_config_identity
from tpen.checkpoint.manifest import CheckpointManifest
from tpen.artifacts import RunResult
from tpen.inspection import RunEvidence, inspect_run

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "core-run-evidence-v2.json"
REFUSED_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "core-run-evidence-v1-refused.json"
PLANNED_YIELD_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "core-run-evidence-v3-planned-yield.json"
UNKNOWN_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "core-run-evidence-v3-unknown.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_REF_PLACEHOLDERS = {
    "content_id": "<sha256:content_id>",
    "manifest_sha256": "<sha256:manifest>",
    "model_sha256": "<sha256:model>",
}
_EXPECTED_PROVENANCE_KEYS = frozenset(
    {
        "command",
        "config_id",
        "cwd",
        "git_branch",
        "git_dirty",
        "git_sha",
        "hostname",
        "python",
        "python_executable",
        "run_dir",
        "run_id",
        "slurm",
        "study_name",
        "tpen_version",
    }
)
_EXPECTED_SLURM_KEYS = frozenset(
    {
        "cpus_per_task",
        "job_id",
        "job_name",
        "job_partition",
        "mem_per_node",
        "submit_dir",
    }
)
_ENVIRONMENT_PLACEHOLDER = "<env>"


def _make_run(root: Path) -> Path:
    run_dir = root / "attempt" / "core"
    checkpoint = run_dir / "checkpoints" / "step_000007"
    checkpoint.mkdir(parents=True)
    model = b"cx00 canonical model payload\n"
    (checkpoint / "model.pt").write_bytes(model)
    CheckpointManifest(
        schema_version=2,
        kind="tpen.checkpoint",
        next_iteration=7,
        completed_updates=5,
        created_at_unix=1_700_000_000.0,
        files={"model": "model.pt"},
        hashes={"model_sha256": hashlib.sha256(model).hexdigest()},
        runtime={},
        provenance={},
    ).write(checkpoint / "manifest.json")
    (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (run_dir / "status.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    identity = resolved_config_identity({"model": {"name": "fixture"}})
    (run_dir / "run_start.json").write_text(
        json.dumps({"resolved_config_identity": identity}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _make_refused_run(root: Path) -> Path:
    run_dir = root / "attempt" / "core"
    checkpoint = run_dir / "checkpoints" / "step_000007"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text("{not valid json\n", encoding="utf-8")
    (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")
    (run_dir / "status.json").write_text('{"status":"completed"}\n', encoding="utf-8")
    identity = resolved_config_identity({"model": {"name": "fixture"}})
    (run_dir / "run_start.json").write_text(
        json.dumps({"resolved_config_identity": identity}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _make_unknown_run(root: Path) -> Path:
    """Create a non-eligible run through the typed S4 result producer."""

    run_dir = root / "attempt" / "core"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status":"unknown"}\n', encoding="utf-8")
    (run_dir / "run-outcome.json").write_text(
        json.dumps(RunResult(status="unknown").to_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    identity = resolved_config_identity({"model": {"name": "fixture"}})
    (run_dir / "run_start.json").write_text(
        json.dumps({"resolved_config_identity": identity}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _make_planned_yield_run(root: Path) -> Path:
    """Run the small public S4 planned-yield setup and return its physical leaf."""

    from omegaconf import OmegaConf

    from tpen.artifacts import ArtifactDestination
    from tpen.callback.planned_yield import PlannedYield
    from tpen.run import PLANNED_YIELD_EXIT_CODE, run_from_config

    config_path = Path(__file__).resolve().parents[1] / "integration" / "artifacts" / "training" / "vmc_smoke.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.run.root = str(root / "logical")
    cfg.run.run_id = "planned-yield"
    cfg.trainer.max_steps = 3
    cfg.sampler.n_walkers = 8
    cfg.sampler.burn_in = 1
    cfg.sampler.n_steps = 1
    for callback in cfg.callbacks:
        if callback.get("_target_") == "tpen.callback.Checkpoint":
            callback.schedule.every_n = 1
    destination = ArtifactDestination(root / "physical")
    # The checkpoint manifest records cwd as provenance. Keep that path under
    # the same temporary prefix as the other producer paths before normalizing.
    destination.physical_root.mkdir(parents=True, exist_ok=True)
    with chdir(destination.physical_root):
        code = run_from_config(
            cfg,
            artifact_destination=destination,
            runtime_callbacks=(PlannedYield(2, 3),),
            raise_exceptions=True,
        )
    if code != PLANNED_YIELD_EXIT_CODE:
        raise RuntimeError(f"planned-yield producer returned unexpected code {code}")
    return destination.physical_root


def _normalize_path(path: str, temporary_root: Path) -> str:
    """Replace one exact temporary prefix while retaining its relative path."""

    try:
        relative = Path(path).resolve().relative_to(temporary_root.resolve())
    except ValueError as exc:
        raise ValueError(f"producer path is outside temporary root: {path}") from exc
    return str(Path("/normalized") / relative)


def _normalize_provenance(value: Any, temporary_root: Path) -> Any:
    """Normalize structural paths and redact environment-specific metadata."""

    if not isinstance(value, dict) or set(value) != _EXPECTED_PROVENANCE_KEYS:
        raise ValueError(
            "checkpoint provenance keys changed: "
            f"expected {_EXPECTED_PROVENANCE_KEYS}, "
            f"got {set(value) if isinstance(value, dict) else type(value)}"
        )
    slurm = value["slurm"]
    if not isinstance(slurm, dict) or set(slurm) != _EXPECTED_SLURM_KEYS:
        raise ValueError(
            "checkpoint provenance slurm keys changed: "
            f"expected {_EXPECTED_SLURM_KEYS}, "
            f"got {set(slurm) if isinstance(slurm, dict) else type(slurm)}"
        )
    for key in ("cwd", "run_dir"):
        if not isinstance(value[key], str):
            raise ValueError(f"checkpoint provenance {key} must be a path string")
    if not isinstance(value["run_id"], str):
        raise ValueError("checkpoint provenance run_id must be a string")
    normalized = {key: _ENVIRONMENT_PLACEHOLDER for key in _EXPECTED_PROVENANCE_KEYS}
    normalized["run_id"] = value["run_id"]
    normalized["cwd"] = _normalize_path(value["cwd"], temporary_root)
    normalized["run_dir"] = _normalize_path(value["run_dir"], temporary_root)
    normalized["slurm"] = {
        key: _ENVIRONMENT_PLACEHOLDER for key in _EXPECTED_SLURM_KEYS
    }
    return normalized


def _placeholder_checkpoint_identity(checkpoint_ref: dict[str, Any]) -> None:
    """Validate and hide path-dependent checkpoint identity hashes."""

    for key, placeholder in _CHECKPOINT_REF_PLACEHOLDERS.items():
        value = checkpoint_ref.get(key)
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"checkpoint_ref.{key} is not a 64-hex SHA-256 value")
        checkpoint_ref[key] = placeholder


def normalize(evidence: RunEvidence, temporary_root: Path) -> dict[str, Any]:
    """Normalize only the known temporary prefix; retain producer values."""

    data = evidence.to_dict()
    data["run_dir"] = _normalize_path(data["run_dir"], temporary_root)
    if data["checkpoint"] is not None:
        data["checkpoint"]["checkpoint_dir"] = _normalize_path(
            data["checkpoint"]["checkpoint_dir"], temporary_root
        )
    if data.get("checkpoint_ref") is not None:
        data["checkpoint_ref"]["checkpoint_dir"] = _normalize_path(
            data["checkpoint_ref"]["checkpoint_dir"], temporary_root
        )
        provenance = data["checkpoint_ref"].get("provenance")
        data["checkpoint_ref"]["provenance"] = _normalize_provenance(
            provenance, temporary_root
        )
        _placeholder_checkpoint_identity(data["checkpoint_ref"])
    if data["refusal"] is not None:
        data["refusal"]["reason"] = data["refusal"]["reason"].replace(
            str(temporary_root.resolve()), "/normalized"
        )
    return data


def regenerate() -> dict[str, Any]:
    """Regenerate from the project environment; never hand-edit the fixture."""
    with tempfile.TemporaryDirectory(prefix="cx00-fixture-") as temporary:
        temporary_root = Path(temporary) / "attempt"
        evidence = inspect_run(_make_run(Path(temporary)))
        normalized = normalize(evidence, temporary_root)
    return normalized


def regenerate_refused() -> dict[str, Any]:
    """Regenerate the v1 refusal fixture from a real corrupt-checkpoint refusal."""
    with tempfile.TemporaryDirectory(prefix="cx00-fixture-refused-") as temporary:
        temporary_root = Path(temporary) / "attempt"
        evidence = inspect_run(_make_refused_run(Path(temporary)))
        normalized = normalize(evidence, temporary_root)
    return normalized


def regenerate_unknown() -> dict[str, Any]:
    """Regenerate the canonical non-eligible unknown-outcome fixture."""

    with tempfile.TemporaryDirectory(prefix="cx01-fixture-unknown-") as temporary:
        temporary_root = Path(temporary) / "attempt"
        evidence = inspect_run(_make_unknown_run(Path(temporary)))
        normalized = normalize(evidence, temporary_root)
    return normalized


def regenerate_planned_yield() -> dict[str, Any]:
    """Regenerate v3 from the real small S4 planned-yield public entry."""

    with tempfile.TemporaryDirectory(prefix="cx01-fixture-planned-yield-") as temporary:
        temporary_root = Path(temporary) / "physical"
        evidence = inspect_run(_make_planned_yield_run(Path(temporary)))
        normalized = normalize(evidence, temporary_root)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIXTURE_PATH)
    parser.add_argument(
        "--fixture",
        choices=("v2", "refused", "unknown", "planned-yield"),
        default="v2",
    )
    args = parser.parse_args()
    payload = {
        "v2": regenerate,
        "refused": regenerate_refused,
        "unknown": regenerate_unknown,
        "planned-yield": regenerate_planned_yield,
    }[args.fixture]()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
