"""Regenerate canonical core-owned boundary fixtures from real producers.

The temporary run deliberately uses a stable logical path.  The producer still
resolves and reads the real status, run-start identity, model, and checkpoint
manifest artifacts; only run-specific absolute paths are normalized before the
fixture is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow the helper to be run directly from the repository root without an
# installed package; test runners already place the repository on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tpen.checkpoint.hashing import resolved_config_identity
from tpen.checkpoint.manifest import CheckpointManifest
from tpen.inspection import RunEvidence, inspect_run

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "core-run-evidence-v2.json"
REFUSED_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "core-run-evidence-v1-refused.json"


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


def _normalize_path(path: str, temporary_root: Path) -> str:
    """Replace one exact temporary prefix while retaining its relative path."""

    try:
        relative = Path(path).resolve().relative_to(temporary_root.resolve())
    except ValueError as exc:
        raise ValueError(f"producer path is outside temporary root: {path}") from exc
    return str(Path("/normalized") / relative)


def normalize(evidence: RunEvidence, temporary_root: Path) -> dict[str, Any]:
    """Normalize only the known temporary prefix; retain producer values."""

    data = evidence.to_dict()
    data["run_dir"] = _normalize_path(data["run_dir"], temporary_root)
    if data["checkpoint"] is not None:
        data["checkpoint"]["checkpoint_dir"] = _normalize_path(
            data["checkpoint"]["checkpoint_dir"], temporary_root
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args()
    payload = regenerate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
