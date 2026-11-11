"""Checkpoint directory artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tpen.artifacts import write_json

COMPLETE_MARKER = "COMPLETE"
LATEST_JSON = "latest.json"


def checkpoint_step_dir_name(step: int) -> str:
    """Return the canonical directory name for a checkpoint step."""

    if int(step) < 0:
        raise ValueError(f"checkpoint step must be nonnegative, got {step}")
    return f"step_{int(step):06d}"


def is_complete_checkpoint_dir(path: str | Path) -> bool:
    """Return whether `path` is a complete checkpoint directory."""

    checkpoint_dir = Path(path)
    return (
        checkpoint_dir.is_dir()
        and not checkpoint_dir.name.endswith(".tmp")
        and (checkpoint_dir / "manifest.json").is_file()
        and (checkpoint_dir / COMPLETE_MARKER).is_file()
    )


def require_complete_checkpoint_dir(path: str | Path) -> Path:
    """Return `path` as a checkpoint directory or fail loudly."""

    checkpoint_dir = Path(path)
    if checkpoint_dir.name.endswith(".tmp"):
        raise ValueError(f"checkpoint tmp directory is not valid: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")
    if not (checkpoint_dir / COMPLETE_MARKER).is_file():
        raise ValueError(f"checkpoint directory lacks COMPLETE marker: {checkpoint_dir}")
    if not (checkpoint_dir / "manifest.json").is_file():
        raise ValueError(f"checkpoint directory lacks manifest.json: {checkpoint_dir}")
    return checkpoint_dir


def read_latest(checkpoint_root: str | Path) -> dict[str, Any]:
    """Read the latest checkpoint pointer from `checkpoint_root/latest.json`."""

    latest_path = Path(checkpoint_root) / LATEST_JSON
    if not latest_path.is_file():
        raise FileNotFoundError(f"latest checkpoint pointer not found: {latest_path}")
    with latest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data.get("checkpoint_dir"):
        raise ValueError(f"invalid latest checkpoint pointer: {latest_path}")
    return data


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Resolve a checkpoint root, latest pointer, or step directory to a valid step dir."""

    candidate = Path(path)
    if candidate.is_file() and candidate.name == LATEST_JSON:
        pointer = read_latest(candidate.parent)
        return require_complete_checkpoint_dir(candidate.parent / str(pointer["checkpoint_dir"]))
    if candidate.is_dir() and (candidate / LATEST_JSON).is_file() and not (candidate / "manifest.json").exists():
        pointer = read_latest(candidate)
        return require_complete_checkpoint_dir(candidate / str(pointer["checkpoint_dir"]))
    return require_complete_checkpoint_dir(candidate)


def write_latest(checkpoint_root: Path, checkpoint_dir: Path, *, step: int, created_at_unix: float) -> None:
    """Atomically update `latest.json` to point at `checkpoint_dir`."""

    latest_path = checkpoint_root / LATEST_JSON
    tmp_path = checkpoint_root / f"{LATEST_JSON}.tmp"
    write_json(
        tmp_path,
        {
            "checkpoint_dir": checkpoint_dir.name,
            "step": int(step),
            "created_at_unix": float(created_at_unix),
        },
    )
    tmp_path.replace(latest_path)


def list_complete_checkpoints(checkpoint_root: str | Path) -> list[Path]:
    """Return complete checkpoint step directories ordered by step."""

    root = Path(checkpoint_root)
    checkpoints = [path for path in root.glob("step_*") if is_complete_checkpoint_dir(path)]
    return sorted(checkpoints, key=_checkpoint_sort_key)


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.removeprefix("step_")), path.name
    except ValueError:
        return -1, path.name
