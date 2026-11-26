"""Checkpoint/model identity checks used by experiment-side gateways."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tpen.checkpoint.hashing import checkpoint_hashes
from tpen.checkpoint.schema import read_manifest


def checkpoint_model_config_hash(checkpoint_dir: str | Path) -> str | None:
    """Read the model-config identity from one validated checkpoint manifest."""

    manifest = read_manifest(Path(checkpoint_dir) / "manifest.json", mode="model_only")
    return manifest.hashes.get("model_config")


def config_model_config_hash(config: Any) -> str | None:
    """Return the model-config identity that a checkpoint would record."""

    return checkpoint_hashes(config).get("model_config")


__all__ = ["checkpoint_model_config_hash", "config_model_config_hash"]
