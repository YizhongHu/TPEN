"""Toolkit boundary for TPEN run and checkpoint path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


class RunPathError(ValueError):
    """A manifest row lacks the fields required to resolve its run path."""


def resolve_run_dir(row: Mapping[str, Any]) -> Path:
    """Resolve a manifest row using TPEN's real artifact layout."""

    from tpen.artifacts import ArtifactManager

    admitted = row.get("run_dir")
    try:
        return ArtifactManager(
            row["results_root"], row["experiment_name"], row["sector"], row["run_id"],
            layout=row.get("run_layout", "nested"),
            logical_run_dir=None if admitted is None else Path(str(admitted)),
        ).run_dir
    except (KeyError, TypeError, ValueError) as exc:
        raise RunPathError("manifest row lacks fields required to resolve run directory") from exc


def resolve_checkpoint_dir(path: str | Path) -> Path:
    """Resolve a complete checkpoint through TPEN's public checkpoint helper."""

    from tpen.checkpoint.artifact import resolve_checkpoint_dir as resolve

    return resolve(path)


__all__ = ["RunPathError", "resolve_checkpoint_dir", "resolve_run_dir"]
