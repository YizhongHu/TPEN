"""Validated protocol and manifest primitives for the Stage-1 atom ladder.

This module intentionally imports no ``tpen`` symbol.  Execution enters TPEN
through the configured ``run_from_config`` gateway in the thin drivers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

SYSTEMS = ("hooke1", "he", "h2")
SYSTEM_IDS = {"hooke1": "hooke_one_electron_omega0.5", "he": "he_atom", "h2": "h2_molecule"}
SYSTEM_CONFIGS = {name: f"configs/{name}" for name in SYSTEMS}


class ProtocolError(ValueError):
    """Raised when a Stage-1 protocol payload is incomplete or inconsistent."""


@dataclass(frozen=True)
class Protocol:
    """The fixed Stage-1 sampling and lineage protocol."""

    smoke_updates: int = 1_000
    smoke_walkers: int = 256
    lineage_updates: int = 50_000
    lineage_walkers: int = 4_096
    lineages: int = 3
    eval_streams: int = 4
    eval_walkers: int = 4_096
    eval_draws: int = 256

    def validate(self) -> None:
        """Reject non-positive sizes and a malformed independent-stream plan."""

        values = self.__dict__
        if any(int(value) <= 0 for value in values.values()):
            raise ProtocolError(f"protocol sizes must be positive: {values}")
        if self.lineages < 3 or self.eval_streams < 4:
            raise ProtocolError("Stage-1 requires three lineages and four eval streams")


def stable_row_id(*, system: str, kind: str, lineage: int = 0, checkpoint: int = 0, stream: int = 0) -> str:
    """Return a stable, opaque row id derived from protocol coordinates."""

    if system not in SYSTEMS:
        raise ProtocolError(f"unknown Stage-1 system {system!r}")
    if kind not in {"smoke", "train", "eval"}:
        raise ProtocolError(f"unknown row kind {kind!r}")
    raw = f"stage1/v1|{system}|{kind}|{lineage}|{checkpoint}|{stream}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def build_manifest(protocol: Protocol, *, checkpoints: list[int], seeds: list[int]) -> list[dict[str, Any]]:
    """Expand each system into smoke, lineage, and fresh-evaluation rows."""

    protocol.validate()
    if len(seeds) != protocol.lineages or len(set(seeds)) != len(seeds):
        raise ProtocolError("seeds must contain one distinct seed per lineage")
    if not checkpoints or any(step <= 0 for step in checkpoints):
        raise ProtocolError("checkpoints must be a non-empty list of positive updates")
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        rows.append({"row_id": stable_row_id(system=system, kind="smoke"), "system": system,
                     "kind": "smoke", "updates": protocol.smoke_updates, "walkers": protocol.smoke_walkers})
        for lineage, seed in enumerate(seeds):
            train_id = stable_row_id(system=system, kind="train", lineage=lineage)
            rows.append({"row_id": train_id, "system": system, "kind": "train", "lineage": lineage,
                         "seed": seed, "updates": protocol.lineage_updates, "walkers": protocol.lineage_walkers})
            for checkpoint in checkpoints:
                for stream in range(protocol.eval_streams):
                    rows.append({"row_id": stable_row_id(system=system, kind="eval", lineage=lineage,
                                                          checkpoint=checkpoint, stream=stream),
                                 "system": system, "kind": "eval", "lineage": lineage,
                                 "checkpoint": checkpoint, "stream": stream,
                                 "walkers": protocol.eval_walkers, "draws": protocol.eval_draws,
                                 "depends_on": train_id})
    return rows


def manifest_hash(rows: list[Mapping[str, Any]]) -> str:
    """Hash a manifest using canonical JSON serialization."""

    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
