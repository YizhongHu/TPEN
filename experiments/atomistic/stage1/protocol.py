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
EVAL_SEED_BASE = 1_000_000
EVAL_SEED_STRIDE = 100


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


def evaluation_seed(*, system: str, lineage: int, checkpoint: int, stream: int) -> int:
    """Derive evaluation seeds in a namespace disjoint from training seeds.

    Training seeds are supplied by the caller.  The reserved million-range
    namespace prevents accidental overlap even when a caller chooses ordinary
    small integer lineage seeds; the stride leaves room for four streams and
    makes the lineage/stream coordinates inspectable in manifests.
    """

    if system not in SYSTEMS or lineage < 0 or checkpoint <= 0 or stream < 0 or stream >= EVAL_SEED_STRIDE:
        raise ProtocolError("invalid lineage/stream coordinate")
    # The large system stride makes every manifest coordinate unique without
    # relying on a probabilistic hash, while retaining human-readable seeds.
    system_index = SYSTEMS.index(system)
    return (EVAL_SEED_BASE + system_index * 1_000_000_000_000
            + checkpoint * EVAL_SEED_STRIDE + lineage * EVAL_SEED_STRIDE + stream)


def training_seed(*, system: str, lineage_seed: int) -> int:
    """Namespace a caller's lineage seed by system for global uniqueness."""

    if system not in SYSTEMS:
        raise ProtocolError(f"unknown Stage-1 system {system!r}")
    return SYSTEMS.index(system) * 1_000_000_000_000 + lineage_seed


def build_manifest(protocol: Protocol, *, checkpoints: list[int], seeds: list[int],
                   results_root: str) -> list[dict[str, Any]]:
    """Expand each system into smoke, lineage, and fresh-evaluation rows."""

    protocol.validate()
    if len(seeds) != protocol.lineages or len(set(seeds)) != len(seeds):
        raise ProtocolError("seeds must contain one distinct seed per lineage")
    if not checkpoints or any(step <= 0 for step in checkpoints):
        raise ProtocolError("checkpoints must be a non-empty list of positive updates")
    if not results_root or results_root.startswith("experiments/"):
        raise ProtocolError("results_root must be an explicit path outside the checkout")
    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        rows.append({"row_id": stable_row_id(system=system, kind="smoke"), "system": system,
                     "kind": "smoke", "config": f"configs/{system}/smoke.yaml",
                     "updates": protocol.smoke_updates, "walkers": protocol.smoke_walkers,
                     "overrides": {"trainer.max_steps": protocol.smoke_updates,
                                   "sampler.n_walkers": protocol.smoke_walkers,
                                   "run.root": f"{results_root}/{system}/smoke",
                                   "run.run_id": "smoke"}})
        for lineage, seed in enumerate(seeds):
            train_id = stable_row_id(system=system, kind="train", lineage=lineage)
            train_root = f"{results_root}/{system}/train"
            train_run_id = f"lineage-{lineage}"
            train_seed_value = training_seed(system=system, lineage_seed=seed)
            rows.append({"row_id": train_id, "system": system, "kind": "train", "lineage": lineage,
                         "config": f"configs/{system}/train.yaml", "seed": train_seed_value,
                         "updates": protocol.lineage_updates, "walkers": protocol.lineage_walkers,
                         "overrides": {"trainer.max_steps": protocol.lineage_updates,
                                       "sampler.n_walkers": protocol.lineage_walkers,
                                       "runtime.seed": train_seed_value,
                                       "run.root": train_root,
                                       "run.run_id": train_run_id}})
            for checkpoint in checkpoints:
                for stream in range(protocol.eval_streams):
                    row_id = stable_row_id(system=system, kind="eval", lineage=lineage,
                                           checkpoint=checkpoint, stream=stream)
                    eval_run_id = f"lineage-{lineage}-checkpoint-{checkpoint:06d}-stream-{stream}"
                    checkpoint_dir = f"{train_root}/{train_run_id}/checkpoints/step_{checkpoint:06d}"
                    eval_seed = evaluation_seed(system=system, lineage=lineage,
                                                checkpoint=checkpoint, stream=stream)
                    rows.append({"row_id": row_id, "system": system, "kind": "eval", "lineage": lineage,
                                 "checkpoint": checkpoint, "stream": stream,
                                 "walkers": protocol.eval_walkers, "draws": protocol.eval_draws,
                                 "config": f"configs/{system}/eval.yaml",
                                 "seed": eval_seed,
                                 "overrides": {"evaluation_sampler.n_walkers": protocol.eval_walkers,
                                               "evaluation_tasks.mcmc_energy.generator.n_draws": protocol.eval_draws,
                                               "evaluation.seed": eval_seed,
                                               "load.path": checkpoint_dir,
                                               "run.root": f"{results_root}/{system}/eval",
                                               "run.run_id": eval_run_id,
                                               "trajectory_identity.stage": "stage1",
                                               "trajectory_identity.run_id": eval_run_id,
                                               "trajectory_identity.attempt_id": f"stream-{stream}",
                                               "trajectory_identity.evaluator_id": f"stage1-{system}",
                                               "trajectory_identity.checkpoint_file": f"{checkpoint_dir}/model.pt",
                                               "trajectory_identity.config_sha256": row_id},
                                 "depends_on": train_id})
    row_seeds = [row["seed"] for row in rows if "seed" in row]
    if len(row_seeds) != len(set(row_seeds)):
        raise ProtocolError("training and evaluation seeds must be globally unique")
    return rows


def manifest_hash(rows: list[Mapping[str, Any]]) -> str:
    """Hash a manifest using canonical JSON serialization."""

    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
