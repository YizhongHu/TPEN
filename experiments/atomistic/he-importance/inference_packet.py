"""Immutable, reference-free packets for expensive independent inference.

This is an evaluation-side boundary.  It may consume the immutable
``IndependentSamplerTestPacket`` made by ``stage_coordinate``, but it never
accepts train-side walker state, ranking-arm walker state, or a reference
value.  Selection and confirmation locking deliberately belong to L4b/L4c.

The status record is an audit record, not a probe of a live process.  In
particular, idle, dead, and completed chains have different required fields,
and every terminal state records a finish time even where no finish
notification was emitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any


class InferencePacketError(ValueError):
    """An inference packet is incomplete, mutable, or mixes evaluation arms."""


class ChainState(str, Enum):
    """Recorded lifecycle states for one independently initialized chain."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    DEAD = "dead"


_TERMINAL_STATES = frozenset(
    {ChainState.COMPLETED, ChainState.ERROR, ChainState.CANCELLED, ChainState.DEAD}
)


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-shaped provenance without changing its values."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(nested) for nested in value)
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Return a string-keyed mapping or name the packet boundary failure."""

    if not isinstance(value, Mapping):
        raise InferencePacketError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise InferencePacketError(f"{label} keys must be strings")
    return value


@dataclass(frozen=True)
class SamplingInterval:
    """The fixed sampling interval for every retained draw of one chain.

    Parameters
    ----------
    burn_in_proposals
        Proposals discarded before retaining a draw.
    proposals_between_draws
        Proposals separating successive retained draws.
    retained_draws
        Number of retained draws.  This is a packet allocation, not an ESS.
    """

    burn_in_proposals: int
    proposals_between_draws: int
    retained_draws: int

    def __post_init__(self) -> None:
        if self.burn_in_proposals < 0:
            raise InferencePacketError("burn_in_proposals must be non-negative")
        if self.proposals_between_draws < 1:
            raise InferencePacketError("proposals_between_draws must be positive")
        if self.retained_draws < 1:
            raise InferencePacketError("retained_draws must be positive")


@dataclass(frozen=True)
class ChainSeedProvenance:
    """Complete, phase-separated seed provenance for one inference chain."""

    training_seed: int
    calibration_seed: int
    inference_seed: int
    chain_seed: int

    def __post_init__(self) -> None:
        values = (
            self.training_seed,
            self.calibration_seed,
            self.inference_seed,
            self.chain_seed,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise InferencePacketError("all seed provenance values must be positive integers")
        if len(set(values)) != len(values):
            raise InferencePacketError("chain seeds must be disjoint from every phase seed")


@dataclass(frozen=True)
class ChainStatus:
    """Self-sufficient lifecycle record for a chain, including terminal failures.

    ``finish_notification_at`` is evidence that an observer was notified, not
    evidence that a chain terminated: error/dead/cancelled states may have no
    notification.  Consumers must use ``state`` and ``finished_at``.
    """

    state: ChainState
    created_at: datetime
    last_activity_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    terminal_reason: str | None = None
    finish_notification_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.last_activity_at < self.created_at:
            raise InferencePacketError("chain activity cannot precede creation")
        if self.state is ChainState.IDLE:
            if any(value is not None for value in (self.started_at, self.finished_at, self.terminal_reason)):
                raise InferencePacketError("idle chain status cannot claim start or termination")
        elif self.state is ChainState.RUNNING:
            if self.started_at is None or self.finished_at is not None or self.terminal_reason is not None:
                raise InferencePacketError("running chain status requires only a start record")
        elif self.state in _TERMINAL_STATES:
            if self.started_at is None or self.finished_at is None:
                raise InferencePacketError("terminal chain status requires start and finish records")
            if self.finished_at < self.started_at:
                raise InferencePacketError("chain finish cannot precede start")
            if self.state is ChainState.COMPLETED:
                if self.terminal_reason is not None:
                    raise InferencePacketError("completed chain status cannot carry a failure reason")
            elif not (self.terminal_reason or "").strip():
                raise InferencePacketError("non-completed terminal status requires a reason")
        else:  # Defensive: Enum construction normally makes this unreachable.
            raise InferencePacketError(f"unknown chain state {self.state!r}")
        if self.finish_notification_at is not None:
            if self.finished_at is None:
                raise InferencePacketError("finish notification requires a terminal status")
            if self.finish_notification_at < self.finished_at:
                raise InferencePacketError("finish notification cannot precede termination")


@dataclass(frozen=True)
class CheckpointReference:
    """A checkpoint bound to its parent content-addressed cell directory."""

    parent_cell_path: Path
    checkpoint_path: Path
    source_content_hash: str
    topology_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        parent = Path(self.parent_cell_path)
        checkpoint = Path(self.checkpoint_path)
        if not parent.is_absolute() or not checkpoint.is_absolute():
            raise InferencePacketError("checkpoint and parent cell paths must be absolute")
        if checkpoint.parent.parent != parent or checkpoint.parent.name != "checkpoints":
            raise InferencePacketError("checkpoint must be bound beneath its parent cell checkpoints directory")
        if len(self.source_content_hash) != 64 or any(character not in "0123456789abcdef" for character in self.source_content_hash):
            raise InferencePacketError("source_content_hash must be lowercase sha256")
        topology = _require_mapping(self.topology_provenance, "topology provenance")
        object.__setattr__(self, "parent_cell_path", parent)
        object.__setattr__(self, "checkpoint_path", checkpoint)
        object.__setattr__(self, "topology_provenance", _freeze(dict(topology)))


@dataclass(frozen=True)
class DistributedCheckpointProvenance:
    """Complete distributed artifact membership for a checkpoint, when used."""

    world_size: int
    rank_artifacts: Mapping[int, Path]

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size < 1:
            raise InferencePacketError("distributed checkpoint world_size must be positive")
        if not isinstance(self.rank_artifacts, Mapping):
            raise InferencePacketError("distributed checkpoint rank_artifacts must be a mapping")
        expected = set(range(self.world_size))
        if set(self.rank_artifacts) != expected:
            raise InferencePacketError("distributed checkpoint artifacts must contain each rank exactly once")
        artifacts = {rank: Path(path) for rank, path in self.rank_artifacts.items()}
        if len(set(artifacts.values())) != self.world_size:
            raise InferencePacketError("distributed checkpoint rank artifacts must be unambiguous")
        object.__setattr__(self, "rank_artifacts", MappingProxyType(artifacts))


@dataclass(frozen=True)
class IndependentChain:
    """One fresh independent chain; it has no inherited walker-state field."""

    chain_id: str
    seeds: ChainSeedProvenance
    interval: SamplingInterval
    status: ChainStatus

    def __post_init__(self) -> None:
        if not isinstance(self.chain_id, str) or not self.chain_id.strip():
            raise InferencePacketError("chain_id must be non-empty")


@dataclass(frozen=True)
class InferencePacket:
    """Immutable launch packet for independent sampler inference only."""

    checkpoint: CheckpointReference
    independent_sampler_inputs: Mapping[str, Any]
    chains: tuple[IndependentChain, ...]
    distributed_checkpoint: DistributedCheckpointProvenance | None

    def __post_init__(self) -> None:
        inputs = _require_mapping(self.independent_sampler_inputs, "independent sampler inputs")
        forbidden = {"training_walker_state", "cheap_arm_walker_state", "ranking_walker_state"}
        if forbidden.intersection(inputs):
            raise InferencePacketError("independent inference packet cannot reuse train or ranking walkers")
        if not self.chains:
            raise InferencePacketError("inference packet requires at least one independent chain")
        if len({chain.chain_id for chain in self.chains}) != len(self.chains):
            raise InferencePacketError("inference packet chain IDs must be unique")
        if len({chain.seeds.chain_seed for chain in self.chains}) != len(self.chains):
            raise InferencePacketError("inference packet chain seeds must be unique")
        if self.distributed_checkpoint is not None:
            for artifact in self.distributed_checkpoint.rank_artifacts.values():
                if not artifact.is_relative_to(self.checkpoint.checkpoint_path):
                    raise InferencePacketError("distributed artifact lies outside its checkpoint directory")
        object.__setattr__(self, "independent_sampler_inputs", _freeze(dict(inputs)))


def launch_inference_packet(
    source_packet: Any,
    checkpoint: CheckpointReference,
    chains: Sequence[IndependentChain],
    *,
    distributed_checkpoint: DistributedCheckpointProvenance | None = None,
) -> InferencePacket:
    """Bind an L2 independent-sampler packet to fresh chains and provenance.

    The small structural interface intentionally avoids importing the
    hyphenated experiment directory as a package.  It verifies that the L2
    packet names the same checkpoint and content identity before copying only
    its already-separated independent-sampler input block.
    """

    try:
        source_checkpoint = Path(source_packet.checkpoint_path)
        source_hash = source_packet.source_content_hash
        sampler_inputs = source_packet.independent_sampler_inputs
    except AttributeError as error:
        raise InferencePacketError("source packet must be an independent-sampler test packet") from error
    if source_checkpoint != checkpoint.checkpoint_path:
        raise InferencePacketError("source packet checkpoint does not match checkpoint reference")
    if source_hash != checkpoint.source_content_hash:
        raise InferencePacketError("source packet content hash does not match checkpoint reference")
    return InferencePacket(
        checkpoint=checkpoint,
        independent_sampler_inputs=sampler_inputs,
        chains=tuple(chains),
        distributed_checkpoint=distributed_checkpoint,
    )
