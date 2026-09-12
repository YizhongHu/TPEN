"""Immutable, reference-free packets for expensive independent inference.

This is an evaluation-side boundary.  It may consume the immutable
``IndependentSamplerTestPacket`` made by ``stage_coordinate``, but it never
accepts train-side walker state, ranking-arm walker state, or a reference
value.  Selection and confirmation locking deliberately belong to L4b/L4c.

The independent-sampler input block is a closed declaration: only the three
declared integer paths are admitted, and the validator descends on that
declaration rather than on the value's type.  The block is read once and the
validated frozen structure is the value stored in the packet.

The declared integer channel is intentionally unbounded.  A Python ``int``
at ``walkers`` or ``sampler.walkers`` can carry an arbitrary payload in one
leaf, limited only by available process memory; for example, the full
4096-by-2-by-3 float64 walker buffer can be represented by a 196602-byte
integer.  Key closure and the single-pass binding, not the provenance extent
budget, are what separate sampler declarations from bulk payloads.

L4a's closure deliberately narrows L2's language at the ``sampler``
position: L2 descends into Sequences applying its own schema, while this
single-arm boundary admits only the declared mapping there.  A sequence-
wrapped mapping can therefore be accepted by L2 and refused here; widening
this boundary is out of scope.

The provenance extent has two separate axes.  Its leaf capacity is 8192
JSON-shaped scalar leaves at depth 6.  Its container axis has no separate
count budget: empty mappings and sequences cost zero leaves, so their breadth
is bounded only by available process memory and the depth limit.  The latter
is an accepted residual and is not priced as part of the sampler declaration.

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
from inspect import getattr_static
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any


class PacketRefusal(str, Enum):
    """Structured identity for each refusal site at this packet boundary."""

    SAMPLER_KEY_NOT_DECLARED = "sampler_key_not_declared"
    SAMPLER_VALUE_NOT_A_DECLARED_SCALAR = "sampler_value_not_a_declared_scalar"
    SAMPLER_SUBTREE_NOT_A_MAPPING = "sampler_subtree_not_a_mapping"
    SAMPLER_KEY_NOT_A_STRING = "sampler_key_not_a_string"
    SAMPLER_SPEC_NOT_CLOSED = "sampler_spec_not_closed"
    PROVENANCE_NOT_A_MAPPING = "provenance_not_a_mapping"
    PROVENANCE_KEY_NOT_A_STRING = "provenance_key_not_a_string"
    PROVENANCE_VALUE_NOT_JSON_SHAPED = "provenance_value_not_json_shaped"
    PROVENANCE_VALUE_NOT_FINITE = "provenance_value_not_finite"
    PROVENANCE_LEAF_BUDGET_EXCEEDED = "provenance_leaf_budget_exceeded"
    PROVENANCE_DEPTH_BUDGET_EXCEEDED = "provenance_depth_budget_exceeded"
    INTERVAL_BURN_IN_NEGATIVE = "interval_burn_in_negative"
    INTERVAL_BURN_IN_NOT_INT = "interval_burn_in_not_int"
    INTERVAL_SPACING_NOT_POSITIVE = "interval_spacing_not_positive"
    INTERVAL_SPACING_NOT_INT = "interval_spacing_not_int"
    INTERVAL_DRAWS_NOT_POSITIVE = "interval_draws_not_positive"
    INTERVAL_DRAWS_NOT_INT = "interval_draws_not_int"
    SEED_NOT_POSITIVE_INT = "seed_not_positive_int"
    SEEDS_NOT_DISJOINT = "seeds_not_disjoint"
    STATUS_ACTIVITY_PRECEDES_CREATION = "status_activity_precedes_creation"
    STATUS_TIMESTAMP_NOT_AWARE_DATETIME = "status_timestamp_not_aware_datetime"
    STATUS_NON_ACTIVITY_PRECEDES_CREATION = "status_non_activity_precedes_creation"
    STATUS_UNKNOWN_STATE = "status_unknown_state"
    STATUS_IDLE_CLAIMS_START_OR_TERMINATION = "status_idle_claims_start_or_termination"
    STATUS_RUNNING_NOT_ONLY_A_START = "status_running_not_only_a_start"
    STATUS_TERMINAL_MISSING_RECORDS = "status_terminal_missing_records"
    STATUS_FINISH_PRECEDES_START = "status_finish_precedes_start"
    STATUS_COMPLETED_CARRIES_REASON = "status_completed_carries_reason"
    STATUS_TERMINAL_MISSING_REASON = "status_terminal_missing_reason"
    STATUS_NOTIFICATION_WITHOUT_TERMINATION = "status_notification_without_termination"
    STATUS_NOTIFICATION_PRECEDES_TERMINATION = "status_notification_precedes_termination"
    CHECKPOINT_PATH_NOT_ABSOLUTE = "checkpoint_path_not_absolute"
    CHECKPOINT_NOT_BOUND_TO_PARENT_CELL = "checkpoint_not_bound_to_parent_cell"
    CHECKPOINT_HASH_NOT_SHA256 = "checkpoint_hash_not_sha256"
    CHECKPOINT_NOT_A_DECLARED_TYPE = "checkpoint_not_a_declared_type"
    DDP_WORLD_SIZE_NOT_POSITIVE = "ddp_world_size_not_positive"
    DDP_RANK_ARTIFACTS_NOT_A_MAPPING = "ddp_rank_artifacts_not_a_mapping"
    DDP_RANK_MEMBERSHIP_INCOMPLETE = "ddp_rank_membership_incomplete"
    DDP_RANK_ARTIFACTS_AMBIGUOUS = "ddp_rank_artifacts_ambiguous"
    DDP_ARTIFACT_OUTSIDE_CHECKPOINT = "ddp_artifact_outside_checkpoint"
    DDP_NOT_A_DECLARED_TYPE = "ddp_not_a_declared_type"
    CHAIN_ID_EMPTY = "chain_id_empty"
    CHAIN_COMPONENT_NOT_DECLARED_TYPE = "chain_component_not_declared_type"
    PACKET_CHAIN_NOT_DECLARED_TYPE = "packet_chain_not_declared_type"
    PACKET_HAS_NO_CHAIN = "packet_has_no_chain"
    PACKET_CHAIN_IDS_NOT_UNIQUE = "packet_chain_ids_not_unique"
    PACKET_CHAIN_SEEDS_NOT_UNIQUE = "packet_chain_seeds_not_unique"
    SOURCE_PACKET_NOT_A_PACKET = "source_packet_not_a_packet"
    SOURCE_CHECKPOINT_MISMATCH = "source_checkpoint_mismatch"
    SOURCE_HASH_MISMATCH = "source_hash_mismatch"


class InferencePacketError(ValueError):
    """An inference packet is incomplete, mutable, or mixes evaluation arms."""

    def __init__(self, message: str, *, refusal: PacketRefusal) -> None:
        super().__init__(message)
        self.refusal = refusal


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

_SAMPLER_INPUT_SPEC = MappingProxyType(
    {
        "walkers": int,
        "burn_in_sweeps": int,
        "sampler": MappingProxyType({"walkers": int}),
    }
)

_ADMITTED_PROVENANCE_SCALARS = frozenset({str, int, float, bool, type(None)})

# 2048 ranks x 4 per-rank fields covers Cannon, Polaris, Aurora, and Frontier
# job shapes.  At 8192 leaves this budget no longer separates a declaration
# from a bulk payload; sampler key closure plus its single-pass binding do that.
_PROVENANCE_LEAF_BUDGET = 8192
_PROVENANCE_DEPTH_BUDGET = 6


def _require_declared_sampler_inputs(value: Any, spec: Any, label: str) -> Any:
    """Validate and freeze the sampler block in one read of each mapping."""

    if isinstance(spec, Mapping):
        if not isinstance(value, Mapping):
            raise InferencePacketError(f"{label} must be a mapping", refusal=PacketRefusal.SAMPLER_SUBTREE_NOT_A_MAPPING)
        frozen: dict[str, Any] = {}
        # This is the only read of the caller-owned mapping.  In particular,
        # do not screen and then freeze it through a second items()/dict pass.
        for key, nested in value.items():
            if type(key) is not str:
                raise InferencePacketError(
                    f"{label} keys must be strings",
                    refusal=PacketRefusal.SAMPLER_KEY_NOT_A_STRING,
                )
            if key not in spec:
                raise InferencePacketError(f"{label} contains undeclared input key {key!r}", refusal=PacketRefusal.SAMPLER_KEY_NOT_DECLARED)
            frozen[key] = _require_declared_sampler_inputs(
                nested, spec[key], f"{label}.{key}"
            )
        return MappingProxyType(frozen)
    if isinstance(spec, type):
        if type(value) is not spec:
            raise InferencePacketError(
                f"{label} must be a declared {spec.__name__} sampling datum",
                refusal=PacketRefusal.SAMPLER_VALUE_NOT_A_DECLARED_SCALAR,
            )
        return value
    raise InferencePacketError(f"{label} has no closed declaration", refusal=PacketRefusal.SAMPLER_SPEC_NOT_CLOSED)


def _freeze_provenance(value: Any, label: str) -> Mapping[str, Any]:
    """Freeze a caller-declared provenance mapping, refusing open values."""

    if not isinstance(value, Mapping):
        raise InferencePacketError(
            f"{label} must be a mapping",
            refusal=PacketRefusal.PROVENANCE_NOT_A_MAPPING,
        )
    return _freeze_provenance_node(value, label, 0, [_PROVENANCE_LEAF_BUDGET])


def _freeze_provenance_node(
    value: Any, label: str, depth: int, budget: list[int]
) -> Any:
    if depth > _PROVENANCE_DEPTH_BUDGET:
        raise InferencePacketError(
            f"{label} nests deeper than a provenance declaration",
            refusal=PacketRefusal.PROVENANCE_DEPTH_BUDGET_EXCEEDED,
        )
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise InferencePacketError(
                    f"{label} keys must be strings",
                    refusal=PacketRefusal.PROVENANCE_KEY_NOT_A_STRING,
                )
            frozen[key] = _freeze_provenance_node(
                nested, f"{label}.{key}", depth + 1, budget
            )
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_provenance_node(item, f"{label}[{index}]", depth + 1, budget)
            for index, item in enumerate(value)
        )
    if type(value) in _ADMITTED_PROVENANCE_SCALARS:
        budget[0] -= 1
        if budget[0] < 0:
            raise InferencePacketError(f"{label} exceeds the provenance declaration budget", refusal=PacketRefusal.PROVENANCE_LEAF_BUDGET_EXCEEDED)
        if type(value) is float and not isfinite(value):
            raise InferencePacketError(
                f"{label} must be finite",
                refusal=PacketRefusal.PROVENANCE_VALUE_NOT_FINITE,
            )
        return value
    raise InferencePacketError(f"{label} is not JSON-shaped provenance", refusal=PacketRefusal.PROVENANCE_VALUE_NOT_JSON_SHAPED)


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
        if type(self.burn_in_proposals) is not int:
            raise InferencePacketError(
                "burn_in_proposals must be an integer",
                refusal=PacketRefusal.INTERVAL_BURN_IN_NOT_INT,
            )
        if self.burn_in_proposals < 0:
            raise InferencePacketError(
                "burn_in_proposals must be non-negative",
                refusal=PacketRefusal.INTERVAL_BURN_IN_NEGATIVE,
            )
        if type(self.proposals_between_draws) is not int:
            raise InferencePacketError(
                "proposals_between_draws must be an integer",
                refusal=PacketRefusal.INTERVAL_SPACING_NOT_INT,
            )
        if self.proposals_between_draws < 1:
            raise InferencePacketError(
                "proposals_between_draws must be positive",
                refusal=PacketRefusal.INTERVAL_SPACING_NOT_POSITIVE,
            )
        if type(self.retained_draws) is not int:
            raise InferencePacketError(
                "retained_draws must be an integer",
                refusal=PacketRefusal.INTERVAL_DRAWS_NOT_INT,
            )
        if self.retained_draws < 1:
            raise InferencePacketError(
                "retained_draws must be positive",
                refusal=PacketRefusal.INTERVAL_DRAWS_NOT_POSITIVE,
            )


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
            raise InferencePacketError(
                "all seed provenance values must be positive integers",
                refusal=PacketRefusal.SEED_NOT_POSITIVE_INT,
            )
        if len(set(values)) != len(values):
            raise InferencePacketError(
                "chain seeds must be disjoint from every phase seed",
                refusal=PacketRefusal.SEEDS_NOT_DISJOINT,
            )


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
        if type(self.state) is not ChainState:
            raise InferencePacketError(
                f"unknown chain state {self.state!r}",
                refusal=PacketRefusal.STATUS_UNKNOWN_STATE,
            )
        timestamps = (
            self.created_at,
            self.last_activity_at,
            self.started_at,
            self.finished_at,
            self.finish_notification_at,
        )
        if any(
            timestamp is not None
            and (
                type(timestamp) is not datetime
                or timestamp.tzinfo is None
                or timestamp.utcoffset() is None
            )
            for timestamp in timestamps
        ):
            raise InferencePacketError(
                "chain status timestamps must be exact aware datetimes",
                refusal=PacketRefusal.STATUS_TIMESTAMP_NOT_AWARE_DATETIME,
            )
        if self.terminal_reason is not None and type(self.terminal_reason) is not str:
            raise InferencePacketError(
                "terminal reason must be a string",
                refusal=PacketRefusal.STATUS_TERMINAL_MISSING_REASON,
            )
        if self.last_activity_at < self.created_at:
            raise InferencePacketError(
                "chain activity cannot precede creation",
                refusal=PacketRefusal.STATUS_ACTIVITY_PRECEDES_CREATION,
            )
        if any(
            timestamp is not None and timestamp < self.created_at
            for timestamp in (
                self.started_at,
                self.finished_at,
                self.finish_notification_at,
            )
        ):
            raise InferencePacketError(
                "chain status cannot precede creation",
                refusal=PacketRefusal.STATUS_NON_ACTIVITY_PRECEDES_CREATION,
            )
        if self.state is ChainState.IDLE:
            if any(value is not None for value in (self.started_at, self.finished_at, self.terminal_reason)):
                raise InferencePacketError("idle chain status cannot claim start or termination", refusal=PacketRefusal.STATUS_IDLE_CLAIMS_START_OR_TERMINATION)
        elif self.state is ChainState.RUNNING:
            if self.started_at is None or self.finished_at is not None or self.terminal_reason is not None:
                raise InferencePacketError("running chain status requires only a start record", refusal=PacketRefusal.STATUS_RUNNING_NOT_ONLY_A_START)
        elif self.state in _TERMINAL_STATES:
            if self.started_at is None or self.finished_at is None:
                raise InferencePacketError(
                    "terminal chain status requires start and finish records",
                    refusal=PacketRefusal.STATUS_TERMINAL_MISSING_RECORDS,
                )
            if self.finished_at < self.started_at:
                raise InferencePacketError(
                    "chain finish cannot precede start",
                    refusal=PacketRefusal.STATUS_FINISH_PRECEDES_START,
                )
            if self.last_activity_at < self.finished_at:
                raise InferencePacketError(
                    "chain activity cannot precede finish",
                    refusal=PacketRefusal.STATUS_FINISH_PRECEDES_START,
                )
            if self.state is ChainState.COMPLETED:
                if self.terminal_reason is not None:
                    raise InferencePacketError(
                        "completed chain status cannot carry a failure reason",
                        refusal=PacketRefusal.STATUS_COMPLETED_CARRIES_REASON,
                    )
            elif not (self.terminal_reason or "").strip():
                raise InferencePacketError(
                    "non-completed terminal status requires a reason",
                    refusal=PacketRefusal.STATUS_TERMINAL_MISSING_REASON,
                )
        else:  # Defensive: Enum construction normally makes this unreachable.
            raise InferencePacketError(
                f"unknown chain state {self.state!r}",
                refusal=PacketRefusal.STATUS_UNKNOWN_STATE,
            )
        if self.finish_notification_at is not None:
            if self.finished_at is None:
                raise InferencePacketError(
                    "finish notification requires a terminal status",
                    refusal=PacketRefusal.STATUS_NOTIFICATION_WITHOUT_TERMINATION,
                )
            if self.finish_notification_at < self.finished_at:
                raise InferencePacketError(
                    "finish notification cannot precede termination",
                    refusal=PacketRefusal.STATUS_NOTIFICATION_PRECEDES_TERMINATION,
                )


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
            raise InferencePacketError(
                "checkpoint and parent cell paths must be absolute",
                refusal=PacketRefusal.CHECKPOINT_PATH_NOT_ABSOLUTE,
            )
        parent = parent.resolve()
        checkpoint = checkpoint.resolve()
        if checkpoint.parent.parent != parent or checkpoint.parent.name != "checkpoints":
            raise InferencePacketError(
                "checkpoint must be bound beneath its parent cell checkpoints directory",
                refusal=PacketRefusal.CHECKPOINT_NOT_BOUND_TO_PARENT_CELL,
            )
        if (
            type(self.source_content_hash) is not str
            or len(self.source_content_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_content_hash
            )
        ):
            raise InferencePacketError("source_content_hash must be lowercase sha256", refusal=PacketRefusal.CHECKPOINT_HASH_NOT_SHA256)
        topology = _freeze_provenance(self.topology_provenance, "topology provenance")
        object.__setattr__(self, "parent_cell_path", parent)
        object.__setattr__(self, "checkpoint_path", checkpoint)
        object.__setattr__(self, "topology_provenance", topology)


@dataclass(frozen=True)
class DistributedCheckpointProvenance:
    """Complete distributed artifact membership for a checkpoint, when used."""

    world_size: int
    rank_artifacts: Mapping[int, Path]

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size < 1:
            raise InferencePacketError(
                "distributed checkpoint world_size must be positive",
                refusal=PacketRefusal.DDP_WORLD_SIZE_NOT_POSITIVE,
            )
        if not isinstance(self.rank_artifacts, Mapping):
            raise InferencePacketError(
                "distributed checkpoint rank_artifacts must be a mapping",
                refusal=PacketRefusal.DDP_RANK_ARTIFACTS_NOT_A_MAPPING,
            )
        rank_artifacts = dict(self.rank_artifacts.items())
        if any(type(rank) is not int for rank in rank_artifacts):
            raise InferencePacketError(
                "distributed checkpoint ranks must be exact integers",
                refusal=PacketRefusal.DDP_RANK_MEMBERSHIP_INCOMPLETE,
            )
        expected = set(range(self.world_size))
        if set(rank_artifacts) != expected:
            raise InferencePacketError(
                "distributed checkpoint artifacts must contain each rank exactly once",
                refusal=PacketRefusal.DDP_RANK_MEMBERSHIP_INCOMPLETE,
            )
        artifacts = {
            rank: Path(path).resolve() for rank, path in rank_artifacts.items()
        }
        if len(set(artifacts.values())) != self.world_size:
            raise InferencePacketError(
                "distributed checkpoint rank artifacts must be unambiguous",
                refusal=PacketRefusal.DDP_RANK_ARTIFACTS_AMBIGUOUS,
            )
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
            raise InferencePacketError(
                "chain_id must be non-empty",
                refusal=PacketRefusal.CHAIN_ID_EMPTY,
            )
        if type(self.seeds) is not ChainSeedProvenance:
            raise InferencePacketError("chain seeds must be declared chain seed provenance", refusal=PacketRefusal.CHAIN_COMPONENT_NOT_DECLARED_TYPE)
        if type(self.interval) is not SamplingInterval:
            raise InferencePacketError("chain interval must be a declared sampling interval", refusal=PacketRefusal.CHAIN_COMPONENT_NOT_DECLARED_TYPE)
        if type(self.status) is not ChainStatus:
            raise InferencePacketError("chain status must be a declared chain status", refusal=PacketRefusal.CHAIN_COMPONENT_NOT_DECLARED_TYPE)


@dataclass(frozen=True)
class InferencePacket:
    """Immutable launch packet for independent sampler inference only."""

    checkpoint: CheckpointReference
    independent_sampler_inputs: Mapping[str, Any]
    chains: tuple[IndependentChain, ...]
    distributed_checkpoint: DistributedCheckpointProvenance | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "chains", tuple(self.chains))
        frozen_inputs = _require_declared_sampler_inputs(
            self.independent_sampler_inputs,
            _SAMPLER_INPUT_SPEC,
            "independent sampler inputs",
        )
        # Store exactly the value returned by the one-pass validator.  A
        # second read or a dict(value) conversion would reopen the TOCTOU hole.
        object.__setattr__(self, "independent_sampler_inputs", frozen_inputs)
        if type(self.checkpoint) is not CheckpointReference:
            raise InferencePacketError("inference packet checkpoint must be a declared checkpoint reference", refusal=PacketRefusal.CHECKPOINT_NOT_A_DECLARED_TYPE)
        if self.distributed_checkpoint is not None and type(self.distributed_checkpoint) is not DistributedCheckpointProvenance:
            raise InferencePacketError("inference packet distributed checkpoint must be a declared provenance", refusal=PacketRefusal.DDP_NOT_A_DECLARED_TYPE)
        if not all(type(chain) is IndependentChain for chain in self.chains):
            raise InferencePacketError("inference packet chains must be independent chain records", refusal=PacketRefusal.PACKET_CHAIN_NOT_DECLARED_TYPE)
        if not self.chains:
            raise InferencePacketError(
                "inference packet requires at least one independent chain",
                refusal=PacketRefusal.PACKET_HAS_NO_CHAIN,
            )
        if len({chain.chain_id for chain in self.chains}) != len(self.chains):
            raise InferencePacketError(
                "inference packet chain IDs must be unique",
                refusal=PacketRefusal.PACKET_CHAIN_IDS_NOT_UNIQUE,
            )
        if len({chain.seeds.chain_seed for chain in self.chains}) != len(self.chains):
            raise InferencePacketError(
                "inference packet chain seeds must be unique",
                refusal=PacketRefusal.PACKET_CHAIN_SEEDS_NOT_UNIQUE,
            )
        if self.distributed_checkpoint is not None:
            for artifact in self.distributed_checkpoint.rank_artifacts.values():
                if not artifact.resolve().is_relative_to(
                    self.checkpoint.checkpoint_path.resolve()
                ):
                    raise InferencePacketError(
                        "distributed artifact lies outside its checkpoint directory",
                        refusal=PacketRefusal.DDP_ARTIFACT_OUTSIDE_CHECKPOINT,
                    )


def launch_inference_packet(
    source_packet: Any,
    checkpoint: CheckpointReference,
    chains: Sequence[IndependentChain],
    *,
    distributed_checkpoint: DistributedCheckpointProvenance | None = None,
) -> InferencePacket:
    """Bind an L2 packet to fresh chains and provenance by structural fields.

    The structural interface keeps this boundary independent of L2's
    concrete module type.  It verifies that the source names the same
    checkpoint and content identity before copying only its already-separated
    independent-sampler input block.
    """

    def read_source_attribute(name: str) -> Any:
        try:
            getattr_static(source_packet, name)
        except AttributeError as error:
            raise InferencePacketError(
                "source packet must be an independent-sampler test packet",
                refusal=PacketRefusal.SOURCE_PACKET_NOT_A_PACKET,
            ) from error
        return getattr(source_packet, name)

    source_checkpoint = Path(read_source_attribute("checkpoint_path"))
    source_hash = read_source_attribute("source_content_hash")
    sampler_inputs = read_source_attribute("independent_sampler_inputs")
    if source_checkpoint != checkpoint.checkpoint_path:
        raise InferencePacketError(
            "source packet checkpoint does not match checkpoint reference",
            refusal=PacketRefusal.SOURCE_CHECKPOINT_MISMATCH,
        )
    if source_hash != checkpoint.source_content_hash:
        raise InferencePacketError(
            "source packet content hash does not match checkpoint reference",
            refusal=PacketRefusal.SOURCE_HASH_MISMATCH,
        )
    return InferencePacket(
        checkpoint=checkpoint,
        independent_sampler_inputs=sampler_inputs,
        chains=tuple(chains),
        distributed_checkpoint=distributed_checkpoint,
    )
