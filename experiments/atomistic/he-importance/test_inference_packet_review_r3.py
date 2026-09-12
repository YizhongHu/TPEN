"""Round-three review arms for the L4a inference-packet boundary.

Each arm is a witness for one reviewer-designed finding.  Arms are written
remedy-agnostically where a finding admits more than one correct fix.
"""

from __future__ import annotations

import copy
import importlib.util
import pickle
import sys
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest


def _reuse_or_load(module_name: str, path: Path) -> object:
    """Reuse a module already loaded from path before creating an alias."""

    if module_name in sys.modules:
        return sys.modules[module_name]
    target = path.resolve()
    for candidate in tuple(sys.modules.values()):
        candidate_path = getattr(candidate, "__file__", None)
        if candidate_path is not None and Path(candidate_path).resolve() == target:
            sys.modules.update({module_name: candidate})
            return candidate
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.update({module_name: module})
    spec.loader.exec_module(module)
    return module


inference_packet = _reuse_or_load(
    "he_importance_inference_packet", Path(__file__).with_name("inference_packet.py")
)


_HASH = "a" * 64
_CREATED = datetime(2026, 9, 9, 4, 30, tzinfo=UTC)


def _checkpoint(tmp_path: Path, topology: object = None) -> object:
    parent = (tmp_path / "O1" / _HASH).resolve()
    return inference_packet.CheckpointReference(
        parent_cell_path=parent,
        checkpoint_path=parent / "checkpoints" / "update-00050000",
        source_content_hash=_HASH,
        topology_provenance={"world_size": 1, "launcher": "test"}
        if topology is None
        else topology,
    )


def _seeds(index: int) -> object:
    return inference_packet.ChainSeedProvenance(
        training_seed=100 + index,
        calibration_seed=200 + index,
        inference_seed=300 + index,
        chain_seed=400 + index,
    )


def _chain(index: int) -> object:
    status = inference_packet.ChainStatus(
        state=inference_packet.ChainState.IDLE,
        created_at=_CREATED,
        last_activity_at=_CREATED,
    )
    return inference_packet.IndependentChain(
        chain_id=f"chain-{index}",
        seeds=_seeds(index),
        interval=inference_packet.SamplingInterval(10_000, 10, 32_768),
        status=status,
    )


def _source(checkpoint: object, inputs: object | None = None) -> object:
    return SimpleNamespace(
        checkpoint_path=checkpoint.checkpoint_path,
        source_content_hash=checkpoint.source_content_hash,
        independent_sampler_inputs={"sampler": {"walkers": 4_096}}
        if inputs is None
        else inputs,
    )


def _packet(tmp_path: Path) -> object:
    checkpoint = _checkpoint(tmp_path)
    return inference_packet.launch_inference_packet(
        _source(checkpoint), checkpoint, (_chain(0),)
    )


class _RecordingTZ(tzinfo):
    """Fixed -10h zone that records every ``utcoffset`` interrogation."""

    def __init__(self) -> None:
        self.calls = 0

    def utcoffset(self, dt: datetime | None) -> timedelta:
        self.calls += 1
        return timedelta(hours=-10)

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, dt: datetime | None) -> str:
        return "rec"


class _FlipTZ(tzinfo):
    """-10h for the first ``flip_after`` reads, then +10h for every later read."""

    def __init__(self, flip_after: int) -> None:
        self.flip_after = flip_after
        self.calls = 0

    def utcoffset(self, dt: datetime | None) -> timedelta:
        self.calls += 1
        if self.calls <= self.flip_after:
            return timedelta(hours=-10)
        return timedelta(hours=+10)

    def dst(self, dt: datetime | None) -> timedelta:
        return timedelta(0)

    def tzname(self, dt: datetime | None) -> str:
        return "flip"


def _completed_status(finished_zone: tzinfo) -> object:
    return inference_packet.ChainStatus(
        state=inference_packet.ChainState.COMPLETED,
        created_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        started_at=datetime(2026, 1, 2, 0, 0, tzinfo=UTC),
        last_activity_at=datetime(2026, 1, 2, 11, 0, tzinfo=UTC),
        finished_at=datetime(2026, 1, 2, 0, 30, tzinfo=finished_zone),
        terminal_reason=None,
    )


def test_a_stored_terminal_record_keeps_a_fixed_chronology_after_construction() -> None:
    """A record admitted as ordered must still read as ordered once stored."""

    # Calibrate: count the offset reads one successful construction consumes,
    # so the flip lands exactly after validation and before any later read.
    recorder = _RecordingTZ()
    _completed_status(recorder)
    try:
        status = _completed_status(_FlipTZ(recorder.calls))
    except inference_packet.InferencePacketError:
        return
    assert status.finished_at.astimezone(UTC) >= status.started_at.astimezone(UTC)


def test_extreme_aware_timestamps_are_refused_with_a_declared_member() -> None:
    """Timestamps whose UTC instant overflows must refuse, not crash."""

    floor = datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))
    with pytest.raises(inference_packet.InferencePacketError):
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.IDLE,
            created_at=floor,
            last_activity_at=floor,
        )


def test_terminal_activity_before_finish_is_not_reported_as_finish_precedes_start() -> None:
    """Activity-before-finish must not be attributed to the finish/start member."""

    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.COMPLETED,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            last_activity_at=datetime(2026, 1, 5, tzinfo=UTC),
            started_at=datetime(2026, 1, 2, tzinfo=UTC),
            finished_at=datetime(2026, 1, 10, tzinfo=UTC),
            terminal_reason=None,
        )
    assert (
        excinfo.value.refusal
        is not inference_packet.PacketRefusal.STATUS_FINISH_PRECEDES_START
    )


def test_a_running_chain_cannot_record_activity_before_its_start() -> None:
    """A running chain's last activity cannot predate the start it records."""

    with pytest.raises(inference_packet.InferencePacketError):
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.RUNNING,
            created_at=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
            last_activity_at=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            started_at=datetime(2026, 1, 5, 0, 0, tzinfo=UTC),
        )


def test_a_valid_packet_survives_a_pickle_roundtrip(tmp_path: Path) -> None:
    """A launch packet must cross a process boundary to reach a scheduler."""

    packet = _packet(tmp_path)
    assert pickle.loads(pickle.dumps(packet)) == packet


def test_a_valid_packet_can_be_deep_copied(tmp_path: Path) -> None:
    """An immutable packet must be copyable without losing its contents."""

    packet = _packet(tmp_path)
    assert copy.deepcopy(packet) == packet


def test_a_non_pathlike_source_checkpoint_path_is_refused_not_crashed(
    tmp_path: Path,
) -> None:
    """A source whose checkpoint path is not path-like must refuse, not crash."""

    checkpoint = _checkpoint(tmp_path)
    bad_source = SimpleNamespace(
        checkpoint_path=5,
        source_content_hash=checkpoint.source_content_hash,
        independent_sampler_inputs={"sampler": {"walkers": 4_096}},
    )
    with pytest.raises(inference_packet.InferencePacketError):
        inference_packet.launch_inference_packet(
            bad_source, checkpoint, (_chain(0),)
        )


def test_a_non_pathlike_rank_artifact_is_refused_not_crashed() -> None:
    """A rank artifact that is not path-like must refuse, not crash."""

    with pytest.raises(inference_packet.InferencePacketError):
        inference_packet.DistributedCheckpointProvenance(1, {0: 5})


def test_a_non_iterable_chains_argument_is_refused_not_crashed(
    tmp_path: Path,
) -> None:
    """A non-iterable chains argument must refuse, not crash."""

    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError):
        inference_packet.InferencePacket(checkpoint, {"walkers": 4_096}, 5, None)


class _GetattrSource:
    """A structural source whose only attribute interface is ``__getattr__``."""

    def __init__(self, fields: dict[str, object]) -> None:
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> object:
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name]
        raise AttributeError(name)


def test_a_getattr_backed_structural_source_is_not_falsely_refused(
    tmp_path: Path,
) -> None:
    """A functional structural source must not be refused for its lookup route."""

    checkpoint = _checkpoint(tmp_path)
    source = _GetattrSource(
        {
            "checkpoint_path": checkpoint.checkpoint_path,
            "source_content_hash": checkpoint.source_content_hash,
            "independent_sampler_inputs": {"sampler": {"walkers": 4_096}},
        }
    )
    packet = inference_packet.launch_inference_packet(
        source, checkpoint, (_chain(0),)
    )
    assert type(packet) is inference_packet.InferencePacket


def test_topology_and_distributed_world_size_cannot_contradict(
    tmp_path: Path,
) -> None:
    """A packet cannot record two different world sizes for one checkpoint."""

    checkpoint = _checkpoint(tmp_path, topology={"world_size": 1})
    artifacts = {
        rank: checkpoint.checkpoint_path / f"rank-{rank}.pt" for rank in range(2)
    }
    distributed = inference_packet.DistributedCheckpointProvenance(2, artifacts)
    with pytest.raises(inference_packet.InferencePacketError):
        inference_packet.launch_inference_packet(
            _source(checkpoint),
            checkpoint,
            (_chain(0), _chain(1)),
            distributed_checkpoint=distributed,
        )


def test_an_honest_zoneinfo_terminal_record_reads_the_same_chronology_on_every_read() -> None:
    """An honest civil-time record across a DST fold reads identically twice."""

    zone = ZoneInfo("America/New_York")
    started = datetime(2026, 11, 1, 1, 45, tzinfo=zone, fold=0)
    finished = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1)
    status = inference_packet.ChainStatus(
        state=inference_packet.ChainState.COMPLETED,
        created_at=datetime(2026, 11, 1, 0, 0, tzinfo=zone),
        started_at=started,
        finished_at=finished,
        last_activity_at=finished,
        terminal_reason=None,
    )
    first_started = status.started_at.astimezone(UTC)
    second_started = status.started_at.astimezone(UTC)
    first_finished = status.finished_at.astimezone(UTC)
    second_finished = status.finished_at.astimezone(UTC)
    assert first_started == second_started
    assert first_finished == second_finished
    assert first_finished > first_started


def test_ordinary_utc_range_timestamps_do_not_overflow_the_instant_policy() -> None:
    """Timestamps at the far end of the representable UTC range stay admissible."""

    created = datetime(9999, 1, 1, tzinfo=UTC)
    status = inference_packet.ChainStatus(
        state=inference_packet.ChainState.COMPLETED,
        created_at=created,
        started_at=created + timedelta(days=1),
        finished_at=created + timedelta(days=2),
        last_activity_at=created + timedelta(days=2),
        terminal_reason=None,
    )
    assert status.state is inference_packet.ChainState.COMPLETED


def test_refusal_member_names_are_substring_free() -> None:
    """No refusal name may shadow another, or a mention census over-counts."""

    members = list(inference_packet.PacketRefusal)
    shadowing = [
        (left.name, right.name)
        for left in members
        for right in members
        if left is not right and left.name in right.name
    ]
    assert shadowing == []
