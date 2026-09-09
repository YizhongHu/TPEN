"""Contract tests for expensive independent-sampler inference packets."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "he_importance_inference_packet", Path(__file__).with_name("inference_packet.py")
)
assert _SPEC is not None and _SPEC.loader is not None
inference_packet = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = inference_packet
_SPEC.loader.exec_module(inference_packet)


_HASH = "a" * 64
_CREATED = datetime(2026, 9, 9, 4, 30, tzinfo=UTC)


def _checkpoint(tmp_path: Path) -> object:
    parent = (tmp_path / "O1" / _HASH).resolve()
    return inference_packet.CheckpointReference(
        parent_cell_path=parent,
        checkpoint_path=parent / "checkpoints" / "update-00050000",
        source_content_hash=_HASH,
        topology_provenance={"world_size": 1, "launcher": "test"},
    )


def _seeds(index: int) -> object:
    return inference_packet.ChainSeedProvenance(
        training_seed=100 + index,
        calibration_seed=200 + index,
        inference_seed=300 + index,
        chain_seed=400 + index,
    )


def _chain(index: int, state: object = inference_packet.ChainState.IDLE) -> object:
    if state is inference_packet.ChainState.IDLE:
        status = inference_packet.ChainStatus(state=state, created_at=_CREATED, last_activity_at=_CREATED)
    elif state is inference_packet.ChainState.RUNNING:
        status = inference_packet.ChainStatus(
            state=state,
            created_at=_CREATED,
            started_at=_CREATED + timedelta(seconds=1),
            last_activity_at=_CREATED + timedelta(seconds=2),
        )
    else:
        reason = None if state is inference_packet.ChainState.COMPLETED else "recorded terminal failure"
        status = inference_packet.ChainStatus(
            state=state,
            created_at=_CREATED,
            started_at=_CREATED + timedelta(seconds=1),
            finished_at=_CREATED + timedelta(seconds=3),
            last_activity_at=_CREATED + timedelta(seconds=3),
            terminal_reason=reason,
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
        independent_sampler_inputs={"sampler": {"walkers": 4_096}} if inputs is None else inputs,
    )


def test_launch_packet_keeps_independent_inputs_and_all_chain_provenance_immutable(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    packet = inference_packet.launch_inference_packet(
        _source(checkpoint), checkpoint, (_chain(0), _chain(1))
    )
    assert packet.chains[0].status.state is inference_packet.ChainState.IDLE
    assert packet.chains[0].seeds.chain_seed != packet.chains[1].seeds.chain_seed
    assert packet.checkpoint.parent_cell_path == checkpoint.checkpoint_path.parent.parent
    with pytest.raises(TypeError):
        packet.independent_sampler_inputs["sampler"]["walkers"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        packet.checkpoint.topology_provenance["world_size"] = 8  # type: ignore[index]


@pytest.mark.parametrize(
    "state",
    [
        inference_packet.ChainState.COMPLETED,
        inference_packet.ChainState.ERROR,
        inference_packet.ChainState.CANCELLED,
        inference_packet.ChainState.DEAD,
    ],
)
def test_every_terminal_state_is_self_describing_without_a_notification(state: object) -> None:
    """Terminal error paths do not get mistaken for an unobserved completion."""

    chain = _chain(0, state)
    assert chain.status.finished_at is not None
    assert chain.status.finish_notification_at is None
    assert chain.status.state is state
    if state is inference_packet.ChainState.COMPLETED:
        assert chain.status.terminal_reason is None
    else:
        assert chain.status.terminal_reason == "recorded terminal failure"


def test_idle_dead_and_completed_are_distinguishable_from_recorded_status_alone() -> None:
    statuses = [_chain(0, inference_packet.ChainState.IDLE).status]
    statuses.extend(_chain(index, state).status for index, state in enumerate(
        (inference_packet.ChainState.DEAD, inference_packet.ChainState.COMPLETED), start=1
    ))
    assert [status.state.value for status in statuses] == ["idle", "dead", "completed"]
    assert statuses[0].finished_at is None
    assert all(status.finished_at is not None for status in statuses[1:])


@pytest.mark.parametrize(
    "state",
    [inference_packet.ChainState.ERROR, inference_packet.ChainState.CANCELLED, inference_packet.ChainState.DEAD],
)
def test_non_success_terminal_status_requires_its_own_failure_witness(state: object) -> None:
    with pytest.raises(inference_packet.InferencePacketError, match="requires a reason"):
        inference_packet.ChainStatus(
            state=state,
            created_at=_CREATED,
            started_at=_CREATED,
            finished_at=_CREATED + timedelta(seconds=1),
            last_activity_at=_CREATED + timedelta(seconds=1),
        )


@pytest.mark.parametrize(
    "walker_key",
    ["training_walker_state", "cheap_arm_walker_state", "ranking_walker_state"],
)
def test_inherited_train_or_cheap_walker_state_is_rejected_by_the_packet_boundary(
    tmp_path: Path, walker_key: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError, match="cannot reuse train or ranking walkers"):
        inference_packet.launch_inference_packet(
            _source(checkpoint, {walker_key: {"opaque": "state"}}), checkpoint, (_chain(0),)
        )


def test_checkpoint_reference_pins_parent_cell_not_only_checkpoint_name(tmp_path: Path) -> None:
    parent = (tmp_path / "O1" / _HASH).resolve()
    with pytest.raises(inference_packet.InferencePacketError, match="bound beneath its parent cell"):
        inference_packet.CheckpointReference(
            parent_cell_path=parent,
            checkpoint_path=(tmp_path / "other" / "checkpoints" / "update-00050000").resolve(),
            source_content_hash=_HASH,
            topology_provenance={},
        )


@pytest.mark.parametrize(
    "rank_artifacts",
    [
        {0: Path("rank-0")},
        {0: Path("same"), 1: Path("same")},
        {0: Path("rank-0"), 2: Path("rank-2")},
    ],
)
def test_partial_or_ambiguous_distributed_checkpoint_artifacts_are_rejected(rank_artifacts: object) -> None:
    with pytest.raises(inference_packet.InferencePacketError, match="artifacts must contain each rank exactly once|unambiguous"):
        inference_packet.DistributedCheckpointProvenance(world_size=2, rank_artifacts=rank_artifacts)


def test_distributed_artifact_must_remain_under_the_bound_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    distributed = inference_packet.DistributedCheckpointProvenance(
        world_size=1, rank_artifacts={0: (tmp_path / "foreign-rank-0").resolve()}
    )
    with pytest.raises(inference_packet.InferencePacketError, match="outside its checkpoint directory"):
        inference_packet.launch_inference_packet(
            _source(checkpoint), checkpoint, (_chain(0),), distributed_checkpoint=distributed
        )


def test_source_packet_checkpoint_and_hash_are_both_binding(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    source = _source(checkpoint)
    source.source_content_hash = "b" * 64
    with pytest.raises(inference_packet.InferencePacketError, match="content hash does not match"):
        inference_packet.launch_inference_packet(source, checkpoint, (_chain(0),))
