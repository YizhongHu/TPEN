"""Contract tests for expensive independent-sampler inference packets."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple
from types import MappingProxyType, SimpleNamespace

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "he_importance_inference_packet", Path(__file__).with_name("inference_packet.py")
)
assert _SPEC is not None and _SPEC.loader is not None
inference_packet = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = inference_packet
_SPEC.loader.exec_module(inference_packet)


_L2_TEST_SPEC = importlib.util.spec_from_file_location(
    "he_importance_test_stage_coordinate", Path(__file__).with_name("test_stage_coordinate.py")
)
assert _L2_TEST_SPEC is not None and _L2_TEST_SPEC.loader is not None
l2_tests = importlib.util.module_from_spec(_L2_TEST_SPEC)
sys.modules[_L2_TEST_SPEC.name] = l2_tests
_L2_TEST_SPEC.loader.exec_module(l2_tests)
stage_coordinate = l2_tests.stage_coordinate


_HASH = "a" * 64
_CREATED = datetime(2026, 9, 9, 4, 30, tzinfo=UTC)
_WALKER_PAYLOAD = {
    "marker": 0.5731904418265731,
    "positions": [0.1, 0.2],
    "provenance": "COPIED-FROM-TRAINING-RUN-WALKERS",
}


@dataclass(frozen=True)
class _WalkerBlob:
    marker: float
    positions: tuple[float, float]


class _WalkerRow(NamedTuple):
    marker: float
    positions: tuple[float, float]


class _LyingObject:
    def __getattr__(self, name: str) -> object:
        return {"marker": 0.5731904418265731}[name]


@dataclass(frozen=True)
class _ChainLookalike:
    chain_id: str
    seeds: object
    interval: object
    status: object


@dataclass(frozen=True)
class _ForgedIndependentChain(inference_packet.IndependentChain):
    def __post_init__(self) -> None:
        # This simulates a frozen-dataclass subclass that suppresses the base
        # guard; the packet must still require the exact declared type.
        pass


@dataclass(frozen=True)
class _ForgedSeedProvenance(inference_packet.ChainSeedProvenance):
    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True)
class _ForgedSamplingInterval(inference_packet.SamplingInterval):
    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True)
class _ForgedChainStatus(inference_packet.ChainStatus):
    def __post_init__(self) -> None:
        pass


class _IntMarker(IntEnum):
    VALUE = 1


class _StringMarker(str):
    pass


class _TwoFacedMapping(Mapping[str, object]):
    """Return a safe value on the first items() read and a marker on the next."""

    def __init__(self) -> None:
        self.items_reads = 0

    def __getitem__(self, key: str) -> object:
        if key != "walkers":
            raise KeyError(key)
        return 4_096

    def __iter__(self):
        return iter(("walkers",))

    def __len__(self) -> int:
        return 1

    def items(self):
        self.items_reads += 1
        value = 4_096 if self.items_reads == 1 else _WALKER_PAYLOAD
        return {"walkers": value}.items()


class _TwoFacedDict(dict[str, object]):
    """Store a marker but lie through items(), exposing dict fast paths."""

    def __init__(self) -> None:
        super().__init__({"walkers": _WALKER_PAYLOAD})
        self.items_reads = 0

    def items(self):
        self.items_reads += 1
        return {"walkers": 4_096}.items()


def _assert_refusal(excinfo: pytest.ExceptionInfo[BaseException], refusal: object) -> None:
    assert type(excinfo.value) is inference_packet.InferencePacketError
    assert excinfo.value.refusal is refusal


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


def _chain(index: int, state: object = inference_packet.ChainState.IDLE) -> object:
    if state is inference_packet.ChainState.IDLE:
        status = inference_packet.ChainStatus(
            state=state, created_at=_CREATED, last_activity_at=_CREATED
        )
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


_DEFAULT_INPUTS = object()


def _source(checkpoint: object, inputs: object = _DEFAULT_INPUTS) -> object:
    return SimpleNamespace(
        checkpoint_path=checkpoint.checkpoint_path,
        source_content_hash=checkpoint.source_content_hash,
        independent_sampler_inputs={"sampler": {"walkers": 4_096}}
        if inputs is _DEFAULT_INPUTS
        else inputs,
    )


def _real_l2_packet(tmp_path: Path, sampler_inputs: object) -> object:
    """Produce an L2 IndependentSamplerTestPacket through the real producer."""

    packets = stage_coordinate.materialize_job_packets(
        l2_tests._packet_source_cells(tmp_path),
        stage_coordinate.CheckpointCadence(1_000, (1_000,)),
        {"statistic": "logabs_variance"},
        (stage_coordinate.RankingStatistic.LOGABS_VARIANCE,),
        sampler_inputs,
        ddp_provenance={"launcher": "operator-supplied"},
    )
    return packets.independent_sampler_test[0]


def _checkpoint_for(packet: object) -> object:
    """Bind an L4a CheckpointReference to a real L2 packet's checkpoint."""

    return inference_packet.CheckpointReference(
        parent_cell_path=packet.checkpoint_path.parent.parent,
        checkpoint_path=packet.checkpoint_path,
        source_content_hash=packet.source_content_hash,
        topology_provenance={"world_size": 1, "launcher": "test"},
    )


def test_packet_inputs_and_topology_remain_immutable(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    packet = inference_packet.launch_inference_packet(
        _source(checkpoint), checkpoint, (_chain(0), _chain(1))
    )
    assert packet.chains[0].status.state is inference_packet.ChainState.IDLE
    assert packet.chains[0].seeds.chain_seed != packet.chains[1].seeds.chain_seed
    assert type(packet.independent_sampler_inputs) is MappingProxyType
    assert type(packet.independent_sampler_inputs["sampler"]) is MappingProxyType
    assert type(packet.checkpoint.topology_provenance) is MappingProxyType
    with pytest.raises(TypeError):
        packet.independent_sampler_inputs["sampler"]["walkers"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        packet.checkpoint.topology_provenance["world_size"] = 8  # type: ignore[index]


def test_sampler_mapping_is_screened_and_stored_in_one_pass(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    inputs = _TwoFacedMapping()
    packet = inference_packet.launch_inference_packet(
        _source(checkpoint, inputs), checkpoint, (_chain(0),)
    )
    assert packet.independent_sampler_inputs == {"walkers": 4_096}
    assert inputs.items_reads == 1


def test_sampler_dict_subclass_items_is_screened_and_stored_in_one_pass(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    inputs = _TwoFacedDict()
    packet = inference_packet.launch_inference_packet(
        _source(checkpoint, inputs), checkpoint, (_chain(0),)
    )
    assert packet.independent_sampler_inputs == {"walkers": 4_096}
    assert inputs.items_reads == 1


def test_real_producer_walker_payload_under_the_declared_walkers_key_is_refused_by_l4a(
    tmp_path: Path,
) -> None:
    l2_packet = _real_l2_packet(tmp_path, {"walkers": _WALKER_PAYLOAD})
    assert l2_packet.independent_sampler_inputs["walkers"]["marker"] == 0.5731904418265731
    checkpoint = _checkpoint_for(l2_packet)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(l2_packet, checkpoint, (_chain(0),))
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_VALUE_NOT_A_DECLARED_SCALAR)


def test_real_producer_walker_payload_nested_under_sampler_is_refused_by_l4a(
    tmp_path: Path,
) -> None:
    l2_packet = _real_l2_packet(tmp_path, {"sampler": {"walkers": _WALKER_PAYLOAD}})
    assert l2_packet.independent_sampler_inputs["sampler"]["walkers"]["marker"] == 0.5731904418265731
    checkpoint = _checkpoint_for(l2_packet)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(l2_packet, checkpoint, (_chain(0),))
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_VALUE_NOT_A_DECLARED_SCALAR)


def test_direct_source_interface_walker_payload_under_an_undeclared_key_is_refused_by_l4a(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint, {"train_walker_state": _WALKER_PAYLOAD}),
            checkpoint,
            (_chain(0),),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_KEY_NOT_DECLARED)


@pytest.mark.parametrize(
    "walker_key",
    ["training_walker_state", "cheap_arm_walker_state", "ranking_walker_state"],
)
def test_the_three_historically_forbidden_names_are_refused_by_l4a_key_closure(
    tmp_path: Path, walker_key: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint, {walker_key: _WALKER_PAYLOAD}),
            checkpoint,
            (_chain(0),),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_KEY_NOT_DECLARED)


@pytest.mark.parametrize(
    "value",
    [0.5731904418265731, "4096", True, (1, 2, 3), MappingProxyType({"n": 4_096})],
    ids=["float", "str", "bool", "small_tuple", "mappingproxy"],
)
def test_declared_scalar_shape_refuses_a_non_integer_at_a_declared_key(
    tmp_path: Path, value: object
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint, {"walkers": value}), checkpoint, (_chain(0),)
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_VALUE_NOT_A_DECLARED_SCALAR)


def test_declared_key_closure_refuses_an_undeclared_key_carrying_an_admitted_scalar(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint, {"walkers": 4_096, "burn_in": 100}),
            checkpoint,
            (_chain(0),),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_KEY_NOT_DECLARED)


def test_declared_subtree_requires_a_mapping_where_the_spec_declares_one(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint, {"sampler": 4_096}), checkpoint, (_chain(0),)
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_SUBTREE_NOT_A_MAPPING)


def test_sampler_input_keys_must_be_strings(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint, {1: 4_096}), checkpoint, (_chain(0),)
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_KEY_NOT_A_STRING)


@pytest.mark.parametrize("value", [[], "sampler", 4_096, None], ids=["list", "str", "int", "None"])
def test_sampler_inputs_must_be_a_mapping(tmp_path: Path, value: object) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(_source(checkpoint, value), checkpoint, (_chain(0),))
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_SUBTREE_NOT_A_MAPPING)


def test_the_sampler_spec_language_admits_no_caller_open_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(
        inference_packet,
        "_SAMPLER_INPUT_SPEC",
        MappingProxyType({"walkers": None}),
    )
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint, {"walkers": 4_096}), checkpoint, (_chain(0),)
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_SPEC_NOT_CLOSED)


@pytest.mark.parametrize(
    "value",
    [_WalkerBlob(0.1, (0.2, 0.3)), _WalkerRow(0.1, (0.2, 0.3)), SimpleNamespace(marker=0.1), _LyingObject()],
    ids=["dataclass", "namedtuple", "simplenamespace", "lying_getattr"],
)
def test_a_record_type_at_a_declared_scalar_key_is_refused(
    tmp_path: Path, value: object
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint, {"walkers": value}), checkpoint, (_chain(0),)
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_VALUE_NOT_A_DECLARED_SCALAR)


@pytest.mark.parametrize(
    "value",
    [
        _WalkerBlob(0.1, (0.2, 0.3)),
        Path("/x"),
        datetime(2026, 1, 1, tzinfo=UTC),
        {"set-item"},
        b"bytes",
        lambda: None,
    ],
    ids=["dataclass", "path", "datetime", "set", "bytes", "callable"],
)
def test_a_record_type_inside_topology_provenance_is_refused_rather_than_frozen_verbatim(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _checkpoint(tmp_path, {"launcher": value})
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PROVENANCE_VALUE_NOT_JSON_SHAPED)


@pytest.mark.parametrize(
    "value",
    [inference_packet.ChainState.IDLE, _IntMarker.VALUE, _StringMarker("marker")],
    ids=["chain_state", "int_enum", "str_subclass"],
)
def test_str_and_int_subclasses_are_not_admitted_as_provenance_scalars(
    tmp_path: Path, value: object
) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _checkpoint(tmp_path, {"launcher": value})
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PROVENANCE_VALUE_NOT_JSON_SHAPED)


def test_a_namedtuple_of_json_scalars_is_normalized_to_a_plain_tuple(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, {"launcher": _WalkerRow(0.1, (0.2, 0.3))})
    frozen = checkpoint.topology_provenance["launcher"]
    assert type(frozen) is tuple
    assert not isinstance(frozen, _WalkerRow)
    assert frozen == (0.1, (0.2, 0.3))


def _deep_provenance(depth: int) -> object:
    value: object = 1
    for index in reversed(range(depth)):
        value = {f"level_{index}": value}
    return value


def test_bulk_json_payload_in_topology_provenance_is_refused_by_the_leaf_budget(
    tmp_path: Path,
) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _checkpoint(tmp_path, {"positions": [0.1] * 8_193})
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PROVENANCE_LEAF_BUDGET_EXCEEDED)


def test_topology_provenance_depth_budget_refuses_a_deeply_nested_declaration(
    tmp_path: Path,
) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _checkpoint(tmp_path, _deep_provenance(8))
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PROVENANCE_DEPTH_BUDGET_EXCEEDED)


@pytest.mark.parametrize("leaf_count", [2, 8_192, 8_193], ids=["2", "8192", "8193"])
def test_the_declared_provenance_budget_admits_a_real_declaration_and_refuses_one_leaf_more(
    tmp_path: Path, leaf_count: int
) -> None:
    topology = {f"leaf_{index}": index for index in range(leaf_count)}
    if leaf_count <= 8_192:
        checkpoint = _checkpoint(tmp_path, topology)
        assert len(checkpoint.topology_provenance) == leaf_count
    else:
        with pytest.raises(inference_packet.InferencePacketError) as excinfo:
            _checkpoint(tmp_path, topology)
        _assert_refusal(excinfo, inference_packet.PacketRefusal.PROVENANCE_LEAF_BUDGET_EXCEEDED)


@pytest.mark.parametrize(
    "chain",
    [
        SimpleNamespace(chain_id="c", seeds=SimpleNamespace(chain_seed=400), interval=None, status=None),
        {"chain_id": "c"},
        _ChainLookalike("c", SimpleNamespace(chain_seed=400), None, None),
        _ForgedIndependentChain("c", _seeds(0), inference_packet.SamplingInterval(1, 1, 1), None),
    ],
    ids=["simplenamespace", "dict", "dataclass_lookalike", "subclass"],
)
def test_packet_refuses_a_chain_that_is_not_an_independent_chain_record(
    tmp_path: Path, chain: object
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint), checkpoint, (chain,)
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PACKET_CHAIN_NOT_DECLARED_TYPE)


@pytest.mark.parametrize(
    ("component", "bad_value"),
    [
        ("seeds", SimpleNamespace(chain_seed=400)),
        ("interval", SimpleNamespace()),
        ("status", SimpleNamespace()),
        ("seeds", _ForgedSeedProvenance(100, 200, 300, 400)),
        ("interval", _ForgedSamplingInterval(1, 1, 1)),
        (
            "status",
            _ForgedChainStatus(
                inference_packet.ChainState.IDLE,
                _CREATED,
                _CREATED,
            ),
        ),
    ],
    ids=[
        "seeds",
        "interval",
        "status",
        "seeds_subclass",
        "interval_subclass",
        "status_subclass",
    ],
)
def test_independent_chain_refuses_a_component_that_is_not_its_declared_provenance_type(
    component: str, bad_value: object,
) -> None:
    values = {
        "chain_id": "chain-0",
        "seeds": _seeds(0),
        "interval": inference_packet.SamplingInterval(1, 1, 1),
        "status": _chain(0).status,
    }
    values[component] = bad_value
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.IndependentChain(**values)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHAIN_COMPONENT_NOT_DECLARED_TYPE)


@pytest.mark.parametrize("field", ["checkpoint", "distributed_checkpoint"])
def test_packet_requires_declared_checkpoint_and_distributed_checkpoint_types(
    tmp_path: Path, field: str
) -> None:
    checkpoint = _checkpoint(tmp_path)
    fake_checkpoint = SimpleNamespace(
        checkpoint_path=checkpoint.checkpoint_path,
        source_content_hash=_HASH,
        topology_provenance={"unfrozen": object()},
    )
    fake_distributed = SimpleNamespace(world_size=1, rank_artifacts={})
    kwargs = {
        "checkpoint": checkpoint,
        "independent_sampler_inputs": {"walkers": 4_096},
        "chains": (_chain(0),),
        "distributed_checkpoint": None,
    }
    if field == "checkpoint":
        kwargs["checkpoint"] = fake_checkpoint
        expected = inference_packet.PacketRefusal.CHECKPOINT_NOT_A_DECLARED_TYPE
    else:
        kwargs["distributed_checkpoint"] = fake_distributed
        expected = inference_packet.PacketRefusal.DDP_NOT_A_DECLARED_TYPE
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.InferencePacket(**kwargs)
    _assert_refusal(excinfo, expected)


@pytest.mark.parametrize("value", ["1", 0, -1, True], ids=["non_int", "zero", "negative", "bool"])
def test_seed_provenance_requires_positive_integers(value: object) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainSeedProvenance(value, 2, 3, 4)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SEED_NOT_POSITIVE_INT)


@pytest.mark.parametrize(
    "values",
    [
        (100, 200, 300, 100),
        (100, 200, 100, 400),
        (100, 100, 300, 400),
    ],
    ids=["chain_eq_training", "chain_eq_inference", "training_eq_calibration"],
)
def test_seed_provenance_requires_disjoint_phase_and_chain_seeds(values: tuple[int, ...]) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainSeedProvenance(*values)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SEEDS_NOT_DISJOINT)


@pytest.mark.parametrize(
    "field,value,refusal",
    [
        ("burn_in_proposals", -1, inference_packet.PacketRefusal.INTERVAL_BURN_IN_NEGATIVE),
        ("proposals_between_draws", 0, inference_packet.PacketRefusal.INTERVAL_SPACING_NOT_POSITIVE),
        ("retained_draws", 0, inference_packet.PacketRefusal.INTERVAL_DRAWS_NOT_POSITIVE),
    ],
    ids=["burn_in_negative", "spacing_zero", "draws_zero"],
)
def test_sampling_interval_requires_its_declared_bounds(
    field: str, value: int, refusal: object
) -> None:
    values = {"burn_in_proposals": 1, "proposals_between_draws": 1, "retained_draws": 1}
    values[field] = value
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.SamplingInterval(**values)
    _assert_refusal(excinfo, refusal)


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {
            "started_at": _CREATED + timedelta(seconds=1),
            "finished_at": _CREATED + timedelta(seconds=2),
            "last_activity_at": _CREATED + timedelta(seconds=2),
        },
        {
            "started_at": _CREATED,
            "last_activity_at": _CREATED + timedelta(seconds=1),
            "terminal_reason": "not allowed",
        },
    ],
    ids=["no_start", "with_finish", "with_reason"],
)
def test_running_status_requires_only_a_start_record(fields: dict[str, object]) -> None:
    values = {
        "state": inference_packet.ChainState.RUNNING,
        "created_at": _CREATED,
        "last_activity_at": _CREATED,
        **fields,
    }
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(**values)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_RUNNING_NOT_ONLY_A_START)


def test_running_status_is_constructible_and_records_only_a_start() -> None:
    status = _chain(0, inference_packet.ChainState.RUNNING).status
    assert status.started_at is not None
    assert status.finished_at is None
    assert status.terminal_reason is None


@pytest.mark.parametrize(
    "field",
    ["started_at", "finished_at", "terminal_reason"],
    ids=["started_at", "finished_at", "terminal_reason"],
)
def test_idle_status_cannot_claim_start_or_termination(field: str) -> None:
    values = {"state": inference_packet.ChainState.IDLE, "created_at": _CREATED, "last_activity_at": _CREATED}
    values[field] = _CREATED if field != "terminal_reason" else "not idle"
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(**values)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_IDLE_CLAIMS_START_OR_TERMINATION)


def test_completed_status_cannot_carry_a_failure_reason() -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.COMPLETED,
            created_at=_CREATED,
            started_at=_CREATED,
            finished_at=_CREATED + timedelta(seconds=1),
            last_activity_at=_CREATED + timedelta(seconds=1),
            terminal_reason="failure",
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_COMPLETED_CARRIES_REASON)


def test_chain_activity_cannot_precede_creation() -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.IDLE,
            created_at=_CREATED,
            last_activity_at=_CREATED - timedelta(seconds=1),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_ACTIVITY_PRECEDES_CREATION)


@pytest.mark.parametrize("missing", ["no_start", "no_finish"], ids=["no_start", "no_finish"])
def test_terminal_status_requires_start_and_finish_records(missing: str) -> None:
    values = {
        "state": inference_packet.ChainState.ERROR,
        "created_at": _CREATED,
        "started_at": _CREATED,
        "finished_at": _CREATED + timedelta(seconds=1),
        "last_activity_at": _CREATED + timedelta(seconds=1),
        "terminal_reason": "failure",
    }
    values["started_at" if missing == "no_start" else "finished_at"] = None
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(**values)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_TERMINAL_MISSING_RECORDS)


def test_terminal_finish_cannot_precede_start() -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.ERROR,
            created_at=_CREATED,
            started_at=_CREATED + timedelta(seconds=2),
            finished_at=_CREATED + timedelta(seconds=1),
            last_activity_at=_CREATED + timedelta(seconds=2),
            terminal_reason="failure",
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_FINISH_PRECEDES_START)


def test_finish_notification_requires_a_terminal_status() -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.IDLE,
            created_at=_CREATED,
            last_activity_at=_CREATED,
            finish_notification_at=_CREATED,
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_NOTIFICATION_WITHOUT_TERMINATION)


def test_finish_notification_cannot_precede_termination() -> None:
    finished = _CREATED + timedelta(seconds=2)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.ERROR,
            created_at=_CREATED,
            started_at=_CREATED,
            finished_at=finished,
            last_activity_at=finished,
            terminal_reason="failure",
            finish_notification_at=finished - timedelta(seconds=1),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_NOTIFICATION_PRECEDES_TERMINATION)


@pytest.mark.parametrize("state", ["idle", "running", "paused", "completed"], ids=["idle", "running", "paused", "completed"])
def test_a_bare_string_is_not_accepted_as_a_chain_state(state: str) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state=state,
            created_at=_CREATED,
            started_at=_CREATED,
            finished_at=_CREATED + timedelta(seconds=1),
            last_activity_at=_CREATED + timedelta(seconds=1),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_UNKNOWN_STATE)


def test_a_bare_completed_state_with_reason_is_refused() -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state="completed",
            created_at=_CREATED,
            started_at=_CREATED,
            finished_at=_CREATED + timedelta(seconds=1),
            last_activity_at=_CREATED + timedelta(seconds=1),
            terminal_reason="failure",
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_UNKNOWN_STATE)
    with pytest.raises(inference_packet.InferencePacketError) as enum_excinfo:
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.COMPLETED,
            created_at=_CREATED,
            started_at=_CREATED,
            finished_at=_CREATED + timedelta(seconds=1),
            last_activity_at=_CREATED + timedelta(seconds=1),
            terminal_reason="failure",
        )
    _assert_refusal(enum_excinfo, inference_packet.PacketRefusal.STATUS_COMPLETED_CARRIES_REASON)


def test_every_chain_state_member_has_a_declared_lifecycle_rule() -> None:
    for state in inference_packet.ChainState:
        assert state in {
            inference_packet.ChainState.IDLE,
            inference_packet.ChainState.RUNNING,
            *inference_packet._TERMINAL_STATES,
        }


def test_source_packet_checkpoint_path_is_binding(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    source = _source(checkpoint)
    source.checkpoint_path = checkpoint.checkpoint_path.parent / "other"
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(source, checkpoint, (_chain(0),))
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SOURCE_CHECKPOINT_MISMATCH)


def test_source_packet_content_hash_is_binding(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    source = _source(checkpoint)
    source.source_content_hash = "b" * 64
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(source, checkpoint, (_chain(0),))
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SOURCE_HASH_MISMATCH)


@pytest.mark.parametrize(
    "source",
    [
        SimpleNamespace(source_content_hash=_HASH, independent_sampler_inputs={}),
        SimpleNamespace(checkpoint_path=Path("/x"), independent_sampler_inputs={}),
        SimpleNamespace(checkpoint_path=Path("/x"), source_content_hash=_HASH),
        object(),
    ],
    ids=["missing_checkpoint_path", "missing_hash", "missing_inputs", "plain_object"],
)
def test_a_non_packet_source_object_is_refused(tmp_path: Path, source: object) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(source, checkpoint, (_chain(0),))
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SOURCE_PACKET_NOT_A_PACKET)


@pytest.mark.parametrize(
    "parent,checkpoint_path",
    [
        (Path("relative-cell"), Path("/absolute/checkpoints/update")),
        (Path("/absolute/cell"), Path("relative-checkpoints/update")),
    ],
    ids=["parent", "checkpoint"],
)
def test_checkpoint_and_parent_paths_must_be_absolute(parent: Path, checkpoint_path: Path) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.CheckpointReference(parent, checkpoint_path, _HASH, {})
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHECKPOINT_PATH_NOT_ABSOLUTE)


@pytest.mark.parametrize(
    "value",
    ["short", "b" * 65, "A" * 64, "g" * 64],
    ids=["short", "long", "uppercase", "nonhex"],
)
def test_source_content_hash_must_be_lowercase_sha256(tmp_path: Path, value: str) -> None:
    parent = (tmp_path / "O1" / _HASH).resolve()
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.CheckpointReference(
            parent, parent / "checkpoints" / "update", value, {}
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHECKPOINT_HASH_NOT_SHA256)


@pytest.mark.parametrize(
    "topology",
    [[], {"nested": {1: "bad"}}],
    ids=["not_a_mapping", "non_str_key"],
)
def test_topology_provenance_must_be_a_string_keyed_mapping(
    tmp_path: Path, topology: object
) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _checkpoint(tmp_path, topology)
    expected = (
        inference_packet.PacketRefusal.PROVENANCE_NOT_A_MAPPING
        if not isinstance(topology, Mapping)
        else inference_packet.PacketRefusal.PROVENANCE_KEY_NOT_A_STRING
    )
    _assert_refusal(excinfo, expected)


def test_nested_sequences_inside_topology_provenance_are_frozen_to_tuples(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, {"ranks": [0, 1]})
    assert checkpoint.topology_provenance["ranks"] == (0, 1)
    assert type(checkpoint.topology_provenance["ranks"]) is tuple
    with pytest.raises(TypeError):
        checkpoint.topology_provenance["ranks"][0] = 2  # type: ignore[index]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
def test_non_finite_provenance_values_are_refused(tmp_path: Path, value: float) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _checkpoint(tmp_path, {"x": value})
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PROVENANCE_VALUE_NOT_FINITE)


@pytest.mark.parametrize("value", [0, -1, "1", True], ids=["zero", "negative", "non_int", "bool"])
def test_distributed_world_size_must_be_a_positive_int(value: object) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.DistributedCheckpointProvenance(value, {})
    _assert_refusal(excinfo, inference_packet.PacketRefusal.DDP_WORLD_SIZE_NOT_POSITIVE)


@pytest.mark.parametrize("value", [[], None, 1], ids=["list", "none", "int"])
def test_distributed_rank_artifacts_must_be_a_mapping(value: object) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.DistributedCheckpointProvenance(1, value)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.DDP_RANK_ARTIFACTS_NOT_A_MAPPING)


@pytest.mark.parametrize(
    "artifacts",
    [{0: Path("rank-0")}, {0: Path("rank-0"), 1: Path("rank-1"), 2: Path("rank-2")}],
    ids=["partial", "extra_rank"],
)
def test_distributed_rank_membership_must_be_exact(artifacts: dict[int, Path]) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.DistributedCheckpointProvenance(2, artifacts)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.DDP_RANK_MEMBERSHIP_INCOMPLETE)


def test_distributed_rank_artifacts_must_be_unambiguous() -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.DistributedCheckpointProvenance(
            2, {0: Path("same"), 1: Path("same")}
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.DDP_RANK_ARTIFACTS_AMBIGUOUS)


def test_distributed_artifact_must_remain_under_the_bound_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    distributed = inference_packet.DistributedCheckpointProvenance(
        world_size=1, rank_artifacts={0: (tmp_path / "foreign-rank-0").resolve()}
    )
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint), checkpoint, (_chain(0),), distributed_checkpoint=distributed
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.DDP_ARTIFACT_OUTSIDE_CHECKPOINT)


def test_type_shaped_guards_are_reachable_because_annotations_are_not_runtime_enforced() -> None:
    """A TypeCheckError here means the six type-shaped guards became vacuous."""

    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainSeedProvenance("1", 2, 3, 4)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SEED_NOT_POSITIVE_INT)


@pytest.mark.parametrize(
    "sampler_inputs",
    [
        {"walkers": 4_096, "burn_in_sweeps": 100},
        {"sampler": {"walkers": 4_096}},
        {"walkers": 4_096},
    ],
    ids=["walkers_and_burn_in", "sampler_walkers", "walkers_only"],
)
def test_the_real_producer_declaration_shapes_are_accepted(
    tmp_path: Path, sampler_inputs: dict[str, object]
) -> None:
    l2_packet = _real_l2_packet(tmp_path, sampler_inputs)
    checkpoint = _checkpoint_for(l2_packet)
    packet = inference_packet.launch_inference_packet(l2_packet, checkpoint, (_chain(0),))
    assert packet.independent_sampler_inputs == sampler_inputs


def test_the_real_producer_fixture_is_the_l2_test_modules_own() -> None:
    assert l2_tests._packet_source_cells.__module__ == "he_importance_test_stage_coordinate"
    assert stage_coordinate.materialize_job_packets is getattr(
        l2_tests.stage_coordinate, "materialize_job_packets"
    )
