"""Contract tests for expensive independent-sampler inference packets."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple
from types import MappingProxyType, SimpleNamespace
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


l2_tests = _reuse_or_load(
    "he_importance_test_stage_coordinate",
    Path(__file__).with_name("test_stage_coordinate.py"),
)
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


class _AlwaysEqualHash:
    """Non-string source value that lies to equality-based comparisons."""

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


class _LyingHash(str):
    """String subclass whose comparison claims to match every checkpoint."""

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


class _SameContentDistinctIdentity(str):
    """Equal-looking content with deliberately distinct equality/hash behavior."""

    def __new__(cls, value: str, identity_hash: int) -> "_SameContentDistinctIdentity":
        result = str.__new__(cls, value)
        result.identity_hash = identity_hash
        return result

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return self.identity_hash


class _WhitespaceLooksValid(str):
    def strip(self, chars: str | None = None) -> str:
        return "looks-valid"


@dataclass
class _RawMarker:
    value: str


class _MutableProvenanceKey(str):
    def __new__(cls, value: str, marker: _RawMarker) -> "_MutableProvenanceKey":
        result = str.__new__(cls, value)
        result.marker = marker
        return result


class _CountingOffset(tzinfo):
    def __init__(self, offset: object) -> None:
        self.offset = offset
        self.calls = 0

    def utcoffset(self, value: datetime | None) -> object:
        self.calls += 1
        return self.offset if self.calls == 1 else None


class _RaisingOffset(tzinfo):
    def utcoffset(self, value: datetime | None) -> timedelta:
        raise RuntimeError("offset probe failed")


class _ItemsMapping(Mapping):
    def __init__(self, entries: list[object]) -> None:
        self.entries = entries

    def __iter__(self):
        return iter(())

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, key: object) -> object:
        raise KeyError(key)

    def items(self):
        return iter(self.entries)


class _PathLike:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __fspath__(self) -> str:
        return str(self.path)


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


def test_sampler_mapping_items_must_be_pairs(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.InferencePacket(
            checkpoint,
            _ItemsMapping([("walkers", 4_096, "extra")]),
            (_chain(0),),
            None,
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_ITEMS_NOT_PAIRS)



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


def test_source_packet_checkpoint_path_accepts_a_symlinked_spelling(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(checkpoint.checkpoint_path.parent.parent, target_is_directory=True)
    source = _source(checkpoint)
    source.checkpoint_path = alias / "checkpoints" / checkpoint.checkpoint_path.name

    packet = inference_packet.launch_inference_packet(
        source, checkpoint, (_chain(0),)
    )
    assert packet.checkpoint is checkpoint


def test_source_checkpoint_path_wrong_type_has_distinct_refusal(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    source = _source(checkpoint)
    source.checkpoint_path = 5
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(source, checkpoint, (_chain(0),))
    _assert_refusal(
        excinfo, inference_packet.PacketRefusal.SOURCE_CHECKPOINT_PATH_NOT_PATHLIKE
    )


def test_source_checkpoint_path_accepts_an_os_pathlike_value(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    source = _source(checkpoint)
    source.checkpoint_path = _PathLike(checkpoint.checkpoint_path)
    packet = inference_packet.launch_inference_packet(
        source, checkpoint, (_chain(0),)
    )
    assert packet.checkpoint is checkpoint


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
    assert (
        Path(l2_tests._packet_source_cells.__globals__["__file__"]).resolve()
        == Path(__file__).with_name("test_stage_coordinate.py").resolve()
    )
    assert (
        Path(stage_coordinate.materialize_job_packets.__globals__["__file__"]).resolve()
        == Path(__file__).with_name("stage_coordinate.py").resolve()
    )


@pytest.mark.parametrize("source_hash", [_AlwaysEqualHash(), _LyingHash("b" * 64)])
def test_source_hash_comparison_cannot_admit_a_nonmatching_caller_value(
    tmp_path: Path, source_hash: object
) -> None:
    checkpoint = _checkpoint(tmp_path)
    source = _source(checkpoint)
    source.source_content_hash = source_hash
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(source, checkpoint, (_chain(0),))
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SOURCE_HASH_MISMATCH)


def test_ordinary_matching_source_hash_remains_a_positive_control(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    packet = inference_packet.launch_inference_packet(
        _source(checkpoint), checkpoint, (_chain(0),)
    )
    assert packet.checkpoint is checkpoint


def test_chain_ids_are_unique_by_underlying_plain_string_content(
    tmp_path: Path,
) -> None:
    del tmp_path
    for index in (0, 1):
        valid = _chain(index)
        with pytest.raises(inference_packet.InferencePacketError) as excinfo:
            inference_packet.IndependentChain(
                _SameContentDistinctIdentity("same", 101 + index),
                valid.seeds,
                valid.interval,
                valid.status,
            )
        _assert_refusal(excinfo, inference_packet.PacketRefusal.CHAIN_ID_NOT_A_STRING)


def test_chain_id_empty_check_cannot_be_redirected_by_str_subclass_strip() -> None:
    valid = _chain(0)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.IndependentChain(
            _WhitespaceLooksValid("   "), valid.seeds, valid.interval, valid.status
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHAIN_ID_NOT_A_STRING)


def test_chain_id_type_violation_has_distinct_refusal() -> None:
    valid = _chain(0)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.IndependentChain(
            5, valid.seeds, valid.interval, valid.status
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHAIN_ID_NOT_A_STRING)


def test_disposed_topology_key_subclass_is_retained_as_residual_measurement(
    tmp_path: Path,
) -> None:
    """Disposed residual only; it is not adoptable closure evidence."""

    marker = _RawMarker("training-walker")
    key = _MutableProvenanceKey("topology", marker)
    try:
        checkpoint = _checkpoint(tmp_path)
        checkpoint = inference_packet.CheckpointReference(
            checkpoint.parent_cell_path,
            checkpoint.checkpoint_path,
            checkpoint.source_content_hash,
            {key: "preserved-content"},
        )
    except inference_packet.InferencePacketError as exc:
        assert exc.refusal is inference_packet.PacketRefusal.PROVENANCE_KEY_NOT_A_STRING
        return

    stored_key = next(iter(checkpoint.topology_provenance))
    assert type(stored_key) is not str
    assert stored_key == "topology"
    marker.value = "mutated-training-walker"
    assert stored_key.marker.value == "mutated-training-walker"


def test_the_same_mutable_key_fixture_is_refused_in_sampler_inputs(
    tmp_path: Path,
) -> None:
    marker = _RawMarker("training-walker")
    key = _MutableProvenanceKey("walkers", marker)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.InferencePacket(
            _checkpoint(tmp_path), {key: 4_096}, (_chain(0),), None
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_KEY_NOT_A_STRING)


def _completed_status(started_at: datetime, finished_at: datetime) -> object:
    return inference_packet.ChainStatus(
        state=inference_packet.ChainState.COMPLETED,
        created_at=datetime(2026, 11, 1, 0, 0, tzinfo=ZoneInfo("America/New_York")),
        last_activity_at=finished_at,
        started_at=started_at,
        finished_at=finished_at,
    )


def test_chain_status_stores_all_timestamps_as_canonical_utc_instants() -> None:
    zone = ZoneInfo("America/New_York")
    created = datetime(2026, 11, 1, 0, 0, tzinfo=zone)
    started = datetime(2026, 11, 1, 1, 45, fold=0, tzinfo=zone)
    finished = datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=zone)
    notification = datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=zone)
    assert created.astimezone(UTC) == datetime(2026, 11, 1, 4, 0, tzinfo=UTC)
    assert started.astimezone(UTC) == datetime(2026, 11, 1, 5, 45, tzinfo=UTC)
    assert finished.astimezone(UTC) == datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
    status = inference_packet.ChainStatus(
        state=inference_packet.ChainState.ERROR,
        created_at=created,
        last_activity_at=finished,
        started_at=started,
        finished_at=finished,
        terminal_reason="fold test",
        finish_notification_at=notification,
    )
    expected = (
        datetime(2026, 11, 1, 4, 0, tzinfo=UTC),
        datetime(2026, 11, 1, 6, 30, tzinfo=UTC),
        datetime(2026, 11, 1, 5, 45, tzinfo=UTC),
        datetime(2026, 11, 1, 6, 30, tzinfo=UTC),
        datetime(2026, 11, 1, 6, 30, tzinfo=UTC),
    )
    for stored, literal in zip(
        (
            status.created_at,
            status.last_activity_at,
            status.started_at,
            status.finished_at,
            status.finish_notification_at,
        ),
        expected,
    ):
        assert type(stored) is datetime
        assert stored.tzinfo is UTC
        assert stored.fold == 0
        assert stored == literal
    assert status.created_at.tzinfo is not zone
    assert status.started_at.tzinfo is not zone
    assert status.finished_at.tzinfo is not zone


def test_chain_status_reads_each_timestamp_tzinfo_once_and_detaches_it() -> None:
    offsets = [_CountingOffset(timedelta(hours=-5)) for _ in range(5)]
    values = (
        datetime(2026, 1, 1, 0, 0, tzinfo=offsets[0]),
        datetime(2026, 1, 1, 0, 4, tzinfo=offsets[1]),
        datetime(2026, 1, 1, 0, 1, tzinfo=offsets[2]),
        datetime(2026, 1, 1, 0, 2, tzinfo=offsets[3]),
        datetime(2026, 1, 1, 0, 3, tzinfo=offsets[4]),
    )
    status = inference_packet.ChainStatus(
        inference_packet.ChainState.ERROR,
        values[0],
        values[1],
        values[2],
        values[3],
        "counted offset",
        values[4],
    )
    assert [counter.calls for counter in offsets] == [1, 1, 1, 1, 1]
    for stored in (
        status.created_at,
        status.last_activity_at,
        status.started_at,
        status.finished_at,
        status.finish_notification_at,
    ):
        assert stored.tzinfo is UTC
        stored.astimezone(UTC)
    assert [counter.calls for counter in offsets] == [1, 1, 1, 1, 1]


@pytest.mark.parametrize("field", ["created_at", "last_activity_at"])
def test_required_missing_status_timestamp_has_declared_refusal(field: str) -> None:
    values: dict[str, object] = {
        "state": inference_packet.ChainState.IDLE,
        "created_at": _CREATED,
        "last_activity_at": _CREATED,
    }
    values[field] = None
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(**values)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_TIMESTAMP_NOT_AWARE_DATETIME)


@pytest.mark.parametrize(
    "tz_value", [None, "raises"], ids=["none-offset", "ordinary-exception"]
)
def test_unavailable_status_timestamp_offset_has_declared_refusal(tz_value: object) -> None:
    offset: tzinfo = _CountingOffset(None) if tz_value is None else _RaisingOffset()
    value = datetime(2026, 1, 1, tzinfo=offset)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.IDLE, value, value
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_TIMESTAMP_NOT_AWARE_DATETIME)


@pytest.mark.parametrize(
    "offset", [0, timedelta(hours=24)], ids=["non-timedelta", "out-of-range"]
)
def test_datetime_method_rejects_invalid_timestamp_offsets(offset: object) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.IDLE,
            *[datetime(2026, 1, 1, tzinfo=_CountingOffset(offset)) for _ in range(2)],
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_TIMESTAMP_NOT_AWARE_DATETIME)


@pytest.mark.parametrize(
    "value",
    [
        datetime.min.replace(tzinfo=timezone(timedelta(hours=1))),
        datetime.max.replace(tzinfo=timezone(-timedelta(hours=1))),
    ],
    ids=["minimum-plus-one-hour", "maximum-minus-one-hour"],
)
def test_status_timestamp_utc_range_overflow_has_distinct_refusal(value: datetime) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.IDLE, value, value
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_TIMESTAMP_OUT_OF_UTC_RANGE)


@pytest.mark.parametrize("value", [datetime.min, datetime.max], ids=["min-utc", "max-utc"])
def test_utc_boundary_timestamps_remain_accepted(value: datetime) -> None:
    value = value.replace(tzinfo=UTC)
    status = inference_packet.ChainStatus(
        inference_packet.ChainState.IDLE, value, value
    )
    assert status.created_at == value
    assert status.last_activity_at == value
    assert status.created_at.tzinfo is UTC


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("idle-creation", inference_packet.PacketRefusal.STATUS_ACTIVITY_PRECEDES_CREATION),
        ("running-creation", inference_packet.PacketRefusal.STATUS_NON_ACTIVITY_PRECEDES_CREATION),
        ("terminal-creation", inference_packet.PacketRefusal.STATUS_NON_ACTIVITY_PRECEDES_CREATION),
        ("error-notification-creation", inference_packet.PacketRefusal.STATUS_NON_ACTIVITY_PRECEDES_CREATION),
        ("finish-start", inference_packet.PacketRefusal.STATUS_FINISH_PRECEDES_START),
        ("activity-finish", inference_packet.PacketRefusal.STATUS_ACTIVITY_PRECEDES_FINISH),
        ("notification-finish", inference_packet.PacketRefusal.STATUS_NOTIFICATION_PRECEDES_TERMINATION),
        ("running-activity-start", inference_packet.PacketRefusal.STATUS_ACTIVITY_PRECEDES_START),
    ],
    ids=[
        "idle-creation",
        "running-creation",
        "terminal-creation",
        "error-notification-creation",
        "finish-start",
        "activity-finish",
        "notification-finish",
        "running-activity-start",
    ],
)
def test_dst_fold_pins_each_declared_chronology_refusal(
    case: str, expected: object
) -> None:
    zone = ZoneInfo("America/New_York")
    boundary = datetime(2026, 11, 1, 0, 0, tzinfo=zone)
    early = datetime(2026, 11, 1, 1, 45, fold=0, tzinfo=zone)
    late = datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=zone)
    assert boundary.astimezone(UTC) == datetime(2026, 11, 1, 4, tzinfo=UTC)
    assert early.astimezone(UTC) == datetime(2026, 11, 1, 5, 45, tzinfo=UTC)
    assert late.astimezone(UTC) == datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
    values: dict[str, object] = {
        "state": inference_packet.ChainState.IDLE,
        "created_at": late,
        "last_activity_at": early,
        "started_at": None,
        "finished_at": None,
        "terminal_reason": None,
        "finish_notification_at": None,
    }
    if case == "running-creation":
        values.update(state=inference_packet.ChainState.RUNNING, started_at=early, last_activity_at=late)
    elif case == "terminal-creation":
        values.update(state=inference_packet.ChainState.COMPLETED, started_at=late, finished_at=early, last_activity_at=late)
    elif case == "error-notification-creation":
        values.update(state=inference_packet.ChainState.ERROR, started_at=late, finished_at=late, last_activity_at=late, terminal_reason="failure", finish_notification_at=early)
    elif case == "finish-start":
        values.update(state=inference_packet.ChainState.COMPLETED, created_at=boundary, started_at=late, finished_at=early, last_activity_at=late)
    elif case == "activity-finish":
        values.update(state=inference_packet.ChainState.COMPLETED, created_at=boundary, started_at=early, finished_at=late, last_activity_at=early)
    elif case == "notification-finish":
        values.update(state=inference_packet.ChainState.ERROR, created_at=boundary, started_at=early, finished_at=late, last_activity_at=late, terminal_reason="failure", finish_notification_at=early)
    elif case == "running-activity-start":
        values.update(state=inference_packet.ChainState.RUNNING, created_at=boundary, started_at=late, last_activity_at=early)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(**values)
    _assert_refusal(excinfo, expected)


def test_equal_finish_and_start_are_accepted() -> None:
    instant = _CREATED + timedelta(seconds=1)
    status = inference_packet.ChainStatus(
        inference_packet.ChainState.COMPLETED,
        _CREATED,
        instant,
        instant,
        instant,
    )
    assert status.started_at == status.finished_at


def test_equal_notification_and_finish_are_accepted() -> None:
    instant = _CREATED + timedelta(seconds=2)
    status = inference_packet.ChainStatus(
        inference_packet.ChainState.ERROR,
        _CREATED,
        instant,
        _CREATED + timedelta(seconds=1),
        instant,
        "failure",
        instant,
    )
    assert status.finish_notification_at == status.finished_at


@pytest.mark.parametrize(
    ("started_at", "finished_at", "should_refuse"),
    [
        (
            datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 11, 1, 1, 45, fold=0, tzinfo=ZoneInfo("America/New_York")),
            True,
        ),
        (
            datetime(2026, 11, 1, 1, 45, fold=0, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=ZoneInfo("America/New_York")),
            False,
        ),
    ],
    ids=["finish-actually-before-start", "finish-actually-after-start"],
)
def test_dst_fold_status_order_is_chronological(
    started_at: datetime, finished_at: datetime, should_refuse: bool
) -> None:
    if should_refuse:
        assert started_at.astimezone(UTC) > finished_at.astimezone(UTC)
        with pytest.raises(inference_packet.InferencePacketError) as excinfo:
            _completed_status(started_at, finished_at)
        _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_FINISH_PRECEDES_START)
    else:
        assert started_at.astimezone(UTC) < finished_at.astimezone(UTC)
        status = _completed_status(started_at, finished_at)
        assert status.finished_at == finished_at.astimezone(UTC)
        assert status.finished_at.tzinfo is UTC
        assert status.finished_at.fold == 0


def test_plain_aware_chronology_controls_remain_directional() -> None:
    earlier = datetime(2026, 11, 1, 1, 30, tzinfo=UTC)
    later = datetime(2026, 11, 1, 1, 45, tzinfo=UTC)
    status = inference_packet.ChainStatus(
        state=inference_packet.ChainState.COMPLETED,
        created_at=datetime(2026, 11, 1, 0, 0, tzinfo=UTC),
        last_activity_at=later,
        started_at=earlier,
        finished_at=later,
    )
    assert status.finished_at == later
    assert status.finished_at.tzinfo is UTC
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.COMPLETED,
            created_at=datetime(2026, 11, 1, 0, 0, tzinfo=UTC),
            last_activity_at=earlier,
            started_at=later,
            finished_at=earlier,
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_FINISH_PRECEDES_START)


def test_only_finish_before_creation_reports_creation_precedence() -> None:
    created = datetime(2026, 9, 12, 10, 0, tzinfo=UTC)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.COMPLETED,
            created_at=created,
            started_at=created,
            finished_at=datetime(2026, 9, 12, 9, 0, tzinfo=UTC),
            last_activity_at=datetime(2026, 9, 12, 11, 0, tzinfo=UTC),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_NON_ACTIVITY_PRECEDES_CREATION)


def test_only_notification_before_creation_reports_creation_precedence() -> None:
    created = datetime(2026, 9, 12, 10, 0, tzinfo=UTC)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state=inference_packet.ChainState.ERROR,
            created_at=created,
            started_at=created,
            finished_at=datetime(2026, 9, 12, 11, 0, tzinfo=UTC),
            last_activity_at=datetime(2026, 9, 12, 11, 0, tzinfo=UTC),
            terminal_reason="reviewed error",
            finish_notification_at=datetime(2026, 9, 12, 9, 0, tzinfo=UTC),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_NON_ACTIVITY_PRECEDES_CREATION)


@pytest.mark.parametrize(
    "state",
    [
        inference_packet.ChainState.COMPLETED,
        inference_packet.ChainState.ERROR,
        inference_packet.ChainState.CANCELLED,
        inference_packet.ChainState.DEAD,
    ],
    ids=["COMPLETED", "ERROR", "CANCELLED", "DEAD"],
)
@pytest.mark.parametrize("notified", [False, True], ids=["without-notification", "with-notification"])
def test_each_valid_terminal_status_is_accepted(state: object, notified: bool) -> None:
    created = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
    started = datetime(2026, 9, 12, 12, 1, tzinfo=UTC)
    finished = datetime(2026, 9, 12, 12, 2, tzinfo=UTC)
    notification = datetime(2026, 9, 12, 12, 3, tzinfo=UTC) if notified else None
    reason = None if state is inference_packet.ChainState.COMPLETED else "reviewed terminal outcome"
    status = inference_packet.ChainStatus(
        state=state,
        created_at=created,
        last_activity_at=finished,
        started_at=started,
        finished_at=finished,
        terminal_reason=reason,
        finish_notification_at=notification,
    )
    assert status.state is state
    assert status.terminal_reason is reason
    assert status.finish_notification_at == notification


@pytest.mark.parametrize("depth", [6, 7], ids=["depth-6", "depth-7"])
def test_exact_provenance_depth_boundary(depth: int, tmp_path: Path) -> None:
    topology: object = 1
    for index in reversed(range(depth)):
        topology = {f"level_{index}": topology}
    if depth == 6:
        assert _checkpoint(tmp_path, topology=topology).topology_provenance
    else:
        with pytest.raises(inference_packet.InferencePacketError) as excinfo:
            _checkpoint(tmp_path, topology=topology)
        _assert_refusal(excinfo, inference_packet.PacketRefusal.PROVENANCE_DEPTH_BUDGET_EXCEEDED)


@pytest.mark.parametrize(
    "position", ["training_seed", "calibration_seed", "inference_seed", "chain_seed"]
)
def test_each_seed_position_requires_positive_exact_int(position: str) -> None:
    values = dict(training_seed=101, calibration_seed=201, inference_seed=301, chain_seed=401)
    values[position] = 0
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainSeedProvenance(**values)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SEED_NOT_POSITIVE_INT)


@pytest.mark.parametrize(
    "position", ["training_seed", "calibration_seed", "inference_seed", "chain_seed"]
)
def test_each_seed_position_rejects_bad_type(position: str) -> None:
    values = dict(training_seed=101, calibration_seed=201, inference_seed=301, chain_seed=401)
    values[position] = "not-an-int"
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainSeedProvenance(**values)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SEED_NOT_POSITIVE_INT)


def test_ddp_rank_artifacts_are_independently_immutable(tmp_path: Path) -> None:
    ranks = {0: tmp_path / "checkpoint" / "rank-0"}
    provenance = inference_packet.DistributedCheckpointProvenance(1, ranks)
    assert type(provenance.rank_artifacts) is MappingProxyType
    with pytest.raises(TypeError):
        provenance.rank_artifacts[0] = tmp_path / "checkpoint" / "other"  # type: ignore[index]


def test_ddp_rank_artifact_value_wrong_type_has_distinct_refusal() -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.DistributedCheckpointProvenance(1, {0: 5})
    _assert_refusal(
        excinfo, inference_packet.PacketRefusal.DDP_RANK_ARTIFACT_NOT_PATHLIKE
    )


def test_ddp_rank_artifact_accepts_an_os_pathlike_value(tmp_path: Path) -> None:
    path = tmp_path / "rank-0"
    provenance = inference_packet.DistributedCheckpointProvenance(
        1, {0: _PathLike(path)}
    )
    assert provenance.rank_artifacts[0] == path.resolve()


def test_ddp_rank_artifact_mapping_items_must_be_pairs(tmp_path: Path) -> None:
    malformed = _ItemsMapping([(0, tmp_path / "rank-0", "extra")])
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.DistributedCheckpointProvenance(1, malformed)
    _assert_refusal(
        excinfo, inference_packet.PacketRefusal.DDP_RANK_ARTIFACT_ITEMS_NOT_PAIRS
    )


def test_ddp_rank_artifact_pair_shape_is_distinct_from_path_type(tmp_path: Path) -> None:
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.DistributedCheckpointProvenance(
            1, _ItemsMapping([(0, 5)])
        )
    _assert_refusal(
        excinfo, inference_packet.PacketRefusal.DDP_RANK_ARTIFACT_NOT_PATHLIKE
    )


def test_valid_packet_accepts_complete_ddp_artifacts_inside_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    artifacts = {
        rank: checkpoint.checkpoint_path / f"rank-{rank}.safetensors"
        for rank in range(2)
    }
    distributed = inference_packet.DistributedCheckpointProvenance(2, artifacts)
    packet = inference_packet.launch_inference_packet(
        _source(checkpoint), checkpoint, (_chain(0),), distributed_checkpoint=distributed
    )
    assert packet.distributed_checkpoint is distributed
    assert set(packet.distributed_checkpoint.rank_artifacts) == {0, 1}


def test_topology_mapping_is_independently_immutable(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, topology={"launcher": "review"})
    assert type(checkpoint.topology_provenance) is MappingProxyType
    with pytest.raises(TypeError):
        checkpoint.topology_provenance["launcher"] = "mutated"  # type: ignore[index]


@pytest.mark.parametrize("nested", [False, True], ids=["root", "nested"])
def test_topology_provenance_mapping_items_must_be_pairs(
    tmp_path: Path, nested: bool
) -> None:
    malformed = _ItemsMapping([("launcher", "test", "extra")])
    topology = {"outer": malformed} if nested else malformed
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _checkpoint(tmp_path, topology=topology)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PROVENANCE_ITEMS_NOT_PAIRS)


def test_caller_open_sampler_leaf_refuses_actual_mapping_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(
        inference_packet,
        "_SAMPLER_INPUT_SPEC",
        MappingProxyType({"walkers": None}),
    )
    payload = {"marker": 0.1, "positions": [0.2, 0.3]}
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint, inputs={"walkers": payload}),
            checkpoint,
            (_chain(0),),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_SPEC_NOT_CLOSED)


def test_materialize_job_packets_is_the_actual_stage_coordinate_producer() -> None:
    path = Path(__file__).resolve()
    spec = importlib.util.spec_from_file_location("review_r2_contract_suite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    producer = module.stage_coordinate.materialize_job_packets
    defining_file = Path(producer.__globals__["__file__"]).resolve()
    expected = path.with_name("stage_coordinate.py").resolve()
    assert defining_file == expected


class _ReviewTwoFacedRanks(Mapping):
    """Report declared ranks through iteration and different ranks through items."""

    def __init__(self, stored: dict[int, Path], advertised: dict[int, int]) -> None:
        self._stored = stored
        self._advertised = advertised

    def __iter__(self):
        return iter(self._advertised)

    def __len__(self) -> int:
        return len(self._stored)

    def __getitem__(self, key: int) -> Path:
        return self._stored[key]

    def items(self):
        return self._stored.items()


class _ReviewLyingRankDict(dict[int, Path]):
    """Keep screened ranks in dict storage while items supplies other ranks."""

    def __init__(self, items_ranks: dict[int, Path]) -> None:
        super().__init__({0: Path("screened-rank-0"), 1: Path("screened-rank-1")})
        self._items_ranks = items_ranks

    def items(self):
        return self._items_ranks.items()


def _review_packet(checkpoint: object, chains: object, distributed: object = None) -> object:
    """Construct a direct packet for a review arm."""
    return inference_packet.InferencePacket(
        checkpoint, {"walkers": 4_096}, chains, distributed
    )


class _ReviewForgedSamplerKey(str):
    """Hash and compare as walkers while retaining a different string payload."""

    def __eq__(self, other: object) -> bool:
        return True if other == "walkers" else str.__eq__(self, other)

    def __hash__(self) -> int:
        return hash("walkers")


class _ReviewTwoFacedSamplerMapping(Mapping[str, object]):
    """Return declared walkers on its first items read and a payload thereafter."""

    def __init__(self) -> None:
        self.items_reads = 0

    def __iter__(self):
        return iter(("walkers",))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> object:
        if key != "walkers":
            raise KeyError(key)
        return 4_096

    def items(self):
        self.items_reads += 1
        value: object = 4_096 if self.items_reads == 1 else {"payload": "second read"}
        return {"walkers": value}.items()


class _ReviewAlwaysAfter:
    """Make every less-than temporal check report false."""

    def __lt__(self, other: object) -> bool:
        return False


class _ReviewPropertyAttributeErrorSource:
    """Expose the source interface but raise AttributeError inside one property."""

    def __init__(self, checkpoint: object) -> None:
        self.checkpoint_path = checkpoint.checkpoint_path
        self.source_content_hash = checkpoint.source_content_hash

    @property
    def independent_sampler_inputs(self) -> object:
        return self.inner_attribute_that_does_not_exist


class _ReviewTopologyStringSubclass(str):
    """A distinct string type for the nested provenance-key green hunt."""


def test_distributed_rank_membership_is_screened_on_a_different_read_than_it_stores(
    tmp_path: Path,
) -> None:
    """One rank mapping read must govern both validation and storage."""
    ranks = _ReviewTwoFacedRanks(
        {77: (tmp_path / "a").resolve(), 88: (tmp_path / "b").resolve()},
        {0: 1, 1: 1},
    )
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.DistributedCheckpointProvenance(2, ranks)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.DDP_RANK_MEMBERSHIP_INCOMPLETE)


def test_a_dict_subclass_overriding_items_diverges_the_stored_ranks_from_the_screened_ranks(
    tmp_path: Path,
) -> None:
    """A dict subclass cannot make its storage and items view disagree."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.DistributedCheckpointProvenance(
            2,
            _ReviewLyingRankDict(
                {77: (tmp_path / "a").resolve(), 88: (tmp_path / "b").resolve()}
            ),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.DDP_RANK_MEMBERSHIP_INCOMPLETE)


def test_a_plain_dict_of_ranks_is_screened_and_stored_consistently(tmp_path: Path) -> None:
    """A plain mapping's declared ranks are preserved in its frozen provenance."""
    provenance = inference_packet.DistributedCheckpointProvenance(
        2,
        {0: (tmp_path / "a").resolve(), 1: (tmp_path / "b").resolve()},
    )
    assert set(provenance.rank_artifacts) == {0, 1}


def test_a_distributed_artifact_escapes_the_checkpoint_through_parent_traversal() -> None:
    """Normalized artifacts outside the checkpoint are refused."""
    checkpoint = inference_packet.CheckpointReference(
        Path("/cell"), Path("/cell/checkpoints/x.pt"), _HASH, {"world_size": 2}
    )
    artifact = Path("/cell/checkpoints/x.pt/../../../../etc/passwd")
    distributed = inference_packet.DistributedCheckpointProvenance(
        2,
        {0: artifact, 1: Path("/cell/checkpoints/x.pt/rank-1")},
    )
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _review_packet(checkpoint, (_chain(0),), distributed)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.DDP_ARTIFACT_OUTSIDE_CHECKPOINT)


def test_a_sibling_distributed_artifact_is_still_refused() -> None:
    """A lexically foreign sibling artifact still reaches the refusal."""
    checkpoint = inference_packet.CheckpointReference(
        Path("/cell"), Path("/cell/checkpoints/x.pt"), _HASH, {"world_size": 2}
    )
    distributed = inference_packet.DistributedCheckpointProvenance(
        2,
        {
            0: Path("/cell/checkpoints/y.pt/rank-0"),
            1: Path("/cell/checkpoints/x.pt/rank-1"),
        },
    )
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _review_packet(checkpoint, (_chain(0),), distributed)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.DDP_ARTIFACT_OUTSIDE_CHECKPOINT)


def test_a_checkpoint_path_ending_in_a_parent_traversal_is_bound_to_its_own_cell() -> None:
    """A checkpoint path normalized to its parent cell is refused."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.CheckpointReference(
            Path("/cell"), Path("/cell/checkpoints/.."), _HASH, {"world_size": 1}
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHECKPOINT_NOT_BOUND_TO_PARENT_CELL)


def test_a_more_obvious_checkpoint_escape_is_refused_by_the_spelling_guard() -> None:
    """An obvious checkpoint escape still reaches the binding refusal."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.CheckpointReference(
            Path("/cell"), Path("/cell/checkpoints/../../etc/x"), _HASH, {"world_size": 1}
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHECKPOINT_NOT_BOUND_TO_PARENT_CELL)


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        Path("/cell/blobs/x.pt"),
        Path("/cell/checkpoints/sub/x.pt"),
        Path("/other/checkpoints/x.pt"),
    ],
    ids=["wrong_directory", "one_level_too_deep", "different_cell"],
)
def test_a_checkpoint_outside_its_parent_cell_checkpoints_directory_is_refused(
    checkpoint_path: Path,
) -> None:
    """Ordinary unbound checkpoint paths still reach the binding refusal."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.CheckpointReference(
            Path("/cell"), checkpoint_path, _HASH, {"world_size": 1}
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHECKPOINT_NOT_BOUND_TO_PARENT_CELL)


def test_a_list_of_chains_is_materialized_and_immutable(tmp_path: Path) -> None:
    """Direct construction freezes a list of chains into a tuple."""
    packet = _review_packet(_checkpoint(tmp_path), [_chain(0)])
    assert type(packet.chains) is tuple
    with pytest.raises(AttributeError):
        packet.chains.append("not a chain")


def test_an_iterator_of_chains_is_materialized_and_admitted(
    tmp_path: Path,
) -> None:
    """Direct construction accepts an iterator after eagerly tuple-ing it."""
    packet = _review_packet(_checkpoint(tmp_path), iter([_chain(0)]))
    assert type(packet.chains) is tuple
    assert len(packet.chains) == 1


@pytest.mark.parametrize("route", ["direct", "launch"], ids=["direct", "launch"])
def test_non_iterable_chains_has_declared_refusal(tmp_path: Path, route: str) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        if route == "direct":
            _review_packet(checkpoint, 5)
        else:
            inference_packet.launch_inference_packet(
                _source(checkpoint), checkpoint, 5  # type: ignore[arg-type]
            )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PACKET_CHAINS_NOT_ITERABLE)


class _OneShotChains:
    def __init__(self, chain: object) -> None:
        self.chain = chain
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("chains were iterated more than once")
        return iter((self.chain,))


def test_a_one_shot_chain_iterable_is_consumed_once(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    chains = _OneShotChains(_chain(0))
    packet = _review_packet(checkpoint, chains)
    assert type(packet.chains) is tuple
    assert chains.iterations == 1


def test_an_empty_sequence_of_chains_is_refused_with_its_declared_member(tmp_path: Path) -> None:
    """A true empty sequence still takes the packet's declared empty-chain path."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _review_packet(_checkpoint(tmp_path), ())
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PACKET_HAS_NO_CHAIN)


def test_a_forged_string_key_defeats_the_declared_sampler_key_closure(tmp_path: Path) -> None:
    """Sampler keys require exact strings before equality-based closure."""
    key = _ReviewForgedSamplerKey("COPIED-FROM-TRAINING-RUN-WALKERS")
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.InferencePacket(
            _checkpoint(tmp_path), {key: 4_096}, (_chain(0),), None
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_KEY_NOT_A_STRING)


def test_the_forged_key_survives_a_json_shaped_export(tmp_path: Path) -> None:
    """A forged sampler key cannot reach a JSON-shaped export."""
    key = _ReviewForgedSamplerKey("COPIED-FROM-TRAINING-RUN-WALKERS")
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.InferencePacket(
            _checkpoint(tmp_path), {key: 4_096}, (_chain(0),), None
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_KEY_NOT_A_STRING)


def test_a_plain_undeclared_key_is_still_refused(tmp_path: Path) -> None:
    """A normal undeclared key reaches the declared closure refusal."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.InferencePacket(
            _checkpoint(tmp_path), {"not_declared": 4_096}, (_chain(0),), None
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_KEY_NOT_DECLARED)


def test_a_nested_forged_string_key_defeats_the_declared_sampler_key_closure(
    tmp_path: Path,
) -> None:
    """Nested sampler keys require exact strings too."""
    key = _ReviewForgedSamplerKey("COPIED-FROM-TRAINING-RUN-WALKERS")
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.InferencePacket(
            _checkpoint(tmp_path), {"sampler": {key: 4_096}}, (_chain(0),), None
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_KEY_NOT_A_STRING)


def test_the_single_pass_read_holds_at_the_nested_sampler_position(tmp_path: Path) -> None:
    """The nested sampler mapping is screened and frozen from one items read."""
    inputs = _ReviewTwoFacedSamplerMapping()
    packet = inference_packet.InferencePacket(
        _checkpoint(tmp_path), {"sampler": inputs}, (_chain(0),), None
    )
    assert packet.independent_sampler_inputs["sampler"] == {"walkers": 4_096}
    assert inputs.items_reads == 1


@pytest.mark.parametrize(
    ("interval", "field"),
    [
        ((0.5, 1.5, 2.5), "burn_in_proposals"),
        ((10, 1.5, 5), "proposals_between_draws"),
        ((0, 1, float("inf")), "retained_draws"),
        ((10, 2, 2.5), "retained_draws"),
        ((False, 1, 1), "burn_in_proposals"),
    ],
    ids=[
        "fractional",
        "spacing_fractional",
        "infinite",
        "retained_fractional",
        "bool",
    ],
)
def test_sampling_interval_admits_non_integer_proposal_counts(
    interval: tuple[object, object, object], field: str
) -> None:
    """Every sampling interval count requires an exact integer."""
    refusal = {
        "burn_in_proposals": inference_packet.PacketRefusal.INTERVAL_BURN_IN_NOT_INT,
        "proposals_between_draws": inference_packet.PacketRefusal.INTERVAL_SPACING_NOT_INT,
        "retained_draws": inference_packet.PacketRefusal.INTERVAL_DRAWS_NOT_INT,
    }[field]
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.SamplingInterval(*interval)
    _assert_refusal(excinfo, refusal)


def test_seed_provenance_requires_an_exact_int_where_the_interval_does_not() -> None:
    """Seed provenance retains its exact-int refusal control."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainSeedProvenance(1.0, 2, 3, 4)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SEED_NOT_POSITIVE_INT)


@pytest.mark.parametrize(
    "source_hash",
    [list(_HASH), tuple(_HASH)],
    ids=["list", "tuple"],
)
def test_a_sequence_of_hex_characters_is_accepted_as_a_sha256_source_hash(
    source_hash: object,
) -> None:
    """A source hash must be an exact string before its content is screened."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.CheckpointReference(
            Path("/cell"), Path("/cell/checkpoints/x"), source_hash, {"world_size": 1}
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHECKPOINT_HASH_NOT_SHA256)


def test_a_bytes_source_hash_raises_typeerror_instead_of_a_declared_refusal() -> None:
    """Bytes source hashes take the declared hash refusal path."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.CheckpointReference(
            Path("/cell"), Path("/cell/checkpoints/x"), b"a" * 64, {"world_size": 1}
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHECKPOINT_HASH_NOT_SHA256)


@pytest.mark.parametrize("source_hash", ["a" * 63, "g" * 64], ids=["short", "non_hex"])
def test_malformed_string_hashes_are_still_refused(source_hash: str) -> None:
    """Malformed ordinary strings still reach the declared hash refusal."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.CheckpointReference(
            Path("/cell"), Path("/cell/checkpoints/x"), source_hash, {"world_size": 1}
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHECKPOINT_HASH_NOT_SHA256)


def test_status_ordering_guards_are_vacuous_against_a_caller_supplied_comparison() -> None:
    """Timestamp types are checked before caller-defined ordering runs."""
    value = _ReviewAlwaysAfter()
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.COMPLETED, value, value, value, value
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_TIMESTAMP_NOT_AWARE_DATETIME)


def test_mixing_naive_and_aware_timestamps_raises_typeerror_instead_of_a_declared_refusal() -> None:
    """Mixed timestamp awareness reaches the packet refusal vocabulary."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.IDLE,
            datetime(2026, 1, 1),
            datetime(2026, 1, 1, tzinfo=UTC),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_TIMESTAMP_NOT_AWARE_DATETIME)


def test_a_fully_naive_status_record_is_refused() -> None:
    """A status record requires aware timestamps."""
    created = datetime(2026, 1, 1)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.COMPLETED,
            created,
            created + timedelta(seconds=2),
            created + timedelta(seconds=1),
            created + timedelta(seconds=2),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_TIMESTAMP_NOT_AWARE_DATETIME)


def test_a_non_string_terminal_reason_raises_attributeerror_instead_of_a_declared_refusal() -> None:
    """Terminal reasons require exact strings before string operations."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.ERROR,
            _CREATED,
            _CREATED + timedelta(seconds=2),
            _CREATED + timedelta(seconds=1),
            _CREATED + timedelta(seconds=2),
            5,
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_TERMINAL_MISSING_REASON)


def test_real_aware_datetime_activity_violations_are_still_refused() -> None:
    """Real aware datetimes still exercise activity-order refusal."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.IDLE, _CREATED, _CREATED - timedelta(1)
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_ACTIVITY_PRECEDES_CREATION)


def test_real_aware_datetime_finish_violations_are_still_refused() -> None:
    """Real aware datetimes still exercise finish-order refusal."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.COMPLETED,
            _CREATED,
            _CREATED + timedelta(seconds=2),
            _CREATED + timedelta(seconds=2),
            _CREATED + timedelta(seconds=1),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_FINISH_PRECEDES_START)


def test_the_provenance_leaf_budget_does_not_bound_empty_container_breadth(tmp_path: Path) -> None:
    """The disposed container-axis behavior remains measurable and admitted."""
    reference = _checkpoint(tmp_path, topology={"ranks": [[] for _ in range(20_000)]})
    assert len(reference.topology_provenance["ranks"]) == 20_000


def test_the_provenance_leaf_budget_does_not_bound_empty_mapping_breadth(tmp_path: Path) -> None:
    """Empty mappings remain outside the leaf budget's priced axis."""
    reference = _checkpoint(
        tmp_path, topology={"ranks": {str(index): {} for index in range(20_000)}}
    )
    assert len(reference.topology_provenance["ranks"]) == 20_000


def test_actual_provenance_leaves_still_exhaust_the_budget(tmp_path: Path) -> None:
    """More than 8192 scalar leaves reaches the leaf-budget refusal."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _checkpoint(tmp_path, topology={"leaves": list(range(8193))})
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PROVENANCE_LEAF_BUDGET_EXCEEDED)


def test_deep_provenance_still_exhausts_the_depth_budget(tmp_path: Path) -> None:
    """Eight nested mapping levels reaches the depth-budget refusal."""
    topology: dict[str, object] = {}
    cursor = topology
    for index in range(8):
        child: dict[str, object] = {}
        cursor[str(index)] = child
        cursor = child
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _checkpoint(tmp_path, topology=topology)
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PROVENANCE_DEPTH_BUDGET_EXCEEDED)


def test_an_attributeerror_raised_inside_a_source_packet_property_is_misattributed_to_a_missing_interface(
    tmp_path: Path,
) -> None:
    """An inner property AttributeError is not relabeled as a missing field."""
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(AttributeError) as excinfo:
        inference_packet.launch_inference_packet(
            _ReviewPropertyAttributeErrorSource(checkpoint), checkpoint, (_chain(0),)
        )
    assert "inner_attribute_that_does_not_exist" in str(excinfo.value)


@pytest.mark.parametrize(
    (
        "state",
        "created_at",
        "last_activity_at",
        "started_at",
        "finished_at",
        "reason",
        "notified_at",
        "attribute",
    ),
    [
        (
            inference_packet.ChainState.RUNNING,
            datetime(2026, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            None,
            None,
            None,
            "started_at",
        ),
        (
            inference_packet.ChainState.COMPLETED,
            datetime(2026, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            None,
            None,
            "finished_at",
        ),
        (
            inference_packet.ChainState.COMPLETED,
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 20, tzinfo=UTC),
            None,
            None,
            "last_activity_at",
        ),
        (
            inference_packet.ChainState.ERROR,
            datetime(2026, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 10, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            "x",
            datetime(2026, 1, 3, tzinfo=UTC),
            "finish_notification_at",
        ),
    ],
    ids=[
        "running_started_before_creation",
        "terminal_wholly_before_creation",
        "finish_after_last_activity",
        "notified_wholly_before_creation",
    ],
)
def test_status_records_refuse_precedence_violations(
    state: object,
    created_at: datetime,
    last_activity_at: datetime,
    started_at: datetime,
    finished_at: datetime | None,
    reason: str | None,
    notified_at: datetime | None,
    attribute: str,
) -> None:
    """Every timestamp position must satisfy the status chronology."""
    del attribute
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state,
            created_at,
            last_activity_at,
            started_at,
            finished_at,
            reason,
            notified_at,
        )
    expected = (
        inference_packet.PacketRefusal.STATUS_ACTIVITY_PRECEDES_FINISH
        if finished_at is not None and last_activity_at < finished_at
        else inference_packet.PacketRefusal.STATUS_NON_ACTIVITY_PRECEDES_CREATION
    )
    _assert_refusal(excinfo, expected)


def test_creation_precedence_is_enforced_only_against_last_activity() -> None:
    """Creation precedence applies to start time as well as activity time."""
    created = datetime(2026, 1, 10, tzinfo=UTC)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.RUNNING,
            created,
            created,
            datetime(2026, 1, 1, tzinfo=UTC),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_NON_ACTIVITY_PRECEDES_CREATION)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            inference_packet.ChainState.IDLE,
            created,
            datetime(2026, 1, 1, tzinfo=UTC),
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_ACTIVITY_PRECEDES_CREATION)


def test_distributed_rank_keys_are_not_type_required(tmp_path: Path) -> None:
    """DDP rank keys require exact integers, including against bool equality."""
    for ranks in (
        {True: (tmp_path / "a").resolve(), 0: (tmp_path / "b").resolve()},
        {0.0: (tmp_path / "c").resolve(), 1.0: (tmp_path / "d").resolve()},
    ):
        with pytest.raises(inference_packet.InferencePacketError) as excinfo:
            inference_packet.DistributedCheckpointProvenance(2, ranks)
        _assert_refusal(excinfo, inference_packet.PacketRefusal.DDP_RANK_MEMBERSHIP_INCOMPLETE)


def test_launch_inference_packet_tolerates_an_iterator_of_chains(tmp_path: Path) -> None:
    """The launch route eagerly tuples an iterator of valid chains."""
    checkpoint = _checkpoint(tmp_path)
    source = SimpleNamespace(
        checkpoint_path=checkpoint.checkpoint_path,
        source_content_hash=checkpoint.source_content_hash,
        independent_sampler_inputs={"walkers": 4_096},
    )
    packet = inference_packet.launch_inference_packet(
        source, checkpoint, iter([_chain(0)])
    )
    assert type(packet.chains) is tuple


def test_a_nested_non_string_shaped_key_in_topology_provenance(tmp_path: Path) -> None:
    """A nested str subclass remains the disposed topology-key measurement."""
    key = _ReviewTopologyStringSubclass("nested")
    reference = _checkpoint(tmp_path, topology={"outer": {key: "value"}})
    nested_key = next(iter(reference.topology_provenance["outer"]))
    assert type(nested_key) is _ReviewTopologyStringSubclass


def test_the_contract_suites_immutability_arm_does_not_cover_the_chains_field() -> None:
    """The lifted R3 arm pins the missing tuple assertion in the old suite."""
    contract_text = Path(__file__).read_text()
    start = contract_text.index("def test_packet_inputs_and_topology_remain_immutable")
    end = contract_text.index("\ndef test_", start + 1)
    body = contract_text[start:end]
    assert "assert type(packet.chains) is tuple" not in body
    assert "assert isinstance(packet.chains, tuple)" not in body


def test_chain_id_must_not_be_empty() -> None:
    """Chain provenance requires a non-blank chain identifier."""
    valid = _chain(0)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.IndependentChain(
            "   ", valid.seeds, valid.interval, valid.status
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHAIN_ID_EMPTY)


def test_packet_requires_at_least_one_chain(tmp_path: Path) -> None:
    """A packet cannot launch without chain provenance."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _review_packet(_checkpoint(tmp_path), ())
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PACKET_HAS_NO_CHAIN)


def test_packet_chain_ids_must_be_unique(tmp_path: Path) -> None:
    """A packet cannot carry two chains under one identifier."""
    first = _chain(0)
    second = _chain(1)
    duplicate_id = inference_packet.IndependentChain(
        first.chain_id, second.seeds, second.interval, second.status
    )
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _review_packet(_checkpoint(tmp_path), (first, duplicate_id))
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PACKET_CHAIN_IDS_NOT_UNIQUE)


def test_packet_chain_seeds_must_be_unique(tmp_path: Path) -> None:
    """A packet cannot carry two chains under one chain seed."""
    first = _chain(0)
    second = _chain(1)
    duplicate_seed = inference_packet.IndependentChain(
        second.chain_id, first.seeds, second.interval, second.status
    )
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        _review_packet(_checkpoint(tmp_path), (first, duplicate_seed))
    _assert_refusal(excinfo, inference_packet.PacketRefusal.PACKET_CHAIN_SEEDS_NOT_UNIQUE)


def test_the_refusal_coverage_census_over_the_whole_enum() -> None:
    """The review census ratchet is closed after all members are pinned."""
    contract_text = Path(__file__).read_text()
    assert "match" + "=" not in contract_text
    uncovered = {
        member.name
        for member in inference_packet.PacketRefusal
        if member.name not in contract_text
    }
    assert uncovered == set()


def test_refusal_member_names_and_values_are_pairwise_substring_free() -> None:
    members = tuple(inference_packet.PacketRefusal)
    for index, left in enumerate(members):
        for right in members[index + 1 :]:
            assert left.name not in right.name
            assert right.name not in left.name
            assert left.value not in right.value
            assert right.value not in left.value


def test_review_module_is_the_same_object_as_the_contract_suite_loads() -> None:
    """Both dynamic loads reuse the already-registered module objects."""
    inference_alias = _reuse_or_load(
        "he_importance_inference_packet_review_probe",
        Path(__file__).with_name("inference_packet.py"),
    )
    l2_alias = _reuse_or_load(
        "he_importance_test_stage_coordinate_review_probe",
        Path(__file__).with_name("test_stage_coordinate.py"),
    )
    assert inference_alias is inference_packet
    assert l2_alias is l2_tests
    assert l2_alias.stage_coordinate is stage_coordinate


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (state, reason)
        for state in (
            inference_packet.ChainState.ERROR,
            inference_packet.ChainState.DEAD,
            inference_packet.ChainState.CANCELLED,
        )
        for reason in (None, "", "   ")
    ],
    ids=[
        f"{state.value}-{reason_id}"
        for state in (
            inference_packet.ChainState.ERROR,
            inference_packet.ChainState.DEAD,
            inference_packet.ChainState.CANCELLED,
        )
        for reason_id in ("none", "blank", "whitespace")
    ],
)
def test_a_non_completed_terminal_status_requires_a_reason(
    state: object, reason: str | None
) -> None:
    """Every non-completed terminal state rejects absent or blank reasons."""
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.ChainStatus(
            state,
            _CREATED,
            _CREATED + timedelta(seconds=2),
            _CREATED + timedelta(seconds=1),
            _CREATED + timedelta(seconds=2),
            reason,
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.STATUS_TERMINAL_MISSING_REASON)
