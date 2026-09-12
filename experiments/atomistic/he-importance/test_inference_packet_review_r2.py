"""R2 reviewer probes for adversarial identity and provenance values.

These tests are intentionally separate from the contract suite.  They record
the reviewer-designed boundary expectations without changing production code.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_boundary() -> object:
    name = "he_importance_inference_packet_review_r2"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("inference_packet.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inference_packet = _load_boundary()
_HASH = "a" * 64
_CREATED = datetime(2026, 9, 12, tzinfo=UTC)


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


def _checkpoint(tmp_path: Path) -> object:
    parent = (tmp_path / "O1" / _HASH).resolve()
    return inference_packet.CheckpointReference(
        parent,
        parent / "checkpoints" / "update-00050000",
        _HASH,
        {"world_size": 1, "launcher": "review"},
    )


def _chain(index: int, chain_id: str | None = None) -> object:
    status = inference_packet.ChainStatus(
        state=inference_packet.ChainState.IDLE,
        created_at=_CREATED,
        last_activity_at=_CREATED,
    )
    seeds = inference_packet.ChainSeedProvenance(
        training_seed=100 + index,
        calibration_seed=200 + index,
        inference_seed=300 + index,
        chain_seed=400 + index,
    )
    return inference_packet.IndependentChain(
        f"chain-{index}" if chain_id is None else chain_id,
        seeds,
        inference_packet.SamplingInterval(10_000, 10, 32_768),
        status,
    )


def _source(checkpoint: object, source_hash: object = _HASH, inputs: object = None) -> object:
    return SimpleNamespace(
        checkpoint_path=checkpoint.checkpoint_path,
        source_content_hash=source_hash,
        independent_sampler_inputs={"walkers": 4_096} if inputs is None else inputs,
    )


def _assert_refusal(excinfo: pytest.ExceptionInfo[BaseException], refusal: object) -> None:
    assert type(excinfo.value) is inference_packet.InferencePacketError
    assert excinfo.value.refusal is refusal


@pytest.mark.parametrize("source_hash", [_AlwaysEqualHash(), _LyingHash("b" * 64)])
def test_source_hash_comparison_cannot_admit_a_nonmatching_caller_value(
    tmp_path: Path, source_hash: object
) -> None:
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.launch_inference_packet(
            _source(checkpoint, source_hash), checkpoint, (_chain(0),)
        )
    assert excinfo.value.refusal in {
        inference_packet.PacketRefusal.SOURCE_HASH_MISMATCH,
        inference_packet.PacketRefusal.CHECKPOINT_HASH_NOT_SHA256,
    }


def test_ordinary_matching_source_hash_remains_a_positive_control(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    packet = inference_packet.launch_inference_packet(
        _source(checkpoint), checkpoint, (_chain(0),)
    )
    assert packet.checkpoint is checkpoint


def test_chain_ids_are_unique_by_underlying_plain_string_content(tmp_path: Path) -> None:
    first = _chain(0, _SameContentDistinctIdentity("same", 101))
    second = _chain(1, _SameContentDistinctIdentity("same", 202))
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.InferencePacket(
            checkpoint, {"walkers": 4_096}, (first, second), None
        )
    _assert_refusal(
        excinfo, inference_packet.PacketRefusal.PACKET_CHAIN_IDS_NOT_UNIQUE
    )


def test_chain_id_empty_check_cannot_be_redirected_by_str_subclass_strip() -> None:
    valid = _chain(0)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.IndependentChain(
            _WhitespaceLooksValid("   "), valid.seeds, valid.interval, valid.status
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.CHAIN_ID_EMPTY)


def test_topology_provenance_key_is_normalized_or_refused(tmp_path: Path) -> None:
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
    assert type(stored_key) is str
    assert stored_key == "topology"
    marker.value = "mutated-training-walker"
    assert not hasattr(stored_key, "marker")


def test_the_same_mutable_key_fixture_is_refused_in_sampler_inputs(tmp_path: Path) -> None:
    marker = _RawMarker("training-walker")
    key = _MutableProvenanceKey("walkers", marker)
    with pytest.raises(inference_packet.InferencePacketError) as excinfo:
        inference_packet.InferencePacket(
            _checkpoint(tmp_path), {key: 4_096}, (_chain(0),), None
        )
    _assert_refusal(excinfo, inference_packet.PacketRefusal.SAMPLER_KEY_NOT_A_STRING)
