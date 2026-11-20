"""Contract tests for the separate, bounded HI ranking evaluator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
import importlib.util
from pathlib import Path
import re
import sys
from types import MappingProxyType
from typing import Callable

import pytest


_SPEC = importlib.util.spec_from_file_location("he_importance_ranking_evaluator", Path(__file__).with_name("ranking_evaluator.py"))
assert _SPEC is not None and _SPEC.loader is not None
ranking_evaluator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ranking_evaluator
_SPEC.loader.exec_module(ranking_evaluator)

_EVALUATOR_SPEC = importlib.util.spec_from_file_location(
    "he_importance_l3a_evaluator", Path(__file__).with_name("evaluator.py")
)
assert _EVALUATOR_SPEC is not None and _EVALUATOR_SPEC.loader is not None
l3a_evaluator = importlib.util.module_from_spec(_EVALUATOR_SPEC)
sys.modules[_EVALUATOR_SPEC.name] = l3a_evaluator
_EVALUATOR_SPEC.loader.exec_module(l3a_evaluator)


def _request(tmp_path: Path, observations: tuple[float, ...] = (0.1, 0.2)) -> object:
    cell = tmp_path / "cell-a"
    return ranking_evaluator.CheckpointRankingRequest(
        "checkpoint-a", cell / "checkpoints" / "1000.pt", cell, MappingProxyType({"stream": "ranking-a"}), observations, 2
    )


def test_ranking_output_is_ordinal_and_discards_mechanical_values(tmp_path: Path) -> None:
    signal = ranking_evaluator.evaluate_checkpoint(_request(tmp_path))
    assert signal == ranking_evaluator.RankingSignal("checkpoint-a", "keep", 0)
    assert set(signal.__dict__) == {"checkpoint_id", "disposition", "ordinal"}


def test_output_screen_refuses_nested_reference_content() -> None:
    with pytest.raises(ranking_evaluator.RankingEvaluationError, match="reference_energy"):
        ranking_evaluator.validate_ranking_output(
            {"checkpoint_id": {"reference_energy": 1.0}, "disposition": "keep", "ordinal": 0}
        )


@pytest.mark.parametrize("forbidden_key", ["reference_error", "e_ref", "energy", "local_energy"])
def test_blinding_traverses_frozen_nested_containers(forbidden_key: str, tmp_path: Path) -> None:
    request = _request(tmp_path)
    request = ranking_evaluator.CheckpointRankingRequest(
        request.checkpoint_id, request.checkpoint_path, request.cell_directory,
        MappingProxyType({"outer": (MappingProxyType({forbidden_key: 1}),)}), request.mechanical_observations, request.max_observations,
    )
    with pytest.raises(ranking_evaluator.RankingEvaluationError, match=forbidden_key):
        ranking_evaluator.evaluate_checkpoint(request)


@pytest.mark.parametrize("observations, expected", [((0.1,), "keep"), ((float("nan"),), "defer"), ((float("inf"),), "defer")])
def test_declared_nonfinite_policy_is_defer(observations: tuple[float, ...], expected: str, tmp_path: Path) -> None:
    assert ranking_evaluator.evaluate_checkpoint(_request(tmp_path, observations)).disposition == expected


def test_rng_provenance_is_required(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request = ranking_evaluator.CheckpointRankingRequest(
        request.checkpoint_id, request.checkpoint_path, request.cell_directory,
        MappingProxyType({}), request.mechanical_observations, request.max_observations,
    )
    with pytest.raises(ranking_evaluator.RankingEvaluationError, match="RNG provenance"):
        ranking_evaluator.evaluate_checkpoint(request)


def test_cost_bound_rejects_more_observations_than_declared(tmp_path: Path) -> None:
    with pytest.raises(ranking_evaluator.RankingEvaluationError, match="declared bounded cost"):
        ranking_evaluator.evaluate_checkpoint(_request(tmp_path, (0.1, 0.2, 0.3)))


def test_cost_bound_is_not_caller_selectable(tmp_path: Path) -> None:
    request = _request(tmp_path, tuple(float(index) for index in range(5000)))
    request = ranking_evaluator.CheckpointRankingRequest(
        request.checkpoint_id, request.checkpoint_path, request.cell_directory,
        request.rng_provenance, request.mechanical_observations, 10**9,
    )
    with pytest.raises(ranking_evaluator.RankingEvaluationError, match="evaluator bound"):
        ranking_evaluator.evaluate_checkpoint(request)


@pytest.mark.parametrize(
    ("dirname", "same_cell"),
    [("checkpoints", True), ("other", True), ("checkpoints", False), ("other", False)],
)
def test_checkpoint_binding_asserts_parent_cell_directory(
    dirname: str, same_cell: bool, tmp_path: Path
) -> None:
    request = _request(tmp_path)
    checkpoint_path = request.cell_directory / dirname / "1000.pt"
    cell_directory = request.cell_directory if same_cell else tmp_path / "cell-b"
    request = replace(request, checkpoint_path=checkpoint_path, cell_directory=cell_directory)
    if dirname == "checkpoints" and same_cell:
        assert ranking_evaluator.evaluate_checkpoint(request).disposition == "keep"
    else:
        with pytest.raises(ranking_evaluator.RankingEvaluationError):
            ranking_evaluator.evaluate_checkpoint(request)


def test_checkpoint_binding_uses_directory_identity(tmp_path: Path) -> None:
    cell = tmp_path / "cell-a"
    request = ranking_evaluator.CheckpointRankingRequest(
        "checkpoint-a", cell / "checkpoints" / ".." / "checkpoints" / "1000.pt", cell,
        MappingProxyType({"stream": "ranking-a"}), (0.1,), 2,
    )
    assert ranking_evaluator.evaluate_checkpoint(request).disposition == "keep"


def test_blinding_refuses_forbidden_string_values() -> None:
    with pytest.raises(ranking_evaluator.RankingEvaluationError, match="reference_energy"):
        ranking_evaluator.validate_ranking_output(
            {"checkpoint_id": "reference_energy=-2.903724", "disposition": "keep", "ordinal": 0}
        )


class _PlainObjectWitness:
    def __init__(self, payload: object) -> None:
        self.payload = payload


@dataclass
class _DataclassMappingHybrid(dict):
    payload: object

    def __post_init__(self) -> None:
        dict.__init__(self, {"container": "benign"})


@dataclass
class _DataclassSequenceHybrid(list):
    payload: object

    def __post_init__(self) -> None:
        list.__init__(self, ["benign"])


class _PrivateSlotWitness:
    __slots__ = ("__payload",)

    def __init__(self, payload: object) -> None:
        type(self).__dict__["_PrivateSlotWitness__payload"].__set__(self, payload)


class _SelfReferentialWitness:
    def __init__(self, sibling: object = "benign") -> None:
        self.self = self
        self.sibling = sibling


class _BenignEnum(Enum):
    ITEM = "benign"


@pytest.mark.parametrize(
    ("payload", "request_suffix", "output_suffix"),
    [
        (_PlainObjectWitness("reference_energy"), "payload.payload", "checkpoint_id.payload"),
        (b"reference_energy", "payload", "checkpoint_id"),
        (Path("reference_energy"), "payload", "checkpoint_id"),
        (deque(["reference_energy"]), "payload[0]", "checkpoint_id[0]"),
        (_PrivateSlotWitness("reference_energy"), "payload.__payload", "checkpoint_id.__payload"),
    ],
    ids=["plain-object", "bytes", "path", "deque", "private-slot"],
)
@pytest.mark.parametrize("route", ["request", "output"])
def test_l3b_shared_walker_reaches_new_structural_and_normalized_leaves(
    payload: object, request_suffix: str, output_suffix: str, route: str, tmp_path: Path
) -> None:
    if route == "request":
        label = f"ranking request.rng_provenance.{request_suffix}"
    else:
        label = f"ranking output.{output_suffix}"
    expected = f"{label} contains forbidden value 'reference_energy'"
    with pytest.raises(ranking_evaluator.RankingEvaluationError, match=re.escape(expected)) as exc:
        if route == "request":
            request = replace(_request(tmp_path), rng_provenance={"payload": payload})
            ranking_evaluator.evaluate_checkpoint(request)
        else:
            ranking_evaluator.validate_ranking_output(
                {"checkpoint_id": payload, "disposition": "keep", "ordinal": 0}
            )
    assert exc.type is ranking_evaluator.RankingEvaluationError


@pytest.mark.parametrize("key", [b"benign", Path("benign"), b"reference_energy"])
@pytest.mark.parametrize("route", ["request", "output"])
def test_l3b_refuses_non_string_mapping_keys_before_normalization(
    key: object, route: str, tmp_path: Path
) -> None:
    if route == "request":
        expected = "ranking request.rng_provenance keys must be strings"
        with pytest.raises(ranking_evaluator.RankingEvaluationError, match=re.escape(expected)) as exc:
            ranking_evaluator.evaluate_checkpoint(
                replace(_request(tmp_path), rng_provenance={key: "benign"})
            )
    else:
        expected = "ranking output.checkpoint_id keys must be strings"
        with pytest.raises(ranking_evaluator.RankingEvaluationError, match=re.escape(expected)) as exc:
            ranking_evaluator.validate_ranking_output(
                {"checkpoint_id": {key: "benign"}, "disposition": "keep", "ordinal": 0}
            )
    assert exc.type is ranking_evaluator.RankingEvaluationError


@pytest.mark.parametrize("hybrid", [_DataclassMappingHybrid, _DataclassSequenceHybrid])
@pytest.mark.parametrize("payload", ["reference_energy", "benign"])
@pytest.mark.parametrize("route", ["request", "output"])
def test_l3b_dataclass_fields_precede_hybrid_container_dispatch(
    hybrid: Callable[[object], object], payload: str, route: str, tmp_path: Path
) -> None:
    value = hybrid(payload)
    if payload == "benign":
        if route == "request":
            ranking_evaluator.evaluate_checkpoint(
                replace(_request(tmp_path), rng_provenance={"payload": value})
            )
        else:
            ranking_evaluator.validate_ranking_output(
                {"checkpoint_id": value, "disposition": "keep", "ordinal": 0}
            )
        return

    if route == "request":
        label = "ranking request.rng_provenance.payload.payload"
        with pytest.raises(ranking_evaluator.RankingEvaluationError, match=re.escape(
            f"{label} contains forbidden value 'reference_energy'"
        )) as exc:
            ranking_evaluator.evaluate_checkpoint(
                replace(_request(tmp_path), rng_provenance={"payload": value})
            )
    else:
        label = "ranking output.checkpoint_id.payload"
        with pytest.raises(ranking_evaluator.RankingEvaluationError, match=re.escape(
            f"{label} contains forbidden value 'reference_energy'"
        )) as exc:
            ranking_evaluator.validate_ranking_output(
                {"checkpoint_id": value, "disposition": "keep", "ordinal": 0}
            )
    assert exc.type is ranking_evaluator.RankingEvaluationError


@pytest.mark.parametrize("route", ["request", "output"])
def test_l3b_undecorated_plain_object_benign_control(route: str, tmp_path: Path) -> None:
    value = _PlainObjectWitness("benign")
    if route == "request":
        ranking_evaluator.evaluate_checkpoint(
            replace(_request(tmp_path), rng_provenance={"payload": value})
        )
    else:
        ranking_evaluator.validate_ranking_output(
            {"checkpoint_id": value, "disposition": "keep", "ordinal": 0}
        )


def test_l3b_shared_walker_preserves_benign_cycles_enum_and_deque(tmp_path: Path) -> None:
    ranking_evaluator.evaluate_checkpoint(
        replace(_request(tmp_path), rng_provenance={"payload": _SelfReferentialWitness()})
    )
    ranking_evaluator.evaluate_checkpoint(
        replace(_request(tmp_path), rng_provenance={"payload": _BenignEnum.ITEM})
    )
    ranking_evaluator.evaluate_checkpoint(
        replace(_request(tmp_path), rng_provenance={"payload": deque(["benign"])})
    )

    with pytest.raises(
        ranking_evaluator.RankingEvaluationError,
        match=re.escape("ranking request.rng_provenance.payload.sibling contains forbidden value 'reference_energy'"),
    ):
        ranking_evaluator.evaluate_checkpoint(
            replace(
                _request(tmp_path),
                rng_provenance={"payload": _SelfReferentialWitness("reference_energy")},
            )
        )


def test_l3b_energy_floor_remains_stricter_than_l3a() -> None:
    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        ranking_evaluator._refuse_blinded_content({"energy": 1}, "ranking request")
    l3a_evaluator._refuse_blinding_content({"energy": 1}, "evaluator input")
    l3a_evaluator._refuse_blinding_content({"local_energy": 1}, "evaluator input")


def test_selector_side_energy_peek_mutant_is_rejected() -> None:
    mutant = {"checkpoint_id": "checkpoint-a", "disposition": "keep", "ordinal": 1, "mechanical_value": -2.9}
    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        ranking_evaluator.validate_ranking_output(mutant)


def test_b_directory_identity_accepts_alias_and_preserves_request(tmp_path: Path) -> None:
    cell = tmp_path / "actual"
    (cell / "checkpoints").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(cell, target_is_directory=True)
    request = replace(
        _request(tmp_path),
        cell_directory=alias,
        checkpoint_path=cell / "checkpoints" / "1000.pt",
    )
    assert ranking_evaluator.evaluate_checkpoint(request).disposition == "keep"
    assert request.cell_directory == alias


@pytest.mark.parametrize(
    ("count", "ceiling", "allowed"),
    [(1, 1, True), (1024, 1024, True), (1025, 1024, False), (1025, 1025, False),
     (1, 10**9, False), (0, 1, False), (1, 0, False)],
)
def test_b_cost_bound_edges(tmp_path: Path, count: int, ceiling: int, allowed: bool) -> None:
    request = replace(
        _request(tmp_path),
        mechanical_observations=(0.1,) * count,
        max_observations=ceiling,
    )
    if allowed:
        assert ranking_evaluator.evaluate_checkpoint(request).disposition == "keep"
    else:
        with pytest.raises(ranking_evaluator.RankingEvaluationError):
            ranking_evaluator.evaluate_checkpoint(request)


def test_b_rng_presence_is_required(tmp_path: Path) -> None:
    request = replace(_request(tmp_path), rng_provenance=MappingProxyType({}))
    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        ranking_evaluator.evaluate_checkpoint(request)


@pytest.mark.parametrize("route", ["ranking request", "ranking output"])
@pytest.mark.parametrize("channel", ["key", "value"])
def test_b_each_screen_alone(
    route: str, channel: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ranking_evaluator._refuse_blinded_content
    seen: list[str] = []

    def only_target(value: object, path: str) -> None:
        seen.append(path)
        if path == route or path.startswith(route + ".") or path.startswith(route + "["):
            original(value, path)

    monkeypatch.setattr(ranking_evaluator, "_refuse_blinded_content", only_target)
    only_target({"reference_energy": 1}, "disabled control")
    bad = {"reference_energy": 1} if channel == "key" else "local_energy"
    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        if route == "ranking request":
            request = replace(_request(tmp_path), rng_provenance=MappingProxyType({"stream": bad}))
            ranking_evaluator.evaluate_checkpoint(request)
        else:
            ranking_evaluator.validate_ranking_output(
                {"checkpoint_id": bad, "disposition": "keep", "ordinal": 0}
            )
    assert route in seen


@pytest.mark.parametrize(
    "change",
    [{"ordinal": -1}, {"ordinal": True}, {"ordinal": 0.5},
     {"disposition": "other"}, {"mechanical_value": -2.8}],
)
def test_b_ordinal_schema_rejects_invalid_limbs(change: dict[str, object]) -> None:
    signal: dict[str, object] = {"checkpoint_id": "cp", "disposition": "keep", "ordinal": 0}
    signal.update(change)
    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        ranking_evaluator.validate_ranking_output(signal)


@pytest.mark.parametrize(
    ("value", "disposition", "ordinal"),
    [(0.1, "keep", 0), (float("nan"), "defer", 1),
     (float("inf"), "defer", 1), (-float("inf"), "defer", 1)],
)
def test_b_nonfinite_declared_policy_retains_no_observation(
    tmp_path: Path, value: float, disposition: str, ordinal: int
) -> None:
    result = ranking_evaluator.evaluate_checkpoint(
        replace(_request(tmp_path), mechanical_observations=(value,))
    )
    assert result == ranking_evaluator.RankingSignal("checkpoint-a", disposition, ordinal)
    assert set(vars(result)) == {"checkpoint_id", "disposition", "ordinal"}


@pytest.mark.parametrize(
    "token",
    ["reference", "reference_energy", "e_ref", "accuracy", "x-reference-y", "E-REF", "ACCURACY"],
)
@pytest.mark.parametrize("channel", ["key", "value"])
def test_l3b_common_frozen_predicate_at_visited_nodes(token: str, channel: str) -> None:
    leaf = {token: 1} if channel == "key" else {"label": token}
    nested = MappingProxyType({"outer": [MappingProxyType({"inner": (leaf,)})]})
    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        ranking_evaluator._refuse_blinded_content(nested, "probe")


@pytest.mark.parametrize("token", ["energy", "local_energy", "some-energy-value", "LOCAL-ENERGY"])
@pytest.mark.parametrize("channel", ["key", "value"])
def test_l3b_refuses_energy_tokens_in_both_channels(token: str, channel: str) -> None:
    """The frozen adjudication keeps L3b's energy-token refusal floor."""
    payload = {token: 1} if channel == "key" else {"label": token}
    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        ranking_evaluator._refuse_blinded_content(payload, "L3b forbidden payload")


def test_r2_b_non_string_mapping_key_is_refused_with_declared_error() -> None:
    """The non-string-key limb must raise the declared error, not an AttributeError."""

    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        ranking_evaluator._refuse_blinded_content({1: "x"}, "probe")


def test_r2_b_non_string_key_is_refused_at_nested_node() -> None:
    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        ranking_evaluator._refuse_blinded_content({"outer": ({3: "z"},)}, "probe")


def test_r2_b_output_missing_required_key_is_refused() -> None:
    """A subset key set must be refused with the declared error, not a KeyError."""

    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        ranking_evaluator.validate_ranking_output({"checkpoint_id": "cp", "disposition": "keep"})


def test_r2_b_output_non_mapping_with_matching_key_names_is_refused() -> None:
    """A non-Mapping whose iteration yields exactly the key names must still be refused."""

    with pytest.raises(ranking_evaluator.RankingEvaluationError):
        ranking_evaluator.validate_ranking_output(["checkpoint_id", "disposition", "ordinal"])


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")], ids=["nan", "posinf", "neginf"])
@pytest.mark.parametrize("nonfinite_first", [False, True], ids=["finite-first", "nonfinite-first"])
def test_r3_mixed_observations_defer(tmp_path, nonfinite, nonfinite_first):
    observations = (nonfinite, 0.25) if nonfinite_first else (0.25, nonfinite)
    cell = tmp_path / "cell"
    request = ranking_evaluator.CheckpointRankingRequest(
        "cp", cell / "checkpoints" / "1.pt", cell,
        MappingProxyType({"stream": "ranking-a"}), observations, 2,
    )
    result = ranking_evaluator.evaluate_checkpoint(request)
    assert result == ranking_evaluator.RankingSignal("cp", "defer", 1)
    assert set(vars(result)) == {"checkpoint_id", "disposition", "ordinal"}
