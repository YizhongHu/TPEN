"""Contract tests for the separate, bounded HI ranking evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType

import pytest


_SPEC = importlib.util.spec_from_file_location("he_importance_ranking_evaluator", Path(__file__).with_name("ranking_evaluator.py"))
assert _SPEC is not None and _SPEC.loader is not None
ranking_evaluator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ranking_evaluator
_SPEC.loader.exec_module(ranking_evaluator)


def _request(tmp_path: Path, observations: tuple[float, ...] = (0.1, 0.2)) -> object:
    cell = tmp_path / "cell-a"
    return ranking_evaluator.CheckpointRankingRequest(
        "checkpoint-a", cell / "checkpoints" / "1000.pt", cell, MappingProxyType({"stream": "ranking-a"}), observations, 2
    )


def test_ranking_output_is_ordinal_and_discards_mechanical_values(tmp_path: Path) -> None:
    signal = ranking_evaluator.evaluate_checkpoint(_request(tmp_path))
    assert signal == ranking_evaluator.RankingSignal("checkpoint-a", "keep", 0)
    assert set(signal.__dict__) == {"checkpoint_id", "disposition", "ordinal"}


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


def test_cost_bound_rejects_more_observations_than_declared(tmp_path: Path) -> None:
    with pytest.raises(ranking_evaluator.RankingEvaluationError, match="declared bounded cost"):
        ranking_evaluator.evaluate_checkpoint(_request(tmp_path, (0.1, 0.2, 0.3)))


@pytest.mark.parametrize("path", ["checkpoints/1000.pt", "other/1000.pt"])
def test_checkpoint_binding_asserts_parent_cell_directory(path: str, tmp_path: Path) -> None:
    request = _request(tmp_path)
    request = ranking_evaluator.CheckpointRankingRequest(
        request.checkpoint_id, tmp_path / "other-cell" / path, request.cell_directory,
        request.rng_provenance, request.mechanical_observations, request.max_observations,
    )
    with pytest.raises(ranking_evaluator.RankingEvaluationError, match="parent cell directory"):
        ranking_evaluator.evaluate_checkpoint(request)


def test_selector_side_energy_peek_mutant_is_rejected() -> None:
    mutant = {"checkpoint_id": "checkpoint-a", "disposition": "keep", "ordinal": 1, "energy": -2.9}
    with pytest.raises(ranking_evaluator.RankingEvaluationError, match="ordinal signal fields"):
        ranking_evaluator.validate_ranking_output(mutant)
