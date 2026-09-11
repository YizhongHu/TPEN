"""Bounded, mechanics-only checkpoint ranking for the HI scan.

This module is deliberately a separate arm from independent-sampler
evaluation.  It accepts a fixed, finite sequence of mechanical observations
and emits only an ordinal keep/defer signal; it has no sampler inputs and no
analysis-result representation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any


class RankingEvaluationError(ValueError):
    """A cheap-ranking request or signal violates its mechanics-only contract."""


@dataclass(frozen=True)
class CheckpointRankingRequest:
    """One bounded-cost, post-checkpoint mechanics request.

    ``cell_directory`` is an identity-bearing witness: checkpoints are valid
    only when they reside under that exact parent cell directory.  The maximum
    cost is ``max_observations`` mechanical observations per checkpoint.
    """

    checkpoint_id: str
    checkpoint_path: Path
    cell_directory: Path
    rng_provenance: Mapping[str, Any]
    mechanical_observations: tuple[float, ...]
    max_observations: int


@dataclass(frozen=True)
class RankingSignal:
    """An ordinal mechanics signal, with no value suitable for analysis."""

    checkpoint_id: str
    disposition: str
    ordinal: int


_FORBIDDEN_TOKENS = frozenset(
    {
        "accuracy",
        "e_ref",
        "energy",
        "local_energy",
        "reference",
        "reference_energy",
    }
)
_SIGNAL_KEYS = frozenset({"checkpoint_id", "disposition", "ordinal"})


def evaluate_checkpoint(request: CheckpointRankingRequest) -> RankingSignal:
    """Evaluate bounded mechanics and return a ranking-only ordinal signal.

    This evaluator's declared nonfinite policy is ``defer``: a nonfinite
    mechanical observation produces a deferred ordinal signal rather than
    raising like the trainer seam.  The result retains no observation values.
    """

    _refuse_blinded_content(request, "ranking request")
    _validate_request(request)
    disposition = "keep" if all(value == value and abs(value) != float("inf") for value in request.mechanical_observations) else "defer"
    # This is a mechanics-only preorder: eligible checkpoints precede deferred
    # ones.  It exposes no aggregate or per-observation numeric result.
    signal = RankingSignal(request.checkpoint_id, disposition, 0 if disposition == "keep" else 1)
    validate_ranking_output(signal)
    return signal


def validate_ranking_output(output: Any) -> None:
    """Reject any ranking output that carries a value-like analysis field."""

    content = _dataclass_members(output) if is_dataclass(output) else output
    if not isinstance(content, Mapping) or frozenset(content) != _SIGNAL_KEYS:
        raise RankingEvaluationError("ranking output must contain only ordinal signal fields")
    _refuse_blinded_content(content, "ranking output")
    if type(content["ordinal"]) is not int or content["ordinal"] < 0:
        raise RankingEvaluationError("ranking output ordinal must be a nonnegative integer")
    if content["disposition"] not in {"keep", "defer"}:
        raise RankingEvaluationError("ranking output disposition is not declared")


def _validate_request(request: CheckpointRankingRequest) -> None:
    if request.max_observations <= 0:
        raise RankingEvaluationError("max_observations must be positive")
    if not request.mechanical_observations or len(request.mechanical_observations) > request.max_observations:
        raise RankingEvaluationError("mechanical observations exceed the declared bounded cost")
    if not request.rng_provenance:
        raise RankingEvaluationError("ranking jobs require distinct RNG provenance")
    if request.checkpoint_path.parent.name != "checkpoints" or request.checkpoint_path.parent.parent != request.cell_directory:
        raise RankingEvaluationError("checkpoint path must bind to its parent cell directory")


def _refuse_blinded_content(value: Any, path: str) -> None:
    """Traverse mappings, frozen containers, and dataclasses by structure."""

    if is_dataclass(value):
        _refuse_blinded_content(_dataclass_members(value), path)
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise RankingEvaluationError(f"{path} keys must be strings")
            normalized = key.lower().replace("-", "_")
            if normalized in _FORBIDDEN_TOKENS or "reference" in normalized or "energy" in normalized:
                raise RankingEvaluationError(f"{path} contains forbidden field {key!r}")
            _refuse_blinded_content(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _refuse_blinded_content(nested, f"{path}[{index}]")


def _dataclass_members(value: Any) -> Mapping[str, Any]:
    """Expose dataclass members without deepcopying frozen mapping proxies."""

    return {field.name: getattr(value, field.name) for field in fields(value)}
