"""Bounded, mechanics-only checkpoint ranking for the HI scan.

This module is deliberately a separate arm from independent-sampler
evaluation.  It accepts a fixed, finite sequence of mechanical observations
and emits only an ordinal keep/defer signal; it has no sampler inputs and no
analysis-result representation.  Its blinding screen delegates structural
reachability and leaf normalization to the study-owned shared walker while
retaining this evaluator's stricter token policy and error labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any


class RankingEvaluationError(ValueError):
    """A cheap-ranking request or signal violates its mechanics-only contract."""


_WALKER_NAME = "he_importance_blinding_traversal"
_WALKER_PATH = Path(__file__).with_name("blinding_traversal.py")
_WALKER_SPEC = importlib.util.spec_from_file_location(_WALKER_NAME, _WALKER_PATH)
assert _WALKER_SPEC is not None and _WALKER_SPEC.loader is not None
_WALKER = sys.modules.get(_WALKER_NAME)
if _WALKER is None:
    _WALKER = importlib.util.module_from_spec(_WALKER_SPEC)
    sys.modules[_WALKER_NAME] = _WALKER
    _WALKER_SPEC.loader.exec_module(_WALKER)


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
MAX_MECHANICAL_OBSERVATIONS = 1024


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
    if request.max_observations > MAX_MECHANICAL_OBSERVATIONS:
        raise RankingEvaluationError("max_observations exceeds the evaluator bound")
    if not request.mechanical_observations or len(request.mechanical_observations) > request.max_observations:
        raise RankingEvaluationError("mechanical observations exceed the declared bounded cost")
    if not request.rng_provenance:
        raise RankingEvaluationError("ranking jobs require distinct RNG provenance")
    checkpoint_directory = request.checkpoint_path.parent.resolve()
    cell_directory = request.cell_directory.resolve()
    if checkpoint_directory.name != "checkpoints" or checkpoint_directory.parent != cell_directory:
        raise RankingEvaluationError("checkpoint path must bind to its parent cell directory")


def _refuse_blinded_content(value: Any, path: str) -> None:
    """Traverse structural and normalized content with L3b's frozen policy.

    The shared walker owns reachability, normalization, and cycle handling.  It
    does not own a token set or an exception policy: those remain local to L3b
    so its stricter ``energy`` token floor and established labels are unchanged.
    """

    def is_forbidden(text: str) -> bool:
        normalized = text.lower().replace("-", "_")
        return _is_forbidden_token(normalized)

    def error_factory(path: str, content: Any, reason: str) -> Exception:
        if reason == "key-type":
            parent = getattr(path, "parent", path)
            return RankingEvaluationError(f"{parent} keys must be strings")
        if reason == "key":
            parent = getattr(path, "parent", path.rsplit(".", 1)[0])
            return RankingEvaluationError(f"{parent} contains forbidden field {content!r}")
        if reason == "decode":
            return RankingEvaluationError(f"{path} contains undecodable blinding content")
        return RankingEvaluationError(f"{path} contains forbidden value {content!r}")

    _WALKER.walk_blinding_content(
        value,
        path,
        is_forbidden=is_forbidden,
        error_factory=error_factory,
    )


def _is_forbidden_token(normalized: str) -> bool:
    return normalized in _FORBIDDEN_TOKENS or "reference" in normalized or "energy" in normalized


def _dataclass_members(value: Any) -> Mapping[str, Any]:
    """Expose dataclass members without deepcopying frozen mapping proxies."""

    return {field.name: getattr(value, field.name) for field in fields(value)}
