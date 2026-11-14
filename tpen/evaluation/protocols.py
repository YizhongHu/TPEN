"""Protocols for composable evaluation components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from tpen.evaluation.bundle import EvaluationBundle, GeneratedConfigurations
from tpen.evaluation.results import SummaryResult
from tpen.evaluation.task import ArtifactLevel, FailurePolicy


@dataclass(frozen=True)
class EvaluationContext:
    """Evaluator-local context shared by one task's components."""

    namespace: str
    artifact_level: ArtifactLevel
    task_failure_policy: FailurePolicy
    device: torch.device | None
    dtype: torch.dtype | None
    seed: int | None
    run_dir: Path
    task_output_dir: Path
    metadata: Mapping[str, Any]
    resolved_config_identity: Mapping[str, str] | None = None
    model_provenance: str = "constructed"
    checkpoint_sha256: str | None = None


class Generator(Protocol):
    """Protocol for evaluation configuration generators."""

    name: str

    def generate(
        self,
        *,
        model: torch.nn.Module | None,
        context: EvaluationContext,
    ) -> GeneratedConfigurations:
        """Generate configurations and bookkeeping metadata."""
        ...


class Calculator(Protocol):
    """Protocol for primitive evaluation calculations."""

    name: str

    def calculate(
        self,
        *,
        model: torch.nn.Module,
        bundle: EvaluationBundle,
        context: EvaluationContext,
    ) -> EvaluationBundle:
        """Return an updated bundle with raw primitive values."""
        ...


class Summary(Protocol):
    """Protocol for scalar summaries and artifact writers."""

    name: str
    required_fields: frozenset[str]

    def summarize(
        self,
        *,
        bundle: EvaluationBundle,
        context: EvaluationContext,
        namespace: str,
    ) -> SummaryResult:
        """Summarize raw bundle values into metrics/artifacts."""
        ...


__all__ = ["Calculator", "EvaluationContext", "Generator", "Summary"]
