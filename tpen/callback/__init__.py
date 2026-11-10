"""Callback primitives for configured TPEN runs."""

from __future__ import annotations

from .base import Callback, StatefulCallback
from .cadence import Cadence, CadenceGate, StepCadence, StepCadenceGate, SubscriptionGroup
from .metadata import Metadata
from .snapshot import ConfigSnapshot, ResolvedConfigSnapshot
from .terminal_logging import configure_terminal_logging


def __getattr__(name: str) -> object:
    """Load torch-dependent callback classes only when they are requested.

    A `StatefulCallback` declaring a ``state_type`` has to import that state
    class at class-creation time, and importing anything from ``tpen.training``
    or ``tpen.evaluation`` runs that package's ``__init__``, which pulls in
    torch. Every such callback therefore has to be loaded lazily to keep
    ``import tpen.callback`` torch-free. `FailureLog` needs no state but shares
    a module with `ArtifactIndex`, so it is loaded lazily with it.
    """

    if name == "ArtifactIndex":
        from .evaluation import ArtifactIndex

        return ArtifactIndex
    if name == "Checkpoint":
        from .checkpoint import Checkpoint

        return Checkpoint
    if name == "InitialCheckpoint":
        from .initial_checkpoint import InitialCheckpoint

        return InitialCheckpoint
    if name == "ModuleNorms":
        from .module_norms import ModuleNorms

        return ModuleNorms
    if name == "ParameterActivity":
        from .parameter_activity import ParameterActivity

        return ParameterActivity
    if name == "Status":
        from .status import Status

        return Status
    if name == "FailureLog":
        from .evaluation import FailureLog

        return FailureLog
    if name == "RuntimeEquivariance":
        from .equivariance import RuntimeEquivariance

        return RuntimeEquivariance
    if name == "DataIntegrity":
        from .health import DataIntegrity

        return DataIntegrity
    if name == "GradientStats":
        from .health import GradientStats

        return GradientStats
    if name == "SamplerHealth":
        from .health import SamplerHealth

        return SamplerHealth
    if name == "FactorScalars":
        from .factor_scalars import FactorScalars

        return FactorScalars
    if name == "ElectronNucleusDistanceQuantiles":
        from .electron_nucleus_distance import ElectronNucleusDistanceQuantiles

        return ElectronNucleusDistanceQuantiles
    if name == "DiagnosticTiming":
        from .timing import DiagnosticTiming

        return DiagnosticTiming
    if name == "EvaluationComponentTiming":
        from .timing import EvaluationComponentTiming

        return EvaluationComponentTiming
    if name == "EvaluationTiming":
        from .timing import EvaluationTiming

        return EvaluationTiming
    if name == "ResourceUsage":
        from .resource_usage import ResourceUsage

        return ResourceUsage
    if name == "RunTiming":
        from .timing import RunTiming

        return RunTiming
    if name == "TrainPhaseTiming":
        from .timing import TrainPhaseTiming

        return TrainPhaseTiming
    if name == "TrainStepTiming":
        from .timing import TrainStepTiming

        return TrainStepTiming
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ArtifactIndex",
    "Cadence",
    "CadenceGate",
    "Callback",
    "Checkpoint",
    "ConfigSnapshot",
    "DataIntegrity",
    "DiagnosticTiming",
    "EvaluationComponentTiming",
    "EvaluationTiming",
    "ElectronNucleusDistanceQuantiles",
    "FailureLog",
    "FactorScalars",
    "GradientStats",
    "InitialCheckpoint",
    "Metadata",
    "ModuleNorms",
    "ParameterActivity",
    "ResolvedConfigSnapshot",
    "ResourceUsage",
    "RunTiming",
    "RuntimeEquivariance",
    "SamplerHealth",
    "StatefulCallback",
    "Status",
    "StepCadence",
    "StepCadenceGate",
    "SubscriptionGroup",
    "TrainPhaseTiming",
    "TrainStepTiming",
    "configure_terminal_logging",
]
