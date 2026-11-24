"""Callback primitives for configured TPEN runs."""

from __future__ import annotations

from .base import (
    Callback,
    ManagedCallback,
    ManagedLifecycleError,
    ResourceOwnershipClaim,
    ResourceOwnershipConflictError,
    RunResource,
    StatefulCallback,
    canonicalize_allocator_reset_scope,
    claims_conflict,
    is_managed_callback,
    validate_run_resource,
)
from .cadence import Cadence, CadenceGate, StepCadence, StepCadenceGate, SubscriptionGroup
from .metadata import Metadata
from .snapshot import ConfigSnapshot, ResolvedConfigSnapshot
from .terminal_logging import configure_terminal_logging

#: Closed torch-free set of shipped non-managed callback dotted targets
#: (package exports, subpackage re-exports, and defining-module paths).
#: Preflight classification uses this table to skip eager ``get_class`` imports
#: of ordinary shipped callbacks (whose modules pull in ``torch``) until inside
#: the managed ``PREPARING`` scope.
_KNOWN_NON_MANAGED_CALLBACK_TARGETS: frozenset[str] = frozenset(
    {
        "tpen.callback.AmplitudeRiskFraction",
        "tpen.callback.amplitude_risk.AmplitudeRiskFraction",
        "tpen.callback.ArtifactIndex",
        "tpen.callback.evaluation.ArtifactIndex",
        "tpen.callback.Checkpoint",
        "tpen.callback.checkpoint.Checkpoint",
        "tpen.callback.ConfigSnapshot",
        "tpen.callback.snapshot.ConfigSnapshot",
        "tpen.callback.DataIntegrity",
        "tpen.callback.health.DataIntegrity",
        "tpen.callback.health.data_integrity.DataIntegrity",
        "tpen.callback.DiagnosticTiming",
        "tpen.callback.timing.DiagnosticTiming",
        "tpen.callback.timing.diagnostic_timing.DiagnosticTiming",
        "tpen.callback.ElectronNucleusDistanceQuantiles",
        "tpen.callback.electron_nucleus_distance.ElectronNucleusDistanceQuantiles",
        "tpen.callback.EvaluationComponentTiming",
        "tpen.callback.timing.EvaluationComponentTiming",
        "tpen.callback.timing.evaluation_component_timing.EvaluationComponentTiming",
        "tpen.callback.EvaluationTiming",
        "tpen.callback.timing.EvaluationTiming",
        "tpen.callback.timing.evaluation_timing.EvaluationTiming",
        "tpen.callback.FactorScalars",
        "tpen.callback.factor_scalars.FactorScalars",
        "tpen.callback.FailureLog",
        "tpen.callback.evaluation.FailureLog",
        "tpen.callback.GradientStats",
        "tpen.callback.health.GradientStats",
        "tpen.callback.health.gradient_stats.GradientStats",
        "tpen.callback.InitialCheckpoint",
        "tpen.callback.initial_checkpoint.InitialCheckpoint",
        "tpen.callback.Metadata",
        "tpen.callback.metadata.Metadata",
        "tpen.callback.ModuleNorms",
        "tpen.callback.module_norms.ModuleNorms",
        "tpen.callback.OptimizerState",
        "tpen.callback.optimizer_state.OptimizerState",
        "tpen.callback.ParameterActivity",
        "tpen.callback.parameter_activity.ParameterActivity",
        "tpen.callback.PlannedYield",
        "tpen.callback.planned_yield.PlannedYield",
        "tpen.callback.RawTrainingBatchRecorder",
        "tpen.callback.raw_training_batch_recorder.RawTrainingBatchRecorder",
        "tpen.callback.ResolvedConfigSnapshot",
        "tpen.callback.snapshot.ResolvedConfigSnapshot",
        "tpen.callback.ResourceUsage",
        "tpen.callback.resource_usage.ResourceUsage",
        "tpen.callback.RunTiming",
        "tpen.callback.timing.RunTiming",
        "tpen.callback.timing.run_timing.RunTiming",
        "tpen.callback.RuntimeEquivariance",
        "tpen.callback.equivariance.RuntimeEquivariance",
        "tpen.callback.SamplerHealth",
        "tpen.callback.health.SamplerHealth",
        "tpen.callback.health.sampler_health.SamplerHealth",
        "tpen.callback.Status",
        "tpen.callback.status.Status",
        "tpen.callback.TrainPhaseTiming",
        "tpen.callback.timing.TrainPhaseTiming",
        "tpen.callback.timing.train_phase_timing.TrainPhaseTiming",
        "tpen.callback.TrainStepTiming",
        "tpen.callback.timing.TrainStepTiming",
        "tpen.callback.timing.train_step_timing.TrainStepTiming",
        "tpen.callback.TrainingBatchRecorder",
        "tpen.callback.raw_training_batch_recorder.TrainingBatchRecorder",
    }
)


def __getattr__(name: str) -> object:
    """Load torch-dependent callback classes only when they are requested.

    A `StatefulCallback` declaring a ``state_type`` has to import that state
    class at class-creation time, and importing anything from ``tpen.training``
    or ``tpen.evaluation`` runs that package's ``__init__``, which pulls in
    torch. Every such callback therefore has to be loaded lazily to keep
    ``import tpen.callback`` torch-free. `FailureLog` needs no state but shares
    a module with `ArtifactIndex`, so it is loaded lazily with it.
    """

    if name == "AmplitudeRiskFraction":
        from .amplitude_risk import AmplitudeRiskFraction

        return AmplitudeRiskFraction
    if name == "ArtifactIndex":
        from .evaluation import ArtifactIndex

        return ArtifactIndex
    if name == "Checkpoint":
        from .checkpoint import Checkpoint

        return Checkpoint
    if name == "PlannedYield":
        from .planned_yield import PlannedYield

        return PlannedYield
    if name == "PlannedYieldSignal":
        from .planned_yield import PlannedYieldSignal

        return PlannedYieldSignal
    if name == "InitialCheckpoint":
        from .initial_checkpoint import InitialCheckpoint

        return InitialCheckpoint
    if name == "ModuleNorms":
        from .module_norms import ModuleNorms

        return ModuleNorms
    if name == "OptimizerState":
        from .optimizer_state import OptimizerState

        return OptimizerState
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
    if name in {
        "AllocatorSource",
        "LinuxCgroupV2Source",
        "LinuxMemorySource",
        "LinuxProcessTreeSource",
        "NvidiaDeviceSource",
        "RUsageSource",
        "ResourceSamples",
    }:
        from . import resource_samples

        return getattr(resource_samples, name)
    if name in {"RawTrainingBatchRecorder", "TrainingBatchRecorder"}:
        from .raw_training_batch_recorder import RawTrainingBatchRecorder

        return RawTrainingBatchRecorder
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
    "AllocatorSource",
    "AmplitudeRiskFraction",
    "ArtifactIndex",
    "Cadence",
    "CadenceGate",
    "Callback",
    "Checkpoint",
    "PlannedYield",
    "PlannedYieldSignal",
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
    "LinuxCgroupV2Source",
    "LinuxMemorySource",
    "LinuxProcessTreeSource",
    "ManagedCallback",
    "ManagedLifecycleError",
    "Metadata",
    "ModuleNorms",
    "NvidiaDeviceSource",
    "OptimizerState",
    "ParameterActivity",
    "RUsageSource",
    "RawTrainingBatchRecorder",
    "ResolvedConfigSnapshot",
    "ResourceOwnershipClaim",
    "ResourceOwnershipConflictError",
    "ResourceSamples",
    "ResourceUsage",
    "RunResource",
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
    "TrainingBatchRecorder",
    "canonicalize_allocator_reset_scope",
    "claims_conflict",
    "configure_terminal_logging",
    "is_managed_callback",
    "validate_run_resource",
]
