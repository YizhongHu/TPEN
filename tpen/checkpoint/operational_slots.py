"""Owner-defined registry for operational callback and logger destinations.

The manifest is the source of truth for both config-identity scrubbing and
routed artifact admission.  Entries are keyed by the exact dotted target or
runtime type name; no inheritance or path-name inference is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationalSlot:
    """One exact operational target and its admitted routing behavior."""

    kind: str | None = None
    fields: tuple[str, ...] = ()
    routed: bool = False
    sink_free: bool = False
    non_operational_reasons: tuple[tuple[str, str], ...] = ()


def _callback(*fields: str, routed: bool = False) -> OperationalSlot:
    return OperationalSlot(kind="callback", fields=fields, routed=routed)


def _logger(*fields: str, routed: bool = False) -> OperationalSlot:
    return OperationalSlot(kind="logger", fields=fields, routed=routed)


def _sink_free() -> OperationalSlot:
    return OperationalSlot(routed=True, sink_free=True)


# Keep every package alias, defining-module target, and compatibility prefix
# explicit.  The identity census tests this complete manifest against the
# public callback/logger constructors, while routed=True marks the concrete
# runtime types that ArtifactDestination may rebind.
OPERATIONAL_SLOT_MANIFEST: dict[str, OperationalSlot] = {
    "tpen.callback.ConfigSnapshot": _callback("output_path"),
    "tpen.callback.ResolvedConfigSnapshot": _callback("output_path"),
    "tpen.callback.Metadata": _callback("output_path"),
    "tpen.callback.Status": _callback("output_path"),
    "tpen.callback.Checkpoint": _callback("output_dir"),
    "tpen.callback.InitialCheckpoint": _callback("output_dir"),
    "tpen.callback.ArtifactIndex": _callback("path"),
    "tpen.callback.FailureLog": _callback("path"),
    "tpen.callback.ParameterActivity": _callback("output_path"),
    "tpen.callback.RawTrainingBatchRecorder": _callback("path"),
    "tpen.callback.TrainingBatchRecorder": _callback("path"),
    "tpen.callback.RuntimeEquivariance": _callback("artifact_dir"),
    "tpen.callback.checkpoint.Checkpoint": _callback("output_dir", routed=True),
    "tpen.callback.equivariance.RuntimeEquivariance": _callback(
        "artifact_dir", routed=True
    ),
    "tpen.callback.evaluation.ArtifactIndex": _callback("path", routed=True),
    "tpen.callback.evaluation.FailureLog": _callback("path", routed=True),
    "tpen.callback.initial_checkpoint.InitialCheckpoint": _callback(
        "output_dir", routed=True
    ),
    "tpen.callback.metadata.Metadata": _callback("output_path", routed=True),
    "tpen.callback.parameter_activity.ParameterActivity": _callback(
        "output_path", routed=True
    ),
    "tpen.callback.raw_training_batch_recorder.RawTrainingBatchRecorder": _callback(
        "path", routed=True
    ),
    "tpen.callback.raw_training_batch_recorder.TrainingBatchRecorder": _callback(
        "path"
    ),
    "tpen.callback.snapshot.ConfigSnapshot": _callback("output_path", routed=True),
    "tpen.callback.snapshot.ResolvedConfigSnapshot": _callback(
        "output_path", routed=True
    ),
    "tpen.callback.status.Status": _callback("output_path", routed=True),
    "spenn.callback.ConfigSnapshot": _callback("output_path"),
    "spenn.callback.ResolvedConfigSnapshot": _callback("output_path"),
    "spenn.callback.Metadata": _callback("output_path"),
    "spenn.callback.Status": _callback("output_path"),
    "spenn.callback.Checkpoint": _callback("output_dir"),
    "spenn.callback.InitialCheckpoint": _callback("output_dir"),
    "spenn.callback.ArtifactIndex": _callback("path"),
    "spenn.callback.FailureLog": _callback("path"),
    "spenn.callback.ParameterActivity": _callback("output_path"),
    "spenn.callback.RawTrainingBatchRecorder": _callback("path"),
    "spenn.callback.TrainingBatchRecorder": _callback("path"),
    "spenn.callback.RuntimeEquivariance": _callback("artifact_dir"),
    "tpen.logging.CSV": _logger("path"),
    "tpen.logging.JSONL": _logger("path"),
    "tpen.logging.WandB": _logger("dir"),
    "tpen.logging.csv.CSV": _logger("path", routed=True),
    "tpen.logging.jsonl.JSONL": _logger("path", routed=True),
    "tpen.logging.wandb.WandB": _logger("dir"),
    "spenn.logging.CSV": _logger("path"),
    "spenn.logging.JSONL": _logger("path"),
    "spenn.logging.WandB": _logger("dir"),
    "tpen.callback.amplitude_risk.AmplitudeRiskFraction": _sink_free(),
    "tpen.callback.electron_nucleus_distance.ElectronNucleusDistanceQuantiles": _sink_free(),
    "tpen.callback.factor_scalars.FactorScalars": _sink_free(),
    "tpen.callback.module_norms.ModuleNorms": _sink_free(),
    "tpen.callback.optimizer_state.OptimizerState": _sink_free(),
    "tpen.callback.resource_usage.ResourceUsage": _sink_free(),
    "tpen.callback.health.data_integrity.DataIntegrity": _sink_free(),
    "tpen.callback.health.gradient_stats.GradientStats": _sink_free(),
    "tpen.callback.health.sampler_health.SamplerHealth": _sink_free(),
    "tpen.callback.timing.diagnostic_timing.DiagnosticTiming": _sink_free(),
    "tpen.callback.timing.evaluation_component_timing.EvaluationComponentTiming": _sink_free(),
    "tpen.callback.timing.evaluation_timing.EvaluationTiming": _sink_free(),
    "tpen.callback.timing.run_timing.RunTiming": _sink_free(),
    "tpen.callback.timing.train_phase_timing.TrainPhaseTiming": _sink_free(),
    "tpen.callback.timing.train_step_timing.TrainStepTiming": _sink_free(),
    "tpen.callback.planned_yield.PlannedYield": _sink_free(),
}


def config_path_fields(kind: str) -> dict[str, set[str]]:
    """Return a mutable identity table for callback or logger targets."""

    return {
        target: set(slot.fields)
        for target, slot in OPERATIONAL_SLOT_MANIFEST.items()
        if slot.kind == kind and slot.fields
    }


def non_operational_path_fields() -> dict[str, dict[str, str]]:
    """Return explicitly named non-operational path fields, if any."""

    return {
        target: dict(slot.non_operational_reasons)
        for target, slot in OPERATIONAL_SLOT_MANIFEST.items()
        if slot.non_operational_reasons
    }


def routed_sink_fields() -> dict[str, tuple[str, ...]]:
    """Return exact runtime writer types and their rebindable fields."""

    return {
        target: slot.fields
        for target, slot in OPERATIONAL_SLOT_MANIFEST.items()
        if slot.routed and slot.fields
    }


def sink_free_types() -> frozenset[str]:
    """Return exact runtime types that are admitted without routing fields."""

    return frozenset(
        target
        for target, slot in OPERATIONAL_SLOT_MANIFEST.items()
        if slot.routed and slot.sink_free
    )


__all__ = [
    "OPERATIONAL_SLOT_MANIFEST",
    "OperationalSlot",
    "config_path_fields",
    "non_operational_path_fields",
    "routed_sink_fields",
    "sink_free_types",
]
