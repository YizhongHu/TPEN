"""Training-loop namespace."""

from tpen.training.optim import make_optimizer
from tpen.training.qgt import (
    DampingPolicy,
    QGTOperator,
    SolveDiagnostics,
    solve_parameter_space,
    solve_sample_space,
)
from tpen.training.score_geometry import (
    SCORE_CONVENTION_VERSION,
    ScoreConventions,
    ScoreGeometry,
    build_energy_residual,
    build_score_geometry,
    build_score_geometry_from_rows,
    flatten_parameter_score_blocks,
    layout_convention_fingerprint,
    unflatten_to_layout,
)
from tpen.training.sr import (
    SR_STATE_VERSION,
    SRPolicy,
    SRTelemetry,
    StochasticReconfigurationUpdate,
)
from tpen.training.state import TrainerState, TrainingTiming
from tpen.training.statistics import (
    IdentityStatisticsReducer,
    StatisticsReducer,
    StatisticsSums,
    center_statistics,
)
from tpen.training.trainer import VMCTrainer
from tpen.training.update import (
    AutogradUpdateInput,
    LegacyAutogradUpdate,
    MinimalUpdateDiagnostics,
    ModelParameterBinding,
    ScoreUpdateInput,
    UpdateDiagnostics,
    UpdateMethodDescription,
    VMCStepData,
    VMCUpdateMethod,
    VMCUpdateResult,
    VMCUpdateState,
    deserialize_parameter_layout,
    serialize_parameter_layout,
)
from tpen.training.vmc import (
    VMCObjectiveResult,
    compute_vmc_objective,
    hamiltonian_term_metric_prefix,
    summarize_local_energy_terms,
    summarize_logabs,
)

__all__ = [
    "TrainerState",
    "TrainingTiming",
    "AutogradUpdateInput",
    "LegacyAutogradUpdate",
    "MinimalUpdateDiagnostics",
    "ModelParameterBinding",
    "ScoreUpdateInput",
    "UpdateDiagnostics",
    "UpdateMethodDescription",
    "VMCStepData",
    "VMCUpdateMethod",
    "VMCUpdateResult",
    "VMCUpdateState",
    "deserialize_parameter_layout",
    "serialize_parameter_layout",
    "SCORE_CONVENTION_VERSION",
    "SR_STATE_VERSION",
    "DampingPolicy",
    "QGTOperator",
    "SRPolicy",
    "SRTelemetry",
    "ScoreConventions",
    "ScoreGeometry",
    "SolveDiagnostics",
    "StochasticReconfigurationUpdate",
    "build_energy_residual",
    "build_score_geometry",
    "build_score_geometry_from_rows",
    "flatten_parameter_score_blocks",
    "layout_convention_fingerprint",
    "solve_parameter_space",
    "solve_sample_space",
    "unflatten_to_layout",
    "IdentityStatisticsReducer",
    "StatisticsReducer",
    "StatisticsSums",
    "center_statistics",
    "VMCObjectiveResult",
    "VMCTrainer",
    "compute_vmc_objective",
    "hamiltonian_term_metric_prefix",
    "make_optimizer",
    "summarize_local_energy_terms",
    "summarize_logabs",
]
