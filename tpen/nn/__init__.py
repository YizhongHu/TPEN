"""Neural-network component namespace for TPEN."""

from tpen.nn.activation import (
    ChannelActivationAxes,
    ChannelPreservingMLPActivation,
    GaussianActivation,
    OrderMLPLayout,
    OrderMLPSpec,
)
from tpen.nn.basis import (
    ElectronBasis,
    ElectronBasisFeatures,
    HookeHermiteBasis,
    HookeOrbitalBasis,
    RawCoordinateBasis,
)
from tpen.nn.context import TPENForwardContext
from tpen.nn.coordinate_envelopes import (
    CoordinateEnvelope,
    GaussianCoordinateEnvelope,
    GaussianDecayGate,
)
from tpen.nn.cusp import (
    ElectronElectronCusp,
    ElectronNucleusCuspEvaluation,
    ElectronNucleusCusp,
    ElectronNucleusCuspLaw,
    LinearElectronNucleusCuspLaw,
    CurvatureElectronNucleusCuspLaw,
)
from tpen.nn.embedding import Embedding
from tpen.nn.envelope import (
    AdditiveEnvelope,
    Envelope,
    GaussianConfinement,
    HookeGaussianConfinement,
)
from tpen.nn.equivariant_mixing import EquivariantMixing
from tpen.nn.composite_mixing import CompositeMixing
from tpen.nn.linear_equivariant_mixing import LinearEquivariantMixing
from tpen.nn.mixing_kernel import execute_binary, execute_unary
from tpen.nn.factor import AdditiveCusp, LogAmplitudeFactor
from tpen.nn.forward import (
    CoordinateGradientRequest,
    CoordinateGradientProvider,
    MaterializedParameterScoreRequest,
    ParameterScoreRequest,
    ParameterScoreProvider,
    WavefunctionForwardRequest,
    WavefunctionRequestProvider,
)
from tpen.nn.initialization import SeededLinear, TorchInitializer
from tpen.nn.interaction_config import (
    InteractionMode,
    ProducerFamily,
    ResolvedInteractionConfig,
    normalize_interaction_mode,
    normalize_producer_order,
)
from tpen.nn.mlp import MLP
from tpen.nn.normalization import RMSNorm
from tpen.nn.path_aggregation import PathAggregation
from tpen.nn.tpen_layer import TPENLayer
from tpen.nn.tpen_wave_function import TPENWaveFunction
from tpen.nn.tpen_stack import TPENStack
from tpen.nn.update import ReplaceUpdater, ResidualUpdater, Updater

__all__ = [
    "AdditiveCusp",
    "AdditiveEnvelope",
    "ChannelActivationAxes",
    "ChannelPreservingMLPActivation",
    "CoordinateEnvelope",
    "CoordinateGradientRequest",
    "CoordinateGradientProvider",
    "ElectronBasis",
    "ElectronBasisFeatures",
    "ElectronElectronCusp",
    "ElectronNucleusCusp",
    "ElectronNucleusCuspEvaluation",
    "ElectronNucleusCuspLaw",
    "Embedding",
    "Envelope",
    "EquivariantMixing",
    "CompositeMixing",
    "LinearEquivariantMixing",
    "execute_binary",
    "execute_unary",
    "GaussianActivation",
    "GaussianCoordinateEnvelope",
    "GaussianDecayGate",
    "GaussianConfinement",
    "HookeGaussianConfinement",
    "HookeHermiteBasis",
    "HookeOrbitalBasis",
    "LinearElectronNucleusCuspLaw",
    "LogAmplitudeFactor",
    "MLP",
    "MaterializedParameterScoreRequest",
    "OrderMLPLayout",
    "OrderMLPSpec",
    "ParameterScoreRequest",
    "ParameterScoreProvider",
    "PathAggregation",
    "RawCoordinateBasis",
    "RMSNorm",
    "ReplaceUpdater",
    "ResidualUpdater",
    "SeededLinear",
    "TPENForwardContext",
    "TPENLayer",
    "TPENWaveFunction",
    "TPENStack",
    "TorchInitializer",
    "CurvatureElectronNucleusCuspLaw",
    "Updater",
    "WavefunctionForwardRequest",
    "WavefunctionRequestProvider",
    "ProducerFamily",
    "InteractionMode",
    "ResolvedInteractionConfig",
    "normalize_interaction_mode",
    "normalize_producer_order",
]
