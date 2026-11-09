"""Evaluation configuration generators."""

from __future__ import annotations

from tpen.evaluation.generators.atom import HeliumRadialGridGenerator
from tpen.evaluation.generators.factor_response import ReequilibratedFactorGenerator
from tpen.evaluation.generators.hooke import (
    CuspGridGenerator,
    HookeOrbitalGenerator,
    StratifiedGeometryGenerator,
    TailGridGenerator,
)
from tpen.evaluation.generators.helium_atlas import (
    HeliumAngularShellGenerator,
    HeliumCenterOfMassEscapeGenerator,
    HeliumElectronElectronApproachGenerator,
    HeliumElectronNucleusApproachGenerator,
    HeliumOneElectronEscapeGenerator,
)
from tpen.evaluation.generators.mcmc import MCMCGenerator
from tpen.evaluation.generators.orbits import (
    ExchangeOrbitGenerator,
    PermutationOrbitGenerator,
    RotationOrbitGenerator,
)
from tpen.evaluation.generators.trajectory import (
    SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY,
    TRAJECTORY_METADATA_KEY,
    TrajectoryMCMCGenerator,
)
from tpen.evaluation.generators.readout_trajectory import ReadoutTrajectoryMCMCGenerator

__all__ = [
    "SAMPLER_TRAJECTORY_DIAGNOSTICS_KEY",
    "TRAJECTORY_METADATA_KEY",
    "CuspGridGenerator",
    "ExchangeOrbitGenerator",
    "HookeOrbitalGenerator",
    "HeliumRadialGridGenerator",
    "HeliumAngularShellGenerator",
    "HeliumCenterOfMassEscapeGenerator",
    "HeliumElectronElectronApproachGenerator",
    "HeliumElectronNucleusApproachGenerator",
    "HeliumOneElectronEscapeGenerator",
    "MCMCGenerator",
    "PermutationOrbitGenerator",
    "ReequilibratedFactorGenerator",
    "ReadoutTrajectoryMCMCGenerator",
    "RotationOrbitGenerator",
    "StratifiedGeometryGenerator",
    "TailGridGenerator",
    "TrajectoryMCMCGenerator",
]
