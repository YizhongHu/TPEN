"""Public immutable records for the independent TPEN executor project."""

from .contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from .results import AllocationObservation, ManifestAttempt, OutcomeReceipt, RunManifest, StartReceipt

__version__ = "0.1.0"

__all__ = ["__version__"]
