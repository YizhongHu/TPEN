"""Public immutable records for the independent TPEN executor project."""

from .contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from .configuration import StagedConfig, StagedHydra
from .programs import AllocationProgram, WorkerProgram
from .results import AllocationObservation, ManifestAttempt, OutcomeReceipt, RunManifest, StartReceipt, decode_manifest, encode_manifest

__version__ = "0.1.0"

__all__ = [
    "__version__", "AllocationId", "AllocationObservation", "AllocationProgram",
    "AttemptId", "BackendJobId", "ManifestAttempt", "OutcomeReceipt",
    "ProgramSpecRef", "ReadyGroup", "RunManifest", "StagedConfig", "StagedHydra",
    "StartReceipt", "TaskSpec", "WorkerProgram",
    "decode_manifest", "encode_manifest",
]
