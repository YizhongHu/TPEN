"""Public immutable records for the independent TPEN executor project."""

from .contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from .results import AllocationObservation, ManifestAttempt, OutcomeReceipt, RunManifest, StartReceipt, decode_manifest, encode_manifest

__version__ = "0.1.0"

__all__ = [
    "__version__", "AllocationId", "AllocationObservation", "AttemptId",
    "BackendJobId", "ManifestAttempt", "OutcomeReceipt", "ProgramSpecRef",
    "ReadyGroup", "RunManifest", "StartReceipt", "TaskSpec",
    "decode_manifest", "encode_manifest",
]
