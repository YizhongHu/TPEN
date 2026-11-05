"""Public immutable records for the independent TPEN executor project."""

from importlib import import_module

from .contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from .allocations import ConfiguredAllocation
from .delivery import PreparedInputBundle
from .environments import PreparedPythonEnvironment, SourceScript
from .results import AllocationObservation, ManifestAttempt, OutcomeReceipt, OUTCOME_STATUSES, RunManifest, StartReceipt, decode_manifest, encode_manifest
from .runners import CoreRunEvidence, EvidenceUnavailable, TPENCommand, TPENEvidence
from .storage import AttemptFence, DirectoryStore, FenceRejected
from .tasks import ConfiguredRun
from .lifecycle import AttachedAllocation
from .workers import ProcessOutcome, SubprocessWorker

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "AllocationProgram": (".programs", "AllocationProgram"),
    "MaterializedProgram": (".programs", "MaterializedProgram"),
    "StagedConfig": (".configuration", "StagedConfig"),
    "StagedHydra": (".configuration", "StagedHydra"),
    "WorkerProgram": (".programs", "WorkerProgram"),
}


def __getattr__(name: str):
    """Load Hydra-dependent exports only when a caller requests them."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "__version__", "AllocationId", "AllocationObservation", "AllocationProgram",
    "AttemptId", "AttachedAllocation", "AttemptFence", "BackendJobId", "ConfiguredAllocation", "ConfiguredRun", "CoreRunEvidence", "DirectoryStore", "EvidenceUnavailable", "FenceRejected", "ManifestAttempt", "MaterializedProgram", "OUTCOME_STATUSES", "OutcomeReceipt", "ProcessOutcome",
    "ProgramSpecRef", "ReadyGroup", "RunManifest", "StagedConfig", "StagedHydra",
    "PreparedInputBundle", "PreparedPythonEnvironment", "SourceScript", "StartReceipt",
    "TaskSpec", "TPENCommand", "TPENEvidence", "SubprocessWorker", "WorkerProgram",
    "decode_manifest", "encode_manifest",
]
