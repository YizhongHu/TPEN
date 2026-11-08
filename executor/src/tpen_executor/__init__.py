"""Public immutable records for the independent TPEN executor project."""

from importlib import import_module

from .contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from .allocations import ConfiguredAllocation
from .delivery import PreparedInputBundle
from .environments import PreparedPythonEnvironment, SourceScript
from .results import AcceptedSegment, AllocationObservation, ManifestAttempt, OutcomeReceipt, OUTCOME_STATUSES, RunManifest, SciencePlacementReceipt, StartReceipt, decode_manifest, encode_manifest
from .runners import CoreRunEvidence, EvidenceUnavailable, TPENCommand, TPENEvidence
from .recovery import CoreProfileReference, ProfileCheck, ProfileUnavailable
from .storage import AttemptFence, DirectoryStore, FenceRejected
from .tasks import ConfiguredRun
from .lifecycle import AttachedAllocation
from .workers import ProcessOutcome, SubprocessWorker
from .placement import AllocationSlot, BindingError, ChildPlacementObservation, PlacementRefusal, WorkerBinding, binding_from_allocation, bindings_from_allocation, parse_cpu_set, parse_proc_start_ticks, read_child_placement, slots_from_allocation

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "AllocationProgram": (".programs", "AllocationProgram"),
    "MaterializedProgram": (".programs", "MaterializedProgram"),
    "StagedConfig": (".configuration", "StagedConfig"),
    "StagedHydra": (".configuration", "StagedHydra"),
    "WorkerProgram": (".programs", "WorkerProgram"),
    "LocalDispatcher": (".dispatch", "LocalDispatcher"),
    "ParslDispatcher": (".dispatch", "ParslDispatcher"),
    "ParslUnavailableError": (".dispatch", "ParslUnavailableError"),
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
    "__version__", "AcceptedSegment", "AllocationId", "AllocationObservation", "AllocationProgram",
    "AttemptId", "AttachedAllocation", "AttemptFence", "BackendJobId", "ConfiguredAllocation", "ConfiguredRun", "CoreRunEvidence", "DirectoryStore", "EvidenceUnavailable", "FenceRejected", "ManifestAttempt", "MaterializedProgram", "OUTCOME_STATUSES", "OutcomeReceipt", "ProcessOutcome",
    "ProgramSpecRef", "ProfileCheck", "ProfileUnavailable", "ReadyGroup", "RunManifest", "SciencePlacementReceipt", "StagedConfig", "StagedHydra", "LocalDispatcher", "ParslDispatcher", "ParslUnavailableError",
    "PreparedInputBundle", "PreparedPythonEnvironment", "SourceScript", "StartReceipt",
    "TaskSpec", "TPENCommand", "TPENEvidence", "SubprocessWorker", "WorkerProgram", "CoreProfileReference",
    "AllocationSlot", "BindingError", "ChildPlacementObservation", "PlacementRefusal", "WorkerBinding", "binding_from_allocation", "bindings_from_allocation", "parse_cpu_set", "parse_proc_start_ticks", "read_child_placement", "slots_from_allocation",
    "decode_manifest", "encode_manifest",
]
