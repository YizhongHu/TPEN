"""Runtime timing callbacks."""

from __future__ import annotations

from .evaluation_component_timing import EvaluationComponentTiming
from .evaluation_timing import EvaluationTiming
from .run_timing import RunTiming
from .train_phase_timing import TrainPhaseTiming
from .train_step_timing import TrainStepTiming
from .sampling_timing import SamplingScopeTiming


def __getattr__(name: str) -> object:
    """Load timing callbacks that resolve a domain state class only on demand.

    A `tpen.callback.StatefulCallback` declares its ``state_type`` as a ClassVar,
    which forces the state class to be imported at class-creation time. For the
    evaluation domain that runs `tpen.evaluation`'s ``__init__``, which pulls in
    torch, so `DiagnosticTiming` has to be loaded lazily to keep
    ``import tpen.callback.timing`` torch-free. The callbacks that only need
    event TYPES resolve those inside their ``__init__`` and stay eager.
    """

    if name == "DiagnosticTiming":
        from .diagnostic_timing import DiagnosticTiming

        return DiagnosticTiming
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DiagnosticTiming",
    "EvaluationComponentTiming",
    "EvaluationTiming",
    "RunTiming",
    "TrainPhaseTiming",
    "TrainStepTiming",
    "SamplingScopeTiming",
]
