"""Opt-in native profiling with a code-owned training window.

The module is deliberately Torch-profiler free at import time.  Native
profiler objects are invocation-local and are never part of checkpoint state.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tpen.artifacts import RunContext
from tpen.events import Ended, Event as TypedEvent, Occurrence, Operation, Started, Subscription, started, ended
from tpen.observability_artifacts import (
    ObservabilityBundleWriter,
    ObservabilityManifest,
    ObservabilityProfileStatus,
    ObservabilityProfileSummary,
    RunArtifactSink,
)
from .base import (
    Callback,
    ManagedLifecycleError,
    ResourceOwnershipClaim,
    RunResource,
)
from .cadence import SubscriptionGroup

if TYPE_CHECKING:
    from tpen.sampling.events import SamplerBurnIn, SamplerProduction
    from tpen.training.events import CollectSamples, TrainingIteration


@dataclass(frozen=True)
class _EnabledEventTypes:
    """Runtime-resolved event classes owned by an enabled profiler."""

    training_iteration: type[Operation]
    collect_samples: type[Operation]
    sampler_burn_in: type[Operation]
    sampler_production: type[Operation]


def _load_enabled_event_types() -> _EnabledEventTypes:
    """Load the fixed event classes only on the enabled path."""

    from tpen.sampling.events import SamplerBurnIn, SamplerProduction
    from tpen.training.events import CollectSamples, TrainingIteration

    return _EnabledEventTypes(
        training_iteration=TrainingIteration,
        collect_samples=CollectSamples,
        sampler_burn_in=SamplerBurnIn,
        sampler_production=SamplerProduction,
    )


@dataclass(frozen=True)
class TrainingIterationWindow:
    """Immutable durable-step window for one native profile.

    Parameters
    ----------
    start_step : int
        First durable training step considered by the window.
    warmup_steps : int
        Number of selected iterations used for backend warm-up.
    active_steps : int
        Number of selected iterations captured after warm-up.
    """

    start_step: int
    warmup_steps: int
    active_steps: int

    def __post_init__(self) -> None:
        for name in ("start_step", "warmup_steps", "active_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0 or (name == "active_steps" and value == 0):
                raise ValueError(f"{name} has an invalid value {value!r}")
        if self.start_step + self.warmup_steps + self.active_steps > (1 << 63) - 1:
            raise ValueError("profile window overflows the durable-step range")

    @property
    def end_step(self) -> int:
        """Return the inclusive final selected durable step."""

        return self.start_step + self.warmup_steps + self.active_steps - 1

    def phase(self, step: int) -> str:
        """Classify a durable step as before, warmup, active, or after."""

        if step < self.start_step:
            return "before"
        offset = step - self.start_step
        if offset < self.warmup_steps:
            return "warmup"
        if offset < self.warmup_steps + self.active_steps:
            return "active"
        return "after"


@dataclass(frozen=True)
class TorchOperatorSummary:
    """Bounded conversion of a backend ``key_averages`` table."""

    max_rows: int = 64

    def __post_init__(self) -> None:
        if isinstance(self.max_rows, bool) or not isinstance(self.max_rows, int) or self.max_rows < 1:
            raise ValueError("max_rows must be a positive integer")

    def rows(self, backend: Any) -> tuple[Mapping[str, Any], ...]:
        """Return at most ``max_rows`` JSON-safe operator rows."""

        table = backend.key_averages() if hasattr(backend, "key_averages") else ()
        result: list[Mapping[str, Any]] = []
        for item in list(table)[: self.max_rows]:
            name = getattr(item, "key", getattr(item, "name", type(item).__name__))
            result.append({"name": str(name), "repr": str(item)[:512]})
        return tuple(result)


@dataclass(frozen=True)
class ChromeTraceExport:
    """Explicit, contained Chrome trace export request."""

    relative_path: str = "trace.json"

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if path.is_absolute() or ".." in path.parts or not self.relative_path.strip():
            raise ValueError("ChromeTraceExport.relative_path must stay inside the profile bundle")
        object.__setattr__(self, "relative_path", "/".join(path.parts))


class _DisabledScope(AbstractContextManager[None]):
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        return False


class _NativeSession(RunResource, AbstractContextManager["_NativeSession"]):
    """Invocation-local backend and range state."""

    def __init__(self, callback: "NativeProfiler", context: RunContext) -> None:
        self.callback = callback
        self.context = context
        self.backend: Any = None
        self.writer: ObservabilityBundleWriter | None = None
        self.started = False
        self.stopped = False
        self.stop_requested = False
        self.missed_window = False
        self._range_stack: list[tuple[type[Any], int]] = []
        self._range_contexts: list[Any] = []
        self._typed_counts: dict[type[Any], int] = {}
        self._steps: list[int] = []
        self._overshoot_steps = 0
        self._primary_error: BaseException | None = None

    def __getstate__(self) -> Any:
        raise TypeError("NativeProfiler live state cannot be serialized")

    def __reduce__(self) -> Any:
        raise TypeError("NativeProfiler live state cannot be serialized")

    def __reduce_ex__(self, protocol: Any) -> Any:
        del protocol
        raise TypeError("NativeProfiler live state cannot be serialized")

    def _make_backend(self) -> Any:
        if self.callback.backend_factory is not None:
            return self.callback.backend_factory()
        # This is the only import of torch.profiler in the module.
        from torch.profiler import ProfilerActivity, profile  # type: ignore

        return profile(activities=[ProfilerActivity.CPU], record_shapes=False)

    def _start(self) -> None:
        self.backend = self._make_backend()
        start = getattr(self.backend, "start", None)
        if callable(start):
            start()
        self.started = True

    def checkpoint(self) -> None:
        """Advance the backend cooperatively when it is active."""

        if self.started and not self.stopped and not self.stop_requested:
            step = getattr(self.backend, "step", None)
            if callable(step):
                step()

    def request_stop(self) -> None:
        """Request a nonblocking stop before the managed scope exits."""

        self.stop_requested = True

    def _stop(self) -> None:
        if not self.started or self.stopped:
            return
        stop = getattr(self.backend, "stop", None)
        if callable(stop):
            stop()
        self.stopped = True

    def __enter__(self) -> "_NativeSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del tb
        self._primary_error = exc
        try:
            self._stop()
            self.callback._persist(self)
        except BaseException:
            # Diagnostics never replace an application exception.  A cleanup
            # error is raised only when the application itself succeeded.
            if exc is None:
                raise
        finally:
            self.callback._finish(self)
        return False


class NativeProfiler(Callback):
    """Opt-in managed native profiler with fixed typed ranges."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        window: TrainingIterationWindow | Mapping[str, Any] | None = None,
        summary: TorchOperatorSummary | Mapping[str, Any] | None = None,
        sink: RunArtifactSink | Mapping[str, Any] | None = None,
        chrome_trace: ChromeTraceExport | Mapping[str, Any] | None = None,
        observer_id: str = "native-profiler",
        backend_factory: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(f"NativeProfiler received unsupported keys: {sorted(kwargs)}")
        self.enabled = bool(enabled)
        self.window = self._coerce_window(window)
        self.summary = self._coerce_summary(summary)
        self.sink = RunArtifactSink(relative_dir="native-profiling") if sink is None else self._coerce_sink(sink)
        self.chrome_trace = self._coerce_trace(chrome_trace)
        self.observer_id = str(observer_id)
        self.backend_factory = backend_factory
        self._active_session: _NativeSession | None = None
        self._unsafe = False
        self._observed = False
        self._event_types = _load_enabled_event_types() if self.enabled else None
        super().__init__(typed_groups=self._typed_groups(self._event_types) if self._event_types else ())

    @staticmethod
    def _typed_groups(event_types: _EnabledEventTypes) -> tuple[SubscriptionGroup, ...]:
        """Build the fixed event plan only for an enabled profiler."""

        return (SubscriptionGroup(selectors=(
            started(event_types.training_iteration), ended(event_types.training_iteration),
            started(event_types.collect_samples), ended(event_types.collect_samples),
            started(event_types.sampler_burn_in), ended(event_types.sampler_burn_in),
            started(event_types.sampler_production), ended(event_types.sampler_production),
        )),)

    @staticmethod
    def _coerce_window(value: Any) -> TrainingIterationWindow:
        if value is None:
            return TrainingIterationWindow(0, 0, 1)
        if isinstance(value, TrainingIterationWindow):
            return value
        if isinstance(value, Mapping):
            return TrainingIterationWindow(**dict(value))
        raise TypeError("window must be TrainingIterationWindow or a mapping")

    @staticmethod
    def _coerce_summary(value: Any) -> TorchOperatorSummary:
        if value is None:
            return TorchOperatorSummary()
        if isinstance(value, TorchOperatorSummary):
            return value
        if isinstance(value, Mapping):
            return TorchOperatorSummary(**dict(value))
        raise TypeError("summary must be TorchOperatorSummary or a mapping")

    @staticmethod
    def _coerce_sink(value: Any) -> RunArtifactSink:
        if isinstance(value, RunArtifactSink):
            return value
        if isinstance(value, Mapping):
            return RunArtifactSink(**dict(value))
        raise TypeError("sink must be RunArtifactSink or a mapping")

    @staticmethod
    def _coerce_trace(value: Any) -> ChromeTraceExport | None:
        if value is None or isinstance(value, ChromeTraceExport):
            return value
        if isinstance(value, Mapping):
            return ChromeTraceExport(**dict(value))
        raise TypeError("chrome_trace must be ChromeTraceExport or a mapping")

    def __getstate__(self) -> Any:
        raise TypeError("NativeProfiler is invocation-local and cannot be serialized")

    def __reduce__(self) -> Any:
        raise TypeError("NativeProfiler is invocation-local and cannot be serialized")

    def __reduce_ex__(self, protocol: Any) -> Any:
        del protocol
        raise TypeError("NativeProfiler is invocation-local and cannot be serialized")

    def ownership_claims(self, *, artifact_manager: Any = None, invocation_id: str | None = None, **_: Any) -> list[ResourceOwnershipClaim]:
        if not self.enabled:
            return []
        claims = [ResourceOwnershipClaim(kind="torch_profiler", scope="process")]
        if artifact_manager is not None and invocation_id is not None:
            claims.append(self.sink.ownership_claim(artifact_manager, invocation_id, self.observer_id))
        return claims

    def run_scope(self, context: RunContext) -> AbstractContextManager[RunResource | None]:
        if not self.enabled:
            return _DisabledScope()
        if self._unsafe or self._active_session is not None:
            raise ManagedLifecycleError("NativeProfiler cannot be concurrently re-entered")
        session = _NativeSession(self, context)
        self._active_session = session
        return session

    def _finish(self, session: _NativeSession) -> None:
        if self._active_session is session:
            self._active_session = None
        if not session.stopped and session.started:
            self._unsafe = True

    def handle_occurrence_impl(self, occurrence: Occurrence[TypedEvent], context: RunContext) -> None:
        session = self._active_session
        if session is None:
            return
        event_types = self._event_types
        if event_types is None:
            return
        event = occurrence.event
        if isinstance(event, Started) and isinstance(event.operation, event_types.training_iteration):
            step = event.operation.step
            session._steps.append(step)
            session._typed_counts[event_types.training_iteration] = session._typed_counts.get(event_types.training_iteration, 0) + 1
            phase = self.window.phase(step)
            if phase == "warmup" or phase == "active":
                if not session.started:
                    session._start()
                session._range_stack.append((type(event.operation), occurrence.count))
                recorder = getattr(session.backend, "record_function", None)
                if callable(recorder):
                    range_context = recorder(type(event.operation).__name__)
                    range_context.__enter__()
                    session._range_contexts.append(range_context)
            elif phase == "after":
                if session.started:
                    session._overshoot_steps += 1
                session.request_stop()
            else:
                session.missed_window = True
            return
        if isinstance(event, Ended) and isinstance(event.operation, event_types.training_iteration):
            if session.started and not session.stopped:
                session.checkpoint()
                if self.window.phase(event.operation.step) == "active" and event.operation.step == self.window.end_step:
                    session.request_stop()
            if session._range_stack:
                session._range_stack.pop()
            if session._range_contexts:
                session._range_contexts.pop().__exit__(None, None, None)
            return
        if isinstance(event, (Started, Ended)) and isinstance(event.operation, (
            event_types.collect_samples, event_types.sampler_burn_in, event_types.sampler_production
        )):
            op_type = type(event.operation)
            if isinstance(event, Started):
                session._typed_counts[op_type] = session._typed_counts.get(op_type, 0) + 1
            if isinstance(event, Started) and session.started:
                session._range_stack.append((type(event.operation), occurrence.count))
                recorder = getattr(session.backend, "record_function", None)
                if callable(recorder):
                    range_context = recorder(op_type.__name__)
                    range_context.__enter__()
                    session._range_contexts.append(range_context)
            elif isinstance(event, Ended) and session._range_stack:
                session._range_stack.pop()
                if session._range_contexts:
                    session._range_contexts.pop().__exit__(None, None, None)

    def _persist(self, session: _NativeSession) -> None:
        # Persistence is best effort, but uses the existing bounded profile schema.
        if session.context is None or not hasattr(session.context, "artifact_manager"):
            return
        now = time.monotonic_ns()
        status = "not_started" if session.missed_window and not session.started else ("closed" if session.stopped else "open")
        completeness = "none" if status == "not_started" else ("complete" if session.stopped else "partial")
        rows = self.summary.rows(session.backend) if session.started else ()
        window_id = f"step-{self.window.start_step}-{self.window.end_step}"
        writer = ObservabilityBundleWriter(session.context.artifact_manager, session.context.invocation_id, sink=self.sink)
        manifest_base = dict(invocation_id=session.context.invocation_id, run_id=session.context.metadata.run_id,
                             producer={"name": "tpen.callback.native_profiler", "version": "1"},
                             configuration_digest=hashlib.sha256(b"native-profiler").hexdigest(),
                             inventory_id=f"inventory-{session.context.invocation_id}", descriptor_set_id=f"descriptors-{session.context.invocation_id}",
                             observers=({"observer_id": self.observer_id},), limits={"max_rows": self.summary.max_rows})
        writer.write_open_manifest(ObservabilityManifest(lifecycle_state="open", data_completeness="unknown", **manifest_base))
        writer.write_profile_summary(ObservabilityProfileSummary(session.context.invocation_id, self.observer_id, window_id,
            "TorchOperatorSummary", self.summary.max_rows, len(rows), tuple(rows)))
        if self.chrome_trace is not None and session.started:
            trace_path = writer.bundle_dir / self.chrome_trace.relative_path
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_bundle = writer.bundle_dir.resolve()
            resolved_trace = trace_path.resolve()
            resolved_trace.relative_to(resolved_bundle)
            export = getattr(session.backend, "export_chrome_trace", None)
            if callable(export):
                export(str(trace_path))
        counts = {op.__name__: count for op, count in session._typed_counts.items()}
        writer.write_profile_status(ObservabilityProfileStatus(session.context.invocation_id, self.observer_id, window_id,
            "operation", ("CPU",), ("CPU",) if session.started else (), counts, "process-monotonic", now,
            status, completeness, "diagnostic", "TorchOperatorSummary", "complete" if session.stopped else "partial",
            durable_steps=tuple(session._steps), max_rows=self.summary.max_rows,
            export_mode="chrome_trace" if self.chrome_trace else "summary",
            activity_diagnostics=(f"overshoot_steps={session._overshoot_steps}",)))
        writer.finalize_manifest(ObservabilityManifest(lifecycle_state="closed" if session.stopped else "unsafe",
            data_completeness=completeness, **manifest_base))


__all__ = ["ChromeTraceExport", "NativeProfiler", "TorchOperatorSummary", "TrainingIterationWindow"]
