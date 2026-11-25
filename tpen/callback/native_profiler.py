"""Opt-in native profiling with a code-owned training window.

The module is deliberately Torch-profiler free at import time. A live native
session, its writer, and its range contexts are invocation-local and refuse
serialization.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, TYPE_CHECKING, TypeAlias

from tpen.artifacts import ArtifactManager, RunContext
from tpen.events import Ended, Event as TypedEvent, Occurrence, Operation, Started
from tpen.observability_artifacts import (
    ObservabilityBundleWriter,
    ObservabilityArtifactRef,
    ObservabilityManifest,
    ObservabilityProfileStatus,
    ObservabilityProfileSummary,
    RunArtifactSink,
)

from .base import Callback, ManagedLifecycleError, ResourceOwnershipClaim, RunResource
from .cadence import SubscriptionGroup

if TYPE_CHECKING:
    from tpen.sampling.events import SamplerBurnIn, SamplerProduction
    from tpen.training.events import CollectSamples, TrainingIteration


class _OperatorRow(Protocol):
    """Minimum native operator-row shape needed by the bounded exporter."""

    key: object


class _ProfilerBackend(Protocol):
    """Required backend API; missing methods are never silently ignored."""

    def start(self) -> None: ...
    def step(self) -> None: ...
    def stop(self) -> None: ...
    def key_averages(self) -> Sequence[_OperatorRow]: ...
    def record_function(self, label: str) -> AbstractContextManager[object]: ...
    def export_chrome_trace(self, path: str) -> None: ...


_BackendFactory: TypeAlias = Callable[[], _ProfilerBackend]


class _RawTorchProfiler(Protocol):
    """Typed surface of the object returned by ``torch.profiler.profile``."""

    def start(self) -> None: ...
    def step(self) -> None: ...
    def stop(self) -> None: ...
    def key_averages(self) -> Sequence[_OperatorRow]: ...
    def export_chrome_trace(self, path: str) -> None: ...


class _TorchProfilerBackend:
    """Adapter for the real profiler and module-level record-function API."""

    def __init__(
        self,
        raw: _RawTorchProfiler,
        record_function: Callable[[str], AbstractContextManager[object]],
    ) -> None:
        self._raw = raw
        self._record_function = record_function

    def start(self) -> None:
        self._raw.start()

    def step(self) -> None:
        self._raw.step()

    def stop(self) -> None:
        self._raw.stop()

    def key_averages(self) -> Sequence[_OperatorRow]:
        return self._raw.key_averages()

    def record_function(self, label: str) -> AbstractContextManager[object]:
        return self._record_function(label)

    def export_chrome_trace(self, path: str) -> None:
        self._raw.export_chrome_trace(path)


@dataclass(frozen=True)
class _EnabledEventTypes:
    """Runtime-resolved event classes owned by an enabled profiler."""

    training_iteration: type[Operation]
    collect_samples: type[Operation]
    sampler_burn_in: type[Operation]
    sampler_production: type[Operation]


def _load_enabled_event_types() -> _EnabledEventTypes:
    """Load fixed event classes only on the enabled path."""

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
    """Immutable durable-step window for one native profile."""

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
        if isinstance(self.max_rows, bool) or not isinstance(self.max_rows, int):
            raise TypeError("max_rows must be an integer")
        if self.max_rows < 1:
            raise ValueError("max_rows must be a positive integer")

    def collect(
        self, backend: _ProfilerBackend
    ) -> tuple[tuple[Mapping[str, object], ...], int, bool]:
        """Return bounded rows, the true source count, and truncation state."""

        table = list(backend.key_averages())
        rows = tuple(
            {"name": str(item.key), "repr": str(item)[:512]}
            for item in table[: self.max_rows]
        )
        return rows, len(table), len(table) > len(rows)

    def rows(self, backend: _ProfilerBackend) -> tuple[Mapping[str, object], ...]:
        """Return at most ``max_rows`` JSON-safe operator rows."""

        return self.collect(backend)[0]


@dataclass(frozen=True)
class ChromeTraceExport:
    """Opt-in code-owned Chrome trace export with no path knobs."""

    def __post_init__(self) -> None:
        """Reject accidental fields when constructed through a mapping."""

        if self.__dict__:
            raise TypeError("ChromeTraceExport does not accept configuration fields")


class _DisabledScope(AbstractContextManager[None]):
    """No-op scope for the disabled path."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        del exc_type, exc, tb
        return False


@dataclass
class _RangeFrame:
    """One identity-paired native range context."""

    operation_type: type[Operation]
    occurrence_count: int
    context: AbstractContextManager[object]


class _NativeSession(RunResource, AbstractContextManager["_NativeSession"]):
    """Invocation-local backend, schedule, writer, and range state."""

    def __init__(self, callback: "NativeProfiler", context: RunContext) -> None:
        self.callback = callback
        self.context = context
        self.backend: _ProfilerBackend | None = None
        self.writer: ObservabilityBundleWriter | None = None
        self.activation_attempted = False
        self.started = False
        self.stopped = False
        self.stop_attempted = False
        self.stop_requested = False
        self.window_started = False
        self.window_completed = False
        self.window_start_not_observed = False
        self.current_step: int | None = None
        self._range_stack: list[_RangeFrame] = []
        self._typed_counts: dict[type[Operation], int] = {}
        self._steps: list[int] = []
        self._overshoot_steps = 0
        self._diagnostics: list[str] = []
        self._start_bracket_ns: tuple[int, int] | None = None
        self._stop_bracket_ns: tuple[int, int] | None = None
        self._activation_ns = 0
        self._closed = False
        self._activation_failed = False
        self.trace_path: Path | None = None
        self.trace_ref: ObservabilityArtifactRef | None = None

    def __getstate__(self) -> object:
        raise TypeError("NativeProfiler live state cannot be serialized")

    def __reduce__(self) -> object:
        raise TypeError("NativeProfiler live state cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> object:
        del protocol
        raise TypeError("NativeProfiler live state cannot be serialized")

    @property
    def unsafe(self) -> bool:
        """Whether activation or quiescence could not be confirmed."""

        return self._activation_failed or (self.activation_attempted and not self.stopped)

    def _diagnostic(self, label: str, exc: BaseException | None = None) -> None:
        message = label if exc is None else f"{label}:{type(exc).__name__}:{exc}"
        if message not in self._diagnostics:
            self._diagnostics.append(message)

    def _make_backend(self) -> _ProfilerBackend:
        if self.callback.backend_factory is not None:
            return self.callback.backend_factory()
        # Imports remain inside enabled activation. The adapter deliberately
        # binds module-level record_function, not an optional backend method.
        from torch.profiler import ProfilerActivity, profile, record_function, schedule

        window = self.callback.window
        if window is None:
            raise ValueError("enabled NativeProfiler requires an explicit window")
        raw = profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=False,
            schedule=schedule(
                wait=0,
                warmup=window.warmup_steps,
                active=window.active_steps,
                repeat=1,
            ),
        )
        return _TorchProfilerBackend(raw, record_function)

    def _open_writer(self) -> None:
        if self.writer is not None:
            return
        self.writer = self.callback._new_writer(self.context)
        if self.callback.trace is not None:
            self.trace_path = self.writer.reserve_external_artifact(
                self.callback._trace_relative_path(), kind="trace"
            )
        self.writer.write_open_manifest(self.callback._open_manifest(self.context))

    def _start(self) -> None:
        """Attempt activation once; partial-start failures remain diagnostic."""

        if self.activation_attempted or self.stop_requested:
            return
        self.activation_attempted = True
        self._activation_ns = time.monotonic_ns()
        try:
            self._open_writer()
            self.backend = self._make_backend()
            started_before = time.monotonic_ns()
            self.backend.start()
            started_after = time.monotonic_ns()
            self._start_bracket_ns = (started_before, started_after)
            self.started = True
        except BaseException as exc:
            self._activation_failed = True
            self._diagnostic(
                "trace_reservation_failed" if self.trace_path is None and self.callback.trace is not None
                else "backend_start_failed",
                exc,
            )

    def checkpoint(self) -> None:
        """Keep cooperative lifecycle checkpoints separate from backend steps."""

    def request_stop(self) -> None:
        """Request a nonblocking stop and prevent later activation/ranges."""

        self.stop_requested = True

    def _unwind_ranges(self) -> None:
        while self._range_stack:
            frame = self._range_stack.pop()
            try:
                frame.context.__exit__(None, None, None)
            except BaseException as exc:
                self._diagnostic("range_exit_failed", exc)

    def _stop(self) -> None:
        if self.stop_attempted:
            return
        if self.backend is None or (not self.started and not self.activation_attempted):
            return
        self.stop_attempted = True
        self._unwind_ranges()
        started_before = time.monotonic_ns()
        try:
            self.backend.stop()
            self.stopped = True
        except BaseException as exc:
            self._diagnostic("backend_stop_failed", exc)
        finally:
            self._stop_bracket_ns = (started_before, time.monotonic_ns())

    def _close_range(self, operation: Operation, count: int) -> None:
        if not self._range_stack:
            return
        frame = self._range_stack[-1]
        if frame.operation_type is not type(operation) or frame.occurrence_count != count:
            self._diagnostic(f"range_lifo_mismatch:{type(operation).__name__}:{count}")
            return
        self._range_stack.pop()
        try:
            frame.context.__exit__(None, None, None)
        except BaseException as exc:
            self._diagnostic("range_exit_failed", exc)

    def _open_range(self, operation: Operation, count: int) -> None:
        if self.backend is None or not self.started or self.stopped or self.stop_requested:
            return
        try:
            range_context = self.backend.record_function(type(operation).__name__)
            range_context.__enter__()
            self._range_stack.append(_RangeFrame(type(operation), count, range_context))
        except BaseException as exc:
            self._diagnostic("range_start_failed", exc)

    def __enter__(self) -> "_NativeSession":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        del exc_type, tb
        if self._closed:
            return False
        self._closed = True
        self._stop()
        self.callback._persist(self, application_error=exc)
        self.callback._finish(self)
        return False

    def _export_trace(self) -> None:
        """Synchronously export and register the optional native trace."""

        if self.callback.trace is None or self.backend is None or not self.started:
            return
        writer = self.writer
        trace_path = self.trace_path
        if writer is None or trace_path is None:
            self._diagnostic("trace_export_failed:reservation_unavailable")
            return
        try:
            trace_path = writer.prepare_external_artifact(
                self.callback._trace_relative_path()
            )
            # This direct typed call is intentional: missing backend capability
            # is an explicit diagnostic failure, never a silent success.
            self.backend.export_chrome_trace(str(trace_path))
            self.trace_ref = writer.register_external_artifact(
                self.callback._trace_relative_path(), kind="trace"
            )
        except BaseException as exc:
            self.trace_ref = None
            self._diagnostic("trace_export_failed", exc)


class NativeProfiler(Callback):
    """Opt-in managed native profiler with fixed typed ranges."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        window: TrainingIterationWindow | Mapping[str, object] | None = None,
        summary: TorchOperatorSummary | Mapping[str, object] | None = None,
        trace: ChromeTraceExport | Mapping[str, object] | None = None,
        sink: RunArtifactSink | Mapping[str, object] | None = None,
        observer_id: str = "native-profiler",
        backend_factory: _BackendFactory | None = None,
        **kwargs: object,
    ) -> None:
        if kwargs:
            raise TypeError(f"NativeProfiler received unsupported keys: {sorted(kwargs)}")
        self.enabled = bool(enabled)
        self.window = self._coerce_window(window)
        if self.enabled and self.window is None:
            raise ValueError("enabled NativeProfiler requires an explicit window")
        self.summary = self._coerce_summary(summary)
        self.trace = self._coerce_trace(trace)
        self.sink = RunArtifactSink(relative_dir="native-profiling") if sink is None else self._coerce_sink(sink)
        self.observer_id = observer_id
        self.backend_factory = backend_factory
        self._active_session: _NativeSession | None = None
        self._unsafe = False
        self._event_types = _load_enabled_event_types() if self.enabled else None
        super().__init__(typed_groups=self._typed_groups(self._event_types) if self._event_types else ())

    @staticmethod
    def _typed_groups(event_types: _EnabledEventTypes) -> tuple[SubscriptionGroup, ...]:
        """Build the fixed event plan only for an enabled profiler."""

        from tpen.events import ended, started

        return (SubscriptionGroup(selectors=(
            started(event_types.training_iteration), ended(event_types.training_iteration),
            started(event_types.collect_samples), ended(event_types.collect_samples),
            started(event_types.sampler_burn_in), ended(event_types.sampler_burn_in),
            started(event_types.sampler_production), ended(event_types.sampler_production),
        )),)

    @staticmethod
    def _coerce_window(value: TrainingIterationWindow | Mapping[str, object] | None) -> TrainingIterationWindow | None:
        if value is None:
            return None
        if isinstance(value, TrainingIterationWindow):
            return value
        if isinstance(value, Mapping):
            return TrainingIterationWindow(**dict(value))
        raise TypeError("window must be TrainingIterationWindow or a mapping")

    @staticmethod
    def _coerce_summary(value: TorchOperatorSummary | Mapping[str, object] | None) -> TorchOperatorSummary:
        if value is None:
            return TorchOperatorSummary()
        if isinstance(value, TorchOperatorSummary):
            return value
        if isinstance(value, Mapping):
            return TorchOperatorSummary(**dict(value))
        raise TypeError("summary must be TorchOperatorSummary or a mapping")

    @staticmethod
    def _coerce_trace(value: ChromeTraceExport | Mapping[str, object] | None) -> ChromeTraceExport | None:
        if value is None:
            return None
        if isinstance(value, ChromeTraceExport):
            return value
        if isinstance(value, Mapping) and not value:
            # Empty mappings preserve Hydra construction compatibility while
            # keeping the exporter entirely code-owned and zero-knob.
            return ChromeTraceExport()
        raise TypeError("trace must be ChromeTraceExport, an empty mapping, or None")

    @staticmethod
    def _coerce_sink(value: RunArtifactSink | Mapping[str, object]) -> RunArtifactSink:
        if type(value) is RunArtifactSink:
            return value
        if isinstance(value, Mapping):
            return RunArtifactSink(**dict(value))
        raise TypeError("sink must be exactly RunArtifactSink or a mapping")

    def _trace_relative_path(self) -> str:
        """Return the fixed bundle-relative Chrome trace destination."""

        window = self.window
        if window is None:
            raise ValueError("enabled NativeProfiler requires an explicit window")
        window_id = f"step-{window.start_step}-{window.end_step}"
        return f"profiles/{self.observer_id}/{window_id}/chrome-trace.json"

    def __getstate__(self) -> object:
        raise TypeError("NativeProfiler is invocation-local and cannot be serialized")

    def __reduce__(self) -> object:
        raise TypeError("NativeProfiler is invocation-local and cannot be serialized")

    def __reduce_ex__(self, protocol: object) -> object:
        del protocol
        raise TypeError("NativeProfiler is invocation-local and cannot be serialized")

    def is_managed_scope_active(self) -> bool:
        """Expose unsafe ownership to the core lifecycle registry."""

        return self._active_session is not None or self._unsafe

    def ownership_claims(
        self,
        *,
        artifact_manager: ArtifactManager | None = None,
        invocation_id: str | None = None,
        **_: object,
    ) -> list[ResourceOwnershipClaim]:
        """Return process and exact profile-artifact ownership claims."""

        if not self.enabled:
            return []
        claims = [ResourceOwnershipClaim(kind="torch_profiler", scope="process")]
        if artifact_manager is None or invocation_id is None:
            return claims
        bundle = self.sink.resolve_bundle_dir(artifact_manager, invocation_id)
        window = self.window
        if window is None:
            raise ValueError("enabled NativeProfiler requires an explicit window")
        window_id = f"step-{window.start_step}-{window.end_step}"
        rels = (
            "manifest.json",
            f"profiles/{self.observer_id}/{window_id}/profile-status.json",
            f"profiles/{self.observer_id}/{window_id}/summary.json",
        )
        if self.trace is not None:
            rels += (self._trace_relative_path(),)
        for relative in rels:
            target = bundle / relative
            claims.append(ResourceOwnershipClaim(
                kind="artifact_writer", scope=str(target.resolve(strict=False))
            ))
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
        if session.unsafe:
            self._unsafe = True

    def _new_writer(self, context: RunContext) -> ObservabilityBundleWriter:
        return ObservabilityBundleWriter(context.artifact_manager, context.invocation_id, sink=self.sink)

    def _open_manifest(self, context: RunContext) -> ObservabilityManifest:
        return ObservabilityManifest(
            invocation_id=context.invocation_id, lifecycle_state="open", data_completeness="unknown",
            run_id=context.metadata.run_id, producer={"name": "tpen.callback.native_profiler", "version": "1"},
            configuration_digest=hashlib.sha256(b"native-profiler").hexdigest(),
            inventory_id=f"inventory-{context.invocation_id}", descriptor_set_id=f"descriptors-{context.invocation_id}",
            observers=({"observer_id": self.observer_id},), limits={"max_rows": self.summary.max_rows},
        )

    def handle_occurrence_impl(self, occurrence: Occurrence[TypedEvent], context: RunContext) -> None:
        del context
        session = self._active_session
        event_types = self._event_types
        window = self.window
        if session is None or event_types is None or window is None:
            return
        event = occurrence.event
        if isinstance(event, Started) and isinstance(event.operation, event_types.training_iteration):
            step = event.operation.step
            session._typed_counts[event_types.training_iteration] = session._typed_counts.get(event_types.training_iteration, 0) + 1
            phase = window.phase(step)
            if phase == "before":
                session.current_step = step
                return
            if phase == "after":
                if not session.window_started:
                    session.window_start_not_observed = True
                    session._diagnostic("window_start_not_observed")
                else:
                    session._overshoot_steps += 1
                session.request_stop()
                return
            if not session.window_started:
                if step != window.start_step:
                    session.window_start_not_observed = True
                    session._diagnostic("window_start_not_observed")
                    session.request_stop()
                    return
                session.window_started = True
                session.current_step = step
                session._start()
            if session.started and not session.stop_requested:
                session._steps.append(step)
                session.current_step = step
                session._open_range(event.operation, occurrence.count)
            return
        if isinstance(event, Ended) and isinstance(event.operation, event_types.training_iteration):
            session._close_range(event.operation, occurrence.count)
            if session.started and not session.stopped and not session.stop_requested:
                try:
                    if session.backend is not None and event.operation.step <= window.end_step:
                        session.backend.step()
                    if event.operation.step == window.end_step:
                        session.window_completed = True
                        session.request_stop()
                        session._stop()
                except BaseException as exc:
                    session._diagnostic("backend_step_failed", exc)
            return
        fixed_types = (event_types.collect_samples, event_types.sampler_burn_in, event_types.sampler_production)
        if isinstance(event, Started) and isinstance(event.operation, fixed_types):
            op_type = type(event.operation)
            session._typed_counts[op_type] = session._typed_counts.get(op_type, 0) + 1
            if session.started and session.current_step is not None and not session.stop_requested:
                session._open_range(event.operation, occurrence.count)
        elif isinstance(event, Ended) and isinstance(event.operation, fixed_types):
            session._close_range(event.operation, occurrence.count)

    def _persist(self, session: _NativeSession, *, application_error: BaseException | None) -> None:
        """Persist bounded status and summary without replacing application errors."""

        try:
            session._open_writer()
            writer = session.writer
            if writer is None or self.window is None:
                return
            rows: tuple[Mapping[str, object], ...] = ()
            total_rows = 0
            summary_truncated = False
            summary_succeeded = not session.started
            if session.backend is not None and session.started:
                try:
                    rows, total_rows, summary_truncated = self.summary.collect(session.backend)
                    summary_succeeded = True
                except BaseException as exc:
                    summary_succeeded = False
                    session._diagnostic("summary_failed", exc)
            session._export_trace()
            if application_error is not None:
                session._diagnostic(f"application_error={type(application_error).__name__}")
            if session._overshoot_steps:
                session._diagnostic(f"overshoot_steps={session._overshoot_steps}")
            if summary_truncated:
                session._diagnostic("summary_rows_truncated")
            window_id = f"step-{self.window.start_step}-{self.window.end_step}"
            if session.window_start_not_observed:
                lifecycle_state, completeness = "not_started", "none"
            elif session.unsafe:
                lifecycle_state, completeness = "unsafe", "partial"
            else:
                lifecycle_state = "closed"
                completeness = "complete" if session.window_completed else "partial"
            export_failed = any(
                item.startswith(("backend_", "persistence_", "summary_failed", "trace_"))
                for item in session._diagnostics
            )
            if summary_succeeded is False:
                completeness = "partial"
            if session.trace_ref is None and self.trace is not None and session.started:
                completeness = "partial"
            diagnostics = tuple(session._diagnostics)
            requested = ("CPU",)
            accepted = requested if session.started else ()
            writer.write_profile_summary(ObservabilityProfileSummary(
                session.context.invocation_id, self.observer_id, window_id, "TorchOperatorSummary",
                self.summary.max_rows, total_rows, rows, truncated=summary_truncated,
                truncation_reasons=("max_rows_exceeded",) if summary_truncated else (),
                source_provenance={"requested_activities": list(requested), "accepted_activities": list(accepted)},
            ))
            writer.write_profile_status(ObservabilityProfileStatus(
                invocation_id=session.context.invocation_id,
                observer_id=self.observer_id,
                window_id=window_id,
                window_kind="operation",
                requested_activities=requested,
                accepted_activities=accepted,
                observed_event_counts={op.__name__: count for op, count in session._typed_counts.items()},
                clock_domain_id="process-monotonic",
                activation_ns=session._activation_ns,
                lifecycle_state=lifecycle_state,
                data_completeness=completeness,
                guarantee_class="diagnostic",
                exporter=(
                    "TorchOperatorSummary+ChromeTraceExport"
                    if self.trace is not None
                    else "TorchOperatorSummary"
                ),
                export_status=(
                    "failed"
                    if export_failed
                    else ("not_started" if not session.started else "complete")
                ),
                requested_start_ns=None,
                requested_end_ns=None,
                start_call_bracket_ns=session._start_bracket_ns, stop_call_bracket_ns=session._stop_bracket_ns,
                backend_event_bounds={"status": "unavailable", "reason_code": "cooperative_boundary"},
                stop_overshoot_bounds_seconds=None, activity_diagnostics=diagnostics,
                durable_steps=tuple(session._steps), export_mode="summary",
                artifacts=(session.trace_ref,) if session.trace_ref is not None else None,
                truncation_reasons=("max_rows_exceeded",) if summary_truncated else None,
            ))
            writer.finalize_manifest(ObservabilityManifest(
                session.context.invocation_id, "unsafe" if session.unsafe else "closed", completeness,
                session.context.metadata.run_id, {"name": "tpen.callback.native_profiler", "version": "1"},
                hashlib.sha256(b"native-profiler").hexdigest(), f"inventory-{session.context.invocation_id}",
                f"descriptors-{session.context.invocation_id}", ({"observer_id": self.observer_id},),
                {"max_rows": self.summary.max_rows},
            ))
        except BaseException as exc:
            # Export and persistence are diagnostic-only after activation.
            session._diagnostic("persistence_failed", exc)


__all__ = ["NativeProfiler", "TorchOperatorSummary", "TrainingIterationWindow"]
