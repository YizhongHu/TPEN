"""Focused tests for O1 managed observability lifecycle and readiness contract.

Covers Slice A of the observability contracts:
- Torch-free imports and core/experiments/executor isolation
- ``RunContextReadiness`` state machine, central operation gating, and early metadata
- ``ManagedCallback`` / ``RunResource`` / ``ResourceOwnershipClaim`` helpers
- Legacy / disabled path characterization
- Preflight refusals leaving zero directories, artifacts, or entered scopes
- Early context contract, preparation checkpoints, and partial-enter rollback
- Preparation failure before ``READY``, failure/yield precedence, and LIFO teardown
- First typed-context binding and repeated invocations
- Programmatic ``open_run_context`` and unscoped ``prepare_run_context`` refusal
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from contextlib import contextmanager
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

from omegaconf import DictConfig, OmegaConf
import pytest
from typeguard import TypeCheckError, suppress_type_checks

import tpen.artifacts as artifacts_mod
from tpen.artifacts import (
    ArtifactDestination,
    ArtifactManager,
    ArtifactRoutingError,
    RunClock,
    RunContext,
    RunContextReadiness,
    RunContextReadinessError,
    RunIdentityError,
    RunResult,
    build_run_metadata,
)
from tpen.callback import (
    Cadence,
    Callback,
    ManagedLifecycleError,
    ResourceOwnershipClaim,
    ResourceOwnershipConflictError,
    ResourceUsage,
    RunResource,
    StatefulCallback,
    SubscriptionGroup,
    canonicalize_allocator_reset_scope,
    claims_conflict,
    is_managed_callback,
    validate_run_resource,
)
from tpen.config_schema import ClosedSchemaError
from tpen.distributed import ExecutionTopology, ProfileRecord, ProfileScope
from tpen.events import DomainState, Event, Occurrence, Operation, Subscription
from tpen.hi_schema import HI_TRAIN_SCHEMA
from tpen.process_resources import ProcessResourceResult
import tpen.run as run_mod
from tpen.run import (
    _ManagedLifecycleRegistry,
    open_run_context,
    prepare_run_context,
    run_from_config,
)
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tpen.runner import Runner
from tests.helpers.run_context import RecordingLogger, make_run_context

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _isolated_managed_registry(monkeypatch: pytest.MonkeyPatch) -> _ManagedLifecycleRegistry:
    """Give each test a fresh process-wide registry so unsafe-state tests stay isolated."""

    fresh = _ManagedLifecycleRegistry()
    monkeypatch.setattr(run_mod, "_MANAGED_LIFECYCLE_REGISTRY", fresh)
    return fresh


# ---------------------------------------------------------------------------
# Lightweight torch-free test doubles (module-level for Hydra _target_ lookup)
# ---------------------------------------------------------------------------


class _PulseEvent(Event):
    pass


class _WorkOp(Operation):
    pass


class _DummyState(DomainState):
    pass


class _NoopRunner(Runner):
    def run(self, context: RunContext) -> RunResult:
        del context
        return RunResult(status="completed")


class _ReturnedFailedRunner(Runner):
    def run(self, context: RunContext) -> RunResult:
        del context
        return RunResult(status="failed", error="suite task failed")


class _RaisingRunner(Runner):
    def run(self, context: RunContext) -> RunResult:
        del context
        raise RuntimeError("runner exploded")


class _YieldingRunner(Runner):
    def run(self, context: RunContext) -> RunResult:
        from tpen.callback.planned_yield import PlannedYieldSignal

        del context
        raise PlannedYieldSignal(
            RunResult(
                status="planned_yield",
                next_iteration=5,
                completed_updates=5,
                target_iterations=10,
            )
        )


class _CheckpointFailureRunner(Runner):
    def run(self, context: RunContext) -> RunResult:
        from tpen.callback.checkpoint import CheckpointPublicationError

        del context
        err = OSError("disk full")
        setattr(err, "tpen_publication_stage", "commit")
        raise CheckpointPublicationError(
            err,
            next_iteration=3,
            completed_updates=3,
            target_iterations=10,
        )


class _KeyboardInterruptRunner(Runner):
    def run(self, context: RunContext) -> RunResult:
        del context
        raise KeyboardInterrupt("user interrupt")


class _RecordingLogger:
    """Configured logger tracking log/finish calls and optional failure injection."""

    instances: list[_RecordingLogger] = []

    def __init__(
        self,
        *,
        name: str = "logger",
        fail_on_finish: bool = False,
        trace: list[str] | None = None,
    ) -> None:
        self.name = name
        self.fail_on_finish = fail_on_finish
        self.trace = trace
        self.records: list[Any] = []
        self.finished = 0
        if self.trace is not None:
            self.trace.append(f"init:{self.name}")
        self.instances.append(self)

    def log(self, record: Any) -> None:
        self.records.append(record)

    def finish(self) -> None:
        self.finished += 1
        if self.trace is not None:
            self.trace.append(f"finish:{self.name}")
        if self.fail_on_finish:
            raise RuntimeError(f"logger finish boom: {self.name}")


class _OrdinaryCallback(Callback):
    """Ordinary callback recording construction and observed readiness/events."""

    instances: list[_OrdinaryCallback] = []

    def __init__(
        self,
        *,
        name: str = "ordinary",
        fail_on_init: bool = False,
        fail_on_event: str | None = None,
        trace: list[str] | None = None,
    ) -> None:
        if trace is not None:
            trace.append(f"init:{name}")
        if fail_on_init:
            raise RuntimeError(f"ordinary init boom: {name}")
        super().__init__(
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(RunStarted),
                        Subscription.of(RunCompleted),
                        Subscription.of(RunFailed),
                    )
                ),
            )
        )
        self.name = name
        self.fail_on_event = fail_on_event
        self.trace = trace
        self.events: list[tuple[str, RunContextReadiness]] = []
        self.instances.append(self)

    def handle_occurrence_impl(
        self, occurrence: Occurrence[Any], context: RunContext
    ) -> None:
        event_name = type(occurrence.event).__name__
        self.events.append((event_name, context.readiness))
        if self.trace is not None:
            self.trace.append(f"event:{self.name}:{event_name}")
        if self.fail_on_event == event_name:
            raise RuntimeError(f"ordinary callback boom on {event_name}")


class _RecordingHandle:
    """Minimal RunResource handle recording checkpoints and stop requests."""

    def __init__(
        self,
        *,
        name: str,
        trace: list[str],
        fail_on_stop: bool = False,
        fail_on_checkpoint: bool = False,
    ) -> None:
        self.name = name
        self.trace = trace
        self.fail_on_stop = fail_on_stop
        self.fail_on_checkpoint = fail_on_checkpoint
        self.checkpoints = 0
        self.stop_requests = 0

    def checkpoint(self) -> None:
        self.checkpoints += 1
        self.trace.append(f"checkpoint:{self.name}")
        if self.fail_on_checkpoint:
            raise RuntimeError(f"checkpoint boom: {self.name}")

    def request_stop(self) -> None:
        self.stop_requests += 1
        self.trace.append(f"request_stop:{self.name}")
        if self.fail_on_stop:
            raise RuntimeError(f"request_stop boom: {self.name}")


class _ManagedScopeCM:
    """Explicit context manager so tests can control __enter__/__exit__ return/raise."""

    def __init__(
        self,
        owner: _ManagedTestCallback,
        context: RunContext,
    ) -> None:
        self.owner = owner
        self.context = context

    def __enter__(self) -> Any:
        self.owner.enter_readiness.append(self.context.readiness)
        self.owner.enter_metadata_extra.append(dict(self.context.metadata.extra))
        self.owner.enter_metadata_status.append(self.context.metadata.status)
        self.owner.enter_invocation_ids.append(self.context.invocation_id)
        self.owner.trace.append(f"enter:{self.owner.name}")
        if self.owner.fail_on_enter:
            if self.owner.leave_active_on_enter_failure:
                self.owner._active = True
            raise RuntimeError(f"enter boom: {self.owner.name}")
        self.owner._active = True
        self.owner.activated_token = f"active-{self.owner.reset_count}"
        if self.owner.return_invalid_handle:
            return object()
        if self.owner.return_none_handle:
            return None
        handle = _RecordingHandle(
            name=self.owner.name,
            trace=self.owner.trace,
            fail_on_stop=self.owner.fail_on_stop,
            fail_on_checkpoint=self.owner.fail_on_checkpoint,
        )
        self.owner.handles.append(handle)
        return handle

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        del exc_tb
        self.owner.exit_readiness.append(self.context.readiness)
        self.owner.exit_exc_types.append(exc_type)
        self.owner.exit_exc_values.append(exc_val)
        self.owner.trace.append(f"exit:{self.owner.name}")
        if not self.owner.leave_active_on_exit:
            self.owner._active = False
        if self.owner.fail_on_exit:
            raise RuntimeError(f"exit boom: {self.owner.name}")
        return self.owner.suppress_in_exit


class _ManagedTestCallback(Callback):
    """Configurable managed callback for lifecycle, rollback, and ownership tests."""

    instances: list[_ManagedTestCallback] = []

    def __init__(
        self,
        *,
        name: str = "managed",
        claims: list[ResourceOwnershipClaim] | None = None,
        claim_specs: list[dict[str, str]] | None = None,
        output_path: str | Path | None = None,
        fail_on_enter: bool = False,
        fail_on_stop: bool = False,
        fail_on_exit: bool = False,
        fail_on_checkpoint: bool = False,
        suppress_in_exit: bool = False,
        return_invalid_handle: bool = False,
        return_none_handle: bool = False,
        arrive_active: bool = False,
        leave_active_on_enter_failure: bool = False,
        leave_active_on_exit: bool = False,
        probability: float = 1.0,
        trace: list[str] | None = None,
    ) -> None:
        super().__init__(
            probability=probability,
            typed_groups=(
                SubscriptionGroup(
                    selectors=(
                        Subscription.of(RunStarted),
                        Subscription.of(RunCompleted),
                        Subscription.of(RunFailed),
                    ),
                    cadence=None if probability == 1.0 else Cadence(probability=probability),
                ),
            ),
        )
        self.name = name
        self.output_path = None if output_path is None else Path(output_path)
        self._claims = list(claims or [])
        if claim_specs:
            for spec in claim_specs:
                self._claims.append(
                    ResourceOwnershipClaim(kind=spec["kind"], scope=spec["scope"])
                )
        self.fail_on_enter = fail_on_enter
        self.fail_on_stop = fail_on_stop
        self.fail_on_exit = fail_on_exit
        self.fail_on_checkpoint = fail_on_checkpoint
        self.suppress_in_exit = suppress_in_exit
        self.return_invalid_handle = return_invalid_handle
        self.return_none_handle = return_none_handle
        self._active = arrive_active
        self.leave_active_on_enter_failure = leave_active_on_enter_failure
        self.leave_active_on_exit = leave_active_on_exit
        self.trace = [] if trace is None else trace
        self.handles: list[_RecordingHandle] = []
        self.enter_readiness: list[RunContextReadiness] = []
        self.enter_metadata_extra: list[dict[str, Any]] = []
        self.enter_metadata_status: list[str] = []
        self.enter_invocation_ids: list[str] = []
        self.exit_readiness: list[RunContextReadiness] = []
        self.exit_exc_types: list[Any] = []
        self.exit_exc_values: list[Any] = []
        self.events: list[tuple[str, str | None]] = []
        self.reset_count = 0
        self.activated_token: str | None = None
        self.instances.append(self)

    def _reset_typed_state(self) -> None:
        self.reset_count += 1
        self.activated_token = None
        self.trace.append(f"reset:{self.name}:{self.reset_count}")

    def is_managed_scope_active(self) -> bool:
        return self._active

    def ownership_claims(self, cfg: DictConfig) -> list[ResourceOwnershipClaim]:
        del cfg
        return list(self._claims)

    def run_scope(self, context: RunContext) -> _ManagedScopeCM:
        return _ManagedScopeCM(self, context)

    def handle_occurrence_impl(
        self, occurrence: Occurrence[Any], context: RunContext
    ) -> None:
        del context
        event_name = type(occurrence.event).__name__
        self.events.append((event_name, self.activated_token))
        self.trace.append(f"event:{self.name}:{event_name}")


def _build_managed_callback(name: str = "from_factory") -> _ManagedTestCallback:
    """Module-level factory returning a managed callback for Hydra _target_ tests."""

    return _ManagedTestCallback(name=name)


class _BadScopeAttrCallback(Callback):
    """Module-level callback with a non-callable run_scope attribute."""

    run_scope = "not-callable"  # type: ignore[assignment]


class _GcProbingManagedCallback(_ManagedTestCallback):
    """Managed callback that triggers cyclic GC on its first active-state check."""

    def __init__(self, *, name: str = "gc_probe") -> None:
        super().__init__(name=name)
        self._trigger_gc_once = True

    def is_managed_scope_active(self) -> bool:
        if self._trigger_gc_once:
            self._trigger_gc_once = False
            gc.collect()
        return super().is_managed_scope_active()


class _OpaqueAllocatorProbe:
    """Allocator probe double exposing no device attribute (conservative 'unknown')."""

    def reset(self) -> Any:
        return None

    def read(self) -> Any:
        return None


class _DeviceAllocatorProbe:
    """Allocator probe double exposing a concrete configured device."""

    def __init__(self, device: str) -> None:
        self.device = device

    def reset(self) -> Any:
        return None

    def read(self) -> Any:
        return None


def _base_cfg(tmp_path: Path, **overrides: Any) -> DictConfig:
    data: dict[str, Any] = {
        "experiment": {"name": "obs", "sector": "unit", "run_name": "obs_unit"},
        "run": {"root": str(tmp_path / "outputs"), "run_id": "run_001", "dir": None},
        "runtime": {"device": "cpu", "dtype": "float64", "seed": 0},
        "terminal": {"enabled": False},
        "runner": {"_target_": f"{__name__}._NoopRunner"},
        "callbacks": [],
        "loggers": [],
    }
    data.update(overrides)
    return OmegaConf.create(data)


# ---------------------------------------------------------------------------
# 1. Torch-free and boundary hygiene
# ---------------------------------------------------------------------------


def test_core_modules_stay_torch_free_and_free_of_experiments_or_executor(
    tmp_path: Path,
) -> None:
    """Core lifecycle modules import without torch, experiments, or tpen_executor."""

    (tmp_path / "torch.py").write_text('raise AssertionError("torch imported")\n', encoding="utf-8")
    env = os.environ.copy()
    pythonpath = [str(tmp_path), str(_REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tpen.artifacts, tpen.callback, tpen.callback.base, tpen.run; "
                "bad = [m for m in ('torch', 'tpen.training', 'experiments', 'tpen_executor') "
                "if m in sys.modules]; "
                "assert not bad, bad; "
                "print('OK')"
            ),
        ],
        cwd=_REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"

    for rel in (
        "tpen/artifacts.py",
        "tpen/callback/base.py",
        "tpen/callback/__init__.py",
        "tpen/run.py",
    ):
        tree = ast.parse((_REPO_ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("experiments", "tpen_executor")), (rel, alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("experiments", "tpen_executor")), (rel, node.module)


# ---------------------------------------------------------------------------
# 2. RunContextReadiness & RunContext gating
# ---------------------------------------------------------------------------


def test_run_context_readiness_transitions_and_gating(tmp_path: Path) -> None:
    """Central operations refuse non-READY states without advancing counts or writing."""

    logger = RecordingLogger()
    ctx = make_run_context(tmp_path, loggers=[logger])
    assert ctx.readiness is RunContextReadiness.READY
    assert isinstance(ctx.invocation_id, str) and ctx.invocation_id.strip()

    # Construct a PREPARING context directly to test pre-ready gating.
    preparing = RunContext(
        cfg=ctx.cfg,
        source_cfg=ctx.source_cfg,
        artifact_manager=ctx.artifact_manager,
        metadata=ctx.metadata,
        clock=ctx.clock,
        loggers=[logger],
        initial_readiness=RunContextReadiness.PREPARING,
    )
    assert preparing.readiness is RunContextReadiness.PREPARING
    assert preparing.invocation_id != ctx.invocation_id

    profile_record = ProfileRecord(
        scope=ProfileScope.PROCESS,
        monotonic_time=0.0,
        topology=ExecutionTopology.single_process(device="cpu"),
        process=ProcessResourceResult(
            user_cpu_seconds=0.1,
            system_cpu_seconds=0.05,
            read_block_operations=0,
            write_block_operations=0,
            voluntary_context_switches=1,
            involuntary_context_switches=0,
            peak_rss_mb=32.0,
        ),
    )
    # Positive control: READY context reaches the profile-writer check rather than failing readiness.
    with pytest.raises(RuntimeError, match="rank-local profile writer is not configured") as ready_exc:
        ctx.write_profile(profile_record)
    assert not isinstance(ready_exc.value, RunContextReadinessError)

    for state in (RunContextReadiness.PREPARING, RunContextReadiness.FINALIZING, RunContextReadiness.CLOSED):
        target = RunContext(
            cfg=ctx.cfg,
            source_cfg=ctx.source_cfg,
            artifact_manager=ctx.artifact_manager,
            metadata=ctx.metadata,
            clock=ctx.clock,
            loggers=[logger],
            initial_readiness=state,
        )
        with pytest.raises(RunContextReadinessError, match="emit"):
            target.emit(_PulseEvent())
        with pytest.raises(RunContextReadinessError, match="scope"):
            with target.scope(_WorkOp()):
                pass
        with pytest.raises(RunContextReadinessError, match="log"):
            target.log({"x": 1.0})
        with pytest.raises(RunContextReadinessError, match="write_profile"):
            target.write_profile(profile_record)
        with pytest.raises(RunContextReadinessError, match="_dispatch_occurrence"):
            target._dispatch_occurrence(target._occurrence(event=_PulseEvent(), count=1))
        assert target._occurrence_counts == {}
        assert logger.records == []

    # Valid transition graph: PREPARING -> READY -> FINALIZING -> CLOSED.
    preparing._transition_readiness(RunContextReadiness.READY)
    with pytest.raises(RunContextReadinessError):
        preparing._transition_readiness(RunContextReadiness.PREPARING)
    preparing._transition_readiness(RunContextReadiness.FINALIZING)
    with pytest.raises(RunContextReadinessError):
        preparing._transition_readiness(RunContextReadiness.READY)
    preparing._transition_readiness(RunContextReadiness.CLOSED)
    with pytest.raises(RunContextReadinessError):
        preparing._transition_readiness(RunContextReadiness.FINALIZING)

    # PREPARING -> FINALIZING -> CLOSED is valid for preparation failures.
    aborted = RunContext(
        cfg=ctx.cfg,
        source_cfg=ctx.source_cfg,
        artifact_manager=ctx.artifact_manager,
        metadata=ctx.metadata,
        clock=ctx.clock,
        initial_readiness=RunContextReadiness.PREPARING,
    )
    with pytest.raises(RunContextReadinessError):
        aborted._transition_readiness(RunContextReadiness.CLOSED)
    aborted._transition_readiness(RunContextReadiness.FINALIZING)
    aborted._transition_readiness(RunContextReadiness.CLOSED)

    with pytest.raises((TypeError, TypeCheckError)):
        RunContext(
            cfg=ctx.cfg,
            source_cfg=ctx.source_cfg,
            artifact_manager=ctx.artifact_manager,
            metadata=ctx.metadata,
            clock=ctx.clock,
            initial_readiness="ready",  # type: ignore[arg-type]
        )
    with suppress_type_checks():
        with pytest.raises(TypeError):
            RunContext(
                cfg=ctx.cfg,
                source_cfg=ctx.source_cfg,
                artifact_manager=ctx.artifact_manager,
                metadata=ctx.metadata,
                clock=ctx.clock,
                initial_readiness="ready",  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError):
        RunContext(
            cfg=ctx.cfg,
            source_cfg=ctx.source_cfg,
            artifact_manager=ctx.artifact_manager,
            metadata=ctx.metadata,
            clock=ctx.clock,
            invocation_id="   ",
        )


def test_run_context_subclass_and_new_default_to_ready(tmp_path: Path) -> None:
    """Subclasses skipping super().__init__() and object.__new__(RunContext) default to READY."""

    base = make_run_context(tmp_path)

    class _NoSuperContext(RunContext):
        def __init__(self) -> None:
            self.artifact_manager = base.artifact_manager
            self.metadata = base.metadata
            self.clock = base.clock
            self.callbacks = []
            self._occurrence_counts = {}

    sub = _NoSuperContext()
    assert sub.readiness is RunContextReadiness.READY
    assert sub._checkpoint_hook is None
    sub._require_ready("emit")
    sub._dispatch_occurrence(sub._occurrence(event=_PulseEvent(), count=1))
    assert sub.emit(_PulseEvent()).count == 1
    with pytest.raises(RunContextReadinessError):
        sub._transition_readiness(RunContextReadiness.PREPARING)
    sub._transition_readiness(RunContextReadiness.FINALIZING)
    assert sub.readiness is RunContextReadiness.FINALIZING
    assert RunContext._readiness is RunContextReadiness.READY
    with pytest.raises(RunContextReadinessError, match="emit"):
        sub._require_ready("emit")

    raw = object.__new__(RunContext)
    assert raw.readiness is RunContextReadiness.READY
    assert raw._checkpoint_hook is None
    raw._require_ready("write_profile")
    raw.artifact_manager = base.artifact_manager
    raw.metadata = base.metadata
    raw.clock = base.clock
    raw.callbacks = []
    raw._dispatch_occurrence(raw._occurrence(event=_PulseEvent(), count=1))
    with pytest.raises(RunContextReadinessError):
        raw._transition_readiness(RunContextReadiness.PREPARING)
    raw._transition_readiness(RunContextReadiness.FINALIZING)
    assert raw.readiness is RunContextReadiness.FINALIZING
    assert RunContext._readiness is RunContextReadiness.READY


def test_build_run_metadata_include_hardware_false_skips_hardware_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """include_hardware=False builds valid identity/git metadata with empty extra."""

    called: list[bool] = []
    monkeypatch.setattr(
        artifacts_mod,
        "collect_hardware_metadata",
        lambda **kwargs: (called.append(True), {"hardware": kwargs})[1],
    )
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "exp", "run_name": "exp_run"},
            "run": {"run_id": "id1", "dir": str(tmp_path / "run")},
            "runtime": {"device": "cpu", "dtype": "float64"},
        }
    )

    early = build_run_metadata(cfg, command="cmd", config_path="cfg.yaml", include_hardware=False)
    assert called == []
    assert early.extra == {}
    assert early.run_id == "id1"
    assert early.status == "initialized"
    assert isinstance(early.dirty_worktree, bool)

    full = build_run_metadata(cfg, command="cmd", config_path="cfg.yaml", include_hardware=True)
    assert called == [True]
    assert "hardware" in full.extra


def test_dispatch_occurrence_checkpoint_hook_runs_before_and_after_without_reordering(
    tmp_path: Path,
) -> None:
    """Checkpoint hook brackets dispatch cleanly and does not swallow exceptions."""

    order: list[str] = []

    class _OrderCb(Callback):
        def __init__(self, label: str, *, explode: bool = False) -> None:
            super().__init__(
                typed_groups=(SubscriptionGroup(selectors=(Subscription.of(_PulseEvent),)),)
            )
            self.label = label
            self.explode = explode

        def handle_occurrence_impl(self, occurrence: Occurrence[Any], context: RunContext) -> None:
            del occurrence, context
            order.append(f"cb:{self.label}")
            if self.explode:
                raise RuntimeError("cb boom")

    ctx = make_run_context(tmp_path, callbacks=[_OrderCb("a"), _OrderCb("b")])
    ctx._checkpoint_hook = lambda: order.append("checkpoint")

    occ = ctx.emit(_PulseEvent())
    assert occ.count == 1
    assert order == ["checkpoint", "cb:a", "cb:b", "checkpoint"]

    # Failing callback still propagates untouched.
    ctx2 = make_run_context(tmp_path / "err", callbacks=[_OrderCb("boom", explode=True)])
    ctx2._checkpoint_hook = lambda: order.append("checkpoint2")
    with pytest.raises(RuntimeError, match="cb boom"):
        ctx2.emit(_PulseEvent())


# ---------------------------------------------------------------------------
# 3. Callback base protocol, ownership claim, and canonicalization unit tests
# ---------------------------------------------------------------------------


def test_ownership_claim_canonicalization_and_conflict_rules(tmp_path: Path) -> None:
    """Conservative allocator/profiler/writer conflict rules match r5 section 7.5."""

    assert canonicalize_allocator_reset_scope("cpu") is None
    assert canonicalize_allocator_reset_scope("CPU:0") is None
    assert canonicalize_allocator_reset_scope("cuda") == "cuda"
    assert canonicalize_allocator_reset_scope("CUDA:0") == "cuda:0"
    assert canonicalize_allocator_reset_scope("xpu:2") == "xpu:2"
    assert canonicalize_allocator_reset_scope("mps") == "mps"
    assert canonicalize_allocator_reset_scope("rocm") == "rocm"
    assert canonicalize_allocator_reset_scope("hip") == "rocm"
    assert canonicalize_allocator_reset_scope("hip:1") == "rocm:1"
    for ambiguous in ("", "   ", "auto", "gpu", "cuda:", "cuda:-1", "cuda:abc", "meta", "cuda:0:1"):
        assert canonicalize_allocator_reset_scope(ambiguous) == "unknown"

    with pytest.raises(ValueError):
        ResourceOwnershipClaim(kind="bogus", scope="x")
    with pytest.raises(ValueError):
        ResourceOwnershipClaim(kind="artifact_writer", scope="   ")
    with pytest.raises(ValueError):
        ResourceOwnershipClaim(kind="torch_profiler", scope="cuda:0")

    prof1 = ResourceOwnershipClaim(kind="torch_profiler", scope="process")
    prof2 = ResourceOwnershipClaim(kind="torch_profiler", scope="process")
    assert claims_conflict(prof1, prof2) is True

    c_cuda = ResourceOwnershipClaim(kind="allocator_reset", scope="cuda")
    c_cuda0 = ResourceOwnershipClaim(kind="allocator_reset", scope="cuda:0")
    c_cuda1 = ResourceOwnershipClaim(kind="allocator_reset", scope="cuda:1")
    c_xpu0 = ResourceOwnershipClaim(kind="allocator_reset", scope="xpu:0")
    c_rocm = ResourceOwnershipClaim(kind="allocator_reset", scope="rocm")
    c_hip0 = ResourceOwnershipClaim(kind="allocator_reset", scope="hip:0")
    c_unk = ResourceOwnershipClaim(kind="allocator_reset", scope="unknown")
    assert claims_conflict(c_cuda, c_cuda0) is True
    assert claims_conflict(c_cuda0, c_cuda0) is True
    assert claims_conflict(c_cuda0, c_cuda1) is False
    assert claims_conflict(c_cuda0, c_xpu0) is False
    assert claims_conflict(c_rocm, c_hip0) is True
    assert claims_conflict(c_unk, c_cuda0) is True
    assert claims_conflict(c_unk, c_unk) is True
    assert claims_conflict(prof1, c_cuda0) is False

    w1 = ResourceOwnershipClaim(kind="artifact_writer", scope=str(tmp_path / "a" / ".." / "out.jsonl"))
    w2 = ResourceOwnershipClaim(kind="artifact_writer", scope=str(tmp_path / "out.jsonl"))
    w3 = ResourceOwnershipClaim(kind="artifact_writer", scope=str(tmp_path / "other.jsonl"))
    assert claims_conflict(w1, w2) is True
    assert claims_conflict(w1, w3) is False


def test_is_managed_callback_and_validate_run_resource_strictness() -> None:
    """Only _CallbackCore subclasses declaring callable run_scope are recognized."""

    class _NonCallbackWithScope:
        @contextmanager
        def run_scope(self, context: Any) -> Iterator[None]:
            del context
            yield None

    class _BadScopeAttr(Callback):
        run_scope = "not-callable"  # type: ignore[assignment]

    class _StatefulManaged(StatefulCallback[_DummyState]):
        state_type = _DummyState

        @contextmanager
        def run_scope(self, context: Any) -> Iterator[None]:
            del context
            yield None

    ordinary = _OrdinaryCallback()
    # Ad-hoc instance attribute must NOT be treated as a managed callback.
    ordinary.run_scope = lambda ctx: None  # type: ignore[attr-defined]
    assert is_managed_callback(ordinary) is False
    assert is_managed_callback(_NonCallbackWithScope()) is False
    assert is_managed_callback(_ManagedTestCallback()) is True
    assert is_managed_callback(_StatefulManaged()) is True
    with pytest.raises(TypeError, match="run_scope"):
        is_managed_callback(_BadScopeAttr())

    validate_run_resource(None)
    handle = _RecordingHandle(name="h", trace=[])
    assert isinstance(handle, RunResource)
    validate_run_resource(handle)
    with pytest.raises(TypeError, match="checkpoint"):
        validate_run_resource(object())

    class _MissingStop:
        def checkpoint(self) -> None:
            pass

    with pytest.raises(TypeError, match="request_stop"):
        validate_run_resource(_MissingStop())


# ---------------------------------------------------------------------------
# 4. Disabled / legacy characterization
# ---------------------------------------------------------------------------


def test_disabled_legacy_path_preserves_exact_order_and_artifacts(tmp_path: Path) -> None:
    """With no managed callbacks, legacy order, artifacts, and readiness stay intact."""

    trace: list[str] = []
    cb = _OrdinaryCallback(name="legacy_cb", trace=trace)
    cfg = _base_cfg(
        tmp_path,
        callbacks=[
            {"_target_": "tpen.callback.Metadata", "output_path": "${run.dir}/metadata.json"}
        ],
        loggers=[
            {"_target_": "tpen.logging.JSONL", "path": "${run.dir}/metrics.jsonl"}
        ],
    )

    rc = run_from_config(cfg, config_path="legacy.yaml", command="pytest", runtime_callbacks=[cb])
    assert rc == 0
    assert cb.events == [
        ("RunStarted", RunContextReadiness.READY),
        ("RunCompleted", RunContextReadiness.READY),
    ]
    run_dir = tmp_path / "outputs" / "obs" / "unit" / "run_001"
    assert (run_dir / "run_start.json").is_file()
    assert json.loads((run_dir / "metadata.json").read_text())["status"] == "completed"
    assert json.loads((run_dir / "run-outcome.json").read_text())["status"] == "completed"
    assert not (run_dir / "observability").exists()


# ---------------------------------------------------------------------------
# 5. Preflight refusals leaving zero dirs, artifacts, or entered scopes
# ---------------------------------------------------------------------------


def test_preflight_hi_firewall_refuses_before_dirs_or_scope_enter(tmp_path: Path) -> None:
    """HI closed schema refusal leaves zero outputs and never enters managed scope."""

    managed = _ManagedTestCallback(name="m")
    cfg = _base_cfg(
        tmp_path,
        schema=HI_TRAIN_SCHEMA,
        optimizer={"_target_": "torch.optim.Adam", "lr": 0.005},
        system={"nuclei": {"reference_energy": -2.9}},
    )
    with pytest.raises(ClosedSchemaError):
        run_from_config(cfg, raise_exceptions=True, runtime_callbacks=[managed])
    assert managed.enter_readiness == []
    assert not (tmp_path / "outputs").exists()


def test_preflight_destination_freshness_and_routed_unregistered_managed_callback(
    tmp_path: Path,
) -> None:
    """Non-fresh destination and unregistered routed managed callback refuse preflight."""

    # Non-fresh physical destination
    non_fresh = tmp_path / "non_fresh"
    non_fresh.mkdir()
    (non_fresh / "stale.txt").write_text("old")
    managed = _ManagedTestCallback(name="m")
    with pytest.raises(ArtifactRoutingError):
        run_from_config(
            _base_cfg(tmp_path),
            raise_exceptions=True,
            artifact_destination=ArtifactDestination(physical_root=non_fresh),
            runtime_callbacks=[managed],
        )
    assert managed.enter_readiness == []
    assert [p.name for p in non_fresh.iterdir()] == ["stale.txt"]

    # Unregistered managed callback (both runtime and configured) refused before make_dirs()
    fresh_dest = tmp_path / "fresh_dest"
    with pytest.raises(ArtifactRoutingError):
        run_from_config(
            _base_cfg(tmp_path),
            raise_exceptions=True,
            artifact_destination=ArtifactDestination(physical_root=fresh_dest),
            runtime_callbacks=[managed],
        )
    assert managed.enter_readiness == []
    assert not fresh_dest.exists()

    cfg_managed = _base_cfg(
        tmp_path,
        callbacks=[{"_target_": f"{__name__}._ManagedTestCallback", "name": "cfg_m"}],
    )
    with pytest.raises(ArtifactRoutingError):
        run_from_config(
            cfg_managed,
            raise_exceptions=True,
            artifact_destination=ArtifactDestination(physical_root=fresh_dest),
        )
    assert not fresh_dest.exists()


def test_preflight_run_identity_error_leaves_zero_dirs_or_scopes(tmp_path: Path) -> None:
    """Multi-rank run-id disagreement refuses before make_dirs() or scope activation."""

    managed = _ManagedTestCallback(name="m")
    cfg = _base_cfg(tmp_path)
    cfg.run.run_id = None
    topology = ExecutionTopology(
        global_rank=0,
        global_size=2,
        local_rank=0,
        local_size=2,
        node_rank=0,
        node_size=1,
        host="localhost",
        pid=1234,
        device="cpu",
    )
    with pytest.raises(RunIdentityError):
        run_from_config(cfg, raise_exceptions=True, topology=topology, runtime_callbacks=[managed])
    assert managed.enter_readiness == []
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "typed_groups",
        "selectors",
        "subscriptions",
        "triggers",
        "event_selector",
        "event_selectors",
        "event_type",
        "event_types",
        "operation",
        "operation_selector",
        "operation_selectors",
        "operation_type",
        "operation_types",
    ],
)
def test_free_form_configured_selector_keys_refused_before_dirs(
    tmp_path: Path, forbidden_key: str
) -> None:
    """Every forbidden selector key (including nested) fails preflight with ValueError."""

    cfg = _base_cfg(
        tmp_path,
        callbacks=[
            {
                "_target_": f"{__name__}._OrdinaryCallback",
                "nested": {forbidden_key: ["RunStarted"]},
            }
        ],
    )
    with pytest.raises(ValueError, match=forbidden_key):
        prepare_run_context(cfg)
    assert not (tmp_path / "outputs").exists()

    managed = _ManagedTestCallback(name="m")
    with pytest.raises(ValueError, match=forbidden_key):
        run_from_config(cfg, raise_exceptions=True, runtime_callbacks=[managed])
    assert managed.enter_readiness == []
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    "forbidden_target",
    [
        "tpen.events.Subscription",
        "tpen.callback.cadence.SubscriptionGroup",
        "tpen.callback.SubscriptionGroup",
    ],
)
def test_forbidden_selector_targets_refused_before_dirs(
    tmp_path: Path, forbidden_target: str
) -> None:
    """Configured Subscription / SubscriptionGroup targets fail preflight with ValueError."""

    cfg = _base_cfg(
        tmp_path,
        callbacks=[
            {
                "_target_": f"{__name__}._OrdinaryCallback",
                "custom": {"_target_": forbidden_target},
            }
        ],
    )
    with pytest.raises(ValueError, match="forbidden"):
        run_from_config(cfg, raise_exceptions=True)
    assert not (tmp_path / "outputs").exists()


def test_preflight_ownership_conflicts_leave_zero_dirs_and_scopes(tmp_path: Path) -> None:
    """Profiler, allocator-reset (including ResourceUsage), writer, and active-reuse conflicts refuse preflight."""

    # 1. Two torch_profiler claims
    m1 = _ManagedTestCallback(
        name="p1", claims=[ResourceOwnershipClaim("torch_profiler", "process")]
    )
    m2 = _ManagedTestCallback(
        name="p2", claims=[ResourceOwnershipClaim("torch_profiler", "process")]
    )
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(_base_cfg(tmp_path / "c1"), raise_exceptions=True, runtime_callbacks=[m1, m2])
    assert m1.enter_readiness == [] and m2.enter_readiness == []
    assert not (tmp_path / "c1" / "outputs").exists()

    # 2. Indexless vs indexed allocator_reset
    a1 = _ManagedTestCallback(name="a1", claims=[ResourceOwnershipClaim("allocator_reset", "cuda")])
    a2 = _ManagedTestCallback(name="a2", claims=[ResourceOwnershipClaim("allocator_reset", "cuda:0")])
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(_base_cfg(tmp_path / "c2"), raise_exceptions=True, runtime_callbacks=[a1, a2])
    assert not (tmp_path / "c2" / "outputs").exists()

    # 3. Managed allocator_reset vs configured legacy ResourceUsage on cuda:0
    cfg_ru_cuda = _base_cfg(
        tmp_path / "c3",
        runtime={"device": "cuda:0", "dtype": "float64", "seed": 0},
        callbacks=[{"_target_": "tpen.callback.ResourceUsage"}],
    )
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(cfg_ru_cuda, raise_exceptions=True, runtime_callbacks=[a1])
    assert a1.enter_readiness == []
    assert not (tmp_path / "c3" / "outputs").exists()

    # 4. Managed allocator_reset vs runtime ResourceUsage with opaque probe on CPU config
    ru_opaque = ResourceUsage(allocator_probe=_OpaqueAllocatorProbe())
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(
            _base_cfg(tmp_path / "c4"),
            raise_exceptions=True,
            runtime_callbacks=[a2, ru_opaque],
        )
    assert a2.enter_readiness == []
    assert not (tmp_path / "c4" / "outputs").exists()

    # 5. Managed allocator_reset on cuda:0 with ResourceUsage on CPU (no probe) succeeds
    cfg_ru_cpu = _base_cfg(
        tmp_path / "c5",
        runtime={"device": "cpu", "dtype": "float64", "seed": 0},
        callbacks=[{"_target_": "tpen.callback.ResourceUsage"}],
    )
    assert run_from_config(cfg_ru_cpu, raise_exceptions=True, runtime_callbacks=[a2]) == 0
    assert len(a2.enter_readiness) == 1

    # 6. Conflicting artifact_writer canonical paths
    w1 = _ManagedTestCallback(
        name="w1",
        claims=[ResourceOwnershipClaim("artifact_writer", str(tmp_path / "shared" / "a.jsonl"))],
    )
    w2 = _ManagedTestCallback(
        name="w2",
        claims=[ResourceOwnershipClaim("artifact_writer", str(tmp_path / "shared" / "." / "a.jsonl"))],
    )
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(_base_cfg(tmp_path / "c6"), raise_exceptions=True, runtime_callbacks=[w1, w2])
    assert not (tmp_path / "c6" / "outputs").exists()

    # 7. Arriving already active & duplicate instance in same invocation
    already_active = _ManagedTestCallback(name="active", arrive_active=True)
    with pytest.raises(ManagedLifecycleError, match="already active"):
        run_from_config(
            _base_cfg(tmp_path / "c7"),
            raise_exceptions=True,
            runtime_callbacks=[already_active],
        )
    assert not (tmp_path / "c7" / "outputs").exists()

    dup = _ManagedTestCallback(name="dup")
    with pytest.raises(ManagedLifecycleError, match="duplicate"):
        run_from_config(
            _base_cfg(tmp_path / "c8"),
            raise_exceptions=True,
            runtime_callbacks=[dup, dup],
        )
    assert not (tmp_path / "c8" / "outputs").exists()

    # 8. Active re-entry via nested open_run_context refused
    nested_cb = _ManagedTestCallback(name="nested")
    with open_run_context(_base_cfg(tmp_path / "c9a"), runtime_callbacks=[nested_cb]):
        with pytest.raises(ManagedLifecycleError, match="already active"):
            with open_run_context(_base_cfg(tmp_path / "c9b"), runtime_callbacks=[nested_cb]):
                pass
    assert not (tmp_path / "c9b" / "outputs").exists()


# ---------------------------------------------------------------------------
# 6. Early context, preparation checkpoints, callback order, and routing
# ---------------------------------------------------------------------------


def test_early_context_contract_checkpoints_and_configured_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Managed scope observes PREPARING with empty extra; ordinary callbacks never see PREPARING."""

    trace: list[str] = []
    _OrdinaryCallback.instances.clear()
    _RecordingLogger.instances.clear()

    class _ProbingManagedCallback(_ManagedTestCallback):
        def run_scope(self, context: RunContext) -> Any:
            @contextmanager
            def _cm() -> Iterator[RunResource]:
                assert context.readiness is RunContextReadiness.PREPARING
                assert context.metadata.extra == {}
                assert context.metadata.status == "initialized"
                with pytest.raises(RunContextReadinessError):
                    context.emit(_PulseEvent())
                with pytest.raises(RunContextReadinessError):
                    context.log({"x": 1.0})
                trace.append("enter:probing")
                handle = _RecordingHandle(name="probing", trace=trace)
                self.handles.append(handle)
                try:
                    yield handle
                finally:
                    assert context.readiness is RunContextReadiness.FINALIZING
                    trace.append("exit:probing")

            return _cm()

    # Register test classes for routed admission too to prove managed sinks route once.
    qualified_probing = f"{_ProbingManagedCallback.__module__}.{_ProbingManagedCallback.__qualname__}"
    qualified_ordinary = f"{_OrdinaryCallback.__module__}.{_OrdinaryCallback.__qualname__}"
    qualified_logger = f"{_RecordingLogger.__module__}.{_RecordingLogger.__qualname__}"
    monkeypatch.setitem(run_mod._ARTIFACT_SINK_FIELDS, qualified_probing, ("output_path",))
    monkeypatch.setattr(
        run_mod,
        "_SINK_FREE_TYPES",
        run_mod._SINK_FREE_TYPES | {qualified_ordinary, qualified_logger},
    )

    dest = tmp_path / "physical_leaf"
    logical_root = tmp_path / "logical"
    runtime_ord = _OrdinaryCallback(name="rt_ord", trace=trace)
    managed_rt = _ProbingManagedCallback(
        name="probing",
        output_path=logical_root / "obs" / "unit" / "run_001" / "managed.out",
        trace=trace,
    )
    cfg = _base_cfg(
        tmp_path,
        run={"root": str(logical_root), "run_id": "run_001", "dir": None},
        callbacks=[
            {"_target_": f"{__name__}._OrdinaryCallback", "name": "cfg_ord_0"},
        ],
        loggers=[
            {"_target_": f"{__name__}._RecordingLogger", "name": "cfg_log_0"},
        ],
    )

    rc = run_from_config(
        cfg,
        raise_exceptions=True,
        artifact_destination=ArtifactDestination(physical_root=dest),
        runtime_callbacks=[runtime_ord, managed_rt],
    )
    assert rc == 0
    # Managed sink was routed exactly once to physical_leaf.
    assert managed_rt.output_path == dest / "managed.out"
    receipt = json.loads((dest / "artifact_routing.json").read_text())
    assert receipt["routed_sinks"][f"{qualified_probing}.output_path"] == str(dest / "managed.out")

    # Ordinary configured callback was constructed AFTER managed scope entered and only saw READY.
    cfg_ord = next(cb for cb in _OrdinaryCallback.instances if cb.name == "cfg_ord_0")
    assert cfg_ord.events == [
        ("RunStarted", RunContextReadiness.READY),
        ("RunCompleted", RunContextReadiness.READY),
    ]
    assert runtime_ord.events == [
        ("RunStarted", RunContextReadiness.READY),
        ("RunCompleted", RunContextReadiness.READY),
    ]
    assert managed_rt.handles[0].checkpoints > 0
    assert managed_rt.handles[0].stop_requests == 1
    assert trace.index("enter:probing") < trace.index("event:rt_ord:RunStarted")
    assert trace.index("request_stop:probing") < trace.index("exit:probing")


# ---------------------------------------------------------------------------
# 7. Partial-enter rollback
# ---------------------------------------------------------------------------


def test_partial_enter_rollback_for_enter_failure_and_invalid_handle(tmp_path: Path) -> None:
    """Partial-enter failure rolls back all earlier scopes without masking primary error."""

    # Case A: M2.__enter__ raises; M1 gets request_stop + __exit__; M2.__exit__ NOT called.
    trace_a: list[str] = []
    m1 = _ManagedTestCallback(name="m1", trace=trace_a)
    m2 = _ManagedTestCallback(name="m2", fail_on_enter=True, trace=trace_a)
    with pytest.raises(RuntimeError, match="enter boom: m2"):
        with open_run_context(_base_cfg(tmp_path / "a"), runtime_callbacks=[m1, m2]):
            pass
    assert trace_a == [
        "reset:m1:1",
        "enter:m1",
        "reset:m2:1",
        "enter:m2",
        "checkpoint:m1",
        "request_stop:m1",
        "exit:m1",
    ]
    assert m1.exit_readiness == [RunContextReadiness.FINALIZING]
    assert m1.is_managed_scope_active() is False

    # Case B: M2.__enter__ succeeds but returns invalid handle -> M2.__exit__ AND M1.__exit__ run LIFO.
    trace_b: list[str] = []
    b1 = _ManagedTestCallback(name="b1", trace=trace_b)
    b2 = _ManagedTestCallback(name="b2", return_invalid_handle=True, trace=trace_b)
    with pytest.raises(TypeError, match="checkpoint"):
        with open_run_context(_base_cfg(tmp_path / "b"), runtime_callbacks=[b1, b2]):
            pass
    assert trace_b.index("request_stop:b1") < trace_b.index("exit:b2") < trace_b.index("exit:b1")

    # Case C: Secondary exit error during rollback does NOT mask primary enter error and marks b1 unsafe.
    trace_c: list[str] = []
    c1 = _ManagedTestCallback(
        name="c1",
        fail_on_exit=True,
        claims=[ResourceOwnershipClaim("torch_profiler", "process")],
        trace=trace_c,
    )
    c2 = _ManagedTestCallback(name="c2", fail_on_enter=True, trace=trace_c)
    with pytest.raises(RuntimeError, match="enter boom: c2"):
        with open_run_context(_base_cfg(tmp_path / "c"), runtime_callbacks=[c1, c2]):
            pass
    # c1 is now marked unsafe and cannot be reused, nor can another profiler claim conflict with it.
    with pytest.raises(ManagedLifecycleError, match="unconfirmed prior cleanup"):
        with open_run_context(_base_cfg(tmp_path / "c_reuse"), runtime_callbacks=[c1]):
            pass
    c_other = _ManagedTestCallback(
        name="c_other",
        claims=[ResourceOwnershipClaim("torch_profiler", "process")],
    )
    with pytest.raises(ResourceOwnershipConflictError, match="unsafe prior claim"):
        with open_run_context(_base_cfg(tmp_path / "c_conflict"), runtime_callbacks=[c_other]):
            pass


# ---------------------------------------------------------------------------
# 8. Preparation failure before READY
# ---------------------------------------------------------------------------


def test_preparation_failure_before_ready_writes_bootstrap_error_without_run_failed(
    tmp_path: Path,
) -> None:
    """Failure during PREPARING writes bootstrap error.json, skips RunFailed, and closes scopes."""

    trace: list[str] = []
    managed = _ManagedTestCallback(name="m", trace=trace)
    cfg = _base_cfg(
        tmp_path,
        callbacks=[
            {"_target_": f"{__name__}._OrdinaryCallback", "name": "ok_ord"},
            {"_target_": f"{__name__}._OrdinaryCallback", "name": "bad_ord", "fail_on_init": True},
        ],
    )
    rc = run_from_config(cfg, config_path="prep.yaml", command="pytest", runtime_callbacks=[managed])
    assert rc == 1

    run_dir = tmp_path / "outputs" / "obs" / "unit" / "run_001"
    error_doc = json.loads((run_dir / "error.json").read_text())
    assert error_doc["phase"] == "bootstrap"
    assert "ordinary init boom: bad_ord" in error_doc["exception_message"]
    assert not (run_dir / "occurrences.jsonl").exists()
    assert not (run_dir / "events.jsonl").exists()
    assert managed.events == []
    assert managed.handles[0].stop_requests == 1
    assert managed.exit_readiness == [RunContextReadiness.FINALIZING]


# ---------------------------------------------------------------------------
# 9. Ready execution, failure/yield precedence, and teardown guarantees
# ---------------------------------------------------------------------------


def test_ready_outcomes_yield_checkpoint_error_and_early_callback_failure(
    tmp_path: Path,
) -> None:
    """Managed path preserves return/raise/yield/publication semantics and always cleans up."""

    # Returned failed runner -> exit code 1, clean close
    m_fail_ret = _ManagedTestCallback(name="fail_ret")
    cfg_ret = _base_cfg(tmp_path / "ret", runner={"_target_": f"{__name__}._ReturnedFailedRunner"})
    assert run_from_config(cfg_ret, runtime_callbacks=[m_fail_ret]) == 1
    assert m_fail_ret.handles[0].stop_requests == 1
    assert len(m_fail_ret.exit_readiness) == 1

    # Planned yield -> exit code 2, run-outcome.json == planned_yield
    m_yield = _ManagedTestCallback(name="yield")
    cfg_yield = _base_cfg(tmp_path / "yield", runner={"_target_": f"{__name__}._YieldingRunner"})
    assert run_from_config(cfg_yield, runtime_callbacks=[m_yield]) == 2
    assert (
        json.loads((tmp_path / "yield" / "outputs" / "obs" / "unit" / "run_001" / "run-outcome.json").read_text())[
            "status"
        ]
        == "planned_yield"
    )
    assert m_yield.handles[0].stop_requests == 1

    # CheckpointPublicationError -> exit code 1, status == unknown
    m_ckpt = _ManagedTestCallback(name="ckpt")
    cfg_ckpt = _base_cfg(
        tmp_path / "ckpt", runner={"_target_": f"{__name__}._CheckpointFailureRunner"}
    )
    assert run_from_config(cfg_ckpt, runtime_callbacks=[m_ckpt]) == 1
    ckpt_outcome = json.loads(
        (tmp_path / "ckpt" / "outputs" / "obs" / "unit" / "run_001" / "run-outcome.json").read_text()
    )
    assert ckpt_outcome["status"] == "unknown"
    assert ckpt_outcome["publication_stage"] == "commit"
    assert m_ckpt.handles[0].stop_requests == 1

    # Earlier callback failure still triggers stop + exit even with probability=0.0
    m_prob0 = _ManagedTestCallback(name="prob0", probability=0.0)
    failing_cb = _OrdinaryCallback(name="boom_start", fail_on_event="RunStarted")
    with pytest.raises(RuntimeError, match="ordinary callback boom on RunStarted"):
        run_from_config(
            _base_cfg(tmp_path / "cb_fail"),
            raise_exceptions=True,
            runtime_callbacks=[failing_cb, m_prob0],
        )
    assert m_prob0.events == []  # never delivered due to failure / probability=0.0
    assert m_prob0.handles[0].stop_requests == 1
    assert len(m_prob0.exit_readiness) == 1


def test_seeding_runner_instantiation_and_event_writer_failures_still_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seeding, runner-instantiation, and event-writer failures all emit/record properly and close scopes."""

    # 1. Seeding failure after READY: error.json phase=bootstrap, RunFailed emitted, scope closed
    m_seed = _ManagedTestCallback(name="seed_fail")
    cfg_seed = _base_cfg(
        tmp_path / "seed",
        runtime={"device": "cpu", "dtype": "float64", "seed": "not-an-int"},
    )
    assert run_from_config(cfg_seed, runtime_callbacks=[m_seed]) == 1
    seed_dir = tmp_path / "seed" / "outputs" / "obs" / "unit" / "run_001"
    assert json.loads((seed_dir / "error.json").read_text())["phase"] == "bootstrap"
    assert [name for name, _ in m_seed.events] == ["RunFailed"]
    assert m_seed.handles[0].stop_requests == 1
    assert len(m_seed.exit_readiness) == 1

    # 2. Runner instantiation failure after RunStarted: error.json phase=bootstrap, RunFailed emitted, scope closed
    m_runner = _ManagedTestCallback(name="runner_fail")
    cfg_runner = _base_cfg(tmp_path / "runner_bad", runner={"_target_": "builtins.object"})
    assert run_from_config(cfg_runner, runtime_callbacks=[m_runner]) == 1
    runner_dir = tmp_path / "runner_bad" / "outputs" / "obs" / "unit" / "run_001"
    assert json.loads((runner_dir / "error.json").read_text())["phase"] == "bootstrap"
    assert [name for name, _ in m_runner.events] == ["RunStarted", "RunFailed"]
    assert m_runner.handles[0].stop_requests == 1
    assert len(m_runner.exit_readiness) == 1

    # 3. Event-writer failure on RunStarted still runs request_stop + __exit__
    m_writer = _ManagedTestCallback(name="writer_fail")
    monkeypatch.setattr(
        artifacts_mod,
        "write_occurrence_artifact",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("event disk full")),
    )
    with pytest.raises(OSError, match="event disk full"):
        run_from_config(
            _base_cfg(tmp_path / "writer"),
            raise_exceptions=True,
            runtime_callbacks=[m_writer],
        )
    assert m_writer.handles[0].stop_requests == 1
    assert len(m_writer.exit_readiness) == 1


def test_teardown_ordering_no_suppression_and_cleanup_error_precedence(
    tmp_path: Path,
) -> None:
    """ALL request_stop run before ANY __exit__; __exit__ cannot suppress; error precedence holds."""

    # 1. ALL request_stop before ANY __exit__ (LIFO), __exit__ returning True cannot suppress primary error,
    #    and multiple cleanup failures still attempt every logger/stop/exit.
    trace: list[str] = []
    m1 = _ManagedTestCallback(
        name="m1", suppress_in_exit=True, fail_on_stop=True, trace=trace
    )
    m2 = _ManagedTestCallback(
        name="m2", suppress_in_exit=True, fail_on_exit=True, trace=trace
    )
    cfg = _base_cfg(
        tmp_path / "multi",
        runner={"_target_": f"{__name__}._RaisingRunner"},
        loggers=[
            {"_target_": f"{__name__}._RecordingLogger", "name": "l1", "fail_on_finish": True},
            {"_target_": f"{__name__}._RecordingLogger", "name": "l2"},
        ],
    )
    _RecordingLogger.instances.clear()
    with pytest.raises(RuntimeError, match="runner exploded"):
        run_from_config(cfg, raise_exceptions=True, runtime_callbacks=[m1, m2])

    # Both loggers were finished despite l1 failing.
    assert [lg.finished for lg in _RecordingLogger.instances] == [1, 1]
    # Both request_stop calls precede both __exit__ calls, and exits are reverse LIFO (m2 then m1).
    stop_indices = [trace.index("request_stop:m1"), trace.index("request_stop:m2")]
    exit_indices = [trace.index("exit:m2"), trace.index("exit:m1")]
    assert max(stop_indices) < min(exit_indices)
    assert exit_indices[0] < exit_indices[1]

    # 2. Successful application + mandatory cleanup failure -> fails invocation, preserves run-outcome.json!
    m_clean_fail = _ManagedTestCallback(name="clean_fail", fail_on_exit=True)
    cfg_ok = _base_cfg(tmp_path / "ok_clean_fail")
    assert run_from_config(cfg_ok, raise_exceptions=False, runtime_callbacks=[m_clean_fail]) == 1
    outcome_ok = json.loads(
        (tmp_path / "ok_clean_fail" / "outputs" / "obs" / "unit" / "run_001" / "run-outcome.json").read_text()
    )
    assert outcome_ok["status"] == "completed"

    # 3. Yielded application + cleanup failure -> fails invocation (returns 1 / raises), preserves planned_yield!
    m_yield_clean_fail = _ManagedTestCallback(name="yield_clean_fail", fail_on_exit=True)
    cfg_yield = _base_cfg(
        tmp_path / "yield_clean_fail",
        runner={"_target_": f"{__name__}._YieldingRunner"},
    )
    with pytest.raises(RuntimeError, match="exit boom: yield_clean_fail"):
        run_from_config(cfg_yield, raise_exceptions=True, runtime_callbacks=[m_yield_clean_fail])
    outcome_yield = json.loads(
        (tmp_path / "yield_clean_fail" / "outputs" / "obs" / "unit" / "run_001" / "run-outcome.json").read_text()
    )
    assert outcome_yield["status"] == "planned_yield"

    # 4. Unconfirmed quiescence (is_managed_scope_active() stays True after __exit__) fails & marks unsafe
    m_unconfirmed = _ManagedTestCallback(name="unconfirmed", leave_active_on_exit=True)
    with pytest.raises(ManagedLifecycleError, match="failed to confirm quiescence"):
        run_from_config(
            _base_cfg(tmp_path / "unconfirmed"),
            raise_exceptions=True,
            runtime_callbacks=[m_unconfirmed],
        )
    with pytest.raises(ManagedLifecycleError, match="unconfirmed prior cleanup"):
        run_from_config(
            _base_cfg(tmp_path / "unconfirmed2"),
            raise_exceptions=True,
            runtime_callbacks=[m_unconfirmed],
        )


def test_keyboard_interrupt_runs_managed_cleanup_and_propagates(tmp_path: Path) -> None:
    """BaseException (KeyboardInterrupt) still stops/closes scopes and propagates untouched."""

    managed = _ManagedTestCallback(name="kb", suppress_in_exit=True)
    cfg = _base_cfg(tmp_path, runner={"_target_": f"{__name__}._KeyboardInterruptRunner"})
    with pytest.raises(KeyboardInterrupt, match="user interrupt"):
        run_from_config(cfg, raise_exceptions=False, runtime_callbacks=[managed])
    assert managed.handles[0].stop_requests == 1
    assert managed.exit_exc_types == [KeyboardInterrupt]


# ---------------------------------------------------------------------------
# 10. First typed-context binding & repeated invocations
# ---------------------------------------------------------------------------


def test_first_typed_context_binding_preserves_live_state_and_refreshes_on_reuse(
    tmp_path: Path,
) -> None:
    """_reset_typed_state runs before run_scope, NOT wiped by first RunStarted, refreshed on run 2."""

    trace: list[str] = []
    managed = _ManagedTestCallback(name="reusable", trace=trace)

    assert run_from_config(_base_cfg(tmp_path / "r1"), runtime_callbacks=[managed]) == 0
    assert managed.reset_count == 1
    assert managed.events == [
        ("RunStarted", "active-1"),
        ("RunCompleted", "active-1"),
    ]

    assert run_from_config(_base_cfg(tmp_path / "r2"), runtime_callbacks=[managed]) == 0
    assert managed.reset_count == 2
    assert managed.events[2:] == [
        ("RunStarted", "active-2"),
        ("RunCompleted", "active-2"),
    ]
    assert trace.index("reset:reusable:1") < trace.index("enter:reusable")
    assert trace.index("exit:reusable") < trace.index("reset:reusable:2")


# ---------------------------------------------------------------------------
# 11. Programmatic APIs: open_run_context and prepare_run_context refusal
# ---------------------------------------------------------------------------


def test_open_run_context_managed_and_unmanaged_and_prepare_refusal(tmp_path: Path) -> None:
    """open_run_context guarantees cleanup; prepare_run_context refuses managed callbacks early."""

    # Unscoped prepare_run_context refuses both runtime and configured managed callbacks before make_dirs()
    managed = _ManagedTestCallback(name="m")
    with pytest.raises(ManagedLifecycleError, match="open_run_context"):
        prepare_run_context(_base_cfg(tmp_path / "refuse_rt"), runtime_callbacks=[managed])
    assert not (tmp_path / "refuse_rt" / "outputs").exists()

    cfg_managed = _base_cfg(
        tmp_path / "refuse_cfg",
        callbacks=[{"_target_": f"{__name__}._ManagedTestCallback", "name": "cfg_m"}],
    )
    with pytest.raises(ManagedLifecycleError, match="open_run_context"):
        prepare_run_context(cfg_managed)
    assert not (tmp_path / "refuse_cfg" / "outputs").exists()

    # open_run_context with managed callback: normal exit, exception, and KeyboardInterrupt
    with open_run_context(_base_cfg(tmp_path / "open_ok"), runtime_callbacks=[managed]) as ctx:
        assert ctx.readiness is RunContextReadiness.READY
        assert managed.enter_readiness[-1] is RunContextReadiness.PREPARING
    assert ctx.readiness is RunContextReadiness.CLOSED
    assert managed.handles[-1].stop_requests == 1

    with pytest.raises(ValueError, match="body error"):
        with open_run_context(_base_cfg(tmp_path / "open_err"), runtime_callbacks=[managed]) as ctx_err:
            assert ctx_err.readiness is RunContextReadiness.READY
            raise ValueError("body error")
    assert ctx_err.readiness is RunContextReadiness.CLOSED
    assert managed.exit_exc_types[-1] is ValueError

    with pytest.raises(KeyboardInterrupt, match="ctrl-c"):
        with open_run_context(_base_cfg(tmp_path / "open_kb"), runtime_callbacks=[managed]) as ctx_kb:
            raise KeyboardInterrupt("ctrl-c")
    assert ctx_kb.readiness is RunContextReadiness.CLOSED
    assert managed.exit_exc_types[-1] is KeyboardInterrupt

    # open_run_context WITHOUT managed callbacks validates/appends runtime_callbacks and closes loggers
    _RecordingLogger.instances.clear()
    ord_cb = _OrdinaryCallback(name="rt_plain")
    cfg_plain = _base_cfg(
        tmp_path / "open_plain",
        loggers=[{"_target_": f"{__name__}._RecordingLogger", "name": "plain_log"}],
    )
    with open_run_context(cfg_plain, runtime_callbacks=[ord_cb]) as ctx_plain:
        assert ctx_plain.readiness is RunContextReadiness.READY
        assert ord_cb in ctx_plain.callbacks
    assert ctx_plain.readiness is RunContextReadiness.CLOSED
    assert _RecordingLogger.instances[0].finished == 1


# ---------------------------------------------------------------------------
# 12. T2 review discriminating tests (Findings 1-6)
# ---------------------------------------------------------------------------


def test_unsafe_callback_gc_during_registry_lock_does_not_deadlock(tmp_path: Path) -> None:
    """Collecting an unsafe callback weakref during registry lock acquisition must not deadlock."""

    _ManagedTestCallback.instances.clear()
    rc1 = run_from_config(
        _base_cfg(
            tmp_path / "r1",
            callbacks=[
                {
                    "_target_": f"{__name__}._ManagedTestCallback",
                    "name": "cfg_unsafe",
                    "fail_on_exit": True,
                }
            ],
        ),
        raise_exceptions=False,
    )
    assert rc1 == 1
    assert len(_ManagedTestCallback.instances) == 1

    old_threshold = gc.get_threshold()
    was_enabled = gc.isenabled()
    result: list[int] = []
    errors: list[BaseException] = []

    def _run_second() -> None:
        try:
            gc.set_threshold(1, 1, 1)
            rc2 = run_from_config(
                _base_cfg(tmp_path / "r2"),
                runtime_callbacks=[_GcProbingManagedCallback(name="ok")],
            )
            result.append(rc2)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            gc.set_threshold(*old_threshold)

    # Pause automatic GC before dropping the last external strong reference so
    # the cfg_unsafe <-> RunContext cycle stays alive until cyclic GC runs
    # while _MANAGED_LIFECYCLE_REGISTRY._lock is held in Run 2.
    gc.disable()
    try:
        _ManagedTestCallback.instances.clear()
        thread = threading.Thread(target=_run_second, daemon=True)
        thread.start()
        thread.join(timeout=3.0)
    finally:
        gc.set_threshold(*old_threshold)
        if was_enabled:
            gc.enable()

    assert not thread.is_alive(), (
        "Run 2 deadlocked inside _MANAGED_LIFECYCLE_REGISTRY._lock while collecting "
        "an unsafe callback weakref during cyclic GC"
    )
    assert errors == [], f"unexpected exception in worker thread: {errors!r}"
    assert result == [0]


def test_interpolated_callback_list_entry_uses_managed_session(tmp_path: Path) -> None:
    """Interpolated callback list entries (callbacks: ['${obs_cb}']) must enter managed session."""

    _ManagedTestCallback.instances.clear()
    cfg = _base_cfg(
        tmp_path / "run",
        obs_cb={
            "_target_": f"{__name__}._ManagedTestCallback",
            "name": "interp_cb",
        },
        callbacks=["${obs_cb}"],
    )
    rc = run_from_config(cfg, raise_exceptions=True)
    assert rc == 0
    assert len(_ManagedTestCallback.instances) == 1
    managed = _ManagedTestCallback.instances[0]
    assert managed.name == "interp_cb"
    assert managed.enter_readiness == [RunContextReadiness.PREPARING]
    assert [name for name, _ in managed.events] == ["RunStarted", "RunCompleted"]
    assert len(managed.handles) == 1
    assert managed.handles[0].stop_requests == 1
    assert managed.exit_readiness == [RunContextReadiness.FINALIZING]

    _ManagedTestCallback.instances.clear()
    cfg2 = _base_cfg(
        tmp_path / "open",
        obs_cb={
            "_target_": f"{__name__}._ManagedTestCallback",
            "name": "interp_open",
        },
        callbacks=["${obs_cb}"],
    )
    with open_run_context(cfg2) as ctx:
        assert ctx.readiness is RunContextReadiness.READY
        assert len(_ManagedTestCallback.instances) == 1
        managed2 = _ManagedTestCallback.instances[0]
        assert managed2.name == "interp_open"
        assert managed2.enter_readiness == [RunContextReadiness.PREPARING]
        assert managed2.is_managed_scope_active() is True
    assert ctx.readiness is RunContextReadiness.CLOSED
    assert managed2.handles[0].stop_requests == 1
    assert managed2.exit_readiness == [RunContextReadiness.FINALIZING]
    assert managed2.is_managed_scope_active() is False


def test_factory_target_returning_managed_callback_refused_by_prepare_run_context(
    tmp_path: Path,
) -> None:
    """Factory function _target_ returning a ManagedCallback must be refused by prepare_run_context."""

    cfg = _base_cfg(
        tmp_path,
        callbacks=[
            {
                "_target_": f"{__name__}._build_managed_callback",
                "name": "fac",
            }
        ],
    )
    with pytest.raises(ManagedLifecycleError):
        prepare_run_context(cfg)


def test_routed_preflight_refusal_does_not_mutate_active_or_conflicting_runtime_managed_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routed preflight refusal must not mutate sink fields on active or conflicting callbacks."""

    qualified_managed = f"{_ManagedTestCallback.__module__}.{_ManagedTestCallback.__qualname__}"
    monkeypatch.setitem(run_mod._ARTIFACT_SINK_FIELDS, qualified_managed, ("output_path",))

    logical_root = tmp_path / "logical"
    cfg = _base_cfg(
        tmp_path,
        run={"root": str(logical_root), "run_id": "run_001", "dir": None},
    )
    logical_out = logical_root / "obs" / "unit" / "run_001" / "m.out"

    # Case A: active re-entry refusal must not mutate output_path of live callback.
    m = _ManagedTestCallback(name="live", output_path=logical_out)
    with open_run_context(cfg, runtime_callbacks=[m]):
        with pytest.raises(ManagedLifecycleError, match="already active"):
            with open_run_context(
                cfg,
                artifact_destination=ArtifactDestination(physical_root=tmp_path / "dest_nested"),
                runtime_callbacks=[m],
            ):
                pass
        assert m.output_path == logical_out

    # Case B: ownership conflict refusal must not mutate output_path before acquire().
    m1 = _ManagedTestCallback(
        name="p1",
        output_path=logical_out,
        claims=[ResourceOwnershipClaim("torch_profiler", "process")],
    )
    m2 = _ManagedTestCallback(
        name="p2",
        claims=[ResourceOwnershipClaim("torch_profiler", "process")],
    )
    with pytest.raises(ResourceOwnershipConflictError):
        run_from_config(
            cfg,
            raise_exceptions=True,
            artifact_destination=ArtifactDestination(physical_root=tmp_path / "dest_conflict"),
            runtime_callbacks=[m1, m2],
        )
    assert m1.output_path == logical_out

    dest_ok = tmp_path / "dest_ok"
    rc = run_from_config(
        cfg,
        raise_exceptions=True,
        artifact_destination=ArtifactDestination(physical_root=dest_ok),
        runtime_callbacks=[m1],
    )
    assert rc == 0
    assert m1.output_path == dest_ok / "m.out"


def test_failed_enter_leaving_callback_active_retains_unsafe_ownership_claims(
    tmp_path: Path,
) -> None:
    """Failed __enter__ leaving is_managed_scope_active() True must retain unsafe claims."""

    m1 = _ManagedTestCallback(
        name="half_enter",
        fail_on_enter=True,
        leave_active_on_enter_failure=True,
        claims=[ResourceOwnershipClaim("torch_profiler", "process")],
    )
    with pytest.raises(RuntimeError, match="enter boom: half_enter"):
        with open_run_context(_base_cfg(tmp_path / "r1"), runtime_callbacks=[m1]):
            pass
    assert m1.is_managed_scope_active() is True

    m2 = _ManagedTestCallback(
        name="second_prof",
        claims=[ResourceOwnershipClaim("torch_profiler", "process")],
    )
    with pytest.raises(ResourceOwnershipConflictError, match="unsafe prior claim"):
        with open_run_context(_base_cfg(tmp_path / "r2"), runtime_callbacks=[m2]):
            pass


def test_preflight_destination_freshness_and_ordinary_target_import_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Destination freshness precedes target imports, and ordinary targets wait for managed scope."""

    # Case A: non-fresh destination must raise ArtifactRoutingError before importing callback targets.
    dest = tmp_path / "non_fresh"
    dest.mkdir()
    (dest / "stale.txt").write_text("old", encoding="utf-8")
    cfg_a = _base_cfg(
        tmp_path / "a",
        callbacks=[{"_target_": f"{__name__}._BadScopeAttrCallback"}],
    )
    with pytest.raises(ArtifactRoutingError):
        run_from_config(
            cfg_a,
            raise_exceptions=True,
            artifact_destination=ArtifactDestination(physical_root=dest),
        )
    with pytest.raises(ArtifactRoutingError):
        with open_run_context(
            cfg_a,
            artifact_destination=ArtifactDestination(physical_root=dest),
        ):
            pass

    # Case B: shipped ordinary callback targets must not be imported via get_class before run_scope enters.
    real_get_class = run_mod.get_class
    trace: list[str] = []

    def _spy_get_class(path: str) -> Any:
        trace.append(f"get_class:{path}")
        return real_get_class(path)

    monkeypatch.setattr(run_mod, "get_class", _spy_get_class)
    managed = _ManagedTestCallback(name="m", trace=trace)
    with open_run_context(
        _base_cfg(
            tmp_path / "b",
            callbacks=[
                {
                    "_target_": "tpen.callback.Status",
                    "output_path": "${run.dir}/status.json",
                }
            ],
        ),
        runtime_callbacks=[managed],
    ):
        pass
    enter_idx = trace.index("enter:m")
    status_indices = [
        idx for idx, item in enumerate(trace) if item == "get_class:tpen.callback.Status"
    ]
    assert not any(idx < enter_idx for idx in status_indices), (
        f"ordinary callback target imported before managed scope entered: {trace}"
    )


def test_secondary_cleanup_errors_logged_in_open_run_context_and_on_keyboard_interrupt(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Secondary cleanup errors must be logged on tpen.bootstrap in open_run_context and on KeyboardInterrupt."""

    bootstrap_logger = logging.getLogger("tpen.bootstrap")
    bootstrap_logger.addHandler(caplog.handler)
    try:
        # Case A: open_run_context body raises ValueError while managed __exit__ raises.
        with caplog.at_level(logging.ERROR, logger="tpen.bootstrap"):
            with pytest.raises(ValueError, match="body boom"):
                with open_run_context(
                    _base_cfg(tmp_path / "open"),
                    runtime_callbacks=[_ManagedTestCallback(name="m_open", fail_on_exit=True)],
                ):
                    raise ValueError("body boom")
            assert any("exit boom: m_open" in rec.getMessage() for rec in caplog.records), (
                f"expected secondary cleanup error in tpen.bootstrap logs, got "
                f"{[rec.getMessage() for rec in caplog.records]!r}"
            )

        # Case B: run_from_config runner raises KeyboardInterrupt while managed __exit__ raises.
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="tpen.bootstrap"):
            with pytest.raises(KeyboardInterrupt, match="user interrupt"):
                run_from_config(
                    _base_cfg(
                        tmp_path / "kb",
                        runner={"_target_": f"{__name__}._KeyboardInterruptRunner"},
                    ),
                    runtime_callbacks=[_ManagedTestCallback(name="m_kb", fail_on_exit=True)],
                )
            assert any("exit boom: m_kb" in rec.getMessage() for rec in caplog.records), (
                f"expected secondary cleanup error on KeyboardInterrupt in tpen.bootstrap logs, "
                f"got {[rec.getMessage() for rec in caplog.records]!r}"
            )
    finally:
        bootstrap_logger.removeHandler(caplog.handler)


@pytest.mark.parametrize(
    "forbidden_target",
    [
        "tpen.events.Subscription.of",
        "tpen.events.Subscription.started",
        "tpen.events.Subscription.ended",
        "tpen.events.started",
        "tpen.events.ended",
        "tpen.callback.SubscriptionGroup.from_config",
    ],
)
def test_forbidden_selector_factory_targets_refused_before_dirs(
    tmp_path: Path, forbidden_target: str
) -> None:
    """Subscription factory callables and helper targets fail preflight leaving zero dirs."""

    cfg = _base_cfg(
        tmp_path / "direct",
        callbacks=[
            {
                "_target_": f"{__name__}._OrdinaryCallback",
                "custom": {"_target_": forbidden_target},
            }
        ],
    )
    with pytest.raises(ValueError, match="forbidden"):
        prepare_run_context(cfg)
    assert not (tmp_path / "direct" / "outputs").exists()

    with pytest.raises(ValueError, match="forbidden"):
        run_from_config(cfg, raise_exceptions=True)
    assert not (tmp_path / "direct" / "outputs").exists()

    cfg_interp = _base_cfg(
        tmp_path / "interp",
        bad_cb={
            "_target_": f"{__name__}._OrdinaryCallback",
            "custom": {"_target_": forbidden_target},
        },
        callbacks=["${bad_cb}"],
    )
    with pytest.raises(ValueError, match="forbidden"):
        run_from_config(cfg_interp, raise_exceptions=True)
    assert not (tmp_path / "interp" / "outputs").exists()


def test_known_non_managed_callback_targets_consistent_with_callback_exports() -> None:
    """Every shipped non-managed callback export and defining path is registered cleanly."""

    import tpen.callback as callback_pkg
    from tpen.callback import _KNOWN_NON_MANAGED_CALLBACK_TARGETS
    from tpen.callback.base import _CallbackCore, _is_managed_callback_class

    base_classes = {Callback, StatefulCallback}
    concrete_exports: list[tuple[str, type]] = []
    for name in callback_pkg.__all__:
        obj = getattr(callback_pkg, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, _CallbackCore)
            and obj not in base_classes
        ):
            concrete_exports.append((name, obj))

    assert concrete_exports, "expected shipped concrete callbacks in tpen.callback.__all__"
    for name, cls in concrete_exports:
        if _is_managed_callback_class(cls):
            assert f"tpen.callback.{name}" not in _KNOWN_NON_MANAGED_CALLBACK_TARGETS
            assert f"{cls.__module__}.{cls.__qualname__}" not in _KNOWN_NON_MANAGED_CALLBACK_TARGETS
        else:
            assert f"tpen.callback.{name}" in _KNOWN_NON_MANAGED_CALLBACK_TARGETS
            assert f"{cls.__module__}.{cls.__qualname__}" in _KNOWN_NON_MANAGED_CALLBACK_TARGETS

    for target in _KNOWN_NON_MANAGED_CALLBACK_TARGETS:
        resolved = run_mod.get_class(target)
        assert isinstance(resolved, type) and issubclass(resolved, _CallbackCore)
        assert not _is_managed_callback_class(resolved)
