"""Command-line entrypoint for configured TPEN runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import logging
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from hydra.utils import get_class, instantiate
from hydra.errors import InstantiationException
from omegaconf import DictConfig, ListConfig, OmegaConf

from tpen.accelerator import TorchAllocatorPeakProbe, seed_all as accelerator_seed_all
from tpen.artifacts import (
    ArtifactDestination,
    ArtifactManager,
    ArtifactRoutingError,
    RunContext,
    RunResult,
    build_run_metadata,
    resolve_run_clock,
    resolve_run_id,
    write_artifact_routing_receipt,
    write_error_artifact,
    write_run_outcome_artifact,
    write_run_start_artifact,
)
from tpen.distributed import (
    ExecutionTopology,
    RankLocalJSONLWriter,
    execution_topology_from_facts,
)
from tpen.callback import configure_terminal_logging
from tpen.checkpoint.hashing import populate_resolved_config_defaults
from tpen.config import register_resolvers
from tpen.dependencies import OptionalDependencyError, require_torch
from tpen.events import Event as TypedEvent
from tpen.hi_schema import validate_hi_train_config
from tpen.run_events import RunCompleted, RunFailed, RunStarted
from tpen.runner import Runner
from tpen.callback.planned_yield import PlannedYield, PlannedYieldSignal
from tpen.callback.checkpoint import CheckpointPublicationError

# Register custom OmegaConf resolvers (e.g. tpen.basis_feature_dim) before any
# config is loaded or resolved on the run path.
register_resolvers()

PLANNED_YIELD_EXIT_CODE = 2

# Bootstrap diagnostics get their own channel so a run's logging configuration can
# silence or route fatal pre-context errors independently. Spelled once: the channel
# name is read by tests/conftest.py logger isolation, so a partial rename would
# silently disable that isolation rather than fail loudly.
_BOOTSTRAP_LOGGER_NAME = "tpen.bootstrap"


def main(argv: Sequence[str] | None = None) -> int:
    """Run one configured TPEN runner from the command line."""

    _install_bootstrap_stderr_logger()
    args = parse_args(argv)
    command = " ".join(["run.py", *(sys.argv[1:] if argv is None else argv)])
    try:
        cfg = load_config(str(args.config), args.overrides)
    except Exception as exc:
        _print_fatal(
            exc,
            phase="bootstrap",
            traceback_text=traceback.format_exc(),
            command=command,
            config_path=str(args.config),
        )
        return 1
    try:
        _preflight_optional_dependencies(cfg)
    except OptionalDependencyError as exc:
        _print_fatal(exc, phase="bootstrap", command=command, config_path=str(args.config))
        return 1
    return run_from_config(cfg, config_path=str(args.config), command=command)


@dataclass
class _BootstrapState:
    run_dir: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse configured-run command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return parser.parse_args(argv)


def load_config(config_path: str, overrides: Sequence[str] | None = None) -> DictConfig:
    """Load a YAML config and apply OmegaConf dotlist overrides."""

    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def _nonempty_text(value: object) -> str | None:
    """Normalize one configured durable name, preserving only usable text."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def prepare_run_context(
    cfg: DictConfig,
    *,
    config_path: str | None = None,
    command: str | None = None,
    bootstrap: _BootstrapState | None = None,
    topology: ExecutionTopology | None = None,
    artifact_destination: ArtifactDestination | None = None,
    runtime_callbacks: Sequence[object] = (),
) -> RunContext:
    """Resolve run metadata, artifact paths, callbacks, and loggers.

    Callbacks and loggers are configured at the config root and owned by the
    `RunContext`; runners dispatch into ``context.callbacks`` and log through
    ``context.log``.

    Parameters
    ----------
    artifact_destination : tpen.artifacts.ArtifactDestination or None, optional
        Operational, launcher-supplied physical leaf this run's artifact
        sinks must write beneath, outside ``cfg``. When ``None`` (the
        default), sinks write under the logical ``run.root``/``run.dir`` tree
        exactly as before. When supplied, the recorded ``run.dir`` and every
        sink a known operational-slot registry covers are routed to the
        claimed physical leaf while the resolved config `cfg` records and
        other callers see stays logically unchanged -- see
        :func:`_route_artifact_sinks`.
    runtime_callbacks : sequence of objects, optional
        Operational observers. In routed mode (`artifact_destination`
        supplied), they are admitted, routed, validated, and appended into
        ``context.callbacks`` here, before this context is returned, so a
        launcher-supplied observer is bound before `RunStarted` is ever
        emitted (F4) -- see :func:`run_from_config`, whose docstring
        describes this argument's purpose. Without a destination, they are
        NOT appended here (review-r3 R3.4, restoring the original legacy
        ordering): `run_from_config` validates and appends them itself,
        after this context already exists, so a failing legacy list appends
        none of them rather than all of them regardless of outcome.

    Raises
    ------
    tpen.config_schema.ClosedSchemaError
        When `cfg` declares a closed schema and violates it. Raised before any
        directory, logger or callback exists.
    """

    # PRECONSTRUCTION FIREWALL, ENTRY-POINT-INDEPENDENT COPY. `run_from_config`
    # runs the same check, but a check that lives ONLY there is a property of
    # one entry point rather than of the construction path: a programmatic
    # caller reaching `prepare_run_context` directly -- a script, a notebook, a
    # test harness, another runner -- would walk straight past it and still get
    # a run directory, every configured logger and callback, and a run-start
    # artifact on disk. Everything this function constructs happens below this
    # line, so a refusal here leaves nothing behind.
    #
    # The duplication is deliberate and cheap: `validate_hi_train_config` is a
    # pure function of `cfg` with no side effects, so running it twice on the
    # `run_from_config` path costs one extra sweep and removes the possibility
    # that the two paths ever disagree about what is admissible.
    validate_hi_train_config(cfg)
    # (A) review-r2: a routed destination must be fresh -- absent, or a real,
    # empty, non-symlinked directory -- checked before anything below
    # constructs a directory, a logger, or a callback (R2.1, R2.3).
    if artifact_destination is not None:
        artifact_destination.require_fresh()

    run_clock = resolve_run_clock(cfg)
    source_cfg = _rerunnable_config(
        OmegaConf.create(populate_resolved_config_defaults(cfg, timezone=run_clock.timezone))
    )
    resolved_cfg = OmegaConf.create(populate_resolved_config_defaults(cfg, timezone=run_clock.timezone))
    experiment_name = _nonempty_text(OmegaConf.select(resolved_cfg, "experiment.name"))
    run_name = (
        _nonempty_text(OmegaConf.select(resolved_cfg, "experiment.run_name"))
        or experiment_name
        or "tpen_run"
    )
    OmegaConf.update(resolved_cfg, "experiment.run_name", run_name, merge=False, force_add=True)
    # Run identity is AGREED wherever a launch declares itself, rather than
    # generated per process. `generate_run_id` is process-local, so a
    # distributed launch derived a different id on every rank and wrote to its
    # own artifact root instead of one converged run root.
    #
    # HONEST SCOPE, because the comment would otherwise overclaim: nothing in
    # `tpen/` yet initializes a process group or supplies a multi-rank
    # `ExecutionTopology`, so on the launches TPEN can perform TODAY this still
    # resolves through the serial branch. The agreement is the mechanism the
    # distributed runtime will use, tested against real multi-rank process
    # groups; it is not yet reached by a production caller.
    #
    # `resolve_run_id` is reached unconditionally, including when the id is
    # already explicit, and that is load-bearing rather than tidiness: a
    # resolution entered only on the null branch would leave a rank whose config
    # carries an explicit id outside the collective, so its peers would block
    # until the process-group timeout. Reaching it on every path also means the
    # ranks' explicit ids are checked for agreement instead of being trusted.
    #
    # It runs here, above every construction below, so a disagreeing launch is
    # refused while the run directory still does not exist.
    configured_run_id = OmegaConf.select(resolved_cfg, "run.run_id", default=None)
    run_id = resolve_run_id(configured_run_id, run_name, clock=run_clock, topology=topology)
    if configured_run_id is None:
        OmegaConf.update(resolved_cfg, "run.run_id", run_id, merge=False, force_add=True)
    experiment_name = experiment_name or "experiment"
    sector = str(OmegaConf.select(resolved_cfg, "experiment.sector", default="default"))
    root = Path(str(OmegaConf.select(resolved_cfg, "run.root", default="outputs")))
    layout = str(OmegaConf.select(resolved_cfg, "run.layout", default="nested"))
    # The ADMITTED logical `run.dir`, read before anything below overwrites
    # it. In routed mode only, this -- not a directory recomputed from
    # root/experiment/sector/run_id -- is what every eagerly-resolved HI sink
    # was already baked against (F1): HI can resolve a layout whose directory
    # does not match what `root`/`experiment`/`sector`/`run_id` alone would
    # recompute (for example an intermediate path segment under a `flat`
    # layout). Legacy (no destination) behavior is unaffected: it always
    # recomputes, exactly as before -- and (D) review-r2 R2.5: the read
    # itself is gated on routed mode, so a legacy caller's unresolved,
    # discarded `run.dir` placeholder never has to resolve at all.
    admitted_run_dir = (
        _nonempty_text(OmegaConf.select(resolved_cfg, "run.dir", default=None))
        if artifact_destination is not None
        else None
    )
    artifact_manager = ArtifactManager(
        root,
        experiment_name,
        sector,
        str(run_id),
        layout=layout,
        destination=artifact_destination,
        logical_run_dir=(
            Path(admitted_run_dir)
            if artifact_destination is not None and admitted_run_dir is not None
            else None
        ),
    )
    if bootstrap is not None:
        bootstrap.run_dir = artifact_manager.run_dir
    # `run.dir` is recorded as the LOGICAL path, never the physical one, even
    # when `artifact_destination` is supplied: several allocations of one
    # scientific cell must record the same constant C. `artifact_manager.run_dir`
    # (physical when a destination is bound) is reserved for actual disk writes
    # below -- see `_route_artifact_sinks` for the already-instantiated sinks
    # whose output paths were baked from this same logical value before this
    # function ever ran (HI resolves its config eagerly).
    logical_run_dir = artifact_manager.logical_run_dir
    OmegaConf.update(resolved_cfg, "run.dir", str(logical_run_dir), merge=False, force_add=True)
    OmegaConf.resolve(resolved_cfg)
    artifact_manager.make_dirs()
    _configure_terminal_logging(resolved_cfg)

    routed = artifact_destination is not None
    # (B) review-r2 R2.2: in routed mode, each configured sink's `_target_`
    # class is resolved and admitted BEFORE `instantiate()` ever constructs
    # it -- an unregistered writer or subclass can no longer write to its
    # logical destination before a later, construction-side probe refuses
    # it. Legacy (no destination) construction is unchanged.
    loggers = _instantiate_sequence(
        OmegaConf.select(resolved_cfg, "loggers", default=[]), routed=routed
    )
    callbacks = _instantiate_sequence(
        OmegaConf.select(resolved_cfg, "callbacks", default=[]), routed=routed
    )
    # CONFIGURED validation is unconditional and happens here, before this
    # context (or anything else below) is constructed (review-r3 R3.4,
    # restoring the original pre-S3 legacy ordering): fail-loud interface
    # validation only, confirming these objects expose the lifecycle methods
    # without invoking any behavior (no handle/log/finish). A refusal here
    # leaves no `RunContext` bound at all, so `run_from_config`'s `finally`
    # teardown (which calls `logger.finish()` on every context logger) can
    # never reach an invalid configured logger -- gating this validation on
    # `routed`, or deferring it until after the context existed, let exactly
    # that AttributeError mask the original, already-handled refusal.
    _validate_callbacks(callbacks)
    _validate_loggers(loggers)
    runtime_callbacks_list = list(runtime_callbacks)
    if routed:
        # Runtime callbacks have no `_target_` to resolve ahead of
        # construction -- they already exist -- so admission checks their
        # exact type instead (F3/F4). In routed mode only, runtime callbacks
        # must also be validated and merged here, strictly before
        # `RunStarted` (F4): legacy (no-destination) runtime validation and
        # merging is restored to `run_from_config` instead (E/R3.4) -- see
        # its docstring -- so a legacy runtime-callback failure appends none
        # of them, rather than merging first and validating after.
        for runtime_callback in runtime_callbacks_list:
            _admit_runtime_sink(runtime_callback)
        _validate_callbacks(runtime_callbacks_list)
        # Runtime callbacks join the list -- and are routed along with every
        # configured callback (F4) -- before this context, and therefore
        # `RunStarted`, exists at all.
        callbacks = [*callbacks, *runtime_callbacks_list]
    routed_sinks: dict[str, str] = {}
    if routed:
        routed_sinks = _route_artifact_sinks(
            [*callbacks, *loggers], logical_run_dir=logical_run_dir, destination=artifact_destination
        )
    metadata = build_run_metadata(resolved_cfg, command=command, config_path=config_path, clock=run_clock)
    if topology is None:
        try:
            device_identity = TorchAllocatorPeakProbe(metadata.device).identity()
        except RuntimeError:
            # Identity is optional telemetry: torch-free, malformed-device, and
            # ancient-torch environments still need to prepare their context.
            device_identity = None
        topology = ExecutionTopology.single_process(
            device=metadata.device, device_identity=device_identity
        )
    context = RunContext(
        cfg=resolved_cfg,
        source_cfg=source_cfg,
        artifact_manager=artifact_manager,
        metadata=metadata,
        clock=run_clock,
        callbacks=callbacks,
        loggers=loggers,
        topology=topology,
        profile_writer=(
            None
            if topology.global_rank is None
            else RankLocalJSONLWriter(
                artifact_manager.run_dir,
                topology,
                path=artifact_manager.path(
                    "profiles", topology.rank_path_component, "resources.jsonl"
                ),
            )
        ),
    )
    write_run_start_artifact(context)
    if artifact_destination is not None:
        write_artifact_routing_receipt(
            context,
            destination=artifact_destination,
            logical_run_dir=logical_run_dir,
            routed=routed_sinks,
        )
    return context


def run_from_config(
    cfg: DictConfig,
    *,
    config_path: str | None = None,
    command: str | None = None,
    raise_exceptions: bool = False,
    topology: ExecutionTopology | Mapping[str, object] | None = None,
    runtime_callbacks: Sequence[object] = (),
    restore_request: Any = None,
    artifact_destination: ArtifactDestination | None = None,
) -> int:
    """Instantiate and execute the configured runner.

    Parameters
    ----------
    cfg : DictConfig
        Resolved run configuration.
    config_path, command : str or None, optional
        Provenance recorded in run metadata.
    raise_exceptions : bool, optional
        When ``True``, re-raise the original exception after the status update,
        exception event, and logger teardown. The default ``False`` preserves
        CLI-style ``return 1`` behavior; tests and debugging can set ``True`` to
        surface the original traceback.
    topology : ExecutionTopology or mapping or None, optional
        Launcher-supplied execution topology forwarded to the context consumer
        for this call. Mapping facts are normalized to ``ExecutionTopology``
        before context construction. When omitted, ``prepare_run_context``
        applies its normal single-process fallback.
    runtime_callbacks : sequence of objects, optional
        Operational observers appended after configured callbacks. They are
        supplied by the launcher, outside the immutable science configuration;
        for example, a bounded prefix can stop after a published checkpoint
        without changing the trainer configuration that restore validates.
    restore_request : tpen.execution.recovery.OperationalRestoreRequest or None, optional
        Closed, versioned operational-start request carrying an E1-validated
        full-state restore decision, supplied by the launcher outside ``cfg``.
        When present, it is injected at the already-existing ``Train`` restore
        boundary (the runner's ``load`` argument) after construction and
        before ``fit``; ``cfg`` and its resolved ``run.dir``/``trainer.max_steps``
        are unchanged. A concrete configured ``runner.load`` that disagrees
        with the request's decision is refused rather than silently shadowed
        or silently preferred. Typed only as ``Any`` here so that importing
        ``tpen.run`` never pulls in ``tpen.execution`` (and its backend
        adapters) for callers that never pass this argument.
    artifact_destination : tpen.artifacts.ArtifactDestination or None, optional
        Operational, launcher-supplied physical leaf this run's artifact
        sinks must write beneath, outside ``cfg``. Forwarded to
        :func:`prepare_run_context`; see its docstring for the routing
        contract. ``cfg`` and the recorded ``run.dir`` are unaffected.

    Returns
    -------
    int
        ``0`` on success, ``1`` on a handled failure (when
        ``raise_exceptions=False``). A runner that RETURNS
        ``RunResult(status="failed")`` -- an evaluation suite whose tasks
        failed, which raises nothing -- is a handled failure too and also
        returns ``1``. It used to return ``0``, so a failed evaluation exited
        successfully and every launcher that reads an exit code recorded it as
        a success.
    """

    _install_bootstrap_stderr_logger()
    bootstrap = _BootstrapState()
    context: RunContext | None = None
    runner: Runner | None = None
    try:
        # PRECONSTRUCTION FIREWALL. This is the first statement in the try on
        # purpose: `prepare_run_context` below already creates the run
        # directory and instantiates every logger and callback, so a check
        # placed after it would be enforcing a rule against a run that had
        # already begun to exist. Nothing above this line constructs anything,
        # so a refusal here leaves no directory, no callback and no model.
        #
        # A configuration that declares no HI schema passes straight through --
        # see `tpen.hi_schema` on why the firewall is opt-in.
        #
        # `prepare_run_context` enforces the same rule for callers that bypass
        # this entry point. Keeping the check here as well means a refusal is
        # reported through this function's own failure path (status update,
        # exception event, logger teardown, `return 1`) rather than escaping as
        # a bare exception from a helper.
        validate_hi_train_config(cfg)
        if topology is not None:
            topology = execution_topology_from_facts(topology)
        if topology is None:
            context = prepare_run_context(
                cfg,
                config_path=config_path,
                command=command,
                bootstrap=bootstrap,
                artifact_destination=artifact_destination,
                runtime_callbacks=runtime_callbacks,
            )
        else:
            context = prepare_run_context(
                cfg,
                config_path=config_path,
                command=command,
                bootstrap=bootstrap,
                topology=topology,
                artifact_destination=artifact_destination,
                runtime_callbacks=runtime_callbacks,
            )
        if artifact_destination is None:
            # (E)/(R3.4) legacy runtime-callback phase: configured callbacks
            # and loggers were already validated inside `prepare_run_context`
            # (unconditionally, before `context` existed). Runtime callbacks
            # are validated here instead -- restoring the original ordering --
            # and merged into `context.callbacks` only once validation
            # succeeds, so a failing runtime-callback list appends NONE of
            # them, rather than all of them having already been merged inside
            # `prepare_run_context` regardless of outcome. A refusal here
            # still runs through the `except` branch below with `context`
            # bound: the failure artifact is written and every logger's
            # `finish()` still runs in the `finally` clause, instead of
            # escaping with the context only half-reported.
            _validate_callbacks(list(runtime_callbacks))
            context.callbacks.extend(runtime_callbacks)
        _seed_runtime_rngs(context.cfg)
        context.emit(RunStarted())
        runner = _instantiate_runner(context)
        if restore_request is not None:
            runner = _apply_operational_restore(context.cfg, runner, restore_request)
        for callback in context.callbacks:
            if isinstance(callback, PlannedYield):
                callback.validate_target(runner)
        result = runner.run(context)
        # The harness owns the whole run lifecycle, including this boundary,
        # which the runners emitted the ``run_end`` string for. See
        # `tpen.run_events` for why one emitter rather than three.
        #
        # The runner's own verdict rides the event. It cannot be read off
        # `context.metadata` by the callbacks instead: the copy below has to
        # stay AFTER the emit, because `Metadata` assigns ``metadata.status``
        # itself while handling this event and would otherwise overwrite a
        # failed suite's status with the boundary's own name.
        status = result.status if isinstance(result, RunResult) else "completed"
        if isinstance(result, RunResult):
            write_run_outcome_artifact(context, result)
        context.emit(RunCompleted(status=status, result=result if isinstance(result, RunResult) else None))
        if isinstance(result, RunResult):
            context.metadata.status = result.status
        # A failed suite raises nothing -- the runner returned -- but the process
        # must not claim success. Same exit code as the raising path: no caller
        # distinguishes between nonzero codes, and inventing a second one would
        # be a new contract for no consumer.
        return 0 if status == "completed" else 1
    except PlannedYieldSignal as signal:
        result = signal.result
        if context is not None:
            context.metadata.status = result.status
            write_run_outcome_artifact(context, result)
            try:
                context.emit(RunCompleted(status=result.status, result=result))
            except Exception as exc:
                context.metadata.status = "failed"
                _write_error_if_possible(context, exc, phase="run_completed", traceback_text=traceback.format_exc())
                _emit_typed_event_if_possible(
                    context, RunFailed(type(exc).__name__, str(exc))
                )
                return 1
        return 0 if result.status == "completed" else (PLANNED_YIELD_EXIT_CODE if result.status == "planned_yield" else 1)
    except CheckpointPublicationError as exc:
        result = RunResult(
            status="unknown", error=str(exc.original),
            next_iteration=exc.next_iteration, completed_updates=exc.completed_updates,
        )
        if context is not None:
            context.metadata.status = result.status
            write_run_outcome_artifact(context, result)
            context.emit(RunCompleted(status=result.status, result=result))
        return 1
    except Exception as exc:
        phase = _failure_phase(exc, context=context, runner=runner)
        traceback_text = traceback.format_exc()
        payload = {
            "exception": exc,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "phase": phase,
            "traceback": traceback_text,
        }
        if context is not None:
            context.metadata.status = "failed"
            _write_error_if_possible(context, exc, phase=phase, traceback_text=traceback_text)
            # ONE typed event for the two legacy strings below, which carry the
            # same payload and are distinguished by no consumer -- see `RunFailed`.
            # Guarded exactly as they are: this runs while the context may be
            # half-constructed, and an emit that raised here would mask the
            # original exception.
            _emit_typed_event_if_possible(
                context,
                RunFailed(
                    exception_type=str(payload["exception_type"]),
                    exception_message=str(payload["exception_message"]),
                ),
            )
        elif bootstrap.run_dir is not None:
            _write_error_if_possible(
                bootstrap.run_dir,
                exc,
                phase=phase,
                traceback_text=traceback_text,
                command=command,
                config_path=config_path,
            )
        _print_fatal(
            exc,
            phase=phase,
            traceback_text=traceback_text,
            run_dir=context.run_dir if context is not None else bootstrap.run_dir,
            command=command,
            config_path=config_path,
        )
        if raise_exceptions:
            raise
        return 1
    finally:
        if context is not None:
            for logger in context.loggers:
                logger.finish()


def _validate_callbacks(callbacks: list[object]) -> None:
    """Fail loudly if a configured callback lacks typed dispatch.

    This checks the interface shape only; callback behavior is invoked lazily
    through normal lifecycle events, never during setup.
    """

    for index, callback in enumerate(callbacks):
        if not callable(getattr(callback, "handle_occurrence", None)):
            raise TypeError(
                f"callbacks[{index}]={type(callback).__name__} must expose callable "
                "handle_occurrence(occurrence, context)"
            )


def _validate_loggers(loggers: list[object]) -> None:
    """Fail loudly if a configured logger lacks callable ``log``/``finish``.

    This checks the interface shape only; logger behavior is invoked lazily when
    records are logged and during normal run teardown, never during setup.
    """

    for index, logger in enumerate(loggers):
        if not callable(getattr(logger, "log", None)):
            raise TypeError(
                f"loggers[{index}]={type(logger).__name__} must expose callable log(record)"
            )
        if not callable(getattr(logger, "finish", None)):
            raise TypeError(
                f"loggers[{index}]={type(logger).__name__} must expose callable finish()"
            )


#: Closed registry of operational output-path fields a destination may rebind.
#: Keyed by the defining module's qualified class name -- the same shape as
#: `tpen.checkpoint.hashing._CALLBACK_OUTPUT_FIELDS`, which independently
#: excludes these same slots from the science-identity hash because they are
#: operational rather than scientific. Matching is by EXACT qualified type
#: name only (review-r3 R3.1): a subclass of one of these registered writers
#: is refused, not silently admitted through its ancestor, because a subclass
#: can add its own unaudited constructor- or method-side write that this
#: registry's field list would never see. core-owned writes that instead go
#: through `RunContext.path`/`ArtifactManager.path` are already physical
#: because `ArtifactManager.run_dir` itself is bound to the destination.
_ARTIFACT_SINK_FIELDS: dict[str, tuple[str, ...]] = {
    "tpen.callback.snapshot.ConfigSnapshot": ("output_path",),
    "tpen.callback.snapshot.ResolvedConfigSnapshot": ("output_path",),
    "tpen.callback.metadata.Metadata": ("output_path",),
    "tpen.callback.status.Status": ("output_path",),
    "tpen.callback.checkpoint.Checkpoint": ("output_dir",),
    "tpen.callback.initial_checkpoint.InitialCheckpoint": ("output_dir",),
    "tpen.callback.evaluation.ArtifactIndex": ("path",),
    "tpen.callback.evaluation.FailureLog": ("path",),
    "tpen.callback.parameter_activity.ParameterActivity": ("output_path",),
    "tpen.callback.equivariance.RuntimeEquivariance": ("artifact_dir",),
    "tpen.callback.raw_training_batch_recorder.RawTrainingBatchRecorder": ("path",),
    "tpen.logging.csv.CSV": ("path",),
    "tpen.logging.jsonl.JSONL": ("path",),
}

#: Closed registry of callback/logger types known to own NO constructor-side
#: output path at all (review-r2 B/R2.2). Membership is checked by EXACT
#: qualified type name, never by MRO: a subclass of one of these might add
#: its own output attribute, and inferring sink-freedom from the absence of
#: an attribute is exactly the heuristic this registry replaces. Every
#: core callback/logger reachable from `experiments/atomistic/he-importance`
#: or `tests/integration/artifacts/training/vmc_smoke.yaml` that is not a
#: registered writer above is listed here; anything else is refused.
_SINK_FREE_TYPES: frozenset[str] = frozenset(
    {
        "tpen.callback.amplitude_risk.AmplitudeRiskFraction",
        "tpen.callback.electron_nucleus_distance.ElectronNucleusDistanceQuantiles",
        "tpen.callback.factor_scalars.FactorScalars",
        "tpen.callback.module_norms.ModuleNorms",
        "tpen.callback.optimizer_state.OptimizerState",
        "tpen.callback.resource_usage.ResourceUsage",
        "tpen.callback.health.data_integrity.DataIntegrity",
        "tpen.callback.health.gradient_stats.GradientStats",
        "tpen.callback.health.sampler_health.SamplerHealth",
        "tpen.callback.timing.diagnostic_timing.DiagnosticTiming",
        "tpen.callback.timing.evaluation_component_timing.EvaluationComponentTiming",
        "tpen.callback.timing.evaluation_timing.EvaluationTiming",
        "tpen.callback.timing.run_timing.RunTiming",
        "tpen.callback.timing.train_phase_timing.TrainPhaseTiming",
        "tpen.callback.timing.train_step_timing.TrainStepTiming",
        "tpen.callback.planned_yield.PlannedYield",
    }
)


def _registered_sink_fields(sink: object) -> tuple[str, ...] | None:
    """Return the registered output fields for ``sink``'s EXACT type, if any.

    Matched by exact qualified type name only (review-r3 R3.1): no MRO walk,
    so a subclass of a registered writer is not matched through its ancestor.
    """

    qualified = f"{type(sink).__module__}.{type(sink).__qualname__}"
    return _ARTIFACT_SINK_FIELDS.get(qualified)


def _class_is_admitted_sink(klass: type) -> bool:
    """Return whether ``klass`` is an EXACT closed-admission sink type.

    Both the registered-writer registry and the sink-free allowlist are
    matched by EXACT qualified type name only (review-r3 R3.1): a subclass
    of either -- which might add its own unaudited constructor- or
    method-side write this check cannot see -- is refused, not silently
    admitted through its ancestor.
    """

    qualified = f"{klass.__module__}.{klass.__qualname__}"
    return qualified in _ARTIFACT_SINK_FIELDS or qualified in _SINK_FREE_TYPES


def _admit_configured_target(target: str) -> None:
    """Refuse an unregistered configured sink BEFORE it is constructed (B).

    Resolves ``target`` (a configured ``_target_`` dotted path) to its class
    without instantiating it, and admits it only when
    :func:`_class_is_admitted_sink` accepts that EXACT class. An unregistered
    type, or an unregistered subclass of a registered or sink-free type, is
    refused here -- before construction can write anything to its logical
    destination.
    """

    klass = get_class(target)
    if not _class_is_admitted_sink(klass):
        raise ArtifactRoutingError(
            f"{target} is not a registered routed writer or a known sink-free "
            "type; refusing to admit it to a routed run before construction"
        )


def _admit_runtime_sink(sink: object) -> None:
    """Refuse an unregistered, already-constructed runtime sink (F3/F4/R3.1).

    Runtime callbacks have no ``_target_`` to resolve ahead of construction,
    so admission is checked against the sink's EXACT type instead: a
    registered-writer type or a sink-free type, matched by exact qualified
    name only, with no MRO/subclass admission and no inference from the
    presence or absence of an output-path-shaped attribute (review-r3 R3.1
    rejects both as open sink admission). A genuinely sink-free runtime
    observer must be an exact member of one of the two closed registries --
    for example a registered sink-free callback type -- not merely an ad hoc
    object lacking a path attribute.
    """

    qualified = f"{type(sink).__module__}.{type(sink).__qualname__}"
    if qualified in _ARTIFACT_SINK_FIELDS or qualified in _SINK_FREE_TYPES:
        return
    raise ArtifactRoutingError(
        f"{qualified} is not a registered routed writer or a known sink-free "
        "type; refusing to admit it to a routed run"
    )


def _route_artifact_sinks(
    sinks: list[object],
    *,
    logical_run_dir: Path,
    destination: ArtifactDestination,
) -> dict[str, str]:
    """Rebind every known concrete sink field from the logical leaf to the physical one.

    A field is rebound only when it is currently set to a concrete value
    falling under ``logical_run_dir`` -- exactly the shape HI's eager
    ``${run.dir}`` resolution bakes into these constructors before
    `run_from_config` ever runs. `ArtifactDestination.bind` refuses a value
    that does not fall under ``logical_run_dir`` (an unsupported or unknown
    destination) and refuses a mapped path that would not stay under the
    claimed physical root (an escape).

    CLOSED SINK CONTRACT (F3, closed by construction-side admission per
    review-r2 B, exact-type only per review-r3 R3.1): by the time a sink
    reaches here, every configured sink already passed
    :func:`_admit_configured_target` before it was constructed, and every
    runtime sink already passed :func:`_admit_runtime_sink`. This function's
    own check -- a sink whose EXACT type is not in `_ARTIFACT_SINK_FIELDS` is
    admitted only when it is an exact member of `_SINK_FREE_TYPES` -- is
    therefore a second, cheap confirmation rather than the admission's first
    line of defense; it exists so this function stays correct for any direct
    caller that bypasses those gates (as several unit tests do).

    A field whose registered writer exposes an owner-defined
    ``rebind_<field>`` method (for example `tpen.logging.csv.CSV.rebind_path`,
    which also resets destination-dependent header state, F6) is rebound
    through that method instead of a bare ``setattr``.

    Returns
    -------
    dict of str to str
        ``"<qualified type name>.<field>"`` to the physical path it was
        rebound to, for every field actually rebound.

    Raises
    ------
    tpen.artifacts.ArtifactRoutingError
        When a sink's exact type is not a registered routed writer and is
        not a known sink-free type.
    """

    routed: dict[str, str] = {}
    for sink in sinks:
        qualified = f"{type(sink).__module__}.{type(sink).__qualname__}"
        fields = _registered_sink_fields(sink)
        if fields is None:
            if qualified not in _SINK_FREE_TYPES:
                raise ArtifactRoutingError(
                    f"{qualified} is not a registered routed writer or a known "
                    "sink-free type; refusing to admit it to a routed run before "
                    "it can write outside the claimed physical destination"
                )
            continue
        for field in fields:
            current = getattr(sink, field, None)
            if current is None:
                continue
            physical = destination.bind(logical_run_dir, Path(current))
            rebind = getattr(sink, f"rebind_{field}", None)
            if callable(rebind):
                rebind(physical)
            else:
                setattr(sink, field, physical)
            routed[f"{qualified}.{field}"] = str(physical)
    return routed


def _instantiate_sequence(items: ListConfig | list | tuple | None, *, routed: bool = False) -> list:
    """Instantiate each configured ``_target_`` item.

    When ``routed`` is true (B, review-r2), each item's ``_target_`` class is
    resolved and admission-checked via :func:`_admit_configured_target`
    BEFORE ``instantiate()`` constructs it -- an unregistered writer never
    gets to run its constructor, let alone write anything, in a routed run.
    """

    instantiated = []
    for item in items or []:
        if isinstance(item, DictConfig) and "_target_" in item:
            if routed:
                _admit_configured_target(str(item["_target_"]))
            instantiated.append(instantiate(item))
        else:
            instantiated.append(item)
    return instantiated


def _seed_runtime_rngs(cfg: DictConfig) -> None:
    """Seed process RNGs from ``runtime.seed`` before runner construction."""

    seed = OmegaConf.select(cfg, "runtime.seed", default=None)
    if seed is None:
        return
    seed_int = int(seed)
    random.seed(seed_int)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed_int % (2**32 - 1))
    if _config_requires_torch(OmegaConf.to_container(cfg, resolve=False)):
        torch = require_torch(feature="seeded configured TPEN run")
        torch.manual_seed(seed_int)
        accelerator_seed_all(seed_int, feature="seeded configured TPEN run")


def _instantiate_runner(context: RunContext) -> Runner:
    runner_cfg = context.cfg.runner.copy()
    # Callbacks and loggers are configured at the config root and owned by the
    # RunContext; a runner must not own them.
    for forbidden in ("callbacks", "loggers"):
        if forbidden in runner_cfg:
            raise ValueError(
                f"runner config must not own {forbidden!r}; configure it at the config root."
            )
    try:
        runner = instantiate(runner_cfg)
    except InstantiationException as exc:
        # Hydra wraps constructor/configuration failures, but the run artifact
        # contract records the underlying failure identity. Preserve the
        # original exception when Hydra attached it as the cause.
        if exc.__cause__ is not None:
            raise exc.__cause__ from exc
        raise
    if not isinstance(runner, Runner):
        raise TypeError(f"runner must instantiate to tpen.runner.Runner, got {type(runner)!r}")
    return runner


def _apply_operational_restore(cfg: DictConfig, runner: Runner, restore_request: Any) -> Runner:
    """Inject an E1-validated restore decision at the runner's existing load boundary.

    This does not construct, resolve, or mutate ``cfg``: it only checks the
    already-resolved ``runner.load`` leaf for a concrete conflict and then
    sets the live runner's ``load`` attribute, the same boundary a configured
    ``runner.load`` already reaches (``tpen.runner.train.Train.__init__``).
    Also refuses, before any restore or ``fit``, a live constructed
    ``runner.trainer.max_steps`` that disagrees with the request's fixed
    target horizon -- the constructed trainer object, not a hashed config
    section, is what ``VMCTrainer.fit`` actually steps against.
    """

    # Lazy import: ``tpen.execution`` pulls in backend adapters (for example
    # Balsam) that ordinary configured runs -- the overwhelming majority of
    # callers, which never pass ``restore_request`` -- must not be made to
    # import.
    from tpen.execution.recovery import OperationalRestoreRequest

    if not isinstance(restore_request, OperationalRestoreRequest):
        raise TypeError("restore_request must be an OperationalRestoreRequest")
    if not hasattr(runner, "load"):
        raise TypeError(
            f"runner {type(runner).__name__} does not support operational restore injection"
        )
    live_max_steps = getattr(getattr(runner, "trainer", None), "max_steps", None)
    if live_max_steps != restore_request.target_iterations:
        raise ValueError(
            "operational restore request's fixed target horizon "
            f"(target_iterations={restore_request.target_iterations}) does not match "
            f"the live runner's trainer.max_steps={live_max_steps!r}"
        )
    configured_load = OmegaConf.select(cfg, "runner.load", default=None)
    if OmegaConf.is_config(configured_load):
        configured_load = OmegaConf.to_container(configured_load, resolve=True)
    if _is_unconfigured_load(configured_load):
        configured_load = None
    decision_load = restore_request.load
    if configured_load is not None and dict(configured_load) != decision_load:
        raise ValueError(
            "operational restore request conflicts with a concrete configured runner.load"
        )
    runner.load = decision_load
    # Carry the full validated request to the live restore boundary, not just
    # its closed load shape (R2.3): ``Train.run`` uses this to revalidate the
    # selected ref immediately before restore and to bind restored progress
    # to the request's counters before `fit`, so a checkpoint replaced after
    # request construction -- or one whose restored counters merely disagree
    # -- cannot reach training. Additive: a runner that never reads
    # ``restore_binding`` behaves exactly as before.
    runner.restore_binding = restore_request
    return runner


def _is_unconfigured_load(configured_load: Any) -> bool:
    """Treat the supported no-load form ``{mode: none, path: None}`` as unset.

    ``${load}`` resolving to this shape (``tpen/execution/invocation.py``'s
    ``_has_concrete_config_path``) is an existing supported representation of
    "no configured load", not a concrete competing load, so it must not
    refuse an otherwise admissible operational restore request.
    """

    if not isinstance(configured_load, Mapping):
        return False
    return configured_load.get("path") is None and configured_load.get("mode") in (None, "none")


def _configure_terminal_logging(cfg: DictConfig) -> None:
    terminal = OmegaConf.select(cfg, "terminal", default=None)
    if terminal is None:
        return
    configure_terminal_logging(
        enabled=bool(OmegaConf.select(terminal, "enabled", default=True)),
        level=str(OmegaConf.select(terminal, "level", default="info")),
        color=str(OmegaConf.select(terminal, "color", default="auto")),
    )


def _install_bootstrap_stderr_logger() -> None:
    """Install a minimal stderr logger for fatal bootstrap diagnostics."""

    logger = logging.getLogger(_BOOTSTRAP_LOGGER_NAME)
    logger.setLevel(logging.ERROR)
    for handler in logger.handlers:
        if getattr(handler, "_tpen_bootstrap_handler", False):
            return
    handler = logging.StreamHandler(sys.stderr)
    handler._tpen_bootstrap_handler = True
    handler.setLevel(logging.ERROR)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def _failure_phase(
    exc: BaseException,
    *,
    context: RunContext | None,
    runner: Runner | None,
) -> str:
    phase = getattr(exc, "_tpen_failure_phase", None)
    if phase is not None:
        return str(phase)
    if context is None or runner is None:
        return "bootstrap"
    return "run"


def _write_error_if_possible(
    target: RunContext | Path,
    exc: BaseException,
    *,
    phase: str,
    traceback_text: str,
    command: str | None = None,
    config_path: str | None = None,
) -> None:
    try:
        write_error_artifact(
            target,
            exc,
            phase=phase,
            traceback_text=traceback_text,
            command=command,
            config_path=config_path,
        )
    except Exception as artifact_exc:  # pragma: no cover - disk/runtime dependent
        logging.getLogger(_BOOTSTRAP_LOGGER_NAME).error(
            "FATAL: failed to write error.json: %s: %s",
            type(artifact_exc).__name__,
            artifact_exc,
        )


def _emit_typed_event_if_possible(context: RunContext, event: TypedEvent) -> None:
    """Emit one typed event on the failure path without masking the failure.

    This reports the typed failure event without masking the original error; it
    reason: this runs after the run has already raised, possibly from a
    half-constructed context, so a callback or a disk error here must not replace
    the exception the user needs to see.
    """

    try:
        context.emit(event)
    except Exception as event_exc:  # pragma: no cover - callback/runtime dependent
        logging.getLogger(_BOOTSTRAP_LOGGER_NAME).error(
            "FATAL: failed to emit %s while reporting failure: %s: %s",
            type(event).__name__,
            type(event_exc).__name__,
            event_exc,
        )


def _print_fatal(
    exc: BaseException,
    *,
    phase: str,
    traceback_text: str | None = None,
    run_dir: Path | None = None,
    command: str | None = None,
    config_path: str | None = None,
) -> None:
    """Print a fatal diagnostic to stderr regardless of terminal settings."""

    parts = [f"FATAL {phase} error: {type(exc).__name__}: {exc}"]
    load_path = getattr(exc, "_tpen_load_path", None)
    load_mode = getattr(exc, "_tpen_load_mode", None)
    if load_path is not None:
        parts.append(f"load.path: {load_path}")
    if load_mode is not None:
        parts.append(f"load.mode: {load_mode}")
    if run_dir is not None:
        parts.append(f"run_dir: {run_dir}")
    if config_path is not None:
        parts.append(f"config: {config_path}")
    if command is not None:
        parts.append(f"command: {command}")
    if traceback_text:
        parts.append(traceback_text.rstrip())
    print("\n".join(parts), file=sys.stderr, flush=True)


def _preflight_optional_dependencies(cfg: DictConfig) -> None:
    """Fail early with actionable optional-dependency errors for configured targets."""

    if _config_requires_torch(OmegaConf.to_container(cfg, resolve=False)):
        require_torch(feature="configured TPEN run")


def _config_requires_torch(value: object) -> bool:
    if isinstance(value, dict):
        target = value.get("_target_")
        if isinstance(target, str) and _target_requires_torch(target):
            return True
        return any(_config_requires_torch(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_config_requires_torch(item) for item in value)
    return False


def _target_requires_torch(target: str) -> bool:
    return target.startswith(
        (
            "torch.",
            "tpen.nn.",
            "tpen.training.",
            "tpen.sampling.",
            "tpen.physics.",
            "tpen.diagnostics.",
            "tpen.equivariance.checks.",
            "tpen.runner.Train",
            "tpen.runner.Evaluate",
        )
    )


def _rerunnable_config(cfg: DictConfig) -> DictConfig:
    snapshot = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(snapshot, "run.run_id", None, merge=False, force_add=True)
    OmegaConf.update(snapshot, "run.dir", None, merge=False, force_add=True)
    return snapshot


__all__ = [
    "load_config",
    "main",
    "parse_args",
    "prepare_run_context",
    "run_from_config",
]
