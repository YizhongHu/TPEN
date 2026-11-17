"""Composable evaluation task runner."""

from __future__ import annotations

import traceback as traceback_module
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import torch
from omegaconf import OmegaConf

from tpen.checkpoint.hashing import file_sha256, resolved_config_identity
from tpen.checkpoint.replay import (
    CheckpointReplaySemantics,
    REPLAY_SEMANTICS_FILENAME,
    write_checkpoint_replay_semantics,
)
from tpen.evaluation.bundle import EvaluationBundle
from tpen.evaluation.events import (
    CalculatorRun,
    ComponentFailed,
    EvaluationTaskRun,
    GeneratorRun,
    SummaryRun,
)
from tpen.evaluation.protocols import EvaluationContext
from tpen.evaluation.results import ArtifactRecord, EvaluationFailure, EvaluationResult, MetricScalar, TaskResult
from tpen.evaluation.state import EvaluationRunState
from tpen.evaluation.task import ArtifactLevel, EvaluationTask, FailurePolicy, coerce_task


class Evaluator:
    """Run a sequence of evaluation tasks against one model."""

    def __init__(
        self,
        *,
        tasks: Sequence[EvaluationTask | Mapping[str, object]],
        namespace: str,
        artifact_level: ArtifactLevel = "metrics_only",
        task_failure_policy: FailurePolicy = "continue",
        seed: int | None = None,
    ) -> None:
        self.tasks = tuple(coerce_task(task) for task in tasks)
        self.namespace = str(namespace).strip("/")
        if not self.namespace:
            raise ValueError("Evaluator namespace must be non-empty")
        if artifact_level not in ("metrics_only", "summaries", "records"):
            raise ValueError(f"unsupported artifact_level {artifact_level!r}")
        if task_failure_policy not in ("continue", "fail_fast"):
            raise ValueError(f"unsupported task_failure_policy {task_failure_policy!r}")
        self.artifact_level = artifact_level
        self.task_failure_policy = task_failure_policy
        self.seed = seed

    def evaluate(
        self,
        *,
        model: torch.nn.Module,
        context: Any,
        replay_semantics: CheckpointReplaySemantics | None = None,
    ) -> EvaluationResult:
        """Run all configured tasks and return aggregate metrics.

        Parameters
        ----------
        model : torch.nn.Module
            Wavefunction model every task is evaluated against.
        context : RunContext
            Run-level context. It owns typed emission, so it is the only
            reporting channel the evaluator needs: the string-event ``emit``
            callable this method used to take carried the same moments in a
            flattened payload that four callbacks re-parsed, and every one of
            them now observes the typed occurrences instead.
        """

        # Two unrelated context types meet here: `context` is the run-level
        # `tpen.artifacts.RunContext` that owns typed emission, while
        # `base_context`/`task_context` are `EvaluationContext` values. The
        # alias keeps the two visually distinct wherever both are passed on.
        run_context = context
        base_context = self._context_from_run_context(context)
        # One state object for the whole suite, updated in place: `scope`
        # captures this reference at entry, so its identity must stay stable.
        state = EvaluationRunState()
        task_results: list[TaskResult] = []
        full_metrics: dict[str, MetricScalar] = {}
        failures: list[EvaluationFailure] = []
        artifacts: list[ArtifactRecord] = []
        if replay_semantics is not None:
            replay_path = write_checkpoint_replay_semantics(
                replay_semantics, run_context.run_dir
            )
            artifacts.append(
                ArtifactRecord(
                    name=REPLAY_SEMANTICS_FILENAME,
                    kind="checkpoint_replay_semantics",
                    path=replay_path,
                    metadata={"content_id": replay_semantics.content_id()},
                )
            )

        for task in self.tasks:
            task_output_dir = _resolve_task_output_dir(task.output_dir, run_dir=base_context.run_dir)
            # Materialize the task directory before running so summaries can write
            # task-local artifacts without each re-creating it.
            task_output_dir.mkdir(parents=True, exist_ok=True)
            task_context = replace(
                base_context,
                namespace=task.namespace,
                artifact_level=task.artifact_level or base_context.artifact_level,
                task_output_dir=task_output_dir,
            )
            # Clear the previous task's result before this task's scope opens, so
            # a handler at the `Started` boundary cannot read a stale one.
            state.task_result = None
            task_run = EvaluationTaskRun(
                name=task.name, namespace=task.namespace, output_dir=task_output_dir
            )
            with run_context.scope(task_run, state=state):
                result = self._evaluate_task(
                    model=model,
                    task=task,
                    context=task_context,
                    run_context=run_context,
                    state=state,
                    replay_semantics=replay_semantics,
                )
                # Written inside the scope so the `Ended` boundary observes it.
                state.task_result = result
            task_results.append(result)
            failures.extend(result.failures)
            artifacts.extend(result.artifacts)
            for key, value in result.metrics.items():
                full_metrics[f"{result.namespace}/{key}"] = value
            if result.failed and self.task_failure_policy == "fail_fast":
                break

        status = _aggregate_status(task_results)
        return EvaluationResult(
            status=status,
            metrics=full_metrics,
            task_results=tuple(task_results),
            artifacts=tuple(artifacts),
            failures=tuple(failures),
            replay_semantics=replay_semantics,
        )

    def _context_from_run_context(self, context: Any) -> EvaluationContext:
        device = _torch_device(getattr(getattr(context, "metadata", None), "device", None))
        dtype = _torch_dtype(getattr(getattr(context, "metadata", None), "dtype", None))
        run_dir = _context_run_dir(context) or Path(".")
        identity = _evaluation_identity_from_run_context(context)
        return EvaluationContext(
            namespace=self.namespace,
            artifact_level=self.artifact_level,
            task_failure_policy=self.task_failure_policy,
            device=device,
            dtype=dtype,
            seed=self.seed,
            run_dir=run_dir,
            task_output_dir=run_dir,
            metadata={},
            resolved_config_identity=identity.resolved_config_identity,
            model_provenance=identity.model_provenance,
            checkpoint_sha256=identity.checkpoint_sha256,
        )

    def _evaluate_task(
        self,
        *,
        model: torch.nn.Module,
        task: EvaluationTask,
        context: EvaluationContext,
        run_context: Any,
        state: EvaluationRunState,
        replay_semantics: CheckpointReplaySemantics | None,
    ) -> TaskResult:
        output_dir = context.task_output_dir
        failures: list[EvaluationFailure] = []
        artifacts: list[ArtifactRecord] = []
        metrics: dict[str, MetricScalar] = {}
        task_failed = False
        partial_failed = False
        bundle: EvaluationBundle | None = None

        # Each component is bracketed by its own typed scope. The evaluator owns
        # the boundaries but no timing policy: `EvaluationComponentTiming`
        # decides whether and how to measure them. `scope` emits ``Ended`` from a
        # ``finally``, so a component scope closes on failure as well as success,
        # and the failure is then reported separately as `ComponentFailed`.
        try:
            with run_context.scope(
                GeneratorRun(name=_component_name(task.generator)), state=state
            ):
                generated = task.generator.generate(model=model, context=context)
            bundle = EvaluationBundle(generated=generated)
        except Exception as exc:
            failure = _failure(context, task=task, component=task.generator, component_type="generator", exc=exc)
            failures.append(failure)
            result = _task_result(
                task,
                output_dir,
                "failed",
                metrics,
                artifacts,
                failures,
                replay_semantics,
            )
            run_context.emit(ComponentFailed(failure=failure), state=state)
            return result

        for calculator in task.calculators:
            try:
                with run_context.scope(
                    CalculatorRun(name=_component_name(calculator)), state=state
                ):
                    bundle = calculator.calculate(model=model, bundle=bundle, context=context)
            except Exception as exc:
                failure = _failure(context, task=task, component=calculator, component_type="calculator", exc=exc)
                failures.append(failure)
                task_failed = True
                run_context.emit(ComponentFailed(failure=failure), state=state)
                if context.task_failure_policy == "fail_fast":
                    break

        if bundle is not None:
            for summary in task.summaries:
                if not _summary_dependencies_present(summary, bundle):
                    if task_failed:
                        continue
                    failure = _missing_dependency_failure(context, task=task, summary=summary)
                    failures.append(failure)
                    partial_failed = True
                    run_context.emit(ComponentFailed(failure=failure), state=state)
                    continue
                try:
                    with run_context.scope(
                        SummaryRun(name=_component_name(summary)), state=state
                    ):
                        result = summary.summarize(bundle=bundle, context=context, namespace=task.namespace)
                    _merge_metrics(metrics, result.metrics, component_name=_component_name(summary))
                except Exception as exc:
                    failure = _failure(context, task=task, component=summary, component_type="summary", exc=exc)
                    failures.append(failure)
                    partial_failed = True
                    run_context.emit(ComponentFailed(failure=failure), state=state)
                    continue
                artifacts.extend(result.artifacts)

        if task_failed:
            status = "failed"
        elif partial_failed:
            status = "partial_failed"
        else:
            status = "success"
        # No event is emitted here. The legacy path split this single moment in
        # two by comparing the status string -- ``task_end`` or ``task_failed``
        # -- so a subscriber could tell them apart. The typed path does not need
        # the split: the caller writes this result onto `EvaluationRunState`
        # before the task scope closes, so `Ended[EvaluationTaskRun]` carries the
        # outcome, status included, on both paths (ADR-E008).
        return _task_result(
            task,
            output_dir,
            status,
            metrics,
            artifacts,
            failures,
            replay_semantics,
        )


def _task_result(
    task: EvaluationTask,
    output_dir: Path,
    status: str,
    metrics: Mapping[str, MetricScalar],
    artifacts: Sequence[ArtifactRecord],
    failures: Sequence[EvaluationFailure],
    replay_semantics: CheckpointReplaySemantics | None,
) -> TaskResult:
    return TaskResult(
        name=task.name,
        namespace=task.namespace,
        output_dir=output_dir,
        status=status,  # type: ignore[arg-type]
        metrics=dict(metrics),
        artifacts=tuple(artifacts),
        failures=tuple(failures),
        replay_semantics=replay_semantics,
    )


def _summary_dependencies_present(summary: object, bundle: EvaluationBundle) -> bool:
    required = getattr(summary, "required_fields", frozenset())
    return all(getattr(bundle, field, None) is not None for field in required)


def _missing_dependency_failure(context: EvaluationContext, *, task: EvaluationTask, summary: object) -> EvaluationFailure:
    required = sorted(getattr(summary, "required_fields", frozenset()))
    return EvaluationFailure(
        task=task.name,
        component=_component_name(summary),
        component_type="summary",
        error_type="MissingBundleField",
        message=f"summary requires missing bundle field(s): {required}",
        traceback=None,
    )


def _failure(
    context: EvaluationContext,
    *,
    task: EvaluationTask,
    component: object,
    component_type: str,
    exc: Exception,
) -> EvaluationFailure:
    return EvaluationFailure(
        task=task.name,
        component=_component_name(component),
        component_type=component_type,  # type: ignore[arg-type]
        error_type=type(exc).__name__,
        message=str(exc),
        traceback=traceback_module.format_exc(),
    )


def _component_name(component: object | None) -> str | None:
    if component is None:
        return None
    name = getattr(component, "name", None)
    return str(name) if name is not None else type(component).__name__


def _merge_metrics(target: dict[str, MetricScalar], values: Mapping[str, MetricScalar], *, component_name: str | None) -> None:
    for key, value in values.items():
        if key in target:
            raise ValueError(f"metric key collision for {key!r} from {component_name or 'summary'}")
        target[key] = value


def _resolve_task_output_dir(value: Path | str, *, run_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(run_dir) / path


def _aggregate_status(task_results: Sequence[TaskResult]) -> str:
    # Any configured task failure (full or partial) fails the suite; there is no
    # `required` flag to downgrade it. `success_with_warnings` is reserved for
    # non-task-critical issues (e.g. skipped tasks), not broken evaluation tasks.
    if any(task.failed for task in task_results):
        return "failed"
    if any(task.status == "skipped" for task in task_results):
        return "success_with_warnings"
    return "success"


def _torch_device(value: object) -> torch.device | None:
    if value in (None, ""):
        return None
    return torch.device(str(value))


def _torch_dtype(value: object) -> torch.dtype | None:
    if value in (None, ""):
        return None
    dtype = getattr(torch, str(value))
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported evaluation dtype {value!r}")
    return dtype


def _context_run_dir(context: Any) -> Path | None:
    try:
        return context.run_dir
    except AttributeError:
        return None


class _EvaluationIdentity(NamedTuple):
    """Run-derived identity values passed to evaluation components."""

    resolved_config_identity: Mapping[str, str] | None
    model_provenance: str
    checkpoint_sha256: str | None


def _evaluation_identity_from_run_context(context: Any) -> _EvaluationIdentity:
    """Return the identity bound to the run that owns ``context``.

    The core run-start artifact and evaluation records must use the same
    resolved-config identity. Checkpoint-backed evaluation rows use the
    explicit model file already supplied by the trajectory join configuration;
    constructed controls deliberately carry no invented checkpoint digest.
    """

    cfg = getattr(context, "cfg", None)
    source_cfg = getattr(context, "source_cfg", None)
    metadata = getattr(context, "metadata", None)
    if cfg is None or source_cfg is None or metadata is None:
        return _EvaluationIdentity(None, "constructed", None)

    try:
        config_identity = resolved_config_identity(
            source_cfg,
            run_values={
                "root": OmegaConf.select(cfg, "run.root", default=None),
                "run_id": getattr(metadata, "run_id"),
                "dir": getattr(metadata, "run_dir"),
                "layout": OmegaConf.select(cfg, "run.layout", default=None),
                "timezone": getattr(getattr(context, "clock", None), "timezone"),
            },
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        # Non-core programmatic contexts do not own a resolved run config.
        # They remain usable for v1 evaluation and must opt in explicitly to
        # v2 with a complete identity on EvaluationContext itself.
        config_identity = None

    try:
        checkpoint_file = OmegaConf.select(
            cfg, "trajectory_identity.checkpoint_file", default=None
        )
    except (KeyError, TypeError, ValueError):
        checkpoint_file = None
    if checkpoint_file in (None, "", "???"):
        return _EvaluationIdentity(config_identity, "constructed", None)

    checkpoint_path = Path(str(checkpoint_file))
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(getattr(metadata, "run_dir")) / checkpoint_path
    return _EvaluationIdentity(config_identity, "checkpoint", file_sha256(checkpoint_path))


__all__ = ["Evaluator"]
