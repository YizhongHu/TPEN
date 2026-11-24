"""Configured restart gateway for one Stage-1 training lineage.

Stage-1 owns the study identity checks in this module. Checkpoint discovery,
compatibility, and restore construction remain core responsibilities exposed
through :func:`tpen.inspection.inspect_run` and the normal
``tpen.run.run_from_config`` gateway.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from omegaconf import OmegaConf

from experiments.atomistic.stage1.protocol import (
    validate_gateway_binding,
    validate_resolved_train_config,
)
from experiments.atomistic.stage1.train import build_train_config
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.inspection import RunEvidence, inspect_run
from tpen.run import run_from_config


class RestartCause(str, Enum):
    """Stable reasons for refusing a configured restart."""

    MISSING_RESUME = "missing_resume_input"
    INVALID_RESUME = "invalid_resume_input"
    RUN_DIRECTORY_MISMATCH = "run_directory_mismatch"
    HORIZON_MISMATCH = "horizon_mismatch"


class RestartRefusal(ValueError):
    """A restart input was not admissible; the runner was not called."""

    def __init__(self, message: str, *, cause: RestartCause = RestartCause.INVALID_RESUME) -> None:
        self.cause = cause
        super().__init__(message)


class StartDecision(str, Enum):
    """The only start decision supported by this requested-resume gateway."""

    RESTORE = "restore"


@dataclass(frozen=True)
class RestartSelection:
    """Public inspection facts selected for one configured restart."""

    run_dir: Path
    checkpoint_dir: Path
    next_iteration: int
    completed_updates: int
    inspection: RunEvidence
    decision: StartDecision = StartDecision.RESTORE

    @property
    def content_id(self) -> str | None:
        """Return the public checkpoint content id when the inspection has one."""

        if self.inspection.checkpoint_ref is None:
            return None
        value = self.inspection.checkpoint_ref.get("content_id")
        return value if isinstance(value, str) else None


@dataclass(frozen=True)
class RestartResult:
    """Result of one configured ``run_from_config`` invocation."""

    selection: RestartSelection
    outcome: str
    exit_code: int
    runner_load: Mapping[str, str]


def _configured_recorder_overlay(cfg: Any, overlay: str | None) -> str | None:
    """Return the configured diagnostics overlay when recorder resume is present."""

    callbacks = OmegaConf.select(cfg, "callbacks", default=())
    for callback in callbacks or ():
        target = callback.get("_target_") if isinstance(callback, Mapping) else None
        if target == "tpen.callback.RawTrainingBatchRecorder":
            configured_overlay = OmegaConf.select(cfg, "stage1_overlay.path", default=None)
            return str(configured_overlay or overlay or "<unspecified overlay>")
    return None


def _refuse_unsupported_resume_callbacks(cfg: Any, overlay: str | None) -> None:
    """Refuse the deferred recorder contract before checkpoint inspection or launch."""

    configured_overlay = _configured_recorder_overlay(cfg, overlay)
    if configured_overlay is not None:
        raise RestartRefusal(
            "RawTrainingBatchRecorder cannot be resumed by Stage-1; "
            "core item 7eaf9e0c (Executor lane ea7125c1) owns deferred recorder "
            f"resume semantics; refusing overlay {configured_overlay}",
            cause=RestartCause.INVALID_RESUME,
        )


def _target_iterations(cfg: Any, row: Mapping[str, Any]) -> int:
    """Require the resolved trainer horizon and manifest horizon to agree."""

    target = OmegaConf.select(cfg, "trainer.max_steps", default=None)
    expected = row.get("updates")
    if type(target) is not int or type(expected) is not int or target <= 0 or target != expected:
        raise RestartRefusal(
            "resolved trainer.max_steps does not match the manifest update horizon",
            cause=RestartCause.HORIZON_MISMATCH,
        )
    return target


def _config_run_dir(cfg: Any) -> Path:
    """Resolve the output directory from the concrete producer config."""

    fields = {
        "results_root": OmegaConf.select(cfg, "run.root", default=None),
        "experiment_name": OmegaConf.select(cfg, "experiment.name", default=None),
        "sector": OmegaConf.select(cfg, "experiment.sector", default=None),
        "run_id": OmegaConf.select(cfg, "run.run_id", default=None),
    }
    if any(value is None or not str(value).strip() for value in fields.values()):
        raise RestartRefusal(
            "resolved config lacks a complete run identity",
            cause=RestartCause.RUN_DIRECTORY_MISMATCH,
        )
    try:
        return resolve_run_dir(fields)
    except Exception as exc:
        raise RestartRefusal(
            "resolved config run identity is not admissible",
            cause=RestartCause.RUN_DIRECTORY_MISMATCH,
        ) from exc


def _admitted_run_dir(cfg: Any, row: Mapping[str, Any], run_dir: str | Path | None) -> Path:
    """Require the row, config, and compatibility path to name one run."""

    try:
        row_dir = resolve_run_dir(row)
    except Exception as exc:
        raise RestartRefusal(
            "manifest row run identity is not admissible",
            cause=RestartCause.RUN_DIRECTORY_MISMATCH,
        ) from exc
    config_dir = _config_run_dir(cfg)
    if config_dir.resolve() != row_dir.resolve():
        raise RestartRefusal(
            f"config-derived run directory differs from manifest run directory: config={config_dir} manifest={row_dir}",
            cause=RestartCause.RUN_DIRECTORY_MISMATCH,
        )
    if run_dir is not None and Path(run_dir).resolve() != config_dir.resolve():
        raise RestartRefusal(
            f"explicit run_dir differs from admitted run directory: explicit={run_dir} admitted={config_dir}",
            cause=RestartCause.RUN_DIRECTORY_MISMATCH,
        )
    return config_dir


def _checkpoint_facts(
    evidence: RunEvidence, *, run_dir: Path, target: int,
) -> tuple[Path, int, int]:
    """Validate public checkpoint facts without inspecting core files."""

    if evidence.verdict == "refused" or evidence.refusal is not None:
        reason = (
            evidence.refusal.get("reason", "public run inspection refused")
            if evidence.refusal else "public run inspection refused"
        )
        cause = RestartCause.MISSING_RESUME if evidence.refusal and evidence.refusal.get("code") in {
            "missing-run-dir", "missing-status", "missing-checkpoint", "no-train-resume-checkpoint",
        } else RestartCause.INVALID_RESUME
        raise RestartRefusal(reason, cause=cause)
    checkpoint = evidence.checkpoint
    if not isinstance(checkpoint, Mapping):
        raise RestartRefusal(
            "public run inspection returned no checkpoint facts",
            cause=RestartCause.MISSING_RESUME,
        )
    checkpoint_value = checkpoint.get("checkpoint_dir")
    next_iteration = checkpoint.get("next_iteration")
    completed_updates = checkpoint.get("completed_updates")
    if (
        not isinstance(checkpoint_value, str)
        or type(next_iteration) is not int
        or type(completed_updates) is not int
        or next_iteration < 0
        or completed_updates < 0
        or next_iteration > target
        or completed_updates > target
    ):
        raise RestartRefusal("public checkpoint facts are incompatible with the requested horizon")
    checkpoint_dir = Path(checkpoint_value).resolve()
    checkpoint_root = (run_dir / "checkpoints").resolve()
    try:
        checkpoint_dir.relative_to(checkpoint_root)
    except ValueError as exc:
        raise RestartRefusal(
            "public checkpoint is outside the admitted run directory",
            cause=RestartCause.INVALID_RESUME,
        ) from exc
    if not checkpoint_dir.is_dir():
        raise RestartRefusal("public checkpoint directory is missing", cause=RestartCause.MISSING_RESUME)
    if next_iteration == target or completed_updates == target:
        raise RestartRefusal(
            "public checkpoint already reaches the full Stage-1 horizon",
            cause=RestartCause.INVALID_RESUME,
        )
    return checkpoint_dir, next_iteration, completed_updates


def select_restart(
    *, cfg: Any, row: Mapping[str, Any], run_dir: str | Path | None = None,
    inspector: Callable[[str | Path], RunEvidence] = inspect_run,
) -> RestartSelection:
    """Select one concrete checkpoint through the public inspection surface.

    A requested restart never falls back to a fresh run. Missing, ambiguous,
    torn, or incompatible public evidence raises before the runner is called.
    """

    target = _target_iterations(cfg, row)
    admitted = _admitted_run_dir(cfg, row, run_dir)
    evidence = inspector(admitted)
    checkpoint_dir, next_iteration, completed_updates = _checkpoint_facts(
        evidence, run_dir=admitted, target=target,
    )
    return RestartSelection(
        run_dir=admitted,
        checkpoint_dir=checkpoint_dir,
        next_iteration=next_iteration,
        completed_updates=completed_updates,
        inspection=evidence,
    )


def _remove_initial_checkpoint(cfg: Any) -> None:
    """Do not write a new checkpoint-zero artifact while restoring checkpoint zero."""

    callbacks = list(OmegaConf.select(cfg, "callbacks", default=[]))
    callbacks = [
        callback for callback in callbacks
        if callback.get("_target_") != "tpen.callback.InitialCheckpoint"
    ]
    OmegaConf.update(cfg, "callbacks", callbacks, merge=False, force_add=True)


def configure_attempt_local_diagnostics(cfg: Any, attempt_dir: str | Path) -> Any:
    """Keep the study-owned diagnostic path binder available to callers."""

    attempt_dir = Path(attempt_dir)
    callbacks = list(OmegaConf.select(cfg, "callbacks", default=[]))
    for callback in callbacks:
        target = callback.get("_target_")
        if target == "tpen.callback.RawTrainingBatchRecorder":
            callback["path"] = str(attempt_dir / "training_history")
        elif target == "tpen.callback.ParameterActivity":
            callback["output_path"] = str(attempt_dir / "parameter_activity.json")
        elif target == "tpen.callback.RunTiming":
            callback["output_path"] = str(attempt_dir / "run_timing.json")
    OmegaConf.update(cfg, "callbacks", callbacks, merge=False, force_add=True)
    loggers = list(OmegaConf.select(cfg, "loggers", default=[]))
    for logger in loggers:
        if logger.get("_target_") == "tpen.logging.JSONL":
            logger["path"] = str(attempt_dir / "metrics.jsonl")
    OmegaConf.update(cfg, "loggers", loggers, merge=False, force_add=True)
    return cfg


def _set_resume_load(cfg: Any, checkpoint_dir: Path) -> dict[str, str]:
    """Set the exact configured load delta consumed by ``run_from_config``."""

    load = {"mode": "train_resume", "path": str(checkpoint_dir)}
    OmegaConf.update(cfg, "runner.load", load, merge=False, force_add=True)
    return load


def execute_restart(
    *, cfg: Any, row: Mapping[str, Any], config_path: str | Path,
    binding: str | None = None, overrides: list[str] | None = None,
    kind: str = "train", overlay: str | None = None,
    overlay_sha256: str | None = None, command: str = "stage1 restart",
    run: Callable[..., int] | None = None,
    run_dir: str | Path | None = None,
    inspector: Callable[[str | Path], RunEvidence] = inspect_run,
) -> RestartResult:
    """Run one approved configured resume through the ordinary gateway."""

    if binding is not None:
        row = validate_gateway_binding(
            binding=binding, config=str(config_path), overrides=overrides or [], kind=kind,
            overlay=overlay, overlay_sha256=overlay_sha256,
        )
    validate_resolved_train_config(cfg, row)
    _refuse_unsupported_resume_callbacks(cfg, overlay)
    selection = select_restart(cfg=cfg, row=row, run_dir=run_dir, inspector=inspector)
    _remove_initial_checkpoint(cfg)
    runner_load = _set_resume_load(cfg, selection.checkpoint_dir)
    runner = run_from_config if run is None else run
    exit_code = int(runner(cfg, config_path=str(config_path), command=command))
    return RestartResult(
        selection=selection,
        outcome="completed" if exit_code == 0 else "failed",
        exit_code=exit_code,
        runner_load=runner_load,
    )


def restart_train_from_row(
    *, row: Mapping[str, Any], config_path: str | Path, overrides: list[str],
    binding: str, overlay: str | None = None, overlay_sha256: str | None = None,
    command: str = "stage1 restart",
) -> RestartResult:
    """Build one row-bound config and execute its requested resume."""

    resolved = validate_gateway_binding(
        binding=binding, config=str(config_path), overrides=overrides, kind="train",
        overlay=overlay, overlay_sha256=overlay_sha256,
    )
    cfg = build_train_config(
        resolved, config_path=config_path, overrides=overrides,
        overlay=overlay, overlay_sha256=overlay_sha256,
    )
    return execute_restart(
        cfg=cfg, row=resolved, config_path=config_path, command=command,
    )


__all__ = [
    "RestartCause", "RestartRefusal", "RestartResult", "RestartSelection",
    "StartDecision", "configure_attempt_local_diagnostics", "execute_restart",
    "restart_train_from_row", "select_restart",
]
