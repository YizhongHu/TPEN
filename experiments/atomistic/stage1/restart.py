"""Restartable start protocol for one admitted Stage-1 train lineage.

The module is deliberately a protocol adapter.  Checkpoint publication,
native recovery admission, typed restore construction, and gateway/config
validation remain owned by their respective core modules.  This layer only
coordinates their public interfaces for one existing run directory.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from omegaconf import DictConfig, OmegaConf

from experiments.atomistic.stage1.protocol import (
    STAGE1_PROTOCOL,
    validate_gateway_binding,
    validate_resolved_train_config,
)
from experiments.atomistic.stage1.train import prepare_initial_checkpoint_config
from experiments.toolkit.run_paths import resolve_run_dir
from tpen.checkpoint import CheckpointRef, configured_load_mode, list_complete_checkpoints
from tpen.execution import (
    CheckpointEvidence,
    OperationalRestoreRequest,
    ReceiptExistsError,
    ScienceIntent,
    create_json_receipt,
    discover_published_checkpoint,
    select_recovery,
    validate_native_recovery_profile,
)
from tpen.run import load_config, run_from_config


_STEP_NAME = re.compile(r"step_(?P<step>[0-9]{6})$")
_ATTEMPT_SEQUENCE = re.compile(r"attempt-(?P<number>[0-9]{6})\.json$")
ATTEMPT_RECEIPT_SCHEMA = "stage1/restart-attempt/v1"
OUTCOME_RECEIPT_SCHEMA = "stage1/restart-outcome/v1"


class RestartCause(str, Enum):
    """Typed causes for every parked or refused restart decision."""

    DISCOVERY_ERROR = "checkpoint_discovery_error"
    NEWER_COMPLETE_CHECKPOINT = "newer_complete_checkpoint"
    RECOVERY_REFUSAL = "recovery_refusal"
    RESTORE_REQUEST_INVALID = "restore_request_invalid"
    NO_PUBLICATION = "no_publication_at_admitted_run_dir"
    POST_RUN_DISCOVERY_ERROR = "post_run_checkpoint_discovery_error"
    INVOCATION_ERROR = "restart_invocation_error"
    RUN_DIRECTORY_MISMATCH = "run_directory_mismatch"


class RestartRefusal(ValueError):
    """A deterministic refusal that must not start model or driver work."""

    def __init__(
        self, message: str, *, cause: RestartCause = RestartCause.INVOCATION_ERROR,
    ) -> None:
        self.cause = cause
        super().__init__(message)


class RestartPark(ValueError):
    """A reconciliation condition that must remain parked."""


class StartDecision(str, Enum):
    """The only decisions allowed for one Stage-1 start."""

    FRESH = "fresh"
    RESTORE = "restore"
    COMPLETED = "completed"
    PARK = "park"


@dataclass(frozen=True)
class RestartApproval:
    """External replacement approval consumed by the takeover gate.

    This object carries proof; it does not create takeover authority.  The
    orchestrator remains responsible for producing the terminal/quiescence
    and budget facts represented here.
    """

    predecessor_content_id: str
    terminal_pbs_proof: str
    quiescent: bool
    adopted_budget: str
    lineage_id: str
    config_identity: str
    overlay_identity: str
    target_iterations: int

    def validate(self, *, expected: Mapping[str, Any]) -> None:
        """Validate approval facts against the immutable selected lineage."""

        if not self.terminal_pbs_proof.strip():
            raise RestartRefusal("takeover approval lacks terminal PBS proof")
        if not self.quiescent:
            raise RestartRefusal("takeover approval lacks predecessor quiescence")
        if not self.adopted_budget.strip():
            raise RestartRefusal("takeover approval lacks an adopted recovery budget")
        checks = {
            "predecessor_content_id": self.predecessor_content_id,
            "lineage_id": self.lineage_id,
            "config_identity": self.config_identity,
            "overlay_identity": self.overlay_identity,
            "target_iterations": self.target_iterations,
        }
        for key, actual in checks.items():
            if actual != expected.get(key):
                raise RestartRefusal(f"takeover approval {key} does not match lineage")


@dataclass(frozen=True)
class RestartSelection:
    """Immutable result of the read-only checkpoint decision."""

    decision: StartDecision
    run_dir: Path
    predecessor: CheckpointRef | None
    recovery: RecoveryDecision | None
    restore_request: OperationalRestoreRequest | None
    reason: str = ""
    cause: RestartCause | None = None

    @property
    def content_id(self) -> str:
        """Return the predecessor identity, or the stable initial marker."""

        return "initial" if self.predecessor is None else self.predecessor.content_id


@dataclass(frozen=True)
class RestartResult:
    """Durable result of one restartable invocation."""

    selection: RestartSelection
    lease_path: Path
    outcome_path: Path
    outcome: str
    exit_code: int | None
    receipt: Mapping[str, Any]


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RestartRefusal(f"{field} must be non-empty text")
    return value


def _target_iterations(cfg: Any, row: Mapping[str, Any]) -> int:
    """Read the immutable 100k horizon from the resolved config and row."""

    target = OmegaConf.select(cfg, "trainer.max_steps", default=None)
    expected = row.get("updates", STAGE1_PROTOCOL.lineage_updates)
    if type(target) is not int or target != expected or target != STAGE1_PROTOCOL.lineage_updates:
        raise RestartRefusal("resolved trainer.max_steps is not the fixed Stage-1 horizon")
    return target


def _identity(row: Mapping[str, Any], cfg: Any, target: int) -> dict[str, Any]:
    """Derive lease identity only from already admitted row/config values."""

    config_identity = row.get("config_sha256")
    if isinstance(config_identity, Mapping):
        config_identity = config_identity.get(row.get("system"))
    return {
        "lineage_id": _text(row.get("row_id"), "row_id"),
        "config_identity": _text(config_identity, "config_identity"),
        "overlay_identity": str(row.get("overlay_sha256") or "none"),
        "target_iterations": target,
        "system": _text(row.get("system"), "system"),
        "run_id": _text(row.get("run_id"), "run_id"),
    }


def _complete_refs(checkpoint_root: Path) -> list[CheckpointRef]:
    """Read complete step directories as typed refs, preserving refusals."""

    refs: list[CheckpointRef] = []
    for path in list_complete_checkpoints(checkpoint_root):
        try:
            refs.append(CheckpointRef.from_directory(path))
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            raise RestartPark(f"complete checkpoint is not a valid train-resume ref: {exc}") from exc
    return refs


def _check_newer_complete(*, checkpoint_root: Path, published: CheckpointRef | None) -> None:
    """Park when a complete step is newer than the CX-08 publication."""

    refs = _complete_refs(checkpoint_root) if checkpoint_root.is_dir() else []
    floor = -1 if published is None else published.next_iteration
    newer = [ref for ref in refs if ref.next_iteration > floor]
    if newer:
        newest = max(newer, key=lambda ref: ref.next_iteration)
        raise RestartPark(
            "complete checkpoint is newer than the published predecessor: "
            f"{newest.checkpoint_dir}"
        )


def _science_intent(cfg: Any, row: Mapping[str, Any], target: int) -> ScienceIntent:
    """Capture a JSON-shaped intent without constructing model objects."""

    science_input = {
        "system": row.get("system"),
        "system_id": row.get("system_id"),
        "lineage": row.get("lineage", 0),
        "seed": row.get("seed"),
        "sampler_seed": row.get("sampler_seed"),
        "config_sha256": row.get("config_sha256"),
        "overlay_sha256": row.get("overlay_sha256"),
    }
    source = OmegaConf.to_container(cfg, resolve=False)
    return ScienceIntent(
        science_input=science_input, target_iterations=target, source_config=source,
    )


def _config_run_dir(cfg: Any) -> Path:
    """Resolve the run directory that the un-routed core gateway will write."""

    fields = {
        "results_root": OmegaConf.select(cfg, "run.root", default=None),
        "experiment_name": OmegaConf.select(cfg, "experiment.name", default=None),
        "sector": OmegaConf.select(cfg, "experiment.sector", default=None),
        "run_id": OmegaConf.select(cfg, "run.run_id", default=None),
    }
    if any(value is None or not str(value).strip() for value in fields.values()):
        raise RestartRefusal("resolved config lacks a complete run identity")
    try:
        return resolve_run_dir(fields)
    except Exception as exc:
        raise RestartRefusal("resolved config run identity is not admissible") from exc


def _admitted_run_dir(
    *, cfg: Any, row: Mapping[str, Any], run_dir: str | Path | None,
) -> Path:
    """Require the row, config, and optional compatibility argument to agree."""

    try:
        row_dir = resolve_run_dir(row)
    except Exception as exc:
        raise RestartRefusal("manifest row run identity is not admissible") from exc
    config_dir = _config_run_dir(cfg)
    if config_dir.resolve() != row_dir.resolve():
        raise RestartRefusal(
            "config-derived run directory differs from manifest run directory: "
            f"config={config_dir} manifest={row_dir}",
            cause=RestartCause.RUN_DIRECTORY_MISMATCH,
        )
    if run_dir is not None and Path(run_dir).resolve() != config_dir.resolve():
        raise RestartRefusal(
            "explicit run_dir differs from the admitted config-derived run directory: "
            f"explicit={run_dir} admitted={config_dir}",
            cause=RestartCause.RUN_DIRECTORY_MISMATCH,
        )
    return config_dir


def _remove_initial_checkpoint(cfg: Any) -> None:
    """Ensure a restored zero boundary is observed, never rewritten."""

    callbacks = list(OmegaConf.select(cfg, "callbacks", default=[]))
    callbacks = [
        callback for callback in callbacks
        if callback.get("_target_") != "tpen.callback.InitialCheckpoint"
    ]
    OmegaConf.update(cfg, "callbacks", callbacks, merge=False, force_add=True)


def configure_attempt_local_diagnostics(cfg: Any, attempt_dir: str | Path) -> Any:
    """Bind every configured CSV/JSONL diagnostic sink to this attempt."""

    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
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


def select_restart(
    *, cfg: Any, row: Mapping[str, Any], run_dir: str | Path | None = None,
) -> RestartSelection:
    """Select fresh, typed restore, completed, or parked without invoking core."""

    target = _target_iterations(cfg, row)
    resolved_run_dir = _admitted_run_dir(cfg=cfg, row=row, run_dir=run_dir)
    checkpoint_root = resolved_run_dir / "checkpoints"
    try:
        published = discover_published_checkpoint(checkpoint_root)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        return RestartSelection(
            StartDecision.PARK, resolved_run_dir, None, None, None,
            reason=str(exc), cause=RestartCause.DISCOVERY_ERROR,
        )
    try:
        _check_newer_complete(checkpoint_root=checkpoint_root, published=published)
    except RestartPark as exc:
        return RestartSelection(
            StartDecision.PARK, resolved_run_dir, published, None, None,
            reason=str(exc), cause=RestartCause.NEWER_COMPLETE_CHECKPOINT,
        )
    if published is None:
        return RestartSelection(StartDecision.FRESH, resolved_run_dir, None, None, None)

    complete_terminal = (
        published.next_iteration == target
        and published.completed_updates == target
        and (checkpoint_root / f"step_{target:06d}").is_dir()
    )
    if complete_terminal:
        return RestartSelection(StartDecision.COMPLETED, resolved_run_dir, published, None, None)

    intent = _science_intent(cfg, row, target)
    evidence = CheckpointEvidence.from_ref(
        published, science_input=intent.science_input, target_iterations=target
    )
    recovery = select_recovery(intent, phase="recovery", evidence=evidence)
    if recovery.refusal is not None:
        return RestartSelection(
            StartDecision.PARK, resolved_run_dir, published, recovery, None,
            reason=recovery.refusal.message, cause=RestartCause.RECOVERY_REFUSAL,
        )
    try:
        request = OperationalRestoreRequest(recovery)
    except (TypeError, ValueError) as exc:
        return RestartSelection(
            StartDecision.PARK, resolved_run_dir, published, recovery, None,
            reason=str(exc), cause=RestartCause.RESTORE_REQUEST_INVALID,
        )
    return RestartSelection(StartDecision.RESTORE, resolved_run_dir, published, recovery, request)


def _write_outcome(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish an outcome receipt through CX-08's exclusive primitive."""

    try:
        create_json_receipt(path, payload)
    except ReceiptExistsError as exc:
        raise RestartRefusal(f"outcome receipt already exists: {path}") from exc


def _durable_status(run_dir: Path) -> tuple[str | None, dict[str, Any]]:
    """Read status facts after return; never infer success from exit code."""

    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return None, {}
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, {"status_error": str(exc)}
    if not isinstance(payload, dict):
        return None, {"status_error": "status.json is not an object"}
    status = payload.get("status")
    return status if isinstance(status, str) else None, payload


def _active_attempt_lease_paths(attempts: Path) -> list[Path]:
    """Return lease receipts whose owners have not published an outcome.

    A successor checkpoint changes the lease key, so checking only the
    selected predecessor key would allow an in-flight initial attempt to race
    its successor.  The outcome sibling is the CX-08-backed relinquishment
    record for every earlier attempt; absent that sibling, the lease remains
    active and a successor must be refused or explicitly taken over.
    """

    if not attempts.is_dir():
        return []
    superseded: set[str] = set()
    for takeover_path in sorted(attempts.glob("attempt-*.json")):
        if takeover_path.name.endswith(".outcome.json"):
            continue
        try:
            payload = json.loads(takeover_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        takeover = payload.get("takeover") if isinstance(payload, dict) else None
        if not isinstance(takeover, dict) or takeover.get("approved") is not True:
            continue
        names = takeover.get("superseded_attempts", [])
        if isinstance(names, list):
            superseded.update(name for name in names if isinstance(name, str))

    active: list[Path] = []
    for lease_path in sorted(attempts.glob("*.json")):
        if lease_path.name.endswith(".outcome.json"):
            continue
        if lease_path.name in superseded:
            continue
        if not (attempts / f"{lease_path.stem}.outcome.json").is_file():
            active.append(lease_path)
    return active


def _reject_configured_restore_on_fresh(cfg: Any) -> None:
    """Reject concrete raw restore input before acquiring a restart lease."""

    try:
        mode = configured_load_mode(cfg)
    except ValueError as exc:
        raise RestartRefusal("configured load is not an admitted typed restore") from exc
    if mode != "none":
        raise RestartRefusal(
            "fresh restart refuses concrete configured load; use the selected "
            "OperationalRestoreRequest"
        )


def _attempt_number(path: Path) -> int | None:
    match = _ATTEMPT_SEQUENCE.fullmatch(path.name)
    return None if match is None else int(match.group("number"))


def _claim_attempt(
    *, attempts: Path, selection: RestartSelection, identity: Mapping[str, Any],
    takeover: RestartApproval | None, superseded_attempts: list[Path],
) -> tuple[str, Path]:
    """Exclusively claim the next monotonic lineage attempt sequence number."""

    existing = [
        number for path in attempts.glob("attempt-*.json")
        if (number := _attempt_number(path)) is not None
    ]
    number = max(existing, default=-1) + 1
    while True:
        attempt_key = f"attempt-{number:06d}"
        lease_path = attempts / f"{attempt_key}.json"
        lease_payload: dict[str, Any] = {
            "schema": ATTEMPT_RECEIPT_SCHEMA,
            "lineage": dict(identity),
            "predecessor_content_id": selection.content_id,
            "attempt_sequence": number,
            "attempt_key": attempt_key,
            "decision": selection.decision.value,
            "run_dir": str(selection.run_dir),
        }
        if takeover is not None:
            lease_payload["takeover"] = {
                "approved": True,
                "predecessor_content_id": takeover.predecessor_content_id,
                "lineage_id": takeover.lineage_id,
                "config_identity": takeover.config_identity,
                "overlay_identity": takeover.overlay_identity,
                "target_iterations": takeover.target_iterations,
                "terminal_pbs_proof": takeover.terminal_pbs_proof,
                "quiescent": takeover.quiescent,
                "adopted_budget": takeover.adopted_budget,
                "superseded_attempts": [path.name for path in superseded_attempts],
            }
        try:
            create_json_receipt(lease_path, lease_payload)
        except ReceiptExistsError:
            number += 1
            continue
        return attempt_key, lease_path


def _write_claim_refusal(
    *, outcome_path: Path, lease_path: Path, selection: RestartSelection,
    identity: Mapping[str, Any], attempt_key: str, reason: str,
) -> None:
    """Close a claimed sequence with durable refusal evidence before raising."""

    _write_outcome(outcome_path, {
        "schema": OUTCOME_RECEIPT_SCHEMA,
        "lease": str(lease_path),
        "lineage": dict(identity),
        "run_dir": str(selection.run_dir),
        "attempt": attempt_key,
        "decision": selection.decision.value,
        "outcome": "parked",
        "exit_code": None,
        "published_content_id": None,
        "next_iteration": None,
        "logs": {"attempt_dir": str(selection.run_dir / "attempts" / attempt_key)},
        "cause": RestartCause.INVOCATION_ERROR.value,
        "reason": reason,
    })


def execute_restart(
    *, cfg: Any, row: Mapping[str, Any], config_path: str | Path,
    binding: str | None = None, overrides: list[str] | None = None,
    kind: str = "train", overlay: str | None = None,
    overlay_sha256: str | None = None, command: str = "stage1 restart",
    run: Callable[..., int] = run_from_config,
    profile_validator: Callable[[Any], None] = validate_native_recovery_profile,
    run_dir: str | Path | None = None,
    takeover: RestartApproval | None = None,
) -> RestartResult:
    """Run one admitted lineage and publish a truthful per-attempt outcome."""

    if binding is not None:
        row = validate_gateway_binding(
            binding=binding, config=str(config_path), overrides=overrides or [], kind=kind,
            overlay=overlay, overlay_sha256=overlay_sha256,
        )
    target = _target_iterations(cfg, row)
    validate_resolved_train_config(cfg, row)
    profile_validator(cfg)
    selection = select_restart(cfg=cfg, row=row, run_dir=run_dir)
    identity = _identity(row, cfg, target)
    attempts = selection.run_dir / "attempts"
    if selection.decision is StartDecision.FRESH:
        _reject_configured_restore_on_fresh(cfg)

    # Every takeover, including replacement of an initial owner, consumes the
    # same complete approval contract.  The initial marker is a real owner
    # identity for this purpose, not an exception to the gate.
    expected_approval = {
        **identity,
        "predecessor_content_id": selection.content_id,
    }
    active_leases = _active_attempt_lease_paths(attempts)
    if takeover is not None:
        takeover.validate(expected=expected_approval)
    elif active_leases:
        raise RestartRefusal(
            "occupied predecessor lease requires orchestrator takeover approval: "
            + ", ".join(str(path) for path in active_leases)
        )

    attempts.mkdir(parents=True, exist_ok=True)
    # The sequence receipt is the lineage-wide admission barrier.  A takeover
    # claims the next number too and durably names every active predecessor it
    # is authorized to supersede; old receipts are never rewritten.
    superseded_attempts = active_leases if takeover is not None else []
    attempt_key, lease_path = _claim_attempt(
        attempts=attempts, selection=selection, identity=identity,
        takeover=takeover, superseded_attempts=superseded_attempts,
    )
    outcome_path = attempts / f"{attempt_key}.outcome.json"

    # Selection is read-only before the claim, but publication may advance in
    # the gap.  Re-derive it after the sequence claim and reject any other
    # live owner before entering the driver.
    try:
        refreshed_selection = select_restart(cfg=cfg, row=row, run_dir=run_dir)
        if takeover is not None and refreshed_selection.content_id != expected_approval[
            "predecessor_content_id"
        ]:
            raise RestartRefusal("takeover selection changed after lineage claim")
        selection = refreshed_selection
        if selection.decision is StartDecision.FRESH:
            _reject_configured_restore_on_fresh(cfg)
        post_claim_active = [
            path for path in _active_attempt_lease_paths(attempts)
            if path.resolve() != lease_path.resolve()
        ]
        if takeover is not None:
            post_claim_active = [
                path for path in post_claim_active
                if path not in superseded_attempts
            ]
        if post_claim_active:
            reason = (
                "occupied lineage attempt after sequence claim: "
                + ", ".join(str(path) for path in post_claim_active)
            )
            _write_claim_refusal(
                outcome_path=outcome_path, lease_path=lease_path,
                selection=selection, identity=identity,
                attempt_key=attempt_key, reason=reason,
            )
            raise RestartRefusal(reason)
    except RestartRefusal:
        if outcome_path.is_file():
            raise
        _write_claim_refusal(
            outcome_path=outcome_path, lease_path=lease_path,
            selection=selection, identity=identity,
            attempt_key=attempt_key, reason="selection invalid after lineage claim",
        )
        raise

    attempt_dir = attempts / attempt_key
    try:
        if selection.decision is StartDecision.RESTORE:
            _remove_initial_checkpoint(cfg)
        if selection.decision in {StartDecision.FRESH, StartDecision.RESTORE}:
            configure_attempt_local_diagnostics(cfg, attempt_dir)
        exit_code: int | None = None
        if selection.decision is StartDecision.FRESH:
            exit_code = run(cfg, config_path=str(config_path), command=command)
        elif selection.decision is StartDecision.RESTORE:
            if selection.restore_request is None:
                raise RestartRefusal("restore decision lacks OperationalRestoreRequest")
            exit_code = run(
                cfg, config_path=str(config_path), command=command,
                restore_request=selection.restore_request,
            )
        status, status_payload = _durable_status(selection.run_dir)
        try:
            published_after = discover_published_checkpoint(selection.run_dir / "checkpoints")
        except Exception as exc:
            raise RestartPark(str(exc) or RestartCause.POST_RUN_DISCOVERY_ERROR.value) from exc
        cause = selection.cause
        reason = selection.reason
        if selection.decision is StartDecision.PARK:
            outcome = "parked"
            cause = cause or RestartCause.INVOCATION_ERROR
            reason = reason or cause.value
        elif selection.decision is StartDecision.COMPLETED:
            outcome = "completed" if status in {None, "completed"} else status
        elif published_after is not None and published_after.next_iteration == target:
            outcome = "completed" if status in {None, "completed"} else status
        elif status in {"failed", "planned_yield"}:
            outcome = status
        else:
            outcome = "parked"
            cause = RestartCause.NO_PUBLICATION
            reason = (
                "no published checkpoint at admitted run directory: "
                f"{selection.run_dir}"
            )
        receipt = {
            "schema": OUTCOME_RECEIPT_SCHEMA,
            "lease": str(lease_path),
            "lineage": identity,
            "run_dir": str(selection.run_dir),
            "attempt": attempt_key,
            "decision": selection.decision.value,
            "outcome": outcome,
            "exit_code": exit_code,
            "status": status,
            "status_facts": status_payload,
            "published_content_id": None if published_after is None else published_after.content_id,
            "next_iteration": None if published_after is None else published_after.next_iteration,
            "logs": {"attempt_dir": str(attempt_dir)},
            "cause": None if cause is None else cause.value,
            "reason": reason,
        }
    except Exception as exc:
        cause = selection.cause or (
            RestartCause.POST_RUN_DISCOVERY_ERROR
            if isinstance(exc, RestartPark)
            else RestartCause.INVOCATION_ERROR
        )
        reason = str(exc).strip() or cause.value
        receipt = {
            "schema": OUTCOME_RECEIPT_SCHEMA,
            "lease": str(lease_path),
            "lineage": identity,
            "run_dir": str(selection.run_dir),
            "attempt": attempt_key,
            "decision": selection.decision.value,
            "outcome": "parked",
            "exit_code": None,
            "published_content_id": None,
            "next_iteration": None,
            "logs": {"attempt_dir": str(attempt_dir)},
            "cause": cause.value,
            "reason": reason,
        }
        outcome = "parked"
        exit_code = None
    _write_outcome(outcome_path, receipt)
    return RestartResult(selection, lease_path, outcome_path, outcome, exit_code, receipt)


def restart_train_from_row(
    *, row: Mapping[str, Any], config_path: str | Path, overrides: list[str],
    binding: str, overlay: str | None = None, overlay_sha256: str | None = None,
    command: str = "stage1 restart",
) -> RestartResult:
    """Convenience gateway used by tests and a future scheduler wrapper."""

    resolved = validate_gateway_binding(
        binding=binding, config=str(config_path), overrides=overrides, kind="train",
        overlay=overlay, overlay_sha256=overlay_sha256,
    )
    cfg = load_config(str(config_path), overrides)
    if overlay is not None:
        from experiments.atomistic.stage1.protocol import compose_train_overlay

        compose_train_overlay(cfg, overlay=overlay, expected_sha256=overlay_sha256)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    prepare_initial_checkpoint_config(cfg)
    return execute_restart(
        cfg=cfg, row=resolved, config_path=config_path, command=command,
    )


__all__ = [
    "ATTEMPT_RECEIPT_SCHEMA", "OUTCOME_RECEIPT_SCHEMA", "RestartApproval",
    "RestartPark", "RestartRefusal", "RestartResult", "RestartSelection",
    "RestartCause", "StartDecision", "configure_attempt_local_diagnostics", "execute_restart",
    "restart_train_from_row", "select_restart",
]
