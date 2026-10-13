"""Worker-start materialization and backend-neutral outcome contracts.

This module is the boundary used by an actual worker process.  It owns the
filesystem transaction which gives every start a fresh output namespace and a
durable receipt before the configured TPEN entry point is called.  It does not
submit work or import a backend, and it does not infer scientific success from
an operating-system or scheduler exit code.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from tpen.checkpoint.artifact import resolve_checkpoint_dir
from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME, read_publications
from tpen.checkpoint.reference import CheckpointRef
from tpen.checkpoint.schema import read_manifest
from tpen.config_schema import ClosedSchemaError

from .batch import ReadyBatch, ReadyItem
from .config import canonical_payload_bytes, payload_digest, recovery_selection_digest
from .recovery import (
    CheckpointEvidence,
    RecoveryDecision,
    RecoveryOutcome,
    RecoveryPhase,
    ScienceIntent,
    select_recovery,
)

START_RECEIPT_SCHEMA = "tpen.execution/receipt-worker-start-v1"
OUTCOME_RECEIPT_SCHEMA = "tpen.execution/receipt-worker-outcome-v1"
START_RECEIPT_NAME = "start.json"
OUTCOME_RECEIPT_NAME = "outcome.json"

# These are the only configuration/refusal classes this gateway treats as a
# deterministic terminal result.  A training-time ValueError (or another
# ordinary application exception) remains unknown and is parked.  Checkpoint
# gates predate a shared exception base, so their named owner modules are
# checked below in addition to this explicit configuration type.
_CONFIG_REFUSAL_TYPES = (ClosedSchemaError,)
_CHECKPOINT_GATE_MODULES = frozenset(
    {
        "tpen.checkpoint.artifact",
        "tpen.checkpoint.catalog",
        "tpen.checkpoint.manifest",
        "tpen.checkpoint.payload",
        "tpen.checkpoint.reference",
        "tpen.checkpoint.restore",
        "tpen.checkpoint.schema",
    }
)
_CHECKPOINT_GATE_EXCEPTION_TYPES = (KeyError, TypeError, ValueError)


class InvocationOutcome(str, Enum):
    """Durable classification of one worker invocation."""

    COMPLETED = "completed"
    TERMINAL = "terminal"
    PARKED = "parked"


class CoreStatus(str, Enum):
    """Status values supplied by the configured TPEN core."""

    COMPLETED = "completed"
    TERMINAL = "terminal"
    PARKED = "parked"
    UNKNOWN = "unknown"


class ReceiptBindingError(ValueError):
    """Raised when a receipt is reused for a different immutable binding."""


def _text(value: Any, name: str) -> str:
    """Validate one required durable text field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _path_component(value: Any, name: str) -> str:
    """Validate an opaque identity before using it as a path component."""

    value = _text(value, name)
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{name} must be one opaque path component")
    return value


def _digest(value: Any, name: str) -> str:
    """Validate one lowercase SHA-256 digest."""

    value = _text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    """Validate one counter without coercing booleans or floats."""

    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _check_schema(data: Mapping[str, Any], expected: str) -> None:
    """Reject a receipt serialized under another schema."""

    if not isinstance(data, Mapping) or data.get("schema") != expected:
        raise ValueError(f"receipt schema is not {expected!r}")


def _innermost_tpen_module(exc: BaseException) -> str | None:
    """Return the innermost TPEN module in the exception traceback.

    The innermost TPEN frame is the closest available owner of a refusal.  It
    avoids classifying an application error merely because it passed through
    an outer checkpoint-related frame, while still allowing the checkpoint
    gate's own ``ValueError``/``KeyError``/``TypeError`` to be terminal.
    """

    owner: str | None = None
    traceback = exc.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__")
        if isinstance(module_name, str) and module_name.startswith("tpen."):
            owner = module_name
        traceback = traceback.tb_next
    return owner


def _is_terminal_core_refusal(exc: BaseException) -> bool:
    """Classify only named core config/checkpoint refusal paths as terminal."""

    if isinstance(exc, _CONFIG_REFUSAL_TYPES):
        return True
    return (
        isinstance(exc, _CHECKPOINT_GATE_EXCEPTION_TYPES)
        and _innermost_tpen_module(exc) in _CHECKPOINT_GATE_MODULES
    )


@dataclass(frozen=True, slots=True)
class CoreOutcomeFacts:
    """Core-owned facts needed to classify a worker result.

    Parameters
    ----------
    status : CoreStatus or str
        Explicit core classification.  A plain ``int`` returned by
        ``run_from_config`` is deliberately not a completion fact.
    target_iterations, next_iteration, completed_updates : int
        The fixed target and the two distinct progress counters reported by
        core.  Both counters are retained in the receipt; neither is inferred
        from the other.
    reason : str, optional
        Durable explanation for a terminal, parked, or unknown result.
    exception_type : str, optional
        Exception identity when a configured run failed before reporting core
        facts.
    """

    status: CoreStatus | str
    target_iterations: int
    next_iteration: int
    completed_updates: int
    reason: str = ""
    exception_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CoreStatus(self.status))
        for name in ("target_iterations", "next_iteration", "completed_updates"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if self.exception_type is not None:
            object.__setattr__(self, "exception_type", _text(self.exception_type, "exception_type"))

    @classmethod
    def completed(
        cls, *, target_iterations: int, next_iteration: int, completed_updates: int
    ) -> "CoreOutcomeFacts":
        """Build explicit completion facts from the configured core."""

        return cls(
            status=CoreStatus.COMPLETED,
            target_iterations=target_iterations,
            next_iteration=next_iteration,
            completed_updates=completed_updates,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe facts."""

        return {
            "status": self.status.value,
            "target_iterations": self.target_iterations,
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
            "reason": self.reason,
            "exception_type": self.exception_type,
        }


@dataclass(frozen=True, slots=True)
class WorkerStartReceipt:
    """Durable binding written before configured-run construction begins."""

    schema: ClassVar[str] = START_RECEIPT_SCHEMA
    lineage_id: str
    batch_id: str
    logical_invocation_id: str
    invocation_id: str
    fresh_output_id: str
    output_path: str
    ready_batch_digest: str
    source_config_digest: str
    payload_digest: str
    science_input_digest: str
    target_iterations: int
    next_iteration: int
    completed_updates: int
    recovery_outcome: str
    recovery_selection_digest: str
    checkpoint_content_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "lineage_id",
            "batch_id",
            "logical_invocation_id",
            "invocation_id",
            "fresh_output_id",
            "output_path",
            "recovery_outcome",
            "recovery_selection_digest",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "ready_batch_digest",
            "source_config_digest",
            "payload_digest",
            "science_input_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("target_iterations", "next_iteration", "completed_updates"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        if self.checkpoint_content_id is not None:
            object.__setattr__(self, "checkpoint_content_id", _digest(self.checkpoint_content_id, "checkpoint_content_id"))

    @property
    def receipt_id(self) -> str:
        """Return the immutable digest of this receipt."""

        return payload_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe receipt data."""

        return {
            "schema": self.schema,
            "lineage_id": self.lineage_id,
            "batch_id": self.batch_id,
            "logical_invocation_id": self.logical_invocation_id,
            "invocation_id": self.invocation_id,
            "fresh_output_id": self.fresh_output_id,
            "output_path": self.output_path,
            "ready_batch_digest": self.ready_batch_digest,
            "source_config_digest": self.source_config_digest,
            "payload_digest": self.payload_digest,
            "science_input_digest": self.science_input_digest,
            "target_iterations": self.target_iterations,
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
            "recovery_outcome": self.recovery_outcome,
            "recovery_selection_digest": self.recovery_selection_digest,
            "checkpoint_content_id": self.checkpoint_content_id,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorkerStartReceipt":
        """Deserialize one worker-start receipt."""

        if not isinstance(data, Mapping):
            raise TypeError("worker-start receipt must be a mapping")
        _check_schema(data, cls.schema)
        return cls(**{name: data.get(name) for name in (
            "lineage_id", "batch_id", "logical_invocation_id",
            "invocation_id", "fresh_output_id", "output_path", "ready_batch_digest",
            "source_config_digest", "payload_digest", "science_input_digest",
            "target_iterations", "next_iteration", "completed_updates",
            "recovery_outcome", "recovery_selection_digest", "checkpoint_content_id",
        )})


@dataclass(frozen=True, slots=True)
class WorkerOutcomeReceipt:
    """Durable core outcome bound to one worker-start receipt."""

    schema: ClassVar[str] = OUTCOME_RECEIPT_SCHEMA
    start_receipt_digest: str
    outcome: InvocationOutcome | str
    status: CoreStatus | str
    target_iterations: int
    next_iteration: int
    completed_updates: int
    reason: str
    exception_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_receipt_digest", _digest(self.start_receipt_digest, "start_receipt_digest"))
        object.__setattr__(self, "outcome", InvocationOutcome(self.outcome))
        object.__setattr__(self, "status", CoreStatus(self.status))
        for name in ("target_iterations", "next_iteration", "completed_updates"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(self, "reason", str(self.reason))
        if self.exception_type is not None:
            object.__setattr__(self, "exception_type", _text(self.exception_type, "exception_type"))

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe receipt data."""

        return {
            "schema": self.schema,
            "start_receipt_digest": self.start_receipt_digest,
            "outcome": self.outcome.value,
            "status": self.status.value,
            "target_iterations": self.target_iterations,
            "next_iteration": self.next_iteration,
            "completed_updates": self.completed_updates,
            "reason": self.reason,
            "exception_type": self.exception_type,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "WorkerOutcomeReceipt":
        """Deserialize one worker-outcome receipt."""

        if not isinstance(data, Mapping):
            raise TypeError("worker-outcome receipt must be a mapping")
        _check_schema(data, cls.schema)
        return cls(**{name: data.get(name) for name in (
            "start_receipt_digest", "outcome", "status",
            "target_iterations", "next_iteration", "completed_updates",
            "reason", "exception_type",
        )})


@dataclass(frozen=True, slots=True)
class WorkerInvocation:
    """Result of one gateway start and its optional configured-run call."""

    start: WorkerStartReceipt
    outcome: WorkerOutcomeReceipt
    run_result: Any = None

    @property
    def output_path(self) -> Path:
        """Return this start's fresh science output namespace."""

        return Path(self.start.output_path)


def validate_start_receipt(receipt: WorkerStartReceipt, batch: ReadyBatch, item: ReadyItem) -> None:
    """Refuse a start receipt whose immutable binding changed."""

    if receipt.lineage_id != batch.lineage_id or receipt.batch_id != batch.batch_id:
        raise ReceiptBindingError("worker-start receipt is bound to a different lineage or batch")
    if receipt.ready_batch_digest != batch.identity:
        raise ReceiptBindingError("worker-start receipt is bound to a different ready batch")
    if receipt.source_config_digest != batch.source_config_digest:
        raise ReceiptBindingError("worker-start receipt is bound to a different source configuration")
    if receipt.payload_digest != batch.payload_digest:
        raise ReceiptBindingError("worker-start receipt is bound to a different payload")
    if receipt.science_input_digest != batch.science_input_digest:
        raise ReceiptBindingError("worker-start receipt is bound to different science input")
    if receipt.logical_invocation_id != item.invocation_id:
        raise ReceiptBindingError("worker-start receipt names a different invocation")
    if receipt.target_iterations != batch.target_iterations:
        raise ReceiptBindingError("worker-start receipt uses a different target horizon")


def classify_core_outcome(
    facts: CoreOutcomeFacts | None,
    *,
    decision: RecoveryDecision,
) -> tuple[InvocationOutcome, str]:
    """Classify core facts without treating scheduler success as completion.

    A completion is accepted only when core explicitly reports completion and
    ``next_iteration`` reaches the fixed target.  ``completed_updates`` is a
    distinct fact: skipped updates may make it smaller, but it may never be
    negative or exceed ``next_iteration``.  This is an execution contract,
    not proof that scientific work happened: a known optimizer/model binding
    hazard can still advance counters and write checkpoints while doing no
    meaningful update.
    """

    if decision.refusal is not None:
        return InvocationOutcome.PARKED, decision.refusal.message
    if facts is None:
        return InvocationOutcome.PARKED, "core did not provide completion facts; park for reconciliation"
    if facts.status is CoreStatus.COMPLETED:
        if facts.target_iterations != decision.target_iterations:
            return InvocationOutcome.TERMINAL, "core reported a different fixed target horizon"
        if facts.completed_updates > facts.next_iteration:
            return (
                InvocationOutcome.TERMINAL,
                "core completion facts have completed_updates greater than next_iteration",
            )
        if facts.next_iteration != decision.target_iterations:
            # Core claims completion, but its durable iteration counter has
            # not reached this immutable target.  That is ambiguous for a
            # recovery controller, so reconcile it instead of retrying or
            # declaring success.
            return (
                InvocationOutcome.PARKED,
                "core reported completion before the fixed target horizon; park for reconciliation",
            )
        return InvocationOutcome.COMPLETED, "core reported target completion"
    if facts.status is CoreStatus.TERMINAL:
        return InvocationOutcome.TERMINAL, facts.reason or "core reported a deterministic terminal failure"
    return InvocationOutcome.PARKED, facts.reason or "core outcome is ambiguous; park for reconciliation"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write and fsync JSON, then publish it with a same-directory rename."""

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temporary, flags, 0o600)
    try:
        encoded = canonical_payload_bytes(payload)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(encoded)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON receipt mapping."""

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"receipt is not a mapping: {path}")
    return value


def _run_status_artifact(output_path: Path) -> tuple[Path, dict[str, Any]] | None:
    """Read the single durable run-status artifact emitted by configured TPEN."""

    status_paths = tuple(output_path.rglob("status.json"))
    if not status_paths:
        return None
    if len(status_paths) != 1:
        raise ValueError(
            f"configured run wrote {len(status_paths)} status artifacts under {output_path}"
        )
    return status_paths[0].parent, _read_json(status_paths[0])


def _committed_checkpoint(run_dir: Path) -> CheckpointRef | None:
    """Read the latest committed full-state checkpoint through checkpoint APIs."""

    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return None

    catalog_path = checkpoint_root / PUBLICATION_CATALOG_FILENAME
    if catalog_path.is_file():
        publications = read_publications(catalog_path)
        if not publications:
            return None
        ref = max(publications, key=lambda candidate: candidate.next_iteration)
    else:
        # ``latest.json`` is the checkpoint package's durable convenience
        # pointer.  The resolver also requires the COMPLETE marker, so a
        # temporary or half-written directory cannot become completion facts.
        ref = CheckpointRef.from_directory(resolve_checkpoint_dir(checkpoint_root))

    # CheckpointRef owns identity and byte validation; the schema reader owns
    # the mode-specific full-state gate.  Do not infer counters from filenames
    # or trainer JSON in this execution layer.
    read_manifest(ref.checkpoint_dir / "manifest.json", mode="train_resume")
    ref.validate()
    return ref


def _core_outcome_from_artifacts(
    output_path: Path, *, decision: RecoveryDecision
) -> CoreOutcomeFacts:
    """Materialize core outcome facts from the run and checkpoint artifacts.

    A configured run's integer return value is a process result, not a core
    completion claim.  Completion therefore requires the run-owned completed
    status and a committed train-resume checkpoint carrying both counters.
    """

    status_result = _run_status_artifact(output_path)
    if status_result is None:
        return CoreOutcomeFacts(
            status=CoreStatus.UNKNOWN,
            target_iterations=decision.target_iterations,
            next_iteration=decision.next_iteration,
            completed_updates=decision.completed_updates,
            reason="core wrote no durable run status; park for reconciliation",
        )
    run_dir, status = status_result
    run_status = status.get("status")
    if run_status == "failed":
        return CoreOutcomeFacts(
            status=CoreStatus.TERMINAL,
            target_iterations=decision.target_iterations,
            next_iteration=decision.next_iteration,
            completed_updates=decision.completed_updates,
            reason="core run status is failed",
        )
    if run_status != "completed":
        return CoreOutcomeFacts(
            status=CoreStatus.UNKNOWN,
            target_iterations=decision.target_iterations,
            next_iteration=decision.next_iteration,
            completed_updates=decision.completed_updates,
            reason=f"core run status is {run_status!r}; park for reconciliation",
        )

    try:
        ref = _committed_checkpoint(run_dir)
    except OSError as exc:
        return CoreOutcomeFacts(
            status=CoreStatus.UNKNOWN,
            target_iterations=decision.target_iterations,
            next_iteration=decision.next_iteration,
            completed_updates=decision.completed_updates,
            reason=f"checkpoint artifact I/O failed; park for reconciliation: {exc}",
            exception_type=type(exc).__name__,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return CoreOutcomeFacts(
            status=CoreStatus.TERMINAL if _is_terminal_core_refusal(exc) else CoreStatus.UNKNOWN,
            target_iterations=decision.target_iterations,
            next_iteration=decision.next_iteration,
            completed_updates=decision.completed_updates,
            reason=(
                f"invalid committed final checkpoint: {exc}"
                if _is_terminal_core_refusal(exc)
                else f"checkpoint artifact refusal is ambiguous; park for reconciliation: {exc}"
            ),
            exception_type=type(exc).__name__,
        )
    if ref is None:
        return CoreOutcomeFacts(
            status=CoreStatus.PARKED,
            target_iterations=decision.target_iterations,
            next_iteration=decision.next_iteration,
            completed_updates=decision.completed_updates,
            reason="no committed final checkpoint; park for reconciliation",
        )
    return CoreOutcomeFacts(
        status=CoreStatus.COMPLETED,
        target_iterations=decision.target_iterations,
        next_iteration=ref.next_iteration,
        completed_updates=ref.completed_updates,
    )


class WorkerStartGateway:
    """Materialize one actual worker start before calling configured TPEN.

    The shared-filesystem guarantee is deliberately narrow: exclusive
    directory creation, a fsynced temporary receipt, same-directory rename,
    and parent-directory fsync make the published receipt durable before the
    callback is entered.  A crash before rename can leave a namespace and a
    temporary file; a crash after rename but before outcome publication leaves
    a start receipt with no outcome.  Both states park during reconciliation.
    """

    def __init__(self, output_root: str | Path, *, id_factory: Callable[[], str] | None = None) -> None:
        self.output_root = Path(output_root)
        self._id_factory = id_factory or (lambda: uuid4().hex)

    def start(
        self,
        *,
        batch: ReadyBatch,
        intent: ScienceIntent,
        phase: RecoveryPhase | str,
        run_once: Callable[[RecoveryDecision, Path], Any],
        item: ReadyItem | None = None,
        evidence: CheckpointEvidence | Mapping[str, Any] | str | Path | None = None,
        attempt_id: str | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> WorkerInvocation:
        """Select recovery, publish a start receipt, then call TPEN once.

        ``run_once`` is called exactly once only after the start receipt is
        durable.  It receives the E1 decision (including its existing
        ``train_resume`` load overlay) and the fresh output path.  It must
        return :class:`CoreOutcomeFacts` when it wants a completed outcome;
        legacy integer return values are intentionally parked as unknown.
        """

        if not isinstance(intent, ScienceIntent):
            raise TypeError("intent must be a ScienceIntent")
        if intent.science_input_digest != batch.science_input_digest:
            raise ReceiptBindingError("science intent is bound to a different ready batch")
        if intent.target_iterations != batch.target_iterations:
            raise ReceiptBindingError("science intent uses a different target horizon")
        selected_item = item or self._single_item(batch)
        if selected_item not in batch.items:
            raise ReceiptBindingError("worker item is absent from the ready batch")
        if not callable(run_once):
            raise TypeError("run_once must be callable")

        decision = select_recovery(intent, phase=phase, evidence=evidence)
        lineage_id = _path_component(batch.lineage_id, "batch.lineage_id")
        batch_id = _path_component(batch.batch_id, "batch.batch_id")
        actual_id = _path_component(attempt_id or self._id_factory(), "attempt_id")
        namespace = self.output_root / lineage_id / batch_id / actual_id
        try:
            namespace.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ReceiptBindingError("worker-start namespace already exists; refuse receipt reuse") from exc
        output_path = namespace / "science"
        output_path.mkdir()
        if fault_injector is not None:
            fault_injector("namespace_created")

        start = WorkerStartReceipt(
            lineage_id=batch.lineage_id,
            batch_id=batch.batch_id,
            logical_invocation_id=selected_item.invocation_id,
            invocation_id=actual_id,
            fresh_output_id=self._id_factory(),
            output_path=str(output_path),
            ready_batch_digest=batch.identity,
            source_config_digest=batch.source_config_digest,
            payload_digest=batch.payload_digest,
            science_input_digest=batch.science_input_digest,
            target_iterations=decision.target_iterations,
            next_iteration=decision.next_iteration,
            completed_updates=decision.completed_updates,
            recovery_outcome=decision.outcome.value,
            recovery_selection_digest=recovery_selection_digest(decision),
            checkpoint_content_id=(
                None if decision.checkpoint is None else decision.checkpoint.content_id
            ),
        )
        if fault_injector is not None:
            fault_injector("before_receipt_publish")
        _atomic_write_json(namespace / START_RECEIPT_NAME, start.to_dict())
        if fault_injector is not None:
            fault_injector("receipt_published")

        run_result: Any = None
        facts: CoreOutcomeFacts | None = None
        exception_type: str | None = None
        try:
            if decision.refusal is None:
                run_result = run_once(decision, output_path)
                facts = run_result if isinstance(run_result, CoreOutcomeFacts) else None
            else:
                run_result = None
        except Exception as exc:
            exception_type = type(exc).__name__
            terminal_refusal = _is_terminal_core_refusal(exc)
            facts = CoreOutcomeFacts(
                status=CoreStatus.TERMINAL if terminal_refusal else CoreStatus.UNKNOWN,
                target_iterations=decision.target_iterations,
                next_iteration=decision.next_iteration,
                completed_updates=decision.completed_updates,
                reason=(
                    str(exc)
                    if terminal_refusal
                    else (
                        f"transient I/O during configured run; park for reconciliation: {exc}"
                        if isinstance(exc, OSError)
                        else "configured run raised outside a named deterministic refusal; park for reconciliation"
                    )
                ),
                exception_type=exception_type,
            )
        outcome, reason = classify_core_outcome(facts, decision=decision)
        outcome_receipt = WorkerOutcomeReceipt(
            start_receipt_digest=start.receipt_id,
            outcome=outcome,
            status=(facts.status if facts is not None else CoreStatus.UNKNOWN),
            target_iterations=(facts.target_iterations if facts is not None else decision.target_iterations),
            next_iteration=(facts.next_iteration if facts is not None else decision.next_iteration),
            completed_updates=(facts.completed_updates if facts is not None else decision.completed_updates),
            reason=reason,
            exception_type=exception_type if exception_type is not None else (facts.exception_type if facts else None),
        )
        _atomic_write_json(namespace / OUTCOME_RECEIPT_NAME, outcome_receipt.to_dict())
        return WorkerInvocation(start=start, outcome=outcome_receipt, run_result=run_result)

    def start_configured(
        self,
        *,
        cfg: Any,
        batch: ReadyBatch,
        intent: ScienceIntent,
        phase: RecoveryPhase | str,
        item: ReadyItem | None = None,
        evidence: CheckpointEvidence | Mapping[str, Any] | str | Path | None = None,
        attempt_id: str | None = None,
        fault_injector: Callable[[str], None] | None = None,
        run_kwargs: Mapping[str, Any] | None = None,
    ) -> WorkerInvocation:
        """Run configured TPEN once after publishing the worker-start receipt."""

        kwargs = dict(run_kwargs or {})
        # The direct API keeps its historical ``raise_exceptions=False``
        # default.  A worker gateway must see the named core refusal itself so
        # it can classify it; an integer ``1`` would erase that distinction.
        kwargs["raise_exceptions"] = True

        def run_once(decision: RecoveryDecision, output_path: Path) -> Any:
            run_configured_once(cfg, decision=decision, output_path=output_path, **kwargs)
            return _core_outcome_from_artifacts(output_path, decision=decision)

        return self.start(
            batch=batch,
            intent=intent,
            phase=phase,
            run_once=run_once,
            item=item,
            evidence=evidence,
            attempt_id=attempt_id,
            fault_injector=fault_injector,
        )

    @staticmethod
    def _single_item(batch: ReadyBatch) -> ReadyItem:
        """Require an unambiguous item when callers omit one."""

        if len(batch.items) != 1:
            raise ValueError("item is required for a batch with multiple ready items")
        return batch.items[0]

    @staticmethod
    def reconcile(namespace: str | Path) -> WorkerOutcomeReceipt | None:
        """Read or durably park a start whose outcome is still ambiguous.

        A namespace without a published start receipt returns ``None`` because
        there is no immutable binding from which an outcome receipt could be
        safely fabricated.  A published start with no outcome is explicitly
        parked and written as an outcome receipt; reconciliation never calls
        the configured runner or resubmits work.
        """

        namespace = Path(namespace)
        path = namespace / OUTCOME_RECEIPT_NAME
        if path.is_file():
            return WorkerOutcomeReceipt.from_mapping(_read_json(path))
        start_path = namespace / START_RECEIPT_NAME
        if not start_path.is_file():
            return None
        start = WorkerStartReceipt.from_mapping(_read_json(start_path))
        parked = WorkerOutcomeReceipt(
            start_receipt_digest=start.receipt_id,
            outcome=InvocationOutcome.PARKED,
            status=CoreStatus.UNKNOWN,
            target_iterations=start.target_iterations,
            next_iteration=start.next_iteration,
            completed_updates=start.completed_updates,
            reason="worker start has no durable core outcome; park for reconciliation",
        )
        _atomic_write_json(path, parked.to_dict())
        return parked


def run_configured_once(cfg: Any, *, decision: RecoveryDecision, output_path: Path, **kwargs: Any) -> Any:
    """Call legacy ``run_from_config`` once with a worker-local overlay.

    The import is intentionally lazy: importing ``tpen.execution`` does not
    import Hydra, torch, or arbitrary science constructors.  The copied config
    receives the E1 load overlay and unique output root only for this worker;
    callers using :func:`tpen.run.run_from_config` directly are unchanged.
    """

    from omegaconf import OmegaConf
    from tpen.run import run_from_config

    worker_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if decision.load is not None:
        if OmegaConf.select(worker_cfg, "load", default=None) is not None:
            OmegaConf.update(worker_cfg, "load", decision.load, merge=False)
        else:
            OmegaConf.update(worker_cfg, "runner.load", decision.load, merge=False, force_add=True)
    OmegaConf.update(worker_cfg, "run.root", str(output_path), merge=False, force_add=True)
    return run_from_config(worker_cfg, **kwargs)


__all__ = [
    "CoreOutcomeFacts",
    "CoreStatus",
    "InvocationOutcome",
    "OUTCOME_RECEIPT_NAME",
    "OUTCOME_RECEIPT_SCHEMA",
    "ReceiptBindingError",
    "START_RECEIPT_NAME",
    "START_RECEIPT_SCHEMA",
    "WorkerInvocation",
    "WorkerOutcomeReceipt",
    "WorkerStartGateway",
    "WorkerStartReceipt",
    "classify_core_outcome",
    "run_configured_once",
    "validate_start_receipt",
]
