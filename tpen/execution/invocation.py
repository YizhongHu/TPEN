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
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from tpen.checkpoint.artifact import LATEST_JSON, list_complete_checkpoints, resolve_checkpoint_dir
from tpen.checkpoint.catalog import PUBLICATION_CATALOG_FILENAME, read_publications, reconcile_publication
from tpen.checkpoint.reference import CheckpointRef
from tpen.checkpoint.schema import read_manifest
from tpen.config_schema import ClosedSchemaError

from ._codec import decode_fields
from .batch import ReadyBatch, ReadyItem
from .config import (
    canonical_payload_bytes,
    payload_digest,
    recovery_selection_digest,
    source_config_digest,
)
from .recovery import (
    CheckpointEvidence,
    RecoveryDecision,
    RecoveryOutcome,
    RecoveryPhase,
    RefusalCode,
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

    if isinstance(exc, ReceiptBindingError) or isinstance(exc, _CONFIG_REFUSAL_TYPES):
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
    declared_overlays: Mapping[str, Any] = field(default_factory=dict)

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
        if not isinstance(self.declared_overlays, Mapping):
            raise TypeError("declared_overlays must be a mapping")
        object.__setattr__(self, "declared_overlays", dict(self.declared_overlays))

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
            "declared_overlays": dict(self.declared_overlays),
        }

    @classmethod
    def from_mapping(cls, data: Any) -> "WorkerStartReceipt":
        """Deserialize a closed, typed worker receipt record."""

        fields = {
            "lineage_id": str, "batch_id": str, "logical_invocation_id": str,
            "invocation_id": str, "fresh_output_id": str, "output_path": str,
            "ready_batch_digest": str, "source_config_digest": str,
            "payload_digest": str, "science_input_digest": str,
            "target_iterations": int, "next_iteration": int, "completed_updates": int,
            "recovery_outcome": str, "recovery_selection_digest": str,
        }
        decode_fields(
            data, fields,
            optional={"checkpoint_content_id": (str, type(None)), "declared_overlays": Mapping},
            schema=cls.schema, name="WorkerStartReceipt",
        )
        return cls(
            **{name: data[name] for name in fields},
            checkpoint_content_id=data.get("checkpoint_content_id"),
            declared_overlays=data.get("declared_overlays", {}),
        )


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
    refusal_code: str | None = None
    exception_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_receipt_digest", _digest(self.start_receipt_digest, "start_receipt_digest"))
        object.__setattr__(self, "outcome", InvocationOutcome(self.outcome))
        object.__setattr__(self, "status", CoreStatus(self.status))
        for name in ("target_iterations", "next_iteration", "completed_updates"):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        object.__setattr__(self, "reason", str(self.reason))
        if self.refusal_code is not None:
            object.__setattr__(self, "refusal_code", RefusalCode(self.refusal_code).value)
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
            "refusal_code": self.refusal_code,
            "exception_type": self.exception_type,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> "WorkerOutcomeReceipt":
        """Deserialize a closed, typed worker receipt record."""

        fields = {
            "start_receipt_digest": str, "outcome": str, "status": str,
            "target_iterations": int, "next_iteration": int, "completed_updates": int,
            "reason": str,
        }
        decode_fields(
            data, fields,
            optional={"refusal_code": (str, type(None)), "exception_type": (str, type(None))},
            schema=cls.schema, name="WorkerOutcomeReceipt",
        )
        return cls(
            **{name: data[name] for name in fields},
            refusal_code=data.get("refusal_code"), exception_type=data.get("exception_type"),
        )


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
        if decision.refusal.code in {
            RefusalCode.INCOMPATIBLE,
            RefusalCode.UNSUPPORTED,
            RefusalCode.MODEL_ONLY,
        }:
            return InvocationOutcome.TERMINAL, decision.refusal.message
        # Missing or incomplete checkpoint evidence remains a reconciliation
        # condition.  Later layers must park it without silently restarting.
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
    """Fsync JSON and exclusively publish it without replacing a prior receipt."""

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
        os.link(temporary, path)
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


def _reconcile_unpublished_checkpoint(checkpoint_root: Path) -> CheckpointRef | None:
    """Repair missing publication indexes for the latest committed directory."""

    candidates = [CheckpointRef.from_directory(path) for path in list_complete_checkpoints(checkpoint_root)]
    if not candidates:
        return None
    ref = max(candidates, key=lambda candidate: candidate.next_iteration)
    # Gate the full-state payload before publishing any execution completion
    # evidence. The checkpoint package owns every non-destructive index write.
    read_manifest(ref.checkpoint_dir / "manifest.json", mode="train_resume")
    return reconcile_publication(checkpoint_root, ref.checkpoint_dir)


def _committed_checkpoint(run_dir: Path) -> CheckpointRef | None:
    """Read the latest committed full-state checkpoint through checkpoint APIs."""

    checkpoint_root = run_dir / "checkpoints"
    if not checkpoint_root.is_dir():
        return None

    catalog_path = checkpoint_root / PUBLICATION_CATALOG_FILENAME
    if catalog_path.is_file():
        publications = read_publications(catalog_path)
        # A readable catalog is an index, not the authority over newer
        # committed checkpoint directories.  Scan all complete directories
        # through CheckpointRef and the train-resume manifest gate, then let
        # the checkpoint package reconcile any missing publication indexes.
        # Keep the catalog read above outside the fallback so a corrupt
        # catalog still refuses instead of silently falling back.
        ref = _reconcile_unpublished_checkpoint(checkpoint_root)
        if ref is None and publications:
            ref = max(publications, key=lambda candidate: candidate.next_iteration)
    elif not (checkpoint_root / LATEST_JSON).is_file():
        # A COMPLETE directory may have landed before either publication
        # index. Reconcile it instead of resolving the root as a checkpoint.
        ref = _reconcile_unpublished_checkpoint(checkpoint_root)
    else:
        # ``latest.json`` is the checkpoint package's durable convenience
        # pointer.  The resolver also requires the COMPLETE marker, so a
        # temporary or half-written directory cannot become completion facts.
        ref = CheckpointRef.from_directory(resolve_checkpoint_dir(checkpoint_root))

    if ref is None:
        return None
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
    directory creation, a fsynced temporary receipt, exclusive hard-link
    publication, and parent-directory fsync make the receipt durable before
    the callback is entered. A crash before publication can leave a namespace
    and a temporary file; a crash after start publication can leave a start
    receipt with no outcome. Both states park during reconciliation.
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
        declared_overlays: Mapping[str, Any] | None = None,
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
        if source_config_digest(intent.source_config) != batch.source_config_digest:
            raise ReceiptBindingError(
                "science intent source configuration is bound to a different ready batch"
            )
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
            declared_overlays=dict(declared_overlays or {}),
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
            refusal_code=(decision.refusal.code.value if decision.refusal is not None else None),
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
        declared_overlays: Mapping[str, Any] | None = None,
    ) -> WorkerInvocation:
        """Run configured TPEN once after publishing the worker-start receipt."""

        def run_once(decision: RecoveryDecision, output_path: Path) -> CoreOutcomeFacts:
            return run_bound_configured_once(
                cfg, intent=intent, decision=decision, output_path=output_path,
                declared_overlays=declared_overlays, **dict(run_kwargs or {}),
            )

        return self.start(
            batch=batch,
            intent=intent,
            phase=phase,
            run_once=run_once,
            item=item,
            evidence=evidence,
            attempt_id=attempt_id,
            fault_injector=fault_injector,
            declared_overlays=declared_overlays,
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
        start_path = namespace / START_RECEIPT_NAME
        if not start_path.is_file():
            return None
        start = WorkerStartReceipt.from_mapping(_read_json(start_path))
        path = namespace / OUTCOME_RECEIPT_NAME
        if path.is_file():
            outcome = WorkerOutcomeReceipt.from_mapping(_read_json(path))
            if outcome.start_receipt_digest != start.receipt_id:
                raise ReceiptBindingError(
                    "worker outcome receipt is bound to a different start receipt"
                )
            return outcome
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


_RECOVERY_CONFIG_PATHS = frozenset({"load", "runner.load"})
_OPERATIONAL_OVERLAY_PATHS = frozenset({"run.root", "trainer.max_steps"})


def _config_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten config-shaped mappings without resolving arbitrary targets."""

    if isinstance(value, Mapping):
        leaves: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_config_leaves(child, child_prefix))
        return leaves
    return {prefix: value}


def _is_config_path(path: str, prefixes: frozenset[str]) -> bool:
    """Return whether a leaf is a named path or one of its descendants."""

    return any(path == prefix or path.startswith(f"{prefix}.") for prefix in prefixes)


def _has_concrete_config_path(config: Mapping[str, Any], path: str) -> bool:
    """Distinguish effective no-load values from concrete INITIAL loads."""

    from omegaconf import OmegaConf
    from omegaconf.errors import OmegaConfBaseException

    try:
        # Resolve only the load node, including references such as ${load};
        # unrelated science targets remain opaque captured syntax.
        value = OmegaConf.select(OmegaConf.create(config), path, default=None)
        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except OmegaConfBaseException as exc:
        raise ReceiptBindingError(f"cannot resolve configured load at {path!r}") from exc
    if value is None:
        return False
    if isinstance(value, Mapping):
        return value.get("path") is not None or value.get("mode") not in (None, "none")
    return True


def _materialize_configured_worker_config(
    cfg: Any,
    *,
    intent: ScienceIntent,
    decision: RecoveryDecision,
    output_path: Path,
    declared_overlays: Mapping[str, Any] | None,
) -> Any:
    """Build a runner config from captured intent plus explicit worker overlays.

    The submitter's captured source configuration is the immutable base.  The
    supplied config is checked only as a syntax witness: source-owned leaves
    must match, recovery ``load`` is owned by the E1 decision, and the only
    other accepted differences are named operational overlays.  OmegaConf is
    used for data copying only; no ``_target_`` value is resolved here.
    """

    from omegaconf import OmegaConf

    # ``ScienceIntent`` stores nested mapping proxies for immutability; use
    # its JSON-shaped thawed representation rather than asking OmegaConf to
    # construct a config from those proxy objects.
    source = OmegaConf.to_container(
        OmegaConf.create(intent.to_dict()["source_config"]), resolve=False
    )
    supplied = OmegaConf.to_container(OmegaConf.create(cfg), resolve=False)
    if not isinstance(source, Mapping) or not isinstance(supplied, Mapping):
        raise ReceiptBindingError("captured and supplied worker configs must be mappings")

    overlays = dict(declared_overlays or {})
    unknown_overlays = set(overlays).difference(_OPERATIONAL_OVERLAY_PATHS)
    if unknown_overlays:
        raise ReceiptBindingError(
            "worker config contains undeclared operational overlays: "
            + ", ".join(sorted(unknown_overlays))
        )
    overlay_values = _config_leaves(overlays)
    source_values = _config_leaves(source)
    supplied_values = _config_leaves(supplied)
    missing_captured = [
        path
        for path in source_values
        if not _is_config_path(path, _RECOVERY_CONFIG_PATHS)
        and path not in supplied_values
    ]
    if missing_captured:
        raise ReceiptBindingError(
            "worker config omits captured source leaves: "
            + ", ".join(sorted(missing_captured))
        )
    for path, value in supplied_values.items():
        if _is_config_path(path, _RECOVERY_CONFIG_PATHS) or path == "run.root":
            continue
        expected = source_values.get(path, _MISSING_CONFIG_VALUE)
        if expected is _MISSING_CONFIG_VALUE:
            expected = overlay_values.get(path, _MISSING_CONFIG_VALUE)
        # A legacy configured caller may carry additional schema leaves that
        # were not part of the captured science snapshot.  Those leaves remain
        # opaque to this binding check; captured leaves and explicit overlays
        # are the only values this gateway authorizes or rewrites.
        if expected is not _MISSING_CONFIG_VALUE and value != expected:
            raise ReceiptBindingError(
                f"worker config leaf {path!r} differs from immutable source intent"
            )

    for path in _RECOVERY_CONFIG_PATHS:
        if decision.load is None and (
            _has_concrete_config_path(source, path)
            or _has_concrete_config_path(supplied, path)
        ):
            raise ReceiptBindingError(
                "INITIAL configured run contains an undeclared concrete runner.load"
            )

    # The captured source is authoritative for every leaf it contains.  A
    # configured caller may still supply the surrounding runner schema (for
    # example, legacy configs whose capture records only the science-facing
    # subset), so retain those already-validated opaque leaves in the supplied
    # mapping.  No supplied leaf can replace a captured value, and no target
    # is resolved while making this copy.
    worker_cfg = OmegaConf.create(supplied)
    for path, value in source_values.items():
        if _is_config_path(path, _RECOVERY_CONFIG_PATHS):
            continue
        if path in supplied_values:
            OmegaConf.update(worker_cfg, path, value, merge=False, force_add=True)
    for path, value in overlays.items():
        OmegaConf.update(worker_cfg, path, value, merge=False, force_add=True)
    OmegaConf.update(worker_cfg, "run.root", str(output_path), merge=False, force_add=True)
    return worker_cfg


def run_bound_configured_once(
    cfg: Any,
    *,
    intent: ScienceIntent,
    decision: RecoveryDecision,
    output_path: Path,
    declared_overlays: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> CoreOutcomeFacts:
    """Run a configured adapter callback through the gateway's intent binding.

    Call this from the ``run_once`` seam of a worker-start gateway, after its
    durable start receipt. Backend adapters share this materialization with
    ``start_configured`` without importing or constructing a science object.
    """

    worker_cfg = _materialize_configured_worker_config(
        cfg, intent=intent, decision=decision, output_path=output_path,
        declared_overlays=declared_overlays,
    )
    # The legacy direct entry retains its default; adapter calls must retain
    # the actual core exception for deterministic refusal classification.
    kwargs["raise_exceptions"] = True
    run_configured_once(worker_cfg, decision=decision, output_path=output_path, **kwargs)
    return _core_outcome_from_artifacts(output_path, decision=decision)


_MISSING_CONFIG_VALUE = object()


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
        # The configured runner consumes ``runner.load``.  Keep the
        # top-level mirror for callers whose config exposes it, but never let
        # a concrete runner value shadow the selected E1 overlay.
        OmegaConf.update(worker_cfg, "runner.load", decision.load, merge=False, force_add=True)
        if "load" in worker_cfg:
            OmegaConf.update(worker_cfg, "load", decision.load, merge=False)
        effective_runner_load = OmegaConf.select(worker_cfg, "runner.load", default=None)
        if OmegaConf.is_config(effective_runner_load):
            effective_runner_load = OmegaConf.to_container(effective_runner_load, resolve=True)
        if effective_runner_load != decision.load:
            raise ValueError("worker overlay did not reach the configured runner.load")
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
    "run_bound_configured_once",
    "run_configured_once",
    "validate_start_receipt",
]
