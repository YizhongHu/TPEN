"""Balsam lineage registration and worker-start adapter.

The adapter owns only the small durable boundary between one prepared
``ReadyBatch`` and a Balsam client.  It does not implement a scheduler,
reaper, retry controller, or backend lifecycle.  The real optional Balsam
package is intentionally imported by a deployment seam, never at module import
time; source tests can therefore use the protocol below without the optional
dependency installed.

Balsam API facts relied on
---------------------------
The historical registration and launched-application observations are
recorded as capability evidence in TO lane item ``34b3e37f`` note
``formal-review-r1-findings`` and E5 item ``2692a8b0`` note
``e5b-attempt1-disposition``.  They show a persistent Job acknowledgement
with an exposed ``id`` and a launched application receiving that identity; the
seam therefore accepts an integer or string id and records its canonical
string form.  These notes are evidence, not an installed-deployment claim.

The compute-side proxy/site prerequisites and the decision to keep live API
observation on the submitter/reconciler side are recorded in E5 item
``2692a8b0`` notes ``e5-facility-and-prerequisites`` and
``e5b-final-not-established``.  Worker start consequently does not observe
the live API.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Callable, Protocol
from uuid import uuid4

from .batch import ReadyBatch, ReadyItem
from .config import ExecutionConfig, canonical_payload_bytes, payload_digest
from .invocation import WorkerInvocation, WorkerStartGateway
from .recovery import CheckpointEvidence, RecoveryPhase, ScienceIntent

BALSAM_INTENT_SCHEMA = "tpen.execution/balsam-registration-intent-v1"
BALSAM_REGISTRATION_SCHEMA = "tpen.execution/balsam-registration-v1"
BALSAM_PARK_SCHEMA = "tpen.execution/balsam-registration-park-v1"
BALSAM_REFUSAL_SCHEMA = "tpen.execution/balsam-registration-refusal-v1"
BALSAM_OBSERVATION_SCHEMA = "tpen.execution/balsam-observation-v1"
BALSAM_WORKER_PARK_SCHEMA = "tpen.execution/balsam-worker-park-v1"
BALSAM_OBSERVATION_PARK_SCHEMA = "tpen.execution/balsam-observation-park-v1"
REGISTRATION_INTENT_NAME = "registration-intent.json"
REGISTRATION_RECEIPT_NAME = "registration.json"
REGISTRATION_PARK_NAME = "registration-park.json"
REGISTRATION_REFUSAL_NAME = "registration-refusal.json"
OBSERVATION_NAME = "observation.json"
OBSERVATION_PARK_NAME = "observation-park.json"
WORKER_PARK_NAME = "worker-park.json"
REGISTRATION_LOCK_NAME = "registration.lock"


class BalsamAdapterError(RuntimeError):
    """Base class for typed adapter refusals and backend ambiguity."""


class BalsamBindingError(BalsamAdapterError, ValueError):
    """Raised when a lineage is reused with a changed immutable intent."""


class BalsamPermanentRefusal(BalsamAdapterError, ValueError):
    """Raised for an explicit, deterministic backend registration refusal."""


class BalsamAcknowledgementUnknown(BalsamAdapterError):
    """Raised when a registration call may have succeeded but its ack is lost."""


class BalsamJobState(str, Enum):
    """Backend observation states relevant to this adapter boundary."""

    ACTIVE = "active"
    ALLOCATION_LOST = "allocation_lost"
    COMPLETED = "completed"
    PERMANENT_FAILURE = "permanent_failure"
    UNKNOWN = "unknown"


class BalsamAdapterStatus(str, Enum):
    """Result of the adapter boundary before or after an E3 invocation."""

    STARTED = "started"
    PARKED = "parked"


@dataclass(frozen=True, slots=True)
class BalsamJobObservation:
    """One observation returned by the backend client seam.

    Parameters
    ----------
    job_id : str
        Opaque persistent backend Job identity.
    state : BalsamJobState or str
        Backend-owned lifecycle observation.  ``allocation_lost`` is a normal
        Balsam lifecycle observation, not a core completion or failure claim.
    reason : str, optional
        Durable explanation for an unknown or permanent observation.
    metadata : mapping, optional
        Backend observation data kept separate from science intent.
    """

    job_id: str | int
    state: BalsamJobState | str
    reason: str = ""
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        job_id = _canonical_job_id(self.job_id)
        if job_id is None:
            raise ValueError("job_id must be a non-empty string or integer")
        object.__setattr__(self, "job_id", job_id)
        object.__setattr__(self, "state", BalsamJobState(self.state))
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        metadata = {} if self.metadata is None else dict(self.metadata)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe backend observation mapping."""

        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "reason": self.reason,
            "metadata": dict(self.metadata or {}),
        }


class BalsamClient(Protocol):
    """Minimal lazy-client seam implemented by a Balsam deployment adapter.

    A deployment may wrap ``balsam.api`` here.  The core adapter never imports
    or resolves that optional package; only the submitter-side method calls the
    protocol's registration operation.
    """

    def register_job(
        self, *, lineage_id: str, batch: ReadyBatch, backend_options: Mapping[str, Any]
    ) -> Any:
        """Register one persistent backend Job and return its opaque ack."""

    def observe_job(self, job_id: str) -> BalsamJobObservation | Mapping[str, Any]:
        """Observe the existing Job without creating or resubmitting it."""


def load_balsam_api() -> Any:
    """Load the optional Balsam API only when a deployment requests it.

    The returned module is deliberately not wrapped into a submission policy
    here: deployment code adapts its installed Balsam client to
    :class:`BalsamClient`.  Importing :mod:`tpen.execution` therefore remains
    safe on submitters and in environments without Balsam installed.
    """

    try:
        from balsam import api
    except ImportError as exc:  # pragma: no cover - Balsam is not a test dependency
        raise BalsamPermanentRefusal("the optional Balsam API is not installed") from exc
    return api


def _text(value: Any, name: str) -> str:
    """Validate an opaque durable text field."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _path_component(value: Any, name: str) -> str:
    """Validate an identity before using it as a filesystem component."""

    value = _text(value, name)
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{name} must be one opaque path component")
    return value


def _digest(value: Any, name: str) -> str:
    """Validate a lowercase SHA-256 digest."""

    value = _text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Publish one JSON record with a same-directory durable rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(canonical_payload_bytes(data))
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


def _create_exclusive_marker(path: Path) -> bool:
    """Create a durable one-line marker, returning false when it exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write("registration-owner\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        if fd >= 0:
            os.close(fd)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one durable adapter record."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"adapter record is not a mapping: {path}")
    return data


@dataclass(frozen=True, slots=True)
class BalsamRegistrationIntent:
    """Immutable registration request written before the client call."""

    schema: ClassVar[str] = BALSAM_INTENT_SCHEMA
    lineage_id: str
    batch_id: str
    ready_batch_digest: str
    source_config_digest: str
    payload_digest: str
    backend_options_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "lineage_id", _text(self.lineage_id, "lineage_id"))
        object.__setattr__(self, "batch_id", _text(self.batch_id, "batch_id"))
        for name in (
            "ready_batch_digest",
            "source_config_digest",
            "payload_digest",
            "backend_options_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic durable intent mapping."""

        return {
            "schema": self.schema,
            "lineage_id": self.lineage_id,
            "batch_id": self.batch_id,
            "ready_batch_digest": self.ready_batch_digest,
            "source_config_digest": self.source_config_digest,
            "payload_digest": self.payload_digest,
            "backend_options_digest": self.backend_options_digest,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BalsamRegistrationIntent":
        """Deserialize and validate a registration intent."""

        if data.get("schema") != cls.schema:
            raise ValueError(f"registration intent schema is not {cls.schema!r}")
        return cls(
            lineage_id=data.get("lineage_id", ""),
            batch_id=data.get("batch_id", ""),
            ready_batch_digest=data.get("ready_batch_digest", ""),
            source_config_digest=data.get("source_config_digest", ""),
            payload_digest=data.get("payload_digest", ""),
            backend_options_digest=data.get("backend_options_digest", ""),
        )


@dataclass(frozen=True, slots=True)
class BalsamRegistrationReceipt:
    """Durable acknowledgement for one persistent Balsam Job."""

    schema: ClassVar[str] = BALSAM_REGISTRATION_SCHEMA
    lineage_id: str
    batch_id: str
    job_id: str
    ready_batch_digest: str
    source_config_digest: str
    payload_digest: str
    backend_options_digest: str

    def __post_init__(self) -> None:
        for name in ("lineage_id", "batch_id", "job_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "ready_batch_digest",
            "source_config_digest",
            "payload_digest",
            "backend_options_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic durable acknowledgement mapping."""

        return {
            "schema": self.schema,
            "lineage_id": self.lineage_id,
            "batch_id": self.batch_id,
            "job_id": self.job_id,
            "ready_batch_digest": self.ready_batch_digest,
            "source_config_digest": self.source_config_digest,
            "payload_digest": self.payload_digest,
            "backend_options_digest": self.backend_options_digest,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BalsamRegistrationReceipt":
        """Deserialize and validate a registration acknowledgement."""

        if data.get("schema") != cls.schema:
            raise ValueError(f"registration receipt schema is not {cls.schema!r}")
        return cls(
            lineage_id=data.get("lineage_id", ""),
            batch_id=data.get("batch_id", ""),
            job_id=data.get("job_id", ""),
            ready_batch_digest=data.get("ready_batch_digest", ""),
            source_config_digest=data.get("source_config_digest", ""),
            payload_digest=data.get("payload_digest", ""),
            backend_options_digest=data.get("backend_options_digest", ""),
        )


@dataclass(frozen=True, slots=True)
class BalsamStartResult:
    """Result of one adapter start and its routed E3 invocation."""

    status: BalsamAdapterStatus
    registration: BalsamRegistrationReceipt | None
    observation: BalsamJobObservation | None
    invocation: WorkerInvocation | None
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", BalsamAdapterStatus(self.status))
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")


def _canonical_job_id(value: Any) -> str | None:
    """Normalize the opaque Balsam Job id without accepting booleans."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _job_id(ack: Any) -> str | None:
    """Extract and canonicalize a Job identity from a client acknowledgement."""

    value = ack
    if isinstance(ack, Mapping):
        value = next((ack.get(key) for key in ("job_id", "id") if key in ack), None)
    elif not isinstance(ack, (str, int)) or isinstance(ack, bool):
        value = next((getattr(ack, key, None) for key in ("job_id", "id") if hasattr(ack, key)), None)
    return _canonical_job_id(value)


def _observation(value: BalsamJobObservation | Mapping[str, Any], job_id: str) -> BalsamJobObservation:
    """Normalize one backend observation without importing Balsam types."""

    expected_job_id = _canonical_job_id(job_id)
    if expected_job_id is None:
        raise BalsamBindingError("registration receipt has an invalid Job identity")
    if isinstance(value, BalsamJobObservation):
        if value.job_id != expected_job_id:
            raise BalsamBindingError("backend observation names a different Job")
        return value
    if not isinstance(value, Mapping):
        raise TypeError("observe_job must return BalsamJobObservation or a mapping")
    observed_job_id = _canonical_job_id(value.get("job_id", expected_job_id))
    if observed_job_id != expected_job_id:
        raise BalsamBindingError("backend observation names a different Job")
    return BalsamJobObservation(
        job_id=observed_job_id,
        state=value.get("state", BalsamJobState.UNKNOWN),
        reason=value.get("reason", ""),
        metadata=value.get("metadata", {}),
    )


class BalsamAdapter:
    """Bind one immutable batch to one Balsam Job and route starts through E3.

    Parameters
    ----------
    execution : ExecutionConfig
        Configuration selecting the ``"balsam"`` backend lego.
    client : BalsamClient
        Explicit client seam.  A deployment wrapper may lazily construct the
        real Balsam client outside this module.
    gateway : WorkerStartGateway
        E3 worker-start gateway used for every actual process start.
    receipt_root : path-like
        Shared filesystem root for the lineage registration records.
    """

    def __init__(
        self,
        execution: ExecutionConfig,
        client: BalsamClient,
        gateway: WorkerStartGateway,
        receipt_root: str | Path,
    ) -> None:
        if not isinstance(execution, ExecutionConfig):
            raise TypeError("execution must be an ExecutionConfig")
        if execution.backend != "balsam":
            raise BalsamPermanentRefusal(
                f"BalsamAdapter requires backend='balsam', got {execution.backend!r}"
            )
        if not hasattr(client, "register_job") or not hasattr(client, "observe_job"):
            raise TypeError("client must implement register_job and observe_job")
        if not callable(getattr(gateway, "start", None)):
            raise TypeError("gateway must provide a callable start method")
        self.execution = execution
        self.client = client
        self.gateway = gateway
        self.receipt_root = Path(receipt_root)

    def start(
        self,
        *,
        batch: ReadyBatch,
        intent: ScienceIntent,
        phase: RecoveryPhase | str,
        run_once: Callable[..., Any],
        job_id: str | int,
        item: ReadyItem | None = None,
        evidence: CheckpointEvidence | Mapping[str, Any] | str | Path | None = None,
        attempt_id: str | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> BalsamStartResult:
        """Route one Balsam-launched process through E3 without backend calls.

        ``job_id`` is supplied by the Balsam launch wrapper.  This worker-side
        method deliberately never calls ``register_job`` or ``observe_job``:
        registration and backend observation belong to the submitter or
        reconciler.  A missing, parked, or mismatched durable registration is
        therefore a durable worker park rather than an implicit submission.
        """

        self._validate_batch(batch)
        launched_job_id = _canonical_job_id(job_id)
        if launched_job_id is None:
            reason = "launched Balsam Job identity is missing or invalid; park for reconciliation"
            self._write_worker_park(batch, reason)
            return BalsamStartResult(
                status=BalsamAdapterStatus.PARKED,
                registration=None,
                observation=None,
                invocation=None,
                reason=reason,
            )
        registration = self._load_registration(batch)
        if registration is None:
            reason = "registration acknowledgement is unavailable or parked; worker must not submit"
            self._write_worker_park(batch, reason)
            return BalsamStartResult(
                status=BalsamAdapterStatus.PARKED,
                registration=None,
                observation=None,
                invocation=None,
                reason=reason,
            )
        if registration.job_id != launched_job_id:
            reason = "launched Balsam Job identity differs from the durable registration; park"
            self._write_worker_park(batch, reason)
            return BalsamStartResult(
                status=BalsamAdapterStatus.PARKED,
                registration=registration,
                observation=None,
                invocation=None,
                reason=reason,
            )

        # Backend state is intentionally absent here.  E3 owns checkpoint
        # selection, fresh output, start ordering, and the science outcome.
        invocation = self.gateway.start(
            batch=batch,
            intent=intent,
            phase=phase,
            run_once=run_once,
            item=item,
            evidence=evidence,
            attempt_id=attempt_id,
            fault_injector=fault_injector,
        )
        return BalsamStartResult(
            status=BalsamAdapterStatus.STARTED,
            registration=registration,
            observation=None,
            invocation=invocation,
        )

    def register_lineage(self, batch: ReadyBatch) -> BalsamRegistrationReceipt | None:
        """Register one persistent Job from the submitter side only.

        This is the sole adapter method that can call ``register_job``.  The
        durable intent is published before that call, and an absent or
        ambiguous acknowledgement is parked without blind resubmission.
        """

        self._validate_batch(batch)
        return self._ensure_registration(batch)

    def observe_lineage(self, batch: ReadyBatch) -> BalsamJobObservation | None:
        """Observe and durably record backend state from a reconciler.

        Observation is intentionally separate from worker start.  Allocation
        loss remains a backend lifecycle state; a permanent backend refusal is
        raised as a typed refusal, while unknown client failures are parked.
        """

        self._validate_batch(batch)
        registration = self._load_registration(batch)
        if registration is None:
            self._write_observation_park(batch, "registration acknowledgement is unavailable; observe later")
            return None
        try:
            observed = _observation(self.client.observe_job(registration.job_id), registration.job_id)
        except BalsamPermanentRefusal:
            raise
        except Exception as exc:
            self._write_observation_park(
                batch, f"backend Job observation is unknown; park for reconciliation: {exc}"
            )
            return None
        self._write_observation(batch, observed)
        if observed.state is BalsamJobState.PERMANENT_FAILURE:
            raise BalsamPermanentRefusal(observed.reason or "Balsam reported permanent Job failure")
        return observed

    def _validate_batch(self, batch: ReadyBatch) -> None:
        """Refuse a batch not bound to this configured backend lineage."""

        if not isinstance(batch, ReadyBatch):
            raise TypeError("batch must be a ReadyBatch")
        if batch.execution.backend != "balsam":
            raise BalsamBindingError("ready batch does not select the Balsam backend")
        for name in ("lineage_id", "source_config_digest", "payload_digest"):
            if getattr(batch, name) != getattr(self.execution, name):
                raise BalsamBindingError(f"batch {name} differs from execution configuration")

    def _intent(self, batch: ReadyBatch) -> BalsamRegistrationIntent:
        """Build the immutable intent that identifies one registration."""

        return BalsamRegistrationIntent(
            lineage_id=batch.lineage_id,
            batch_id=batch.batch_id,
            ready_batch_digest=batch.identity,
            source_config_digest=batch.source_config_digest,
            payload_digest=batch.payload_digest,
            backend_options_digest=payload_digest(self.execution.backend_options),
        )

    def _paths(self, lineage_id: str) -> tuple[Path, Path, Path, Path, Path]:
        """Return registration paths for one lineage."""

        directory = self.receipt_root / _path_component(lineage_id, "lineage_id")
        return (
            directory / REGISTRATION_INTENT_NAME,
            directory / REGISTRATION_RECEIPT_NAME,
            directory / REGISTRATION_PARK_NAME,
            directory / REGISTRATION_REFUSAL_NAME,
            directory / REGISTRATION_LOCK_NAME,
        )

    def _lineage_directory(self, lineage_id: str) -> Path:
        """Return the validated shared-filesystem directory for a lineage."""

        return self.receipt_root / _path_component(lineage_id, "lineage_id")

    def _load_registration(self, batch: ReadyBatch) -> BalsamRegistrationReceipt | None:
        """Load only a complete, binding-valid registration for worker use."""

        expected = self._intent(batch)
        intent_path, receipt_path, park_path, refusal_path, lock_path = self._paths(batch.lineage_id)
        if receipt_path.is_file() and not intent_path.is_file():
            raise BalsamPermanentRefusal("registration acknowledgement exists without its intent")
        if not intent_path.is_file():
            return None
        stored_intent = BalsamRegistrationIntent.from_mapping(_read_json(intent_path))
        if stored_intent != expected:
            raise BalsamBindingError("lineage is already bound to a different Balsam intent")
        if not lock_path.is_file():
            raise BalsamPermanentRefusal("registration intent exists without its ownership marker")
        if refusal_path.is_file():
            refusal = _read_json(refusal_path)
            if refusal.get("schema") != BALSAM_REFUSAL_SCHEMA:
                raise BalsamPermanentRefusal("registration refusal record has an unsupported schema")
            raise BalsamPermanentRefusal(str(refusal.get("reason", "Balsam registration was refused")))
        if not receipt_path.is_file():
            return None
        receipt = BalsamRegistrationReceipt.from_mapping(_read_json(receipt_path))
        self._validate_receipt(receipt, expected)
        return receipt

    def _ensure_registration(self, batch: ReadyBatch) -> BalsamRegistrationReceipt | None:
        """Load or create the one acknowledged persistent Job for a lineage."""

        expected = self._intent(batch)
        intent_path, receipt_path, park_path, refusal_path, lock_path = self._paths(batch.lineage_id)
        owns_registration = False
        if intent_path.is_file():
            stored_intent = BalsamRegistrationIntent.from_mapping(_read_json(intent_path))
            if stored_intent != expected:
                raise BalsamBindingError("lineage is already bound to a different Balsam intent")
            if not lock_path.is_file():
                raise BalsamPermanentRefusal("registration intent exists without its ownership marker")
        elif receipt_path.is_file():
            raise BalsamPermanentRefusal("registration acknowledgement exists without its intent")
        else:
            # The marker remains after acknowledgement.  It is a tiny
            # shared-filesystem claim preventing two process starts from both
            # calling the client before either acknowledgement is visible.
            if _create_exclusive_marker(lock_path):
                owns_registration = True
                _atomic_write_json(intent_path, expected.to_dict())
            else:
                self._write_park(batch, "another registration owner is unresolved; park for reconciliation")
                return None

        # Both submitter and worker use the same loader, so a validated
        # acknowledgement outranks a stale pre-ack race park while refusal
        # precedence remains identical on both paths.
        registration = self._load_registration(batch)
        if registration is not None:
            return registration
        if park_path.is_file():
            # A prior ambiguous client call is durable evidence against a
            # second registration.  Do not let a fresh process silently retry.
            return None

        # An intent and ownership marker without an acknowledgement may be a
        # crashed process between the durable intent and the client call (or
        # after the client call but before the ack was persisted).  It is
        # deliberately not safe to guess which; preserve the no-resubmit park.
        if not owns_registration and intent_path.is_file() and not receipt_path.is_file() and lock_path.is_file():
            self._write_park(batch, "registration acknowledgement is unresolved; park without resubmitting")
            return None

        try:
            ack = self.client.register_job(
                lineage_id=batch.lineage_id,
                batch=batch,
                backend_options=self.execution.backend_options,
            )
        except BalsamPermanentRefusal as exc:
            self._write_refusal(batch, str(exc) or "Balsam client refused registration")
            raise
        except BalsamAcknowledgementUnknown:
            self._write_park(batch, "registration acknowledgement was lost; park without resubmitting")
            return None
        except Exception as exc:
            # A client exception after the intent is published is ambiguous:
            # the request may have reached Balsam before the local failure.
            self._write_park(batch, f"registration acknowledgement is unknown: {exc}")
            return None

        job_id = _job_id(ack)
        if job_id is None:
            self._write_park(batch, "client returned no Job acknowledgement; park without resubmitting")
            return None
        receipt = BalsamRegistrationReceipt(
            lineage_id=expected.lineage_id,
            batch_id=expected.batch_id,
            job_id=job_id,
            ready_batch_digest=expected.ready_batch_digest,
            source_config_digest=expected.source_config_digest,
            payload_digest=expected.payload_digest,
            backend_options_digest=expected.backend_options_digest,
        )
        self._validate_receipt(receipt, expected)
        _atomic_write_json(receipt_path, receipt.to_dict())
        return receipt

    @staticmethod
    def _validate_receipt(
        receipt: BalsamRegistrationReceipt, expected: BalsamRegistrationIntent
    ) -> None:
        """Refuse a durable acknowledgement with changed immutable bindings."""

        for name in (
            "lineage_id",
            "batch_id",
            "ready_batch_digest",
            "source_config_digest",
            "payload_digest",
            "backend_options_digest",
        ):
            if getattr(receipt, name) != getattr(expected, name):
                raise BalsamBindingError(f"registration receipt {name} does not match its intent")

    def _write_park(self, batch: ReadyBatch, reason: str) -> None:
        """Persist an ambiguous registration outcome without resubmission."""

        _, _, park_path, _, _ = self._paths(batch.lineage_id)
        _atomic_write_json(
            park_path,
            {
                "schema": BALSAM_PARK_SCHEMA,
                "lineage_id": batch.lineage_id,
                "batch_id": batch.batch_id,
                "ready_batch_digest": batch.identity,
                "reason": reason,
            },
        )

    def _write_worker_park(self, batch: ReadyBatch, reason: str) -> None:
        """Persist a worker-start park without blocking later registration."""

        path = self._lineage_directory(batch.lineage_id) / WORKER_PARK_NAME
        _atomic_write_json(
            path,
            {
                "schema": BALSAM_WORKER_PARK_SCHEMA,
                "lineage_id": batch.lineage_id,
                "batch_id": batch.batch_id,
                "ready_batch_digest": batch.identity,
                "reason": reason,
            },
        )

    def _write_observation(self, batch: ReadyBatch, observed: BalsamJobObservation) -> None:
        """Persist one backend observation for a submitter-side reconciler."""

        path = self._lineage_directory(batch.lineage_id) / OBSERVATION_NAME
        _atomic_write_json(
            path,
            {
                "schema": BALSAM_OBSERVATION_SCHEMA,
                "lineage_id": batch.lineage_id,
                "batch_id": batch.batch_id,
                "ready_batch_digest": batch.identity,
                "registration_job_id": observed.job_id,
                "state": observed.state.value,
                "reason": observed.reason,
                "metadata": dict(observed.metadata or {}),
            },
        )

    def _write_observation_park(self, batch: ReadyBatch, reason: str) -> None:
        """Persist an observation uncertainty without blocking worker start."""

        path = self._lineage_directory(batch.lineage_id) / OBSERVATION_PARK_NAME
        _atomic_write_json(
            path,
            {
                "schema": BALSAM_OBSERVATION_PARK_SCHEMA,
                "lineage_id": batch.lineage_id,
                "batch_id": batch.batch_id,
                "ready_batch_digest": batch.identity,
                "reason": reason,
            },
        )

    def _write_refusal(self, batch: ReadyBatch, reason: str) -> None:
        """Persist a deterministic registration refusal for later starts."""

        _, _, _, refusal_path, _ = self._paths(batch.lineage_id)
        _atomic_write_json(
            refusal_path,
            {
                "schema": BALSAM_REFUSAL_SCHEMA,
                "lineage_id": batch.lineage_id,
                "batch_id": batch.batch_id,
                "ready_batch_digest": batch.identity,
                "reason": reason,
            },
        )


__all__ = [
    "BALSAM_INTENT_SCHEMA",
    "BALSAM_OBSERVATION_SCHEMA",
    "BALSAM_OBSERVATION_PARK_SCHEMA",
    "BALSAM_PARK_SCHEMA",
    "BALSAM_REGISTRATION_SCHEMA",
    "BALSAM_REFUSAL_SCHEMA",
    "BALSAM_WORKER_PARK_SCHEMA",
    "BalsamAcknowledgementUnknown",
    "BalsamAdapter",
    "BalsamAdapterError",
    "BalsamAdapterStatus",
    "BalsamBindingError",
    "BalsamClient",
    "BalsamJobObservation",
    "BalsamJobState",
    "BalsamPermanentRefusal",
    "BalsamRegistrationIntent",
    "BalsamRegistrationReceipt",
    "BalsamStartResult",
    "load_balsam_api",
    "REGISTRATION_INTENT_NAME",
    "OBSERVATION_NAME",
    "OBSERVATION_PARK_NAME",
    "REGISTRATION_PARK_NAME",
    "REGISTRATION_RECEIPT_NAME",
    "REGISTRATION_LOCK_NAME",
    "REGISTRATION_REFUSAL_NAME",
    "WORKER_PARK_NAME",
]
