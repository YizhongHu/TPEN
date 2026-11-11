"""Versioned remote-leaf wire and worker-owned member execution (S3 Layer A).

This module owns the boundary between a dispatch controller and a leaf that
may run in another process or on another node.  Only canonical JSON bytes
cross that boundary:

* :class:`LeafRequest` is the controller-built, immutable, primitive-only
  request for one fenced attempt.  It binds the group, task, attempt,
  allocation, backend job and slot identities, the configured payload and its
  digest, the fenced output paths, and the prepared science environment.
* :func:`run_leaf` is the leaf callable.  Given only the request bytes, it
  reconstructs the prepared environment, runs science through the unchanged
  :data:`~tpen_executor.runners.RUN_SHIM` worker path, then runs public
  evidence inspection in the same leaf, and returns :class:`LeafResult` bytes.
* :class:`LeafResult` carries only plain outcome, evidence and teardown
  facts.  :func:`validate_leaf_result` is the controller-side check that a
  result answers exactly its request.

The leaf has no ``DirectoryStore``, lifecycle, manifest, controller callback,
scheduler or retry authority: it never begins, finishes or accepts an
attempt, it launches science at most once, and it returns facts rather than
publishing them.  Durable publication stays with the controller.  The S2b
worker-owned GPU sampler and slot quarantine are unchanged because the leaf
launches through the same :class:`~tpen_executor.workers.SubprocessWorker`;
their worker-only keys travel in the prepared environment exactly as for a
local launch.  Before that launch the leaf applies local dispatch slot
admission against the durable quarantine markers under the request's store
root, so a slot is never admitted on the strength of its overlay alone.  It
also refuses a placed slot whose declared CPUs are not all permitted to the
worker it runs on (S3 Layer B review F2): the controller's affinity says
nothing about a remote worker.

Layer A does not wire this module into any dispatcher, so local and
ThreadPool dispatch are unchanged.  Remote submission (Layer B) and remote
cancellation/teardown acknowledgement (Layer C) are out of scope: a leaf
interrupted by a ``BaseException`` returns no result.

Contract mutant controls R1-R5 are listed in
``executor/tests/test_s3_transport_wire.py`` (``S3A_MUTANT_TABLE``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
import socket
import sys
from types import MappingProxyType
from typing import Any, ClassVar

from .._codec import _freeze, canonical_bytes, decode_json, digest, fields, thaw
from .._records import frozen_record
from ..environments import PreparedPythonEnvironment, SourceScript
from ..gpu_sampler import GPU_SAMPLER_OUTPUT_ENV, SLOT_QUARANTINE_ENV, slot_quarantine_dir
from ..placement import PLACEMENT_REFUSAL_ENV, parse_cpu_set
from ..runners import CoreRunEvidence, EvidenceUnavailable, TPENCommand, TPENEvidence
from ..workers import SubprocessWorker
from .contract import (
    _admit_slot,
    _child_precore_refusal,
    _evidence_payload,
    _inside,
    _member_paths,
    _without_gpu_sampler,
)

LEAF_REQUEST_SCHEMA = "tpen-executor/leaf-request-v1"
LEAF_RESULT_SCHEMA = "tpen-executor/leaf-result-v1"

# Child placement identities written by the controller overlay
# (:func:`tpen_executor.dispatch.contract._with_child_placement`).  When a
# prepared environment carries one, it must name this request's attempt.
_PLACEMENT_IDENTITY_KEYS = {
    "TPEN_PLACEMENT_TASK_ID": "task_id",
    "TPEN_PLACEMENT_ATTEMPT_ID": "attempt_id",
    "TPEN_PLACEMENT_GROUP_DIGEST": "group_digest",
    "TPEN_PLACEMENT_ALLOCATION_ID": "allocation_id",
    "TPEN_PLACEMENT_BACKEND_JOB_ID": "backend_job_id",
    "TPEN_PLACEMENT_SLOT_ID": "slot_id",
    "TPEN_PLACEMENT_PAYLOAD_DIGEST": "payload_digest",
    "TPEN_PLACEMENT_OUTPUT_DIR": "output_dir",
}
# Identity fields echoed by every result and compared with the request.
_IDENTITY_FIELDS = (
    "group_digest", "task_id", "attempt_id", "allocation_id", "backend_job_id", "slot_id", "payload_digest",
)
_REQUEST_PLAIN_FIELDS = (
    *_IDENTITY_FIELDS, "config", "overrides", "root", "output_dir", "timeout",
)
_SCIENCE_KEYS = {"argv", "returncode", "timed_out", "pid", "proc_start_ticks", "stdout_path", "stderr_path"}
_ERROR_KEYS = {"type", "message"}
TEARDOWN_SCHEMA = "tpen-executor/leaf-teardown-v1"
_TEARDOWN_KEYS = {
    "schema", "group_digest", "task_id", "attempt_id", "allocation_id",
    "backend_job_id", "child_group_reaped", "inspection",
}
_TEARDOWN_INSPECTIONS = frozenset({"complete", "not-run"})
_CANCEL_MARKER = "cancel.json"
_PHASES = ((), ("science",), ("science", "inspection"))
INSPECTION_STATES = frozenset({"not-run", "evidence", "unavailable", "error"})
# Typed placement refusals a leaf reports, before science, for the CPUs of its
# own worker; the controller publishes them as placement refusals.
WORKER_AFFINITY_REFUSAL_CODES = frozenset(
    {"placement-cpu-outside-worker-affinity", "placement-worker-affinity-unavailable"}
)


class LeafRefused(RuntimeError):
    """Raised inside a leaf that refuses to launch science for a request."""


class LeafPlacementRefused(LeafRefused):
    """Raised inside a leaf whose placed slot CPUs are unavailable on its worker.

    Parameters
    ----------
    code : str
        One of :data:`WORKER_AFFINITY_REFUSAL_CODES`.
    reason : str
        Non-empty human-readable reason.

    Notes
    -----
    The message is ``"<code>: <reason>"``, so the plain launch-error fact
    of the result carries the typed code across the wire.
    """

    def __init__(self, code: str, reason: str) -> None:
        if code not in WORKER_AFFINITY_REFUSAL_CODES or type(reason) is not str or not reason:
            raise ValueError("worker placement refusal needs a known code and a reason")
        self.code = code
        self.reason = reason
        super().__init__(f"{code}: {reason}")


class LeafResultRejected(ValueError):
    """Raised by the controller when a leaf result does not answer its request."""


@frozen_record
class LeafTeardownReceipt:
    """Leaf-authored proof that its child group is quiescent.

    This is deliberately plain data.  A controller future, scheduler state,
    worker exit, or DFK cleanup can never manufacture this acknowledgement.
    """

    schema: ClassVar[str] = TEARDOWN_SCHEMA
    group_digest: str
    task_id: str
    attempt_id: str
    allocation_id: str
    backend_job_id: str
    child_group_reaped: bool
    inspection: str

    def __post_init__(self) -> None:
        for name in ("group_digest", "task_id", "attempt_id", "allocation_id", "backend_job_id"):
            _text(getattr(self, name), name)
        if type(self.child_group_reaped) is not bool or self.inspection not in _TEARDOWN_INSPECTIONS:
            raise ValueError("invalid leaf teardown facts")

    @classmethod
    def for_request(cls, request: "LeafRequest", *, inspection: str, child_group_reaped: bool) -> "LeafTeardownReceipt":
        return cls(request.group_digest, request.task_id, request.attempt_id,
                   request.allocation_id, request.backend_job_id, child_group_reaped, inspection)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "group_digest": self.group_digest,
                "task_id": self.task_id, "attempt_id": self.attempt_id,
                "allocation_id": self.allocation_id, "backend_job_id": self.backend_job_id,
                "child_group_reaped": self.child_group_reaped, "inspection": self.inspection}

    @classmethod
    def from_mapping(cls, data: Any) -> "LeafTeardownReceipt":
        fields(data, _TEARDOWN_KEYS, cls.schema, "leaf teardown receipt")
        return cls(*(data[name] for name in (
            "group_digest", "task_id", "attempt_id", "allocation_id", "backend_job_id",
            "child_group_reaped", "inspection")))


def _require_plain(value: Any, path: str) -> None:
    """Refuse every value that is not exact, immutable JSON-shaped builtins.

    Parameters
    ----------
    value : object
        Candidate wire value.  ``None``, exact ``str``/``int``/``float``/
        ``bool`` and exact tuples of those are plain.
    path : str
        Field name used in the error message.

    Raises
    ------
    TypeError
        If ``value`` (or a nested item) is anything else, including a
        subclass of a builtin, a lock, a future, a process handle or a
        controller callback.
    """

    if value is None or type(value) in (str, int, float, bool):
        return
    if type(value) is tuple:
        for item in value:
            _require_plain(item, f"{path}[]")
        return
    raise TypeError(f"{path} is not plain wire data: {type(value).__name__}")


def _require_json(value: Any, path: str) -> None:
    """Refuse every nested value of a fact that is not exact JSON-shaped data.

    Parameters
    ----------
    value : object
        Candidate fact.  Exact ``dict``/mapping-proxy containers with exact
        ``str`` keys, exact ``list``/``tuple`` containers, and plain leaves
        (:func:`_require_plain`) are accepted at every depth.
    path : str
        Field name used in the error message.

    Raises
    ------
    TypeError
        If any nested key, container or leaf is anything else, including a
        ``str`` subclass that :func:`~tpen_executor._codec._freeze` would
        otherwise retain together with whatever live handle it carries.
    """

    if type(value) in (dict, MappingProxyType):
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} key is not plain wire data: {type(key).__name__}")
            _require_json(item, f"{path}.{key}")
        return
    if type(value) in (list, tuple):
        for item in value:
            _require_json(item, f"{path}[]")
        return
    _require_plain(value, path)


def _path_component(value: str, name: str) -> str:
    """Validate one identity used as a single store path component."""

    # The store names attempts ``groups/<group>/<task>/<attempt>``; an
    # identity that is not one component could bind a different directory.
    if value in {".", ".."} or os.sep in value or (os.altsep is not None and os.altsep in value):
        raise ValueError(f"{name} is not a single store path component")
    return value


def _text(value: Any, name: str) -> str:
    """Validate non-empty request or result text."""

    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _absolute(value: Any, name: str) -> str:
    """Validate an absolute, already lexically normalized path."""

    _text(value, name)
    # The controller derives every fenced path lexically; a request whose
    # path would normalize differently is refused rather than rewritten.
    if not os.path.isabs(value) or os.path.abspath(os.path.normpath(value)) != value:
        raise ValueError(f"{name} must be an absolute normalized path")
    return value


def _error_fact(error: BaseException) -> dict[str, str]:
    """Return the plain type/message fact recorded for one leaf-side error."""

    return {"type": type(error).__name__, "message": str(error)}


def _cancel_path(request: "LeafRequest") -> str:
    return os.path.join(request.output_dir, _CANCEL_MARKER)


def _cancel_identity(request: "LeafRequest") -> dict[str, str]:
    return {"schema": "tpen-executor/leaf-cancel-v1",
            **{name: getattr(request, name) for name in
               ("group_digest", "task_id", "attempt_id", "allocation_id", "backend_job_id")}}


def _cancel_requested(request: "LeafRequest") -> bool:
    try:
        marker = decode_json(Path(_cancel_path(request)).read_bytes())
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return False
    return isinstance(marker, Mapping) and all(marker.get(key) == value for key, value in _cancel_identity(request).items())


@frozen_record
class LeafRequest:
    """Immutable, primitive-only request for one fenced leaf attempt.

    Parameters
    ----------
    group_digest, task_id, attempt_id, allocation_id, backend_job_id : str
        Controller-owned identities of the begun attempt.
    slot_id : str or None
        Slot named by the prepared environment's ``TPEN_SLOT_ID``.
    config : str
        Absolute configured-run Hydra configuration path.
    overrides : tuple of str
        Configured-run dot-list overrides.
    payload_digest : str
        Digest of ``config`` and ``overrides``; it must match them.
    root : str
        Lexical attempt-store root.  It binds the fenced output path and
        names the slot's durable quarantine directory; the leaf never opens
        the store through it.
    output_dir : str
        Fenced attempt output directory ``<root>/groups/<group_digest>/
        <task_id>/<attempt_id>``.
    environment : PreparedPythonEnvironment
        Prepared science environment after every controller overlay.
    timeout : int, float or None
        Positive science and inspection timeout in seconds.

    Raises
    ------
    TypeError
        If any value is not plain wire data or has the wrong type.
    ValueError
        If paths, digests or the timeout are invalid; if the output directory
        is not the store directory of this group, task and attempt; if the
        payload digest or a placement identity in the environment crosses
        this request; or if a carried slot quarantine directory is not the
        one derived from ``root`` (or a sampled launch lacks it).

    Notes
    -----
    Construction is the only gate: a request that exists is internally
    consistent, so :meth:`decode` re-runs every check on the wire form.
    """

    schema: ClassVar[str] = LEAF_REQUEST_SCHEMA
    group_digest: str
    task_id: str
    attempt_id: str
    allocation_id: str
    backend_job_id: str
    slot_id: str | None
    config: str
    overrides: tuple[str, ...]
    payload_digest: str
    root: str
    output_dir: str
    environment: PreparedPythonEnvironment
    timeout: int | float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.overrides, list):
            object.__setattr__(self, "overrides", tuple(self.overrides))
        # Primitive gate first: no live object rides in any field, even one
        # disguised as a builtin subclass that later content checks accept.
        for name in _REQUEST_PLAIN_FIELDS:
            _require_plain(getattr(self, name), name)
        if type(self.environment) is not PreparedPythonEnvironment:
            raise TypeError(f"environment is not plain wire data: {type(self.environment).__name__}")
        # The descriptor validates its own python and env as exact strings but
        # accepts any SourceScript subclass; only the exact setup is wire data.
        if self.environment.setup is not None and type(self.environment.setup) is not SourceScript:
            raise TypeError(f"environment.setup is not plain wire data: {type(self.environment.setup).__name__}")
        for name in ("group_digest", "task_id", "attempt_id", "allocation_id", "backend_job_id"):
            _text(getattr(self, name), name)
        if self.slot_id is not None:
            _text(self.slot_id, "slot_id")
        # Same rule as ConfiguredRun: absolute, but not necessarily normalized.
        if not os.path.isabs(_text(self.config, "config")):
            raise ValueError("config must be an absolute path")
        if not isinstance(self.overrides, tuple) or any(not isinstance(item, str) for item in self.overrides):
            raise TypeError("overrides must be a sequence of strings")
        _absolute(self.root, "root")
        _absolute(self.output_dir, "output_dir")
        try:
            # The dispatcher-owned path rule: output strictly below the root.
            _member_paths(self)
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        if self.timeout is not None and type(self.timeout) not in (int, float):
            raise TypeError("timeout must be an int, a float or None")
        if self.timeout is not None and (not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("timeout must be finite and positive or None")
        # Same digest as the controller placement overlay and receipts.
        if self.payload_digest != digest({"config": self.config, "overrides": list(self.overrides)}):
            raise ValueError("payload_digest does not match the configured payload")
        values = self.environment.env
        if values.get("TPEN_SLOT_ID") != self.slot_id:
            raise ValueError("slot_id does not match the prepared environment TPEN_SLOT_ID")
        if PLACEMENT_REFUSAL_ENV in values:
            # A refused placement is completed by the controller; it never
            # reaches a leaf.
            raise ValueError("a placement-refused environment cannot be sent to a leaf")
        crossed = sorted(
            key for key, name in _PLACEMENT_IDENTITY_KEYS.items()
            if key in values and values[key] != getattr(self, name)
        )
        if crossed:
            raise ValueError(f"prepared environment placement identity crosses the request: {','.join(crossed)}")
        # The slot quarantine directory is derived from this request's store
        # root and the placement slot identity, exactly as the controller
        # overlay derives it; a carried directory may only repeat it.  A
        # sampled launch always carries it, so its durable marker lands where
        # admission looks.
        quarantine_dir = slot_quarantine_dir(self.root, values)
        carried = values.get(SLOT_QUARANTINE_ENV)
        if carried is not None and carried != quarantine_dir:
            raise ValueError("prepared environment slot quarantine directory is not the store-derived one")
        if carried is None and quarantine_dir is not None and GPU_SAMPLER_OUTPUT_ENV in values:
            raise ValueError("a sampled launch must carry its store-derived slot quarantine directory")
        # The fenced output is the store's directory of exactly this group,
        # task and attempt, independently of any optional placement identity.
        expected = os.path.join(
            self.root,
            "groups",
            _path_component(self.group_digest, "group_digest"),
            _path_component(self.task_id, "task_id"),
            _path_component(self.attempt_id, "attempt_id"),
        )
        if self.output_dir != expected:
            raise ValueError("output_dir does not bind this group, task and attempt below the store root")

    @classmethod
    def for_attempt(
        cls,
        fence: Any,
        allocation: Any,
        configured: Any,
        environment: PreparedPythonEnvironment,
        timeout: int | float | None = None,
    ) -> "LeafRequest":
        """Build the request for one begun attempt on the controller.

        Parameters
        ----------
        fence : AttemptFence
            Store-issued fence of the begun attempt.
        allocation : ConfiguredAllocation
            Allocation that owns the fence.
        configured : ConfiguredRun
            Configured run of the fenced task.
        environment : PreparedPythonEnvironment
            Science environment after the controller binding, context-key and
            placement overlays.
        timeout : int, float or None, optional
            Positive science and inspection timeout in seconds.

        Returns
        -------
        LeafRequest
            Validated request; the fence and allocation objects themselves are
            not retained.

        Raises
        ------
        ValueError
            If the fence and allocation identities disagree, the configured
            run belongs to another task, or any request invariant fails.
        """

        if fence.allocation_id != allocation.allocation_id.value:
            raise ValueError("fence allocation does not match the allocation")
        if configured.id != fence.task_id:
            raise ValueError("configured run does not belong to the fenced task")
        return cls(
            fence.group_digest,
            fence.task_id,
            fence.attempt_id.value,
            allocation.allocation_id.value,
            allocation.backend_job_id.value,
            environment.env.get("TPEN_SLOT_ID"),
            configured.config,
            tuple(configured.overrides),
            digest({"config": configured.config, "overrides": list(configured.overrides)}),
            fence.root,
            fence.output_dir,
            environment,
            timeout,
        )

    def paths(self) -> tuple[str, str, str, str, str, str]:
        """Return output, core, science log and inspection log paths.

        Returns
        -------
        tuple of str
            The same dispatcher-owned paths a local launch of this attempt
            uses (:func:`tpen_executor.dispatch.contract._member_paths`).
        """

        return _member_paths(self)

    def science_args(self) -> tuple[str, ...]:
        """Return the exact science arguments after the prepared interpreter.

        Returns
        -------
        tuple of str
            ``-c RUN_SHIM --config <config> <overrides...>`` followed by the
            flat-layout run-root overrides, exactly as for a local launch.
        """

        output_dir = self.paths()[0]
        return (*TPENCommand(self.config, self.overrides).args, f"run.root={output_dir}", "run.layout=flat", "run.run_id=core")

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation."""

        return {
            "schema": self.schema,
            "group_digest": self.group_digest,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "allocation_id": self.allocation_id,
            "backend_job_id": self.backend_job_id,
            "slot_id": self.slot_id,
            "config": self.config,
            "overrides": list(self.overrides),
            "payload_digest": self.payload_digest,
            "root": self.root,
            "output_dir": self.output_dir,
            "environment": self.environment.to_dict(),
            "timeout": self.timeout,
        }

    def encode(self) -> bytes:
        """Encode this request as canonical JSON bytes."""

        return canonical_bytes(self.to_dict())

    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical wire form."""

        return digest(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "LeafRequest":
        """Decode one strict request mapping.

        Raises
        ------
        TypeError, ValueError
            If keys, schema, types, or any request invariant are invalid.
        """

        keys = {"schema", *(name for name in _REQUEST_PLAIN_FIELDS), "environment"}
        fields(data, keys, cls.schema, "LeafRequest")
        overrides = data["overrides"]
        if type(overrides) is not list:
            raise TypeError("overrides must be a JSON array")
        return cls(
            data["group_digest"],
            data["task_id"],
            data["attempt_id"],
            data["allocation_id"],
            data["backend_job_id"],
            data["slot_id"],
            data["config"],
            tuple(overrides),
            data["payload_digest"],
            data["root"],
            data["output_dir"],
            PreparedPythonEnvironment.from_mapping(data["environment"]),
            data["timeout"],
        )

    @classmethod
    def decode(cls, data: bytes) -> "LeafRequest":
        """Decode canonical request bytes.

        Parameters
        ----------
        data : bytes
            Canonical JSON bytes produced by :meth:`encode`.  An in-process
            object or mapping is refused: only bytes are wire data.

        Raises
        ------
        TypeError, ValueError
            If ``data`` is not bytes, not one JSON object, not a valid
            request, or not byte-identical to the request's canonical
            encoding.
        """

        if type(data) is not bytes:
            raise TypeError("leaf request must be canonical JSON bytes")
        request = cls.from_mapping(decode_json(data.decode("utf-8")))
        # Only the canonical form is wire data: whitespace, key order or
        # number spelling that re-encodes differently is refused.
        if request.encode() != data:
            raise ValueError("leaf request bytes are not canonical")
        return request


def _optional_fact(value: Any, keys: set[str], name: str) -> Mapping[str, Any] | None:
    """Freeze an optional fact mapping with an exact key set."""

    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} has invalid keys")
    frozen = _freeze(value, name)
    if any(type(item) is not str for item in frozen.values()):
        raise TypeError(f"{name} values must be strings")
    return frozen


@frozen_record
class LeafResult:
    """Immutable, primitive-only facts returned by one leaf execution.

    Parameters
    ----------
    request_digest : str
        :meth:`LeafRequest.digest` of the request this result answers.
    group_digest, task_id, attempt_id, allocation_id, backend_job_id, slot_id, payload_digest
        Identities echoed from the request.
    leaf_host, leaf_pid, leaf_python
        Host name, process ID and interpreter of the leaf process.
    phases : tuple of str
        Leaf-side phases that ran, in order: ``()``, ``("science",)`` or
        ``("science", "inspection")``.
    science : mapping or None
        Reaped science :class:`~tpen_executor.workers.ProcessOutcome` facts,
        or ``None`` when no outcome was returned.
    launch_error : mapping or None
        ``{"type", "message"}`` of an ordinary error that prevented a science
        outcome (including a leaf refusal or a quarantined slot).
    child_refusal : mapping or None
        Typed ``RUN_SHIM`` pre-core placement refusal ``{"code", "reason"}``.
    inspection : str
        One of ``not-run``, ``evidence``, ``unavailable`` or ``error``.
    evidence : mapping or None
        Public core evidence decoded in the leaf, present only for
        ``inspection == "evidence"``.
    inspection_error : mapping or None
        ``{"type", "message"}`` for ``unavailable`` or ``error`` inspection.
    teardown : mapping
        Versioned, identity-bound :class:`LeafTeardownReceipt` facts authored
        by the leaf after child-group reaping and inspection completion (or
        explicit non-execution).

    Raises
    ------
    TypeError, ValueError
        If any fact is not plain wire data or the facts are inconsistent.
    """

    schema: ClassVar[str] = LEAF_RESULT_SCHEMA
    request_digest: str
    group_digest: str
    task_id: str
    attempt_id: str
    allocation_id: str
    backend_job_id: str
    slot_id: str | None
    payload_digest: str
    leaf_host: str
    leaf_pid: int
    leaf_python: str
    phases: tuple[str, ...]
    science: Mapping[str, Any] | None
    launch_error: Mapping[str, str] | None
    child_refusal: Mapping[str, str] | None
    inspection: str
    evidence: Mapping[str, Any] | None
    inspection_error: Mapping[str, str] | None
    teardown: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.phases, list):
            object.__setattr__(self, "phases", tuple(self.phases))
        for name in (*_IDENTITY_FIELDS, "request_digest", "leaf_host", "leaf_pid", "leaf_python", "phases", "inspection"):
            _require_plain(getattr(self, name), name)
        for name in ("request_digest", "group_digest", "task_id", "attempt_id", "allocation_id", "backend_job_id", "payload_digest", "leaf_host", "leaf_python"):
            _text(getattr(self, name), name)
        if self.slot_id is not None:
            _text(self.slot_id, "slot_id")
        if type(self.leaf_pid) is not int or self.leaf_pid <= 0:
            raise TypeError("leaf_pid must be a positive integer")
        if self.phases not in _PHASES:
            raise ValueError("phases must be (), ('science',) or ('science', 'inspection')")
        if self.inspection not in INSPECTION_STATES:
            raise ValueError("inspection state is unknown")
        object.__setattr__(self, "science", self._science(self.science))
        object.__setattr__(self, "launch_error", _optional_fact(self.launch_error, _ERROR_KEYS, "launch_error"))
        object.__setattr__(self, "child_refusal", _optional_fact(self.child_refusal, {"code", "reason"}, "child_refusal"))
        object.__setattr__(self, "inspection_error", _optional_fact(self.inspection_error, _ERROR_KEYS, "inspection_error"))
        if self.evidence is not None:
            if not isinstance(self.evidence, Mapping):
                raise TypeError("evidence must be a mapping or None")
            try:
                CoreRunEvidence.from_mapping(thaw(self.evidence))
            except EvidenceUnavailable as error:
                raise ValueError(f"evidence is not valid public core evidence: {error}") from error
            object.__setattr__(self, "evidence", _freeze(self.evidence, "evidence"))
        if not isinstance(self.teardown, Mapping):
            raise ValueError("teardown must be a leaf teardown receipt")
        LeafTeardownReceipt.from_mapping(thaw(self.teardown))
        object.__setattr__(self, "teardown", _freeze(self.teardown, "teardown"))
        # Freezing keeps builtin subclasses at any depth; every retained key
        # and leaf must be exact wire data.
        for name in ("science", "launch_error", "child_refusal", "evidence", "inspection_error", "teardown"):
            _require_json(getattr(self, name), name)
        # Cross-fact consistency: every fact has exactly one leaf-side cause.
        inspected = "inspection" in self.phases
        if (self.science is None) != (self.launch_error is not None):
            raise ValueError("exactly one of science and launch_error must be present")
        if (self.science is not None) != ("science" in self.phases):
            raise ValueError("a science outcome is present exactly for the science phase")
        if inspected == (self.inspection == "not-run"):
            raise ValueError("inspection state disagrees with the inspection phase")
        if inspected and (self.science is None or self.child_refusal is not None):
            raise ValueError("inspection may follow only a science outcome without a pre-core refusal")
        if (self.evidence is not None) != (self.inspection == "evidence"):
            raise ValueError("evidence is present exactly for the evidence inspection state")
        if (self.inspection_error is not None) != (self.inspection in {"unavailable", "error"}):
            raise ValueError("inspection_error is present exactly for failed inspection")

    @staticmethod
    def _science(value: Any) -> Mapping[str, Any] | None:
        """Validate and freeze the reaped science outcome facts."""

        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != _SCIENCE_KEYS:
            raise ValueError("science has invalid keys")
        argv = value["argv"]
        if isinstance(argv, (str, bytes, bytearray)) or not isinstance(argv, Sequence) or not argv or any(type(item) is not str for item in argv):
            raise TypeError("science argv must be a non-empty sequence of strings")
        for key in ("returncode", "pid", "proc_start_ticks"):
            if value[key] is not None and type(value[key]) is not int:
                raise TypeError(f"science {key} must be an integer or None")
        if type(value["timed_out"]) is not bool:
            raise TypeError("science timed_out must be a boolean")
        for key in ("stdout_path", "stderr_path"):
            _text(value[key], f"science {key}")
        return _freeze(value, "science")

    def to_dict(self) -> dict[str, Any]:
        """Return the strict wire representation."""

        return {
            "schema": self.schema,
            "request_digest": self.request_digest,
            **{name: getattr(self, name) for name in _IDENTITY_FIELDS},
            "leaf_host": self.leaf_host,
            "leaf_pid": self.leaf_pid,
            "leaf_python": self.leaf_python,
            "phases": list(self.phases),
            "science": thaw(self.science),
            "launch_error": thaw(self.launch_error),
            "child_refusal": thaw(self.child_refusal),
            "inspection": self.inspection,
            "evidence": thaw(self.evidence),
            "inspection_error": thaw(self.inspection_error),
            "teardown": thaw(self.teardown),
        }

    def encode(self) -> bytes:
        """Encode this result as canonical JSON bytes."""

        return canonical_bytes(self.to_dict())

    @classmethod
    def from_mapping(cls, data: Any) -> "LeafResult":
        """Decode one strict result mapping.

        Raises
        ------
        TypeError, ValueError
            If keys, schema, types, or any fact invariant are invalid.
        """

        keys = {
            "schema", "request_digest", *_IDENTITY_FIELDS, "leaf_host", "leaf_pid", "leaf_python", "phases",
            "science", "launch_error", "child_refusal", "inspection", "evidence", "inspection_error", "teardown",
        }
        fields(data, keys, cls.schema, "LeafResult")
        if type(data["phases"]) is not list:
            raise TypeError("phases must be a JSON array")
        return cls(
            data["request_digest"],
            *(data[name] for name in _IDENTITY_FIELDS),
            data["leaf_host"],
            data["leaf_pid"],
            data["leaf_python"],
            tuple(data["phases"]),
            data["science"],
            data["launch_error"],
            data["child_refusal"],
            data["inspection"],
            data["evidence"],
            data["inspection_error"],
            data["teardown"],
        )

    @classmethod
    def decode(cls, data: bytes) -> "LeafResult":
        """Decode canonical result bytes; an in-process object is refused.

        Raises
        ------
        TypeError, ValueError
            If ``data`` is not bytes, not a valid result, or not
            byte-identical to the result's canonical encoding.
        """

        if type(data) is not bytes:
            raise TypeError("leaf result must be canonical JSON bytes")
        result = cls.from_mapping(decode_json(data.decode("utf-8")))
        if result.encode() != data:
            raise ValueError("leaf result bytes are not canonical")
        return result

    def core_evidence(self) -> CoreRunEvidence | None:
        """Return the leaf-decoded public evidence as a controller record."""

        return None if self.evidence is None else CoreRunEvidence.from_mapping(thaw(self.evidence))

    def result_digest(self) -> str | None:
        """Return the outcome ``result_digest`` a local completion would record."""

        evidence = self.core_evidence()
        return None if evidence is None else digest(_evidence_payload(evidence))

    def status(self) -> str:
        """Return the closed operational status for the controller to publish.

        Returns
        -------
        str
            ``launch_failed`` for a launch error, a pre-core refusal or an
            inspection error; otherwise the reaped science outcome
            (``timed_out``, ``completed`` or ``failed``).  Unavailable evidence
            does not change the status, exactly as for a local completion.
        """

        if self.science is None or self.child_refusal is not None or self.inspection == "error":
            return "launch_failed"
        if self.science["timed_out"]:
            return "timed_out"
        return "completed" if self.science["returncode"] == 0 else "failed"


def validate_leaf_result(request: LeafRequest, result: LeafResult) -> LeafResult:
    """Check on the controller that ``result`` answers exactly ``request``.

    Parameters
    ----------
    request : LeafRequest
        Request the controller sent for one begun attempt.
    result : LeafResult
        Decoded leaf result.

    Returns
    -------
    LeafResult
        ``result`` unchanged when it answers ``request``.

    Raises
    ------
    TypeError
        If either argument has the wrong type.
    LeafResultRejected
        If the request digest, any echoed identity, the science argv or log
        paths, or the evidence run or checkpoint directory cross the
        request; if a science outcome without a pre-core refusal lacks the
        inspection phase; or if the leaf did not confirm its reaped children.
        Every echoed field is compared; the request digest alone is not
        trusted.

    Notes
    -----
    Evidence directories are compared after resolving symbolic links on the
    controller, as the store does when it accepts a segment, because public
    core inspection reports its resolved run directory.
    """

    if type(request) is not LeafRequest or type(result) is not LeafResult:
        raise TypeError("validate_leaf_result requires a LeafRequest and a LeafResult")
    checks = {
        "request": result.request_digest == request.digest(),
        "group": result.group_digest == request.group_digest,
        "task": result.task_id == request.task_id,
        "attempt": result.attempt_id == request.attempt_id,
        "allocation": result.allocation_id == request.allocation_id,
        "backend": result.backend_job_id == request.backend_job_id,
        "slot": result.slot_id == request.slot_id,
        "payload": result.payload_digest == request.payload_digest,
    }
    _output, core_dir, stdout_path, stderr_path, _evidence_stdout, _evidence_stderr = request.paths()
    if result.science is not None:
        checks["argv"] = tuple(result.science["argv"]) == (request.environment.python, *request.science_args())
        checks["logs"] = (result.science["stdout_path"], result.science["stderr_path"]) == (stdout_path, stderr_path)
    # Every science outcome without a pre-core refusal is inspected in the
    # leaf, as in a local completion; a result that skipped inspection is
    # never published as that outcome.
    checks["inspection"] = ("inspection" in result.phases) == (result.science is not None and result.child_refusal is None)
    if result.evidence is not None:
        # Evidence must describe this attempt's core run, not a foreign one.
        real_core = os.path.realpath(core_dir)
        checkpoint = result.evidence["checkpoint"]
        checks["evidence"] = os.path.realpath(result.evidence["run_dir"]) == real_core and (
            checkpoint is None or _inside(real_core, os.path.realpath(checkpoint["checkpoint_dir"]))
        )
    teardown = LeafTeardownReceipt.from_mapping(thaw(result.teardown))
    checks["teardown"] = (
        teardown.group_digest == request.group_digest
        and teardown.task_id == request.task_id
        and teardown.attempt_id == request.attempt_id
        and teardown.allocation_id == request.allocation_id
        and teardown.backend_job_id == request.backend_job_id
        and teardown.child_group_reaped
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise LeafResultRejected(f"leaf result crosses its request: {','.join(failed)}")
    return result


def _require_fresh_output(request: LeafRequest) -> None:
    """Refuse a leaf whose fenced output is absent or already used.

    The controller creates the fenced directory at ``begin`` and nothing else
    writes the science or inspection paths before a launch.  A missing
    directory means the attempt store is not visible to this worker; an
    existing core directory or log means the request is a replay.

    Raises
    ------
    LeafRefused
        If the attempt output is not visible or not fresh.
    """

    output_dir, core_dir, *logs = request.paths()
    if not os.path.isdir(output_dir) or os.path.islink(output_dir):
        raise LeafRefused("fenced attempt output directory is not visible to this leaf")
    used = [path for path in (core_dir, *logs) if os.path.lexists(path)]
    if used:
        raise LeafRefused("fenced attempt output is not fresh: " + ",".join(os.path.basename(path) for path in used))


def _require_worker_cpus(environment: PreparedPythonEnvironment) -> None:
    """Refuse a placed slot whose declared CPUs this worker may not use.

    The controller's own CPU affinity says nothing about a remote worker, so
    the execution-local check runs here, on the worker, before science.  It
    is the same subset rule local dispatch applies to the controller, and is
    independent of the controller placement identity and overlay.  An
    unplaced launch (no ``TPEN_PLACEMENT_DECLARED_CPUS``) sets no affinity in
    :data:`~tpen_executor.runners.RUN_SHIM` and is not checked.

    Raises
    ------
    LeafPlacementRefused
        If the worker affinity is unavailable or lacks a declared CPU.
    PlacementRefusal
        If the declared CPU set is malformed.
    """

    raw = environment.env.get("TPEN_PLACEMENT_DECLARED_CPUS")
    if raw is None:
        return
    declared = set(parse_cpu_set(raw))
    try:
        permitted = set(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        raise LeafPlacementRefused(
            "placement-worker-affinity-unavailable", f"worker CPU affinity is unavailable: {error}",
        ) from error
    if not declared <= permitted:
        raise LeafPlacementRefused(
            "placement-cpu-outside-worker-affinity", "declared slot CPUs are not all permitted to this worker",
        )


class _ReapObserver:
    """Aggregate the worker-observed group-reaped fact across one leaf's launches.

    The teardown receipt's ``child_group_reaped`` must reflect what the
    worker actually confirmed, for every launch it ran (science and, when it
    ran, inspection) -- not a heuristic derived from the science outcome's
    pid or return code alone.
    """

    def __init__(self) -> None:
        self.confirmed = True

    def record(self, outcome: Any) -> None:
        if not outcome.group_reaped:
            self.confirmed = False


class _ReapObservingWorker:
    """Wrap a :class:`SubprocessWorker` to feed every launch to a :class:`_ReapObserver`."""

    def __init__(self, inner: SubprocessWorker, observer: _ReapObserver) -> None:
        self._inner = inner
        self._observer = observer

    def run(self, *args: Any, **kwargs: Any) -> Any:
        outcome = self._inner.run(*args, **kwargs)
        self._observer.record(outcome)
        return outcome


def run_leaf(payload: bytes) -> bytes:
    """Run one fenced member on this worker and return plain result bytes.

    Parameters
    ----------
    payload : bytes
        Canonical :class:`LeafRequest` bytes.  Nothing else is accepted: the
        leaf receives no handle, lock, future, store, lifecycle or callback.

    Returns
    -------
    bytes
        Canonical :class:`LeafResult` bytes.

    Raises
    ------
    TypeError, ValueError
        If ``payload`` is not a valid request; no child is launched.
    BaseException
        A cancellation propagates after the worker has reaped its child; no
        result is returned (remote acknowledgement is Layer C).

    Notes
    -----
    Before science, the leaf refuses a used or invisible fenced output,
    applies local dispatch slot admission
    (:func:`tpen_executor.dispatch.contract._admit_slot`) against the store
    root named by the request, and refuses a placed slot whose declared CPUs
    are not all permitted to this worker (:class:`LeafPlacementRefused`); a
    refusal is a launch error and no child is spawned.  Science runs exactly once through :class:`SubprocessWorker`
    with the request's exact ``RUN_SHIM`` arguments.  Public inspection then
    runs here, on the same worker and through the trusted
    :meth:`~tpen_executor.runners.TPENEvidence.read` call site, under the same
    conditions as a local completion: after any reaped science outcome that
    carries no pre-core placement refusal.  Ordinary errors become facts; the
    leaf never begins, finishes, accepts or publishes anything.
    """

    request = LeafRequest.decode(payload)
    environment = request.environment
    output_dir, core_dir, stdout_path, stderr_path, evidence_stdout, evidence_stderr = request.paths()
    reap_observer = _ReapObserver()
    worker = _ReapObservingWorker(SubprocessWorker(), reap_observer)
    phases: list[str] = []
    science: dict[str, Any] | None = None
    launch_error: dict[str, str] | None = None
    child_refusal: dict[str, str] | None = None
    inspection = "not-run"
    evidence: dict[str, Any] | None = None
    inspection_error: dict[str, str] | None = None
    try:
        _require_fresh_output(request)
        # Local dispatch admission, run in the leaf before any spawn: a placed
        # slot is checked against the durable quarantine markers derived from
        # the request's store root and slot identity, sampled or not and
        # whatever quarantine overlay the environment carries.  A fresh leaf
        # has no in-process controller fence, so only the markers hold it.
        _admit_slot(environment, request)
        # The slot's CPUs must be usable on THIS worker, whatever the
        # controller's affinity is; a refusal spawns no child.
        _require_worker_cpus(environment)
        outcome = worker.run(
            environment,
            request.science_args(),
            cwd=output_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=request.timeout,
            cancel_path=_cancel_path(request),
            cancel_identity=_cancel_identity(request),
        )
    except BaseException as error:
        # The cancellation marker is intentionally handled inside the leaf's
        # worker, which reaps its group before returning.  Catching the
        # cancellation here lets that leaf author the required teardown ack;
        # it is not a controller-side conversion of an arbitrary interrupt.
        launch_error = _error_fact(error)
    else:
        phases.append("science")
        science = {
            "argv": list(outcome.argv),
            "returncode": outcome.returncode,
            "timed_out": outcome.timed_out,
            "pid": outcome.pid,
            "proc_start_ticks": outcome.proc_start_ticks,
            "stdout_path": outcome.stdout_path,
            "stderr_path": outcome.stderr_path,
        }
        # The RUN_SHIM sidecar is the typed source of a pre-core refusal; it
        # suppresses inspection exactly as in the local completion path.
        refusal = _child_precore_refusal(environment, outcome)
        if refusal is not None:
            child_refusal = refusal.to_dict()
    if science is not None and child_refusal is None and not _cancel_requested(request):
        phases.append("inspection")
        try:
            # The reader launches with LaunchPurpose.INSPECTION; the
            # environment only drops the worker-only sampling keys.
            decoded = TPENEvidence().read(
                worker,
                _without_gpu_sampler(environment),
                core_dir,
                cwd=output_dir,
                stdout_path=evidence_stdout,
                stderr_path=evidence_stderr,
                timeout=request.timeout,
                cancel_path=_cancel_path(request),
                cancel_identity=_cancel_identity(request),
            )
        except EvidenceUnavailable as error:
            inspection, inspection_error = "unavailable", _error_fact(error)
        except BaseException as error:
            inspection, inspection_error = "error", _error_fact(error)
        else:
            inspection, evidence = "evidence", _evidence_payload(decoded)
    result = LeafResult(
        request.digest(),
        *(getattr(request, name) for name in _IDENTITY_FIELDS),
        socket.gethostname(),
        os.getpid(),
        sys.executable,
        tuple(phases),
        science,
        launch_error,
        child_refusal,
        inspection,
        evidence,
        inspection_error,
        LeafTeardownReceipt.for_request(
            request, inspection="not-run" if inspection == "not-run" else "complete",
            # An observed fact from the worker's own reap of every launch it
            # ran (science and, when it ran, inspection) -- not derived from
            # the science outcome's pid or return code alone.
            child_group_reaped=science is None or reap_observer.confirmed,
        ).to_dict(),
    )
    return result.encode()


__all__ = [
    "INSPECTION_STATES",
    "LEAF_REQUEST_SCHEMA",
    "LEAF_RESULT_SCHEMA",
    "TEARDOWN_SCHEMA",
    "LeafPlacementRefused",
    "LeafRefused",
    "LeafRequest",
    "LeafResult",
    "LeafResultRejected",
    "LeafTeardownReceipt",
    "run_leaf",
    "validate_leaf_result",
    "WORKER_AFFINITY_REFUSAL_CODES",
]
