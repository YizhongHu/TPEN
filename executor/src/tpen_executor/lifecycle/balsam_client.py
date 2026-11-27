"""Bounded executor-side client for the isolated Balsam control process.

This module deliberately contains no Balsam SDK import.  The SDK lives in the
control-side bridge; the executor sends a fixed module invocation and parses
only :mod:`tpen_executor.lifecycle.balsam_wire` records.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Protocol

from .._codec import canonical_bytes, digest
from ..contracts import BackendJobId, BatchJobId, GroupId
from ..storage import ensure_durable_directory, publish_json_exclusive
from .balsam import BalsamGroupParked, BalsamGroupRefusal, BalsamGroupRegistrationIntent
from .balsam_wire import BalsamWireRequest, BalsamWireResult, decode_result, encode_request


class BalsamBridgeFailure(RuntimeError):
    """The bridge could not produce a trustworthy response."""


class BalsamBridgeAmbiguous(BalsamBridgeFailure):
    """A bridge failure may have happened after an external effect."""


class BalsamBridge(Protocol):
    """Fixed request/response transport used by the process client."""

    def call(self, payload: bytes, *, timeout: float) -> bytes:
        """Send one canonical request and return canonical response bytes."""
        ...


def select_balsam_client_path(value: str | os.PathLike[str] | None = None, *, environ: Mapping[str, str] | None = None) -> Path:
    """Select and validate the Balsam client YAML file before any SDK import.

    Parameters
    ----------
    value : path-like or None
        Explicit client YAML path.  When omitted, ``BALSAM_CLIENT_PATH`` is
        required; there is intentionally no home-directory fallback.
    environ : mapping, optional
        Environment source used by tests and callers with a controlled
        environment.

    Raises
    ------
    BalsamGroupRefusal
        If the path is absent, not a regular file, escapes through a symlink,
        or is writable by another user.
    """
    source = os.environ if environ is None else environ
    raw = value if value is not None else source.get("BALSAM_CLIENT_PATH")
    if raw is None or not str(raw).strip():
        raise BalsamGroupRefusal("BALSAM_CLIENT_PATH must name the client YAML file")
    path = Path(raw)
    if not path.is_absolute():
        raise BalsamGroupRefusal("BALSAM_CLIENT_PATH must be absolute")
    try:
        info = path.lstat()
        parent_info = path.parent.stat()
    except OSError as error:
        raise BalsamGroupRefusal("BALSAM_CLIENT_PATH is unavailable") from error
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise BalsamGroupRefusal("BALSAM_CLIENT_PATH must be a regular, non-symlink file")
    if info.st_mode & 0o022 or parent_info.st_mode & 0o022:
        raise BalsamGroupRefusal("BALSAM_CLIENT_PATH or its parent is writable by another user")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise BalsamGroupRefusal("BALSAM_CLIENT_PATH is not owned by the current user")
    return path


def validate_balsam_client_path(value: str | os.PathLike[str] | None = None, *, environ: Mapping[str, str] | None = None) -> Path:
    """Compatibility alias emphasizing the pre-import validation boundary."""
    return select_balsam_client_path(value, environ=environ)


class SubprocessBalsamBridge:
    """Invoke one fixed control bridge module with bounded arguments."""

    _MODULE = "tpen_balsam_control.bridge"

    def __init__(self, control_python: str | os.PathLike[str], client_path: Path) -> None:
        executable = Path(control_python)
        try:
            info = executable.stat()
        except OSError as error:
            raise BalsamGroupRefusal("control Python is unavailable") from error
        if not stat.S_ISREG(info.st_mode) or not os.access(executable, os.X_OK):
            raise BalsamGroupRefusal("control Python must be an executable file")
        self._argv = (str(executable), "-m", self._MODULE)
        self._client_path = client_path

    def call(self, payload: bytes, *, timeout: float) -> bytes:
        """Run the fixed bridge; stdout is the only accepted response channel."""
        if type(timeout) not in (int, float) or timeout <= 0:
            raise ValueError("bridge timeout must be positive")
        env = dict(os.environ)
        env["BALSAM_CLIENT_PATH"] = str(self._client_path)
        try:
            completed = subprocess.run(self._argv, input=payload, capture_output=True, check=False,
                                       timeout=timeout, env=env)
        except subprocess.TimeoutExpired as error:
            raise BalsamBridgeAmbiguous("control bridge timed out") from error
        except OSError as error:
            raise BalsamBridgeFailure("control bridge could not be launched") from error
        if completed.returncode != 0:
            raise BalsamBridgeAmbiguous("control bridge exited without a response")
        if not completed.stdout:
            raise BalsamBridgeAmbiguous("control bridge returned no response")
        return completed.stdout


class BalsamProcessClient:
    """Fixed, bounded client implementing the executor's Balsam port.

    Mutation intents are written before the bridge call.  A missing or
    malformed acknowledgement is durably parked, so calling the same method
    again cannot issue a second external mutation.
    """

    def __init__(self, control_python: str | os.PathLike[str], *, client_path: str | os.PathLike[str] | None = None,
                 state_dir: str | os.PathLike[str] | None = None, timeout: float = 30.0,
                 bridge: BalsamBridge | None = None) -> None:
        self.client_path = select_balsam_client_path(client_path)
        if type(timeout) not in (int, float) or timeout <= 0:
            raise BalsamGroupRefusal("client timeout must be positive")
        self.timeout = float(timeout)
        self._state_dir = None if state_dir is None else Path(state_dir)
        self._bridge = SubprocessBalsamBridge(control_python, self.client_path) if bridge is None else bridge

    def _call(self, request: BalsamWireRequest) -> BalsamWireResult:
        """Call the bridge and strictly bind its result to the request."""
        try:
            result = decode_result(self._bridge.call(encode_request(request), timeout=self.timeout))
        except BalsamBridgeFailure:
            raise
        except (TypeError, ValueError, OSError) as error:
            raise BalsamBridgeFailure("control bridge returned malformed data") from error
        if result.operation != request.operation or result.group_id != request.group_id:
            raise BalsamBridgeFailure("control bridge response crosses request identity")
        if request.backend_job_id is not None and result.backend_job_id != request.backend_job_id:
            raise BalsamBridgeFailure("control bridge response crosses backend Job identity")
        return result

    @staticmethod
    def _require_ack(request: BalsamWireRequest, result: BalsamWireResult) -> BalsamWireResult:
        if result.status == "refused":
            raise BalsamGroupRefusal(result.reason or "Balsam operation refused")
        if result.status == "ambiguous":
            raise BalsamGroupParked(result.reason or "Balsam operation has unknown effect")
        return result

    def register_group(self, intent: BalsamGroupRegistrationIntent) -> BackendJobId:
        """Register one immutable group and return only a typed Job identity."""
        if not isinstance(intent, BalsamGroupRegistrationIntent):
            raise TypeError("intent must be BalsamGroupRegistrationIntent")
        request = BalsamWireRequest.register(intent.group_id, intent.group_digest, intent.backend_options_digest)
        result = self._require_ack(request, self._call(request))
        if not isinstance(result.backend_job_id, BackendJobId) or result.batch_job_id is not None:
            raise BalsamGroupParked("registration acknowledgement has no typed Job identity")
        return result.backend_job_id

    def observe(self, group_id: GroupId, backend_job_id: BackendJobId) -> BalsamWireResult:
        """Observe one typed Job without interpreting its state as completion."""
        request = BalsamWireRequest.observe(group_id, backend_job_id)
        return self._require_ack(request, self._call(request))

    def observe_job(self, job_id: BackendJobId) -> BalsamWireResult:
        """Reject unbound observation handles rather than guessing a group."""
        raise BalsamGroupRefusal("observe_job requires the bound group identity")

    def submit_batch(self, group_id: GroupId, backend_job_id: BackendJobId, batch_digest: str) -> BatchJobId:
        """Submit one explicitly authorized batch, with no automatic retry."""
        request = BalsamWireRequest.submit(group_id, backend_job_id, batch_digest)
        result = self._mutate(request)
        if not isinstance(result.batch_job_id, BatchJobId):
            raise BalsamGroupParked("submission acknowledgement has no typed BatchJob identity")
        return result.batch_job_id

    def rearm_group(self, group_id: GroupId, backend_job_id: BackendJobId, recovery_digest: str) -> BalsamWireResult:
        """Rearm the same Job exactly once after explicit recovery admission."""
        return self._mutate(BalsamWireRequest.rearm(group_id, backend_job_id, recovery_digest))

    def readback(self, group_id: GroupId, backend_job_id: BackendJobId) -> BalsamWireResult:
        """Read back the bound deployment/job facts without mutation."""
        request = BalsamWireRequest.readback(group_id, backend_job_id)
        return self._require_ack(request, self._call(request))

    def _mutate(self, request: BalsamWireRequest) -> BalsamWireResult:
        """Journal one mutation before calling the bridge and park uncertainty."""
        if self._state_dir is None:
            raise BalsamGroupRefusal("mutations require a durable journal directory")
        ensure_durable_directory(str(self._state_dir))
        prefix = self._state_dir / "balsam-wire" / digest(request.to_dict())
        ensure_durable_directory(str(prefix.parent))
        ensure_durable_directory(str(prefix))
        intent_path = prefix / "intent.json"
        result_path = prefix / "result.json"
        if result_path.exists():
            return self._require_ack(request, decode_result(result_path.read_bytes()))
        try:
            publish_json_exclusive(str(intent_path), canonical_bytes(request.to_dict()))
        except FileExistsError as error:
            raise BalsamGroupParked("mutation intent is already owned; reconcile before retry") from error
        try:
            result = self._call(request)
        except BalsamBridgeAmbiguous as error:
            ambiguous = BalsamWireResult.ambiguous(request, str(error))
            try:
                publish_json_exclusive(str(result_path), ambiguous.encode())
            except OSError:
                pass
            raise BalsamGroupParked(str(error)) from error
        except BalsamBridgeFailure as error:
            raise BalsamGroupParked("mutation acknowledgement is unknown; no automatic retry") from error
        try:
            publish_json_exclusive(str(result_path), result.encode())
        except OSError as error:
            raise BalsamGroupParked("mutation result publication is uncertain") from error
        return self._require_ack(request, result)


__all__ = ["BalsamBridge", "BalsamBridgeAmbiguous", "BalsamBridgeFailure", "BalsamProcessClient",
           "SubprocessBalsamBridge", "select_balsam_client_path", "validate_balsam_client_path"]
