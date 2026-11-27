"""Finite control-side records and injected SDK operations.

The control project owns the optional Balsam dependency.  Only primitive,
typed values cross into this module; executor records remain owned by the
installed executor artifact and are decoded by the bridge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Protocol

from tpen_executor._codec import sha256_text, text


CONTROL_PROJECT_VERSION = "0.1.0"


class BalsamControlError(RuntimeError):
    """Base error for a typed control-side operation."""


class BalsamControlRefusal(BalsamControlError):
    """The requested operation was rejected before an external effect."""


class BalsamControlAmbiguous(BalsamControlError):
    """An operation may have had an external effect and must be reconciled."""


@dataclass(frozen=True, slots=True)
class SDKIdentity:
    """Identity of the loaded control toolchain and shared wire artifact.

    Parameters
    ----------
    sdk_version : str
        Exact imported Balsam version.
    api_module : str
        Module exposing the injected/documented SDK API.
    executor_artifact_digest : str
        SHA-256 digest of the installed executor artifact identity record.
    project_digest : str
        SHA-256 digest of the control project's source manifest.
    lock_digest : str
        SHA-256 digest of the exact control lock file.
    """

    schema: ClassVar[str] = "tpen-balsam-control/sdk-identity-v1"
    sdk_version: str
    api_module: str
    executor_artifact_digest: str
    project_digest: str
    lock_digest: str

    def __post_init__(self) -> None:
        text(self.sdk_version, "sdk_version")
        text(self.api_module, "api_module")
        for name in ("executor_artifact_digest", "project_digest", "lock_digest"):
            sha256_text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class FixedBatchOutputPaths:
    """The only SDK optional parameters exposed by the bridge.

    Parameters
    ----------
    stdout : str
        Absolute path for the batch's standard output.
    stderr : str
        Absolute path for the batch's standard error.
    """

    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        for name in ("stdout", "stderr"):
            value = text(getattr(self, name), name)
            if not Path(value).is_absolute():
                raise BalsamControlRefusal(f"{name} must be absolute")

    def as_optional_params(self) -> dict[str, str]:
        """Return the closed SDK parameter mapping."""

        return {"tpen_stdout": self.stdout, "tpen_stderr": self.stderr}


def fixed_batch_output_paths(values: Mapping[str, str] | FixedBatchOutputPaths) -> FixedBatchOutputPaths:
    """Decode exactly the two supported output path keys.

    Arbitrary SDK parameters are deliberately not accepted.  The values are
    generated from a persisted carrier intent by a later lifecycle slice.
    """

    if isinstance(values, FixedBatchOutputPaths):
        return values
    if not isinstance(values, Mapping) or set(values) != {"tpen_stdout", "tpen_stderr"}:
        raise BalsamControlRefusal("only tpen_stdout and tpen_stderr are supported")
    return FixedBatchOutputPaths(values["tpen_stdout"], values["tpen_stderr"])


class InjectedSDKOperations(Protocol):
    """Finite SDK operation surface used by the bridge.

    Implementations adapt the pinned SDK's documented models.  Tests inject a
    local fake; this protocol never accepts a method name or kwargs bag.
    """

    def register(self, group_id: str, group_digest: str, deployment_digest: str) -> str:
        """Register one immutable group and return its backend identity."""
        ...

    def observe(self, group_id: str, backend_job_id: str) -> str:
        """Observe one bound backend job and return its primitive state."""
        ...

    def submit(self, group_id: str, backend_job_id: str, batch_digest: str) -> str:
        """Submit one explicitly authorized batch and return its identity."""
        ...

    def rearm(self, group_id: str, backend_job_id: str, recovery_digest: str) -> str:
        """Rearm one already-bound job and return its primitive state."""
        ...

    def readback(self, group_id: str, backend_job_id: str) -> str:
        """Read back one bound job and return its primitive state."""
        ...


def make_application_definition(
    api: Any,
    *,
    invoke_group: Callable[[Any], Any],
    exit_code: Callable[[Any, Any], int],
    contract: Any,
    record_failure: Callable[[Any, BaseException], None],
    record_observation: Callable[[Any, object], None],
    application_name: str,
    deployment_digest: str,
) -> type:
    """Build the generated application policy around an injected API.

    Parameters
    ----------
    api : object
        Control-side API object exposing ``ApplicationDefinition``.
    invoke_group : callable
        Generic executor group entrypoint.  It owns scientific completion.
    exit_code : callable
        Existing typed group-exit policy, injected to avoid a second classifier.
    contract : object
        Explicit qualified exit contract consumed by ``exit_code``.
    record_failure, record_observation : callable
        Durable control observation hooks; neither may authorize reentry.
    application_name, deployment_digest : str
        Immutable generated identities attached to the application class.

    Returns
    -------
    type
        An ``ApplicationDefinition`` subclass with explicit non-retry hooks.
    """
    base = getattr(api, "ApplicationDefinition", None)
    if not isinstance(base, type) or not all(callable(value) for value in
                                              (invoke_group, exit_code, record_failure, record_observation)):
        raise BalsamControlRefusal("application requires a typed API and control callbacks")
    if not isinstance(application_name, str) or not application_name.strip():
        raise BalsamControlRefusal("application_name must be non-empty")
    if not isinstance(deployment_digest, str) or len(deployment_digest) != 64:
        raise BalsamControlRefusal("deployment_digest must be a SHA-256 identity")

    class GeneratedApplication(base):
        """Generated app whose hooks never inherit automatic reentry."""

        tpen_application_name = application_name
        tpen_deployment_digest = deployment_digest

        def run(self) -> int:
            """Invoke the group and turn only typed reconciliation into success."""
            job = getattr(self, "job", None)
            try:
                code = exit_code(invoke_group(job), contract)
            except BaseException as error:
                try:
                    record_failure(job, error)
                finally:
                    raise SystemExit(2) from error
            if code:
                raise SystemExit(code)
            return 0

        def handle_timeout(self) -> None:
            """Leave a timed-out job non-runnable for explicit reentry."""
            job = getattr(self, "job", None)
            if job is not None and hasattr(job, "state"):
                job.state = "failed"
            record_observation(job, "timeout-awaiting-explicit-reentry")

        def handle_error(self, error: BaseException | None = None) -> None:
            """Leave an errored job non-runnable without an inherited retry."""
            job = getattr(self, "job", None)
            if job is not None and hasattr(job, "state"):
                job.state = "failed"
            record_observation(job, "error-awaiting-explicit-reentry" if error is None else error)

        def request_rearm(self, *args: Any, **kwargs: Any) -> None:
            """Refuse implicit rearm until the explicit recovery slice exists."""
            del args, kwargs
            raise BalsamControlRefusal("rearm requires an explicit grant-bound transaction")

    return GeneratedApplication
