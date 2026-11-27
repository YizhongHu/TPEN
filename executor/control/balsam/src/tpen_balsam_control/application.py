"""Finite control-side records and injected SDK operations.

The control project owns the optional Balsam dependency.  Only primitive,
typed values cross into this module; executor records remain owned by the
installed executor artifact and are decoded by the bridge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

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
