"""Small explicit adapter seam for the pinned SDK.

The adapter keeps the bridge's finite operation surface separate from the
third-party model objects.  Full site/application lifecycle belongs to later
deployment work; this module only provides injected model-shaped operations.
"""

from __future__ import annotations

from typing import Any

from .application import BalsamControlRefusal, InjectedSDKOperations


PINNED_BALSAM_VERSION = "0.7.0.a24"


def require_pinned_version(version: object) -> str:
    """Return the only SDK version accepted by the control project."""

    if version != PINNED_BALSAM_VERSION:
        raise BalsamControlRefusal("loaded Balsam SDK version does not match the lock")
    return PINNED_BALSAM_VERSION


class DocumentedSDKOperations:
    """Adapter for the five supported operations on an imported SDK API.

    The SDK API object is injected for tests.  A production adapter is built
    only after the client path has been validated by :mod:`bridge`.
    """

    def __init__(self, api: Any) -> None:
        self._api = api

    @classmethod
    def from_imported_sdk(cls) -> InjectedSDKOperations:
        """Import the pinned SDK API at the final import boundary."""

        try:
            from balsam import api
            import balsam
        except ImportError as error:  # pragma: no cover - depends on control env.
            raise BalsamControlRefusal("pinned Balsam SDK is unavailable") from error
        require_pinned_version(getattr(balsam, "__version__", None))
        return cls(api)

    def register(self, group_id: str, group_digest: str, deployment_digest: str) -> str:
        """Refuse until a typed deployment supplies the documented model fields."""

        del group_id, group_digest, deployment_digest
        raise BalsamControlRefusal("group registration requires a typed deployment adapter")

    def observe(self, group_id: str, backend_job_id: str) -> str:
        """Read one model through the finite, bound identity seam."""

        del group_id
        try:
            model_id = int(backend_job_id)
        except (TypeError, ValueError) as error:
            raise BalsamControlRefusal("backend Job identity is not an SDK model id") from error
        try:
            job = self._api.Job.objects.get(id=model_id)
            state = job.state
        except (AttributeError, LookupError, OSError, TypeError, ValueError) as error:
            raise BalsamControlRefusal("SDK API lacks the bound Job observation operation") from error
        if type(state) is not str or not state.strip():
            state = str(state) if state is not None else ""
        if not state.strip():
            raise BalsamControlRefusal("SDK returned a non-primitive Job state")
        return state

    def submit(self, group_id: str, backend_job_id: str, batch_digest: str) -> str:
        """Refuse until a typed deployment supplies the documented model fields."""

        del group_id, backend_job_id, batch_digest
        raise BalsamControlRefusal("batch submission requires a typed deployment adapter")

    def rearm(self, group_id: str, backend_job_id: str, recovery_digest: str) -> str:
        """Refuse until recovery admission supplies an explicit rearm adapter."""

        del group_id, backend_job_id, recovery_digest
        raise BalsamControlRefusal("rearm requires explicit recovery admission")

    def readback(self, group_id: str, backend_job_id: str) -> str:
        """Read one model through the same bound observation operation."""

        return self.observe(group_id, backend_job_id)
