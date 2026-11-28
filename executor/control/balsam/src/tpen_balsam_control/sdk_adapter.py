"""Small explicit adapter seam for the pinned SDK.

The adapter keeps the bridge's finite operation surface separate from the
third-party model objects.  Full site/application lifecycle belongs to later
deployment work; this module only provides injected model-shaped operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .application import (
    BalsamControlAmbiguous,
    BalsamControlRefusal,
    InjectedSDKOperations,
    fixed_batch_output_paths,
)


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

    def submit(self, group_id: str, backend_job_id: str, batch_digest: str,
               stdout_path: str | None = None, stderr_path: str | None = None,
               num_nodes: int | None = None, wall_time_min: int | None = None,
               job_mode: str | None = None, site_id: int | None = None,
               project: str | None = None, queue: str | None = None) -> str:
        """Create one BatchJob from the finite carrier fields.

        Parameters are intentionally primitive and closed at the bridge.  The
        parent directories are made visible and checked before calling the
        model's create operation so a rejected path cannot be mistaken for a
        successful external acknowledgement.
        """

        del batch_digest
        if stdout_path is None or stderr_path is None:
            raise BalsamControlRefusal("batch submission requires generated output paths")
        allocation = (num_nodes, wall_time_min, job_mode, site_id, project, queue)
        if any(value is None for value in allocation):
            raise BalsamControlRefusal("batch submission requires complete allocation fields")
        if type(num_nodes) is not int or num_nodes <= 0 or type(wall_time_min) is not int or wall_time_min <= 0:
            raise BalsamControlRefusal("batch allocation sizes must be positive integers")
        if type(site_id) is not int or site_id <= 0:
            raise BalsamControlRefusal("site_id must be a positive integer")
        if any(type(value) is not str or not value.strip() for value in (job_mode, project, queue)):
            raise BalsamControlRefusal("batch allocation names must be non-empty text")
        paths = fixed_batch_output_paths({"tpen_stdout": stdout_path, "tpen_stderr": stderr_path})
        parents = {Path(paths.stdout).parent, Path(paths.stderr).parent}
        try:
            for parent in parents:
                parent.mkdir(parents=True, exist_ok=True)
                if not parent.is_dir():
                    raise BalsamControlRefusal("BatchJob output parent is not a directory")
        except OSError as error:
            raise BalsamControlRefusal("BatchJob output parent is unavailable") from error
        batch_type = getattr(self._api, "BatchJob", None)
        manager = getattr(batch_type, "objects", None)
        create = getattr(manager, "create", None)
        if not callable(create):
            raise BalsamControlRefusal("SDK API lacks BatchJob.objects.create")
        try:
            record = create(
                num_nodes=num_nodes,
                wall_time_min=wall_time_min,
                job_mode=job_mode,
                site_id=site_id,
                project=project,
                queue=queue,
                optional_params=paths.as_optional_params(),
                filter_tags={
                    "tpen_group_id": group_id,
                    "tpen_backend_job_id": backend_job_id,
                },
            )
            identity = getattr(record, "id", None)
        except (AttributeError, LookupError, OSError, TypeError, ValueError) as error:
            raise BalsamControlAmbiguous("BatchJob create acknowledgement is unknown") from error
        if identity is None or not str(identity).strip():
            raise BalsamControlAmbiguous("BatchJob create returned no identity")
        return str(identity)

    def rearm(self, group_id: str, backend_job_id: str, recovery_digest: str) -> str:
        """Refuse until recovery admission supplies an explicit rearm adapter."""

        del group_id, backend_job_id, recovery_digest
        raise BalsamControlRefusal("rearm requires explicit recovery admission")

    def readback(self, group_id: str, backend_job_id: str) -> str:
        """Read one model through the same bound observation operation."""

        return self.observe(group_id, backend_job_id)
