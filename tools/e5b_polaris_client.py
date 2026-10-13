"""Lazy real-Balsam client for the bounded E5b Polaris qualification.

This module is qualification tooling, not part of :mod:`tpen.execution`.  It
adapts the installed Balsam API to the E4 ``BalsamClient`` seam only when a
submitter explicitly constructs the client.  A worker uses E4's backend-free
path and never imports this module.

Balsam API facts relied on
---------------------------
* ``ApplicationDefinition._set_client``, ``sync``, and ``submit`` are the
  installed Balsam 0.7.0.a24 API, verified on ALCF Polaris on 2026-10-13 by
  ``inspect.getsource(balsam.api.ApplicationDefinition)``.
* ``ApplicationDefinition.submit`` accepts ``workdir``, ``tags``, ``data``,
  ``num_nodes``, ``ranks_per_node``, ``threads_per_rank``, ``gpus_per_rank``,
  and ``wall_time_min``; the same installed source documents ``workdir`` as
  relative to the Site ``data/`` directory.
* ``Job.id`` is an integer-capable field in the installed
  ``balsam.api.Job`` model; E4 canonicalizes it to an opaque string.

The API facts above are implementation facts of the measured installation,
not a claim that Balsam is a TPEN dependency or that another site has the
same installation.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Mapping

from tpen.execution import BalsamJobState


def _state_name(value: Any) -> str:
    """Return a lower-case backend state name without importing Balsam types."""

    raw = getattr(value, "value", value)
    return str(raw).rsplit(".", 1)[-1].lower()


def _map_state(value: Any) -> BalsamJobState:
    """Map observed Balsam lifecycle states to E4's small typed vocabulary."""

    name = _state_name(value)
    if name in {"finished", "completed", "job_finished", "succeeded"}:
        return BalsamJobState.COMPLETED
    if name in {"restart_ready", "allocation_lost", "timeout", "timed_out"}:
        return BalsamJobState.ALLOCATION_LOST
    if name in {"failed", "permanent_failure", "job_failed"}:
        return BalsamJobState.PERMANENT_FAILURE
    if name in {
        "created",
        "awaiting_parents",
        "ready",
        "staged",
        "preprocessed",
        "running",
        "postprocessed",
        "queued",
    }:
        return BalsamJobState.ACTIVE
    return BalsamJobState.UNKNOWN


class PolarisBalsamClient:
    """Adapt one private Polaris Balsam Site to the E4 client protocol.

    Parameters
    ----------
    site_root : path-like
        Private Site directory owned by this qualification.
    backend_options : mapping
        Deployment paths and app settings.  These remain backend data and are
        not placed in the scientific source configuration.
    """

    def __init__(self, site_root: str | Path, backend_options: Mapping[str, Any]):
        self.site_root = Path(site_root)
        self.backend_options = dict(backend_options)

    def _bindings(self) -> tuple[Any, Any, type[Any]]:
        """Load Balsam and bind the configured application to this Site."""

        # This is the only import of the optional dependency in the wrapper.
        from balsam import api
        from balsam.config import SiteConfig

        config = SiteConfig(site_path=self.site_root)
        module = importlib.import_module(
            str(self.backend_options.get("app_module", "tools.e5b_polaris_app"))
        )
        app_class = getattr(
            module,
            str(self.backend_options.get("app_class", "PolarisE5bApp")),
        )
        app_class._set_client(config.client)
        app_class.site = config.site_id
        app_class.environment_variables = self._worker_environment()
        return api, config, app_class

    def _worker_environment(self) -> dict[str, str]:
        """Build serialized app environment from measured deployment paths."""

        checkout = str(self.backend_options.get("checkout_root", ""))
        site_packages = str(self.backend_options.get("site_packages", ""))
        path_entries = [entry for entry in (checkout, site_packages) if entry]
        inherited = os.environ.get("PYTHONPATH", "")
        if inherited:
            path_entries.append(inherited)
        environment = {
            "BALSAM_SITE_PATH": str(self.site_root),
            "TPEN_E5B_ROOT": str(self.backend_options.get("pilot_root", "")),
        }
        if path_entries:
            environment["PYTHONPATH"] = ":".join(path_entries)
        for key in ("LD_LIBRARY_PATH", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            if key in os.environ:
                environment[key] = os.environ[key]
        return environment

    def register_job(self, *, lineage_id: str, batch: Any, backend_options: Mapping[str, Any]) -> Any:
        """Create exactly one persistent Balsam Job for E4 registration."""

        del batch
        api, _config, app_class = self._bindings()
        workdir = Path(str(backend_options.get("workdir", f"e5b-{lineage_id}")))
        data = dict(backend_options.get("job_data", {}))
        data.setdefault("lineage_id", lineage_id)
        app_class.sync()
        job = app_class.submit(
            workdir=workdir,
            tags={"tpen_lineage": lineage_id, "tpen_slice": "e5b"},
            data=data,
            num_nodes=1,
            ranks_per_node=1,
            threads_per_rank=1,
            gpus_per_rank=0,
            wall_time_min=int(backend_options.get("wall_time_min", 5)),
        )
        # Keep the return mapping deliberately small; E4 owns id validation.
        return {"id": getattr(job, "id", None)}

    def observe_job(self, job_id: str) -> dict[str, Any]:
        """Read one backend Job for submitter-side reconciliation."""

        from balsam import api

        value: Any = int(job_id) if str(job_id).isdigit() else job_id
        job = api.Job.objects.get(id=value)
        state = _map_state(getattr(job, "state", None))
        state_data = getattr(job, "state_data", None) or {}
        metadata = {
            "raw_state": _state_name(getattr(job, "state", None)),
            "batch_job_id": getattr(job, "batch_job_id", None),
            "return_code": getattr(job, "return_code", None),
            "workdir": str(getattr(job, "workdir", "")),
            "state_data": dict(state_data) if isinstance(state_data, Mapping) else state_data,
            "run_count": state_data.get("run_count") if isinstance(state_data, Mapping) else None,
        }
        return {
            "job_id": getattr(job, "id", job_id),
            "state": state.value,
            "reason": "",
            "metadata": metadata,
        }


__all__ = ["PolarisBalsamClient"]
