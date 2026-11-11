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
    if name in {"restart_ready", "allocation_lost", "timeout", "timed_out", "run_timeout"}:
        return BalsamJobState.ALLOCATION_LOST
    if name in {"failed", "permanent_failure", "job_failed"}:
        return BalsamJobState.PERMANENT_FAILURE
    if name in {
        "created",
        "awaiting_parents",
        "ready",
        "staged",
        "staged_in",
        "run_done",
        "staged_out",
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
        not placed in the scientific source configuration.  The deployment
        paths ``checkout_root``, ``app_root``, ``site_packages``,
        ``python_exe``, ``pilot_root``, and ``ld_library_prefix`` are required
        runtime inputs; this adapter provides no facility-specific defaults.
    """

    def __init__(self, site_root: str | Path, backend_options: Mapping[str, Any]):
        self.site_root = Path(site_root)
        self.backend_options = dict(backend_options)

    def _bindings(self) -> tuple[Any, Any, type[Any]]:
        """Load Balsam and bind the configured application to this Site."""

        # Validate operator-supplied paths before touching the optional backend.
        environment = self._worker_environment()
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
        app_class.python_exe = self._required_backend_path("python_exe")
        app_class.environment_variables = environment
        return api, config, app_class

    def _required_backend_path(self, key: str) -> str:
        """Return a required configured deployment path.

        Parameters
        ----------
        key : str
            Backend option name whose value must be supplied by the caller.

        Returns
        -------
        str
            The configured path.

        Raises
        ------
        ValueError
            If the option is absent or blank.
        """

        value = self.backend_options.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"required Balsam deployment path {key!r} is missing")
        return str(value)

    def _required_backend_paths(self, key: str) -> list[str]:
        """Return a non-empty configured list of deployment paths."""

        value = self.backend_options.get(key)
        if isinstance(value, (str, bytes)) or value is None:
            raise ValueError(f"required Balsam deployment path list {key!r} is missing")
        paths = [str(entry) for entry in value if str(entry).strip()]
        if not paths:
            raise ValueError(f"required Balsam deployment path list {key!r} is missing")
        return paths

    def _worker_environment(self) -> dict[str, str]:
        """Build app environment by prepending to inherited facility paths."""

        checkout = self._required_backend_path("checkout_root")
        app_root = self._required_backend_path("app_root")
        site_packages = self._required_backend_path("site_packages")
        python_exe = self._required_backend_path("python_exe")
        pilot_root = self._required_backend_path("pilot_root")
        ld_prefix = self._required_backend_paths("ld_library_prefix")

        def prepend(entries: list[str], inherited: str) -> str:
            """Prepend non-empty additions without discarding the base env."""

            values = [entry for entry in entries if entry]
            if inherited:
                values.append(inherited)
            return ":".join(values)

        path_entries = [str(Path(checkout) / "bin") if checkout else "", app_root]
        python_entries = [checkout, site_packages]
        environment = {
            "BALSAM_SITE_PATH": str(self.site_root),
            "TPEN_E5B_ROOT": pilot_root,
            "TPEN_E5B_PYTHON": python_exe,
        }
        environment["PATH"] = prepend(path_entries, os.environ.get("PATH", ""))
        environment["PYTHONPATH"] = prepend(python_entries, os.environ.get("PYTHONPATH", ""))
        environment["LD_LIBRARY_PATH"] = prepend(
            ld_prefix, os.environ.get("LD_LIBRARY_PATH", "")
        )
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
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
