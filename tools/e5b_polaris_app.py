"""Balsam ApplicationDefinition for the E5b Polaris pilot.

This file is staged as qualification tooling.  It is intentionally not
imported by :mod:`tpen.execution`; the launcher imports it only after the
submitter has registered the private Site and App.

Balsam API facts relied on
---------------------------
The installed Polaris Balsam 0.7.0.a24 launcher runs ``ApplicationDefinition
.run`` in ``balsam.site.launcher.python_runner`` and treats a non-zero child
status in ``balsam.site.launcher._serial_mode_worker.Worker`` as an error.  Its
retry set is ``[-11, 1, 255, 12345]`` (the measured installation source at
``/home/rhu/.local/lib/python3.13/site-packages/balsam/site/launcher/
_serial_mode_worker.py:23-26,78-89``).  The wrapper therefore writes a durable
pre-E3 failure record and raises ``SystemExit(2)``: the process is visibly
unsuccessful, while exit 2 is outside that installation's blind-retry set.
This is a qualification deployment fact, not a claim about all Balsam
versions.
"""

from __future__ import annotations

import json
import os
import ctypes
import ctypes.util
import sys
from pathlib import Path
from typing import Any, Mapping

from balsam.api import ApplicationDefinition


PRE_E3_FAILURE_EXIT_CODE = 2


class PreE3EnvironmentError(RuntimeError):
    """Raised when worker bootstrap cannot prove its native environment."""

    def __init__(self, message: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.evidence = dict(evidence)


def _torch_probe() -> dict[str, str]:
    """Import torch and return the identity needed in a bootstrap receipt."""

    import torch

    return {
        "version": str(getattr(torch, "__version__", "unknown")),
        "file": str(getattr(torch, "__file__", "unknown")),
    }


def _library_search(library: str) -> dict[str, Any]:
    """Resolve one shared library using Python and inherited search paths.

    ``ldconfig`` and ``ldd`` are intentionally not used: neither is guaranteed
    to be available in a compute-node job shell.  The evidence retains every
    candidate derived from ``LD_LIBRARY_PATH`` and the Python resolver result.
    """

    inherited = os.environ.get("LD_LIBRARY_PATH", "")
    directories = [entry for entry in inherited.split(os.pathsep) if entry]
    searched_paths = [
        str(Path(directory) / candidate)
        for directory in directories
        for candidate in (f"lib{library}.so.1", f"lib{library}.so")
    ]
    existing_paths = [path for path in searched_paths if Path(path).is_file()]
    ctypes_result = ctypes.util.find_library(library)
    evidence = {
        "library": library,
        "ld_library_path": inherited,
        "searched_paths": searched_paths,
        "existing_paths": existing_paths,
        "ctypes_find_library": ctypes_result,
    }
    candidate = existing_paths[0] if existing_paths else ctypes_result
    if not candidate:
        raise PreE3EnvironmentError(f"could not resolve lib{library}", {"library": evidence})
    try:
        ctypes.CDLL(candidate)
    except OSError as exc:
        evidence["load_error"] = str(exc)
        raise PreE3EnvironmentError(f"could not load {candidate}", {"library": evidence}) from exc
    evidence["loaded"] = candidate
    return evidence


def _environment_precheck() -> dict[str, Any]:
    """Validate native bootstrap dependencies before importing the worker."""

    evidence: dict[str, Any] = {"python": {"executable": sys.executable}}
    try:
        evidence["torch"] = _torch_probe()
        evidence["library"] = _library_search("fabric")
    except PreE3EnvironmentError:
        raise
    except Exception as exc:
        raise PreE3EnvironmentError(
            f"native environment precheck failed: {type(exc).__name__}: {exc}", evidence
        ) from exc
    return evidence


def _write_pre_e3_failure(
    job: object,
    exc: BaseException,
    *,
    precheck: Mapping[str, Any] | None = None,
) -> None:
    """Publish a durable reason when worker bootstrap fails before E3.

    The path is supplied in Balsam ``Job.data`` by the submitter.  The
    fallback is intentionally best-effort: if Balsam cannot provide a pilot
    root, there is no shared location in which this qualification wrapper can
    leave a record.
    """

    data = getattr(job, "data", None) or {}
    root_value = data.get("pilot_root") if isinstance(data, dict) else None
    if not root_value:
        return
    path = Path(str(root_value)) / "pre-e3-failure.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    record = {
        "schema": "tpen.execution/e5b-pre-e3-failure-v1",
        "job_id": str(getattr(job, "id", "unknown")),
        "phase": "before-e3-or-unknown",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "exit_code": PRE_E3_FAILURE_EXIT_CODE,
        "reason": "Balsam application failed before a confirmed E3 outcome; park without retry",
    }
    if precheck is not None:
        record["precheck"] = dict(precheck)
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class PolarisE5bApp(ApplicationDefinition):
    """Run the E4 worker gateway inside each Balsam-launched process."""

    environment_variables = {}
    site = 0

    def shell_preamble(self) -> str:
        """Log inherited paths; Python performs the durable precheck."""

        return r'''set -u -o pipefail
echo "TPEN_E5B_PRECHECK host=$(hostname)"
echo "TPEN_E5B_PRECHECK LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}"
echo "TPEN_E5B_PRECHECK PATH=${PATH-}"
echo "TPEN_E5B_PRECHECK PYTHONPATH=${PYTHONPATH-}"
'''

    def run(self) -> int:
        """Pass the launched Job identity explicitly into the worker gateway."""

        try:
            precheck = _environment_precheck()
            from tools.e5b_polaris_runner import run_attempt

            result = int(run_attempt(job_id=self.job.id, job_data=self.job.data or {}))
        except Exception as exc:
            _write_pre_e3_failure(
                self.job,
                exc,
                precheck=getattr(exc, "evidence", None),
            )
            raise SystemExit(PRE_E3_FAILURE_EXIT_CODE) from exc
        if result:
            # Returning an integer only prints BALSAM-RETURN-VALUE and exits
            # zero.  Convert parks and other non-completions to a non-retry
            # process status while leaving E3's durable receipt as authority.
            raise SystemExit(PRE_E3_FAILURE_EXIT_CODE)
        return 0


__all__ = ["PolarisE5bApp"]
