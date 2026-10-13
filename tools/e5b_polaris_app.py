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
from pathlib import Path
from balsam.api import ApplicationDefinition


PRE_E3_FAILURE_EXIT_CODE = 2


def _write_pre_e3_failure(job: object, exc: BaseException) -> None:
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
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class PolarisE5bApp(ApplicationDefinition):
    """Run the E4 worker gateway inside each Balsam-launched process."""

    environment_variables = {}
    site = 0

    def shell_preamble(self) -> str:
        """Verify the inherited Polaris MPI libraries before Python starts."""

        return r'''set -eu
echo "TPEN_E5B_PRECHECK host=$(hostname)"
echo "TPEN_E5B_PRECHECK LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}"
echo "TPEN_E5B_PRECHECK PATH=${PATH-}"
echo "TPEN_E5B_PRECHECK PYTHONPATH=${PYTHONPATH-}"
PYTHON="${TPEN_E5B_PYTHON:-python}"
ldconfig -p | grep -m1 'libfabric.so.1'
"$PYTHON" -c 'import ctypes.util, torch; print("TPEN_E5B_TORCH", torch.__version__, torch.__file__); print("TPEN_E5B_LIBFABRIC", ctypes.util.find_library("fabric")); assert ctypes.util.find_library("fabric")'
TORCH_GLOBAL="$($PYTHON -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib" / "libtorch_global_deps.so")')"
echo "TPEN_E5B_LDD $TORCH_GLOBAL"
ldd "$TORCH_GLOBAL" | tee /dev/stderr | grep -m1 'libfabric.so.1'
'''

    def run(self) -> int:
        """Pass the launched Job identity explicitly into the worker gateway."""

        try:
            from tools.e5b_polaris_runner import run_attempt

            result = int(run_attempt(job_id=self.job.id, job_data=self.job.data or {}))
        except Exception as exc:
            _write_pre_e3_failure(self.job, exc)
            raise SystemExit(PRE_E3_FAILURE_EXIT_CODE) from exc
        if result:
            # Returning an integer only prints BALSAM-RETURN-VALUE and exits
            # zero.  Convert parks and other non-completions to a non-retry
            # process status while leaving E3's durable receipt as authority.
            raise SystemExit(PRE_E3_FAILURE_EXIT_CODE)
        return 0


__all__ = ["PolarisE5bApp"]
