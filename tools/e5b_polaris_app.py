"""Balsam ApplicationDefinition for the E5b Polaris pilot.

This file is staged as qualification tooling.  It is intentionally not
imported by :mod:`tpen.execution`; the launcher imports it only after the
submitter has registered the private Site and App.
"""

from __future__ import annotations

from balsam.api import ApplicationDefinition


class PolarisE5bApp(ApplicationDefinition):
    """Run the E4 worker gateway inside each Balsam-launched process."""

    environment_variables = {}
    site = 0

    def run(self) -> int:
        """Pass the launched Job identity explicitly into the worker gateway."""

        from tools.e5b_polaris_runner import run_attempt

        return int(run_attempt(job_id=self.job.id, job_data=self.job.data or {}))


__all__ = ["PolarisE5bApp"]
