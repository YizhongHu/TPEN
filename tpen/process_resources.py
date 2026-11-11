"""Typed process resource readings and the stdlib ``resource`` probe.

Linux and macOS expose the seven fields used here through ``getrusage``
(``getrusage(2)`` on `man7.org <https://man7.org/linux/man-pages/man2/getrusage.2.html>`_
and the Apple `getrusage(2) <https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/getrusage.2.html>`_).
Their values may legitimately be zero; this probe therefore treats only a
failed process-level read as unavailable evidence.
"""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Union


_BYTES_PER_MIB = 1024 * 1024


class ResourceScope(Enum):
    """Resource scope supported by this layer."""

    PROCESS = "process"


@dataclass(frozen=True)
class ResourceUnavailable:
    """Evidence that one process resource counter could not be read."""

    reason: str


ResourceReading = Union[float, int, ResourceUnavailable]


@dataclass(frozen=True)
class ProcessResourceBaseline:
    """Process resource counters captured at run start."""

    user_cpu_seconds: ResourceReading
    system_cpu_seconds: ResourceReading
    read_block_operations: ResourceReading
    write_block_operations: ResourceReading
    voluntary_context_switches: ResourceReading
    involuntary_context_switches: ResourceReading
    peak_rss_mb: ResourceReading


@dataclass(frozen=True)
class ProcessResourceResult:
    """Process resource counters consumed for one run."""

    user_cpu_seconds: ResourceReading
    system_cpu_seconds: ResourceReading
    read_block_operations: ResourceReading
    write_block_operations: ResourceReading
    voluntary_context_switches: ResourceReading
    involuntary_context_switches: ResourceReading
    peak_rss_mb: ResourceReading


def _peak_rss_mb(value: ResourceReading) -> ResourceReading:
    """Normalize ``ru_maxrss`` to MiB on both supported Unix conventions."""

    if isinstance(value, ResourceUnavailable):
        return value
    divisor = _BYTES_PER_MIB if sys.platform == "darwin" else 1024
    return float(value) / divisor


def _subtract(current: ResourceReading, baseline: ResourceReading) -> ResourceReading:
    """Subtract counters while retaining an unavailable operand as evidence."""

    if isinstance(current, ResourceUnavailable):
        return current
    if isinstance(baseline, ResourceUnavailable):
        return baseline
    return current - baseline


class ProcessRUsageProbe:
    """Capture process counters from :func:`resource.getrusage`.

    Parameters
    ----------
    scope : ResourceScope, optional
        Only ``ResourceScope.PROCESS`` is supported in this layer.
    """

    def __init__(self, scope: ResourceScope = ResourceScope.PROCESS) -> None:
        if scope is not ResourceScope.PROCESS:
            raise ValueError("ProcessRUsageProbe supports only ResourceScope.PROCESS")
        self.scope = scope

    def read(self) -> ProcessResourceBaseline:
        """Read the current process usage as a baseline-shaped record."""

        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
        except OSError as exc:
            unavailable = ResourceUnavailable(f"{type(exc).__name__}: {exc}")
            return ProcessResourceBaseline(
                user_cpu_seconds=unavailable,
                system_cpu_seconds=unavailable,
                read_block_operations=unavailable,
                write_block_operations=unavailable,
                voluntary_context_switches=unavailable,
                involuntary_context_switches=unavailable,
                peak_rss_mb=unavailable,
            )
        return ProcessResourceBaseline(
            user_cpu_seconds=usage.ru_utime,
            system_cpu_seconds=usage.ru_stime,
            read_block_operations=usage.ru_inblock,
            write_block_operations=usage.ru_oublock,
            voluntary_context_switches=usage.ru_nvcsw,
            involuntary_context_switches=usage.ru_nivcsw,
            peak_rss_mb=_peak_rss_mb(usage.ru_maxrss),
        )

    def result(self, baseline: ProcessResourceBaseline) -> ProcessResourceResult:
        """Read terminal counters and calculate run deltas."""

        current = self.read()
        return ProcessResourceResult(
            user_cpu_seconds=_subtract(current.user_cpu_seconds, baseline.user_cpu_seconds),
            system_cpu_seconds=_subtract(current.system_cpu_seconds, baseline.system_cpu_seconds),
            read_block_operations=_subtract(current.read_block_operations, baseline.read_block_operations),
            write_block_operations=_subtract(current.write_block_operations, baseline.write_block_operations),
            voluntary_context_switches=_subtract(
                current.voluntary_context_switches, baseline.voluntary_context_switches
            ),
            involuntary_context_switches=_subtract(
                current.involuntary_context_switches, baseline.involuntary_context_switches
            ),
            peak_rss_mb=current.peak_rss_mb,
        )


__all__ = [
    "ProcessRUsageProbe",
    "ProcessResourceBaseline",
    "ProcessResourceResult",
    "ResourceScope",
    "ResourceUnavailable",
]
