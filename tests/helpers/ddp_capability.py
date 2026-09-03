"""Typed capability/topology probe for the CPU/Gloo subprocess harness (DF1).

Tests select themselves by capability, never by facility name
(``approved-ddp-program-materialization-2026-08-31``). A missing capability
must produce an explicit, attributable skip -- never a silent vanish and
never an unbounded hang while probing.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

_SPAWN_PROBE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class GlooSubprocessCapability:
    """Result of probing this machine for the CPU/Gloo subprocess harness.

    Parameters
    ----------
    gloo_available : bool
        Whether this torch build supports the Gloo backend at all.
    subprocess_spawn_available : bool
        Whether launching a plain Python subprocess succeeds within a bounded
        timeout. A sandboxed environment that denies subprocess creation
        must fail this explicitly rather than hang the harness.
    reasons : dict
        Capability name -> human reason, populated only for capabilities
        that are False.
    """

    gloo_available: bool
    subprocess_spawn_available: bool
    reasons: Mapping[str, str] = field(default_factory=dict)

    @property
    def sufficient(self) -> bool:
        """Return whether every probed capability is available."""

        return self.gloo_available and self.subprocess_spawn_available


def probe_gloo_capability() -> GlooSubprocessCapability:
    """Probe this machine for the capabilities the harness needs.

    Never raises; a probe failure is recorded as a False capability with a
    reason, not an exception, so callers can always build a skip message.
    """

    gloo_available = bool(torch.distributed.is_available()) and bool(
        torch.distributed.is_gloo_available()
    )
    reasons: dict[str, str] = {}
    if not gloo_available:
        reasons["gloo_available"] = (
            "torch.distributed.is_gloo_available() is False on this torch build"
        )

    subprocess_spawn_available = True
    try:
        completed = subprocess.run(
            [sys.executable, "-c", "pass"],
            timeout=_SPAWN_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            subprocess_spawn_available = False
            reasons["subprocess_spawn_available"] = (
                f"dry-run subprocess exited {completed.returncode}"
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        subprocess_spawn_available = False
        reasons["subprocess_spawn_available"] = (
            f"dry-run subprocess failed: {exc!r}"
        )

    return GlooSubprocessCapability(
        gloo_available=gloo_available,
        subprocess_spawn_available=subprocess_spawn_available,
        reasons=reasons,
    )


def missing_capability_reason(capability: GlooSubprocessCapability, name: str) -> str:
    """Return an explicit, attributable skip reason naming ``name``.

    ``name`` must be a field of :class:`GlooSubprocessCapability`
    (``"gloo_available"`` or ``"subprocess_spawn_available"``). Raises
    ``KeyError`` if ``name`` is not a recorded missing capability -- this
    function is only ever called after confirming the capability is absent.
    """

    detail = capability.reasons[name]
    return f"missing capability {name!r}: {detail}"


__all__ = [
    "GlooSubprocessCapability",
    "missing_capability_reason",
    "probe_gloo_capability",
]
