"""Deterministic rank-sharded fixtures for the Accelerate DDP worker.

TAKEN VERBATIM from ``tests/spikes/native_ddp/fixtures.py`` at DS-N0's
terminalized tip ``36d5900e10521ee3561dc3153714a0c70a25ed4f``. The two spikes
must be scored on IDENTICAL inputs or the DG0 comparison compares fixtures
rather than runtimes, so this file is a copy and must stay one -- if it needs to
change, the change belongs in a note explaining why the comparison survives it.

Copied rather than imported because DS-N0's tree never landed on ``dev``: PR #470
was closed unmerged, and its branch is a rescue push retained only because the
commit would otherwise have been unreachable. Importing across spike packages
would also couple two deliberately independent prototypes.
"""

from __future__ import annotations

import torch


def scientific_fixture(
    world_size: int, rank: int, *, kind: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return feature and energy shards, including the M2 uneven topology."""

    if kind == "m2":
        if world_size != 3:
            raise ValueError("the m2 fixture requires world_size=3")
        shards = (
            (
                ((0.2, 0.1), (-0.5, 0.3), (0.7, -0.2), (0.4, 0.9), (-0.1, 0.6)),
                (1.0, float("nan"), 2.0, -1.0, float("nan")),
            ),
            (
                ((0.6, -0.4), (-0.2, 0.8), (0.5, 0.2)),
                (float("nan"), float("nan"), float("nan")),
            ),
            (
                ((-0.4, 0.3), (0.3, 0.7), (0.9, -0.8), (-0.7, 0.2), (0.1, 0.5), (0.2, -0.3), (0.5, 0.4)),
                (3.0, -2.0, float("nan"), 1.0, 0.5, 2.5, float("nan")),
            ),
        )
    elif kind == "regular":
        if world_size != 2:
            raise ValueError("the regular fixture requires world_size=2")
        shards = (
            (
                ((0.2, 0.1), (-0.5, 0.3), (0.7, -0.2), (0.4, 0.9)),
                (1.0, 2.0, 0.5, float("nan")),
            ),
            (
                ((-0.3, 0.8), (0.4, -0.1), (1.1, 0.2)),
                (3.0, -1.0, 2.0),
            ),
        )
    elif kind == "all_invalid":
        if world_size != 2:
            raise ValueError("the all_invalid fixture requires world_size=2")
        shards = (
            (((0.2, 0.1), (-0.5, 0.3)), (float("nan"), float("inf"))),
            (((0.7, -0.2), (0.4, 0.9), (-0.3, 0.8)), (float("nan"), float("inf"), float("nan"))),
        )
    else:
        raise ValueError(f"unknown scientific fixture {kind!r}")

    features, energy = shards[rank]
    return (
        torch.tensor(features, dtype=torch.float64),
        torch.tensor(energy, dtype=torch.float64),
    )


__all__ = ["scientific_fixture"]
