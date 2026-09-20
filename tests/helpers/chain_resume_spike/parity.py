"""Continuation parity over SUBSEQUENT DRAWS, with first-diverging-channel attribution.

Parity here is a function of what the system draws *after* the restore -- next
proposals, acceptance uniforms, walker positions, the parameter noise term and
the cadence measurements. It is never a comparison of serialized state bytes,
and never a comparison of deterministic outputs alone. Spike DS-A0 compared
deterministic outputs, passed with its RNG entirely discarded, and its
non-vacuity control inherited the same blindness.

:func:`first_divergence` reports *where* two traces first differ, in the order a
step actually emits its channels. A mutation arm then asserts that the reported
channel belongs to the limb it disabled. An arm that merely goes red proves
nothing about which mutant ran: two limbs both produce redness, and only the
channel distinguishes them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .fixture import CHANNELS
from .restore_limbs import LIMB_CHANNELS, RestoreLimb


@dataclass(frozen=True)
class Divergence:
    """The first point at which two traces differ.

    Parameters
    ----------
    index : int
        Position within the compared traces, zero-based.
    step : int or None
        The logical step recorded at that position, when both traces agree on
        it. ``None`` when the traces disagree about the step itself.
    channel : str
        Which observable channel diverged first, in emission order. The
        sentinel ``"length"`` means the traces had different lengths, and
        ``"step"`` means they disagreed on the step index.
    left : Any
        Value from the first trace.
    right : Any
        Value from the second trace.
    """

    index: int
    step: int | None
    channel: str
    left: Any
    right: Any

    def describe(self) -> str:
        """Return a one-line diagnostic naming the channel and both values."""

        return (
            f"first divergence at index {self.index} (step {self.step}) in channel "
            f"{self.channel!r}: {self.left!r} != {self.right!r}"
        )


def first_divergence(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> Divergence | None:
    """Return the first divergence between two traces, or ``None`` if identical.

    Channels are compared in :data:`~tests.helpers.chain_resume_spike.fixture.CHANNELS`
    order within each step, because "first" is only meaningful against the order
    a step actually emits them. A length mismatch is itself reported as a
    divergence rather than raising: an attempt that stopped early is data.

    Parameters
    ----------
    left, right : sequence of dict
        Step traces as produced by ``ContinuationSystem.step_once``.

    Returns
    -------
    Divergence or None
        ``None`` exactly when the two traces are equal.
    """

    for index in range(min(len(left), len(right))):
        left_step, right_step = left[index], right[index]
        if left_step.get("step") != right_step.get("step"):
            return Divergence(
                index=index,
                step=None,
                channel="step",
                left=left_step.get("step"),
                right=right_step.get("step"),
            )
        for channel in CHANNELS:
            if left_step[channel] != right_step[channel]:
                return Divergence(
                    index=index,
                    step=int(left_step["step"]),
                    channel=channel,
                    left=left_step[channel],
                    right=right_step[channel],
                )
    if len(left) != len(right):
        return Divergence(
            index=min(len(left), len(right)),
            step=None,
            channel="length",
            left=len(left),
            right=len(right),
        )
    return None


def channels_fed_by(limb: RestoreLimb) -> frozenset[str]:
    """Return the observable channels ``limb`` feeds."""

    return LIMB_CHANNELS[limb]


def divergence_is_attributable_to(divergence: Divergence, limb: RestoreLimb) -> bool:
    """Return whether ``divergence`` fell in a channel ``limb`` actually feeds.

    This is the attribution test each mutation arm applies. It is stricter than
    "the arm went red": redness alone is produced by every limb, and by an
    unrelated bug, so an arm that only checked redness would pass for the wrong
    mutant as readily as the right one.
    """

    return divergence.channel in LIMB_CHANNELS[limb]


def assert_channel_map_is_disjoint_and_total() -> None:
    """Fail if the limb-to-channel map is not a partition of :data:`CHANNELS`.

    Disjointness makes attribution possible: a channel fed by two limbs could
    not distinguish which mutant ran, and the two clauses would mask each
    other. Totality makes every limb observable: a limb feeding no channel has
    a vacuous mutation arm, which is exactly the DS-A0 shape.

    Exposed as a function rather than written inline in one test so the native
    arm can apply the same check to its own channel map.
    """

    seen: set[str] = set()
    for limb, channels in LIMB_CHANNELS.items():
        overlap = seen & channels
        if overlap:
            raise AssertionError(f"limb {limb.value} shares channels {sorted(overlap)}")
        if not channels:
            raise AssertionError(f"limb {limb.value} feeds no observable channel")
        seen |= channels
    if seen != set(CHANNELS):
        raise AssertionError(
            f"channel map is not total: unmapped {sorted(set(CHANNELS) - seen)}, "
            f"unknown {sorted(seen - set(CHANNELS))}"
        )


__all__ = [
    "Divergence",
    "assert_channel_map_is_disjoint_and_total",
    "channels_fed_by",
    "divergence_is_attributable_to",
    "first_divergence",
]
