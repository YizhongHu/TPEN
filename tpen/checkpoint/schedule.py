"""Semantic schedules for checkpoint publication.

The schedule is deliberately independent of callbacks and training events.  A
caller supplies the durable number of completed optimizer updates and marks a
terminal decision explicitly.  That keeps resume alignment a property of the
schedule itself rather than of a run-local event counter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class CheckpointSchedule(Protocol):
    """Decide whether a checkpoint boundary should be published.

    Implementations must treat ``completed_updates`` as the durable coordinate
    restored from trainer state.  ``terminal`` is a separate semantic boundary
    and must not be inferred from the update cadence.  Every implementation
    must return ``True`` when ``terminal`` is true, regardless of the update
    count.
    """

    def should_run(self, completed_updates: int, terminal: bool = False) -> bool:
        """Return whether this boundary is selected for publication."""


@dataclass(frozen=True, slots=True)
class TerminalOnly:
    """Select terminal boundaries and no periodic update boundary."""

    def should_run(self, completed_updates: int, terminal: bool = False) -> bool:
        """Return ``True`` only for an explicitly terminal boundary."""

        _validate_boundary(completed_updates, terminal)
        return terminal


@dataclass(frozen=True, slots=True)
class EveryNUpdates:
    """Select every ``every_n`` completed optimizer updates and terminal writes.

    Parameters
    ----------
    every_n : int
        Positive interval in durable completed-update counts.  The first
        periodic boundary is update ``every_n``; a resumed run therefore keeps
        the same phase without replaying prior occurrences.
    """

    every_n: int

    def __post_init__(self) -> None:
        _validate_positive_int(self.every_n, "every_n")

    def should_run(self, completed_updates: int, terminal: bool = False) -> bool:
        """Return ``True`` on an interval boundary or terminal boundary."""

        _validate_boundary(completed_updates, terminal)
        if terminal:
            return True
        # Zero completed updates is a valid terminal state, not a periodic
        # update boundary.  This matters for max_steps=0 and vacuum runs.
        return completed_updates > 0 and completed_updates % self.every_n == 0


@dataclass(frozen=True, slots=True, init=False)
class ExplicitUpdates:
    """Select an explicit set of durable completed-update boundaries.

    Parameters
    ----------
    updates : iterable of int
        Positive completed-update counts to select.  The values are
        normalized to a frozen set, making decisions deterministic and
        independent of input ordering or duplicate entries.
    """

    updates: frozenset[int]

    def __init__(self, updates: Iterable[int]) -> None:
        try:
            entries = tuple(updates)
        except TypeError as exc:
            raise TypeError("updates must be an iterable of integers") from exc
        # Validate before deduplication: bool is an int subclass, and
        # frozenset({1, True}) would otherwise erase the invalid bool entry.
        for update in entries:
            _validate_positive_int(update, "updates entry")
        normalized = frozenset(entries)
        object.__setattr__(self, "updates", normalized)

    def should_run(self, completed_updates: int, terminal: bool = False) -> bool:
        """Return ``True`` for a listed update or any terminal boundary."""

        _validate_boundary(completed_updates, terminal)
        if terminal:
            return True
        return completed_updates > 0 and completed_updates in self.updates


def _validate_boundary(completed_updates: int, terminal: bool) -> None:
    """Validate the common schedule decision inputs."""

    if completed_updates < 0:
        raise ValueError("completed_updates must be non-negative")


def _validate_positive_int(value: int, label: str) -> None:
    """Validate a positive integer schedule value."""

    if type(value) is not int:
        raise TypeError(f"{label} must be an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{label} must be at least 1")


__all__ = [
    "CheckpointSchedule",
    "EveryNUpdates",
    "ExplicitUpdates",
    "TerminalOnly",
]
