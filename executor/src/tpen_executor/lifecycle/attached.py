"""Lifecycle adapter for work inside an already configured allocation."""

from __future__ import annotations

from dataclasses import dataclass

from ..allocations import ConfiguredAllocation
from ..contracts import ReadyGroup
from ..results import AllocationObservation, OutcomeReceipt
from ..storage import AttemptFence, DirectoryStore


@dataclass(frozen=True, slots=True)
class AttachedAllocation:
    """Delegate attempt lifecycle to a store without dispatch or scheduler calls."""

    allocation: ConfiguredAllocation
    store: DirectoryStore

    def observe(self, state: str, at_utc: str) -> AllocationObservation:
        """Observe the attached allocation."""
        return self.allocation.observe(state, at_utc)

    def begin(self, group: ReadyGroup, task_id: str, started_at_utc: str = "") -> AttemptFence:
        """Begin one task attempt in the attached allocation."""
        return self.store.begin_attempt(group, task_id, self.allocation, started_at_utc)

    def finish(self, fence: AttemptFence, status: str, finished_at_utc: str, result_digest: str | None = None) -> OutcomeReceipt:
        """Finish one attached task attempt."""
        return self.store.finish_attempt(fence, status, finished_at_utc, result_digest)


__all__ = ["AttachedAllocation"]
