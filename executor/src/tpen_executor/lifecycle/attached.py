"""Lifecycle adapter for work inside an already configured allocation."""

from __future__ import annotations

from dataclasses import dataclass

from ..allocations import ConfiguredAllocation
from ..contracts import ReadyGroup
from ..results import AllocationObservation, OutcomeReceipt
from ..storage import AttemptFence, DirectoryStore, FenceRejected


@dataclass(frozen=True, slots=True)
class AttachedAllocation:
    """Delegate lifecycle to one configured allocation and receipt store.

    Parameters
    ----------
    allocation : ConfiguredAllocation
        Allocation identity that owns every delegated attempt.
    store : DirectoryStore
        Durable store used for fenced start and outcome receipts.

    Notes
    -----
    This adapter performs no dispatch, scheduler, Balsam, or Parsl work.
    Attempt fences are not transferable between attached allocations.
    """

    allocation: ConfiguredAllocation
    store: DirectoryStore

    def observe(self, state: str, at_utc: str) -> AllocationObservation:
        """Create a backend-neutral observation of the attached allocation.

        Parameters
        ----------
        state, at_utc : str
            Backend-neutral state and UTC observation timestamp.

        Returns
        -------
        AllocationObservation
            Observation bound to this adapter's allocation.
        """
        return self.allocation.observe(state, at_utc)

    def begin(self, group: ReadyGroup, task_id: str, started_at_utc: str = "") -> AttemptFence:
        """Begin one task attempt in the attached allocation.

        Parameters
        ----------
        group : ReadyGroup
            Group containing the task.
        task_id : str
            Task member to start.
        started_at_utc : str, optional
            UTC start timestamp; ``""`` selects the documented placeholder.

        Returns
        -------
        AttemptFence
            Fence bound to this allocation and store.
        """
        return self.store.begin_attempt(group, task_id, self.allocation, started_at_utc)

    def finish(self, fence: AttemptFence, status: str, finished_at_utc: str, result_digest: str | None = None) -> OutcomeReceipt:
        """Finish one task attempt owned by this attached allocation.

        Parameters
        ----------
        fence : AttemptFence
            Fence returned by :meth:`begin` for this allocation.
        status : str
            Closed operational outcome status.
        finished_at_utc : str
            UTC completion timestamp.
        result_digest : str or None, optional
            Optional produced-result digest.

        Returns
        -------
        OutcomeReceipt
            Newly published terminal receipt.

        Raises
        ------
        FenceRejected
            If a concrete fence belongs to another allocation or the store
            rejects its stale, crossed, or completed identity.
        """
        if isinstance(fence, AttemptFence) and fence.allocation_id != self.allocation.allocation_id.value:
            raise FenceRejected("fence belongs to another attached allocation")
        return self.store.finish_attempt(fence, status, finished_at_utc, result_digest)


__all__ = ["AttachedAllocation"]
