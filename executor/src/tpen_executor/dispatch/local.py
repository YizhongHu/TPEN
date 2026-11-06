"""Sequential local dispatch for one already-attached allocation."""

from __future__ import annotations

from ..programs import WorkerProgram
from .._records import frozen_record
from .contract import DispatcherContract


@frozen_record
class LocalDispatcher(DispatcherContract):
    """Run independent group members sequentially in one attached allocation.

    Parameters
    ----------
    worker : WorkerProgram
        Unresolved worker handle. It is materialized in the owner process for
        each member using the environment supplied to :meth:`run`.

    Notes
    -----
    Lifecycle, evidence, status, fencing, and accepted-segment rules are owned
    by :class:`~tpen_executor.dispatch.contract.DispatcherContract`. This
    backend only selects the ordinary owner-process subprocess path. It has no
    concurrency, retry, recovery, scheduler, Parsl, or Balsam behavior.
    """

    worker: WorkerProgram

    def __post_init__(self) -> None:
        if not callable(getattr(self.worker, "materialize", None)):
            raise TypeError("worker must be a materializable WorkerProgram handle")


__all__ = ["LocalDispatcher"]
