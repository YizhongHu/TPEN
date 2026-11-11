"""Append-only JSONL sidecar of trajectory-statistics receipts.

The sidecar is the durable side of the producer/consumer boundary. It is
append-only and rejects a second receipt under an identity it already holds:
statistics that silently change under a fixed join key would let two consumers
read the same run and disagree. Correcting a published number requires a new
identity -- a bumped ``evaluator_id``, or a genuinely different attempt -- which
is exactly the audit trail a changed estimate should leave.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from tpen.durable_append import append_record
from tpen.statistics.receipt import (
    TrajectoryStatisticsIdentity,
    TrajectoryStatisticsReceipt,
)

__all__ = ["DuplicateReceiptError", "TrajectoryStatisticsSidecar"]


class DuplicateReceiptError(RuntimeError):
    """Raised when an identity already present in the sidecar is appended again."""


class TrajectoryStatisticsSidecar:
    """Append-only JSONL file of :class:`TrajectoryStatisticsReceipt` records.

    Parameters
    ----------
    path : pathlib.Path or str
        Location of the ``.jsonl`` sidecar. The file and its parent directory
        are created on first append.

    Attributes
    ----------
    path : pathlib.Path
        Resolved sidecar location.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self.path)!r})"

    def read(self) -> tuple[TrajectoryStatisticsReceipt, ...]:
        """Return every receipt in the sidecar, in file order.

        Returns
        -------
        tuple of TrajectoryStatisticsReceipt
            Empty when the sidecar does not exist yet.

        Raises
        ------
        ValueError
            If a line is not valid JSON or is not a valid receipt. A corrupt
            sidecar fails loudly rather than silently returning a short list.
        """

        if not self.path.exists():
            return ()
        receipts: list[TrajectoryStatisticsReceipt] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{self.path}:{line_number}: invalid JSON: {error}") from error
                try:
                    receipts.append(TrajectoryStatisticsReceipt.from_dict(record))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"{self.path}:{line_number}: invalid receipt: {error}") from error
        return tuple(receipts)

    def identities(self) -> tuple[tuple[str, ...], ...]:
        """Return the identity key of every receipt currently stored."""
        return tuple(receipt.identity.as_key() for receipt in self.read())

    def get(
        self, identity: TrajectoryStatisticsIdentity
    ) -> TrajectoryStatisticsReceipt | None:
        """Return the receipt stored under `identity`, or ``None``.

        Parameters
        ----------
        identity : TrajectoryStatisticsIdentity
            Complete join key. Partial keys are not supported: a join on fewer
            than seven fields can silently match the wrong observable or the
            wrong attempt.

        Returns
        -------
        TrajectoryStatisticsReceipt or None
            The matching receipt, or ``None`` when the identity is absent.
        """

        wanted = identity.as_key()
        for receipt in self.read():
            if receipt.identity.as_key() == wanted:
                return receipt
        return None

    def append(self, receipt: TrajectoryStatisticsReceipt) -> None:
        """Append one receipt, refusing to overwrite an existing identity.

        Parameters
        ----------
        receipt : TrajectoryStatisticsReceipt
            The receipt to persist.

        Raises
        ------
        DuplicateReceiptError
            If the sidecar already holds this identity.
        """

        self.extend((receipt,))

    def extend(self, receipts: Iterable[TrajectoryStatisticsReceipt]) -> None:
        """Append several receipts atomically with respect to duplicates.

        Every identity is checked -- against the file and against the rest of
        the batch -- before anything is written, so a duplicate inside the batch
        cannot leave the sidecar half-updated.

        Each receipt is written by its own
        :func:`tpen.durable_append.append_record` call, so every row gets the
        single-write, torn-tail and short-write guarantees rather than only the
        batch as a whole.  That matters because this sidecar is structurally the
        same object as the checkpoint publication catalog -- append-only JSONL,
        :meth:`read` raises on a malformed row, and :meth:`extend` reads the
        file before it writes -- so ONE torn row would block every later append,
        not merely lose itself.  A batch of N costs N opens; the production path
        is :meth:`append`, where N is 1.

        Parameters
        ----------
        receipts : iterable of TrajectoryStatisticsReceipt
            Receipts to persist, in order.

        Raises
        ------
        DuplicateReceiptError
            If any identity is already stored or repeats within the batch.
        """

        pending = tuple(receipts)
        if not pending:
            return

        seen = set(self.identities())
        for receipt in pending:
            key = receipt.identity.as_key()
            if key in seen:
                raise DuplicateReceiptError(
                    "receipt identity already present; the sidecar is immutable. "
                    f"Bump evaluator_id to publish a revised estimate. Identity: {key}"
                )
            seen.add(key)

        for receipt in pending:
            append_record(self.path, json.dumps(receipt.to_dict(), sort_keys=True))

    def __iter__(self) -> Iterator[TrajectoryStatisticsReceipt]:
        return iter(self.read())

    def __len__(self) -> int:
        return len(self.read())
