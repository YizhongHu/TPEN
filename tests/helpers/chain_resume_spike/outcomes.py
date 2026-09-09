"""Typed continuation outcomes for one chain-resume execution attempt.

The shared test contract (gate G1) requires that a clean proactive yield be
distinguishable from both scientific completion and failure retry. Four kinds
are therefore closed into one enum, and two policy questions are answered from
the kind alone rather than from a string comparison at each call site:

* Does this outcome mean the science finished?  Only ``FINAL``.
* Does this outcome spend the bounded transient-retry budget?  Only
  ``TRANSIENT``.

A ``YIELD`` spends no retry budget. That is the whole point of the distinction:
ordinary walltime continuation must not exhaust the quota reserved for genuine
transient failures, or a long chain eliminates itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class OutcomeKind(Enum):
    """Closed set of ways one execution attempt can end.

    ``YIELD``
        The attempt stopped itself at a coherent iteration boundary with a
        valid committed generation, before reaching the total target. Neither
        scientific success nor failure. Continuation is expected and automatic.
    ``FINAL``
        The attempt reached the fixed total target. The chain is done.
    ``FATAL``
        A configuration, checkpoint-semantic or deliberate scientific error.
        Continuation must stop promptly; retrying cannot help.
    ``TRANSIENT``
        An execution failure that a bounded retry may clear.
    """

    YIELD = "yield"
    FINAL = "final"
    FATAL = "fatal"
    TRANSIENT = "transient"


#: Kinds after which the chain is expected to launch another attempt.
CONTINUABLE_KINDS = frozenset({OutcomeKind.YIELD, OutcomeKind.TRANSIENT})

#: Kinds after which the chain stops. Disjoint from :data:`CONTINUABLE_KINDS`,
#: and the union of the two is the whole enum -- asserted by a lane test, so a
#: newly added kind cannot silently fall through both classifications.
TERMINAL_KINDS = frozenset({OutcomeKind.FINAL, OutcomeKind.FATAL})


@dataclass(frozen=True)
class AttemptOutcome:
    """Typed result of one execution attempt.

    Parameters
    ----------
    kind : OutcomeKind
        Which of the four closed outcomes occurred.
    reason : str
        Short machine-stable reason token, e.g. ``"yield_budget_reached"`` or
        ``"fault_injected"``. Free text belongs in ``detail``.
    credited_through : int or None
        Highest logical step this attempt credited, or ``None`` if it credited
        none. Distinct from the checkpoint generation: an attempt can credit
        steps whose generation was never committed, and those steps are
        replayed rather than credited by the next attempt.
    next_parent_generation : int or None
        Committed generation the next attempt should restore from, or ``None``
        when no generation is committed yet.
    detail : str, optional
        Human-readable diagnostic. Never parsed by policy code.
    """

    kind: OutcomeKind
    reason: str
    credited_through: int | None
    next_parent_generation: int | None
    detail: str = ""

    def __post_init__(self) -> None:
        # Explicit because typeguard is scoped to the ``tpen`` package only.
        if not isinstance(self.kind, OutcomeKind):
            raise TypeError(f"kind must be OutcomeKind, got {type(self.kind).__name__}")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError(f"reason must be a non-empty str, got {self.reason!r}")

    @property
    def is_scientific_success(self) -> bool:
        """Return whether the science actually finished.

        A ``YIELD`` is explicitly not a success: reporting it as one is how a
        chain silently stops short of its total target while looking green.
        """

        return self.kind is OutcomeKind.FINAL

    @property
    def consumes_retry_budget(self) -> bool:
        """Return whether this outcome spends the bounded transient-retry quota.

        Only ``TRANSIENT`` does. Ordinary walltime continuation (``YIELD``)
        must not, or a chain long enough to need many links would eliminate
        itself on its own success path.
        """

        return self.kind is OutcomeKind.TRANSIENT

    @property
    def is_continuable(self) -> bool:
        """Return whether the chain should launch another attempt."""

        return self.kind in CONTINUABLE_KINDS

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable mapping form."""

        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "credited_through": self.credited_through,
            "next_parent_generation": self.next_parent_generation,
            "detail": self.detail,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AttemptOutcome":
        """Rebuild an :class:`AttemptOutcome` from :meth:`to_dict` output.

        The kind is looked up against the closed enum by value, never via
        ``getattr`` or any other reflective dispatch -- the same discipline
        :mod:`tests.helpers.ddp_fault_injection` applies to its own enums.
        """

        credited = data["credited_through"]
        parent = data["next_parent_generation"]
        return cls(
            kind=OutcomeKind(data["kind"]),
            reason=str(data["reason"]),
            credited_through=None if credited is None else int(credited),
            next_parent_generation=None if parent is None else int(parent),
            detail=str(data.get("detail", "")),
        )


def write_outcome(path: Path, outcome: AttemptOutcome) -> None:
    """Write ``outcome`` to ``path`` as JSON, durably.

    Flushed and ``fsync``-ed before the handle closes. A fault arm that ends in
    ``SIGKILL`` otherwise loses the buffered tail, which turns a recorded
    outcome into indistinguishable silence.
    """

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(outcome.to_dict(), handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_outcome(path: Path) -> AttemptOutcome:
    """Read an :func:`write_outcome` payload back."""

    return AttemptOutcome.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "CONTINUABLE_KINDS",
    "TERMINAL_KINDS",
    "AttemptOutcome",
    "OutcomeKind",
    "read_outcome",
    "write_outcome",
]
