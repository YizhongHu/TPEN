"""Chain identity and the credited-work ledger.

This module owns the *identity* concept for a chain-resumed run, and nothing
else: which logical run a link belongs to, which execution attempt produced it,
which committed generation it continued from, and how much work has been
credited so far.

Three identities are deliberately distinct, because collapsing any two of them
is how a chain either double-counts work or forgets that a link was replayed:

``run_id``
    Stable across the whole chain. One logical scientific run.
``attempt_id``
    Fresh for every execution attempt, including a retry of an attempt that
    produced nothing. Provenance for "which process wrote this".
``parent_generation``
    The checkpoint step this attempt restored from, or ``None`` for a cold
    start. Explicit rather than inferred, so a replayed tail is attributable.

The ledger enforces the property the shared test contract calls G8: after an
interruption at iteration ``i`` whose last committed generation was ``g < i``,
the replay redoes ``g..i`` and the *credited* total must still be exactly the
fixed target ``T``. The interrupted tail must not be counted twice.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChainIdentity:
    """Identity of one execution attempt within one logical chain-resumed run.

    Parameters
    ----------
    run_id : str
        Stable logical run identifier, shared by every attempt in the chain.
    attempt_id : str
        Identifier unique to this execution attempt. Never reused, including
        by a retry that produced no committed generation.
    attempt_index : int
        Zero-based ordinal of this attempt within the chain. Monotone.
    parent_generation : int or None
        Committed checkpoint step this attempt restored from, or ``None`` for
        a cold start. Recorded explicitly rather than inferred from the
        checkpoint root, so a receipt states what was actually continued.
    """

    run_id: str
    attempt_id: str
    attempt_index: int
    parent_generation: int | None

    def __post_init__(self) -> None:
        # Validated explicitly: pytest runs with ``--typeguard-packages=tpen``,
        # so typeguard does not instrument this package and the annotations
        # above guard nothing at runtime. Same reasoning as
        # ``tpen.callback.cadence.SubscriptionGroup.__post_init__``.
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError(f"run_id must be a non-empty str, got {self.run_id!r}")
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError(f"attempt_id must be a non-empty str, got {self.attempt_id!r}")
        if not isinstance(self.attempt_index, int) or isinstance(self.attempt_index, bool):
            raise TypeError("attempt_index must be an integer")
        if self.attempt_index < 0:
            raise ValueError(f"attempt_index must be non-negative, got {self.attempt_index}")
        if self.parent_generation is not None:
            if not isinstance(self.parent_generation, int) or isinstance(
                self.parent_generation, bool
            ):
                raise TypeError("parent_generation must be an integer or None")
            if self.parent_generation < 0:
                raise ValueError(
                    f"parent_generation must be non-negative, got {self.parent_generation}"
                )

    @property
    def is_cold_start(self) -> bool:
        """Return whether this attempt began without a parent generation."""

        return self.parent_generation is None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-serializable mapping form."""

        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "attempt_index": self.attempt_index,
            "parent_generation": self.parent_generation,
        }

    @classmethod
    def from_mapping(cls, data: dict[str, object]) -> "ChainIdentity":
        """Rebuild a :class:`ChainIdentity` from :meth:`to_dict` output."""

        parent = data["parent_generation"]
        return cls(
            run_id=str(data["run_id"]),
            attempt_id=str(data["attempt_id"]),
            attempt_index=int(data["attempt_index"]),  # type: ignore[arg-type]
            parent_generation=None if parent is None else int(parent),  # type: ignore[arg-type]
        )


def new_attempt(
    run_id: str, attempt_index: int, parent_generation: int | None
) -> ChainIdentity:
    """Mint a fresh attempt identity for ``run_id``.

    The attempt id is a ``uuid4``, so two attempts cannot collide even when a
    controller restarts and re-derives the same ``attempt_index``.

    Parameters
    ----------
    run_id : str
        Stable logical run identifier.
    attempt_index : int
        Zero-based ordinal of this attempt within the chain.
    parent_generation : int or None
        Committed checkpoint step being continued, or ``None`` for a cold start.

    Returns
    -------
    ChainIdentity
        A fresh identity whose ``attempt_id`` has never been used before.
    """

    return ChainIdentity(
        run_id=run_id,
        attempt_id=str(uuid.uuid4()),
        attempt_index=attempt_index,
        parent_generation=parent_generation,
    )


class DoubleCreditError(ValueError):
    """A step was credited twice.

    Raised rather than silently ignored. A replayed tail that quietly
    re-credits its steps is precisely the G8 failure this ledger exists to
    detect, and a ledger that tolerated it would report a healthy total while
    the underlying accounting was wrong.
    """


@dataclass
class CreditedLedger:
    """Monotone record of which logical steps a chain has credited.

    A step is credited once, by whichever attempt actually committed it. An
    attempt that replays ``g..i`` after an interruption re-*executes* those
    steps but must not re-*credit* them, so :meth:`credit` refuses a duplicate
    instead of accepting it.

    Parameters
    ----------
    total_target : int
        Fixed total number of logical steps the whole chain must credit,
        independent of how many attempts it takes.
    """

    total_target: int
    _credited: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not isinstance(self.total_target, int) or isinstance(self.total_target, bool):
            raise TypeError("total_target must be an integer")
        if self.total_target < 0:
            raise ValueError(f"total_target must be non-negative, got {self.total_target}")

    def credit(self, step: int) -> None:
        """Credit one logical ``step``, refusing a duplicate.

        Parameters
        ----------
        step : int
            Zero-based logical step index.

        Raises
        ------
        DoubleCreditError
            If ``step`` has already been credited by this ledger.
        ValueError
            If ``step`` lies outside ``range(total_target)``.
        """

        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError("step must be an integer")
        if not 0 <= step < self.total_target:
            raise ValueError(
                f"step {step} outside the fixed target range [0, {self.total_target})"
            )
        if step in self._credited:
            raise DoubleCreditError(
                f"step {step} was already credited; a replayed tail must re-execute "
                "without re-crediting"
            )
        self._credited.add(step)

    def credit_all(self, steps: tuple[int, ...]) -> None:
        """Credit every step in ``steps``, refusing the first duplicate."""

        for step in steps:
            self.credit(step)

    @property
    def credited_steps(self) -> tuple[int, ...]:
        """Return the credited steps in ascending order."""

        return tuple(sorted(self._credited))

    @property
    def credited_count(self) -> int:
        """Return how many distinct logical steps have been credited."""

        return len(self._credited)

    @property
    def is_complete(self) -> bool:
        """Return whether exactly the whole fixed target has been credited."""

        return self.credited_steps == tuple(range(self.total_target))

    def missing_steps(self) -> tuple[int, ...]:
        """Return the steps in the target range that are still uncredited."""

        return tuple(step for step in range(self.total_target) if step not in self._credited)


__all__ = [
    "ChainIdentity",
    "CreditedLedger",
    "DoubleCreditError",
    "new_attempt",
]
