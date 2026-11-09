"""Cadence primitives for typed callback subscriptions.

Two coordinates are available, and choosing between them is a correctness
decision rather than a style one. `Cadence` counts `tpen.events.Occurrence`
values, which are run-local. `StepCadence` counts a domain's own durable step,
which survives a checkpoint restore.
"""

from __future__ import annotations

import random
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from tpen.events import Subscription


@dataclass(frozen=True)
class Cadence:
    """Occurrence-count schedule for one subscription group.

    Parameters
    ----------
    every_n : int, optional
        Interval between eligible one-based occurrence counts.
    start : int, optional
        First eligible one-based occurrence count.
    max_calls : int or None, optional
        Maximum admitted logical coordinates across the whole group.
    probability : float, optional
        Admission probability after count-window filtering.
    seed : int or None, optional
        Seed for the group-local probability stream.
    """

    every_n: int = 1
    start: int = 1
    max_calls: int | None = None
    probability: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.every_n, int) or isinstance(self.every_n, bool):
            raise TypeError("every_n must be an integer")
        if self.every_n < 1:
            raise ValueError(f"every_n must be at least 1, got {self.every_n}")
        if not isinstance(self.start, int) or isinstance(self.start, bool):
            raise TypeError("start must be an integer")
        if self.start < 1:
            raise ValueError(f"start must be at least 1, got {self.start}")
        if self.max_calls is not None:
            if not isinstance(self.max_calls, int) or isinstance(self.max_calls, bool):
                raise TypeError("max_calls must be an integer or None")
            if self.max_calls < 0:
                raise ValueError(f"max_calls must be non-negative, got {self.max_calls}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")


class CadenceGate:
    """Mutable counter and RNG stream for one immutable `Cadence`.

    Parameters
    ----------
    cadence : Cadence
        Immutable scheduling values owned by this gate.
    """

    def __init__(self, cadence: Cadence) -> None:
        if not isinstance(cadence, Cadence):
            raise TypeError(f"cadence must be Cadence, got {type(cadence).__name__}")
        self.cadence = cadence
        self.num_calls = 0
        self._rng = random.Random(cadence.seed)

    def should_run(self, count: int) -> bool:
        """Return whether one new logical coordinate is admitted."""

        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError(f"count must be a one-based integer, got {count!r}")
        cadence = self.cadence
        if cadence.max_calls is not None and self.num_calls >= cadence.max_calls:
            return False
        if count < cadence.start or (count - cadence.start) % cadence.every_n != 0:
            return False
        if cadence.probability <= 0.0:
            return False
        if cadence.probability < 1.0 and self._rng.random() >= cadence.probability:
            return False
        self.num_calls += 1
        return True

    def reset(self) -> None:
        """Reset the counter and recreate the RNG from the original seed."""

        self.num_calls = 0
        self._rng = random.Random(self.cadence.seed)


@dataclass(frozen=True)
class StepCadence:
    """Schedule keyed on a domain's DURABLE step coordinate.

    This is the sibling of `Cadence`, and the difference between them is the
    coordinate they count, not the arithmetic they apply.

    `Cadence` gates on `tpen.events.Occurrence.count`, which is run-local: it
    restarts at 1 after a checkpoint restore, so a resumed run gated on it fires
    on a different schedule than an uninterrupted one. `StepCadence` gates on
    the durable step the emitting domain carries on its own typed operation, so
    the two runs agree. `tpen.callback.Checkpoint` avoids the same trap on the
    legacy path by gating on the durable ``completed_updates`` counter.

    The window reproduces the legacy string path's (`_CallbackCore.should_run`)
    exactly, so migrating a callback off that path cannot move which steps it
    fires on: a callback is admitted when ``step >= start`` and
    ``(step - start) % every_n == 0``, subject to ``max_calls`` and a
    ``probability`` draw applied in that order.

    Parameters
    ----------
    every_n : int, optional
        Interval between eligible durable steps.
    start : int, optional
        First eligible durable step. Zero-based, because trainer steps are.
        This is the one place `StepCadence` differs from `Cadence`, whose
        one-based occurrence counts start at 1.
    max_calls : int or None, optional
        Maximum admitted steps across the whole run.
    probability : float, optional
        Admission probability after step-window filtering.
    seed : int or None, optional
        Seed for the callback-local probability stream. Kept separate from any
        seed a callback's injected collaborators use, so scheduling randomness
        never perturbs their streams.
    """

    every_n: int = 1
    start: int = 0
    max_calls: int | None = None
    probability: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.every_n, int) or isinstance(self.every_n, bool):
            raise TypeError("every_n must be an integer")
        if self.every_n < 1:
            raise ValueError(f"every_n must be at least 1, got {self.every_n}")
        if not isinstance(self.start, int) or isinstance(self.start, bool):
            raise TypeError("start must be an integer")
        if self.max_calls is not None:
            if not isinstance(self.max_calls, int) or isinstance(self.max_calls, bool):
                raise TypeError("max_calls must be an integer or None")
            if self.max_calls < 0:
                raise ValueError(f"max_calls must be non-negative, got {self.max_calls}")
        # The legacy path validated this in `_CallbackCore.__init__`. A migrated
        # callback routes `probability` here instead, so the check has to move
        # with it or an out-of-range value would silently stop being rejected.
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")


class StepCadenceGate:
    """Mutable counter and RNG stream for one immutable `StepCadence`.

    Deliberately has no ``reset``, unlike `CadenceGate`. The legacy scalar path
    this replaces never reset its ``num_calls`` or its RNG when the owning
    `tpen.artifacts.RunContext` identity changed, and a migration that added a
    reset would change when a reused callback instance fires. Reusing one
    instance across two contexts happens in tests, not in production, where
    Hydra builds a fresh callback per run.

    Parameters
    ----------
    cadence : StepCadence
        Immutable scheduling values owned by this gate.
    """

    def __init__(self, cadence: StepCadence) -> None:
        if not isinstance(cadence, StepCadence):
            raise TypeError(f"cadence must be StepCadence, got {type(cadence).__name__}")
        self.cadence = cadence
        self.num_calls = 0
        self._rng = random.Random(cadence.seed)

    def should_run(self, step: int) -> bool:
        """Return whether the domain's durable ``step`` is admitted.

        Parameters
        ----------
        step : int
            Durable zero-based step read from the emitting domain's typed
            operation -- never from a mutable state object, whose value fields
            are stale at any boundary above their assignment.
        """

        cadence = self.cadence
        if cadence.max_calls is not None and self.num_calls >= cadence.max_calls:
            return False
        if step < cadence.start or (step - cadence.start) % cadence.every_n != 0:
            return False
        if not self._draw():
            return False
        # Counted only on admission, matching the legacy path, where
        # ``num_calls`` advanced in `handle` and only for a delivered event.
        self.num_calls += 1
        return True

    def _draw(self) -> bool:
        """Apply the probability gate using the callback-local RNG."""

        probability = self.cadence.probability
        # The RNG is consumed only for a genuinely probabilistic schedule, so a
        # fully-scheduled callback draws nothing and its stream stays untouched.
        if probability >= 1.0:
            return True
        if probability <= 0.0:
            return False
        return self._rng.random() < probability


def pop_step_cadence(kwargs: MutableMapping[str, Any]) -> StepCadence:
    """Consume the five legacy scheduling scalars from ``kwargs``.

    A callback migrated off the legacy string path keeps the option names its
    configs already spell -- ``every_n_steps``, ``start_step``, ``max_calls``,
    ``probability``, ``seed`` -- but routes them to a durable step window
    instead of to `_CallbackCore`'s trigger path. Exactly those five named keys
    are removed; nothing else in ``kwargs`` is read.

    Parameters
    ----------
    kwargs : MutableMapping
        Constructor keywords, mutated in place.

    Returns
    -------
    StepCadence
        Durable-step schedule equivalent to the legacy scalar window.
    """

    every_n_steps = kwargs.pop("every_n_steps", None)
    max_calls = kwargs.pop("max_calls", None)
    start_step = int(kwargs.pop("start_step", 0))
    return StepCadence(
        # ``None`` meant "no window" on the legacy path. Trainer steps are
        # non-negative, so a window of every step from 0 admits the same set.
        every_n=1 if every_n_steps is None else int(every_n_steps),
        # The legacy path guarded its WHOLE window, ``start_step`` included,
        # behind ``if every_n_steps is not None``, so a ``start_step`` set
        # without an ``every_n_steps`` was silently inert. That is preserved
        # rather than repaired: this migration must not move which steps a
        # callback fires on, and no shipped config sets one without the other.
        # Dropping this line is the whole fix, if that quirk is ever ruled a bug.
        start=0 if every_n_steps is None else start_step,
        max_calls=None if max_calls is None else int(max_calls),
        probability=float(kwargs.pop("probability", 1.0)),
        seed=kwargs.pop("seed", None),
    )


@dataclass(frozen=True)
class SubscriptionGroup:
    """Selectors sharing one optional occurrence cadence decision.

    The group, not the callback class, is the unit that decides whether a
    delivery carries domain state. That follows the cadence precedent directly:
    a cadence gate has always been group-local, so a callback already answers
    "when do I fire?" once per group rather than once per class, and ``stateless``
    makes "what do I receive?" answer at the same granularity.

    Parameters
    ----------
    selectors : tuple of Subscription
        Typed deliveries sharing one logical decision.
    cadence : Cadence or None, optional
        Group-local schedule. ``None`` observes and delivers without a gate.
    stateless : bool, optional
        Declare that this group observes boundaries carrying NO domain state.
        Meaningful only on a `tpen.callback.StatefulCallback`, where it routes
        the group's deliveries to the two-argument
        ``handle_stateless_occurrence_impl`` hook and exempts them from the
        ``state_type`` filter. `tpen.callback.Callback` rejects it, because
        every group on a state-free observer is already delivered state-free
        and the flag there could only mislead a reader into believing some
        sibling group is not.

        Defaults to ``False``, which is the pre-existing behaviour for every
        already-declared group: a `StatefulCallback` group still receives its
        domain's state, and this change adds a capability rather than moving
        any existing delivery.
    """

    selectors: tuple[Subscription, ...]
    cadence: Cadence | None = None
    stateless: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.selectors, tuple):
            raise TypeError("selectors must be a tuple of Subscription values")
        if not self.selectors:
            raise ValueError("a subscription group requires at least one selector")
        if not all(isinstance(selector, Subscription) for selector in self.selectors):
            raise TypeError("selectors must contain only Subscription values")
        if self.cadence is not None and not isinstance(self.cadence, Cadence):
            raise TypeError("cadence must be Cadence or None")
        # Checked explicitly because typeguard instruments parameters and
        # returns, never dataclass FIELDS, so the annotation above guards
        # nothing at runtime. A truthy non-bool would otherwise select the
        # stateless route silently, which is the failure shape this whole
        # change exists to remove.
        if not isinstance(self.stateless, bool):
            raise TypeError("stateless must be a bool")


def validate_subscription_groups(groups: tuple[SubscriptionGroup, ...]) -> None:
    """Reject selectors whose deliveries overlap across cadence groups.

    This is load-bearing beyond cadence. Because no two groups can select the
    same occurrence, at most ONE group matches any occurrence, which is what
    makes it structurally impossible for `tpen.callback.StatefulCallback` to
    call both its stateful and its stateless hook for a single delivery. The
    check is deliberately by ``issubclass`` rather than by identity, so a
    subclass selector in one group and its base in another is rejected too; a
    ``stateless`` group is not exempt from any of it.
    """

    for first_index, first in enumerate(groups):
        for second_index in range(first_index + 1, len(groups)):
            second = groups[second_index]
            for first_selector in first.selectors:
                for second_selector in second.selectors:
                    if _subscriptions_overlap(first_selector, second_selector):
                        raise ValueError(
                            "typed subscription groups have overlapping deliveries: "
                            f"groups {first_index} and {second_index}"
                        )


def _subscriptions_overlap(first: Subscription, second: Subscription) -> bool:
    """Return whether two selectors can deliver the same occurrence."""

    if first.lifecycle is not second.lifecycle:
        return False
    return issubclass(first.subject, second.subject) or issubclass(
        second.subject, first.subject
    )


__all__ = [
    "Cadence",
    "CadenceGate",
    "StepCadence",
    "StepCadenceGate",
    "SubscriptionGroup",
    "pop_step_cadence",
    "validate_subscription_groups",
]
