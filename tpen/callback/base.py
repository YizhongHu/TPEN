"""Base callback event and scheduling primitives."""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, ContextManager, Generic, Protocol, final, runtime_checkable

from tpen.artifacts import RunContext
from tpen.events import DomainState, Ended, Started, StateT
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence

from .cadence import CadenceGate, SubscriptionGroup, validate_subscription_groups

# Named for the same reason `tpen.status` and `tpen.bootstrap` are: a run's
# logging configuration can silence or route this channel on its own.
_LOGGER = logging.getLogger("tpen.callback")


@runtime_checkable
class CallbackContext(Protocol):
    """Minimal context contract shared by typed callback delivery hooks.

    Direct callback tests may use a structural stand-in, while production
    dispatch supplies :class:`tpen.artifacts.RunContext`. Keeping this contract
    narrow preserves runtime typeguard checks without requiring unit-only
    rendering tests to construct the full artifact manager.
    """

    metadata: Any

    def now_iso(self) -> str:
        """Return the current run timestamp in the configured timezone."""


class ManagedLifecycleError(RuntimeError):
    """Raised when a managed callback violates its core lifecycle contract."""


class ResourceOwnershipConflictError(ManagedLifecycleError):
    """Raised when two resource ownership claims cannot safely coexist."""


@runtime_checkable
class RunResource(Protocol):
    """Cooperative control handle returned by a managed callback scope.

    ``checkpoint`` advances due cooperative work at a safe core boundary;
    ``request_stop`` signals a nonblocking stop request so helpers can begin
    stopping before potentially blocking scope exits run.
    """

    def checkpoint(self) -> None:
        """Advance due cooperative work at a safe core boundary."""

    def request_stop(self) -> None:
        """Signal a nonblocking stop request before scope exit."""


class ManagedCallback(Protocol):
    """Optional lifecycle protocol implemented by resource-owning callbacks."""

    def run_scope(self, context: RunContext) -> ContextManager[RunResource | None]:
        """Return a context manager acquiring and releasing callback resources."""


_RESOURCE_OWNERSHIP_KINDS: frozenset[str] = frozenset(
    {"torch_profiler", "allocator_reset", "artifact_writer"}
)
_KNOWN_ACCELERATOR_BACKENDS: dict[str, str] = {
    "cuda": "cuda",
    "rocm": "rocm",
    "hip": "rocm",
    "xpu": "xpu",
    "mps": "mps",
}


@dataclass(frozen=True)
class ResourceOwnershipClaim:
    """Exclusive ownership claim declared for preflight conflict detection.

    Parameters
    ----------
    kind : {"torch_profiler", "allocator_reset", "artifact_writer"}
        Closed resource ownership category.
    scope : str
        Resource scope identifier. ``"torch_profiler"`` requires ``"process"``;
        ``"allocator_reset"`` carries a canonical device scope or ``"unknown"``;
        ``"artifact_writer"`` carries a destination path.
    """

    kind: str
    scope: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _RESOURCE_OWNERSHIP_KINDS:
            raise ValueError(
                f"unsupported ResourceOwnershipClaim kind {self.kind!r}; "
                f"expected one of {sorted(_RESOURCE_OWNERSHIP_KINDS)}"
            )
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("ResourceOwnershipClaim scope must be a nonempty string")
        if self.kind == "torch_profiler" and self.scope != "process":
            raise ValueError(
                "torch_profiler ownership claim requires scope == 'process', "
                f"got {self.scope!r}"
            )


def canonicalize_allocator_reset_scope(device: str) -> str | None:
    """Canonicalize a configured device string for allocator-reset conflict checks.

    Returns ``None`` for CPU targets (which have no accelerator allocator reset
    claim), ``"<backend>"`` for indexless accelerators, ``"<backend>:<index>"``
    for indexed accelerators, and ``"unknown"`` conservatively for any
    unrecognized, empty, or ambiguous device string.
    """

    if not isinstance(device, str):
        return "unknown"
    raw = device.strip().lower()
    if not raw:
        return "unknown"
    if raw == "cpu" or raw.startswith("cpu:"):
        return None
    parts = raw.split(":")
    if len(parts) == 1:
        backend = _KNOWN_ACCELERATOR_BACKENDS.get(parts[0])
        return "unknown" if backend is None else backend
    if len(parts) == 2:
        backend = _KNOWN_ACCELERATOR_BACKENDS.get(parts[0])
        index_text = parts[1]
        if backend is None or not index_text.isdigit():
            return "unknown"
        return f"{backend}:{int(index_text)}"
    return "unknown"


def claims_conflict(
    first: ResourceOwnershipClaim, second: ResourceOwnershipClaim
) -> bool:
    """Return whether two ownership claims cannot safely coexist.

    Conservative per the observability ownership contract: any two
    ``torch_profiler`` claims conflict process-wide; ``artifact_writer`` claims
    conflict when their canonicalized paths match; ``allocator_reset`` claims
    conflict whenever either scope is ``"unknown"`` or both name the same
    backend with at least one indexless scope or identical indices.
    """

    if not isinstance(first, ResourceOwnershipClaim) or not isinstance(
        second, ResourceOwnershipClaim
    ):
        raise TypeError("claims_conflict requires ResourceOwnershipClaim arguments")
    if first.kind != second.kind:
        return False
    if first.kind == "torch_profiler":
        return True
    if first.kind == "artifact_writer":
        first_path = os.path.normpath(str(Path(first.scope).resolve(strict=False)))
        second_path = os.path.normpath(str(Path(second.scope).resolve(strict=False)))
        return first_path == second_path
    # allocator_reset
    first_scope = canonicalize_allocator_reset_scope(first.scope)
    second_scope = canonicalize_allocator_reset_scope(second.scope)
    if first_scope is None or second_scope is None:
        return False
    if first_scope == "unknown" or second_scope == "unknown":
        return True
    first_parts = first_scope.split(":", 1)
    second_parts = second_scope.split(":", 1)
    if first_parts[0] != second_parts[0]:
        return False
    # Same backend: indexless overlaps every concrete index on that backend.
    if len(first_parts) == 1 or len(second_parts) == 1:
        return True
    return first_parts[1] == second_parts[1]


def _is_managed_callback_class(klass: object) -> bool:
    """Return whether ``klass`` is a ``_CallbackCore`` subclass declaring ``run_scope``."""

    if not isinstance(klass, type) or not issubclass(klass, _CallbackCore):
        return False
    scope_attr = getattr(klass, "run_scope", None)
    if scope_attr is None:
        return False
    if not callable(scope_attr) and not isinstance(scope_attr, property):
        raise TypeError(f"{klass.__name__}.run_scope must be callable")
    return True


def is_managed_callback(callback: object) -> bool:
    """Return whether ``callback`` implements the core-owned managed scope protocol.

    Recognizes ``run_scope`` only on ``_CallbackCore`` subclasses that declare
    ``run_scope`` on the class itself, never on unrelated objects or ad-hoc
    instance attributes.
    """

    if not isinstance(callback, _CallbackCore):
        return False
    if getattr(type(callback), "run_scope", None) is None:
        return False
    if not callable(getattr(callback, "run_scope", None)):
        raise TypeError(f"{type(callback).__name__}.run_scope must be callable")
    return True


def validate_run_resource(resource: object) -> None:
    """Validate the optional handle returned by ``ManagedCallback.run_scope``."""

    if resource is None:
        return
    if not callable(getattr(resource, "checkpoint", None)):
        raise TypeError(
            f"RunResource {type(resource).__name__} must expose callable checkpoint()"
        )
    if not callable(getattr(resource, "request_stop", None)):
        raise TypeError(
            f"RunResource {type(resource).__name__} must expose callable request_stop()"
        )


@dataclass
class _TypedGroupState:
    group: SubscriptionGroup
    gate: CadenceGate | None = field(init=False)
    open_pairs: dict[tuple[type[object], int], bool] = field(default_factory=dict)
    # Event shapes already reported as arriving with no state at all, so the
    # diagnostic in `StatefulCallback.handle_occurrence` fires once per shape
    # instead of once per training step.
    reported_missing_state: set[tuple[type[object], type[object] | None]] = field(
        default_factory=set
    )

    def __post_init__(self) -> None:
        self.gate = None if self.group.cadence is None else CadenceGate(self.group.cadence)

    def reset(self) -> None:
        if self.gate is not None:
            self.gate.reset()
        self.open_pairs.clear()
        self.reported_missing_state.clear()


_UNSET_CONTEXT = object()


class _CallbackCore:
    """Scheduling, subscription-plan, and context-reset machinery.

    This holds the typed subscription plan with its per-group cadence gates and
    the reset performed when the owning `RunContext` identity
    changes. It owns no delivery signature of its own, because that is exactly
    what distinguishes `Callback` from `StatefulCallback`.

    Parameters
    ----------
    every_n_steps : int or None, optional
        Optional periodic step filter.
    start_step : int, optional
        First eligible step for periodic callbacks.
    max_calls : int or None, optional
        Maximum number of callback invocations (counts actual executions).
    probability : float, optional
        Probability of running when otherwise scheduled. ``1.0`` always runs,
        ``0.0`` never runs. Applied after the trigger/``every_n_steps``/
        ``start_step`` checks.
    seed : int or None, optional
        Seed for the callback-local RNG used by `probability`. Using a local
        RNG keeps probabilistic scheduling reproducible without perturbing
        global PyTorch randomness.
    typed_groups : iterable of SubscriptionGroup, optional
        Typed selectors with independent occurrence-count cadence gates.
    """

    def __init__(
        self,
        every_n_steps: int | None = None,
        start_step: int = 0,
        max_calls: int | None = None,
        probability: float = 1.0,
        seed: int | None = None,
        *,
        typed_groups: Iterable[SubscriptionGroup] = (),
    ) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        self.every_n_steps = every_n_steps
        self.start_step = int(start_step)
        self.max_calls = max_calls
        self.probability = float(probability)
        self.seed = seed
        self._rng = random.Random(seed)
        self.num_calls = 0
        groups = tuple(typed_groups)
        if not all(isinstance(group, SubscriptionGroup) for group in groups):
            raise TypeError("typed_groups must contain only SubscriptionGroup values")
        validate_subscription_groups(groups)
        self._validate_typed_groups(groups)
        self._typed_group_states = tuple(_TypedGroupState(group) for group in groups)
        self._typed_context: object = _UNSET_CONTEXT

    def _validate_typed_groups(self, groups: tuple[SubscriptionGroup, ...]) -> None:
        """Reject a group plan this delivery shape cannot honour.

        Overridden by both public bases. It exists because ``stateless`` is a
        declaration about DELIVERY, and only the base that owns a delivery
        signature can say whether a given declaration is satisfiable there.
        """

        del groups

    def _draw_probability(self) -> bool:
        """Apply the probability gate using the callback-local RNG."""

        if self.probability >= 1.0:
            return True
        if self.probability <= 0.0:
            return False
        return self._rng.random() < self.probability

    def bind_typed_context(self, context: object) -> None:
        """Bind typed state to ``context`` once, resetting caches on change.

        Core invokes this before entering ``run_scope(context)`` so the first
        event delivered after readiness sees ``self._typed_context is context``
        and does not re-reset live activated resources, while reusing a cleanly
        closed callback with a fresh context resets before re-activation.
        """

        if self._typed_context is context:
            return
        self._typed_context = context
        for group_state in self._typed_group_states:
            group_state.reset()
        self._reset_typed_state()

    def _ensure_typed_context(self, context: object) -> None:
        if not self._typed_group_states or self._typed_context is context:
            return
        self.bind_typed_context(context)

    def _reset_typed_state(self) -> None:
        """Reset subclass caches when typed state moves to a new context."""

    @staticmethod
    def _typed_group_selects(
        group_state: _TypedGroupState, occurrence: Occurrence[TypedEvent]
    ) -> bool:
        """Return whether a selector names this occurrence, gating nothing.

        Kept separate from `_typed_group_delivers`, which advances the group's
        cadence gate and its open-pair table as a side effect of asking. A
        diagnostic that wants only "was this occurrence addressed to this
        group?" must not perturb the schedule by asking.
        """

        return any(
            selector.matches(occurrence.event) for selector in group_state.group.selectors
        )

    @staticmethod
    def _typed_group_delivers(
        group_state: _TypedGroupState, occurrence: Occurrence[TypedEvent]
    ) -> bool:
        event = occurrence.event
        if not isinstance(event, (Started, Ended)):
            if not any(
                selector.matches(event) for selector in group_state.group.selectors
            ):
                return False
            return group_state.gate is None or group_state.gate.should_run(occurrence.count)

        operation = event.operation
        if not any(
            selector.lifecycle is not None and isinstance(operation, selector.subject)
            for selector in group_state.group.selectors
        ):
            return False
        coordinate = (type(operation), occurrence.count)
        if isinstance(event, Started):
            decision = group_state.gate is None or group_state.gate.should_run(
                occurrence.count
            )
            # Cache rejected and maxed decisions too, so Ended never redraws.
            group_state.open_pairs[coordinate] = decision
        else:
            decision = group_state.open_pairs.pop(coordinate, False)
        return decision and any(
            selector.matches(event) for selector in group_state.group.selectors
        )


class Callback(_CallbackCore):
    """Callback that observes typed occurrences without any domain state.

    Delivery is two-argument: ``(occurrence, context)``. This is the base for
    every observer that needs only the moment and the run context -- timing,
    metadata, artifact indexing. It exists as a sibling of `StatefulCallback`
    rather than its parent so that a state-free observer never has to accept and
    ignore a state parameter.

    Constructor parameters are documented on `_CallbackCore`.
    """

    def _validate_typed_groups(self, groups: tuple[SubscriptionGroup, ...]) -> None:
        """Reject a ``stateless`` declaration, which is vacuous on this base.

        Every group here is already delivered without state, so the flag adds
        nothing -- but it would imply, to a reader, that some sibling group on
        the same class IS stateful, which this base cannot express. Refused at
        construction rather than ignored, so the disagreement between what the
        group declares and what the class can do surfaces as an error message
        instead of as a subtly wrong reading.
        """

        if any(group.stateless for group in groups):
            raise TypeError(
                f"{type(self).__name__} is a Callback, whose groups are all "
                "delivered without state; stateless=True is meaningful only on a "
                "StatefulCallback"
            )

    @final
    def handle_occurrence(
        self, occurrence: Occurrence[TypedEvent], context: Any
    ) -> None:
        """Match, gate, and deliver one typed occurrence in group order.

        ``context`` is deliberately widened at this final boundary: the
        runtime context identity is validated before delivery, while concrete
        hooks retain the precise ``RunContext`` contract.
        """

        self._ensure_typed_context(context)
        for group_state in self._typed_group_states:
            if self._typed_group_delivers(group_state, occurrence):
                self.handle_occurrence_impl(occurrence, context)

    def handle_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: RunContext
    ) -> None:
        """Handle one occurrence admitted by a configured typed group."""

        del occurrence, context


class StatefulCallback(_CallbackCore, Generic[StateT]):
    """Callback that observes typed occurrences together with domain state.

    Delivery is decided PER GROUP, not per callback. A group that did not
    declare `tpen.callback.cadence.SubscriptionGroup.stateless` receives the
    emitting domain's own `tpen.events.DomainState` through the three-argument
    ``handle_occurrence_impl``, and only when that state is an instance of this
    callback's ``state_type``; a callback whose domain does not match is
    skipped, not failed, because one run may emit several domains' states and a
    callback simply has nothing to observe outside its own. A group that DID
    declare ``stateless`` receives the two-argument
    ``handle_stateless_occurrence_impl`` instead, and no state filter applies to
    it at all.

    That is a deliberate weakening, recorded as an amendment to ADR-E008. The
    invariant used to be "my handler always receives my domain's state"; it is
    now "each group receives state or not, as declared". It buys the one thing
    the old invariant made inexpressible: a callback that observes both its own
    domain and a boundary belonging to no domain, such as the run lifecycle,
    which carries no state and never will. The alternative was splitting such a
    class in two and changing its config-facing ``_target_``.

    Because `tpen.callback.cadence.validate_subscription_groups` rejects
    overlapping deliveries, at most one group matches any occurrence, so the
    two hooks can never both fire for a single delivery.

    This is NOT a subclass of `Callback`. The two delivery arities differ, so an
    inheritance edge would be a substitutability violation, and it would also
    make the dispatcher's ``isinstance`` discrimination meaningless.

    Constructor parameters are documented on `_CallbackCore`.

    Attributes
    ----------
    state_type : type of DomainState
        The concrete domain state class this callback observes. Every concrete
        subclass must declare it, and `__init_subclass__` rejects one that does
        not. It is the RUNTIME authority for delivery, and it is not redundant
        with the ``StatefulCallback[...]`` type argument: Python does not retain
        a subclass's type argument at runtime, so the type argument is a purely
        static claim and cannot route anything. Declaring the class explicitly
        also means renaming a state class cannot silently redirect delivery,
        the same reason `tpen.training.events.TrainingPhase` makes concrete
        phases declare ``phase_name``.
    """

    # ClassVar: shared per concrete callback class, never per instance, and
    # deliberately left unset here so this base is abstract.
    state_type: ClassVar[type[DomainState]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Reject a subclass that declares no observable domain state."""

        super().__init_subclass__(**kwargs)
        # A stateful callback with no declared domain would silently receive
        # nothing forever, so refuse the class rather than the delivery.
        if not hasattr(cls, "state_type"):
            raise TypeError(
                f"{cls.__name__} must declare a state_type ClassVar; "
                "StatefulCallback itself is abstract"
            )
        state_type = cls.state_type
        if not isinstance(state_type, type) or not issubclass(state_type, DomainState):
            raise TypeError(
                f"{cls.__name__}.state_type must be a DomainState subclass, got "
                f"{state_type!r}"
            )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Reject an instance of the abstract base and forward everything else.

        `__init_subclass__` already covers every subclass, so this covers only
        the undeclared base itself, which would otherwise be accepted here and
        fail with a bare ``AttributeError`` inside dispatch, mid-run. Same
        reasoning as `tpen.training.events.TrainingPhase`, which rejects an
        undeclared phase at construction rather than at metric-key lookup.
        Arguments are forwarded unchanged; `_CallbackCore` documents them.
        """

        if not hasattr(type(self), "state_type"):
            raise TypeError(
                "StatefulCallback is abstract; declare state_type on a subclass"
            )
        super().__init__(*args, **kwargs)

    def _validate_typed_groups(self, groups: tuple[SubscriptionGroup, ...]) -> None:
        """Reject a stateless declaration this class could only swallow.

        Two shapes are refused, both of which would otherwise reintroduce the
        silence this capability exists to remove:

        - a stateless group on a class that never overrode
          ``handle_stateless_occurrence_impl``, whose deliveries would land in
          the inherited no-op and disappear;
        - a plan whose groups are ALL stateless ON A CLASS THAT ALSO NEVER
          OVERRODE ``handle_occurrence_impl``, which declares a ``state_type``
          that can never route anything and should be a `Callback` instead.

        THE SECOND CHECK IS NARROWER THAN IT FIRST LANDED, and the qualifier is
        load-bearing rather than defensive. A group plan is per-INSTANCE while
        ``state_type`` is per-CLASS, and `tpen.callback.Status` is the first
        class whose plan varies by constructor argument: with ``train_lines``
        off it declares only the stateless run-lifecycle group, yet the same
        class with ``train_lines`` on routes `tpen.training.state.TrainerState`
        through ``handle_occurrence_impl``. "Make it a `Callback` instead" is
        false advice there, and refusing the instance would have made the
        shipped default of that callback unconstructible.

        Asking whether the CLASS overrode ``handle_occurrence_impl`` asks the
        question the check was written for -- can this ``state_type`` ever route
        a delivery? -- at the granularity where the answer lives. A class that
        never overrode it cannot consume its domain's state under any
        configuration, which is exactly the wiring error being caught.

        A class with no groups at all is untouched, which is a third claim
        again: it subscribes nothing rather than subscribing state-free.
        """

        stateless = tuple(group for group in groups if group.stateless)
        if not stateless:
            return
        if len(stateless) == len(groups) and (
            type(self).handle_occurrence_impl is StatefulCallback.handle_occurrence_impl
        ):
            raise TypeError(
                f"every subscription group on {type(self).__name__} is stateless, so "
                "its state_type can never route a delivery; make it a Callback"
            )
        if (
            type(self).handle_stateless_occurrence_impl
            is StatefulCallback.handle_stateless_occurrence_impl
        ):
            raise TypeError(
                f"{type(self).__name__} declares a stateless subscription group but "
                "does not override handle_stateless_occurrence_impl, so those "
                "deliveries would be silently discarded"
            )

    @final
    def handle_occurrence(
        self,
        occurrence: Occurrence[TypedEvent],
        context: CallbackContext,
        state: DomainState | None,
    ) -> None:
        """Match, gate, and route one typed occurrence group by group.

        ``state`` is whatever the emitting domain passed -- this callback's own
        domain state, some other domain's, or nothing at all. The discrimination
        happens HERE rather than in `tpen.artifacts.RunContext`, because the
        dispatcher cannot know which group a given occurrence will match and the
        answer differs between them. The narrow ``StateT`` annotation therefore
        lives on ``handle_occurrence_impl``, which is reached only after the
        instance check below.

        The state check runs BEFORE `_typed_group_delivers` for a stateful
        group, which is not incidental: asking that question advances the
        group's cadence gate, so checking it first is what keeps a foreign
        domain's occurrences from consuming schedule the way they never did when
        the dispatcher filtered whole callbacks.
        """

        self._ensure_typed_context(context)
        for index, group_state in enumerate(self._typed_group_states):
            if group_state.group.stateless:
                if self._typed_group_delivers(group_state, occurrence):
                    self.handle_stateless_occurrence_impl(occurrence, context)
                continue
            if not isinstance(state, self.state_type):
                self._report_missing_state(index, group_state, occurrence, state)
                continue
            if self._typed_group_delivers(group_state, occurrence):
                self.handle_occurrence_impl(occurrence, context, state)

    def _report_missing_state(
        self,
        index: int,
        group_state: _TypedGroupState,
        occurrence: Occurrence[TypedEvent],
        state: DomainState | None,
    ) -> None:
        """Log the one skip shape that is a wiring error rather than routine.

        The two skips are told apart by whether ANY state arrived, and the
        distinction is not cosmetic:

        - some other domain's state arrived. Routine, and silent. A run emits
          several domains' boundaries and this callback is not their audience;
          it happens on most occurrences of a mixed run, so a diagnostic here
          would be pure noise.
        - nothing arrived, on a boundary this group actually selected. Nobody
          could have been the audience, so either the emitter omitted its
          ``state=`` or the group should have declared ``stateless=True``. That
          is a mistake every time, and it is precisely the silence that trapped
          `tpen.callback.Status` and `tpen.callback.ArtifactIndex`.

        Reported at WARNING rather than raised, because raising would break the
        pinned behaviour that a boundary emitted without state delivers nothing
        rather than killing the run -- a callback misconfiguration must not take
        down training. Reported once per event shape per group, because the
        selected boundary may be a per-step one.
        """

        if state is not None or not self._typed_group_selects(group_state, occurrence):
            return
        event = occurrence.event
        # Typed discrimination rather than attribute probing: a lifecycle event
        # is keyed by its operation type too, so Started[A] and Started[B] are
        # reported separately.
        if isinstance(event, (Started, Ended)):
            key: tuple[type[object], type[object] | None] = (
                type(event),
                type(event.operation),
            )
            shape = f"{type(event).__name__}[{type(event.operation).__name__}]"
        else:
            key = (type(event), None)
            shape = type(event).__name__
        if key in group_state.reported_missing_state:
            return
        group_state.reported_missing_state.add(key)
        _LOGGER.warning(
            "%s subscription group %d requires %s, but %s was emitted with no domain "
            "state at all, so nothing was delivered. Either the emitter omitted "
            "state=, or the group should declare stateless=True. Reported once per "
            "event shape per group.",
            type(self).__name__,
            index,
            self.state_type.__name__,
            shape,
        )

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: CallbackContext,
        state: StateT,
    ) -> None:
        """Handle one occurrence admitted by a configured typed group."""

        del occurrence, context, state

    def handle_stateless_occurrence_impl(
        self, occurrence: Occurrence[TypedEvent], context: CallbackContext
    ) -> None:
        """Handle one occurrence admitted by a group that declared no state.

        Deliberately NOT named ``handle_occurrence_impl``: the two hooks differ
        in arity, and one name carrying two signatures on one class would make
        it impossible to tell, at an override site, which route the author meant
        -- and a wrong guess would be silent. `_validate_typed_groups` refuses a
        stateless group unless this is overridden, so the inherited no-op is
        reachable only by a class that declared no stateless group.
        """

        del occurrence, context


__all__ = [
    "Callback",
    "CallbackContext",
    "ManagedCallback",
    "ManagedLifecycleError",
    "ResourceOwnershipClaim",
    "ResourceOwnershipConflictError",
    "RunResource",
    "StatefulCallback",
    "canonicalize_allocator_reset_scope",
    "claims_conflict",
    "is_managed_callback",
    "validate_run_resource",
]
