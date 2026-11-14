"""Runtime equivariance checking callback."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

from tpen.artifacts import ArtifactRoutingError, RunContext, write_json
from tpen.events import DomainState
from tpen.events import Event as TypedEvent
from tpen.events import Occurrence, Subscription
from tpen.naming import camel_to_snake
from tpen.training.events import TrainingIterationCompleted
from tpen.training.state import TrainerState

from .base import StatefulCallback
from .cadence import StepCadenceGate, SubscriptionGroup, pop_step_cadence


class RuntimeEquivariance(StatefulCallback[TrainerState]):
    """Schedule one or more runtime equivariance checkers.

    The callback owns *when* to run (``every_n_steps``, ``probability``),
    assigns each checker a stable log name, logs its metrics under
    ``checks/equivariance/<name>``, persists any failure artifact under
    ``artifact_dir``, and raises in ``fail_fast`` mode. Each injected checker
    owns *how* to check (permutation selection, comparison) and returns a
    `tpen.equivariance.checks.EquivarianceCheckResult`.

    The callback ``seed`` controls probabilistic *scheduling*; each checker's own
    ``seed`` controls *which permutations* it selects — separate random streams.

    Parameters
    ----------
    checkers : sequence
        Checker objects exposing ``run(state) -> EquivarianceCheckResult``.
        Must be non-empty; see the note below.
    fail_fast : bool, optional
        Raise when any checker fails.
    artifact_dir : str or pathlib.Path or None, optional
        Root directory for failure artifacts. When ``None``, artifacts are not
        written even if a checker returns one.

    Raises
    ------
    ValueError
        If ``checkers`` is empty.

    Notes
    -----
    **An empty checker list is rejected at construction.** Every record this
    callback publishes is emitted from inside the per-checker loop, so with no
    checkers the loop body never runs and the whole ``checks/equivariance/*``
    namespace simply never appears — while a config carrying ``fail_fast: true``
    reads as though equivariance were being enforced. That is worse than a check
    that passes vacuously: a vacuous pass at least leaves a record with counts in
    it that somebody could notice, whereas a vacuous *absence* leaves nothing to
    notice at all.

    Rejecting at construction rather than emitting a failing record at log time
    is deliberate, for three reasons. There is no log name to hang a record on:
    names are derived per checker, so a no-checker record would have to invent a
    ``checks/equivariance/*`` namespace that exists only to describe a
    misconfiguration. It is a configuration error, not an observation, and it is
    fully decidable before any compute is spent. And it is the only option that
    is loud in both modes — a failing record would crash a ``fail_fast: true``
    run late and would never fire at all under ``fail_fast: false``, which is
    precisely the silent case. The cost accepted in exchange is that the error
    surfaces during run setup instead of at the first checked step.

    This is the only migrated callback whose step coordinate reaches a durable
    path name: `_write_equivariance_artifact` formats ``step_{step:06d}``. It is
    read from the typed event so a future boundary change cannot turn it into
    ``step_-00001``. Each checker separately derives its permutation selection
    from the step it reads off the state; at this boundary the two coordinates
    are equal, so no permutation selection moves.
    """

    state_type: ClassVar[type[DomainState]] = TrainerState

    def __init__(
        self,
        *,
        checkers: "Sequence[Any]",
        fail_fast: bool = True,
        artifact_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        # Validated before anything else: `pop_step_cadence` mutates `kwargs`,
        # and a callback that cannot check anything should not get that far.
        checkers = list(checkers)
        if not checkers:
            raise ValueError(
                "RuntimeEquivariance requires at least one checker, got an empty "
                "`checkers` list. With no checkers this callback logs no "
                "`checks/equivariance/*` record at all, so a run would report no "
                "equivariance metrics while `fail_fast` suggested the checks were "
                "being enforced. Configure a checker, or remove the callback."
            )
        cadence = pop_step_cadence(kwargs)
        super().__init__(
            # Subscriptions are class-owned, never configured: this callback
            # observes completed training iterations and nothing else.
            typed_groups=(
                SubscriptionGroup(
                    selectors=(Subscription.of(TrainingIterationCompleted),)
                ),
            ),
            **kwargs,
        )
        # `cadence=None` on the group above is deliberate: a group `Cadence`
        # gates on the run-local occurrence count, which restarts after a
        # restore. This is the most expensive check in the stack, so a schedule
        # that silently shifted phase on a resumed run would be visible.
        self._steps = StepCadenceGate(cadence)
        self.checkers = checkers
        self.fail_fast = bool(fail_fast)
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        self._artifact_routing_bound = False
        self._checker_log_names = _assign_checker_log_names(self.checkers)

    def rebind_artifact_dir(self, artifact_dir: str | Path) -> None:
        """Bind this callback to an operationally routed artifact destination."""

        self.artifact_dir = Path(artifact_dir)
        self._artifact_routing_bound = True

    def handle_occurrence_impl(
        self,
        occurrence: Occurrence[TypedEvent],
        context: RunContext,
        state: TrainerState,
    ) -> None:
        """Run every checker against the current state and log/persist results."""

        event = occurrence.event
        if not isinstance(event, TrainingIterationCompleted):
            return
        # The coordinate rides the typed event, never `state.step`; see the note
        # on `tpen.training.state.TrainerState`'s value fields in the trainer.
        step = int(event.iteration.step)
        if not self._steps.should_run(step):
            return

        for checker, log_name in zip(self.checkers, self._checker_log_names, strict=True):
            result = checker.run(state)

            metrics = dict(result.metrics)
            metrics["passed"] = bool(result.passed)
            # Published here, from the typed field, rather than left to each
            # checker's free-form metrics: this is the key that can contradict a
            # `passed` reached by comparing nothing, so it must be present on
            # every record regardless of what the checker chose to report.
            metrics["n_comparisons"] = int(result.n_comparisons)
            metrics["checker_class"] = type(checker).__name__

            if result.artifact is not None and self.artifact_dir is not None:
                artifact_path = _write_equivariance_artifact(
                    root=self.artifact_dir,
                    checker_name=log_name,
                    step=step,
                    artifact=result.artifact,
                    enforce_containment=self._artifact_routing_bound,
                )
                metrics["artifact_path"] = str(artifact_path)

            context.log(metrics, step=step, namespace=f"checks/equivariance/{log_name}")

            if self.fail_fast and not result.passed:
                raise RuntimeError(
                    f"RuntimeEquivariance checker {log_name!r} failed at step {step}: "
                    f"{result.failures or metrics}"
                )


_DEFAULT_CHECKER_NAMES = {
    "FullModelEquivarianceChecker": "full_model",
    "TraceEquivarianceChecker": "trace",
}


def _checker_base_name(checker: object) -> str:
    """Return a readable base log name for a checker."""

    class_name = type(checker).__name__
    if class_name in _DEFAULT_CHECKER_NAMES:
        return _DEFAULT_CHECKER_NAMES[class_name]
    explicit = getattr(checker, "name", None)
    if explicit:
        return str(explicit)
    snake = camel_to_snake(class_name)
    for suffix in ("_equivariance_checker", "_checker"):
        if snake.endswith(suffix):
            return snake[: -len(suffix)]
    return snake or class_name.lower()


def _assign_checker_log_names(checkers: Sequence[object]) -> list[str]:
    """Assign stable, de-duplicated log names; warn (do not fail) on duplicates.

    The first instance of a base name keeps it; later duplicates get ``_1``,
    ``_2`` suffixes.
    """

    seen: dict[str, int] = {}
    names: list[str] = []
    for checker in checkers:
        base = _checker_base_name(checker)
        count = seen.get(base, 0)
        if count == 0:
            assigned = base
        else:
            assigned = f"{base}_{count}"
            warnings.warn(
                f"RuntimeEquivariance received duplicate checker name {base!r}; "
                f"using {assigned!r} for the duplicate.",
                stacklevel=3,
            )
        seen[base] = count + 1
        names.append(assigned)
    return names


def _write_equivariance_artifact(
    *,
    root: Path,
    checker_name: str,
    step: int,
    artifact: Mapping[str, Any],
    enforce_containment: bool = False,
) -> Path:
    """Write a failure artifact under ``root/<checker_name>/step_<step>/failure.json``.

    ``checker_name`` is derived from a checker's own, caller-controlled
    ``name`` attribute (``_checker_base_name``), not from anything this
    callback validates. In routed mode, an absolute or traversal name would
    otherwise discard or escape ``root`` through plain ``Path`` joining
    (review-r3 R3.2), so the resolved path is checked beneath ``root`` before
    writing. Legacy callers intentionally retain their existing filesystem
    behavior, including following an existing directory link.
    """

    path = root / checker_name / f"step_{int(step):06d}" / "failure.json"
    if enforce_containment:
        resolved_root = root.resolve()
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ArtifactRoutingError(
                f"RuntimeEquivariance checker name {checker_name!r} would write "
                f"to {resolved_path!s}, outside artifact_dir {root!s}; refusing"
            ) from exc
    write_json(path, dict(artifact))
    return path



__all__ = ["RuntimeEquivariance"]
