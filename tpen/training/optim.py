"""Optimizer and update-method construction from Hydra factories or configs.

This module owns the Hydra-instantiation glue for the two objects that cannot
be fully built at config time, because both need the live model: the optimizer
needs its parameter references, and a stateful update method needs both the
optimizer and the model parameter binding.  Keeping that glue in one module
means neither `tpen.training.update` nor `tpen.training.sr` has to import
Hydra.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Iterable, Union

from hydra.utils import instantiate
from omegaconf import DictConfig

from tpen.dependencies import require_torch
from tpen.training.update import (
    LegacyAutogradUpdate,
    ModelParameterBinding,
    VMCUpdateMethod,
)

torch = require_torch(feature="optimizer construction")


# What a caller may supply as an update method. A configured method arrives as
# a Hydra `_partial_` FACTORY rather than an instance, because a stateful
# method needs the live optimizer and parameter binding. The annotation has to
# admit that shape: typeguard enforces these signatures under test, so a
# narrower one would reject every configured SR run at the trainer boundary.
UpdateMethodSpec = Union[VMCUpdateMethod, Callable[..., VMCUpdateMethod], Mapping, None]


def make_optimizer(factory_or_cfg: Any, params: Iterable[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """Build an optimizer bound to ``params`` from a factory or config.

    Accepts the shapes produced by Hydra instantiation of an optimizer block:

    - a ``functools.partial`` (or any callable) from a ``_partial_: true``
      ``_target_`` block, called as ``factory(params)``;
    - an un-instantiated mapping/`DictConfig` carrying ``_target_``, which is
      instantiated (honoring ``_partial_``) and, if still callable, called with
      ``params``;
    - an already-constructed `torch.optim.Optimizer`, returned unchanged.

    The optimizer YAML never needs a ``params`` key.

    Parameters
    ----------
    factory_or_cfg : Any
        Optimizer factory, config, or instance.
    params : iterable of torch.nn.Parameter
        Parameters the optimizer should manage.

    Returns
    -------
    torch.optim.Optimizer
        Constructed optimizer.

    Raises
    ------
    TypeError
        If ``factory_or_cfg`` cannot be turned into an optimizer.
    """

    if isinstance(factory_or_cfg, torch.optim.Optimizer):
        return factory_or_cfg

    candidate = factory_or_cfg
    if isinstance(candidate, (DictConfig, dict)) and "_target_" in candidate:
        candidate = instantiate(candidate)

    if isinstance(candidate, torch.optim.Optimizer):
        return candidate
    if callable(candidate):
        optimizer = candidate(params)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"optimizer factory must return a torch.optim.Optimizer, got {type(optimizer)!r}")
        return optimizer

    raise TypeError(
        "make_optimizer expects a torch.optim.Optimizer, a callable factory, or a config with "
        f"'_target_'; got {type(factory_or_cfg)!r}"
    )


def make_update_method(
    factory_or_cfg: Any,
    *,
    optimizer: torch.optim.Optimizer,
    model_parameters: ModelParameterBinding,
) -> VMCUpdateMethod | None:
    """Build an update method bound to a live optimizer and parameter binding.

    An update method cannot be fully instantiated by Hydra alone: a stateful
    method owns the optimizer and holds direct references to the model's live
    parameters, and neither exists until the runner has built the model. This
    accepts the same shapes as :func:`make_optimizer`:

    - a ``functools.partial`` (or any callable) from a ``_partial_: true``
      ``_target_`` block, called as ``factory(optimizer, model_parameters=...)``;
    - an un-instantiated mapping/`DictConfig` carrying ``_target_``, which is
      instantiated (honoring ``_partial_``) and, if still callable, completed
      the same way;
    - an already-constructed `VMCUpdateMethod`, returned unchanged;
    - ``None``, returned unchanged so the trainer keeps its legacy default.

    The update-method YAML therefore never needs an ``optimizer`` or
    ``model_parameters`` key.

    Parameters
    ----------
    factory_or_cfg : Any
        Update-method factory, config, instance, or ``None``.
    optimizer : torch.optim.Optimizer
        The optimizer the method should own.
    model_parameters : ModelParameterBinding
        The live parameter domain the method should update.

    Returns
    -------
    VMCUpdateMethod or None
        Constructed update method, or ``None`` when none was configured.

    Raises
    ------
    TypeError
        If `factory_or_cfg` cannot be turned into a `VMCUpdateMethod`.

    Notes
    -----
    This resolves an explicit ``_target_`` to a class; it is not a registry and
    performs no name-keyed lookup. A config names the exact object it wants, so
    a typo is an import error at construction rather than a silent fallback to
    some default method.
    """

    if factory_or_cfg is None:
        return None
    if isinstance(factory_or_cfg, VMCUpdateMethod):
        return factory_or_cfg

    candidate = factory_or_cfg
    if isinstance(candidate, (DictConfig, dict)) and "_target_" in candidate:
        candidate = instantiate(candidate)

    if isinstance(candidate, VMCUpdateMethod):
        return candidate
    if callable(candidate):
        method = candidate(optimizer, model_parameters=model_parameters)
        if not isinstance(method, VMCUpdateMethod):
            raise TypeError(
                f"update-method factory must return a VMCUpdateMethod, got {type(method)!r}"
            )
        return method

    raise TypeError(
        "make_update_method expects a VMCUpdateMethod, a callable factory, a config with "
        f"'_target_', or None; got {type(factory_or_cfg)!r}"
    )


def bind_update_method(
    spec: UpdateMethodSpec,
    *,
    optimizer: torch.optim.Optimizer,
    model_parameters: ModelParameterBinding,
    gradient_clip_norm: float | None = None,
) -> VMCUpdateMethod:
    """Normalize any supported spec -- including ``None`` -- into one method.

    This is the single construction entry point for an update method. It
    differs from :func:`make_update_method` in exactly one way, and that way is
    the point of it: ``make_update_method`` returns ``None`` for a ``None``
    spec, leaving its caller to decide what an unconfigured run should use,
    whereas this function resolves that decision here by building the historical
    `LegacyAutogradUpdate` adapter.

    Parameters
    ----------
    spec : VMCUpdateMethod or callable or Mapping or None
        Update-method instance, Hydra ``_partial_`` factory, un-instantiated
        config carrying ``_target_``, or ``None`` for the default adapter.
    optimizer : torch.optim.Optimizer
        The carrier the method will own. The same object is handed to the
        default adapter, so a run has one carrier however it was configured.
    model_parameters : ModelParameterBinding
        The live parameter domain the method will update.
    gradient_clip_norm : float or None, optional
        Clipping bound forwarded to the default adapter only. A configured
        method carries its own clipping policy, so this is deliberately not
        forced onto one.

    Returns
    -------
    VMCUpdateMethod
        Exactly one constructed method. Never ``None``.

    Notes
    -----
    Why the default lives HERE rather than in the trainer, which is where it
    used to live: the trainer's copy of this decision was reachable only
    through a private selection helper, and the memo that helper used to avoid
    rebuilding was keyed on the spec's identity -- so it did not apply when the
    spec was ``None``, and the default adapter was rebuilt on every selection.
    Putting construction in the module that already owns construction lets the
    trainer hold ONE bound method instead of re-deciding how to build one.

    This resolves an explicit ``_target_`` to a class; it is not a registry and
    performs no name-keyed lookup.
    """

    if spec is None:
        return LegacyAutogradUpdate(
            optimizer,
            gradient_clip_norm,
            model_parameters=model_parameters,
        )
    method = make_update_method(
        spec,
        optimizer=optimizer,
        model_parameters=model_parameters,
    )
    if method is None:
        # Unreachable today: `make_update_method` returns None only for a None
        # spec, which the branch above already consumed, and it type-checks
        # every other return path itself. Kept as a narrowing that fails loudly
        # rather than letting None escape a non-optional return type.
        raise TypeError("update-method spec resolved to no update method")
    return method


__all__ = [
    "UpdateMethodSpec",
    "bind_update_method",
    "make_optimizer",
    "make_update_method",
]
