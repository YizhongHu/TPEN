"""Resolve model parameters into explicitly configured module groups."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def resolve_parameter_groups(
    model: Any,
    groups: Mapping[str, Iterable[str]],
    *,
    ungrouped: str = "ungrouped",
) -> dict[str, tuple[tuple[str, Any], ...]]:
    """Assign every named model parameter to one configured module group.

    Parameters
    ----------
    model : Any
        A PyTorch module exposing ``named_modules`` and ``named_parameters``.
    groups : mapping[str, iterable[str]]
        Group names mapped to module paths. A path matches the module itself
        and all of its descendants. Modules with a non-empty ``trace_name``
        may also be addressed by that stable name, following :class:`Trace`.
    ungrouped : str, optional
        Bucket receiving parameters not covered by a configured group.

    Returns
    -------
    dict
        Group names mapped to ordered ``(parameter_name, parameter)`` pairs.
        Ownership is by parameter identity, not by the spelling of a name:
        tied aliases therefore belong to every selected module that exposes
        the tensor, and an identity claimed by two groups is refused. The
        returned pairs use the first model-level name for each identity, and
        the ungrouped bucket is included only when it is non-empty.

    Raises
    ------
    ValueError
        If a group is empty, uses the reserved ungrouped name, names an
        unknown module, or overlaps another group for any parameter.
    TypeError
        If ``groups`` is not a mapping or a module specification is not text.
    """

    if not isinstance(groups, Mapping):
        raise TypeError("groups must be a mapping of group names to module paths")
    if not isinstance(ungrouped, str) or not ungrouped:
        raise ValueError("ungrouped must be a non-empty string")

    module_names: dict[str, set[int]] = {}
    for path, module in model.named_modules():
        names = {path}
        trace_name = getattr(module, "trace_name", None)
        if trace_name:
            names.add(str(trace_name))
        for name in names:
            module_names.setdefault(name, set()).add(id(module))

    normalized: dict[str, tuple[str, ...]] = {}
    for group, specifications in groups.items():
        if not isinstance(group, str) or not group:
            raise TypeError("group names must be non-empty strings")
        if group == ungrouped:
            raise ValueError(f"group name {group!r} is reserved for ungrouped parameters")
        specs = (specifications,) if isinstance(specifications, str) else tuple(specifications)
        if not specs:
            raise ValueError(f"group {group!r} has no module specifications")
        if any(not isinstance(spec, str) or not spec for spec in specs):
            raise TypeError(f"group {group!r} specifications must be non-empty strings")
        unknown = sorted(spec for spec in specs if spec not in module_names)
        if unknown:
            raise ValueError(f"group {group!r} names unknown module(s): {unknown}")
        normalized[group] = specs

    named_modules = tuple(model.named_modules())
    # ``remove_duplicate=False`` is essential here.  PyTorch's default
    # ``named_parameters`` traversal keeps only the first spelling of a tied
    # parameter, which would make a later alias appear ungrouped and would
    # hide an overlap between two explicitly selected modules.
    named_parameters = tuple(model.named_parameters(remove_duplicate=False))
    canonical_parameters: dict[int, tuple[str, Any]] = {}
    for parameter_name, parameter in named_parameters:
        canonical_parameters.setdefault(id(parameter), (parameter_name, parameter))
    selected_by_group: dict[str, set[int]] = {group: set() for group in normalized}
    for group, specifications in normalized.items():
        for path, module in named_modules:
            if any(
                _module_matches_specification(path, module, specification)
                for specification in specifications
            ):
                selected_by_group[group].update(
                    id(parameter)
                    for _, parameter in module.named_parameters(
                        prefix=path, recurse=True, remove_duplicate=False
                    )
                )

    ownership: dict[int, str] = {}
    for group, identities in selected_by_group.items():
        for identity in identities:
            previous = ownership.setdefault(identity, group)
            if previous != group:
                parameter_name = canonical_parameters[identity][0]
                raise ValueError(
                    f"parameter {parameter_name!r} is covered by overlapping groups "
                    f"[{previous!r}, {group!r}]"
                )

    assignments: dict[str, list[tuple[str, Any]]] = {
        group: [] for group in normalized
    }
    assignments[ungrouped] = []
    for identity, (parameter_name, parameter) in canonical_parameters.items():
        assignments[ownership.get(identity, ungrouped)].append((parameter_name, parameter))

    return {group: tuple(values) for group, values in assignments.items() if values}


def _module_matches_specification(path: str, module: Any, specification: str) -> bool:
    """Return whether a module is addressed by one configured specification."""

    trace_name = getattr(module, "trace_name", None)
    return specification == path or specification == trace_name


__all__ = ["resolve_parameter_groups"]
