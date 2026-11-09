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
        The ungrouped bucket is included only when it is non-empty.

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
        specs = tuple(specifications)
        if not specs:
            raise ValueError(f"group {group!r} has no module specifications")
        if any(not isinstance(spec, str) or not spec for spec in specs):
            raise TypeError(f"group {group!r} specifications must be non-empty strings")
        unknown = sorted(spec for spec in specs if spec not in module_names)
        if unknown:
            raise ValueError(f"group {group!r} names unknown module(s): {unknown}")
        normalized[group] = specs

    named_modules = tuple(model.named_modules())
    assignments: dict[str, list[tuple[str, Any]]] = {
        group: [] for group in normalized
    }
    assignments[ungrouped] = []
    for parameter_name, parameter in model.named_parameters():
        matching: list[str] = []
        for group, specifications in normalized.items():
            if any(
                _parameter_is_under_module(parameter_name, path, module, spec)
                for path, module in named_modules
                for spec in specifications
            ):
                matching.append(group)
        if len(matching) > 1:
            raise ValueError(
                f"parameter {parameter_name!r} is covered by overlapping groups {matching}"
            )
        assignments[matching[0] if matching else ungrouped].append((parameter_name, parameter))

    return {group: tuple(values) for group, values in assignments.items() if values}


def _parameter_is_under_module(
    parameter_name: str, path: str, module: Any, specification: str
) -> bool:
    """Return whether a named parameter belongs to one specified module."""

    trace_name = getattr(module, "trace_name", None)
    if specification != path and specification != trace_name:
        return False
    return not path or parameter_name == path or parameter_name.startswith(path + ".")


__all__ = ["resolve_parameter_groups"]
