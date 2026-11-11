"""Binder coverage for ``sys_module_names``, mutated in both directions.

Every case here was found by an independent reviewer executing a mutant, not by
reading the implementation. They are grouped by the direction they protect,
because the two directions fail in opposite ways and only one is loud:

- **False positive** -- the guard accuses code that never touches ``sys``. Fails
  silently until it blocks a colleague's legitimate study.
- **False negative** -- the guard misses a real mutation inside the subset it
  CLAIMS to cover. Worse than an undocumented gap, because a reader trusting a
  stated limit stops looking.
"""

from __future__ import annotations

import ast

import pytest

from experiments.toolkit.ast_bindings import names_bound_in, sys_module_names


def _names(source: str) -> set[str]:
    return sys_module_names(ast.parse(source))


# --------------------------------------------------------------------------
# The plain positive: the whole point of the helper.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source, expected",
    [
        ("import sys\n", {"sys"}),
        ("import sys as s\n", {"s"}),
        ("import sys, os\n", {"sys"}),
        ("def f():\n    import sys as s\n", {"s"}),
        ("import sys\nimport sys as s\n", {"sys", "s"}),
    ],
)
def test_plain_sys_imports_are_resolved(source: str, expected: set[str]) -> None:
    assert _names(source) == expected


# --------------------------------------------------------------------------
# FALSE POSITIVE direction: a rebound name is not the sys module.
# Each binding form below let a rebound name keep being treated as `sys`.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "binder, source",
    [
        ("assignment", "import sys as s\ns = object()\n"),
        ("parameter", "import sys as s\ndef f(s):\n    pass\n"),
        ("import alias", "import sys as s\nimport os as s\n"),
        ("from-import alias", "import sys as s\nfrom os import path as s\n"),
        ("def name", "import sys as s\ndef s():\n    pass\n"),
        ("async def name", "import sys as s\nasync def s():\n    pass\n"),
        ("class name", "import sys as s\nclass s:\n    pass\n"),
        ("except target", "import sys as s\ntry:\n    pass\nexcept ValueError as s:\n    pass\n"),
        ("with target", "import sys as s\nwith open('f') as s:\n    pass\n"),
        ("for target", "import sys as s\nfor s in []:\n    pass\n"),
        ("comprehension", "import sys as s\n_ = [s for s in []]\n"),
        ("walrus", "import sys as s\nif (s := object()):\n    pass\n"),
        ("match capture", "import sys as s\nmatch 1:\n    case s:\n        pass\n"),
        ("match star", "import sys as s\nmatch []:\n    case [*s]:\n        pass\n"),
        ("match mapping rest", "import sys as s\nmatch {}:\n    case {**s}:\n        pass\n"),
        ("lambda parameter", "import sys as s\nf = lambda s: s\n"),
    ],
)
def test_a_rebound_name_is_not_treated_as_sys(binder: str, source: str) -> None:
    """Whatever rebinds the name, the guard must stop attributing it to sys."""

    assert _names(source) == set(), f"{binder} did not disable the alias"


# --------------------------------------------------------------------------
# FALSE NEGATIVE direction: an annotation binds NOTHING and must not disable.
# --------------------------------------------------------------------------
def test_a_bare_annotation_does_not_disable_the_alias() -> None:
    """``s: object`` records a type and binds nothing.

    It parses as an ``AnnAssign`` whose target is a ``Name`` in ``Store``
    context, so a naive rebinding check treats it as a reassignment and switches
    the guard OFF -- letting an annotated alias mutate ``sys.path`` unobserved.
    That is a false negative produced by the fix for a false positive.
    """

    assert _names("import sys as s\ns: object\ns.path.append('probe')\n") == {"s"}


def test_an_annotated_assignment_with_a_value_does_disable_the_alias() -> None:
    """``s: object = something`` DOES bind, so it must still disable."""

    assert _names("import sys as s\ns: object = object()\n") == set()


# --------------------------------------------------------------------------
# The tracked import must not cancel itself out.
# --------------------------------------------------------------------------
def test_the_tracked_import_is_not_counted_as_a_rebinding() -> None:
    """``import sys as s`` binds ``s``; that is the binding, not a rebinding.

    Without the exemption the helper would subtract every alias it just found
    and always return the empty set -- a guard that silently never fires.
    """

    assert "s" in names_bound_in(ast.parse("import sys as s\n"))
    assert "s" not in names_bound_in(
        ast.parse("import sys as s\n"), ignore_import_of="sys"
    )
    assert _names("import sys as s\n") == {"s"}
