"""Which names a Python source file binds, and which are bound to ``sys``.

This module owns one concept: **deciding, by parsing, whether a given name in a
file still refers to what an ``import`` bound it to**.

Two separate guards need that answer -- the ``sys.path`` rule in
``experiments/atomistic/he-cutover/test_cutover_configs.py`` and the
``sys.modules`` rule in
``tests/unit/experiments/test_study_module_identity.py``.  They had a copy each,
and the copies were wrong in the same way at the same time.  Two copies of a
rule is the shape that produced two of this slice's own defects, so the logic
lives here once.

WHAT THIS IS FOR, and its ceiling
---------------------------------
This is a *guard* helper, not a static analyser.  It answers "was this name
plainly imported and never touched again?" and is deliberately conservative:
when a name might have been rebound, it is dropped, because a guard that
falsely accuses legitimate code is worse than one that misses a case.  A false
accusation surfaces only when it blocks a colleague's work; a miss leaves the
status quo.

It does NOT do scope analysis.  A name rebound inside one function is treated as
rebound everywhere in the file.  That is intentional and is the conservative
direction -- but state the consequence plainly, because it is sharper than
"conservative" suggests:

**THE DROP IS A PER-FILE KILL SWITCH.**  A single function parameter named
``sys`` anywhere in a file disables the rule for that ENTIRE file, including
mutations far from the shadowing.  Measured, not theorised.  The fix for a
false positive installed a blind spot with the same trigger, and the honest
description of the trade is that this instrument prefers silence to accusation
at file granularity.  Making it scope-aware is the static-analysis project this
deliberately is not.
"""

from __future__ import annotations

import ast


def _annotation_only_targets(tree: ast.AST) -> set[int]:
    """Return ``id()`` of ``Name`` nodes that are bare annotations, not bindings.

    ``s: object`` parses as an ``AnnAssign`` whose target is a ``Name`` in
    ``Store`` context, but it binds nothing at all -- it only records an
    annotation.  Treating it as a rebinding silently switched the guards OFF for
    that name, so an annotated alias could then mutate ``sys.path`` unobserved.
    A false NEGATIVE produced by the fix for a false positive.
    """

    return {
        id(node.target)
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and node.value is None
        and isinstance(node.target, ast.Name)
    }


def names_bound_in(tree: ast.AST, *, ignore_import_of: str | None = None) -> set[str]:
    """Return every name the file binds, by any binding form.

    Parameters
    ----------
    tree : ast.AST
        A parsed module.
    ignore_import_of : str, optional
        A module name whose plain ``import`` should NOT count as a binding.
        Callers tracking ``import sys`` pass ``"sys"``, so that the very import
        being tracked does not cancel itself out.

    Returns
    -------
    set of str
        Names bound anywhere in the file.

    Notes
    -----
    Binding forms covered, each because a measured guard case escaped without
    it: assignment and every ``Name``-in-``Store`` form (comprehension targets,
    ``with ... as``, walrus); function and lambda parameters (``ast.arg``, which
    is NOT a ``Name``); ``def``/``async def``/``class`` names; ``except ... as``;
    ``match`` captures (``MatchAs``, ``MatchStar``, ``MatchMapping`` rest); and
    imports other than the one being tracked.
    """

    annotation_only = _annotation_only_targets(tree)
    bound: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store) and id(node) not in annotation_only:
                bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    # A star import binds an unknowable set. Parsing cannot say
                    # whether it rebound the tracked name, so the conservative
                    # answer is "assume it did" -- see STAR_IMPORT below.
                    bound.add("*")
                    continue
                if (
                    isinstance(node, ast.Import)
                    and ignore_import_of is not None
                    and alias.name == ignore_import_of
                ):
                    continue  # the tracked import is a binding, not a rebinding
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.MatchAs):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, ast.MatchStar):
            if node.name:
                bound.add(node.name)
        elif isinstance(node, ast.MatchMapping):
            if node.rest:
                bound.add(node.rest)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            for arg in [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
                *([arguments.vararg] if arguments.vararg else []),
                *([arguments.kwarg] if arguments.kwarg else []),
            ]:
                bound.add(arg.arg)

    return bound


def sys_module_names(tree: ast.AST) -> set[str]:
    """Return names that a plain ``import sys`` bound and nothing else rebinds.

    WHAT THIS MECHANICALLY DOES, stated as syntax because that is all it is.
    It scans for an ``ast.Import`` node carrying an alias whose ``name`` is
    exactly ``"sys"``, and yields ``asname or "sys"``. It then subtracts every
    name bound by anything OTHER THAN a plain ``import sys`` itself, and returns
    nothing at all if the file contains a star import. (The exemption matters:
    ``import sys`` is a binding form for ``sys``, so subtracting it too would
    make this function always return the empty set -- a guard that silently
    never fires.)

    IT IS A SYNTACTIC MATCHER, NOT A SEMANTIC ONE. It does not resolve values,
    does not track scope, and cannot know what a name refers to at runtime. Read
    the previous paragraph as the entire promise; anything phrased as "the guard
    excludes rebound or shadowed names" would be a claim about program meaning
    and this cannot deliver one.

    KNOWN-UNCAUGHT, BY CATEGORY. These are CATEGORIES with examples, **not an
    exhaustive list** -- three further path forms and several ``sys.modules``
    forms were found by later review after an earlier enumeration was written as
    though it were complete. Assume any construct not literally matched above
    evades this.

    1. **The module is never named syntactically.** ``from sys import path`` /
       ``from sys import modules``, with or without ``as``;
       ``importlib.import_module("sys")``; ``__import__("sys")``; reaching it
       through another module's attribute.
    2. **The value is aliased rather than the module.** ``s = sys`` (which also
       disables the name here), ``p = sys.path``, ``m = sys.modules``, or
       passing either into a function that mutates it.
    3. **Scope- or runtime-dependent meaning.** A name bound differently on two
       branches, rebound inside one function only, or produced by ``exec``.

    Callers impose a FOURTH category of their own -- which *operations* on the
    resolved name they match. Each caller documents its own operation set; see
    ``_cross_study_violations`` and ``find_bare_sys_modules_registrations``.

    Closing categories 1-3 needs scope-aware binding analysis with runtime
    awareness, which is a static-analysis project rather than a guard, and is
    deliberately not attempted. A narrow accurate guarantee is worth more than a
    broad one that is wrong in both directions.
    """

    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sys":
                    aliases.add(alias.asname or "sys")

    bound = names_bound_in(tree, ignore_import_of="sys")
    if "*" in bound:
        # STAR_IMPORT: `from x import *` binds a set this parser cannot know, so
        # it may have rebound the tracked name. Conservative direction is to
        # attribute nothing, accepting misses rather than risking accusation.
        return set()
    return aliases - bound
