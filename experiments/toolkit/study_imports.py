"""Study-scoped sibling module loading.

This module owns one concept: **how a module inside an ``experiments/`` study
directory reaches another module in the same study**.

Why this exists
---------------
Study directories are not packages.  Each one puts *its own* directory on
``sys.path`` -- explicitly via ``sys.path.insert(0, STUDY_DIR)``, and implicitly
whenever pytest collects a test from it in prepend import mode.  A sibling
reached with a bare ``import plan`` is therefore looked up as a **top-level**
module and cached in ``sys.modules`` under the bare key ``"plan"``.

``experiments/`` currently has three ``plan.py``, three ``collect.py``, four
``launch.py`` and two ``utils/`` packages.  Under a bare import the first study
loaded wins the bare key, and every study loaded afterwards silently receives
the first study's module.  The loud form of this is an ``AttributeError`` naming
a function that plainly exists; the dangerous form is two studies that happen to
share an API surface, where there is no exception at all -- just the wrong
module, the wrong constants, and a green test suite.

Why the fix has to live in the *owning* module
----------------------------------------------
A consumer that loads a study module by path under a unique name fixes only the
name **it** binds.  It cannot fix the names that the loaded module itself binds:
``collect.py`` still executes ``import plan``, which still resolves through the
shared bare key.  So a consumer cannot protect itself, and the change belongs in
every module that reaches a sibling.

Why not relative imports in real packages
-----------------------------------------
One blocker, and it is sufficient on its own: these modules are **entrypoints
executed directly** (``python plan.py``).  At this revision 32 of the rewritten
files carry an ``if __name__ == "__main__"`` block; re-derive rather than trust
the number -- it moves with the diff.  A directly executed file runs as
``__main__`` with no parent package, so a relative import raises
``ImportError: attempted relative import with no known parent package``.

A CORRECTION, recorded because the superseded reason is the one a future reader
would try to work around.  An earlier version of this docstring gave a second
blocker: that ``he-v1``, ``he-cutover``, ``tpen-pair-scan-v1`` and
``tpen-pair-v1`` contain hyphens, which are not Python identifiers, so the
directories would need renaming.  **That is wrong as stated.**  An independent
reviewer loaded a synthetic hyphenated package containing a relative import via
``importlib.import_module`` and it worked: a hyphen blocks only the ``import
he-v1`` *statement*, not package-hood, and relative imports inside such a
package resolve normally.  (There were also four such directories, not three.)
So renaming is NOT required, and only the direct-execution concern survives.
That superseded rationale was reviewed and explicitly endorsed by the lane
manager before it was checked, and it was still wrong -- an endorsement is not
a verification, and a rationale arriving with one still needs testing.

Loading by path under a study-unique key keeps every filename and every
documented ``python <stage>.py`` invocation working unchanged.

Why EVERY sibling import is routed through here, not just ambiguous ones
------------------------------------------------------------------------
An earlier version converted only siblings whose names collide across studies.
That was wrong, and it broke the one live cross-study boundary.  When some of a
study's modules are loaded study-scoped and others by bare import, the study
ends up with TWO copies of the same files: ``he-cutover``'s ``hev1.py`` received
scoped modules while He-v1's own bare ``import strata`` built a second set.
Two copies of ``strata`` means two distinct ``StratumError`` classes, so
``except hev1.strata.StratumError`` silently failed to catch an error raised
through ``hev1.plan_stage.strata``.

Duplicate identity is a wrong-module defect exactly as a shared bare key is, so
fixing only the collisions traded one for the other.  A study must therefore be
loaded consistently: **all** of its sibling imports go through this module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: Root key that every study-scoped module is registered beneath.  Chosen to be
#: a name no study can supply, so it can never itself be captured.
_NAMESPACE = "_tpen_study"

#: ``experiments/`` -- the boundary a study directory is named relative to, so
#: that two studies with the same directory *basename* under different parents
#: still get distinct keys.
_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent


def study_slug(study_dir: Path) -> str:
    """Return the unique ``sys.modules`` infix for one study directory.

    Parameters
    ----------
    study_dir : Path
        Absolute path to the study directory.

    Returns
    -------
    str
        An identifier-safe, INJECTIVE encoding of the study's path relative to
        ``experiments/``.

    Notes
    -----
    The encoding must be injective, not merely sanitized.  Replacing every
    non-alphanumeric character with ``_`` maps ``new-study`` and ``new_study``
    to the same key, so the second study's ``plan.py`` would silently return the
    first study's already-cached module -- reintroducing precisely the wrong-
    module-without-an-exception failure this module exists to remove.  No pair
    of current directories collides that way, so the defect would have been
    latent until somebody added one.

    Alphanumerics survive; ``_`` doubles to ``__``; every other character
    becomes ``_`` plus its two-digit hex byte.  ``_`` is not a hex digit, so
    ``__`` can never be confused with an escape, and the mapping is reversible.
    ``hooke/tpen-pair-scan-v1`` encodes as ``hooke_2ftpen_2dpair_2dscan_2dv1``.

    Derived from the full relative path rather than the basename, so
    ``a/study`` and ``b/study`` also stay distinct. A study OUTSIDE this
    loader's ``experiments/`` is encoded from its absolute path under an
    ``_abs_`` anchor, so the inside and outside spaces cannot overlap either.

    ONE LOADER INSTANCE SERVES EVERY CHECKOUT in an interpreter -- whichever
    bootstrapped first. Two checkouts exist precisely when two revisions differ,
    so if their ``study_imports.py`` files THEMSELVES differ, the second
    checkout's studies are keyed and cache-validated by the first checkout's
    loader code, silently. No collision results (slugs stay distinct and
    ``_load_leaf`` still validates), but the behaviour is the older loader's.
    Untested and not covered by any test here; recorded rather than fixed.
    """

    resolved = study_dir.resolve()
    try:
        relative = resolved.relative_to(_EXPERIMENTS_ROOT)
        anchor = ""
    except ValueError:
        # A study outside this loader's experiments/ -- the ordinary case when a
        # second checkout is in play. Fall back to the absolute path, MINUS the
        # filesystem root, which is not an identifier.
        #
        # The anchor is not decoration. Without it an outside `/foo/bar` and an
        # inside `experiments/foo/bar` encode identically, which would break the
        # injectivity this function promises -- and it promises it because a
        # collision here silently returns another study's module.
        #
        # WHY `_abs_` IS UNREACHABLE FROM INSIDE, stated correctly. An earlier
        # version of this comment claimed "an encoded segment never begins with
        # `_`". That is FALSE -- `-foo` encodes to `_2dfoo`. The real argument is
        # about the character AFTER a leading `_`: it is either `_` (an escaped
        # underscore) or the first hex digit of a byte, and those digits are
        # confined to {0-7, c-f}. ASCII is 0x00-0x7F, UTF-8 lead bytes are
        # 0xC2-0xF4, and 0xA0-0xBF occur only as continuation bytes, never
        # first. So `_a` cannot begin an encoding and `_abs_` cannot be
        # produced from inside. `test_no_inside_encoding_can_begin_with_the_anchor`
        # holds this by execution rather than by argument.
        #
        # ANY FUTURE ANCHOR MUST REDO THIS ARGUMENT. It is specific to `_a`;
        # an anchor beginning `_c3_` or `_2d` would sit squarely inside the
        # reachable space and collide silently.
        relative = Path(*resolved.parts[1:])
        anchor = "_abs_"

    encoded = [anchor] if anchor else []
    for char in relative.as_posix():
        if char.isalnum():
            encoded.append(char)
        elif char == "_":
            encoded.append("__")
        else:
            encoded.extend(f"_{byte:02x}" for byte in char.encode("utf-8"))
    return "".join(encoded)


def _ensure_namespace_parents(slug: str, study_dir: Path) -> None:
    """Register the ``_tpen_study`` and ``_tpen_study.<slug>`` parent packages.

    Submodule imports and relative imports both walk the parent chain, so the
    parents must exist in ``sys.modules`` before any leaf is executed.
    """

    if _NAMESPACE not in sys.modules:
        root = ModuleType(_NAMESPACE)
        root.__path__ = []  # namespace-style: it owns no directory of its own
        sys.modules[_NAMESPACE] = root

    study_key = f"{_NAMESPACE}.{slug}"
    if study_key not in sys.modules:
        holder = ModuleType(study_key)
        # __path__ points at the study directory, so ``<study_key>.<name>``
        # resolves siblings through ordinary package machinery.
        holder.__path__ = [str(study_dir)]
        sys.modules[study_key] = holder
        setattr(sys.modules[_NAMESPACE], slug, holder)


def _load_leaf(key: str, path: Path, is_package: bool) -> ModuleType:
    """Import one module from ``path`` and register it in ``sys.modules`` as ``key``."""

    expected = (path / "__init__.py") if is_package else path

    cached = sys.modules.get(key)
    if cached is not None:
        # A cache hit must be VALIDATED against the source it was asked for.
        #
        # The key is derived from the study's path relative to `experiments/`,
        # so it is checkout-relative: two checkouts of this repository produce
        # the SAME key for the same study. Returning the cached object without
        # checking would hand a caller in checkout B a module loaded from
        # checkout A -- silently, with no exception, which is the exact failure
        # this module exists to remove, one level up from studies to checkouts.
        #
        # Making the key carry provenance would fix it properly and is a design
        # change; see the filed follow-up. This check does the cheap half, and
        # it is the half that matters: it converts a silent wrong module into a
        # loud error, which is the transformation this module is for.
        cached_file = getattr(cached, "__file__", None)
        if cached_file is None or Path(cached_file).resolve() != expected.resolve():
            raise ImportError(
                f"study module key {key!r} is already bound to "
                f"{cached_file!r}, but {str(expected)!r} was requested. "
                "The cache key is relative to experiments/, so two checkouts "
                "of this repository collide on it. Load studies from one "
                "checkout per interpreter."
            )
        return cached

    if is_package:
        spec = importlib.util.spec_from_file_location(
            key, path / "__init__.py", submodule_search_locations=[str(path)]
        )
    else:
        spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load study module {key!r} from {path}")

    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec so that a cycle between two siblings terminates,
    # exactly as the normal import system does it.
    sys.modules[key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(key, None)
        raise
    return module


def load_study_module(study_dir: Path, dotted: str) -> ModuleType:
    """Return a module from ``study_dir``, loaded by path under a unique key.

    The primitive behind :func:`sibling`.  Use it directly only when the study
    being reached is NOT the caller's own -- a deliberate cross-study boundary.

    Parameters
    ----------
    study_dir : Path
        The study directory that owns ``dotted``.
    dotted : str
        Module to load, relative to ``study_dir``.  A plain module (``"plan"``),
        a package (``"utils"``), or a package submodule (``"utils.layout"``).

    Returns
    -------
    ModuleType
        The requested module.

    Raises
    ------
    ImportError
        If no file backs ``dotted`` inside ``study_dir``.
    """

    study_dir = Path(study_dir).resolve()
    slug = study_slug(study_dir)
    _ensure_namespace_parents(slug, study_dir)

    key = f"{_NAMESPACE}.{slug}"
    current = study_dir
    module: ModuleType | None = None
    for part in dotted.split("."):
        key = f"{key}.{part}"
        package_dir = current / part
        module_file = current / f"{part}.py"
        if package_dir.is_dir() and (package_dir / "__init__.py").exists():
            module = _load_leaf(key, package_dir, is_package=True)
            current = package_dir
        elif module_file.exists():
            module = _load_leaf(key, module_file, is_package=False)
            current = module_file
        else:
            raise ImportError(
                f"no module {part!r} of {dotted!r} in study directory {study_dir}"
            )
        # Bind onto the parent so ``parent.child`` attribute access works and so
        # relative imports inside the child resolve against a populated parent.
        parent = sys.modules.get(key.rsplit(".", 1)[0])
        if parent is not None:
            setattr(parent, part, module)

    assert module is not None  # loop runs at least once for a non-empty name
    return module


def sibling(caller_file: str, dotted: str) -> ModuleType:
    """Return a module from the caller's own study directory.

    Loads by **path** under a study-unique ``sys.modules`` key, so two studies
    can hold same-named siblings at once and neither can capture the other.

    Parameters
    ----------
    caller_file : str
        The calling module's ``__file__``.  Its parent directory is the study.
    dotted : str
        Sibling to load, relative to the study directory.  Either a plain module
        (``"plan"``), a package (``"utils"``), or a package submodule
        (``"utils.layout"``).

    Returns
    -------
    ModuleType
        The requested module.

    Raises
    ------
    ImportError
        If no file backs ``dotted`` inside the study directory.

    Examples
    --------
    >>> plan_stage = sibling(__file__, "plan")            # doctest: +SKIP
    >>> layout = sibling(__file__, "utils.layout")        # doctest: +SKIP
    """

    return load_study_module(Path(caller_file).resolve().parent, dotted)
