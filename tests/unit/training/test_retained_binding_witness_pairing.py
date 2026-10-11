"""The stateful/stateless pairing is enforced, and the enforcement is itself tested.

A retained-guard timing witness is only half a witness. On the STATEFUL path a
single deferral of `bind_update_method`'s retained model or carrier check is
shadowed -- `_resolve_method_state` refuses first. On the STATELESS path it is
not, because that method returns early when `update_state()` is `None`, so
neither the ownership guard nor the parameter guard ever runs. An independent
verifier measured all four retained-binding single deferrals crossing their
boundary on the stateless path (Cannon 52093726).

Pairing the two paths by hand is a convention a future edit can quietly break,
so this module turns it into a property the suite enforces.

THE FIRST VERSION OF THIS MODULE WAS BYPASSABLE, and the bypass was found by an
independent verifier (P7, Cannon 52103983), not by me. It tested

    "stateless" in str(mark.args[0])

a SUBSTRING. A witness parametrised over `not_stateless` contains that substring,
satisfied the check, and then ignored the parameter entirely by calling its
fixture with a fixed `False` -- running the stateful path twice while advertising
both ids. It survived 3/3. The module was not decoration: five other negative
mutants against it failed as intended. It was REAL AND NARROWER THAN ITS NAME,
which is the harder defect to notice, because the thing works and so nobody asks
whether it works for the reason claimed.

Two holes came out of that, and they are not equally closeable:

1. SUBSTRING vs EXACT ARGNAME. Closed here: `argnames` is parsed into a set of
   names and membership is required.
2. DECLARATION IS NOT USE. A witness can declare both ids and still never vary,
   by ignoring the parameter. No metadata check can see that. The partial
   mitigation below is an AST check that the function body REFERENCES the name.
   It does NOT establish that the value reaches the behaviour under test, and
   this docstring says so rather than letting the module's name imply otherwise.

WHAT THIS DOES NOT ESTABLISH: it constrains witnesses that CARRY the marker. A
witness added without the marker is outside this contract by construction.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest

MODULES = (
    "tests.unit.training.test_update_binding_parameter_layout_guards",
    "tests.integration.evaluation.test_hooke_eval_runner",
)

PARAM = "stateless"
REQUIRED_IDS = ("stateful", "stateless")
REQUIRED_VALUES = {False, True}


def _argnames(raw) -> set[str]:
    """Parse `parametrize`'s first argument into a SET of names.

    pytest accepts `"a"`, `"a,b"` and `["a", "b"]`. Membership in this set is the
    predicate; `"stateless" in raw` is NOT, because `not_stateless` satisfies it.
    """

    names = raw.split(",") if isinstance(raw, str) else list(raw)
    return {str(name).strip() for name in names}


def _references_parameter(source: str, parameter: str = PARAM) -> bool:
    """True when the function body LOADS `parameter` somewhere.

    This catches a witness that accepts the argument and then ignores it, which
    is how the P7 decoy ran one path twice. It does NOT prove the value reaches
    the behaviour under test -- only that it is read at all.
    """

    # `textwrap.dedent`, NOT `inspect.cleandoc`. cleandoc treats the first line
    # specially and then dedents the rest, which turns a module-level function
    # body into a bare statement at column 0 and raises IndentationError. Caught
    # locally before this reached the cluster.
    tree = ast.parse(textwrap.dedent(source))
    return any(
        isinstance(node, ast.Name)
        and node.id == parameter
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(tree)
    )


def pairing_violations(marks, source: str) -> list[str]:
    """Return every reason this witness fails the pairing contract."""

    problems: list[str] = []
    parametrize = [m for m in marks if m.name == "parametrize"]
    matching = [m for m in parametrize if PARAM in _argnames(m.args[0])]

    if not matching:
        declared = sorted({n for m in parametrize for n in _argnames(m.args[0])})
        problems.append(
            f"not parametrised over `{PARAM}`; declared argnames {declared}. "
            "The stateful path alone leaves the one-fault stateless gap untested."
        )
        return problems

    for mark in matching:
        values = set(mark.args[1])
        if values != REQUIRED_VALUES:
            problems.append(
                f"parametrised over `{PARAM}` with {sorted(values)}; both paths are "
                "required because only the stateless one is unshadowed."
            )
        ids = tuple(mark.kwargs.get("ids", ()))
        if ids != REQUIRED_IDS:
            problems.append(
                f"ids are {list(ids)}; must be {list(REQUIRED_IDS)} so a failure "
                "names the path it failed on."
            )

    if not _references_parameter(source):
        problems.append(
            f"body never reads `{PARAM}`; the parameter is declared and ignored, "
            "so both ids run the same path."
        )
    return problems


def _marked_witnesses():
    found = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for attr in dir(module):
            obj = getattr(module, attr)
            if not callable(obj) or not attr.startswith("test_"):
                continue
            marks = getattr(obj, "pytestmark", [])
            if any(m.name == "retained_binding_timing" for m in marks):
                found.append((module_name, attr, marks, inspect.getsource(obj)))
    return found


def test_the_marked_witness_set_is_not_empty() -> None:
    """Non-emptiness control: an empty scan would pass every check below."""

    found = _marked_witnesses()
    assert len(found) == 4, (
        "expected the four retained-binding timing witnesses; found "
        f"{[(m, n) for m, n, _, _ in found]}"
    )


@pytest.mark.parametrize("module_name", MODULES)
def test_every_marked_witness_is_paired_over_stateful_and_stateless(
    module_name: str,
) -> None:
    witnesses = [w for w in _marked_witnesses() if w[0] == module_name]
    assert witnesses, f"no marked retained-binding witness found in {module_name}"

    for _module, name, marks, source in witnesses:
        problems = pairing_violations(marks, source)
        assert not problems, f"{name}: " + "; ".join(problems)


# ---------------------------------------------------------------------------
# The enforcement must REJECT crafted bypasses, not merely accept good input.
# A guard never shown to reject anything is an intention with a test file around
# it. The first case below is the independent verifier's P7 decoy.
# ---------------------------------------------------------------------------


def _mark(name, *args, **kwargs):
    return SimpleNamespace(name=name, args=args, kwargs=kwargs)


GOOD_SOURCE = "def w(stateless):\n    return _bound_trainer(stateless)\n"


@pytest.mark.parametrize(
    ("label", "marks", "source", "expected_fragment"),
    [
        (
            "P7_substring_bypass",
            [_mark("parametrize", "not_stateless", [False, True], ids=REQUIRED_IDS)],
            "def w(not_stateless):\n    return _bound_trainer(False)\n",
            "not parametrised over `stateless`",
        ),
        (
            "declared_but_ignored",
            [_mark("parametrize", PARAM, [False, True], ids=REQUIRED_IDS)],
            "def w(stateless):\n    return _bound_trainer(False)\n",
            "never reads `stateless`",
        ),
        (
            "one_value_only",
            [_mark("parametrize", PARAM, [False], ids=REQUIRED_IDS)],
            GOOD_SOURCE,
            "both paths are required",
        ),
        (
            "wrong_ids",
            [_mark("parametrize", PARAM, [False, True], ids=("a", "b"))],
            GOOD_SOURCE,
            "names the path it failed on",
        ),
        (
            "no_parametrize_at_all",
            [],
            GOOD_SOURCE,
            "not parametrised over `stateless`",
        ),
    ],
)
def test_the_enforcement_rejects_a_crafted_bypass(
    label: str, marks, source: str, expected_fragment: str
) -> None:
    problems = pairing_violations(marks, source)
    assert problems, f"{label} was ACCEPTED; the enforcement does not enforce"
    assert any(expected_fragment in p for p in problems), (
        f"{label} was rejected for the wrong reason: {problems}"
    )


def test_a_lawful_witness_is_accepted() -> None:
    """Polarity control: the rejections above must not come from rejecting all."""

    marks = [_mark("parametrize", PARAM, [False, True], ids=REQUIRED_IDS)]
    assert pairing_violations(marks, GOOD_SOURCE) == []


def test_comma_joined_and_sequence_argnames_are_both_parsed() -> None:
    """`parametrize` accepts three argname spellings; all must mean the same."""

    for raw in ("stateless", "other,stateless", ["other", "stateless"]):
        marks = [_mark("parametrize", raw, [False, True], ids=REQUIRED_IDS)]
        assert pairing_violations(marks, GOOD_SOURCE) == [], raw
