"""The stateful/stateless pairing is enforced, not merely conventional.

A retained-guard timing witness is only half a witness. On the STATEFUL path a
single deferral of `bind_update_method`'s retained model or carrier check is
shadowed: `_resolve_method_state` refuses first. On the STATELESS path it is
not, because that method returns early when `update_state()` is `None`, so
neither the ownership guard nor the parameter guard ever runs. An independent
verifier measured all four retained-binding single deferrals crossing their
boundary on the stateless path (Cannon 52093726) after the lane had claimed in
writing that the method-state guard always catches them.

Pairing the two paths by hand is a convention, and a convention is something a
future edit can quietly break: delete one `parametrize` argument and the suite
still passes while half the obligation silently stops being tested. This module
turns the pairing into a property the suite enforces.

WHAT THIS DOES NOT ESTABLISH: it constrains witnesses that CARRY the marker. A
new witness added without the marker is outside this contract. That is a real
limit and the honest place to state it is here rather than in a docstring that
claims more than it checks.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = (
    "tests.unit.training.test_update_binding_parameter_layout_guards",
    "tests.integration.evaluation.test_hooke_eval_runner",
)

REQUIRED_IDS = {"stateful", "stateless"}


def _marked_witnesses():
    found = []
    for name in MODULES:
        module = importlib.import_module(name)
        for attr in dir(module):
            obj = getattr(module, attr)
            if not callable(obj) or not attr.startswith("test_"):
                continue
            marks = getattr(obj, "pytestmark", [])
            if any(m.name == "retained_binding_timing" for m in marks):
                found.append((name, attr, marks))
    return found


def test_the_marked_witness_set_is_not_empty() -> None:
    """A non-emptiness control: an empty scan would pass every check below.

    Without this, deleting every marker turns the enforcement into a vacuous
    pass -- the same false-green polarity that makes an empty file compile.
    """

    assert len(_marked_witnesses()) == 4, (
        "expected the four retained-binding timing witnesses; found "
        f"{[(m, n) for m, n, _ in _marked_witnesses()]}"
    )


@pytest.mark.parametrize("module_name", MODULES)
def test_every_marked_witness_is_paired_over_stateful_and_stateless(
    module_name: str,
) -> None:
    """Each marked witness must be parametrised over BOTH paths."""

    witnesses = [w for w in _marked_witnesses() if w[0] == module_name]
    assert witnesses, f"no marked retained-binding witness found in {module_name}"

    for _module, name, marks in witnesses:
        parametrize = [m for m in marks if m.name == "parametrize"]
        stateless_marks = [m for m in parametrize if "stateless" in str(m.args[0])]
        assert stateless_marks, (
            f"{name} carries the retained_binding_timing marker but is not "
            "parametrised over `stateless`; the stateful path alone leaves the "
            "one-fault stateless gap untested"
        )
        for mark in stateless_marks:
            values = set(mark.args[1])
            assert values == {False, True}, (
                f"{name} is parametrised over `stateless` with {sorted(values)}; "
                "both paths are required because only the stateless one is "
                "unshadowed by the method-state guards"
            )
            ids = set(mark.kwargs.get("ids", ()))
            assert ids == REQUIRED_IDS, (
                f"{name} must label its cases {sorted(REQUIRED_IDS)} so a failure "
                f"names the path it failed on; got {sorted(ids)}"
            )
