"""Formal R4 source control for the E3 accepted polarity documentation."""

from __future__ import annotations

import os
from pathlib import Path


def test_r3_13a_corrected_docstrings_are_present() -> None:
    import pytest

    root = Path(__file__).parents[3]
    layer = os.environ.get("TPEN_REVIEW_LAYER", "CR")
    docs = {
        "E3": ("tests/unit/execution/test_invocation_review_r1.py",),
        "E5": ("tests/unit/execution/test_invocation_review_r1.py",),
        "CR": (
            "tests/unit/execution/test_invocation_review_r1.py",
            "tests/unit/execution/test_chain_resume_review_r1.py",
        ),
    }
    if layer not in docs:
        pytest.skip(f"docstring polarity is not owned by {layer}")
    for relative in docs[layer]:
        source = (root / relative).read_text(encoding="utf-8")
        assert "EXPECTED RED" not in source


def test_docguard_doc_files_parameter_cases_match_layer_ownership() -> None:
    """The parameterized docguard must cover exactly this layer's owned files."""

    import ast
    import pytest

    layer = os.environ.get("TPEN_REVIEW_LAYER", "CR")
    ownership = {
        "E4": {Path("tpen/execution/balsam.py")},
        "E5": {
            Path("tpen/execution/balsam.py"),
            Path("tools/e5b_polaris_app.py"),
            Path("tools/e5b_polaris_pilot.py"),
        },
        "CR": {
            Path("tpen/execution/balsam.py"),
            Path("tools/e5b_polaris_app.py"),
            Path("tools/e5b_polaris_pilot.py"),
        },
    }
    if layer not in ownership:
        pytest.skip(f"no docguard ownership map for layer {layer}")
    root = Path(__file__).parents[3]
    source = root / "tests/unit/execution/test_formal_r1_static.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    assign = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "DOC_FILES"
    )
    actual: set[Path] = set()
    for element in assign.value.elts:
        assert isinstance(element, ast.Call) and element.args
        literal = element.args[0]
        assert isinstance(literal, ast.Constant) and isinstance(literal.value, str)
        actual.add(Path(literal.value))
    assert actual == ownership[layer]


def test_r3_13b_e5_no_backend_call_assertions_are_present() -> None:
    """Each E5 scenario must assert that no backend call was made."""

    import ast

    root = Path(__file__).parents[3]
    path = root / "tests/unit/execution/test_formal_r2_e5.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scenario_names = (
        "test_native_prefix_publishes_evidence_before_bounded_walltime_trap",
        "test_native_prefix_nested_run_dir_resolves_full_checkpoint_control",
        "test_prior_initial_start_without_evidence_parks_before_science",
        "test_first_ever_start_without_evidence_invokes_initial_once",
    )
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in scenario_names:
        node = functions.get(name)
        assert node is not None, f"missing E5 scenario function: {name}"
        assert any(
            isinstance(assertion, ast.Assert)
            and isinstance(assertion.test, ast.Compare)
            and isinstance(assertion.test.left, ast.Name)
            and assertion.test.left.id == "client_calls"
            and len(assertion.test.ops) == 1
            and isinstance(assertion.test.ops[0], ast.Eq)
            and len(assertion.test.comparators) == 1
            and isinstance(assertion.test.comparators[0], ast.List)
            and not assertion.test.comparators[0].elts
            for assertion in ast.walk(node)
        ), f"scenario lacks its own client_calls == [] assertion: {name}"


def test_r3_2_trainer_cursor_unaffected_assertions_still_tautological() -> None:
    """The accepted cursor fix must observe the actual unaffected cursor."""

    import ast

    root = Path(__file__).parents[3]
    path = root / "tests/integration/test_native_checkpoint_recovery.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_real_restore_omission_channels_are_observed"
    )

    def is_cursor_recompute(stmt: ast.AST) -> bool:
        return (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Subscript)
            and isinstance(stmt.targets[0].value, ast.Name)
            and stmt.targets[0].value.id == "result"
            and isinstance(stmt.targets[0].slice, ast.Constant)
            and stmt.targets[0].slice.value == "trainer_cursor"
        )

    limbs = {"optimizer", "model", "sampler_mcmc", "process_rng"}
    tautological: list[str] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "limb"
            and len(test.ops) == 1
            and isinstance(test.comparators[0], ast.Constant)
        ):
            continue
        limb = test.comparators[0].value
        if limb not in limbs:
            continue
        references_cursor = any(
            isinstance(sub, ast.Constant) and sub.value == "trainer_cursor"
            for sub in ast.walk(node)
        )
        recomputes_cursor = any(is_cursor_recompute(stmt) for stmt in ast.walk(node))
        if references_cursor and not recomputes_cursor:
            tautological.append(limb)
    assert not tautological, f"tautological trainer_cursor assertions: {tautological}"
