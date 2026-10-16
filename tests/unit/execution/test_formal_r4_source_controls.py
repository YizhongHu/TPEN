"""Formal R4 source control for the E3 accepted polarity documentation."""

from __future__ import annotations

from pathlib import Path


def test_r3_13a_corrected_docstrings_are_present() -> None:
    root = Path(__file__).parents[3]
    source = (root / "tests/unit/execution/test_invocation_review_r1.py").read_text(
        encoding="utf-8"
    )
    assert "EXPECTED RED" not in source


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
