"""Formal R4 source control for the E3 accepted polarity documentation."""

from __future__ import annotations

import ast
from pathlib import Path


def test_r3_2_trainer_cursor_unaffected_assertions_are_observed() -> None:
    """Every non-cursor omission branch recomputes the actual cursor event."""

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
            and isinstance(stmt.value, ast.Compare)
            and isinstance(stmt.value.left, ast.Name)
            and stmt.value.left.id == "disabled_first_cursor"
            and len(stmt.value.ops) == 1
            and isinstance(stmt.value.ops[0], ast.Eq)
            and len(stmt.value.comparators) == 1
            and isinstance(stmt.value.comparators[0], ast.Name)
            and stmt.value.comparators[0].id == "resume_cursor"
        )

    non_cursor_limbs = {"optimizer", "model", "sampler_mcmc", "process_rng"}
    missing: list[str] = []
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
        if limb not in non_cursor_limbs:
            continue
        references_cursor = any(
            isinstance(sub, ast.Constant) and sub.value == "trainer_cursor"
            for sub in ast.walk(node)
        )
        if references_cursor and not any(is_cursor_recompute(stmt) for stmt in ast.walk(node)):
            missing.append(limb)
    assert not missing, f"unobserved trainer_cursor branches: {missing}"


def test_r3_13a_corrected_docstrings_are_present() -> None:
    root = Path(__file__).parents[3]
    docs = ["tests/unit/execution/test_invocation_review_r1.py"]
    chain_review = "tests/unit/execution/test_chain_resume_review_r1.py"
    if (root / chain_review).is_file():
        docs.append(chain_review)
    for relative in docs:
        source = (root / relative).read_text(encoding="utf-8")
        assert "EXPECTED RED" not in source


def test_docguard_doc_files_parameter_cases_match_layer_ownership() -> None:
    """The parameterized docguard must cover exactly this layer's owned files."""

    import ast
    root = Path(__file__).parents[3]
    # The checked-out layer owns the files it contains; review environment
    # variables must not change the product test's coverage or expectations.
    ownership = {
        path for path in (
            Path("tpen/execution/balsam.py"),
            Path("tools/e5b_polaris_app.py"),
            Path("tools/e5b_polaris_pilot.py"),
        ) if (root / path).is_file()
    }
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
    assert actual == ownership


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
    import pytest

    root = Path(__file__).parents[3]
    path = root / "tests/integration/test_native_checkpoint_recovery.py"
    if not path.exists():
        pytest.skip("native integration owner is not present on this layer")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    try:
        func = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "test_real_restore_omission_channels_are_observed"
        )
    except StopIteration:
        pytest.skip("trainer-cursor control is not owned by this layer")

    def is_cursor_recompute(stmt: ast.AST) -> bool:
        return (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Subscript)
            and isinstance(stmt.targets[0].value, ast.Name)
            and stmt.targets[0].value.id == "result"
            and isinstance(stmt.targets[0].slice, ast.Constant)
            and stmt.targets[0].slice.value == "trainer_cursor"
            and isinstance(stmt.value, ast.Compare)
            and isinstance(stmt.value.left, ast.Name)
            and stmt.value.left.id == "disabled_first_cursor"
            and len(stmt.value.ops) == 1
            and isinstance(stmt.value.ops[0], ast.Eq)
            and len(stmt.value.comparators) == 1
            and isinstance(stmt.value.comparators[0], ast.Name)
            and stmt.value.comparators[0].id == "resume_cursor"
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


_PORTS = {
    "E2": {"test_unknown_fields_are_refused_at_all_nested_wire_levels"},
    "E3": {
        "test_all_refusal_codes_have_explicit_outcomes",
        "test_incompatible_refusal_code_survives_receipt_and_disk",
    },
    "E4": {
        "test_refusal_precedence_is_shared_by_submitter_and_worker",
        "test_review_docstrings_contain_no_local_provenance",
    },
    "E5": {
        "test_runner_json_writer_syncs_file_before_replace_and_directory_after",
        "test_app_pre_e3_writer_syncs_file_before_replace_and_directory_after",
        "test_review_docstrings_contain_no_local_provenance",
    },
    "CR": {
        "test_refusal_precedence_is_shared_by_submitter_and_worker",
        "test_unknown_fields_are_refused_at_all_nested_wire_levels",
        "test_all_refusal_codes_have_explicit_outcomes",
        "test_incompatible_refusal_code_survives_receipt_and_disk",
        "test_runner_json_writer_syncs_file_before_replace_and_directory_after",
        "test_app_pre_e3_writer_syncs_file_before_replace_and_directory_after",
        "test_review_docstrings_contain_no_local_provenance",
    },
}


def _tracked_python_files(root: Path) -> list[Path]:
    import subprocess

    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / entry for entry in completed.stdout.decode("utf-8").split("\0") if entry]


def _function_names(root: Path) -> tuple[set[str], list[str]]:
    names: set[str] = set()
    errors: list[str] = []
    for path in _tracked_python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        names.update(
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    return names, errors


def test_whole_tree_r1_r2_r3_port_census_has_required_functions() -> None:
    root = Path(__file__).parents[3]
    layer = "CR" if (root / "tpen/execution/chain_resume.py").is_file() else "E5"
    assert layer in _PORTS
    names, errors = _function_names(Path(__file__).parents[3])
    assert not errors, f"tracked-tree AST parse failures (fail-closed): {errors}"
    missing = sorted(_PORTS[layer] - names)
    assert not missing, f"missing accepted reviewer functions at {layer}: {missing}"


def test_issue5_executed_arms_derivation_is_not_a_fixed_declaration() -> None:
    """Executed and unmeasured arms must be derived from real measured results."""

    root = Path(__file__).parents[3]
    path = root / "tests/integration/test_native_checkpoint_recovery.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_native_e1_to_e4_recovery_parity_divergence_and_receipt"
    )
    assignments: dict[str, list[ast.Assign]] = {}
    for node in ast.walk(target):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            assignments.setdefault(node.targets[0].id, []).append(node)
    executed = assignments.get("executed_arms", [])
    unmeasured = assignments.get("unmeasured_arms", [])
    assert len(executed) == 1
    assert len(unmeasured) == 1
    assert "real_channel_results" in ast.dump(executed[0].value)
    assert "executed_arms" in ast.dump(unmeasured[0].value)


def _target_before_root_assertion() -> ast.Assert:
    root = Path(__file__).parents[3]
    path = root / "tests/integration/test_native_checkpoint_recovery.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "test_real_checkpoint_boundary_faults_are_typed"
    )
    assertions = [
        node
        for node in ast.walk(target)
        if isinstance(node, ast.Assert) and "before_root" in ast.dump(node)
    ]
    assert len(assertions) == 1
    return assertions[0]


def _before_root_predicate(before_root: Path) -> bool:
    from tpen.checkpoint import list_complete_checkpoints

    expr_source = ast.unparse(_target_before_root_assertion().test)
    namespace = {"before_root": before_root, "list_complete_checkpoints": list_complete_checkpoints}
    return eval(expr_source, namespace)  # noqa: S307 -- trusted repository source


def test_issue6_before_root_predicate_accepts_staged(tmp_path: Path) -> None:
    staged_root = tmp_path / "staged"
    staged_dir = staged_root / "checkpoints" / "step_000001.tmp"
    staged_dir.mkdir(parents=True)
    (staged_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (staged_dir / "COMPLETE").write_text("", encoding="utf-8")
    assert _before_root_predicate(staged_root) is True


def test_issue6_before_root_predicate_rejects_committed(tmp_path: Path) -> None:
    committed_root = tmp_path / "committed"
    committed_dir = committed_root / "checkpoints" / "step_000001"
    committed_dir.mkdir(parents=True)
    (committed_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (committed_dir / "COMPLETE").write_text("", encoding="utf-8")
    assert _before_root_predicate(committed_root) is False
