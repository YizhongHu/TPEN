"""Formal R4 source-boundary and port-census controls."""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

from tpen.checkpoint import CheckpointRef
from tpen.checkpoint.payload import TrainResume
from tpen.execution import (
    CheckpointEvidence,
    RecoveryOutcome,
    RecoveryPhase,
    RefusalCode,
    select_recovery,
)
from tests.unit.execution.test_recovery import _checkpoint, _intent


def test_initial_checkpoint_ref_form_is_not_auto_bound(tmp_path: Path) -> None:
    intent = _intent()
    checkpoint = _checkpoint(tmp_path, payload=TrainResume().to_manifest())
    decision = select_recovery(intent, phase=RecoveryPhase.INITIAL, evidence=CheckpointRef.from_directory(checkpoint))
    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPATIBLE


def test_initial_checkpoint_path_form_is_not_auto_bound(tmp_path: Path) -> None:
    intent = _intent()
    checkpoint = _checkpoint(tmp_path, payload=TrainResume().to_manifest())
    decision = select_recovery(intent, phase=RecoveryPhase.INITIAL, evidence=str(checkpoint))
    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPATIBLE


def test_initial_durable_mapping_remains_the_green_bound_form(tmp_path: Path) -> None:
    intent = _intent()
    checkpoint = _checkpoint(tmp_path, payload=TrainResume().to_manifest())
    evidence = CheckpointEvidence.from_ref(
        CheckpointRef.from_directory(checkpoint),
        science_input_digest=intent.science_input_digest,
        target_iterations=intent.target_iterations,
    ).to_dict()
    decision = select_recovery(intent, phase=RecoveryPhase.INITIAL, evidence=evidence)
    assert decision.outcome is RecoveryOutcome.INITIAL_CHECKPOINT


@pytest.mark.parametrize("field", ("logical_task_id", "fresh_output_id", "result_dir"))
def test_v2_row_fields_are_individually_strings(field: str) -> None:
    from experiments.toolkit.execution_v2 import ready_batch_v2

    row = {
        "invocation_id": "invocation",
        "logical_task_id": "logical",
        "fresh_output_id": "output",
        "argv": ("worker",),
        "result_dir": "results",
    }
    row[field] = ["wrong-type"]
    with pytest.raises(ValueError, match=field):
        ready_batch_v2(
            lineage_id="lineage",
            batch_id="batch",
            source_config_digest="source",
            science_input_digest="science",
            target_iterations=2,
            backend="direct",
            recovery="none",
            payload={"science": {}},
            rows=[row],
        )


def test_v2_row_with_all_string_fields_is_accepted() -> None:
    from experiments.toolkit.execution_v2 import ready_batch_v2

    batch = ready_batch_v2(
        lineage_id="lineage",
        batch_id="batch",
        source_config_digest="source",
        science_input_digest="science",
        target_iterations=2,
        backend="direct",
        recovery="none",
        payload={"science": {}},
        rows=[
            {
                "invocation_id": "invocation",
                "logical_task_id": "logical",
                "fresh_output_id": "output",
                "argv": ("worker",),
                "result_dir": "results",
            }
        ],
    )
    assert batch["items"][0]["logical_task_id"] == "logical"


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
    """Enumerate tracked ``*.py`` paths only; an untracked stray must not pad the census."""

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
        names.update(node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return names, errors


def test_whole_tree_r1_r2_r3_port_census_has_required_functions() -> None:
    layer = os.environ.get("TPEN_REVIEW_LAYER", "CR")
    assert layer in _PORTS
    names, errors = _function_names(Path(__file__).parents[3])
    # Fail closed: a file that cannot be parsed must not be silently excluded
    # from the census, since that would let a missing function hide behind
    # an unrelated syntax error.
    assert not errors, f"tracked-tree AST parse failures (fail-closed): {errors}"
    missing = sorted(_PORTS[layer] - names)
    assert not missing, f"missing accepted reviewer functions at {layer}: {missing}"


_DOC_FILE_OWNERSHIP = {
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


def test_docguard_doc_files_parameter_cases_match_layer_ownership() -> None:
    """E4 owns only balsam.py for the no-local-provenance docguard; E5 adds the app/pilot tool files.

    The function name ``test_review_docstrings_contain_no_local_provenance``
    already exists at E4, so a name-only census cannot see this ownership
    split; this control inspects the ``DOC_FILES`` parametrize tuple itself.
    """

    layer = os.environ.get("TPEN_REVIEW_LAYER", "CR")
    if layer not in _DOC_FILE_OWNERSHIP:
        pytest.skip(f"no docguard ownership map for layer {layer}")
    root = Path(__file__).parents[3]
    path = root / "tests/unit/execution/test_formal_r1_static.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
    assert actual == _DOC_FILE_OWNERSHIP[layer]


def test_r3_2_trainer_cursor_unaffected_assertions_still_tautological() -> None:
    """Source-only B3 residual control.

    ``result["trainer_cursor"]`` is produced once, generically, by comparing
    ``first_subsequent_event(...).value`` between the reference and disabled
    trace at ``resume_step=resume_cursor - 1``.  For the cursor's own channel
    that value IS the step number, so the comparison is true by construction
    regardless of whether the cursor was actually observed as unaffected.
    Only the ``limb == "trainer_cursor"`` branch recomputes a non-tautological
    cursor result; the other branches (optimizer, model, sampler_mcmc,
    process_rng) still assert the generic, tautological value. Disposition 2
    calls this a residual, not a full fix; this control is expected RED until
    that disposition is applied upstream. The direct-first cursor test suite
    (fix 2) and the real omission/positive/naming fixes remain valid GREEN
    controls and are untouched here.
    """

    root = Path(__file__).parents[3]
    path = root / "tests/integration/test_native_checkpoint_recovery.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_real_restore_omission_channels_are_observed"
    )

    def _is_cursor_recompute(stmt: ast.AST) -> bool:
        return (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Subscript)
            and isinstance(stmt.targets[0].value, ast.Name)
            and stmt.targets[0].value.id == "result"
        )

    non_cursor_limbs = {"optimizer", "model", "sampler_mcmc", "process_rng"}
    tautological_branches: list[str] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "limb"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)
        ):
            continue
        limb_literal = test.comparators[0].value
        if limb_literal not in non_cursor_limbs:
            continue
        references_cursor = any(
            isinstance(sub, ast.Constant) and sub.value == "trainer_cursor" for sub in ast.walk(node)
        )
        recomputes_cursor = any(_is_cursor_recompute(stmt) for stmt in ast.walk(node))
        if references_cursor and not recomputes_cursor:
            tautological_branches.append(limb_literal)

    assert not tautological_branches, (
        "still-tautological trainer_cursor assertions in branches: "
        f"{tautological_branches}"
    )


def test_r3_13a_corrected_docstrings_are_present() -> None:
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
    assert layer in docs
    # This is a source-level r3-13a control.  It intentionally goes RED on
    # the current owner/CR stale expected-RED prose; no pytest execution of
    # the prose itself is claimed.  E3 has no chain_resume reviewer file.
    for relative in docs[layer]:
        source = (root / relative).read_text(encoding="utf-8")
        assert "EXPECTED RED" not in source


def test_r3_13b_e5_no_backend_call_assertions_are_present() -> None:
    root = Path(__file__).parents[3]
    e5 = (root / "tests/unit/execution/test_formal_r2_e5.py").read_text(encoding="utf-8")
    tree = ast.parse(e5, filename="tests/unit/execution/test_formal_r2_e5.py")
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


def test_issue5_executed_arms_derivation_is_not_a_fixed_declaration() -> None:
    """R3 issue 5, expected GREEN: source-only derivation-sensitivity oracle.

    ``executed_arms`` must be derived from ``real_channel_results`` (the
    actual measured per-channel outcome mapping) and ``unmeasured_arms``
    must be derived by comparing the expected channel set against
    ``executed_arms``, rather than both being independently fixed
    declarations that happen to agree by construction. This is a
    source-only AST check of the derivation shape in
    ``test_native_e1_to_e4_recovery_parity_divergence_and_receipt`` -- it
    establishes source sensitivity only, not a native science verdict.
    """

    root = Path(__file__).parents[3]
    path = root / "tests/integration/test_native_checkpoint_recovery.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    target_function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "test_native_e1_to_e4_recovery_parity_divergence_and_receipt"
        ),
        None,
    )
    assert target_function is not None, "target function not found"
    assigns: dict[str, list[ast.Assign]] = {}
    for node in ast.walk(target_function):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assigns.setdefault(node.targets[0].id, []).append(node)
    executed_assigns = assigns.get("executed_arms", [])
    unmeasured_assigns = assigns.get("unmeasured_arms", [])
    assert len(executed_assigns) == 1, "expected exactly one executed_arms assignment"
    assert len(unmeasured_assigns) == 1, "expected exactly one unmeasured_arms assignment"
    executed_source = ast.dump(executed_assigns[0].value)
    unmeasured_source = ast.dump(unmeasured_assigns[0].value)
    assert "real_channel_results" in executed_source, (
        "executed_arms is not derived from real_channel_results (the real measured outcome)"
    )
    assert "executed_arms" in unmeasured_source, (
        "unmeasured_arms is not derived by comparing against executed_arms"
    )
