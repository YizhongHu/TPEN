"""Formal R4 source-boundary and port-census controls."""

from __future__ import annotations

import ast
import os
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


def _function_names(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("*.py"):
        if any(part in {".venv", ".git", "__pycache__"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        names.update(node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return names


def test_whole_tree_r1_r2_r3_port_census_has_required_functions() -> None:
    layer = os.environ.get("TPEN_REVIEW_LAYER", "CR")
    assert layer in _PORTS
    missing = sorted(_PORTS[layer] - _function_names(Path(__file__).parents[3]))
    assert not missing, f"missing accepted reviewer functions at {layer}: {missing}"


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
