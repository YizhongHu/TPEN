"""Static contract checks for the backend-neutral B1b2 controller seam."""

from __future__ import annotations

import ast
from pathlib import Path


MODULE = Path(__file__).parents[1] / "src" / "tpen_executor" / "lifecycle" / "controller.py"


MUTANTS = (
    ("drop-job-check", str(MODULE), "if observed_job_id != binding.backend_job_id", "if False", "test_controller_source_has_job_identity_gate"),
    ("drop-prelaunch-recheck", str(MODULE), "store.require_current_controller(group, controller.generation, controller.claimant_nonce)\n            launched", "launched", "test_controller_source_has_prelaunch_recheck"),
    ("successor-fresh-path", str(MODULE), "store.begin_admitted_successor(", "store.begin_attempt(", "test_controller_source_has_admitted_successor_path"),
)


def test_controller_module_compiles_and_exports_run_loop():
    compile(MODULE.read_text(encoding="utf-8"), str(MODULE), "exec")
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    assert any(isinstance(node, ast.FunctionDef) and node.name == "run_group_controller" for node in tree.body)


def test_controller_source_has_job_identity_gate():
    assert "if observed_job_id != binding.backend_job_id" in MODULE.read_text(encoding="utf-8")


def test_controller_source_has_prelaunch_recheck():
    source = MODULE.read_text(encoding="utf-8")
    assert source.count("store.require_current_controller(group, controller.generation, controller.claimant_nonce)") >= 2


def test_controller_source_has_admitted_successor_path():
    source = MODULE.read_text(encoding="utf-8")
    assert "store.begin_admitted_successor(" in source
    assert "store.begin_attempt(" in source
