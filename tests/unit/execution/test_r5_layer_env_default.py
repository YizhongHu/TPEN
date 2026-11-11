"""Formal review round 5 reviewer tests: issue 7 (product tests read the review-process variable TPEN_REVIEW_LAYER, default "CR").

The nested pytest reuses the arm's provisioned environment, read-only and sequentially.
Harness plugin and addopts variables are removed so the nested run cannot write the outer arm's evidence.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _nodeids():
    path = ROOT / "tests/unit/execution/test_formal_r4_source_controls.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    return [f"tests/unit/execution/test_formal_r4_source_controls.py::{name}" for name in (
        "test_r3_13a_corrected_docstrings_are_present",
        "test_docguard_doc_files_parameter_cases_match_layer_ownership",
    ) if name in names]


def _nested(env, tmp_path):
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(name, None)
    nodeids = _nodeids()
    assert nodeids, "observed no layer-gated node ids"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider", f"--basetemp={tmp_path / 'nested-basetemp'}", *nodeids], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    print(proc.stdout)
    print(proc.stderr)
    return proc


# PREDICTED: GREEN at E3 and CR; RED at E4 and E5 (issue 7)
def test_r5_layer_gated_tests_pass_with_default_environment(tmp_path):
    env = os.environ.copy()
    env.pop("TPEN_REVIEW_LAYER", None)
    result = _nested(env, tmp_path)
    assert result.returncode == 0, f"nested returncode={result.returncode}; stdout tail={result.stdout[-3000:]}"


# PREDICTED: GREEN at E3, E4, E5, CR (issue 7 control)
def test_r5_layer_gated_tests_pass_with_declared_layer(tmp_path):
    env = os.environ.copy()
    env["TPEN_REVIEW_LAYER"] = os.environ.get("TPEN_REVIEW_LAYER", "CR")
    result = _nested(env, tmp_path)
    assert result.returncode == 0, f"nested returncode={result.returncode}; stdout tail={result.stdout[-3000:]}"
