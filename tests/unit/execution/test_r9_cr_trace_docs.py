"""R9-5: trace documentation must name the actual proposal hook."""

from __future__ import annotations

import inspect
from pathlib import Path

from tests.helpers.native_chain_resume_trace import install_native_trace
from tpen.execution import chain_resume


def test_r9_trace_docstring_names_actual_proposal_instrumentation_point() -> None:
    # PREDICTED: RED at CR
    doc = inspect.getdoc(install_native_trace) or ""
    assert "``MetropolisSampler.collect_samples`` (proposal)" not in doc
    assert "_propose" in doc


def test_r9_r8_test_comment_no_longer_claims_red_on_current_helper() -> None:
    # PREDICTED: RED at CR
    # Resolve from this file so the leaf does not depend on the pytest CWD.
    repo = Path(__file__).resolve().parents[3]
    text = (repo / "tests/unit/execution/test_r8_sampler_proposal.py").read_text(encoding="utf-8")
    assert "intentionally RED on the current helper" not in text


def test_r9_chain_resume_declaration_names_sampler_proposal_point() -> None:
    # PREDICTED: GREEN (control)
    assert "sampler proposal" in (chain_resume.__doc__ or "")
