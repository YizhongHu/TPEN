"""Tests for the shared, stdlib-only job-name helper."""

from __future__ import annotations

import pytest

from experiments.toolkit.job_names import build_job_name, validate_job_name


def test_build_job_name_uses_sha8_and_lane() -> None:
    assert build_job_name("devsuite", "root", sha="a" * 40) == "devsuite-root-s1-aaaaaaaa"


def test_validate_job_name_enforces_length() -> None:
    with pytest.raises(ValueError, match="too long"):
        validate_job_name("x" * 41)
    validate_job_name("x" * 40)
