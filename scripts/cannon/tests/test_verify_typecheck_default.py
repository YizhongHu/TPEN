"""Static contract tests for verify.sh's type-checking mode selection.

The full runner requires Slurm, an exact-SHA clone, and a prepared shared
environment, so this test checks the local argument-building contract without
network or cluster access.
"""

from __future__ import annotations

import re
from pathlib import Path


VERIFY = Path(__file__).resolve().parents[3] / "scripts" / "cannon" / "verify.sh"


def test_typecheck_defaults_on_and_off_is_explicit() -> None:
    source = VERIFY.read_text()

    assert 'TYPECHECK="${TYPECHECK:-on}"' in source
    assert 'TYPECHECK="${TYPECHECK:-off}"' not in source

    pytest_args = re.search(
        r"pytest_args\(\) \{(?P<body>.*?)\n\}", source, re.DOTALL
    )
    assert pytest_args is not None
    body = pytest_args.group("body")
    assert '[[ "$typecheck" == off ]] && args+=(-o addopts=)' in body
    assert '[[ "$typecheck" == on ]] && args+=(-o addopts=)' not in body

    baseline = re.search(
        r"run_pytest baseline \"\$TYPECHECK\" \"\$PARALLEL\"", source
    )
    assert baseline is not None
