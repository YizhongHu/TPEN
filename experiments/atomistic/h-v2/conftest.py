"""Skip the contract-scale H-v2 pipeline test unless explicitly requested.

Round-1 review finding 6: the h2v2_pipeline-marked (now ``h_v2_pipeline``)
contract-scale run adds several minutes to the standard
``pytest -q experiments/atomistic/h-v2 experiments/atomistic/he-v1``
invocation. Verifiers run it once per head, not once per mutant; this
``--run-h-v2-pipeline`` option is how they opt in explicitly, and the default
suite stays fast for everything else.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-h-v2-pipeline",
        action="store_true",
        default=False,
        help="Run the h_v2_pipeline-marked contract-scale H-v2 test(s).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-h-v2-pipeline"):
        return
    skip_marker = pytest.mark.skip(
        reason="h_v2_pipeline contract-scale test skipped by default; pass --run-h-v2-pipeline to run it"
    )
    for item in items:
        if "h_v2_pipeline" in item.keywords:
            item.add_marker(skip_marker)
