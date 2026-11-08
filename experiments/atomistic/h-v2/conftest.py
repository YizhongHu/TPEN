"""Skip opt-in, contract-scale H-v2 tests unless explicitly requested.

Round-1 review finding 6: the h2v2_pipeline-marked (now ``h_v2_pipeline``)
contract-scale run adds several minutes to the standard
``pytest -q experiments/atomistic/h-v2 experiments/atomistic/he-v1``
invocation. Verifiers run it once per head, not once per mutant; this
``--run-h-v2-pipeline`` option is how they opt in explicitly, and the default
suite stays fast for everything else.

``h_v2_sampler_tail_qualification`` (028867dc) follows the same pattern: it
runs four full 4096-walker x 256-draw fresh evaluations (two alphas x two
streams) and gates the real sampler against the exact radial and
local-energy survival curves, so it is opted into separately with
``--run-h-v2-sampler-tail-qualification``.
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
    parser.addoption(
        "--run-h-v2-sampler-tail-qualification",
        action="store_true",
        default=False,
        help="Run the h_v2_sampler_tail_qualification-marked sampler qualification test(s).",
    )


_OPTIONS = {
    "h_v2_pipeline": "--run-h-v2-pipeline",
    "h_v2_sampler_tail_qualification": "--run-h-v2-sampler-tail-qualification",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for marker, option in _OPTIONS.items():
        if config.getoption(option):
            continue
        skip_marker = pytest.mark.skip(
            reason=f"{marker} test skipped by default; pass {option} to run it"
        )
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip_marker)
