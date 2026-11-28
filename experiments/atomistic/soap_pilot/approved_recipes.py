"""Literal, independently reviewed SOAP P3 admission recipe artifact.

The compressed payload is checked-in data, not generated from composer at
import time.  It contains every approved R2 train, target train, and eval
mapping plus the explicitly permitted destination substitutions and binding
expectations.  p3_render compares supplied packet state against this data.
"""

from __future__ import annotations

import base64
import json
import zlib

_APPROVED_RECIPE_ARTIFACT_VERSION = "soap-p3-r2-recipes-v2"
_APPROVED_RECIPE_ARTIFACT_B64 = (
    "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"
    "BqVf/w1OFEpY"
)

APPROVED_RECIPES = json.loads(
    zlib.decompress(base64.b64decode(_APPROVED_RECIPE_ARTIFACT_B64)).decode("utf-8")
)

# This is deliberately literal packet policy, rather than a projection of
# ``command_for_row`` or the P3 renderer.  Caller-owned PBS execution state is
# validated by p3_render against the unchanged wrapper and substituted only at
# admission.  These markers are template-only and are never admissible.
PACKET_EXPECTATIONS = {
    "env": {
        "keys": (
            "MODE", "SOURCE_SHA", "CHECKOUT_ROOT", "UV_CACHE_DIR",
            "RESULTS_ROOT", "SELECTED_ROW", "APPROVAL_ID", "EXPECTED_QUEUE",
        ),
        "MODE_BY_KIND": {"train": "train", "eval": "eval"},
        "UV_CACHE_DIR": "<CALLER_UV_CACHE_DIR>",
        "APPROVAL_ID": "<CALLER_APPROVAL_ID>",
        "EXPECTED_QUEUE": "<CALLER_EXPECTED_QUEUE>",
    },
    "argv": {
        "driver_by_kind": {"train": "train.py", "eval": "eval.py"},
        "prefix": "experiments/atomistic/stage1",
    },
    "output": {
        "log_template": "logs/row-{row_id}-${{PBS_JOBID}}.log",
        "status_template": "logs/row-{row_id}-${{PBS_JOBID}}.status",
    },
}

__all__ = ["APPROVED_RECIPES", "PACKET_EXPECTATIONS"]
