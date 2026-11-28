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
