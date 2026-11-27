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

_APPROVED_RECIPE_ARTIFACT_VERSION = "soap-p3-r2-recipes-v1"
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
