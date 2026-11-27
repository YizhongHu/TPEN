"""Reject invalid caller state even when the supplied packet agrees with it."""

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.atomistic.soap_pilot import p3_render as p3
from experiments.atomistic.soap_pilot.composer import R2_PROFILES, render_profile


SOURCE = "7b88f833babcb82148c8f975a08b53970838be6a"
ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("route", ("render", "supplied_admission"))
@pytest.mark.parametrize(
    "invalid", ("relative_cache", "missing_cache", "file_cache", "empty_approval",
                "unsafe_approval", "empty_queue", "unsafe_queue"),
)
def test_invalid_execution_state_refuses_matching_packet(
    tmp_path: Path, route: str, invalid: str,
) -> None:
    """Matching expected/actual packet values cannot replace state validation."""

    composed = render_profile(
        R2_PROFILES[1], output_dir=tmp_path / "inputs",
        results_root=str(tmp_path), source_sha=SOURCE, device="cuda",
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    state = {"UV_CACHE_DIR": str(cache), "APPROVAL_ID": "review-approved",
             "EXPECTED_QUEUE": "capacity"}
    args = {"python": "/usr/bin/python3", "checkout_root": str(ROOT)}
    packet = p3.render_row_packet(composed.row, execution_state=state, **args)
    # First prove the row and concrete caller packet are admissible.
    assert p3.admit_row_and_command(
        composed.row, packet=packet, execution_state=state, **args,
    ).packet == packet

    if invalid.endswith("cache"):
        key = "UV_CACHE_DIR"
        if invalid == "relative_cache":
            value = "."  # Existing directory: only the absolute-path rule refuses.
        elif invalid == "missing_cache":
            value = str(tmp_path / "absent-cache")
        else:
            cache_file = tmp_path / "cache-file"
            cache_file.write_text("not a directory", encoding="utf-8")
            value = str(cache_file)
    elif invalid.endswith("approval"):
        key = "APPROVAL_ID"
        value = "" if invalid == "empty_approval" else "approval with spaces"
    else:
        key = "EXPECTED_QUEUE"
        value = "" if invalid == "empty_queue" else "queue with spaces"
    bad_state = {**state, key: value}
    supplied = deepcopy(packet)
    supplied["env"][key] = value
    # No placeholder or packet disagreement can mask the predicate under test.
    with pytest.raises(p3.P3RenderError, match=key):
        if route == "render":
            p3.render_row_packet(composed.row, execution_state=bad_state, **args)
        else:
            p3.admit_row_and_command(
                composed.row, packet=supplied, execution_state=bad_state, **args,
            )
