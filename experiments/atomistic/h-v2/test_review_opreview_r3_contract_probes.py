"""Independent operator-review probes for the round-3 contract delta."""

import json
import re
from pathlib import Path


INPUTS = Path(__file__).with_name("contract_inputs.json")


def load_inputs() -> dict:
    with INPUTS.open(encoding="utf-8") as stream:
        return json.load(stream)


def test_t6_capacity_pilot_has_three_checkpoint_triples_and_distinct_seeds() -> None:
    inputs = load_inputs()
    roster = inputs["measurement"]["seed_roster"]
    pilot_eval = roster["non_science_seed_namespaces"]["capacity_pilot"]["evaluation"]
    streams = inputs["measurement"]["fresh_evaluation"]["streams_per_checkpoint"]
    needed = streams * 3

    assert len(pilot_eval) >= needed, (
        "capacity-pilot evaluation roster is too short for three checkpoint triples: "
        f"len(pilot_eval)={len(pilot_eval)}, streams={streams}, needed={needed}"
    )
    assert len(pilot_eval) == len(set(pilot_eval)), (
        "capacity-pilot evaluation seeds are not pairwise distinct: "
        f"pilot_eval={pilot_eval!r}"
    )


def test_t7_pilot_placement_and_update_ceiling_match() -> None:
    inputs = load_inputs()
    pilot = inputs["admission"]["pilot"]
    placements = inputs["admission"]["pipeline_check"]["placements"]
    parts = inputs["admission"]["pipeline_check"]["parts"]

    slice_match = re.search(r"([0-9a-f]{8})", pilot["implemented_by"])
    assert slice_match is not None, (
        "pilot implemented_by does not contain an 8-hex slice id: "
        f"{pilot['implemented_by']!r}"
    )
    slice_id = slice_match.group(1)
    pilot_facility = pilot["facility"]
    expected_facility = next(
        token for token in ("Polaris", "Cannon") if token in pilot_facility
    )

    for placement_name, placement in placements.items():
        if slice_id not in placement:
            continue
        named_facilities = [
            token for token in ("Polaris", "Cannon") if token in placement
        ]
        assert named_facilities == [expected_facility], (
            "pilot placement names a different facility: "
            f"slice_id={slice_id}, placement={placement_name!r}, "
            f"placement_text={placement!r}, pilot_facility={pilot_facility!r}"
        )

    if slice_id in placements["g"]:
        parts_g = next(part for part in parts if part["id"] == "g")
        updates_max = parts_g["pass_criteria"]["updates_max"]
        assert updates_max >= pilot["length_updates_min"], (
            "pilot update minimum exceeds pipeline-check part g ceiling: "
            f"updates_max={updates_max}, length_updates_min={pilot['length_updates_min']}"
        )
