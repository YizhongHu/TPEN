"""Review r1 (PR 600) control: the exact-hydrogen core fixture carries no
`reference_energy` leak (part F6/the earlier `reference_energy` removal).

Static check of `tests/integration/artifacts/hydrogen/exact_eval.yaml` --
same recursive-scan shape as
`experiments/atomistic/h-v2/test_no_reference_energy_leak.py`, applied here
because that file's scan is scoped to `experiments/atomistic/h-v2/configs/`
and does not cover this core fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

FIXTURE_PATH = Path(__file__).parents[2] / "integration" / "artifacts" / "hydrogen" / "exact_eval.yaml"


def _find_reference_energy_hits(node: Any, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    hits: list[tuple[str, ...]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "reference_energy":
                hits.append(path + (str(key),))
            if isinstance(key, str) and key.endswith("_target_") and isinstance(value, str) and value.endswith(
                "ReferenceEnergySummary"
            ):
                hits.append(path + (str(key), f"<target:{value}>"))
            hits.extend(_find_reference_energy_hits(value, path + (str(key),)))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            hits.extend(_find_reference_energy_hits(item, path + (str(index),)))
    elif isinstance(node, str) and "reference_energy" in node:
        hits.append(path + (f"<mention:{node}>",))
    return hits


def test_exact_eval_fixture_has_no_reference_energy() -> None:
    document = yaml.safe_load(FIXTURE_PATH.read_text(encoding="utf-8"))
    hits = _find_reference_energy_hits(document)
    assert not hits, f"reference_energy found in {FIXTURE_PATH}: {hits}"


def test_reference_energy_walker_catches_a_reintroduced_key_and_summary() -> None:
    """Mutant control: the walker is not vacuously green on this file's shape."""

    mutated = {
        "evaluator": {
            "tasks": [
                {
                    "summaries": [
                        {"_target_": "tpen.evaluation.summaries.reference_energy.ReferenceEnergySummary"},
                    ]
                }
            ]
        },
        "runtime": {"nested": {"reference_energy": -0.5}},
        "note": "see ${runtime.nested.reference_energy}",
    }
    hits = _find_reference_energy_hits(mutated)
    assert ("runtime", "nested", "reference_energy") in hits
    assert any(hit[-1].startswith("<mention:") for hit in hits)
    assert any(hit[-1].startswith("<target:") and hit[-1].endswith("ReferenceEnergySummary>") for hit in hits)
