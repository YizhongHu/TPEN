"""Forbid `reference_energy` anywhere in H-v2's run configs.

Operator inline comment on PR 590 `eval.yaml:27`: no reference energy may
reach training or logging. H-v2's exact ground-state energy is a test
oracle (compared after a run, from `contract_inputs.json` analytic
controls or the independent oracle), never a runtime input a config may
carry into a sampler, trainer, or summary. `ReferenceEnergySummary`
(`tpen.evaluation.summaries.reference_energy`) is the concrete leak this
review caught: it requires a `reference_energy` constructor argument and
logs it, plus the derived `energy_error`, straight into `metrics.jsonl`.

This test recursively scans every YAML config under this directory's
`configs/` folder -- not just the ones named in the triggering review
comment -- for a `reference_energy` mapping key at any nesting depth, and
for any string value (a Hydra interpolation) that mentions it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).parent
CONFIGS_DIR = HERE / "configs"


def _find_reference_energy_paths(node: Any, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Return every path at which a `reference_energy` key or mention occurs."""

    hits: list[tuple[str, ...]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "reference_energy":
                hits.append(path + (str(key),))
            hits.extend(_find_reference_energy_paths(value, path + (str(key),)))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            hits.extend(_find_reference_energy_paths(item, path + (str(index),)))
    elif isinstance(node, str) and "reference_energy" in node:
        hits.append(path + (f"<interpolation:{node}>",))
    return hits


def test_no_h_v2_config_names_a_reference_energy() -> None:
    configs = sorted(CONFIGS_DIR.glob("*.yaml"))
    assert configs, f"no configs found under {CONFIGS_DIR} -- swept corpus looks wrong"
    violations: dict[str, list[tuple[str, ...]]] = {}
    for path in configs:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        hits = _find_reference_energy_paths(document)
        if hits:
            violations[path.name] = hits
    assert not violations, f"reference_energy found in H-v2 configs: {violations}"


def test_reference_energy_detector_catches_a_reintroduced_key() -> None:
    """Mutant control: prove the scanner above is not vacuously green.

    Simulates the exact regression this test guards against -- a
    `reference_energy` key reappearing deep inside a config -- so a future
    change to the walker itself (e.g. one that stops recursing into nested
    mappings) would be caught here rather than silently never firing.
    """

    mutated = {"system": {"nested": {"reference_energy": -0.5}}, "note": "see ${system.reference_energy}"}
    hits = _find_reference_energy_paths(mutated)
    assert ("system", "nested", "reference_energy") in hits
    assert any(hit[-1].startswith("<interpolation:") for hit in hits)
