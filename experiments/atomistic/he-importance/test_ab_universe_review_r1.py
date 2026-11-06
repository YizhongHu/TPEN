"""PR #572 AB-4a pre-review round 1 independent renderer tests."""

from __future__ import annotations

import copy
import importlib.util
import json
from itertools import product
from pathlib import Path
import sys
from typing import Any

from omegaconf import OmegaConf
import pytest


HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stage_coordinate = _load("he_importance_stage_coordinate_review_r1", "stage_coordinate.py")
ab_universe = _load("he_importance_ab_universe_review_r1", "ab_universe.py")
context_mapping = _load("he_importance_context_mapping_review_r1", "context_mapping.py")
pr_tests = _load("he_importance_ab_universe_pr_test_review_r1", "test_ab_universe.py")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _configuration(**values: str) -> dict[str, str]:
    result = {
        "channels": "(32,32)", "embedding": "(128,1)",
        "producer_initialization": ".50", "electron_electron": "1",
        "electron_nucleus": "(.01,1)", "readout_initialization": "equal",
    }
    result.update(values)
    return result


PRODUCERS = ("tensor", "linear/coordinate-neighbor", "linear/orbit-complete",
             "hybrid/coordinate-neighbor", "hybrid/orbit-complete")
ACTIVATIONS = ("SiLU,SiLU", "tanh,tanh", "exact GELU,exact GELU",
               "Softplus beta2,Softplus beta2", "channel-MLP,SiLU",
               "SiLU,channel-MLP", "channel-MLP,channel-MLP")
UPDATES = ("residual", "replacement", "Gaussian residual")
CHANNELS = ("(16,16)", "(24,24)", "(32,32)", "(48,48)", "(40,24)", "(24,40)")
EMBEDDINGS = ("(64,1)", "(128,1)", "(128,2)", "(256,2)")
A9 = (".25", ".50", "1.0")
A10 = (".25", "1", "4")
A11 = ("(.0025,.25)", "(.01,1)", "(.04,4)")
A12 = ("equal", "signed L1", "random-direction L2")


def _blocks() -> tuple[list[str], list[str], list[str]]:
    core = [_canonical(_configuration(producer_path=p, activation=a, update=u))
            for p, a, u in product(PRODUCERS, ACTIVATIONS, UPDATES)]
    capacity = [_canonical(_configuration(producer_path="tensor", activation="SiLU,SiLU",
                                           channels=c, embedding=e, update=u))
                for c, e, u in product(CHANNELS, EMBEDDINGS, UPDATES)]
    initialization = [_canonical(_configuration(
        producer_path="tensor", activation="SiLU,SiLU", producer_initialization=i9,
        electron_electron=i10, electron_nucleus=i11, readout_initialization=i12,
        update="residual")) for i9, i10, i11, i12 in product(A9, A10, A11, A12)]
    return core, capacity, initialization


def _identities(packages: Any) -> set[str]:
    return {_canonical(package["literal_configuration"]) for package in packages}


def test_design_duplicate_structure() -> None:
    core, capacity, initialization = _blocks()
    core_set, capacity_set, init_set = map(set, (core, capacity, initialization))
    assert (len(core), len(capacity), len(initialization)) == (105, 72, 81)
    assert len(core_set & capacity_set) == 3
    assert len(init_set & (core_set | capacity_set)) == 1
    assert len(core_set & capacity_set & init_set) == 1
    assert len(core_set | capacity_set | init_set) == 254
    rendered = ab_universe.render_a_packages()
    assert _identities(rendered) == core_set | capacity_set | init_set
    shared = (core_set & capacity_set) | (init_set & (core_set | capacity_set))
    assert all(sum(_canonical(p["literal_configuration"]) == item for p in rendered) == 1
               for item in shared)
    assert {p["display_name"] for p in rendered if _canonical(p["literal_configuration"]) in shared} <= {
        "A001", "A002", "A003"}
    assert len({p["display_name"] for p in rendered}) == 254
    b = ab_universe.render_b_packages()
    assert {p["display_name"] for p in b} == {f"B{i:03d}" for i in range(1, 61)}
    # Pin the current display-name gaps so any renumbering is an explicit decision.
    assert {f"A{i:03d}" for i in range(1, 259)} - {p["display_name"] for p in rendered} == {
        "A133", "A134", "A135", "A217"}


def test_undeclared_level_added_row_is_detected_as_extra() -> None:
    control = OmegaConf.load(HERE / "configs" / "train.yaml")
    mutant = copy.deepcopy(ab_universe.render_a_packages()[0])
    mutant["literal_configuration"]["activation"] = "undeclared activation"
    mutant["display_name"] = "A-extra"
    packages = list(ab_universe.render_a_packages()) + [mutant]
    assert _identities(packages) > set(pr_tests._expected_a())
    with pytest.raises(context_mapping.UnknownContextLabel):
        ab_universe.compose_package_model(control.model, mutant, context_mapping)


@pytest.mark.parametrize("mutation", ("available", "removed", "relabelled", "reason"))
def test_o_star_mutant_variants(mutation: str) -> None:
    packages = list(ab_universe.render_a_packages())
    pr_tests._assert_o_star_honest(packages)
    mutant = copy.deepcopy(packages[0])
    if mutation == "available":
        mutant["optimizer_slots"][1]["status"] = "available"
        mutant["optimizer_slots"][1]["levels"] = copy.deepcopy(mutant["optimizer_slots"][0]["levels"])
    elif mutation == "removed":
        mutant["optimizer_slots"] = mutant["optimizer_slots"][:1]
    elif mutation == "relabelled":
        mutant["optimizer_slots"][1]["method"] = "adam"
    else:
        mutant["optimizer_slots"][1]["unavailable_reason"] = "pending"
    with pytest.raises(AssertionError):
        pr_tests._assert_o_star_honest((mutant,))


def test_adam_control_levels_match_authority() -> None:
    authority = json.loads((HERE / "optimizer_grid_authority.json").read_text())
    adam = next(f for f in authority["families"] if f["family"] == "adam")
    expected = {**adam["fixed"], "lr": 0.005, "beta2": 0.999}
    assert 0.005 in adam["varied"]["lr"] and 0.999 in adam["varied"]["beta2"]
    assert set(expected) == set(adam["fixed"]) | set(adam["varied"])
    for package in (*ab_universe.render_a_packages(), *ab_universe.render_b_packages()):
        assert package["optimizer_slots"][0]["levels"] == expected


def test_literal_configuration_shape_fits_ab4b_closed_context() -> None:
    base = {"producer_path", "activation", "channels", "embedding", "update",
            "producer_initialization", "electron_electron", "electron_nucleus",
            "readout_initialization"}
    a, b = ab_universe.render_ab_universe().values()
    assert all(set(p["literal_configuration"]) == base and
               all(isinstance(v, str) for v in p["literal_configuration"].values()) for p in a)
    assert all(set(p["literal_configuration"]) == base | {"package"} for p in b)
    assert _identities(a).isdisjoint(_identities(b))
    for producer, update in product(PRODUCERS, UPDATES):
        rows = [p for p in b if p["literal_configuration"]["producer_path"] == producer
                and p["literal_configuration"]["update"] == update]
        assert {p["literal_configuration"]["package"] for p in rows} == {"B00", "B10", "B01", "B11"}


def test_b_pairing_key_is_not_vacuous() -> None:
    rows = list(ab_universe.render_b_packages())
    groups = {}
    for row in rows:
        groups.setdefault(_canonical(row["model_initialization_stream_key"]), []).append(row)
    assert len(groups) == 15
    assert all(len(v) == 4 and len({r["literal_configuration"]["package"] for r in v}) == 4
               for v in groups.values())
    assert all(set(json.loads(k)) == {"stage", "producer_path", "update"} for k in groups)
    mutant = copy.deepcopy(rows)
    for row in mutant:
        row["model_initialization_stream_key"] = {"constant": True}
    # The PR's test_seed_namespaces_and_b_pairing_key would accept this constant-key mutant.
    with pytest.raises(AssertionError):
        assert len({_canonical(r["model_initialization_stream_key"]) for r in mutant}) == 15


def test_seed_labels_bound_to_stage_coordinate() -> None:
    universe = ab_universe.render_ab_universe()
    assert all(p["seed_labels"] == list(stage_coordinate.seed_labels("A")) for p in universe["A"])
    assert all(p["seed_labels"] == list(stage_coordinate.seed_labels("B")) for p in universe["B"])
    assert set(stage_coordinate.seed_labels("A")).isdisjoint(stage_coordinate.seed_labels("B"))
    for axis in ("A", "B"):
        for other in ("Q", "O1", "O2"):
            assert set(stage_coordinate.seed_labels(axis)).isdisjoint(stage_coordinate.seed_labels(other))


def test_renderer_is_pure() -> None:
    first, second = ab_universe.render_ab_universe(), ab_universe.render_ab_universe()
    assert first == second
    for axis in ("A", "B"):
        for key in ("literal_configuration", "optimizer_slots", "seed_labels", "model_initialization_stream_key"):
            assert id(first[axis][0][key]) != id(second[axis][0][key])
    first["A"][0]["literal_configuration"]["activation"] = "mutated"
    first["A"][0]["seed_labels"].append(1)
    assert second["A"][0]["literal_configuration"]["activation"] == "SiLU,SiLU"
    assert second["A"][0]["seed_labels"] == list(stage_coordinate.seed_labels("A"))
