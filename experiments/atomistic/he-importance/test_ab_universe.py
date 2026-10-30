"""Independent acceptance tests for the section-2.8 A/B package renderer."""

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


_HERE = Path(__file__).resolve().parent


def _load(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ab_universe = _load("he_importance_ab_universe_test", "ab_universe.py")
context_mapping = _load("he_importance_context_mapping_for_ab_test", "context_mapping.py")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _configuration(**values: str) -> dict[str, str]:
    result = {
        "channels": "(32,32)",
        "embedding": "(128,1)",
        "producer_initialization": ".50",
        "electron_electron": "1",
        "electron_nucleus": "(.01,1)",
        "readout_initialization": "equal",
    }
    result.update(values)
    return result


# These are deliberately retyped from section 2.8, rather than imported from
# the renderer or its label tables.
_PRODUCERS = (
    "tensor", "linear/coordinate-neighbor", "linear/orbit-complete",
    "hybrid/coordinate-neighbor", "hybrid/orbit-complete",
)
_ACTIVATIONS = (
    "SiLU,SiLU", "tanh,tanh", "exact GELU,exact GELU",
    "Softplus beta2,Softplus beta2", "channel-MLP,SiLU",
    "SiLU,channel-MLP", "channel-MLP,channel-MLP",
)
_UPDATES = ("residual", "replacement", "Gaussian residual")
_CHANNELS = ("(16,16)", "(24,24)", "(32,32)", "(48,48)", "(40,24)", "(24,40)")
_EMBEDDINGS = ("(64,1)", "(128,1)", "(128,2)", "(256,2)")
_A9 = (".25", ".50", "1.0")
_A10 = (".25", "1", "4")
_A11 = ("(.0025,.25)", "(.01,1)", "(.04,4)")
_A12 = ("equal", "signed L1", "random-direction L2")
_B = ("B00", "B10", "B01", "B11")


def _expected_a() -> set[str]:
    blocks: list[dict[str, str]] = []
    blocks.extend(
        _configuration(producer_path=p, activation=a, update=u)
        for p, a, u in product(_PRODUCERS, _ACTIVATIONS, _UPDATES)
    )
    blocks.extend(
        _configuration(producer_path="tensor", activation="SiLU,SiLU", channels=c, embedding=e, update=u)
        for c, e, u in product(_CHANNELS, _EMBEDDINGS, _UPDATES)
    )
    blocks.extend(
        _configuration(
            producer_path="tensor", activation="SiLU,SiLU",
            producer_initialization=i9, electron_electron=i10,
            electron_nucleus=i11, readout_initialization=i12,
        )
        for i9, i10, i11, i12 in product(_A9, _A10, _A11, _A12)
    )
    return {_canonical(item) for item in blocks}


def _expected_b() -> set[str]:
    return {
        _canonical({**_configuration(producer_path=p, activation="SiLU,SiLU", update=u), "package": b})
        for p, u, b in product(_PRODUCERS, _UPDATES, _B)
    }


def _identities(packages: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> set[str]:
    return {_canonical(package["literal_configuration"]) for package in packages}


def _assert_o_star_honest(packages: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> None:
    for package in packages:
        slots = package["optimizer_slots"]
        o_star = next(slot for slot in slots if slot["method"] == "O*")
        assert o_star["status"] == "unavailable"
        assert o_star["levels"] is None
        assert o_star["unavailable_reason"] == "selection output of O2 is pending"


def test_independent_a_and_b_sets_equal_renderer_before_counts() -> None:
    universe = ab_universe.render_ab_universe()
    assert _identities(universe["A"]) == _expected_a()
    assert _identities(universe["B"]) == _expected_b()
    assert len(universe["A"]) == 254
    assert len(universe["B"]) == 60


@pytest.mark.parametrize("axis", ("A", "B"))
def test_literal_identity_is_unique(axis: str) -> None:
    packages = list(ab_universe.render_ab_universe()[axis])
    assert len(packages) == len(_identities(packages))


@pytest.mark.parametrize("axis", ("A", "B"))
def test_drop_one_required_row_is_detected(axis: str) -> None:
    packages = list(ab_universe.render_ab_universe()[axis])
    expected = _expected_a() if axis == "A" else _expected_b()
    dropped = packages.pop()
    assert _identities(packages) != expected
    assert _canonical(dropped["literal_configuration"]) in expected


@pytest.mark.parametrize("display_name", ("A001", "A106", "A178"))
def test_drop_one_required_row_from_each_a_block_is_detected(display_name: str) -> None:
    packages = [
        package for package in ab_universe.render_a_packages()
        if package["display_name"] != display_name
    ]
    assert _identities(packages) != _expected_a()


def test_drop_one_shared_a_duplicate_is_detected() -> None:
    packages = list(ab_universe.render_a_packages())
    packages = [package for package in packages if package["display_name"] != "A001"]
    assert _identities(packages) != _expected_a()


def test_undeclared_level_is_detected() -> None:
    packages = list(ab_universe.render_a_packages())
    packages[0]["literal_configuration"]["activation"] = "undeclared activation"
    assert _identities(packages) != _expected_a()


def test_duplicate_under_new_display_name_is_detected() -> None:
    packages = list(ab_universe.render_a_packages())
    duplicate = copy.deepcopy(packages[0])
    duplicate["display_name"] = "A-renamed-duplicate"
    packages.append(duplicate)
    assert len(packages) != len(_identities(packages))


@pytest.mark.parametrize("axis", ("A", "B"))
def test_o_star_is_explicit_pending_and_never_adam(axis: str) -> None:
    _assert_o_star_honest(ab_universe.render_ab_universe()[axis])
    mutant = copy.deepcopy(ab_universe.render_ab_universe()[axis][0])
    mutant["optimizer_slots"][1] = copy.deepcopy(mutant["optimizer_slots"][0])
    with pytest.raises(AssertionError):
        _assert_o_star_honest((mutant,))


def test_seed_namespaces_and_b_pairing_key() -> None:
    universe = ab_universe.render_ab_universe()
    assert {tuple(package["seed_labels"]) for package in universe["A"]} == {tuple(range(830001, 830009))}
    assert {tuple(package["seed_labels"]) for package in universe["B"]} == {tuple(range(840001, 840009))}
    for producer, update in product(_PRODUCERS, _UPDATES):
        rows = [
            package for package in universe["B"]
            if package["literal_configuration"]["producer_path"] == producer
            and package["literal_configuration"]["update"] == update
        ]
        assert len({
            _canonical(package["model_initialization_stream_key"]) for package in rows
        }) == 1


def test_every_rendered_package_composes_through_ab3_entry_point() -> None:
    control = OmegaConf.load(_HERE / "configs" / "train.yaml")
    for package in (*ab_universe.render_a_packages(), *ab_universe.render_b_packages()):
        composed = ab_universe.compose_package_model(control.model, package, context_mapping)
        assert composed["layers"]
        assert composed["factors"]
