"""Pure enumeration of the consolidated-design A and B package universes.

This module deliberately stops at scientific package descriptions.  It does
not write the intended-configuration inventory, alter the resolver, or admit
any row to a runner.  Model construction remains owned by
``context_mapping.compose_architecture_package``.
"""

from __future__ import annotations

from copy import deepcopy
import json
from itertools import product
from typing import Any, Iterable, Mapping


_CONTROL = {
    "channels": "(32,32)",
    "embedding": "(128,1)",
    "producer_initialization": ".50",
    "electron_electron": "1",
    "electron_nucleus": "(.01,1)",
    "readout_initialization": "equal",
}
_PRODUCERS = (
    "tensor",
    "linear/coordinate-neighbor",
    "linear/orbit-complete",
    "hybrid/coordinate-neighbor",
    "hybrid/orbit-complete",
)
_ACTIVATIONS = (
    "SiLU,SiLU",
    "tanh,tanh",
    "exact GELU,exact GELU",
    "Softplus beta2,Softplus beta2",
    "channel-MLP,SiLU",
    "SiLU,channel-MLP",
    "channel-MLP,channel-MLP",
)
_UPDATES = ("residual", "replacement", "Gaussian residual")
_CHANNELS = ("(16,16)", "(24,24)", "(32,32)", "(48,48)", "(40,24)", "(24,40)")
_EMBEDDINGS = ("(64,1)", "(128,1)", "(128,2)", "(256,2)")
_PRODUCER_INITIALIZATIONS = (".25", ".50", "1.0")
_ELECTRON_ELECTRON = (".25", "1", "4")
_ELECTRON_NUCLEUS = ("(.0025,.25)", "(.01,1)", "(.04,4)")
_READOUT_INITIALIZATIONS = ("equal", "signed L1", "random-direction L2")
_B_PACKAGES = ("B00", "B10", "B01", "B11")

_OPTIMIZER_SLOTS = (
    {
        "method": "adam",
        "status": "available",
        "levels": {
            "lr": 0.005,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "parameter_groups": 1,
            "schedule": "constant",
        },
    },
    {
        "method": "O*",
        "status": "unavailable",
        "levels": None,
        "unavailable_reason": "selection output of O2 is pending",
    },
)
_SEED_LABELS = {"A": tuple(range(830_001, 830_009)), "B": tuple(range(840_001, 840_009))}


def _canonical(value: Any) -> str:
    """Return a stable identity representation for literal configuration data."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _configuration(**values: str) -> dict[str, str]:
    """Build one complete, label-only architecture configuration."""

    result = dict(_CONTROL)
    result.update(values)
    return result


def _package(axis: str, number: int, configuration: Mapping[str, str], package: str | None) -> dict[str, Any]:
    """Render one package while keeping display identity out of its content identity."""

    literal = dict(configuration)
    if package is not None:
        literal["package"] = package
    producer = literal["producer_path"]
    update = literal["update"]
    return {
        "axis": axis,
        "display_name": f"{axis}{number:03d}",
        "literal_configuration": literal,
        "optimizer_slots": deepcopy(_OPTIMIZER_SLOTS),
        "seed_labels": list(_SEED_LABELS[axis]),
        "model_initialization_stream_key": {
            "stage": axis,
            "producer_path": producer,
            "update": update,
        },
    }


def _deduplicate(packages: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Deduplicate only by literal scientific configuration."""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for package in packages:
        key = _canonical(package["literal_configuration"])
        if key not in seen:
            seen.add(key)
            result.append(package)
    return tuple(result)


def render_a_packages() -> tuple[dict[str, Any], ...]:
    """Render the 254 deduplicated A packages from the three design blocks."""

    candidates: list[dict[str, Any]] = []
    number = 1
    for producer, activation, update in product(_PRODUCERS, _ACTIVATIONS, _UPDATES):
        candidates.append(_package("A", number, _configuration(
            producer_path=producer, activation=activation, update=update
        ), None))
        number += 1
    for channels, embedding, update in product(_CHANNELS, _EMBEDDINGS, _UPDATES):
        candidates.append(_package("A", number, _configuration(
            producer_path="tensor", activation="SiLU,SiLU", channels=channels,
            embedding=embedding, update=update
        ), None))
        number += 1
    for producer_initialization, electron_electron, electron_nucleus, readout_initialization in product(
        _PRODUCER_INITIALIZATIONS, _ELECTRON_ELECTRON, _ELECTRON_NUCLEUS, _READOUT_INITIALIZATIONS
    ):
        candidates.append(_package("A", number, _configuration(
            producer_path="tensor", activation="SiLU,SiLU", producer_initialization=producer_initialization,
            electron_electron=electron_electron, electron_nucleus=electron_nucleus,
            readout_initialization=readout_initialization
        ), None))
        number += 1
    return _deduplicate(candidates)


def render_b_packages() -> tuple[dict[str, Any], ...]:
    """Render the 60 B product packages at the declared control settings."""

    packages: list[dict[str, Any]] = []
    number = 1
    for producer, update, package in product(_PRODUCERS, _UPDATES, _B_PACKAGES):
        packages.append(_package("B", number, _configuration(
            producer_path=producer, activation="SiLU,SiLU", update=update
        ), package))
        number += 1
    return tuple(packages)


def render_ab_universe() -> dict[str, tuple[dict[str, Any], ...]]:
    """Return the pure A/B package universe without mutating repository files."""

    return {"A": render_a_packages(), "B": render_b_packages()}


def compose_package_model(model_cfg: Mapping[str, Any], package: Mapping[str, Any], context_mapping: Any) -> dict[str, Any]:
    """Compose a rendered package through the AB-3 context-mapping entry point.

    Parameters
    ----------
    model_cfg
        Literal control model configuration.
    package
        One package returned by :func:`render_ab_universe`.
    context_mapping
        The loaded AB-3 ``context_mapping`` module.  Passing it explicitly
        keeps this renderer independent of the hyphenated experiment path.
    """

    labels = package["literal_configuration"]
    kwargs = {key: labels[key] for key in (
        "producer_path", "activation", "channels", "embedding", "update",
        "producer_initialization", "electron_electron", "electron_nucleus",
        "readout_initialization",
    )}
    return context_mapping.compose_architecture_package(
        model_cfg, package=labels.get("package"), **kwargs
    )


__all__ = ["compose_package_model", "render_a_packages", "render_ab_universe", "render_b_packages"]
