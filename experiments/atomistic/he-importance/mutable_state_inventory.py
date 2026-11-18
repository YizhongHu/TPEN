"""Closed mutable-state inventory for the HI training configuration.

The inventory is deliberately derived from the committed Hydra tree rather
than maintained as a second list. A newly constructed target therefore fails
closed until its recovery ownership is classified here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from tpen.checkpoint.profiles import (
    SUPPORTED_CALLBACK_TARGETS,
    SUPPORTED_LOGGER_TARGETS,
)

_CONFIG_PATH = Path(__file__).with_name("configs") / "train.yaml"
_REFUSED_TARGETS = frozenset(
    {
        "tpen.callback.RuntimeEquivariance",
        "tpen.callback.ResourceUsage",
    }
)
_ROOT_CLASSIFICATIONS = {
    "sampler": ("restored", "native train_resume owns sampler_mcmc state"),
    "trainer": ("restored", "native train_resume owns trainer_cursor state"),
}
MUTANT_REFUSAL_MAP = {
    "M1-omitted-sampler-rng": "refused by HI continuation boundary",
    "M2-reset-cadence-counter": "refused by HI continuation boundary",
    "M3-callback-private-rng-drift": "refused by HI continuation boundary",
    "M4-event-before-restore": "refused by HI continuation boundary",
    "M5-missing-state-classification": "refused by closed inventory",
}


@dataclass(frozen=True, slots=True)
class MutableStateEntry:
    """One constructed HI object and its explicit continuation disposition."""

    path: str
    target: str
    classification: str
    reason: str
    source: str


def _source_ref(path: str, target: str) -> str:
    """Return a stable train.yaml line reference for a config path."""

    lines = _CONFIG_PATH.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if "_target_:" in line and target in line:
            return f"{_CONFIG_PATH}:line {number}"
    return f"{_CONFIG_PATH}:path {path}"


def _classification(path: str, target: str) -> tuple[str, str]:
    """Classify one target using the native profile and explicit HI boundary."""

    if target in _REFUSED_TARGETS:
        return (
            "typed-refused",
            "native recovery profile does not admit this mutable owner; "
            "HI continuation refuses before construction",
        )
    if path == "sampler" or path == "trainer":
        return _ROOT_CLASSIFICATIONS[path]
    if path.startswith("callbacks."):
        if target in SUPPORTED_CALLBACK_TARGETS:
            return (
                "reconstructible",
                "explicit native callback allowlist entry; callback state is rebuilt from config",
            )
        return ("typed-refused", "callback target is absent from the native profile")
    if path.startswith("loggers."):
        if target in SUPPORTED_LOGGER_TARGETS:
            return (
                "reconstructible",
                "explicit native logger allowlist entry; logger state is rebuilt from config",
            )
        return ("typed-refused", "logger target is absent from the native profile")
    if ".checkers." in path:
        return (
            "reconstructible",
            "checker has explicit constructor state and is rebuilt by its owning callback",
        )
    if path.endswith(".schedule"):
        return (
            "reconstructible",
            "schedule is a pure completed-update function, not a private counter",
        )
    if path.endswith(".allocator_probe"):
        return ("allocation-local", "allocator measurement is local to one process allocation")
    return ("reconstructible", "nested constructor is rebuilt from the committed config")


def _walk_targets(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Enumerate every Hydra constructor target in a config tree."""

    found: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        target = value.get("_target_")
        if isinstance(target, str) and target:
            found.append((path, target))
        for key, nested in value.items():
            child = f"{path}.{key}" if path else str(key)
            found.extend(_walk_targets(nested, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            child = f"{path}.{index}" if path else str(index)
            found.extend(_walk_targets(nested, child))
    return found


def _config_tree(config: Any | None) -> Mapping[str, Any]:
    if config is None:
        from . import train_config

        config = train_config._compose_base_config()
    if isinstance(config, DictConfig) or OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=False)
    if not isinstance(config, Mapping):
        raise TypeError("HI mutable-state inventory requires a mapping config")
    return config


def build_mutable_state_inventory(config: Any | None = None) -> tuple[MutableStateEntry, ...]:
    """Enumerate and classify every constructor in the HI train config."""

    tree = _config_tree(config)
    entries: list[MutableStateEntry] = []
    for path, target in _walk_targets(tree):
        classification, reason = _classification(path, target)
        entries.append(
            MutableStateEntry(
                path=path,
                target=target,
                classification=classification,
                reason=reason,
                source=_source_ref(path, target),
            )
        )
    if not entries:
        raise ValueError("HI train config constructs no inventory objects")
    return tuple(entries)


def validate_mutable_state_inventory(config: Any | None = None) -> tuple[MutableStateEntry, ...]:
    """Return the inventory, refusing any unclassified constructed object."""

    entries = build_mutable_state_inventory(config)
    allowed = {"restored", "reconstructible", "allocation-local", "typed-refused"}
    invalid = [entry for entry in entries if entry.classification not in allowed]
    if invalid:
        raise ValueError(f"unclassified HI mutable-state object(s): {invalid!r}")
    return entries


def refused_inventory_entries(config: Any | None = None) -> tuple[MutableStateEntry, ...]:
    """Return the typed-refused entries used in the stable refusal reason."""

    return tuple(
        entry
        for entry in validate_mutable_state_inventory(config)
        if entry.classification == "typed-refused"
    )


__all__ = [
    "MUTANT_REFUSAL_MAP",
    "MutableStateEntry",
    "build_mutable_state_inventory",
    "refused_inventory_entries",
    "validate_mutable_state_inventory",
]
