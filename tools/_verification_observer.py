from __future__ import annotations

"""Pytest-side structured observation for TPEN verification receipts.

The module intentionally imports only the standard library. Pytest discovers
the hook functions by name after the receipt wrapper injects this module with
``-p _verification_observer``.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


OBSERVATION_ENV = "TPEN_VERIFICATION_OBSERVATION_PATH"
NONCE_ENV = "TPEN_VERIFICATION_NONCE"
VERSION = 1
_COUNT_FIELDS = ("passed", "failed", "errors", "skipped", "xfailed", "xpassed")


_SELECTION_OPTION_ALIASES = {
    "keyword": ("keyword",),
    "markexpr": ("markexpr",),
    "stepwise": ("stepwise",),
    "stepwise_skip": ("stepwise_skip",),
    "last_failed": ("last_failed", "lf"),
    # Pytest 6.2 uses ``failedfirst``; do not infer the spelling from the
    # option's long flag. Runtime introspection and the real-config test keep
    # this compatibility map honest across supported pytest versions.
    "failed_first": ("failedfirst", "failed_first", "ff"),
    "maxfail": ("maxfail",),
    # ``-x`` stores its limit in maxfail on supported pytest versions, so an
    # absent exitfirst attribute is an explicit, documented exception.
    "exitfirst": ("exitfirst",),
    "collectonly": ("collectonly",),
    "ignore": ("ignore",),
    "ignore_glob": ("ignore_glob",),
    "deselect": ("deselect",),
    "file_or_dir": ("file_or_dir",),
}
OPTIONAL_ABSENT_OPTIONS = {
    "exitfirst": "-x is represented by maxfail on supported pytest versions"
}


def _option_value(option: Any, names: Tuple[str, ...]) -> Tuple[bool, Optional[str], Any]:
    """Read an option alias and preserve whether introspection found it."""

    for name in names:
        if hasattr(option, name):
            return True, name, getattr(option, name)
    return False, None, None


def _json_value(value: Any) -> Any:
    """Convert simple pytest option values into JSON-safe values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _selection_options(config: Any) -> Dict[str, Any]:
    """Capture resolved selection-bearing ``config.option`` fields."""

    option = config.option
    resolved: Dict[str, Any] = {}
    for logical_name, aliases in _SELECTION_OPTION_ALIASES.items():
        found, attribute, value = _option_value(option, aliases)
        resolved[logical_name] = {
            "state": "FOUND" if found else "ABSENT",
            "attribute": attribute,
            "value": _json_value(value),
        }
    return resolved


def _ini_value(config: Any, name: str, default: Any) -> Any:
    """Read an ini value without making plugin startup fail on odd configs."""

    try:
        return _json_value(config.getini(name))
    except (AttributeError, KeyError, TypeError, ValueError):
        return default


def _write_observation(config: Any, collected: Optional[int], deselected: int) -> None:
    """Write the final pytest facts to the wrapper-provided observation path."""

    path = os.environ.get(OBSERVATION_ENV)
    nonce = os.environ.get(NONCE_ENV)
    if not path or nonce is None:
        return

    terminalreporter = config.pluginmanager.get_plugin("terminalreporter")
    stats: Dict[str, int] = {}
    if terminalreporter is not None:
        for field in _COUNT_FIELDS:
            pytest_key = "error" if field == "errors" else field
            stats[field] = len(terminalreporter.stats.get(pytest_key, []))

    invocation_args = list(config.invocation_params.args)
    payload = {
        "version": VERSION,
        "nonce": nonce,
        "invocation_args": [_json_value(value) for value in invocation_args],
        "ini_addopts": _ini_value(config, "addopts", []),
        "ini_testpaths": _ini_value(config, "testpaths", []),
        "env_pytest_addopts": os.environ.get("PYTEST_ADDOPTS"),
        "options": _selection_options(config),
        "collected": collected,
        "deselected": deselected,
        "stats": stats,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def pytest_configure(config: Any) -> None:
    """Initialize per-session state on the pytest config object."""

    config._tpen_verification_collected = None
    config._tpen_verification_deselected = 0


def pytest_collection_finish(session: Any) -> None:
    """Record the selected item count after collection and deselection hooks."""

    session.config._tpen_verification_collected = len(session.items)


def pytest_deselected(items: List[Any]) -> None:
    """Accumulate the number of items pytest removed from the session."""

    if items:
        # Pytest exposes the current config through the first collected item;
        # the session-finish hook also tolerates a zero-item collection.
        config = getattr(items[0], "config", None)
        if config is not None:
            config._tpen_verification_deselected = (
                getattr(config, "_tpen_verification_deselected", 0) + len(items)
            )


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    """Persist the nonce-bound structured result before pytest exits."""

    _write_observation(
        session.config,
        getattr(session.config, "_tpen_verification_collected", None),
        getattr(session.config, "_tpen_verification_deselected", 0),
    )
