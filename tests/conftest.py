"""Shared pytest configuration and fixtures for the TPEN test suite."""

from __future__ import annotations

import logging

import pytest

# Loggers that the terminal-status callback mutates globally
# (`configure_terminal_logging` adds a StreamHandler and sets ``propagate=False``).
# Without isolation, a test that enables terminal logging leaks
# ``propagate=False`` into later caplog-based unit tests, which then capture no
# records. Reset these loggers to a clean, propagating state before each test
# and restore the pre-test snapshot afterwards.
_ISOLATED_LOGGERS = ("tpen", "tpen.status", "tpen.bootstrap")


def _drop_terminal_handlers(logger: logging.Logger) -> None:
    logger.handlers[:] = [
        handler for handler in logger.handlers if not getattr(handler, "_tpen_terminal_handler", False)
    ]


def _drop_bootstrap_handlers(logger: logging.Logger) -> None:
    logger.handlers[:] = [
        handler for handler in logger.handlers if not getattr(handler, "_tpen_bootstrap_handler", False)
    ]


def _typeguard_instruments_tpen(config: pytest.Config) -> bool:
    """Return whether the typeguard plugin is instrumenting ``tpen`` this run.

    Reads the same CLI option / ini fallback typeguard's own
    ``pytest_configure`` uses (``typeguard._pytest_plugin``), rather than
    guessing from an env var, so this agrees with what actually got
    instrumented regardless of how ``--typeguard-packages`` was set or
    cleared (e.g. a runner disabling typeguard via ``-o addopts=``).
    """

    packages_option = config.getoption("typeguard_packages", default=None) or config.getini(
        "typeguard-packages"
    )
    if not packages_option:
        return False
    packages = (
        [pkg.strip() for pkg in packages_option.split(",")]
        if isinstance(packages_option, str)
        else list(packages_option)
    )
    return packages == [":all:"] or "tpen" in packages


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``requires_typeguard`` tests when typeguard is not instrumenting tpen.

    These tests assert a bare ``TypeCheckError`` with no manual-validation
    fallback, so they always fail when runtime type checking is off (e.g. a
    runner disabling typeguard for a quick pass); the full suite still runs
    them with typeguard on.
    """

    if _typeguard_instruments_tpen(config):
        return
    skip_marker = pytest.mark.skip(
        reason="requires_typeguard: typeguard is not instrumenting tpen this run"
    )
    for item in items:
        if "requires_typeguard" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(autouse=True)
def _restore_tpen_logger_state():
    """Give each test a clean TPEN logger tree and restore it afterwards."""

    saved = {}
    for name in _ISOLATED_LOGGERS:
        logger = logging.getLogger(name)
        saved[name] = (list(logger.handlers), logger.level, logger.propagate)
        # Clear any leaked terminal handler / propagate=False from a prior test.
        _drop_terminal_handlers(logger)
        _drop_bootstrap_handlers(logger)
        logger.propagate = True
    try:
        yield
    finally:
        for name, (handlers, level, propagate) in saved.items():
            logger = logging.getLogger(name)
            logger.handlers[:] = handlers
            logger.setLevel(level)
            logger.propagate = propagate
