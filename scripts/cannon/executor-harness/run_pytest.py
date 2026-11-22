"""Spawn-safe pytest runner with collection/runtime recorder."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import importlib.util

# pytest.main runs IN THIS PROCESS. Importing the harness's own `helpers` package here would cache it in
# sys.modules and shadow the projects' `helpers` test packages (tests/helpers, executor/tests/helpers):
# Cannon job 59350515 failed collection with "cannot import name 'review_targets' from 'helpers'".
# Load path_guard under a private name and drop the harness dir from sys.path before importing pytest.
_HARNESS_DIR = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("_tpen_harness_path_guard", _HARNESS_DIR / "helpers" / "path_guard.py")
_path_guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_path_guard)
assert_write_path = _path_guard.assert_write_path
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").absolute() != _HARNESS_DIR.absolute()]
assert "helpers" not in sys.modules, sys.modules["helpers"]

import pytest  # noqa: E402


class Recorder:
    def __init__(self) -> None:
        self.collected: list[str] = []
        self.deselected: set[str] = set()
        self.reports: list[dict[str, object]] = []
        self.collection: list[dict[str, object]] = []
        self.collection_skips: list[dict[str, object]] = []
        self.rootdir: str | None = None

    def pytest_configure(self, config):
        self.rootdir = str(config.rootpath)

    def pytest_collection_modifyitems(self, session, config, items):
        self.collected = [item.nodeid for item in items]
        for item in items:
            item.user_properties.append(("review_nodeid", item.nodeid))

    def pytest_deselected(self, items):
        self.deselected.update(item.nodeid for item in items)

    def pytest_sessionfinish(self, session, exitstatus):
        # pytest can call collection hooks before the -k deselection hook.
        self.collected = [nodeid for nodeid in self.collected if nodeid not in self.deselected]

    def pytest_runtest_logreport(self, report):
        self.reports.append({"nodeid": report.nodeid, "when": report.when, "outcome": report.outcome, "longrepr": str(report.longrepr)})

    def pytest_collectreport(self, report):
        if report.outcome == "skipped":
            self.collection_skips.append({"nodeid": report.nodeid, "outcome": report.outcome, "longrepr": str(report.longrepr)})
        elif not report.passed:
            self.collection.append({"nodeid": report.nodeid, "outcome": report.outcome, "longrepr": str(report.longrepr)})


def prepare_pytest_paths(args: list[str]) -> None:
    """Create pytest's basetemp parent and TMPDIR for every invocation."""
    try:
        basetemp = Path(args[args.index("--basetemp") + 1])
    except (ValueError, IndexError) as error:
        raise SystemExit("every pytest invocation requires --basetemp") from error
    tmpdir = Path(os.environ.get("TMPDIR", str(basetemp.parent / "tmp")))
    assert_write_path(basetemp.parent)
    assert_write_path(tmpdir)
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: run_pytest.py RECEIPT TEST_ARGS...", file=sys.stderr)
        return 2
    destination = Path(sys.argv[1])
    assert_write_path(destination)
    prepare_pytest_paths(sys.argv[2:])
    recorder = Recorder()
    rc = int(pytest.main(sys.argv[2:], plugins=[recorder]))
    destination.write_text(json.dumps({"rc": rc, "cwd": os.getcwd(), "rootdir": recorder.rootdir, "sys_path": sys.path[:5], "collected": recorder.collected, "deselected": sorted(recorder.deselected), "reports": recorder.reports, "collection": recorder.collection, "collection_skips": recorder.collection_skips}, indent=2), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
