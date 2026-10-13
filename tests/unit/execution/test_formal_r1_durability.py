"""Review R1-8 atomic-writer ordering controls.

The tests prove the fsync/replace sequence around real local filesystem
operations. They do not claim power-loss survival or facility durability.
"""

from __future__ import annotations

import importlib
import os
import stat
import sys
import types
from pathlib import Path

from tools.e5b_polaris_runner import _write_json


def _import_app_with_fake_balsam(monkeypatch):
    fake_api = types.ModuleType("balsam.api")

    class ApplicationDefinition:
        pass

    fake_api.ApplicationDefinition = ApplicationDefinition
    fake_balsam = types.ModuleType("balsam")
    fake_balsam.api = fake_api
    monkeypatch.setitem(sys.modules, "balsam", fake_balsam)
    monkeypatch.setitem(sys.modules, "balsam.api", fake_api)
    sys.modules.pop("tools.e5b_polaris_app", None)
    return importlib.import_module("tools.e5b_polaris_app")


def _spy_filesystem_order(monkeypatch):
    import tpen.execution.invocation as invocation

    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def spy_fsync(fd: int) -> None:
        mode = os.fstat(fd).st_mode
        if stat.S_ISREG(mode):
            events.append("fsync-file")
        elif stat.S_ISDIR(mode):
            events.append("fsync-dir")
        else:
            events.append("fsync-other")
        real_fsync(fd)

    def spy_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(invocation.os, "fsync", spy_fsync)
    monkeypatch.setattr(invocation.os, "replace", spy_replace)
    return events


def _assert_durable_order(events: list[str]) -> None:
    file_sync = events.index("fsync-file")
    replace = events.index("replace")
    directory_sync = events.index("fsync-dir")
    assert file_sync < replace < directory_sync, events


def test_runner_json_writer_syncs_file_before_replace_and_directory_after(
    tmp_path: Path, monkeypatch
) -> None:
    events = _spy_filesystem_order(monkeypatch)

    _write_json(tmp_path / "runner-marker.json", {"ok": True})

    _assert_durable_order(events)


def test_app_pre_e3_writer_syncs_file_before_replace_and_directory_after(
    tmp_path: Path, monkeypatch
) -> None:
    app = _import_app_with_fake_balsam(monkeypatch)
    events = _spy_filesystem_order(monkeypatch)

    class Job:
        id = "job-review-r1"
        data = {"pilot_root": str(tmp_path / "pilot")}

    app._write_pre_e3_failure(Job(), RuntimeError("boom"))

    assert (tmp_path / "pilot" / "pre-e3-failure.json").is_file()
    _assert_durable_order(events)
